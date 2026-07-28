#!/usr/bin/env python3
"""Differentiable per-interval Hamiltonian total derivatives.

This module is deliberately standalone: it does not alter any report or
figure-generation pipeline.  It provides the literal per-interval quantity
requested for the follow-up diagnostic,

    d H_k(U) / d u_k,       |d^2 H_k(U) / d u_k^2|,

where ``U=(u_0,...,u_{n-1})`` is the realized zero-order-hold control vector.
The word *total* is important: the RK4 stage states and the discrete RK4
costates remain functions of the complete vector ``U`` while the derivatives
are taken.  There is no ``detach()``, ``no_grad()``, or ``inference_mode()`` in
the differentiable calculation.

Hamiltonian definition and costate pairing
--------------------------------------------
For classical RK4 there is no single node-costate pairing that gives the
exact interval control derivative.  In particular, neither
``H(N_k, lambda_k, u_k)`` nor ``H(N_k, lambda_{k+1}, u_k)`` is the exact RK4
counterpart.  We therefore define the realized interval Hamiltonian by the
same four RK stages used by the state and running-cost quadrature,

    H_k(U) = sum_s b_s [ L(X_{k,s}(U), u_k)
                         + P_{k,s}(U)^T f(X_{k,s}(U), u_k) ],

with ``b=(1/6,1/3,1/3,1/6)``.  ``P_{k,s}`` are the matching discrete stage
costates returned by :func:`feedback_section5_rk4_reference.discrete_rk4_adjoint`.
Their recursion starts from the *next-node* costate ``lambda_{k+1}`` and adds
the later-stage corrections required by RK4.  Thus this helper follows the
``lambda_{k+1}`` convention in the precise stage-corrected sense; it does not
substitute the raw next-node costate for all four stages.  In the forward-Euler
limit this construction reduces to ``H(N_k, lambda_{k+1}, u_k)``.

This total derivative is a different mathematical object from the standard
PMP partial derivative ``H_u`` (which holds state and costate fixed), and from
the reduced-objective derivative ``d J_hat_h / d u_k``.  The distinction is
intentional and is exposed in the API naming.

The command-line smoke test uses a small grid and checks both derivatives
against centered finite differences of the same scalar ``H_k(U)``.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for search_path in (ROOT, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from feedback_section5_rk4_reference import (  # noqa: E402
    RK4_B,
    discrete_rk4_adjoint,
    dynamics,
    running_cost,
    simulate_open_loop_rk4,
)
from train_paper_pmp_kkt import ProblemConfig, build_params  # noqa: E402


TensorMap = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class IntervalHamiltonianTotalDerivatives:
    """Per-interval Hamiltonian values and current-control total derivatives.

    Every tensor has shape ``(n,)``.  No result is explicitly detached; if the
    caller supplies a graph-connected control tensor, the outputs remain
    graph-connected as supported by the selected ``torch.func`` transform.
    """

    hamiltonian: torch.Tensor
    dH_du_current: torch.Tensor
    d2H_du2_current: torch.Tensor
    abs_d2H_du2_current: torch.Tensor


def _validate_inputs(
    interval_control: torch.Tensor,
    initial_state: torch.Tensor,
    cfg: ProblemConfig,
) -> None:
    if interval_control.ndim != 1 or interval_control.numel() != cfg.n:
        raise ValueError(
            f"interval_control must have shape ({cfg.n},), got "
            f"{tuple(interval_control.shape)}"
        )
    if initial_state.ndim != 1 or initial_state.numel() != cfg.m:
        raise ValueError(
            f"initial_state must have shape ({cfg.m},), got "
            f"{tuple(initial_state.shape)}"
        )
    if interval_control.device != initial_state.device:
        raise ValueError("interval_control and initial_state must share a device")
    if interval_control.dtype != initial_state.dtype:
        raise ValueError("interval_control and initial_state must share a dtype")
    if not interval_control.dtype.is_floating_point:
        raise TypeError("interval_control must use a floating-point dtype")


def build_interval_hamiltonian_map(
    cfg: ProblemConfig,
    params: Dict[str, torch.Tensor],
    initial_state: torch.Tensor,
) -> TensorMap:
    """Return the differentiable map ``U -> (H_0(U),...,H_{n-1}(U))``.

    ``params`` should normally come from :func:`build_params`.  The supplied
    initial state replaces ``params['N0']`` for this trajectory.  It is treated
    as fixed when control derivatives are taken.
    """

    if initial_state.ndim != 1 or initial_state.numel() != cfg.m:
        raise ValueError(
            f"initial_state must have shape ({cfg.m},), got "
            f"{tuple(initial_state.shape)}"
        )
    weights = torch.as_tensor(
        RK4_B, dtype=initial_state.dtype, device=initial_state.device
    ).view(1, 1, 4)
    batched_initial = initial_state.unsqueeze(0)

    def interval_hamiltonian(interval_control: torch.Tensor) -> torch.Tensor:
        _validate_inputs(interval_control, initial_state, cfg)
        states, controls, stage_states = simulate_open_loop_rk4(
            interval_control,
            batched_initial,
            cfg,
            params,
        )
        adjoint = discrete_rk4_adjoint(
            states,
            controls,
            stage_states,
            cfg,
            params,
        )
        stage_controls = controls.unsqueeze(-1).expand(-1, -1, 4)
        stage_dynamics = dynamics(stage_states, stage_controls, params)
        stage_running_cost = running_cost(stage_states, stage_controls, params)
        stage_hamiltonian = stage_running_cost + (
            adjoint.stage_costates * stage_dynamics
        ).sum(dim=-1)
        return (stage_hamiltonian * weights).sum(dim=-1).squeeze(0)

    return interval_hamiltonian


def batched_interval_hamiltonian(
    interval_controls: torch.Tensor,
    cfg: ProblemConfig,
    params: Dict[str, torch.Tensor],
    initial_state: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the stage-consistent interval Hamiltonian for a control batch.

    ``interval_controls`` has shape ``(batch,n)`` and the return value has the
    same leading shape.  This is the batched counterpart of
    :func:`build_interval_hamiltonian_map`; state and discrete-costate solves
    are both repeated for every control row.
    """

    if interval_controls.ndim != 2 or interval_controls.shape[1] != cfg.n:
        raise ValueError(
            f"interval_controls must have shape (batch,{cfg.n}), got "
            f"{tuple(interval_controls.shape)}"
        )
    if initial_state.ndim != 1 or initial_state.numel() != cfg.m:
        raise ValueError(
            f"initial_state must have shape ({cfg.m},), got "
            f"{tuple(initial_state.shape)}"
        )
    batch = interval_controls.shape[0]
    batched_initial = initial_state.to(
        device=interval_controls.device, dtype=interval_controls.dtype
    ).unsqueeze(0).expand(batch, -1)
    states, controls, stage_states = simulate_open_loop_rk4(
        interval_controls,
        batched_initial,
        cfg,
        params,
    )
    adjoint = discrete_rk4_adjoint(
        states,
        controls,
        stage_states,
        cfg,
        params,
    )
    stage_controls = controls.unsqueeze(-1).expand(-1, -1, 4)
    stage_dynamics = dynamics(stage_states, stage_controls, params)
    stage_running_cost = running_cost(stage_states, stage_controls, params)
    stage_hamiltonian = stage_running_cost + (
        adjoint.stage_costates * stage_dynamics
    ).sum(dim=-1)
    weights = torch.as_tensor(
        RK4_B,
        dtype=interval_controls.dtype,
        device=interval_controls.device,
    ).view(1, 1, 4)
    return (stage_hamiltonian * weights).sum(dim=-1)


def current_control_total_derivatives_five_point(
    interval_control: torch.Tensor,
    cfg: ProblemConfig,
    *,
    initial_state: torch.Tensor | None = None,
    params: Dict[str, torch.Tensor] | None = None,
    step: float = 1.0e-2,
    chunk_size: int = 64,
) -> IntervalHamiltonianTotalDerivatives:
    """Compute the literal current-control derivatives on a large time grid.

    A fourth-order, five-point stencil is applied to the complete discrete
    map ``U -> H_k(U)``.  Every perturbed evaluation reruns both the RK4 state
    recursion and its matching discrete-costate recursion, so the resulting
    derivatives include all state and costate dependence on ``u_k``.  The
    batched/chunked implementation avoids materializing the much larger
    higher-order automatic-differentiation graph required by a dense grid.

    The stencil is validated against the nested-JVP implementation in this
    module's small-grid tests.  Perturbations are intentionally not projected
    back into the control box: these are derivatives of the smooth discrete
    Hamiltonian map at the realized control, including at an active bound.
    """

    if step <= 0.0:
        raise ValueError("step must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if params is None:
        params = build_params(cfg, interval_control.device, interval_control.dtype)
    if initial_state is None:
        initial_state = params["N0"]
    initial_state = initial_state.to(
        device=interval_control.device, dtype=interval_control.dtype
    )
    _validate_inputs(interval_control, initial_state, cfg)

    with torch.no_grad():
        base = batched_interval_hamiltonian(
            interval_control.unsqueeze(0), cfg, params, initial_state
        ).squeeze(0)
        first = torch.empty_like(interval_control)
        second = torch.empty_like(interval_control)
        for start in range(0, cfg.n, chunk_size):
            stop = min(start + chunk_size, cfg.n)
            indices = torch.arange(
                start, stop, device=interval_control.device, dtype=torch.long
            )
            directions = torch.nn.functional.one_hot(
                indices, num_classes=cfg.n
            ).to(dtype=interval_control.dtype)
            diagonal_values: dict[int, torch.Tensor] = {}
            row = torch.arange(stop - start, device=interval_control.device)
            for multiple in (-2, -1, 1, 2):
                perturbed = interval_control.unsqueeze(0) + (
                    multiple * step * directions
                )
                values = batched_interval_hamiltonian(
                    perturbed, cfg, params, initial_state
                )
                diagonal_values[multiple] = values[row, indices]
            base_diagonal = base[indices]
            first[indices] = (
                -diagonal_values[2]
                + 8.0 * diagonal_values[1]
                - 8.0 * diagonal_values[-1]
                + diagonal_values[-2]
            ) / (12.0 * step)
            second[indices] = (
                -diagonal_values[2]
                + 16.0 * diagonal_values[1]
                - 30.0 * base_diagonal
                + 16.0 * diagonal_values[-1]
                - diagonal_values[-2]
            ) / (12.0 * step**2)

    return IntervalHamiltonianTotalDerivatives(
        hamiltonian=base,
        dH_du_current=first,
        d2H_du2_current=second,
        abs_d2H_du2_current=second.abs(),
    )


def _coordinate_total_derivatives_chunk(
    interval_hamiltonian: TensorMap,
    interval_control: torch.Tensor,
    directions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate matching-output/matching-input first and second derivatives."""

    def one_direction(direction: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # For direction e_k, dot(J_H(U)e_k, e_k) = d H_k / d u_k.
        def first_directional(control: torch.Tensor) -> torch.Tensor:
            _, tangent = torch.func.jvp(
                interval_hamiltonian,
                (control,),
                (direction,),
            )
            return torch.dot(tangent, direction)

        first = first_directional(interval_control)
        # Differentiating the same scalar once more along e_k gives
        # d^2 H_k / d u_k^2, without materializing n separate dense Hessians.
        _, second = torch.func.jvp(
            first_directional,
            (interval_control,),
            (direction,),
        )
        return first, second

    # Do not vmap the nested JVP over all coordinate directions.  That creates
    # a very large higher-order batched-tangent working set as n grows.  The
    # explicit stack keeps memory essentially at one differentiated rollout,
    # while ``chunk_size`` in the public API controls how many graph results
    # are retained before concatenation.
    pairs = [one_direction(direction) for direction in directions]
    return (
        torch.stack([pair[0] for pair in pairs]),
        torch.stack([pair[1] for pair in pairs]),
    )


def current_control_total_derivatives(
    interval_control: torch.Tensor,
    cfg: ProblemConfig,
    *,
    initial_state: torch.Tensor | None = None,
    params: Dict[str, torch.Tensor] | None = None,
    chunk_size: int = 1,
    method: str = "auto",
    dense_limit: int = 64,
) -> IntervalHamiltonianTotalDerivatives:
    """Compute ``H_k``, ``dH_k/du_k``, and ``|d2H_k/du_k2|`` for every k.

    The coordinate derivatives are evaluated with nested forward-mode JVPs.
    ``method='dense_nested_reverse'`` forms the Jacobian of the Hamiltonian
    vector and then the Jacobian of its diagonal.  It is fast for small grids
    but its higher-order reverse-mode working set grows quickly.  The
    ``'coordinate_jvp'`` method processes coordinate directions independently;
    it has approximately ``O(n^2 m)`` work but bounded per-direction memory.
    ``'auto'`` selects the dense method up to ``dense_limit`` and the JVP method
    above it.

    For ``coordinate_jvp``, directions are processed sequentially inside each
    chunk.  ``chunk_size=1`` is the conservative higher-order-autodiff choice.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if dense_limit <= 0:
        raise ValueError("dense_limit must be positive")
    if method not in {"auto", "dense_nested_reverse", "coordinate_jvp"}:
        raise ValueError(
            "method must be 'auto', 'dense_nested_reverse', or 'coordinate_jvp'"
        )
    if params is None:
        params = build_params(cfg, interval_control.device, interval_control.dtype)
    if initial_state is None:
        initial_state = params["N0"]
    initial_state = initial_state.to(
        device=interval_control.device, dtype=interval_control.dtype
    )
    _validate_inputs(interval_control, initial_state, cfg)
    interval_hamiltonian = build_interval_hamiltonian_map(
        cfg,
        params,
        initial_state,
    )
    values = interval_hamiltonian(interval_control)

    selected_method = method
    if selected_method == "auto":
        selected_method = (
            "dense_nested_reverse"
            if cfg.n <= dense_limit
            else "coordinate_jvp"
        )

    if selected_method == "dense_nested_reverse":
        full_jacobian = torch.func.jacrev(interval_hamiltonian)

        def current_first(control: torch.Tensor) -> torch.Tensor:
            return torch.diagonal(full_jacobian(control))

        first_all = current_first(interval_control)
        second_all = torch.diagonal(
            torch.func.jacrev(current_first)(interval_control)
        )
        return IntervalHamiltonianTotalDerivatives(
            hamiltonian=values,
            dH_du_current=first_all,
            d2H_du2_current=second_all,
            abs_d2H_du2_current=second_all.abs(),
        )

    first_parts: list[torch.Tensor] = []
    second_parts: list[torch.Tensor] = []
    for start in range(0, cfg.n, chunk_size):
        stop = min(start + chunk_size, cfg.n)
        indices = torch.arange(
            start, stop, device=interval_control.device, dtype=torch.long
        )
        directions = torch.nn.functional.one_hot(
            indices, num_classes=cfg.n
        ).to(dtype=interval_control.dtype)
        first, second = _coordinate_total_derivatives_chunk(
            interval_hamiltonian,
            interval_control,
            directions,
        )
        first_parts.append(first)
        second_parts.append(second)

    first_all = torch.cat(first_parts)
    second_all = torch.cat(second_parts)
    return IntervalHamiltonianTotalDerivatives(
        hamiltonian=values,
        dH_du_current=first_all,
        d2H_du2_current=second_all,
        abs_d2H_du2_current=second_all.abs(),
    )


def _finite_difference_validation(
    interval_hamiltonian: TensorMap,
    interval_control: torch.Tensor,
    derivatives: IntervalHamiltonianTotalDerivatives,
    *,
    first_epsilon: float,
    second_epsilon: float,
) -> dict[str, float]:
    """Centered finite-difference validation of the per-coordinate quantities."""

    n = interval_control.numel()
    first_fd = torch.empty_like(interval_control)
    second_fd = torch.empty_like(interval_control)
    base = interval_hamiltonian(interval_control)
    for index in range(n):
        direction = torch.nn.functional.one_hot(
            torch.tensor(index, device=interval_control.device), num_classes=n
        ).to(dtype=interval_control.dtype)
        plus_first = interval_hamiltonian(
            interval_control + first_epsilon * direction
        )[index]
        minus_first = interval_hamiltonian(
            interval_control - first_epsilon * direction
        )[index]
        first_fd[index] = (plus_first - minus_first) / (2.0 * first_epsilon)

        plus_second = interval_hamiltonian(
            interval_control + second_epsilon * direction
        )[index]
        minus_second = interval_hamiltonian(
            interval_control - second_epsilon * direction
        )[index]
        second_fd[index] = (
            plus_second - 2.0 * base[index] + minus_second
        ) / (second_epsilon**2)

    first_error = derivatives.dH_du_current - first_fd
    second_error = derivatives.d2H_du2_current - second_fd
    first_scale = torch.maximum(
        derivatives.dH_du_current.abs().max(),
        torch.ones((), dtype=interval_control.dtype, device=interval_control.device),
    )
    second_scale = torch.maximum(
        derivatives.d2H_du2_current.abs().max(),
        torch.ones((), dtype=interval_control.dtype, device=interval_control.device),
    )
    return {
        "first_max_abs_error": float(first_error.abs().max().item()),
        "first_relative_linf_error": float(
            (first_error.abs().max() / first_scale).item()
        ),
        "second_max_abs_error": float(second_error.abs().max().item()),
        "second_relative_linf_error": float(
            (second_error.abs().max() / second_scale).item()
        ),
    }


def smoke_validation(args: argparse.Namespace) -> dict[str, float]:
    """Run a small differentiability and finite-difference smoke test."""

    torch.set_default_dtype(torch.float64)
    cfg = ProblemConfig(
        T=args.T,
        n=args.n,
        m=args.m,
        umax=3.0,
        beta=0.1,
        alpha=1.0,
        gamma=20.0,
        n0=10.0,
        m_suppression=0.5,
    )
    device = torch.device("cpu")
    params = build_params(cfg, device, torch.float64)
    midpoint = (torch.arange(cfg.n, dtype=torch.float64) + 0.5) / cfg.n
    interval_control = (
        1.1
        + 0.22 * torch.sin(2.0 * math.pi * midpoint)
        + 0.07 * torch.cos(4.0 * math.pi * midpoint)
    )
    interval_hamiltonian = build_interval_hamiltonian_map(
        cfg,
        params,
        params["N0"],
    )
    graph_control = interval_control.clone().requires_grad_(True)
    direct_values = interval_hamiltonian(graph_control)
    if not direct_values.requires_grad:
        raise RuntimeError("Hamiltonian values unexpectedly lost their autograd graph")

    started = time.perf_counter()
    derivatives = current_control_total_derivatives(
        interval_control,
        cfg,
        initial_state=params["N0"],
        params=params,
        chunk_size=args.chunk_size,
        method=args.method,
        dense_limit=args.dense_limit,
    )
    autodiff_seconds = time.perf_counter() - started
    validation = _finite_difference_validation(
        interval_hamiltonian,
        interval_control,
        derivatives,
        first_epsilon=args.first_epsilon,
        second_epsilon=args.second_epsilon,
    )
    validation.update(
        {
            "n": float(cfg.n),
            "m": float(cfg.m),
            "T": float(cfg.T),
            "autodiff_seconds": autodiff_seconds,
            "H_min": float(derivatives.hamiltonian.min().item()),
            "H_max": float(derivatives.hamiltonian.max().item()),
            "dH_du_min": float(derivatives.dH_du_current.min().item()),
            "dH_du_max": float(derivatives.dH_du_current.max().item()),
            "abs_d2H_du2_max": float(
                derivatives.abs_d2H_du2_current.max().item()
            ),
        }
    )
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--m", type=int, default=7)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument(
        "--method",
        choices=("auto", "dense_nested_reverse", "coordinate_jvp"),
        default="auto",
    )
    parser.add_argument("--dense-limit", type=int, default=64)
    parser.add_argument("--first-epsilon", type=float, default=1.0e-5)
    parser.add_argument("--second-epsilon", type=float, default=2.0e-4)
    args = parser.parse_args()
    if args.n <= 0 or args.m <= 0 or args.T <= 0.0:
        raise ValueError("n, m, and T must be positive")
    result = smoke_validation(args)
    for key, value in result.items():
        print(f"{key}: {value:.12g}")


if __name__ == "__main__":
    main()
