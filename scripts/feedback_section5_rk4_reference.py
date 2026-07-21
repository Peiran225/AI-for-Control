#!/usr/bin/env python3
"""Differentiable RK4-ZOH reference for the Section-5 PMP loss.

This module contains no reduced-objective training loss.  Its backward pass
computes the *standard PMP/discrete-optimal-control adjoint*: every realized
interval control is held fixed when the local state derivative is taken.  The
result remains in the outer autograd graph, so a PMP residual assembled from
the returned quantities can still be differentiated all the way to policy
parameters.

For classical RK4, pairing ``N_k`` with ``lambda_{k+1}`` is not an exact
discrete switching formula.  Let ``X_{k,s}`` be the four RK stages, ``P_{k,s}``
their discrete stage costates, and ``b=(1/6,1/3,1/3,1/6)``.  The exact
interval-control derivative for the matching RK4 transcription is

    d J_h / d u_k = h * sum_s b_s H_u(X_{k,s}, P_{k,s}, u_k).

The weighted quantity on the right divided by ``h`` is returned as
``interval_switching``.  It reduces to the familiar
``gamma - phi^T(lambda_{k+1} * N_k)`` for forward Euler, but not for RK4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from train_paper_pmp_kkt import ProblemConfig


RK4_A: tuple[tuple[float, ...], ...] = (
    (0.0, 0.0, 0.0, 0.0),
    (0.5, 0.0, 0.0, 0.0),
    (0.0, 0.5, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
)
RK4_B: tuple[float, ...] = (1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0)


@dataclass(frozen=True)
class RK4AdjointPack:
    """Exact discrete adjoint quantities for an RK4-ZOH transcription."""

    node_costates: torch.Tensor
    stage_costates: torch.Tensor
    stage_switching: torch.Tensor
    interval_switching: torch.Tensor
    control_gradient: torch.Tensor


def tumor_g(state: torch.Tensor) -> torch.Tensor:
    return torch.log1p(state.mean(dim=-1))


def dynamics(
    state: torch.Tensor,
    control: torch.Tensor,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    growth = (
        params["r"]
        - params["phi"] * control.unsqueeze(-1)
        - params["M"] * tumor_g(state).unsqueeze(-1)
    )
    return growth * state


def running_cost(
    state: torch.Tensor,
    control: torch.Tensor,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    return (params["beta"] * state).sum(dim=-1) + params["gamma"] * control


def hamiltonian_state_derivative(
    state: torch.Tensor,
    costate: torch.Tensor,
    control: torch.Tensor,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return H_N while holding the realized control value fixed."""

    d_g = 1.0 / (state.shape[-1] + state.sum(dim=-1))
    drift = (
        params["r"]
        - params["phi"] * control.unsqueeze(-1)
        - params["M"] * tumor_g(state).unsqueeze(-1)
    )
    coupling = (costate * params["M"] * state).sum(dim=-1)
    return (
        params["beta"]
        + costate * drift
        - d_g.unsqueeze(-1) * coupling.unsqueeze(-1)
    )


def switching_function(
    state: torch.Tensor,
    costate: torch.Tensor,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    return params["gamma"] - (
        params["phi"] * costate * state
    ).sum(dim=-1)


def singular_quantities_at_points(
    state: torch.Tensor,
    control: torch.Tensor,
    costate: torch.Tensor,
    params: Dict[str, torch.Tensor],
    *,
    safe_b: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """Continuous order-one identities at same-time state/costate points.

    Arbitrary leading dimensions are accepted.  For an RK4 transcription the
    natural inputs are ``stage_states``, the ZOH control expanded over the
    four stages, and ``stage_costates``.  This avoids the inconsistent
    ``(N_k, lambda_{k+1})`` substitution inside a higher-order continuous
    identity.
    """

    if state.shape != costate.shape:
        raise ValueError("state and costate must have identical shapes")
    if control.shape != state.shape[:-1]:
        raise ValueError(
            f"expected control shape {tuple(state.shape[:-1])}, got {tuple(control.shape)}"
        )
    if safe_b <= 0.0:
        raise ValueError("safe_b must be positive")

    G = tumor_g(state)
    drift0 = params["r"] - params["M"] * G.unsqueeze(-1)
    denominator_g = state.shape[-1] + state.sum(dim=-1)
    phi_mass = (params["phi"] * state).sum(dim=-1)
    coupling = (params["M"] * costate * state).sum(dim=-1)
    rho = coupling / denominator_g

    psi = switching_function(state, costate, params)
    dot_psi = (
        params["phi"] * params["beta"] * state
    ).sum(dim=-1) - rho * phi_mass

    coupling_dot0 = (
        params["M"] * state * (-params["beta"] + rho.unsqueeze(-1))
    ).sum(dim=-1)
    denominator_dot0 = (drift0 * state).sum(dim=-1)
    rho_dot0 = (
        coupling_dot0 / denominator_g
        - rho * denominator_dot0 / denominator_g
    )
    phi_mass_dot0 = (params["phi"] * drift0 * state).sum(dim=-1)
    A = (
        (params["phi"] * params["beta"] * drift0 * state).sum(dim=-1)
        - rho_dot0 * phi_mass
        - rho * phi_mass_dot0
    )
    B = (
        -(params["phi"].square() * params["beta"] * state).sum(dim=-1)
        + rho * (params["phi"].square() * state).sum(dim=-1)
        - rho * phi_mass.square() / denominator_g
    )
    ddot_psi = A + B * control

    numerator = (params["beta"] * drift0 * state).sum(dim=-1)
    denominator = (params["beta"] * params["phi"] * state).sum(dim=-1)
    u_state = numerator / denominator.clamp_min(safe_b)
    signed_safe_b = torch.where(
        B.abs() >= safe_b,
        B,
        torch.where(
            B >= 0.0,
            torch.full_like(B, safe_b),
            torch.full_like(B, -safe_b),
        ),
    )
    return {
        "psi": psi,
        "dot_psi": dot_psi,
        "ddot_psi": ddot_psi,
        "A": A,
        "B": B,
        "u_state": u_state,
        "u_strict": -A / signed_safe_b,
        "B_valid": B.abs() >= safe_b,
    }


def rk4_zoh_step(
    state: torch.Tensor,
    control: torch.Tensor,
    dt: float,
    params: Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance one ZOH interval and return its four RK stage states.

    No positivity clamp is inserted into the map.  For the canonical n=200
    transcription all stages are positive; a caller should fail loudly if
    that ceases to hold.  Avoiding an inactive clamp keeps the numerical map
    and its discrete adjoint unambiguous.
    """

    stage1 = state
    slope1 = dynamics(stage1, control, params)
    stage2 = state + 0.5 * dt * slope1
    slope2 = dynamics(stage2, control, params)
    stage3 = state + 0.5 * dt * slope2
    slope3 = dynamics(stage3, control, params)
    stage4 = state + dt * slope3
    slope4 = dynamics(stage4, control, params)
    next_state = state + (dt / 6.0) * (
        slope1 + 2.0 * slope2 + 2.0 * slope3 + slope4
    )
    stages = torch.stack((stage1, stage2, stage3, stage4), dim=1)
    return next_state, stages


def simulate_open_loop_rk4(
    interval_controls: torch.Tensor,
    initial_state: torch.Tensor,
    cfg: ProblemConfig,
    params: Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return n+1 nodes, n controls, and four stages per interval."""

    if interval_controls.ndim == 1:
        controls = interval_controls.unsqueeze(0).expand(initial_state.shape[0], -1)
    elif interval_controls.ndim == 2:
        controls = interval_controls
    else:
        raise ValueError("interval_controls must have shape (n,) or (batch,n)")
    if controls.shape != (initial_state.shape[0], cfg.n):
        raise ValueError(
            f"expected controls {(initial_state.shape[0], cfg.n)}, got {tuple(controls.shape)}"
        )

    dt = cfg.T / cfg.n
    state = initial_state
    states = [state]
    all_stages = []
    for index in range(cfg.n):
        state, stages = rk4_zoh_step(state, controls[:, index], dt, params)
        states.append(state)
        all_stages.append(stages)
    return torch.stack(states, dim=1), controls, torch.stack(all_stages, dim=1)


def simulate_feedback_rk4(
    model: torch.nn.Module,
    initial_state: torch.Tensor,
    cfg: ProblemConfig,
    params: Dict[str, torch.Tensor],
    *,
    state_blind: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Closed-loop RK4 rollout with a left-endpoint policy query and ZOH action.

    The model interface intentionally matches ``NestedFeedbackTransformer``.
    The policy is queried once at ``(N_k,t_k)`` and the resulting action is
    held fixed at every RK stage in interval k.
    """

    dt = cfg.T / cfg.n
    batch = initial_state.shape[0]
    grid = torch.linspace(
        0.0,
        1.0,
        cfg.n + 1,
        device=initial_state.device,
        dtype=initial_state.dtype,
    )
    base_logits = model.time_logits(grid)[: cfg.n]
    state = initial_state
    states = [state]
    controls = []
    all_stages = []
    for index in range(cfg.n):
        normalized_time = torch.full(
            (batch,),
            index / cfg.n,
            device=initial_state.device,
            dtype=initial_state.dtype,
        )
        control = model.interval_action(
            base_logits[index],
            normalized_time,
            state,
            state_blind=state_blind,
        )
        state, stages = rk4_zoh_step(state, control, dt, params)
        states.append(state)
        controls.append(control)
        all_stages.append(stages)
    return (
        torch.stack(states, dim=1),
        torch.stack(controls, dim=1),
        torch.stack(all_stages, dim=1),
    )


def discrete_rk4_adjoint(
    states: torch.Tensor,
    controls: torch.Tensor,
    stage_states: torch.Tensor,
    cfg: ProblemConfig,
    params: Dict[str, torch.Tensor],
) -> RK4AdjointPack:
    """Compute the exact classical-RK4 discrete adjoint and switching.

    ``states`` and ``stage_states`` must come from :func:`rk4_zoh_step` with
    the same ``controls``.  All operations are ordinary differentiable torch
    operations.  In particular, neither the realized controls nor states are
    detached, so a later PMP residual has a complete outer gradient with
    respect to policy parameters.  The *local* Hamiltonian derivative is
    nevertheless the partial derivative at fixed control, as required by the
    teacher draft's Eq. (23).
    """

    batch = states.shape[0]
    expected_states = (batch, cfg.n + 1, cfg.m)
    expected_controls = (batch, cfg.n)
    expected_stages = (batch, cfg.n, 4, cfg.m)
    if states.shape != expected_states:
        raise ValueError(f"expected states {expected_states}, got {tuple(states.shape)}")
    if controls.shape != expected_controls:
        raise ValueError(f"expected controls {expected_controls}, got {tuple(controls.shape)}")
    if stage_states.shape != expected_stages:
        raise ValueError(
            f"expected stage_states {expected_stages}, got {tuple(stage_states.shape)}"
        )

    dt = cfg.T / cfg.n
    lambda_next = params["alpha"].expand(batch, -1)
    node_costates: list[torch.Tensor | None] = [None] * (cfg.n + 1)
    stage_costates_by_interval: list[torch.Tensor | None] = [None] * cfg.n
    stage_switching_by_interval: list[torch.Tensor | None] = [None] * cfg.n
    interval_switching_by_interval: list[torch.Tensor | None] = [None] * cfg.n
    node_costates[cfg.n] = lambda_next

    for index in range(cfg.n - 1, -1, -1):
        control = controls[:, index]
        interval_stages = stage_states[:, index]
        stage_costates: list[torch.Tensor | None] = [None] * 4
        stage_hn: list[torch.Tensor | None] = [None] * 4
        stage_psi: list[torch.Tensor | None] = [None] * 4

        # Discrete RK stage-adjoint recursion.  The b_j/b_i factors are
        # essential; omitting them does not reproduce the RK4 control
        # derivative even if the node costate looks plausible.
        for stage_index in range(3, -1, -1):
            stage_costate = lambda_next
            for later_index in range(stage_index + 1, 4):
                coefficient = (
                    RK4_B[later_index]
                    * RK4_A[later_index][stage_index]
                    / RK4_B[stage_index]
                )
                if coefficient:
                    later_hn = stage_hn[later_index]
                    assert later_hn is not None
                    stage_costate = stage_costate + dt * coefficient * later_hn
            stage_state = interval_stages[:, stage_index]
            hn = hamiltonian_state_derivative(
                stage_state, stage_costate, control, params
            )
            psi = switching_function(stage_state, stage_costate, params)
            stage_costates[stage_index] = stage_costate
            stage_hn[stage_index] = hn
            stage_psi[stage_index] = psi

        lambda_current = lambda_next
        interval_switching = torch.zeros_like(control)
        for stage_index, weight in enumerate(RK4_B):
            hn = stage_hn[stage_index]
            psi = stage_psi[stage_index]
            assert hn is not None and psi is not None
            lambda_current = lambda_current + dt * weight * hn
            interval_switching = interval_switching + weight * psi

        node_costates[index] = lambda_current
        stage_costates_by_interval[index] = torch.stack(
            [value for value in stage_costates if value is not None], dim=1
        )
        stage_switching_by_interval[index] = torch.stack(
            [value for value in stage_psi if value is not None], dim=1
        )
        interval_switching_by_interval[index] = interval_switching
        lambda_next = lambda_current

    node_tensor = torch.stack(
        [value for value in node_costates if value is not None], dim=1
    )
    stage_costate_tensor = torch.stack(
        [value for value in stage_costates_by_interval if value is not None], dim=1
    )
    stage_switching_tensor = torch.stack(
        [value for value in stage_switching_by_interval if value is not None], dim=1
    )
    interval_switching_tensor = torch.stack(
        [value for value in interval_switching_by_interval if value is not None], dim=1
    )
    return RK4AdjointPack(
        node_costates=node_tensor,
        stage_costates=stage_costate_tensor,
        stage_switching=stage_switching_tensor,
        interval_switching=interval_switching_tensor,
        control_gradient=dt * interval_switching_tensor,
    )


def rk4_running_integral(
    stage_states: torch.Tensor,
    controls: torch.Tensor,
    cfg: ProblemConfig,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """RK4 running-cost quadrature, provided for independent audit tests."""

    stage_cost = running_cost(stage_states, controls.unsqueeze(-1), params)
    weights = torch.as_tensor(
        RK4_B,
        dtype=stage_states.dtype,
        device=stage_states.device,
    )
    return (cfg.T / cfg.n) * (stage_cost * weights).sum(dim=(1, 2))
