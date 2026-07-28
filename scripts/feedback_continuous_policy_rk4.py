#!/usr/bin/env python3
"""Differentiable RK4 reference for a continuously queried feedback policy.

The existing feedback-refinement transcription queries the policy at the
left endpoint of every interval and holds that action fixed through the four
RK4 stages.  The formal evaluator instead calls ``u(N(t), t)`` continuously
inside an adaptive DOP853 solve.  This module narrows that numerical gap
without changing the learned policy or its loss:

* the feedback network is queried at every RK4 stage;
* the frozen time-branch logit follows the same PCHIP interpolant used by the
  DOP853 evaluator; and
* the continuous PMP costate equation is integrated backward with matching
  left, midpoint, and right policy queries.

All Torch operations after construction of the frozen PCHIP time logits stay
in the autograd graph.  Consequently scalar PMP losses assembled from the
returned state, costate, and control nodes differentiate only through the
original feedback-network parameters.  No physical objective, direct target,
or external control correction is introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from scipy.interpolate import PchipInterpolator

from scripts.feedback_section5_rk4_reference import (
    dynamics,
    hamiltonian_state_derivative,
    singular_quantities_at_points,
)
from train_paper_pmp_kkt import ProblemConfig


@dataclass(frozen=True)
class ContinuousFeedbackPack:
    """Node- and midpoint-aligned continuous-policy PMP diagnostics."""

    states: torch.Tensor
    node_controls: torch.Tensor
    midpoint_states: torch.Tensor
    midpoint_controls: torch.Tensor
    costates: torch.Tensor
    midpoint_costates: torch.Tensor
    quantities: Dict[str, torch.Tensor]
    midpoint_quantities: Dict[str, torch.Tensor]


def pchip_midpoint_logits(
    normalized_node_time: torch.Tensor,
    node_raw_logits: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the frozen time-logit PCHIP at interval midpoints.

    The DOP853 diagnostic constructs exactly this interpolant.  The time
    Transformer is frozen during feedback refinement, so moving these values
    through NumPy does not remove any trainable gradient.
    """

    if normalized_node_time.ndim != 1 or node_raw_logits.ndim != 1:
        raise ValueError("time and raw-logit nodes must be one-dimensional")
    if normalized_node_time.shape != node_raw_logits.shape:
        raise ValueError("time and raw-logit nodes must have matching shapes")
    if normalized_node_time.numel() < 2:
        raise ValueError("at least two time nodes are required")
    time_np = normalized_node_time.detach().cpu().numpy().astype(np.float64)
    raw_np = node_raw_logits.detach().cpu().numpy().astype(np.float64)
    if not np.all(np.diff(time_np) > 0.0):
        raise ValueError("time nodes must be strictly increasing")
    midpoint_np = 0.5 * (time_np[:-1] + time_np[1:])
    values = PchipInterpolator(time_np, raw_np)(midpoint_np)
    return torch.as_tensor(
        values,
        device=node_raw_logits.device,
        dtype=node_raw_logits.dtype,
    )


def _policy_action(
    model: torch.nn.Module,
    raw_logit: torch.Tensor,
    normalized_time: torch.Tensor,
    state: torch.Tensor,
    *,
    state_mode: str,
) -> torch.Tensor:
    batch = state.shape[0]
    if raw_logit.ndim == 0:
        raw_logit = raw_logit.expand(batch)
    if normalized_time.ndim == 0:
        normalized_time = normalized_time.expand(batch)
    return model.interval_action(
        raw_logit,
        normalized_time,
        state,
        state_mode=state_mode,
    )


def _controls_at_nodes(
    model: torch.nn.Module,
    states: torch.Tensor,
    normalized_time: torch.Tensor,
    raw_logits: torch.Tensor,
    *,
    state_mode: str,
) -> torch.Tensor:
    """Vectorized policy queries for a complete node trajectory."""

    batch, node_count, phenotype_count = states.shape
    flat_state = states.reshape(batch * node_count, phenotype_count)
    flat_time = (
        normalized_time.unsqueeze(0)
        .expand(batch, -1)
        .reshape(batch * node_count)
    )
    flat_raw = (
        raw_logits.unsqueeze(0)
        .expand(batch, -1)
        .reshape(batch * node_count)
    )
    return _policy_action(
        model,
        flat_raw,
        flat_time,
        flat_state,
        state_mode=state_mode,
    ).reshape(batch, node_count)


def continuous_feedback_state_rk4(
    model: torch.nn.Module,
    initial_state: torch.Tensor,
    cfg: ProblemConfig,
    normalized_node_time: torch.Tensor,
    node_raw_logits: torch.Tensor,
    midpoint_raw_logits: torch.Tensor,
    params: Dict[str, torch.Tensor],
    *,
    state_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate ``dN/dt=f(N,u(N,t))`` with stagewise policy queries."""

    expected_nodes = cfg.n + 1
    if normalized_node_time.shape != (expected_nodes,):
        raise ValueError(
            f"expected {expected_nodes} time nodes, got "
            f"{tuple(normalized_node_time.shape)}"
        )
    if node_raw_logits.shape != (expected_nodes,):
        raise ValueError("node raw logits do not match the integration grid")
    if midpoint_raw_logits.shape != (cfg.n,):
        raise ValueError("midpoint raw logits do not match the intervals")

    step = cfg.T / cfg.n
    batch = initial_state.shape[0]
    state = initial_state
    states = [state]
    midpoint_states: list[torch.Tensor] = []
    for index in range(cfg.n):
        left_time = normalized_node_time[index].expand(batch)
        midpoint_time = (
            0.5
            * (
                normalized_node_time[index]
                + normalized_node_time[index + 1]
            )
        ).expand(batch)
        right_time = normalized_node_time[index + 1].expand(batch)

        control1 = _policy_action(
            model,
            node_raw_logits[index],
            left_time,
            state,
            state_mode=state_mode,
        )
        slope1 = dynamics(state, control1, params)

        stage2 = state + 0.5 * step * slope1
        control2 = _policy_action(
            model,
            midpoint_raw_logits[index],
            midpoint_time,
            stage2,
            state_mode=state_mode,
        )
        slope2 = dynamics(stage2, control2, params)

        stage3 = state + 0.5 * step * slope2
        control3 = _policy_action(
            model,
            midpoint_raw_logits[index],
            midpoint_time,
            stage3,
            state_mode=state_mode,
        )
        slope3 = dynamics(stage3, control3, params)

        stage4 = state + step * slope3
        control4 = _policy_action(
            model,
            node_raw_logits[index + 1],
            right_time,
            stage4,
            state_mode=state_mode,
        )
        slope4 = dynamics(stage4, control4, params)

        next_state = state + (step / 6.0) * (
            slope1 + 2.0 * slope2 + 2.0 * slope3 + slope4
        )
        if not bool(torch.isfinite(next_state).all()):
            raise FloatingPointError("continuous-policy state became nonfinite")
        if not bool((next_state > 0.0).all()):
            raise FloatingPointError(
                "continuous-policy RK4 step produced a nonpositive state"
            )
        midpoint_states.append(0.5 * (stage2 + stage3))
        state = next_state
        states.append(state)
    return torch.stack(states, dim=1), torch.stack(midpoint_states, dim=1)


def continuous_pmp_costate_rk4(
    states: torch.Tensor,
    node_controls: torch.Tensor,
    midpoint_states: torch.Tensor,
    midpoint_controls: torch.Tensor,
    cfg: ProblemConfig,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Integrate the continuous PMP adjoint backward with RK4.

    ``H_N`` is evaluated at the realized action while holding that action
    fixed, as required by the PMP partial derivative.  The realized action
    itself remains graph-connected to the network for the outer refinement
    gradient.
    """

    batch = states.shape[0]
    expected_states = (batch, cfg.n + 1, cfg.m)
    if states.shape != expected_states:
        raise ValueError(f"expected states {expected_states}")
    if node_controls.shape != (batch, cfg.n + 1):
        raise ValueError("node controls do not match the trajectory")
    if midpoint_states.shape != (batch, cfg.n, cfg.m):
        raise ValueError("midpoint states do not match the trajectory")
    if midpoint_controls.shape != (batch, cfg.n):
        raise ValueError("midpoint controls do not match the trajectory")

    step = cfg.T / cfg.n
    costate = params["alpha"].expand(batch, -1)
    values: list[torch.Tensor | None] = [None] * (cfg.n + 1)
    values[cfg.n] = costate

    def rhs(
        state: torch.Tensor,
        current_costate: torch.Tensor,
        control: torch.Tensor,
    ) -> torch.Tensor:
        return -hamiltonian_state_derivative(
            state,
            current_costate,
            control,
            params,
        )

    for index in range(cfg.n - 1, -1, -1):
        slope1 = rhs(
            states[:, index + 1],
            costate,
            node_controls[:, index + 1],
        )
        slope2 = rhs(
            midpoint_states[:, index],
            costate - 0.5 * step * slope1,
            midpoint_controls[:, index],
        )
        slope3 = rhs(
            midpoint_states[:, index],
            costate - 0.5 * step * slope2,
            midpoint_controls[:, index],
        )
        slope4 = rhs(
            states[:, index],
            costate - step * slope3,
            node_controls[:, index],
        )
        costate = costate - (step / 6.0) * (
            slope1 + 2.0 * slope2 + 2.0 * slope3 + slope4
        )
        values[index] = costate
    return torch.stack(values, dim=1)  # type: ignore[arg-type]


def hermite_midpoint_costates(
    states: torch.Tensor,
    node_controls: torch.Tensor,
    costates: torch.Tensor,
    cfg: ProblemConfig,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return fourth-order node-consistent costates at RK4 midpoints.

    On an interval ``[t_k,t_{k+1}]`` with width ``h``, let
    ``q_k=d lambda/dt(t_k)`` and ``q_{k+1}=d lambda/dt(t_{k+1})``.  Cubic
    Hermite interpolation evaluated at the midpoint gives

    ``lambda_{k+1/2} = (lambda_k + lambda_{k+1})/2
                       + h (q_k - q_{k+1})/8``.

    With the fourth-order RK4 node costates this is a fourth-order midpoint
    value.  It is preferable to either RK4 half-stage costate alone: those
    stage values are internal slope predictors, not a collocated approximation
    to ``lambda(t_{k+1/2})``.  All operations remain differentiable.
    """

    batch = states.shape[0]
    expected_states = (batch, cfg.n + 1, cfg.m)
    if states.shape != expected_states:
        raise ValueError(f"expected states {expected_states}")
    if node_controls.shape != (batch, cfg.n + 1):
        raise ValueError("node controls do not match the trajectory")
    if costates.shape != expected_states:
        raise ValueError("node costates do not match the trajectory")

    def rhs(
        state: torch.Tensor,
        current_costate: torch.Tensor,
        control: torch.Tensor,
    ) -> torch.Tensor:
        return -hamiltonian_state_derivative(
            state,
            current_costate,
            control,
            params,
        )

    left_rhs = rhs(
        states[:, :-1],
        costates[:, :-1],
        node_controls[:, :-1],
    )
    right_rhs = rhs(
        states[:, 1:],
        costates[:, 1:],
        node_controls[:, 1:],
    )
    step = cfg.T / cfg.n
    return (
        0.5 * (costates[:, :-1] + costates[:, 1:])
        + (step / 8.0) * (left_rhs - right_rhs)
    )


def continuous_feedback_pmp_pack(
    model: torch.nn.Module,
    initial_state: torch.Tensor,
    cfg: ProblemConfig,
    normalized_node_time: torch.Tensor,
    node_raw_logits: torch.Tensor,
    midpoint_raw_logits: torch.Tensor,
    params: Dict[str, torch.Tensor],
    *,
    state_mode: str,
) -> ContinuousFeedbackPack:
    """Return a node-collocated differentiable continuous-policy PMP pack."""

    states, midpoint_states = continuous_feedback_state_rk4(
        model,
        initial_state,
        cfg,
        normalized_node_time,
        node_raw_logits,
        midpoint_raw_logits,
        params,
        state_mode=state_mode,
    )
    node_controls = _controls_at_nodes(
        model,
        states,
        normalized_node_time,
        node_raw_logits,
        state_mode=state_mode,
    )
    midpoint_time = 0.5 * (
        normalized_node_time[:-1] + normalized_node_time[1:]
    )
    midpoint_controls = _controls_at_nodes(
        model,
        midpoint_states,
        midpoint_time,
        midpoint_raw_logits,
        state_mode=state_mode,
    )
    costates = continuous_pmp_costate_rk4(
        states,
        node_controls,
        midpoint_states,
        midpoint_controls,
        cfg,
        params,
    )
    midpoint_costates = hermite_midpoint_costates(
        states,
        node_controls,
        costates,
        cfg,
        params,
    )
    quantities = singular_quantities_at_points(
        states[:, :-1],
        node_controls[:, :-1],
        costates[:, :-1],
        params,
    )
    midpoint_quantities = singular_quantities_at_points(
        midpoint_states,
        midpoint_controls,
        midpoint_costates,
        params,
    )
    return ContinuousFeedbackPack(
        states=states,
        node_controls=node_controls,
        midpoint_states=midpoint_states,
        midpoint_controls=midpoint_controls,
        costates=costates,
        midpoint_costates=midpoint_costates,
        quantities=quantities,
        midpoint_quantities=midpoint_quantities,
    )
