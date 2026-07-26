#!/usr/bin/env python3
"""Section-5 state--time feedback training from PMP singular conditions.

The physical objective is used only for evaluation, never as a training-loss
term.  Both the legacy Euler collocation and an RK4-ZOH transcription with its
matching four-stage discrete adjoint are available explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_paper_pmp_kkt import (  # noqa: E402
    ProblemConfig,
    TimeTransformer,
    build_params,
    parse_hidden,
    set_seed,
    time_features,
)
from feedback_section5_rk4_reference import (  # noqa: E402
    RK4_B,
    discrete_rk4_adjoint,
    rk4_zoh_step as rk4_stage_step,
    simulate_open_loop_rk4 as simulate_open_loop_rk4_reference,
    singular_quantities_at_points,
)


class NestedFeedbackTransformer(nn.Module):
    """Time-only Transformer plus a zero-initialized state correction.

    With a zero correction head this policy exactly reproduces the supplied
    time-only Transformer on the common grid.  The state branch is the small
    MLP ``h_theta_N`` in Section 5.5.1 of the teacher draft.
    """

    def __init__(
        self,
        m: int,
        umax: float,
        state_scale: float,
        state_hidden: tuple[int, ...],
        d_model: int,
        heads: int,
        layers: int,
        init_u: float,
        correction_gain: float = 1.0,
        state_feature_mode: str = "log_absolute",
        center_state_correction: bool = False,
        action_temperature: float = 1.0,
        action_scale: float = 1.0,
        action_parameterization: str = "logit-temperature",
    ) -> None:
        super().__init__()
        self.m = int(m)
        self.umax = float(umax)
        self.state_log_scale = math.log1p(float(state_scale))
        self.correction_gain = float(correction_gain)
        self.center_state_correction = bool(center_state_correction)
        self.action_temperature = float(action_temperature)
        if self.action_temperature <= 0.0:
            raise ValueError("action_temperature must be positive")
        self.action_scale = float(action_scale)
        if self.action_scale <= 0.0:
            raise ValueError("action_scale must be positive")
        if action_parameterization not in {
            "logit-temperature",
            "linear-raw-box",
        }:
            raise ValueError(
                f"unknown action parameterization: {action_parameterization}"
            )
        self.action_parameterization = action_parameterization
        if state_feature_mode not in {
            "log_absolute",
            "relative_nominal",
            "burden_composition",
            "total_burden",
        }:
            raise ValueError(f"unknown state feature mode: {state_feature_mode}")
        self.state_feature_mode = state_feature_mode
        self.register_buffer(
            "nominal_reference", torch.empty(0), persistent=False
        )
        self.register_buffer("feature_r", torch.empty(0), persistent=False)
        self.register_buffer("feature_phi", torch.empty(0), persistent=False)
        self.time_branch = TimeTransformer(
            d_model, heads, layers, umax, init_u
        )

        widths = (6 + m + 3, *state_hidden, 1)
        layers_state: list[nn.Module] = []
        for index, (in_features, out_features) in enumerate(
            zip(widths[:-1], widths[1:])
        ):
            layers_state.append(nn.Linear(in_features, out_features))
            if index < len(widths) - 2:
                layers_state.append(nn.Tanh())
        self.state_branch = nn.Sequential(*layers_state)
        self.reset_state_branch()

    def reset_state_branch(self) -> None:
        linears = [
            module
            for module in self.state_branch.modules()
            if isinstance(module, nn.Linear)
        ]
        for layer in linears[:-1]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        # This is the exact nested-reduction initialization in Section 5.5.1.
        nn.init.zeros_(linears[-1].weight)
        nn.init.zeros_(linears[-1].bias)

    def load_time_checkpoint(self, path: Path) -> dict:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        problem = checkpoint.get("problem", {})
        if int(problem.get("m", self.m)) != self.m:
            raise ValueError("time-only checkpoint phenotype count mismatch")
        if not math.isclose(float(problem.get("umax", self.umax)), self.umax):
            raise ValueError("time-only checkpoint umax mismatch")
        source_state = checkpoint["model_state"]
        wrapper = dict(checkpoint.get("wrapper", {}))
        wrapper_class = str(wrapper.get("class", ""))
        if wrapper_class == "FixedBoxProjection":
            source_state = {
                key.removeprefix("base."): value
                for key, value in source_state.items()
                if key.startswith("base.")
            }
            self.action_scale = float(wrapper["scale"])
            self.action_temperature = float(wrapper.get("temperature", 1.0))
            self.action_parameterization = "logit-temperature"
        elif wrapper_class == "LinearRawBoxProjection":
            source_state = {
                key.removeprefix("base."): value
                for key, value in source_state.items()
                if key.startswith("base.")
            }
            self.action_scale = 1.0
            self.action_temperature = 1.0
            self.action_parameterization = "linear-raw-box"
        elif wrapper_class == "BoundaryProjectedControl":
            source_state = {
                key.removeprefix("base."): value
                for key, value in source_state.items()
                if key.startswith("base.")
            }
            self.action_scale = float(
                wrapper.get("scale", wrapper.get("initial_scale", 1.0))
            )
        elif any(key.startswith("time_branch.") for key in source_state):
            source_state = {
                key.removeprefix("time_branch."): value
                for key, value in source_state.items()
                if key.startswith("time_branch.")
            }
        self.time_branch.load_state_dict(source_state)
        return checkpoint

    def time_logits(self, normalized_time_grid: torch.Tensor) -> torch.Tensor:
        hidden = self.time_branch.input(time_features(normalized_time_grid))
        hidden = self.time_branch.encoder(hidden.unsqueeze(0)).squeeze(0)
        return self.time_branch.output(hidden).squeeze(-1)

    def set_nominal_reference(self, states: torch.Tensor) -> None:
        if states.ndim != 2 or states.shape[1] != self.m:
            raise ValueError("nominal reference must have shape (steps, m)")
        self.nominal_reference = states.detach().clone()

    def set_feature_vectors(self, r: torch.Tensor, phi: torch.Tensor) -> None:
        if r.shape != (self.m,) or phi.shape != (self.m,):
            raise ValueError("feature vectors r and phi must both have shape (m,)")
        self.feature_r = r.detach().clone()
        self.feature_phi = phi.detach().clone()

    def nominal_state_at(self, normalized_time: torch.Tensor) -> torch.Tensor:
        if self.nominal_reference.numel() == 0:
            raise RuntimeError("a nominal reference trajectory has not been set")
        last = self.nominal_reference.shape[0] - 1
        indices = torch.round(normalized_time * last).long().clamp(0, last)
        return self.nominal_reference[indices]

    def state_features(
        self, normalized_time: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        if normalized_time.ndim == 0:
            normalized_time = normalized_time.expand(state.shape[0])
        if self.state_feature_mode == "log_absolute":
            component = torch.log1p(state) / self.state_log_scale
        else:
            if self.nominal_reference.numel() == 0:
                raise RuntimeError(
                    "relative_nominal features require a nominal reference trajectory"
                )
            reference = self.nominal_state_at(normalized_time)
            if self.state_feature_mode == "relative_nominal":
                component = (state - reference) / reference.clamp_min(1e-8)
                summaries = (
                    component.mean(dim=-1, keepdim=True),
                    component.std(dim=-1, keepdim=True, unbiased=False),
                    component.max(dim=-1, keepdim=True).values,
                )
            elif self.state_feature_mode == "total_burden":
                eps = torch.finfo(state.dtype).eps
                total = state.sum(dim=-1, keepdim=True).clamp_min(eps)
                reference_total = reference.sum(dim=-1, keepdim=True).clamp_min(eps)
                log_total_ratio = torch.log(total / reference_total)
                zeros = torch.zeros_like(log_total_ratio)
                # Preserve the state-branch parameter count while exposing only
                # the scalar total burden.  The phenotype-component slots are
                # deliberately zero, so no composition information can leak
                # into this controlled ablation.
                component = torch.zeros_like(state)
                summaries = (log_total_ratio, zeros, zeros)
            else:
                if self.feature_r.numel() == 0 or self.feature_phi.numel() == 0:
                    raise RuntimeError(
                        "burden_composition features require r and phi vectors"
                    )
                eps = torch.finfo(state.dtype).eps
                total = state.sum(dim=-1, keepdim=True).clamp_min(eps)
                reference_total = reference.sum(dim=-1, keepdim=True).clamp_min(eps)
                composition = state / total
                reference_composition = reference / reference_total
                log_composition = torch.log(composition.clamp_min(eps))
                log_reference = torch.log(reference_composition.clamp_min(eps))
                component = (log_composition - log_composition.mean(dim=-1, keepdim=True)) - (
                    log_reference - log_reference.mean(dim=-1, keepdim=True)
                )
                delta_composition = composition - reference_composition
                r_centered = self.feature_r - self.feature_r.mean()
                phi_centered = self.feature_phi - self.feature_phi.mean()
                r_scale = r_centered.square().mean().sqrt().clamp_min(eps)
                phi_scale = phi_centered.square().mean().sqrt().clamp_min(eps)
                summaries = (
                    torch.log(total / reference_total),
                    (delta_composition * r_centered).sum(dim=-1, keepdim=True)
                    / r_scale,
                    (delta_composition * phi_centered).sum(dim=-1, keepdim=True)
                    / phi_scale,
                )
        if self.state_feature_mode == "log_absolute":
            summaries = (
                component.mean(dim=-1, keepdim=True),
                component.std(dim=-1, keepdim=True, unbiased=False),
                component.max(dim=-1, keepdim=True).values,
            )
        features = torch.cat(
            [
                time_features(normalized_time),
                component,
                *summaries,
            ],
            dim=-1,
        )
        return features

    def state_logits(self, normalized_time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        features = self.state_features(normalized_time, state)
        value = self.state_branch(features).squeeze(-1)
        if self.center_state_correction:
            reference = self.nominal_state_at(normalized_time)
            reference_features = self.state_features(normalized_time, reference)
            reference_value = self.state_branch(reference_features).squeeze(-1)
            value = value - reference_value
        return value

    def interval_action(
        self,
        base_logit: torch.Tensor,
        normalized_time: torch.Tensor,
        state: torch.Tensor,
        *,
        state_blind: bool = False,
        state_mode: str | None = None,
    ) -> torch.Tensor:
        if state_mode is None:
            state_mode = "fixed_query" if state_blind else "feedback"
        if state_mode == "w_zero":
            correction = torch.zeros(
                state.shape[0], device=state.device, dtype=state.dtype
            )
        elif state_mode == "fixed_query":
            correction = self.state_logits(
                normalized_time, torch.full_like(state, 10.0)
            )
        elif state_mode == "feedback":
            correction = self.state_logits(normalized_time, state)
        else:
            raise ValueError(f"unknown state mode: {state_mode}")
        combined_logit = base_logit + self.correction_gain * correction
        if self.action_parameterization == "linear-raw-box":
            return torch.clamp(combined_logit, 0.0, self.umax)
        return torch.clamp(
            self.action_scale
            * self.umax
            * torch.sigmoid(combined_logit / self.action_temperature),
            0.0,
            self.umax,
        )


def sample_componentwise_initial_states(
    count: int,
    cfg: ProblemConfig,
    radius: float,
    device: torch.device,
    dtype: torch.dtype,
    *,
    generator: torch.Generator | None = None,
    include_nominal: bool = False,
) -> torch.Tensor:
    """Sample N0_i = n0 * (1 + radius * Z_i), Z_i iid Uniform[-1,1]."""

    z = 2.0 * torch.rand(
        count,
        cfg.m,
        device=device,
        dtype=dtype,
        generator=generator,
    ) - 1.0
    states = cfg.n0 * (1.0 + float(radius) * z)
    if include_nominal and count:
        states[0] = cfg.n0
    return states


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


def rk4_state_step(
    state: torch.Tensor,
    control: torch.Tensor,
    step: float,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """One differentiable ZOH RK4 state step."""

    k1 = dynamics(state, control, params)
    k2 = dynamics(state + 0.5 * step * k1, control, params)
    k3 = dynamics(state + 0.5 * step * k2, control, params)
    k4 = dynamics(state + step * k3, control, params)
    return state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate_feedback(
    model: NestedFeedbackTransformer,
    initial_state: torch.Tensor,
    cfg: ProblemConfig,
    params: Dict[str, torch.Tensor],
    *,
    state_blind: bool = False,
    state_mode: str | None = None,
    integrator: str = "euler",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return n+1 states, n interval controls, and raw next-state minima."""

    dt = cfg.T / cfg.n
    batch = initial_state.shape[0]
    grid = torch.linspace(
        0.0, 1.0, cfg.n + 1, device=initial_state.device, dtype=initial_state.dtype
    )
    base_logits = model.time_logits(grid)[: cfg.n]
    state = initial_state
    states = [state]
    controls = []
    raw_minima = []
    for index in range(cfg.n):
        time = torch.full(
            (batch,),
            index / cfg.n,
            device=initial_state.device,
            dtype=initial_state.dtype,
        )
        control = model.interval_action(
            base_logits[index],
            time,
            state,
            state_blind=state_blind,
            state_mode=state_mode,
        )
        if integrator == "euler":
            raw_next = state + dt * dynamics(state, control, params)
        elif integrator == "rk4":
            raw_next = rk4_state_step(state, control, dt, params)
        else:
            raise ValueError(f"unknown training integrator: {integrator}")
        controls.append(control)
        raw_minima.append(raw_next.amin(dim=-1))
        state = raw_next.clamp_min(1e-8)
        states.append(state)
    return (
        torch.stack(states, dim=1),
        torch.stack(controls, dim=1),
        torch.stack(raw_minima, dim=1),
    )


def simulate_feedback_rk4_stagewise(
    model: NestedFeedbackTransformer,
    initial_state: torch.Tensor,
    cfg: ProblemConfig,
    params: Dict[str, torch.Tensor],
    *,
    state_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return RK4 nodes, interval controls, four stages, and positivity minima."""

    dt = cfg.T / cfg.n
    batch = initial_state.shape[0]
    grid = torch.linspace(
        0.0, 1.0, cfg.n + 1, device=initial_state.device, dtype=initial_state.dtype
    )
    base_logits = model.time_logits(grid)[: cfg.n]
    state = initial_state
    states = [state]
    controls = []
    all_stages = []
    raw_minima = []
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
            state_mode=state_mode,
        )
        next_state, stages = rk4_stage_step(state, control, dt, params)
        controls.append(control)
        all_stages.append(stages)
        raw_minima.append(
            torch.cat((stages, next_state.unsqueeze(1)), dim=1).amin(dim=(1, 2))
        )
        state = next_state
        states.append(state)
    return (
        torch.stack(states, dim=1),
        torch.stack(controls, dim=1),
        torch.stack(all_stages, dim=1),
        torch.stack(raw_minima, dim=1),
    )


def simulate_open_loop(
    interval_control: torch.Tensor,
    initial_state: torch.Tensor,
    cfg: ProblemConfig,
    params: Dict[str, torch.Tensor],
    *,
    integrator: str = "euler",
) -> torch.Tensor:
    dt = cfg.T / cfg.n
    state = initial_state
    states = [state]
    for index in range(cfg.n):
        control = (
            interval_control[:, index]
            if interval_control.ndim == 2
            else interval_control[index]
        )
        if integrator == "euler":
            state = state + dt * dynamics(state, control, params)
        elif integrator == "rk4":
            state = rk4_state_step(state, control, dt, params)
        else:
            raise ValueError(f"unknown training integrator: {integrator}")
        state = state.clamp_min(1e-8)
        states.append(state)
    return torch.stack(states, dim=1)


def dH_dN(
    state: torch.Tensor,
    costate: torch.Tensor,
    control: torch.Tensor,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    dG = 1.0 / (state.shape[-1] + state.sum(dim=-1))
    drift = (
        params["r"]
        - params["phi"] * control.unsqueeze(-1)
        - params["M"] * tumor_g(state).unsqueeze(-1)
    )
    coupling = (costate * params["M"] * state).sum(dim=-1)
    return (
        params["beta"]
        + costate * drift
        - dG.unsqueeze(-1) * coupling.unsqueeze(-1)
    )


def compute_costate(
    states: torch.Tensor,
    controls: torch.Tensor,
    cfg: ProblemConfig,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Matching adjoint for the left-endpoint forward-Euler transcription."""

    dt = cfg.T / cfg.n
    batch = states.shape[0]
    costate = params["alpha"].expand(batch, -1)
    values: list[torch.Tensor | None] = [None] * (cfg.n + 1)
    values[cfg.n] = costate
    for index in range(cfg.n - 1, -1, -1):
        costate = costate + dt * dH_dN(
            states[:, index], costate, controls[:, index], params
        )
        values[index] = costate
    return torch.stack(values, dim=1)  # type: ignore[arg-type]


def dynamics_jacobian_transpose_vector(
    state: torch.Tensor,
    control: torch.Tensor,
    vector: torch.Tensor,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return ``f_N(state, control)^T vector`` without forming a Jacobian."""

    drift = (
        params["r"]
        - params["phi"] * control.unsqueeze(-1)
        - params["M"] * tumor_g(state).unsqueeze(-1)
    )
    denominator = state.shape[-1] + state.sum(dim=-1)
    coupling = (vector * params["M"] * state).sum(dim=-1) / denominator
    return drift * vector - coupling.unsqueeze(-1)


def rk4_step_discrete_adjoint(
    state: torch.Tensor,
    control: torch.Tensor,
    costate_next: torch.Tensor,
    step: float,
    params: Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Matching discrete adjoint and control gradient for one ZOH RK4 step.

    The recursion differentiates the same four-stage state map and four-stage
    running-cost quadrature used by the independent RK4 objective.  It is
    written as explicit reverse accumulation so ``costate_next`` remains an
    input to the recursion and the result remains differentiable with respect
    to the realized state, control, and later costate values.
    """

    k1 = dynamics(state, control, params)
    state2 = state + 0.5 * step * k1
    k2 = dynamics(state2, control, params)
    state3 = state + 0.5 * step * k2
    k3 = dynamics(state3, control, params)
    state4 = state + step * k3
    k4 = dynamics(state4, control, params)

    beta = params["beta"].expand_as(state)
    a_k4 = (step / 6.0) * costate_next
    a_state4 = (step / 6.0) * beta + dynamics_jacobian_transpose_vector(
        state4, control, a_k4, params
    )
    a_k3 = (step / 3.0) * costate_next + step * a_state4
    a_state3 = (step / 3.0) * beta + dynamics_jacobian_transpose_vector(
        state3, control, a_k3, params
    )
    a_k2 = (step / 3.0) * costate_next + 0.5 * step * a_state3
    a_state2 = (step / 3.0) * beta + dynamics_jacobian_transpose_vector(
        state2, control, a_k2, params
    )
    a_k1 = (step / 6.0) * costate_next + 0.5 * step * a_state2
    costate = (
        costate_next
        + (step / 6.0) * beta
        + a_state4
        + a_state3
        + a_state2
        + dynamics_jacobian_transpose_vector(state, control, a_k1, params)
    )

    control_gradient = step * params["gamma"].expand_as(control)
    for stage_state, stage_adjoint in (
        (state, a_k1),
        (state2, a_k2),
        (state3, a_k3),
        (state4, a_k4),
    ):
        control_gradient = control_gradient - (
            params["phi"] * stage_state * stage_adjoint
        ).sum(dim=-1)
    return costate, control_gradient


def compute_costate_rk4(
    states: torch.Tensor,
    controls: torch.Tensor,
    cfg: ProblemConfig,
    params: Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return RK4-matching node costates and interval reduced gradients."""

    step = cfg.T / cfg.n
    batch = states.shape[0]
    costate = params["alpha"].expand(batch, -1)
    values: list[torch.Tensor | None] = [None] * (cfg.n + 1)
    gradients: list[torch.Tensor | None] = [None] * cfg.n
    values[cfg.n] = costate
    for index in range(cfg.n - 1, -1, -1):
        costate, control_gradient = rk4_step_discrete_adjoint(
            states[:, index],
            controls[:, index],
            costate,
            step,
            params,
        )
        values[index] = costate
        gradients[index] = control_gradient
    return (
        torch.stack(values, dim=1),  # type: ignore[arg-type]
        torch.stack(gradients, dim=1),  # type: ignore[arg-type]
    )


def singular_quantities(
    states: torch.Tensor,
    controls: torch.Tensor,
    costates: torch.Tensor,
    cfg: ProblemConfig,
    params: Dict[str, torch.Tensor],
    *,
    safe_b: float = 1e-8,
    costate_pairing: str = "next",
) -> Dict[str, torch.Tensor]:
    """Continuous singular identities collocated at (N_k, lambda_{k+1})."""

    state = states[:, :-1]
    if costate_pairing == "next":
        costate = costates[:, 1:]
    elif costate_pairing == "current":
        costate = costates[:, :-1]
    else:
        raise ValueError(f"unknown costate pairing: {costate_pairing}")
    G = tumor_g(state)
    drift0 = params["r"] - params["M"] * G.unsqueeze(-1)
    D = cfg.m + state.sum(dim=-1)
    Q = (params["phi"] * state).sum(dim=-1)
    C = (params["M"] * costate * state).sum(dim=-1)
    rho = C / D

    psi = params["gamma"] - (params["phi"] * costate * state).sum(dim=-1)
    dot_psi = (params["phi"] * params["beta"] * state).sum(dim=-1) - rho * Q

    Cdot0 = (params["M"] * state * (-params["beta"] + rho.unsqueeze(-1))).sum(dim=-1)
    Ddot0 = (drift0 * state).sum(dim=-1)
    rhodot0 = Cdot0 / D - rho * Ddot0 / D
    Qdot0 = (params["phi"] * drift0 * state).sum(dim=-1)
    A = (
        (params["phi"] * params["beta"] * drift0 * state).sum(dim=-1)
        - rhodot0 * Q
        - rho * Qdot0
    )
    B = (
        -(params["phi"].square() * params["beta"] * state).sum(dim=-1)
        + rho * (params["phi"].square() * state).sum(dim=-1)
        - rho * Q.square() / D
    )
    ddot_psi = A + B * controls

    numerator = (params["beta"] * drift0 * state).sum(dim=-1)
    denominator = (params["beta"] * params["phi"] * state).sum(dim=-1)
    u_state = numerator / denominator.clamp_min(safe_b)
    signed_safe_B = torch.where(
        B.abs() >= safe_b,
        B,
        torch.where(B >= 0.0, torch.full_like(B, safe_b), torch.full_like(B, -safe_b)),
    )
    u_strict = -A / signed_safe_B
    return {
        "state": state,
        "costate_interval": costate,
        "psi": psi,
        "dot_psi": dot_psi,
        "ddot_psi": ddot_psi,
        "A": A,
        "B": B,
        "u_state": u_state,
        "u_strict": u_strict,
        "B_valid": B.abs() >= safe_b,
    }


def persistence_gate(point_gate: torch.Tensor, window: int) -> torch.Tensor:
    if window <= 1:
        return point_gate
    if window % 2 == 0:
        raise ValueError("persistence window must be odd")
    half = window // 2
    persistent = -F.max_pool1d(
        -point_gate.unsqueeze(1), kernel_size=window, stride=1, padding=half
    ).squeeze(1)
    persistent[:, :half] = 0.0
    persistent[:, -half:] = 0.0
    return persistent


def box_violation(value: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    return torch.relu(lower - value) + torch.relu(value - upper)


def compose_section5_optimality_loss(
    controls: torch.Tensor,
    quantities: Dict[str, torch.Tensor],
    cfg: ProblemConfig,
    args: argparse.Namespace,
) -> Dict[str, torch.Tensor]:
    """Compose either the literal draft loss or the numerical pilot variant.

    ``literal`` follows Sections 5.3--5.4 of the teacher draft as written:
    the closed-form gate uses ``psi`` only, the derivative gate uses ``psi``
    and ``dot(psi)``, both indicators include the stated ``B <= 0`` test, and
    the boundary loss uses the unnormalised control.  The smoothness term is
    added separately by :func:`section5_loss`, as requested after the draft.

    ``lc_live`` is a one-factor numerical correction to the literal derivative
    formulation: it omits ``B <= 0`` from the derivative indicator so that the
    displayed ``ReLU(B)^2`` term remains active when its sign condition is
    violated.  The closed-form formulation is identical to ``literal``.

    ``pilot`` preserves the first-pilot implementation: both gates also
    require temporal persistence, the classifier is detached by default, and
    the Legendre--Clebsch residual remains live when ``B > 0``.  Keeping the
    variants explicit prevents an implementation correction from being
    silently presented as a literal reproduction of the draft equations.
    """

    variant = getattr(args, "loss_variant", "pilot")
    if variant not in {"literal", "lc_live", "pilot"}:
        raise ValueError(f"unknown Section-5 loss variant: {variant}")

    psi = quantities["psi"]
    boundary_psi = quantities.get("boundary_psi", psi)
    dot_psi = quantities["dot_psi"]
    B = quantities["B"]
    quadrature_weight = quantities.get("quadrature_weight")

    def reduce_mean(value: torch.Tensor) -> torch.Tensor:
        if quadrature_weight is None:
            return value.mean()
        return (value * quadrature_weight).sum(dim=-1).mean()
    b_min = float(getattr(args, "b_min", 1e-8))
    if b_min <= 0.0:
        raise ValueError("b_min must be positive")
    psi_scale = float(getattr(args, "psi_scale", 1.0))
    dot_scale = float(getattr(args, "dot_scale", 1.0))
    ddot_scale = float(getattr(args, "ddot_scale", 1.0))
    B_scale = float(getattr(args, "B_scale", 1.0))
    if min(psi_scale, dot_scale, ddot_scale, B_scale) <= 0.0:
        raise ValueError("residual scales must be positive")

    psi_gate = torch.sigmoid(
        (args.singular_eps - psi.abs()) / args.singular_tau
    )
    dot_gate = torch.sigmoid(
        (args.dot_eps - dot_psi.abs()) / args.dot_tau
    )

    if variant in {"literal", "lc_live"}:
        # The displayed equations contain no stop-gradient or persistence
        # operation, so the smooth gate remains in the autodiff graph.
        point_gate = psi_gate if args.option == "cf" else psi_gate * dot_gate
        gate_mode = getattr(args, "gate_gradient_mode", "live")
        if gate_mode == "live":
            gate_for_loss = point_gate
        elif gate_mode == "detached":
            gate_for_loss = point_gate.detach()
        else:
            raise ValueError(f"unknown gate gradient mode: {gate_mode}")
        nonsingular_loss = (
            torch.relu(boundary_psi) * controls
            + torch.relu(-boundary_psi) * (cfg.umax - controls)
        ).square()
        invalid_component = torch.zeros(
            (), device=controls.device, dtype=controls.dtype
        )

        if args.option == "cf":
            candidate = quantities["u_state"]
            admissible = (
                (B.abs() >= b_min)
                & (B <= 0.0)
                & (candidate >= 0.0)
                & (candidate <= cfg.umax)
            ).to(controls.dtype)
            q = gate_for_loss * admissible
            singular_loss = (controls - candidate).square()
        elif args.option == "der":
            candidate = quantities["u_strict"]
            B_condition = B.abs() >= b_min
            if variant == "literal":
                B_condition = B_condition & (B <= 0.0)
            admissible = (
                B_condition
                & (candidate > 0.0)
                & (candidate < cfg.umax)
            ).to(controls.dtype)
            q = gate_for_loss * admissible
            singular_loss = (
                args.w0 * (psi / psi_scale).square()
                + args.w1 * (dot_psi / dot_scale).square()
                + args.w2 * (quantities["ddot_psi"] / ddot_scale).square()
                + args.w_lc * (torch.relu(B) / B_scale).square()
            )
        else:
            raise ValueError(f"unknown singular option: {args.option}")
    else:
        point_gate = psi_gate * dot_gate
        geometry_gate = persistence_gate(
            point_gate, getattr(args, "persistence_window", 3)
        )
        gate_for_loss = (
            geometry_gate.detach()
            if getattr(args, "detach_gate", True)
            else geometry_gate
        )

        lower_ratio = controls / cfg.umax
        nonsingular_loss = (
            torch.relu(boundary_psi) * lower_ratio
            + torch.relu(-boundary_psi) * (1.0 - lower_ratio)
        ).square()
        invalid_component = torch.zeros(
            (), device=controls.device, dtype=controls.dtype
        )
        if args.option == "cf":
            candidate = quantities["u_state"]
            admissible = (
                (candidate >= 0.0) & (candidate <= cfg.umax)
            ).to(controls.dtype)
            q = gate_for_loss * admissible
            singular_loss = (
                (controls - candidate).square()
                + args.w_lc * torch.relu(B).square()
            )
            invalid_component = (
                reduce_mean(
                    gate_for_loss
                    * (1.0 - admissible)
                    * box_violation(candidate, 0.0, cfg.umax).square()
                )
            )
        elif args.option == "der":
            q = gate_for_loss
            singular_loss = (
                args.w0 * (psi / psi_scale).square()
                + args.w1 * (dot_psi / dot_scale).square()
                + args.w2 * (quantities["ddot_psi"] / ddot_scale).square()
                + args.w_lc * (torch.relu(B) / B_scale).square()
            )
        else:
            raise ValueError(f"unknown singular option: {args.option}")

    fixed_candidate_mask = getattr(args, "_fixed_candidate_mask_tensor", None)
    if fixed_candidate_mask is not None:
        fixed_candidate_mask = fixed_candidate_mask.to(
            device=q.device, dtype=q.dtype
        )
        try:
            q = fixed_candidate_mask.expand_as(q)
        except RuntimeError as error:
            raise ValueError(
                "fixed candidate mask shape does not match the current loss grid: "
                f"{tuple(fixed_candidate_mask.shape)} versus {tuple(q.shape)}"
            ) from error

    singular_component = float(getattr(args, "singular_loss_weight", 1.0)) * (
        reduce_mean(q * singular_loss)
    )
    nonsingular_component = float(
        getattr(args, "nonsingular_loss_weight", 1.0)
    ) * reduce_mean((1.0 - q) * nonsingular_loss)
    return {
        "opt_gap": singular_component + nonsingular_component + invalid_component,
        "singular_component": singular_component,
        "nonsingular_component": nonsingular_component,
        "invalid_component": invalid_component,
        "point_gate": point_gate,
        "q": q,
    }


def objective_per_sample(
    states: torch.Tensor,
    controls: torch.Tensor,
    cfg: ProblemConfig,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    dt = cfg.T / cfg.n
    running = (states[:, :-1] * params["beta"]).sum(dim=-1) + params["gamma"] * controls
    terminal = (states[:, -1] * params["alpha"]).sum(dim=-1)
    return terminal + dt * running.sum(dim=-1)


def rk4_objective_per_sample(
    states: torch.Tensor,
    controls: torch.Tensor,
    stage_states: torch.Tensor,
    cfg: ProblemConfig,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Objective using the same RK4 stage quadrature as the state map."""

    weights = torch.as_tensor(
        RK4_B, dtype=states.dtype, device=states.device
    ).view(1, 1, 4)
    stage_controls = controls.unsqueeze(-1).expand_as(weights + stage_states[..., 0])
    running = (
        stage_states * params["beta"]
    ).sum(dim=-1) + params["gamma"] * stage_controls
    terminal = (states[:, -1] * params["alpha"]).sum(dim=-1)
    return terminal + (cfg.T / cfg.n) * (running * weights).sum(dim=(1, 2))


def smoothness_components(
    controls: torch.Tensor, args: argparse.Namespace
) -> Dict[str, torch.Tensor]:
    differences = controls[:, 1:] - controls[:, :-1]
    first = differences.square().mean()
    if controls.shape[1] >= 3:
        second_differences = (
            controls[:, 2:] - 2.0 * controls[:, 1:-1] + controls[:, :-2]
        )
        second = second_differences.square().mean()
    else:
        second = torch.zeros((), device=controls.device, dtype=controls.dtype)
    tau = float(getattr(args, "smooth_max_tau", 0.02))
    if tau <= 0.0:
        raise ValueError("smooth_max_tau must be positive")
    soft_max_jump = tau * (
        torch.logsumexp(differences.abs() / tau, dim=-1)
        - math.log(max(1, differences.shape[-1]))
    )
    max_like = soft_max_jump.square().mean()
    total = (
        float(getattr(args, "smooth_weight", 0.0)) * first
        + float(getattr(args, "smooth_second_weight", 0.0)) * second
        + float(getattr(args, "smooth_max_weight", 0.0)) * max_like
    )
    return {
        "smooth": first,
        "smooth_second": second,
        "smooth_max_like": max_like,
        "smooth_total": total,
    }


def section5_loss(
    model: NestedFeedbackTransformer,
    initial_state: torch.Tensor,
    cfg: ProblemConfig,
    params: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    *,
    state_blind: bool = False,
    state_mode: str | None = None,
) -> Dict[str, torch.Tensor]:
    if state_mode is None:
        state_mode = getattr(args, "state_mode", None)
    if state_mode is None:
        state_mode = (
            "fixed_query"
            if state_blind or bool(getattr(args, "state_blind", False))
            else "feedback"
        )
    integrator = getattr(args, "training_integrator", "euler")
    if integrator == "euler":
        states, controls, raw_minima = simulate_feedback(
            model,
            initial_state,
            cfg,
            params,
            state_blind=state_blind,
            state_mode=state_mode,
            integrator="euler",
        )
        costates = compute_costate(states, controls, cfg, params)
        quantities = singular_quantities(
            states,
            controls,
            costates,
            cfg,
            params,
            safe_b=float(getattr(args, "b_min", 1e-8)),
            costate_pairing="next",
        )
        controls_for_loss = controls
        loss_terms = compose_section5_optimality_loss(
            controls_for_loss, quantities, cfg, args
        )
        point_gate = loss_terms["point_gate"]
        q = loss_terms["q"]
    elif integrator == "rk4":
        states, controls, stage_states, raw_minima = (
            simulate_feedback_rk4_stagewise(
                model,
                initial_state,
                cfg,
                params,
                state_mode=state_mode,
            )
        )
        adjoint = discrete_rk4_adjoint(
            states, controls, stage_states, cfg, params
        )
        costates = adjoint.node_costates
        stage_controls = controls.unsqueeze(-1).expand(-1, -1, 4)
        stage_quantities = singular_quantities_at_points(
            stage_states,
            stage_controls,
            adjoint.stage_costates,
            params,
            safe_b=float(getattr(args, "b_min", 1e-8)),
        )
        weights = torch.as_tensor(
            RK4_B, dtype=controls.dtype, device=controls.device
        ).view(1, 1, 4)
        loss_quantities = {
            **stage_quantities,
            "boundary_psi": adjoint.interval_switching.unsqueeze(-1).expand(
                -1, -1, 4
            ),
            "quadrature_weight": weights,
        }
        loss_terms = compose_section5_optimality_loss(
            stage_controls, loss_quantities, cfg, args
        )
        quantities = {
            name: (value * weights).sum(dim=-1)
            for name, value in stage_quantities.items()
            if name != "B_valid"
        }
        quantities["psi"] = adjoint.interval_switching
        quantities["B_valid"] = quantities["B"].abs() >= float(
            getattr(args, "b_min", 1e-8)
        )
        quantities["boundary_psi"] = adjoint.interval_switching
        quantities["discrete_control_gradient"] = adjoint.control_gradient
        for name, value in stage_quantities.items():
            quantities[f"stage_{name}"] = value
        quantities["stage_q"] = loss_terms["q"]
        quantities["stage_point_gate"] = loss_terms["point_gate"]
        point_gate = (loss_terms["point_gate"] * weights).sum(dim=-1)
        q = (loss_terms["q"] * weights).sum(dim=-1)
    else:
        raise ValueError(f"unknown training integrator: {integrator}")
    optimality_gap = loss_terms["opt_gap"]
    smoothing = smoothness_components(controls, args)
    if integrator == "rk4":
        objective = rk4_objective_per_sample(
            states, controls, stage_states, cfg, params
        )
    else:
        objective = objective_per_sample(states, controls, cfg, params)
    full_gradient_weight = float(
        getattr(
            args,
            "_current_full_gradient_weight",
            getattr(args, "full_gradient_weight", 0.0),
        )
    )
    if full_gradient_weight < 0.0:
        raise ValueError("full_gradient_weight must be nonnegative")
    full_gradient_max_weight = float(
        getattr(
            args,
            "_current_full_gradient_max_weight",
            getattr(args, "full_gradient_max_weight", 0.0),
        )
    )
    if full_gradient_max_weight < 0.0:
        raise ValueError("full_gradient_max_weight must be nonnegative")
    if integrator == "rk4":
        full_gradient = quantities["discrete_control_gradient"]
        full_gradient_residual_mode = str(
            getattr(args, "full_gradient_residual", "raw")
        )
        projection_upper_branch = torch.zeros_like(full_gradient)
        projection_lower_branch = torch.zeros_like(full_gradient)
        if full_gradient_residual_mode == "raw":
            full_gradient_residual = full_gradient
        elif full_gradient_residual_mode == "projected":
            projection_step = float(
                getattr(args, "full_gradient_projection_step", 1.0)
            )
            if projection_step <= 0.0:
                raise ValueError(
                    "full_gradient_projection_step must be positive"
                )
            projection_trial = controls - projection_step * full_gradient
            projection_upper_branch = (
                projection_trial > cfg.umax
            ).to(full_gradient.dtype)
            projection_lower_branch = (
                projection_trial < 0.0
            ).to(full_gradient.dtype)
            full_gradient_residual = (
                controls - torch.clamp(projection_trial, 0.0, cfg.umax)
            ) / projection_step
        else:
            raise ValueError(
                "full_gradient_residual must be 'raw' or 'projected'"
            )
        configured_scale = float(
            getattr(args, "full_gradient_scale", 0.0)
        )
        full_gradient_scale = (
            configured_scale
            if configured_scale > 0.0
            else (cfg.T / cfg.n)
        )
        if full_gradient_scale <= 0.0:
            raise ValueError("full_gradient_scale must be positive")
        full_gradient_scaled = (
            full_gradient_residual / full_gradient_scale
        )
        full_gradient_loss = full_gradient_scaled.square().mean()
        full_gradient_max_tau = float(
            getattr(args, "full_gradient_max_tau", 0.1)
        )
        if full_gradient_max_tau <= 0.0:
            raise ValueError("full_gradient_max_tau must be positive")
        full_gradient_soft_max = full_gradient_max_tau * (
            torch.logsumexp(
                full_gradient_scaled.abs() / full_gradient_max_tau,
                dim=-1,
            )
            - math.log(full_gradient_scaled.shape[-1])
        )
        full_gradient_max_loss = full_gradient_soft_max.square().mean()
    else:
        if full_gradient_weight > 0.0:
            raise ValueError(
                "full reduced-gradient training requires --training_integrator=rk4"
            )
        full_gradient = torch.zeros_like(controls)
        full_gradient_residual = torch.zeros_like(controls)
        projection_upper_branch = torch.zeros_like(controls)
        projection_lower_branch = torch.zeros_like(controls)
        full_gradient_scale = 1.0
        full_gradient_loss = torch.zeros(
            (), device=controls.device, dtype=controls.dtype
        )
        full_gradient_soft_max = torch.zeros(
            controls.shape[0], device=controls.device, dtype=controls.dtype
        )
        full_gradient_max_loss = torch.zeros(
            (), device=controls.device, dtype=controls.dtype
        )
    total_loss = (
        optimality_gap
        + smoothing["smooth_total"]
        + full_gradient_weight * full_gradient_loss
        + full_gradient_max_weight * full_gradient_max_loss
    )
    return {
        "loss": total_loss,
        "opt_gap": optimality_gap,
        "singular_component": loss_terms["singular_component"],
        "nonsingular_component": loss_terms["nonsingular_component"],
        "invalid_component": loss_terms["invalid_component"],
        **smoothing,
        "full_gradient_loss": full_gradient_loss,
        "full_gradient_weighted": full_gradient_weight * full_gradient_loss,
        "full_gradient_max_loss": full_gradient_max_loss,
        "full_gradient_max_weighted": (
            full_gradient_max_weight * full_gradient_max_loss
        ),
        "full_gradient_soft_max": full_gradient_soft_max,
        "full_gradient_scale": torch.as_tensor(
            full_gradient_scale, device=controls.device, dtype=controls.dtype
        ),
        "full_gradient": full_gradient,
        "full_gradient_residual": full_gradient_residual,
        "projection_upper_branch": projection_upper_branch,
        "projection_lower_branch": projection_lower_branch,
        "objective_per_sample": objective,
        "objective": objective.mean(),
        "states": states,
        "controls": controls,
        "costates": costates,
        "raw_minima": raw_minima,
        "point_gate": point_gate,
        "q": q,
        **quantities,
    }


def scalar_metrics(pack: Dict[str, torch.Tensor], cfg: ProblemConfig, args: argparse.Namespace) -> dict[str, float]:
    psi = pack["psi"]
    dot_psi = pack["dot_psi"]
    near = (psi.abs() <= args.singular_eps) & (dot_psi.abs() <= args.dot_eps)
    near_count = near.sum().clamp_min(1)
    controls = pack["controls"]
    if "stage_q" in pack:
        weights = torch.as_tensor(
            RK4_B, device=pack["stage_q"].device, dtype=pack["stage_q"].dtype
        ).view(1, 1, 4)
        weighted_q = pack["stage_q"] * weights
        q_denominator = weighted_q.sum().clamp_min(1e-12)

        def q_rms(name: str) -> float:
            value = pack[f"stage_{name}"]
            result = (weighted_q * value.square()).sum().div(q_denominator).sqrt()
            return float(result.detach().cpu())
    else:
        weighted_q = pack["q"]
        q_denominator = weighted_q.sum().clamp_min(1e-12)

        def q_rms(name: str) -> float:
            result = (
                weighted_q * pack[name].square()
            ).sum().div(q_denominator).sqrt()
            return float(result.detach().cpu())
    return {
        "loss": float(pack["loss"].detach().cpu()),
        "opt_gap": float(pack["opt_gap"].detach().cpu()),
        "singular_component": float(pack["singular_component"].detach().cpu()),
        "nonsingular_component": float(pack["nonsingular_component"].detach().cpu()),
        "invalid_component": float(pack["invalid_component"].detach().cpu()),
        "smooth": float(pack["smooth"].detach().cpu()),
        "smooth_second": float(pack["smooth_second"].detach().cpu()),
        "smooth_max_like": float(pack["smooth_max_like"].detach().cpu()),
        "smooth_total": float(pack["smooth_total"].detach().cpu()),
        "full_gradient_loss": float(
            pack["full_gradient_loss"].detach().cpu()
        ),
        "full_gradient_weighted": float(
            pack["full_gradient_weighted"].detach().cpu()
        ),
        "full_gradient_max_loss": float(
            pack["full_gradient_max_loss"].detach().cpu()
        ),
        "full_gradient_max_weighted": float(
            pack["full_gradient_max_weighted"].detach().cpu()
        ),
        "full_gradient_soft_max_mean": float(
            pack["full_gradient_soft_max"].mean().detach().cpu()
        ),
        "full_gradient_scaled_rms": float(
            pack["full_gradient_loss"].sqrt().detach().cpu()
        ),
        "full_gradient_rms": float(
            pack["full_gradient"].square().mean().sqrt().detach().cpu()
        ),
        "full_gradient_linf": float(
            pack["full_gradient"].abs().max().detach().cpu()
        ),
        "full_gradient_residual_rms": float(
            pack["full_gradient_residual"]
            .square()
            .mean()
            .sqrt()
            .detach()
            .cpu()
        ),
        "full_gradient_residual_linf": float(
            pack["full_gradient_residual"].abs().max().detach().cpu()
        ),
        "projection_upper_count_mean": float(
            pack["projection_upper_branch"].sum(dim=-1).mean().detach().cpu()
        ),
        "projection_lower_count_mean": float(
            pack["projection_lower_branch"].sum(dim=-1).mean().detach().cpu()
        ),
        "objective": float(pack["objective"].detach().cpu()),
        "u_min": float(controls.min().detach().cpu()),
        "u_max": float(controls.max().detach().cpu()),
        "u_mean": float(controls.mean().detach().cpu()),
        "u_box_margin": float(
            torch.minimum(controls, cfg.umax - controls).min().detach().cpu()
        ),
        "max_jump": float((controls[:, 1:] - controls[:, :-1]).abs().max().detach().cpu()),
        "q_mean": float(pack["q"].mean().detach().cpu()),
        "psi_rms": float(pack["psi"].square().mean().sqrt().detach().cpu()),
        "dotpsi_rms": float(
            pack["dot_psi"].square().mean().sqrt().detach().cpu()
        ),
        "ddotpsi_rms": float(
            pack["ddot_psi"].square().mean().sqrt().detach().cpu()
        ),
        "B_rms": float(pack["B"].square().mean().sqrt().detach().cpu()),
        "q_psi_rms": q_rms("psi"),
        "q_dotpsi_rms": q_rms("dot_psi"),
        "q_ddotpsi_rms": q_rms("ddot_psi"),
        "q_B_rms": q_rms("B"),
        "near_fraction": float(near.to(controls.dtype).mean().detach().cpu()),
        "lc_violation_near": float(
            ((pack["B"] > 0.0) & near).sum().to(controls.dtype).div(near_count).detach().cpu()
        ),
        "ddot_abs_near": float(
            (pack["ddot_psi"].abs() * near).sum().div(near_count).detach().cpu()
        ),
        "raw_next_min": float(pack["raw_minima"].min().detach().cpu()),
        "final_mean_N": float(pack["states"][:, -1].mean().detach().cpu()),
    }


def make_fixed_directions(count: int, m: int, seed: int, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return 2.0 * torch.rand(count, m, generator=generator, dtype=dtype) - 1.0


def test_states_from_directions(
    directions: torch.Tensor,
    radius: float,
    cfg: ProblemConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return cfg.n0 * (1.0 + float(radius) * directions.to(device=device, dtype=dtype))


def load_operational_time_control(
    checkpoint_path: Path,
    cfg: ProblemConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    wrapper = dict(checkpoint.get("wrapper", {}))
    wrapper_class = str(wrapper.get("class", ""))
    checkpoint_args = dict(
        checkpoint.get("base_model_args", checkpoint.get("args", {}))
    )
    architecture_keys = ("d_model", "heads", "layers", "init_u")
    source_path = checkpoint.get("source_checkpoint")
    visited: set[Path] = {checkpoint_path.resolve()}
    while any(key not in checkpoint_args for key in architecture_keys) and source_path:
        candidate = Path(source_path).expanduser()
        if not candidate.is_absolute():
            local_candidate = checkpoint_path.parent / candidate
            candidate = local_candidate if local_candidate.exists() else ROOT / candidate
        candidate = candidate.resolve()
        if candidate in visited:
            raise RuntimeError(f"cyclic source_checkpoint chain at {candidate}")
        visited.add(candidate)
        source_checkpoint = torch.load(
            candidate, map_location="cpu", weights_only=False
        )
        for key, value in source_checkpoint.get("args", {}).items():
            checkpoint_args.setdefault(key, value)
        source_path = source_checkpoint.get("source_checkpoint")
    missing = [key for key in architecture_keys if key not in checkpoint_args]
    if missing:
        raise KeyError(
            f"{checkpoint_path}: missing architecture settings {missing}"
        )
    model = TimeTransformer(
        checkpoint_args["d_model"],
        checkpoint_args["heads"],
        checkpoint_args["layers"],
        cfg.umax,
        checkpoint_args["init_u"],
    ).to(device=device, dtype=dtype)
    source_state = checkpoint["model_state"]
    if wrapper_class in {
        "FixedBoxProjection",
        "BoundaryProjectedControl",
        "LinearRawBoxProjection",
    }:
        source_state = {
            key.removeprefix("base."): value
            for key, value in source_state.items()
            if key.startswith("base.")
        }
    elif any(key.startswith("time_branch.") for key in source_state):
        source_state = {
            key.removeprefix("time_branch."): value
            for key, value in source_state.items()
            if key.startswith("time_branch.")
        }
    model.load_state_dict(source_state)
    grid = torch.linspace(0.0, 1.0, cfg.n + 1, device=device, dtype=dtype)
    model.eval()
    control = model(grid)
    if wrapper_class == "FixedBoxProjection":
        probability = torch.clamp(
            control / cfg.umax, 1.0e-8, 1.0 - 1.0e-8
        )
        temperature = float(wrapper.get("temperature", 1.0))
        scale = float(wrapper["scale"])
        control = torch.clamp(
            scale * cfg.umax * torch.sigmoid(torch.logit(probability) / temperature),
            0.0,
            cfg.umax,
        )
    elif wrapper_class == "BoundaryProjectedControl":
        scale = float(wrapper.get("scale", wrapper.get("initial_scale", 1.0)))
        control = torch.clamp(scale * control, 0.0, cfg.umax)
    elif wrapper_class == "LinearRawBoxProjection":
        hidden = model.input(time_features(grid)).unsqueeze(0)
        hidden = model.encoder(hidden).squeeze(0)
        raw_control = model.output(hidden).squeeze(-1)
        control = torch.clamp(raw_control, 0.0, cfg.umax)
    return control[: cfg.n]


def write_history(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def evaluate_test_suite(
    model: NestedFeedbackTransformer,
    cfg: ProblemConfig,
    params: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    checkpoint_path: Path,
    out_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    directions = make_fixed_directions(args.test_size, cfg.m, args.test_seed, torch.float64)
    baseline_control = load_operational_time_control(
        checkpoint_path, cfg, device, dtype
    )
    rows: list[dict] = []
    stress_rows: list[dict] = []
    summaries: dict[str, dict[str, float]] = {}
    initial_state_arrays: dict[str, np.ndarray] = {}
    model.eval()

    def baseline_objective_for(initial: torch.Tensor) -> torch.Tensor:
        expanded_control = baseline_control.expand(initial.shape[0], -1)
        if args.training_integrator == "rk4":
            baseline_states, baseline_controls, baseline_stages = (
                simulate_open_loop_rk4_reference(
                    expanded_control, initial, cfg, params
                )
            )
            return rk4_objective_per_sample(
                baseline_states,
                baseline_controls,
                baseline_stages,
                cfg,
                params,
            )
        baseline_states = simulate_open_loop(
            baseline_control, initial, cfg, params, integrator="euler"
        )
        return objective_per_sample(
            baseline_states, expanded_control, cfg, params
        )

    with torch.no_grad():
        for radius in args.test_radii:
            initial = test_states_from_directions(directions, radius, cfg, device, dtype)
            pack = section5_loss(model, initial, cfg, params, args)
            baseline_objective = baseline_objective_for(initial)
            feedback_objective = pack["objective_per_sample"]
            delta = baseline_objective - feedback_objective
            key = f"radius_{radius:.2f}"
            initial_state_arrays[key] = initial.detach().cpu().numpy()
            summaries[key] = {
                "count": int(initial.shape[0]),
                "feedback_J_mean": float(feedback_objective.mean().cpu()),
                "time_only_J_mean": float(baseline_objective.mean().cpu()),
                "time_only_minus_feedback_mean": float(delta.mean().cpu()),
                "time_only_minus_feedback_median": float(delta.median().cpu()),
                "feedback_win_fraction": float((delta > 0.0).to(dtype).mean().cpu()),
                "feedback_cross_sample_u_std": float(pack["controls"].std(dim=0, unbiased=False).mean().cpu()),
            }
            for index in range(initial.shape[0]):
                rows.append(
                    {
                        "radius": radius,
                        "sample": index,
                        "kind": "random",
                        "feedback_J": float(feedback_objective[index].cpu()),
                        "time_only_J": float(baseline_objective[index].cpu()),
                        "time_only_minus_feedback": float(delta[index].cpu()),
                    }
                )

            trait = torch.linspace(-1.0, 1.0, cfg.m, dtype=torch.float64)
            stress_directions = torch.stack(
                [
                    torch.ones(cfg.m, dtype=torch.float64),
                    -torch.ones(cfg.m, dtype=torch.float64),
                    trait,
                    -trait,
                ]
            )
            stress_names = [
                "all_plus",
                "all_minus",
                "resistant_heavy",
                "sensitive_heavy",
            ]
            stress_initial = test_states_from_directions(
                stress_directions, radius, cfg, device, dtype
            )
            stress_pack = section5_loss(model, stress_initial, cfg, params, args)
            stress_baseline_objective = baseline_objective_for(stress_initial)
            stress_feedback_objective = stress_pack["objective_per_sample"]
            for index, name in enumerate(stress_names):
                stress_rows.append(
                    {
                        "radius": radius,
                        "kind": name,
                        "feedback_J": float(stress_feedback_objective[index].cpu()),
                        "time_only_J": float(stress_baseline_objective[index].cpu()),
                        "time_only_minus_feedback": float(
                            (stress_baseline_objective[index] - stress_feedback_objective[index]).cpu()
                        ),
                    }
                )

    with (out_dir / "test_per_sample.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "test_summary.json").open("w") as handle:
        json.dump(summaries, handle, indent=2, sort_keys=True)
    with (out_dir / "test_stress.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(stress_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(stress_rows)
    np.savez(out_dir / "test_initial_states.npz", directions=directions.numpy(), **initial_state_arrays)


def parse_radii(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def train(args: argparse.Namespace) -> None:
    args.full_gradient_weight = float(
        getattr(args, "full_gradient_weight", 0.0)
    )
    args.full_gradient_ramp_epochs = int(
        getattr(args, "full_gradient_ramp_epochs", 0)
    )
    args.full_gradient_max_weight = float(
        getattr(args, "full_gradient_max_weight", 0.0)
    )
    args.full_gradient_max_tau = float(
        getattr(args, "full_gradient_max_tau", 0.1)
    )
    if args.full_gradient_weight < 0.0:
        raise ValueError("full_gradient_weight must be nonnegative")
    if args.full_gradient_ramp_epochs < 0:
        raise ValueError("full_gradient_ramp_epochs must be nonnegative")
    if args.full_gradient_max_weight < 0.0:
        raise ValueError("full_gradient_max_weight must be nonnegative")
    if args.full_gradient_max_tau <= 0.0:
        raise ValueError("full_gradient_max_tau must be positive")
    set_seed(args.seed)
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = torch.float64 if args.float64 else torch.float32
    cfg = ProblemConfig(
        T=args.T,
        n=args.n,
        m=args.m,
        umax=args.umax,
        beta=args.beta,
        alpha=args.alpha,
        gamma=args.gamma,
        n0=args.n0,
        m_suppression=args.m_suppression,
    )
    params = build_params(cfg, device, dtype)
    model = NestedFeedbackTransformer(
        cfg.m,
        cfg.umax,
        args.state_scale,
        parse_hidden(args.state_hidden),
        args.d_model,
        args.heads,
        args.layers,
        args.init_u,
        args.correction_gain,
        args.state_feature_mode,
        args.center_state_correction,
        args.action_temperature,
        args.action_scale,
        args.action_parameterization,
    ).to(device=device, dtype=dtype)
    time_checkpoint = Path(args.time_checkpoint)
    checkpoint = model.load_time_checkpoint(time_checkpoint)
    args.action_temperature = float(model.action_temperature)
    args.action_scale = float(model.action_scale)
    args.action_parameterization = str(model.action_parameterization)
    checkpoint_problem = checkpoint.get("problem", {})
    if int(checkpoint_problem.get("n", cfg.n)) != cfg.n:
        raise ValueError("time-only checkpoint grid mismatch")
    for key in ("T", "umax", "beta", "alpha", "gamma", "n0", "m_suppression"):
        if not math.isclose(
            float(checkpoint_problem.get(key, getattr(cfg, key))),
            float(getattr(cfg, key)),
        ):
            raise ValueError(f"time-only checkpoint {key} mismatch")
    model.to(device=device, dtype=dtype)
    model.set_feature_vectors(params["r"], params["phi"])

    resume_checkpoint_path: Path | None = None
    state_checkpoint_path: Path | None = None
    if args.resume_checkpoint and args.state_checkpoint:
        raise ValueError("--resume_checkpoint and --state_checkpoint are mutually exclusive")
    if args.resume_checkpoint:
        resume_checkpoint_path = Path(args.resume_checkpoint)
        resume_checkpoint = torch.load(
            resume_checkpoint_path, map_location="cpu", weights_only=False
        )
        resume_problem = resume_checkpoint.get("problem", {})
        for key in ("n", "m"):
            if int(resume_problem.get(key, getattr(cfg, key))) != int(
                getattr(cfg, key)
            ):
                raise ValueError(f"resume checkpoint {key} mismatch")
        for key in ("T", "umax", "beta", "alpha", "gamma", "n0"):
            if not math.isclose(
                float(resume_problem.get(key, getattr(cfg, key))),
                float(getattr(cfg, key)),
            ):
                raise ValueError(f"resume checkpoint {key} mismatch")
        resume_args = resume_checkpoint.get("args", {})
        compatibility_keys = (
            "option",
            "loss_variant",
            "state_feature_mode",
            "center_state_correction",
            "state_mode",
            "training_integrator",
        )
        for key in compatibility_keys:
            if key in resume_args and resume_args[key] != getattr(args, key):
                raise ValueError(
                    f"resume checkpoint {key} mismatch: "
                    f"{resume_args[key]!r} != {getattr(args, key)!r}"
                )
        model.load_state_dict(resume_checkpoint["model_state"])
        model.to(device=device, dtype=dtype)
        model.set_feature_vectors(params["r"], params["phi"])
        print(f"Loaded feedback checkpoint from {resume_checkpoint_path}")
    elif args.state_checkpoint:
        state_checkpoint_path = Path(args.state_checkpoint)
        state_checkpoint = torch.load(
            state_checkpoint_path, map_location="cpu", weights_only=False
        )
        state_problem = state_checkpoint.get("problem", {})
        if int(state_problem.get("m", cfg.m)) != cfg.m:
            raise ValueError("state checkpoint phenotype count mismatch")
        for key in ("T", "umax", "beta", "alpha", "gamma", "n0"):
            if not math.isclose(
                float(state_problem.get(key, getattr(cfg, key))),
                float(getattr(cfg, key)),
            ):
                raise ValueError(f"state checkpoint {key} mismatch")
        state_args = state_checkpoint.get("args", {})
        compatibility_keys = (
            "option",
            "state_feature_mode",
            "center_state_correction",
        )
        for key in compatibility_keys:
            if key in state_args and state_args[key] != getattr(args, key):
                raise ValueError(
                    f"state checkpoint {key} mismatch: "
                    f"{state_args[key]!r} != {getattr(args, key)!r}"
                )
        state_dict = {
            key.removeprefix("state_branch."): value
            for key, value in state_checkpoint["model_state"].items()
            if key.startswith("state_branch.")
        }
        if not state_dict:
            raise ValueError("state checkpoint contains no state-branch parameters")
        model.state_branch.load_state_dict(state_dict)
        model.to(device=device, dtype=dtype)
        model.set_feature_vectors(params["r"], params["phi"])
        print(f"Loaded state branch from {state_checkpoint_path}")

    if args.state_blind:
        if args.state_mode not in {"feedback", "fixed_query"}:
            raise ValueError("--state_blind conflicts with the selected state_mode")
        args.state_mode = "fixed_query"
    if args.state_mode == "w_zero" and (
        args.freeze_time_branch or args.time_unfreeze_epoch > 0
    ):
        raise ValueError(
            "w_zero disables the state branch, so its time branch must remain trainable"
        )
    nominal_initial = torch.full(
        (1, cfg.m), cfg.n0, device=device, dtype=dtype
    )

    def refresh_nominal_reference() -> None:
        with torch.no_grad():
            nominal_states, _, _ = simulate_feedback(
                model,
                nominal_initial,
                cfg,
                params,
                state_mode="w_zero",
                integrator=args.training_integrator,
            )
        model.set_nominal_reference(nominal_states[0])

    refresh_nominal_reference()

    if args.auto_residual_scales:
        with torch.enable_grad():
            scale_pack = section5_loss(
                model,
                nominal_initial,
                cfg,
                params,
                args,
                state_mode="w_zero",
            )

        def nominal_rms(name: str) -> float:
            value = scale_pack[name].detach()
            return max(float(value.square().mean().sqrt().cpu()), 1.0e-8)

        args.psi_scale = nominal_rms("psi")
        args.dot_scale = nominal_rms("dot_psi")
        args.ddot_scale = nominal_rms("ddot_psi")
        args.B_scale = nominal_rms("B")
        print(
            "Auto-calibrated residual scales on the frozen nominal trajectory: "
            f"psi={args.psi_scale:.6g}, dot={args.dot_scale:.6g}, "
            f"ddot={args.ddot_scale:.6g}, B={args.B_scale:.6g}",
            flush=True,
        )

    if args.selection_start_epoch < 0:
        args.selection_start_epoch = (
            args.time_unfreeze_epoch
            if args.time_unfreeze_epoch > 0 and not args.freeze_time_branch
            else 0
        )
    if not 0 <= args.selection_start_epoch <= args.epochs:
        raise ValueError("selection_start_epoch must lie between 0 and epochs")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.test_radii = parse_radii(args.test_radii)
    with (out_dir / "config.json").open("w") as handle:
        json.dump(
            {"args": vars(args), "problem": cfg.__dict__},
            handle,
            indent=2,
            sort_keys=True,
        )

    if args.candidate_mask_mode != "dynamic":
        # Freeze the candidate set produced by the incoming checkpoint on the
        # nominal trajectory.  Broadcasting this time/stage mask across the
        # perturbed-state batch makes this a controlled diagnostic of whether
        # the dynamic gate can reduce its own training weight.
        model.eval()
        with torch.no_grad():
            source_pack = section5_loss(
                model,
                nominal_initial,
                cfg,
                params,
                args,
                state_blind=args.state_blind,
            )
        source_mask = (
            source_pack["stage_q"]
            if args.training_integrator == "rk4"
            else source_pack["q"]
        ).detach()
        if args.candidate_mask_mode == "fixed_binary":
            source_mask = (
                source_mask >= args.fixed_candidate_threshold
            ).to(source_mask.dtype)
        args._fixed_candidate_mask_tensor = source_mask
        np.savez(
            out_dir / "fixed_candidate_mask.npz",
            q=source_mask.cpu().numpy(),
            mode=np.asarray(args.candidate_mask_mode),
            threshold=np.asarray(args.fixed_candidate_threshold),
        )
        print(
            "Froze nominal candidate mask: "
            f"mode={args.candidate_mask_mode} mean={source_mask.mean().item():.6g}",
            flush=True,
        )

    time_parameters = list(model.time_branch.parameters())
    state_parameters = list(model.state_branch.parameters())
    if args.freeze_time_branch or args.time_unfreeze_epoch > 0:
        for parameter in time_parameters:
            parameter.requires_grad_(False)
    if args.state_mode == "w_zero":
        for parameter in state_parameters:
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": time_parameters,
                "lr": (
                    0.0
                    if args.freeze_time_branch or args.time_unfreeze_epoch > 0
                    else args.lr * args.time_lr_scale
                ),
            },
            {"params": state_parameters, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )

    validation_directions = make_fixed_directions(
        args.validation_size, cfg.m, args.validation_seed, torch.float64
    )
    validation_initial = test_states_from_directions(
        validation_directions, args.train_radius, cfg, device, dtype
    )
    generator_device = device if device.type in {"cpu", "cuda"} else torch.device("cpu")
    training_generator = torch.Generator(device=generator_device).manual_seed(
        args.training_sample_seed_base + args.seed
    )
    history: list[dict] = []
    training_history: list[dict] = []
    best_value = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_reference: torch.Tensor | None = None
    best_epoch: int | None = None
    best_combined_full_gradient = float("inf")
    best_full_gradient_value = float("inf")
    best_full_gradient_total_loss = float("inf")
    best_full_gradient_state: dict[str, torch.Tensor] | None = None
    best_full_gradient_reference: torch.Tensor | None = None
    best_full_gradient_epoch: int | None = None
    best_full_gradient_linf_value = float("inf")
    best_full_gradient_linf_total_loss = float("inf")
    best_full_gradient_linf_state: dict[str, torch.Tensor] | None = None
    best_full_gradient_linf_reference: torch.Tensor | None = None
    best_full_gradient_linf_epoch: int | None = None

    def evaluate(epoch: int) -> float:
        nonlocal best_value, best_state, best_reference, best_epoch
        nonlocal best_combined_full_gradient
        nonlocal best_full_gradient_value, best_full_gradient_total_loss
        nonlocal best_full_gradient_state, best_full_gradient_reference
        nonlocal best_full_gradient_epoch
        nonlocal best_full_gradient_linf_value, best_full_gradient_linf_total_loss
        nonlocal best_full_gradient_linf_state, best_full_gradient_linf_reference
        nonlocal best_full_gradient_linf_epoch
        model.eval()
        previous_gradient_weight = getattr(
            args, "_current_full_gradient_weight", None
        )
        previous_gradient_max_weight = getattr(
            args, "_current_full_gradient_max_weight", None
        )
        args._current_full_gradient_weight = args.full_gradient_weight
        args._current_full_gradient_max_weight = (
            args.full_gradient_max_weight
        )
        with torch.no_grad():
            validation_pack = section5_loss(
                model, validation_initial, cfg, params, args,
                state_blind=args.state_blind,
            )
            metrics = scalar_metrics(validation_pack, cfg, args)
        if previous_gradient_weight is None:
            delattr(args, "_current_full_gradient_weight")
        else:
            args._current_full_gradient_weight = previous_gradient_weight
        if previous_gradient_max_weight is None:
            delattr(args, "_current_full_gradient_max_weight")
        else:
            args._current_full_gradient_max_weight = (
                previous_gradient_max_weight
            )
        row = {"epoch": epoch, **metrics}
        history.append(row)
        if epoch >= args.selection_start_epoch and metrics["loss"] < best_value:
            best_value = metrics["loss"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_reference = model.nominal_reference.detach().cpu().clone()
            best_combined_full_gradient = metrics["full_gradient_loss"]
        if (
            epoch >= args.selection_start_epoch
            and metrics["full_gradient_loss"] < best_full_gradient_value
        ):
            best_full_gradient_value = metrics["full_gradient_loss"]
            best_full_gradient_total_loss = metrics["loss"]
            best_full_gradient_epoch = epoch
            best_full_gradient_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_full_gradient_reference = (
                model.nominal_reference.detach().cpu().clone()
            )
        if (
            epoch >= args.selection_start_epoch
            and metrics["full_gradient_residual_linf"]
            < best_full_gradient_linf_value
        ):
            best_full_gradient_linf_value = metrics[
                "full_gradient_residual_linf"
            ]
            best_full_gradient_linf_total_loss = metrics["loss"]
            best_full_gradient_linf_epoch = epoch
            best_full_gradient_linf_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_full_gradient_linf_reference = (
                model.nominal_reference.detach().cpu().clone()
            )
        print(
            f"[{args.option} {epoch:04d}] loss={metrics['loss']:.6g} "
            f"gap={metrics['opt_gap']:.6g} J={metrics['objective']:.6f} "
            f"q={metrics['q_mean']:.3f} near={metrics['near_fraction']:.3f} "
            f"Gbar={metrics['full_gradient_scaled_rms']:.3g} "
            f"Ginf={metrics['full_gradient_residual_linf']:.3g} "
            f"up={metrics['projection_upper_count_mean']:.2f} "
            f"u=({metrics['u_min']:.3f},{metrics['u_max']:.3f}) "
            f"jump={metrics['max_jump']:.3f}",
            flush=True,
        )
        return metrics["loss"]

    evaluate(0)
    for epoch in range(1, args.epochs + 1):
        if (
            not args.freeze_time_branch
            and epoch == args.time_unfreeze_epoch
            and args.time_unfreeze_epoch > 0
        ):
            for parameter in time_parameters:
                parameter.requires_grad_(True)
            optimizer.param_groups[0]["lr"] = args.lr * args.time_lr_scale
            print(
                f"Unfroze time branch at epoch {epoch} with "
                f"lr={optimizer.param_groups[0]['lr']:.6g}",
                flush=True,
            )
        model.train()
        initial = sample_componentwise_initial_states(
            args.batch_size,
            cfg,
            args.train_radius,
            generator_device,
            dtype,
            generator=training_generator,
            include_nominal=True,
        ).to(device=device)
        optimizer.zero_grad(set_to_none=True)
        if args.full_gradient_ramp_epochs > 0:
            full_gradient_ramp = min(
                1.0, epoch / args.full_gradient_ramp_epochs
            )
        else:
            full_gradient_ramp = 1.0
        args._current_full_gradient_weight = (
            args.full_gradient_weight * full_gradient_ramp
        )
        args._current_full_gradient_max_weight = (
            args.full_gradient_max_weight * full_gradient_ramp
        )
        pack = section5_loss(
            model, initial, cfg, params, args, state_blind=args.state_blind
        )
        if float(pack["raw_minima"].min().detach().cpu()) <= 0.0:
            raise RuntimeError(
                "positivity clamp became active; the current adjoint no longer matches"
            )
        training_history.append(
            {
                "epoch": epoch,
                "loss": float(pack["loss"].detach().cpu()),
                "opt_gap": float(pack["opt_gap"].detach().cpu()),
                "full_gradient_loss": float(
                    pack["full_gradient_loss"].detach().cpu()
                ),
                "full_gradient_max_loss": float(
                    pack["full_gradient_max_loss"].detach().cpu()
                ),
                "full_gradient_weight": float(
                    args._current_full_gradient_weight
                ),
                "full_gradient_max_weight": float(
                    args._current_full_gradient_max_weight
                ),
                "learning_rate": float(optimizer.param_groups[1]["lr"]),
            }
        )
        pack["loss"].backward()
        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if args.center_state_correction and not args.freeze_time_branch:
            refresh_nominal_reference()
        if (
            epoch == 1
            or epoch % args.eval_every == 0
            or epoch == args.selection_start_epoch
            or epoch == args.epochs
        ):
            value = evaluate(epoch)
            if (
                args.freeze_time_branch
                or args.time_unfreeze_epoch == 0
                or epoch >= args.time_unfreeze_epoch
            ):
                scheduler.step(value)

    if best_state is None:
        raise RuntimeError("training did not produce a finite checkpoint")
    if best_full_gradient_state is None:
        raise RuntimeError("training did not produce a full-gradient checkpoint")
    if best_full_gradient_linf_state is None:
        raise RuntimeError("training did not produce a full-gradient Linf checkpoint")
    model.load_state_dict(best_state)
    if best_reference is None:
        raise RuntimeError("best checkpoint is missing its nominal reference")
    model.set_nominal_reference(best_reference.to(device=device, dtype=dtype))
    torch.save(
        {
            "model_state": best_state,
            "args": {
                key: value
                for key, value in vars(args).items()
                if not key.startswith("_")
            },
            "problem": cfg.__dict__,
            "best_validation_loss": best_value,
            "best_validation_full_gradient_loss": best_combined_full_gradient,
            "best_epoch": best_epoch,
            "selection_metric": "loss",
            "time_checkpoint": str(time_checkpoint),
            "resume_checkpoint": (
                str(resume_checkpoint_path)
                if resume_checkpoint_path is not None
                else None
            ),
            "state_checkpoint": (
                str(state_checkpoint_path)
                if state_checkpoint_path is not None
                else None
            ),
            "nominal_reference": model.nominal_reference.detach().cpu(),
        },
        out_dir / "best_feedback_section5.pt",
    )
    if best_full_gradient_reference is None:
        raise RuntimeError("full-gradient checkpoint is missing its nominal reference")
    torch.save(
        {
            "model_state": best_full_gradient_state,
            "args": {
                key: value
                for key, value in vars(args).items()
                if not key.startswith("_")
            },
            "problem": cfg.__dict__,
            "best_validation_loss": best_full_gradient_total_loss,
            "best_validation_full_gradient_loss": best_full_gradient_value,
            "best_epoch": best_full_gradient_epoch,
            "selection_metric": "full_gradient_loss",
            "time_checkpoint": str(time_checkpoint),
            "resume_checkpoint": (
                str(resume_checkpoint_path)
                if resume_checkpoint_path is not None
                else None
            ),
            "state_checkpoint": (
                str(state_checkpoint_path)
                if state_checkpoint_path is not None
                else None
            ),
            "nominal_reference": best_full_gradient_reference,
        },
        out_dir / "best_feedback_section5_full_gradient.pt",
    )
    if best_full_gradient_linf_reference is None:
        raise RuntimeError("full-gradient Linf checkpoint is missing its nominal reference")
    torch.save(
        {
            "model_state": best_full_gradient_linf_state,
            "args": {
                key: value
                for key, value in vars(args).items()
                if not key.startswith("_")
            },
            "problem": cfg.__dict__,
            "best_validation_loss": best_full_gradient_linf_total_loss,
            "best_validation_full_gradient_linf": best_full_gradient_linf_value,
            "best_epoch": best_full_gradient_linf_epoch,
            "selection_metric": "full_gradient_residual_linf",
            "time_checkpoint": str(time_checkpoint),
            "resume_checkpoint": (
                str(resume_checkpoint_path)
                if resume_checkpoint_path is not None
                else None
            ),
            "state_checkpoint": (
                str(state_checkpoint_path)
                if state_checkpoint_path is not None
                else None
            ),
            "nominal_reference": best_full_gradient_linf_reference,
        },
        out_dir / "best_feedback_section5_full_gradient_linf.pt",
    )
    write_history(out_dir / "history.csv", history)
    write_history(out_dir / "training_history.csv", training_history)
    evaluate_test_suite(
        model,
        cfg,
        params,
        args,
        time_checkpoint,
        out_dir,
        device,
        dtype,
    )
    print(f"Saved Section-5 pilot to {out_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--option", choices=["cf", "der"], required=True)
    parser.add_argument(
        "--loss_variant",
        choices=["literal", "lc_live", "pilot"],
        default="pilot",
        help="teacher-draft equations, its one-factor live-LC correction, or the first pilot",
    )
    parser.add_argument("--time_checkpoint", default="paper_runs/smoothness_weight_sweep/w3/seed_4/best_pmp_kkt.pt")
    parser.add_argument(
        "--resume_checkpoint",
        default="",
        help=(
            "optional full feedback checkpoint used to continue training; "
            "the time checkpoint still records the underlying time-only branch"
        ),
    )
    parser.add_argument(
        "--state_checkpoint",
        default="",
        help=(
            "optional feedback checkpoint from which only the state-correction "
            "branch is loaded; this permits a new time-only base and a new grid"
        ),
    )
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--m", type=int, default=21)
    parser.add_argument("--umax", type=float, default=3.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=20.0)
    parser.add_argument("--n0", type=float, default=10.0)
    parser.add_argument("--m_suppression", type=float, default=0.5)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--init_u", type=float, default=1.5)
    parser.add_argument("--state_hidden", default="128,128")
    parser.add_argument("--state_scale", type=float, default=15.0)
    parser.add_argument("--correction_gain", type=float, default=1.0)
    parser.add_argument(
        "--action_temperature",
        type=float,
        default=1.0,
        help=(
            "positive temperature of the bounded sigmoid action map; values "
            "below one sharpen its approach to the control bounds"
        ),
    )
    parser.add_argument(
        "--action_scale",
        type=float,
        default=1.0,
        help=(
            "positive global scale applied before the final hard box projection; "
            "automatically inherited from supported wrapped time checkpoints"
        ),
    )
    parser.add_argument(
        "--action_parameterization",
        choices=("logit-temperature", "linear-raw-box"),
        default="logit-temperature",
        help="Action map; automatically inherited from supported time checkpoints.",
    )
    parser.add_argument(
        "--center_state_correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="force the state correction to vanish on the nominal trajectory",
    )
    parser.add_argument(
        "--state_feature_mode",
        choices=[
            "log_absolute",
            "relative_nominal",
            "burden_composition",
            "total_burden",
        ],
        default="log_absolute",
    )
    parser.add_argument(
        "--state_mode",
        choices=["feedback", "fixed_query", "w_zero"],
        default="feedback",
    )
    parser.add_argument(
        "--training_integrator", choices=["euler", "rk4"], default="euler"
    )
    parser.add_argument("--train_radius", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--validation_size", type=int, default=64)
    parser.add_argument("--validation_seed", type=int, default=20260719)
    parser.add_argument("--test_size", type=int, default=64)
    parser.add_argument("--test_seed", type=int, default=20260720)
    parser.add_argument("--test_radii", default="0.05,0.10,0.20")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--time_lr_scale", type=float, default=0.1)
    parser.add_argument(
        "--time_unfreeze_epoch",
        type=int,
        default=0,
        help="freeze the time branch until this epoch; 0 trains it immediately",
    )
    parser.add_argument(
        "--freeze_time_branch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="keep the warm-start time branch fixed for the entire run",
    )
    parser.add_argument(
        "--selection_start_epoch",
        type=int,
        default=-1,
        help="first epoch eligible for best-checkpoint selection; -1 uses unfreeze epoch",
    )
    parser.add_argument(
        "--training_sample_seed_base",
        type=int,
        default=20260721,
        help="paired mini-batch stream seed is this value plus --seed",
    )
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--singular_eps", type=float, default=0.1)
    parser.add_argument("--singular_tau", type=float, default=0.03)
    parser.add_argument("--dot_eps", type=float, default=0.1)
    parser.add_argument("--dot_tau", type=float, default=0.03)
    parser.add_argument("--b_min", type=float, default=1e-8)
    parser.add_argument("--persistence_window", type=int, default=3)
    parser.add_argument("--detach_gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gate_gradient_mode", choices=["live", "detached"], default="live"
    )
    parser.add_argument(
        "--candidate_mask_mode",
        choices=["dynamic", "fixed_soft", "fixed_binary"],
        default="dynamic",
        help=(
            "use the live candidate mask, or freeze the incoming checkpoint's "
            "nominal time/stage mask for a controlled gate-escape diagnostic"
        ),
    )
    parser.add_argument(
        "--fixed_candidate_threshold",
        type=float,
        default=0.5,
        help="threshold used only by --candidate_mask_mode=fixed_binary",
    )
    parser.add_argument("--w0", type=float, default=1.0)
    parser.add_argument("--w1", type=float, default=1.0)
    parser.add_argument("--w2", type=float, default=1.0)
    parser.add_argument("--w_lc", type=float, default=1.0)
    parser.add_argument("--psi_scale", type=float, default=1.0)
    parser.add_argument("--dot_scale", type=float, default=1.0)
    parser.add_argument("--ddot_scale", type=float, default=1.0)
    parser.add_argument("--B_scale", type=float, default=1.0)
    parser.add_argument(
        "--auto_residual_scales",
        action="store_true",
        help="Fix DER residual RMS scales from the incoming nominal trajectory before training.",
    )
    parser.add_argument("--singular_loss_weight", type=float, default=1.0)
    parser.add_argument("--nonsingular_loss_weight", type=float, default=1.0)
    parser.add_argument("--smooth_weight", type=float, default=3.0)
    parser.add_argument("--smooth_second_weight", type=float, default=0.0)
    parser.add_argument("--smooth_max_weight", type=float, default=0.0)
    parser.add_argument("--smooth_max_tau", type=float, default=0.02)
    parser.add_argument(
        "--full_gradient_weight",
        type=float,
        default=0.0,
        help=(
            "weight of the complete RK4 reduced-gradient residual; the "
            "physical objective value itself is not added to the loss"
        ),
    )
    parser.add_argument(
        "--full_gradient_scale",
        type=float,
        default=0.0,
        help=(
            "positive scale for dF_h/du; 0 uses dt, yielding the interval-"
            "averaged H_u scale and a mesh-aware loss"
        ),
    )
    parser.add_argument(
        "--full_gradient_residual",
        choices=["raw", "projected"],
        default="raw",
        help=(
            "train the raw reduced gradient or the box-constrained projected-"
            "gradient mapping u-Pi(u-dF_h/du)"
        ),
    )
    parser.add_argument(
        "--full_gradient_projection_step",
        type=float,
        default=1.0,
        help=(
            "positive step in [u-Pi(u-step*dF_h/du)]/step; changing it "
            "preserves the box-KKT zero set while changing active-set "
            "identification"
        ),
    )
    parser.add_argument(
        "--full_gradient_ramp_epochs",
        type=int,
        default=0,
        help="linearly ramp the full-gradient weight over the first epochs",
    )
    parser.add_argument(
        "--full_gradient_max_weight",
        type=float,
        default=0.0,
        help="weight of a smooth maximum of the scaled full-gradient residual",
    )
    parser.add_argument(
        "--full_gradient_max_tau",
        type=float,
        default=0.1,
        help="log-sum-exp temperature on the interval-averaged H_u scale",
    )
    parser.add_argument("--state_blind", action="store_true")
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--float64", action="store_true")
    parser.add_argument("--out_dir", default="outputs/feedback_section5_pilot")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
