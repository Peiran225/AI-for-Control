#!/usr/bin/env python3
"""Refine feedback policies on a fixed-support dense off-grid rollout.

The time Transformer is frozen and queried with its original 801-token
context.  The state correction is then re-evaluated at the current state on
every fine subinterval.  The default ZOH mode uses the matching discrete RK4
adjoint.  The optional continuous-policy mode re-queries ``u(N,t)`` at every
RK4 stage and integrates the continuous PMP costate equation, matching the
formal DOP853 evaluation semantics more closely at a smaller grid multiplier.
Continuous-policy refinement can optionally collocate the scalar conditions
at both RK4 nodes and midpoints; the default remains node-only for backward
compatibility.  Either mode supplies the three scalar DER residuals or the
closed-form singular-control target.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.feedback_section5_rk4_reference import (  # noqa: E402
    RK4_B,
    discrete_rk4_adjoint,
    rk4_zoh_step,
    singular_quantities_at_points,
)
from scripts.feedback_continuous_policy_rk4 import (  # noqa: E402
    continuous_feedback_pmp_pack,
    pchip_midpoint_logits,
)
from scripts.fixed_nominal_opt_gap import (  # noqa: E402
    evaluate_fixed_nominal_der,
    metric_metadata,
)
from scripts.generate_offgrid_policy_switching_diagnostics import (  # noqa: E402
    DensePolicy,
    fixed_support_query_logits,
    integrate_trajectory,
    raw_time_logits,
)
from scripts.refine_feedback_scalar_lbfgs import (  # noqa: E402
    anchored_states,
    load_feedback_checkpoint,
)
from train_paper_pmp_kkt import ProblemConfig, build_params  # noqa: E402


def fine_problem(cfg: ProblemConfig, multiplier: int) -> ProblemConfig:
    if multiplier < 1:
        raise ValueError("multiplier must be positive")
    return ProblemConfig(
        T=cfg.T,
        n=cfg.n * multiplier,
        m=cfg.m,
        umax=cfg.umax,
        beta=cfg.beta,
        alpha=cfg.alpha,
        gamma=cfg.gamma,
        n0=cfg.n0,
        m_suppression=cfg.m_suppression,
    )


def _balanced_mixed_counts(
    count: int,
    composition_fraction: float,
) -> tuple[int, int]:
    if count < 2:
        raise ValueError(
            "balanced mixed sampling requires at least two stochastic states"
        )
    if not 0.0 < composition_fraction < 1.0:
        raise ValueError(
            "mixed composition fraction must lie strictly between 0 and 1"
        )
    composition_count = int(
        math.floor(count * composition_fraction + 0.5)
    )
    composition_count = min(max(composition_count, 1), count - 1)
    return count - composition_count, composition_count


def balanced_mixed_states(
    count: int,
    seed: int,
    random_radius: float,
    structured_radius: float,
    composition_fraction: float,
    cfg: ProblemConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return nominal, structured, iid, and fixed-total initial states."""

    componentwise_count, composition_count = _balanced_mixed_counts(
        count,
        composition_fraction,
    )
    if not 0.0 <= random_radius <= 0.10:
        raise ValueError("mixed random-state radius must lie in [0, 0.10]")
    if not math.isclose(
        structured_radius,
        0.10,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("the balanced structured anchor must use radius 0.10")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    componentwise_direction = 2.0 * torch.rand(
        componentwise_count,
        cfg.m,
        generator=generator,
        dtype=torch.float64,
    ) - 1.0
    componentwise = cfg.n0 * (
        1.0 + float(random_radius) * componentwise_direction
    )

    composition_direction = 2.0 * torch.rand(
        composition_count,
        cfg.m,
        generator=generator,
        dtype=torch.float64,
    ) - 1.0
    composition_direction = (
        composition_direction
        - composition_direction.mean(dim=-1, keepdim=True)
    )
    composition_direction = composition_direction / composition_direction.abs().amax(
        dim=-1,
        keepdim=True,
    ).clamp_min(torch.finfo(torch.float64).eps)
    amplitude = torch.rand(
        composition_count,
        1,
        generator=generator,
        dtype=torch.float64,
    )
    composition = cfg.n0 * (
        1.0 + float(random_radius) * amplitude * composition_direction
    )

    nominal = torch.full(
        (1, cfg.m),
        cfg.n0,
        dtype=torch.float64,
    )
    structured_direction = torch.linspace(
        -1.0,
        1.0,
        cfg.m,
        dtype=torch.float64,
    )
    structured = cfg.n0 * (
        1.0 + float(structured_radius) * structured_direction
    )
    states = torch.cat(
        (
            nominal,
            structured.unsqueeze(0),
            componentwise,
            composition,
        ),
        dim=0,
    )
    return states.to(device=device, dtype=dtype)


def group_balanced_state_weights(
    componentwise_count: int,
    composition_count: int,
    device: torch.device,
    dtype: torch.dtype,
    structured_weight: float = 1.0,
) -> torch.Tensor:
    """Assign empirical-mean group masses ``(1, s, 1, 1)``."""

    if componentwise_count < 1 or composition_count < 1:
        raise ValueError("each stochastic state group must be nonempty")
    if not math.isfinite(structured_weight) or structured_weight <= 0.0:
        raise ValueError("structured group weight must be finite and positive")
    return torch.cat(
        (
            torch.tensor(
                [1.0, structured_weight],
                device=device,
                dtype=dtype,
            ),
            torch.full(
                (componentwise_count,),
                1.0 / componentwise_count,
                device=device,
                dtype=dtype,
            ),
            torch.full(
                (composition_count,),
                1.0 / composition_count,
                device=device,
                dtype=dtype,
            ),
        )
    )


def scalar_lopt_by_group(
    pack: dict[str, torch.Tensor],
    componentwise_count: int,
    composition_count: int,
    *,
    w0: float,
    w1: float,
    w2: float,
) -> dict[str, torch.Tensor]:
    """Return empirical-mean scalar L_opt for the four state groups."""

    expected_count = 2 + componentwise_count + composition_count
    values = pack["quantities"]["psi"]
    if values.shape[0] != expected_count:
        raise ValueError("state-group counts do not match the scalar batch")
    mask = pack["weighted_mask"].expand_as(values)
    reduction_dims = tuple(range(1, values.ndim))
    denominator = mask.sum(dim=reduction_dims).clamp_min(
        torch.finfo(values.dtype).eps
    )

    per_state = values.new_zeros(values.shape[0])
    for weight, key in (
        (w0, "psi"),
        (w1, "dot_psi"),
        (w2, "ddot_psi"),
    ):
        residual = pack["quantities"][key]
        per_state = per_state + float(weight) * (
            (mask * residual.square()).sum(dim=reduction_dims)
            / denominator
        )
    componentwise_start = 2
    composition_start = componentwise_start + componentwise_count
    return {
        "nominal": per_state[0],
        "structured": per_state[1],
        "iid": per_state[componentwise_start:composition_start].mean(),
        "composition": per_state[composition_start:].mean(),
    }


def group_feasible_selection(
    current: dict[str, torch.Tensor],
    baseline: dict[str, torch.Tensor],
    *,
    nominal_ratio_limit: float = 1.25,
    structured_ratio_limit: float = 2.0,
    iid_ratio_limit: float = 0.8,
    composition_ratio_limit: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Return feasibility and the mean relative iid/composition score."""

    limits = {
        "nominal": nominal_ratio_limit,
        "structured": structured_ratio_limit,
        "iid": iid_ratio_limit,
        "composition": composition_ratio_limit,
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in limits.values()):
        raise ValueError("group-feasibility ratio limits must be positive")
    ratios = {
        name: current[name]
        / baseline[name].clamp_min(torch.finfo(current[name].dtype).eps)
        for name in limits
    }
    feasible = torch.stack(
        [ratios[name] <= limit for name, limit in limits.items()]
    ).all()
    score = 0.5 * (ratios["iid"] + ratios["composition"])
    return feasible, score, ratios


def validate_balanced_mixed_configuration(args: argparse.Namespace) -> None:
    """Require the clean continuous-policy DER protocol in balanced mode."""

    if (
        args.state_group_weighting == "group-balanced"
        and args.state_sampling != "balanced-mixed"
    ):
        raise ValueError(
            "group-balanced state weighting requires balanced-mixed sampling"
        )
    if (
        not math.isclose(args.structured_group_weight, 1.0)
        and args.state_group_weighting != "group-balanced"
    ):
        raise ValueError(
            "structured group weight requires group-balanced state weighting"
        )
    if (
        args.state_group_weighting == "group-balanced"
        and (
            not math.isfinite(args.structured_group_weight)
            or args.structured_group_weight <= 0.0
        )
    ):
        raise ValueError("structured group weight must be finite and positive")
    if args.group_feasible_selection and (
        args.state_sampling != "balanced-mixed"
        or args.state_group_weighting != "group-balanced"
    ):
        raise ValueError(
            "group-feasible selection requires group-balanced mixed states"
        )
    if args.group_feasible_selection and any(
        not math.isfinite(value) or value <= 0.0
        for value in (
            args.feasible_nominal_ratio,
            args.feasible_structured_ratio,
            args.feasible_iid_ratio,
            args.feasible_composition_ratio,
        )
    ):
        raise ValueError("group-feasibility ratio limits must be positive")
    if args.state_sampling != "balanced-mixed":
        return
    if args.option != "der":
        raise ValueError("balanced mixed refinement requires --option der")
    if args.trajectory_mode != "continuous-policy":
        raise ValueError(
            "balanced mixed refinement requires --trajectory-mode "
            "continuous-policy"
        )
    if args.continuous_collocation != "nodes":
        raise ValueError(
            "balanced mixed refinement uses the strict node-query evaluator"
        )
    if not (
        math.isclose(args.w0, 1.0)
        and math.isclose(args.w1, 1.0)
        and math.isclose(args.w2, 4.0)
    ):
        raise ValueError("balanced mixed refinement requires w=(1,1,4)")
    if not math.isclose(args.residual_p, 2.0):
        raise ValueError("balanced mixed refinement requires residual p=2")
    if args.cf_weight != 0.0 or args.cf_scalar_weight != 0.0:
        raise ValueError(
            "balanced mixed DER refinement does not use closed-form targets"
        )
    if (
        args.control_trust_weight != 0.0
        or args.protected_control_trust_weight != 0.0
        or args.protected_state_count != 0
    ):
        raise ValueError(
            "balanced mixed refinement does not use control-trust penalties"
        )
    if args.selection_metric != "scalar":
        raise ValueError(
            "balanced mixed refinement selects only by validation scalar L_opt"
        )
    if args.exclude_nominal_from_training:
        raise ValueError(
            "balanced mixed refinement includes the nominal training state"
        )
    if args.train_seed == args.validation_seed:
        raise ValueError(
            "training and validation stochastic states require independent seeds"
        )
    _balanced_mixed_counts(
        args.train_random_states,
        args.mixed_composition_fraction,
    )
    _balanced_mixed_counts(
        args.validation_random_states,
        args.mixed_composition_fraction,
    )
    if not 0.0 <= args.radius <= 0.10:
        raise ValueError("balanced mixed refinement requires radius <= 0.10")
    if not math.isclose(
        args.structured_radius,
        0.10,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("balanced mixed refinement requires structured radius 0.10")
    if not (
        math.isfinite(args.interval_start)
        and math.isfinite(args.interval_end)
        and 0.0 <= args.interval_start < args.interval_end
    ):
        raise ValueError("balanced mixed refinement requires a fixed interior interval")


def fixed_support_dense_logits(
    model: torch.nn.Module,
    cfg: ProblemConfig,
    multiplier: int,
    *,
    query_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Keep the dense policy queries on the same differentiable Transformer
    # arithmetic path used by scalar refinement and by the final DOP853
    # evaluator.  PyTorch's inference-only fused MHA path is close, but the
    # resulting logit perturbation is large enough to matter for this
    # sensitive controlled ODE.
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    model.time_branch.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    support = torch.linspace(
        0.0, 1.0, cfg.n + 1, device=device, dtype=dtype
    )
    dense = torch.linspace(
        0.0,
        1.0,
        cfg.n * multiplier + 1,
        device=device,
        dtype=dtype,
    )
    support_raw = raw_time_logits(model.time_branch, support)
    dense_raw = torch.empty_like(dense)
    support_indices = torch.arange(
        0, dense.numel(), multiplier, device=device
    )
    dense_raw[support_indices] = support_raw
    off_grid = torch.ones(
        dense.numel(), device=device, dtype=torch.bool
    )
    off_grid[support_indices] = False
    if off_grid.any():
        dense_raw[off_grid] = fixed_support_query_logits(
            model.time_branch,
            support,
            dense[off_grid],
            batch_size=query_batch_size,
        )
    return dense, dense_raw.detach()


def fine_feedback_rollout(
    model: torch.nn.Module,
    initial_state: torch.Tensor,
    cfg: ProblemConfig,
    dense_time: torch.Tensor,
    dense_raw_logits: torch.Tensor,
    params: dict[str, torch.Tensor],
    *,
    state_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = initial_state.shape[0]
    state = initial_state
    states = [state]
    controls: list[torch.Tensor] = []
    stage_states: list[torch.Tensor] = []
    step = cfg.T / cfg.n
    for index in range(cfg.n):
        normalized_time = dense_time[index].expand(batch)
        control = model.interval_action(
            dense_raw_logits[index],
            normalized_time,
            state,
            state_mode=state_mode,
        )
        state, stages = rk4_zoh_step(state, control, step, params)
        states.append(state)
        controls.append(control)
        stage_states.append(stages)
    return (
        torch.stack(states, dim=1),
        torch.stack(controls, dim=1),
        torch.stack(stage_states, dim=1),
    )


def scalar_pack(
    model: torch.nn.Module,
    initial_state: torch.Tensor,
    cfg: ProblemConfig,
    dense_time: torch.Tensor,
    dense_raw_logits: torch.Tensor,
    params: dict[str, torch.Tensor],
    *,
    trajectory_mode: str = "zoh",
    midpoint_raw_logits: torch.Tensor | None = None,
    continuous_collocation: str = "nodes",
    option: str,
    interval_start: float,
    interval_end: float,
    w0: float = 1.0,
    w1: float = 1.0,
    w2: float = 1.0,
    residual_p: float = 2.0,
    cf_weight: float = 1.0,
    cf_scalar_weight: float = 0.0,
    control_trust_weight: float = 0.0,
    anchor_controls: torch.Tensor | None = None,
    state_weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if continuous_collocation not in {"nodes", "nodes-midpoints"}:
        raise ValueError(
            "continuous_collocation must be 'nodes' or 'nodes-midpoints'"
        )
    if trajectory_mode == "zoh":
        if continuous_collocation != "nodes":
            raise ValueError(
                "midpoint collocation is available only in continuous-policy "
                "mode"
            )
        states, controls, stages = fine_feedback_rollout(
            model,
            initial_state,
            cfg,
            dense_time,
            dense_raw_logits,
            params,
            state_mode="feedback",
        )
        adjoint = discrete_rk4_adjoint(
            states, controls, stages, cfg, params
        )
        sample_controls = controls.unsqueeze(-1).expand(-1, -1, 4)
        quantities = singular_quantities_at_points(
            stages,
            sample_controls,
            adjoint.stage_costates,
            params,
        )
        fractions = torch.tensor(
            [0.0, 0.5, 0.5, 1.0],
            device=controls.device,
            dtype=controls.dtype,
        )
        sample_time = (
            torch.arange(
                cfg.n, device=controls.device, dtype=controls.dtype
            ).unsqueeze(-1)
            + fractions
        ) * (cfg.T / cfg.n)
        mask = (
            (sample_time >= interval_start)
            & (sample_time < interval_end)
        ).to(controls.dtype)
        quadrature_weights = torch.as_tensor(
            RK4_B, device=controls.device, dtype=controls.dtype
        ).view(1, 1, 4)
        weighted_mask = mask.unsqueeze(0) * quadrature_weights
        objective_running = (
            (
                stages * params["beta"]
            ).sum(dim=-1)
            + params["gamma"] * sample_controls
        ).mul(quadrature_weights).sum(dim=(1, 2))
    elif trajectory_mode == "continuous-policy":
        if midpoint_raw_logits is None:
            raise ValueError(
                "continuous-policy refinement requires midpoint raw logits"
            )
        continuous = continuous_feedback_pmp_pack(
            model,
            initial_state,
            cfg,
            dense_time,
            dense_raw_logits,
            midpoint_raw_logits,
            params,
            state_mode="feedback",
        )
        states = continuous.states
        controls = continuous.node_controls[:, :-1]
        if continuous_collocation == "nodes":
            sample_controls = controls
            quantities = continuous.quantities
            sample_time = dense_time[:-1] * cfg.T
            collocation_weights = None
        else:
            sample_controls = torch.stack(
                [controls, continuous.midpoint_controls],
                dim=-1,
            )
            quantities = {
                key: torch.stack(
                    [
                        continuous.quantities[key],
                        continuous.midpoint_quantities[key],
                    ],
                    dim=-1,
                )
                for key in continuous.quantities
            }
            midpoint_time = 0.5 * (
                dense_time[:-1] + dense_time[1:]
            )
            sample_time = torch.stack(
                [dense_time[:-1], midpoint_time],
                dim=-1,
            ) * cfg.T
            # Left nodes and interval midpoints form a uniform half-step
            # collocation grid.  Equal weights preserve the node-only loss
            # normalization.
            collocation_weights = torch.full(
                (1, 1, 2),
                0.5,
                device=controls.device,
                dtype=controls.dtype,
            )
        mask = (
            (sample_time >= interval_start)
            & (sample_time < interval_end)
        ).to(controls.dtype)
        weighted_mask = mask.unsqueeze(0)
        if collocation_weights is not None:
            weighted_mask = weighted_mask * collocation_weights
        left_running = (
            (states[:, :-1] * params["beta"]).sum(dim=-1)
            + params["gamma"] * continuous.node_controls[:, :-1]
        )
        midpoint_running = (
            (
                continuous.midpoint_states * params["beta"]
            ).sum(dim=-1)
            + params["gamma"] * continuous.midpoint_controls
        )
        right_running = (
            (states[:, 1:] * params["beta"]).sum(dim=-1)
            + params["gamma"] * continuous.node_controls[:, 1:]
        )
        objective_running = (
            left_running + 4.0 * midpoint_running + right_running
        ).sum(dim=1) / 6.0
    else:
        raise ValueError(
            "trajectory_mode must be 'zoh' or 'continuous-policy'"
        )
    diagnostic_denominator = (
        weighted_mask.sum() * controls.shape[0]
    ).clamp_min(torch.finfo(controls.dtype).eps)
    if state_weights is None:
        loss_mask = weighted_mask
        loss_denominator = diagnostic_denominator
    else:
        if (
            state_weights.ndim != 1
            or state_weights.shape[0] != controls.shape[0]
        ):
            raise ValueError("state weights must have one value per trajectory")
        if not torch.isfinite(state_weights).all():
            raise ValueError("state weights must be finite")
        if (state_weights < 0.0).any() or state_weights.sum() <= 0.0:
            raise ValueError(
                "state weights must be nonnegative with positive sum"
            )
        state_weights = state_weights.to(
            device=controls.device,
            dtype=controls.dtype,
        )
        state_weight_shape = (
            controls.shape[0],
            *([1] * (weighted_mask.ndim - 1)),
        )
        loss_mask = weighted_mask * state_weights.view(state_weight_shape)
        loss_denominator = (
            weighted_mask.sum() * state_weights.sum()
        ).clamp_min(torch.finfo(controls.dtype).eps)

    def mean_square(value: torch.Tensor) -> torch.Tensor:
        return (loss_mask * value.square()).sum() / loss_denominator

    def residual_measure(value: torch.Tensor) -> torch.Tensor:
        if math.isclose(float(residual_p), 2.0):
            return mean_square(value)
        # Express the weighted p-norm on the same squared scale as an MSE.
        # Detaching the scale changes only numerical conditioning, not the
        # represented norm or its optimizer direction.
        scale = value.detach().abs().max().clamp_min(
            torch.finfo(value.dtype).tiny
        )
        powered_mean = (
            loss_mask * (value.abs() / scale).pow(float(residual_p))
        ).sum() / loss_denominator
        return scale.square() * powered_mean.pow(2.0 / float(residual_p))

    if option == "der":
        component_losses = {
            "H_u": residual_measure(quantities["psi"]),
            "dH_u_dt": residual_measure(quantities["dot_psi"]),
            "d2H_u_dt2": residual_measure(quantities["ddot_psi"]),
        }
        loss = (
            float(w0) * component_losses["H_u"]
            + float(w1) * component_losses["dH_u_dt"]
            + float(w2) * component_losses["d2H_u_dt2"]
        )
    elif option == "cf":
        candidate_error = sample_controls - quantities["u_state"]
        component_losses = {
            "closed_form_control": mean_square(candidate_error),
            "H_u": residual_measure(quantities["psi"]),
            "dH_u_dt": residual_measure(quantities["dot_psi"]),
            "d2H_u_dt2": residual_measure(quantities["ddot_psi"]),
        }
        loss = float(cf_weight) * component_losses["closed_form_control"]
        loss = loss + float(cf_scalar_weight) * (
            float(w0) * component_losses["H_u"]
            + float(w1) * component_losses["dH_u_dt"]
            + float(w2) * component_losses["d2H_u_dt2"]
        )
    else:
        raise ValueError("option must be 'cf' or 'der'")
    if control_trust_weight:
        if anchor_controls is None:
            raise ValueError(
                "a nonzero control trust weight requires anchor controls"
            )
        if anchor_controls.shape == sample_controls.shape:
            trust_target = anchor_controls
        elif anchor_controls.shape == controls.shape:
            trust_target = (
                anchor_controls.unsqueeze(-1)
                if sample_controls.ndim == 3
                else anchor_controls
            )
        else:
            raise ValueError(
                "anchor controls do not match either the node or collocation "
                "control shape"
            )
        trust_error = sample_controls - trust_target
        component_losses["control_trust"] = mean_square(trust_error)
        loss = loss + (
            float(control_trust_weight)
            * component_losses["control_trust"]
        )

    objective = (
        (states[:, -1] * params["alpha"]).sum(dim=-1)
        + (cfg.T / cfg.n) * objective_running
    )
    return {
        "loss": loss,
        "component_losses": component_losses,
        "states": states,
        "controls": controls,
        "sample_controls": sample_controls,
        "quantities": quantities,
        "weighted_mask": weighted_mask,
        "denominator": diagnostic_denominator,
        "loss_denominator": loss_denominator,
        "state_weights": state_weights,
        "objective": objective,
    }


def physical_metrics(
    pack: dict[str, torch.Tensor],
    *,
    scale: float,
) -> dict[str, float]:
    weighted_mask = pack["weighted_mask"]
    denominator = pack["denominator"]
    result: dict[str, float] = {}
    for label, key in (
        ("H_u", "psi"),
        ("dH_u_dt", "dot_psi"),
        ("d2H_u_dt2", "ddot_psi"),
    ):
        values = pack["quantities"][key]
        rms = (
            (weighted_mask * values.square()).sum() / denominator
        ).sqrt()
        active = weighted_mask.expand_as(values) > 0.0
        result[f"{label}_physical_rms"] = float(scale * rms.detach().cpu())
        result[f"{label}_physical_max_abs"] = float(
            scale * values[active].abs().max().detach().cpu()
        )
    if "closed_form_control" in pack["component_losses"]:
        candidate_error = (
            pack["sample_controls"]
            - pack["quantities"]["u_state"]
        )
        active = weighted_mask.expand_as(candidate_error) > 0.0
        result["closed_form_control_rms"] = float(
            (
                (weighted_mask * candidate_error.square()).sum()
                / denominator
            )
            .sqrt()
            .detach()
            .cpu()
        )
        result["closed_form_control_max_abs"] = float(
            candidate_error[active].abs().max().detach().cpu()
        )
    result["normalized_objective_mean"] = float(
        pack["objective"].mean().detach().cpu()
    )
    result["physical_objective_mean"] = float(
        scale * pack["objective"].mean().detach().cpu()
    )
    return result


def physical_metrics_by_sample(
    pack: dict[str, torch.Tensor],
    *,
    scale: float,
    names: list[str],
) -> dict[str, dict[str, float]]:
    """Return the same physical diagnostics without pooling initial states."""

    values = pack["quantities"]["psi"]
    if values.shape[0] != len(names):
        raise ValueError("sample names do not match the scalar diagnostic batch")
    weighted_mask = pack["weighted_mask"].expand_as(values)
    result: dict[str, dict[str, float]] = {}
    for sample_index, name in enumerate(names):
        sample_mask = weighted_mask[sample_index] > 0.0
        denominator = weighted_mask[sample_index].sum().clamp_min(
            torch.finfo(values.dtype).eps
        )
        row: dict[str, float] = {}
        for label, key in (
            ("H_u", "psi"),
            ("dH_u_dt", "dot_psi"),
            ("d2H_u_dt2", "ddot_psi"),
        ):
            sample_values = pack["quantities"][key][sample_index]
            row[f"{label}_physical_rms"] = float(
                scale
                * (
                    (
                        weighted_mask[sample_index]
                        * sample_values.square()
                    ).sum()
                    / denominator
                )
                .sqrt()
                .detach()
                .cpu()
            )
            row[f"{label}_physical_max_abs"] = float(
                scale
                * sample_values[sample_mask].abs().max().detach().cpu()
            )
        if "closed_form_control" in pack["component_losses"]:
            candidate_error = (
                pack["sample_controls"][sample_index]
                - pack["quantities"]["u_state"][sample_index]
            )
            row["closed_form_control_rms"] = float(
                (
                    (
                        weighted_mask[sample_index]
                        * candidate_error.square()
                    ).sum()
                    / denominator
                )
                .sqrt()
                .detach()
                .cpu()
            )
            row["closed_form_control_max_abs"] = float(
                candidate_error[sample_mask].abs().max().detach().cpu()
            )
        row["normalized_objective"] = float(
            pack["objective"][sample_index].detach().cpu()
        )
        row["physical_objective"] = float(
            scale * pack["objective"][sample_index].detach().cpu()
        )
        result[name] = row
    return result


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def protected_control_trust(
    sample_controls: torch.Tensor,
    anchor_controls: torch.Tensor,
    protected_state_count: int,
) -> torch.Tensor:
    """Keep the first protected trajectories close to the input checkpoint."""

    if protected_state_count <= 0:
        return sample_controls.new_zeros(())
    if sample_controls.shape != anchor_controls.shape:
        raise ValueError(
            "protected control anchors must match the collocation controls"
        )
    if protected_state_count > sample_controls.shape[0]:
        raise ValueError(
            "protected state count exceeds the available state batch"
        )
    return (
        sample_controls[:protected_state_count]
        - anchor_controls[:protected_state_count]
    ).square().mean()


def reset_state_branch_reproducibly(
    model: NestedFeedbackTransformer,
    seed: int,
) -> None:
    """Reinitialize only the feedback branch while preserving the time policy.

    ``reset_state_branch`` uses Xavier initialization for the hidden layers and
    an exactly zero final affine layer.  Forking the RNG makes this optional
    restart reproducible without changing the caller's global RNG stream.
    """

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model.reset_state_branch()


def run(args: argparse.Namespace) -> None:
    validate_balanced_mixed_configuration(args)
    if not math.isfinite(args.residual_p) or args.residual_p < 2.0:
        raise ValueError("--residual-p must be finite and at least 2")
    if min(args.w0, args.w1, args.w2) < 0.0:
        raise ValueError("scalar residual weights must be nonnegative")
    if args.w0 + args.w1 + args.w2 <= 0.0:
        raise ValueError("at least one scalar residual weight must be positive")
    if min(
        args.cf_weight,
        args.cf_scalar_weight,
        args.control_trust_weight,
    ) < 0.0:
        raise ValueError(
            "closed-form and trust-region weights must be nonnegative"
        )
    if args.option == "cf" and (
        args.cf_weight
        + args.cf_scalar_weight
        + args.control_trust_weight
        <= 0.0
    ):
        raise ValueError("the closed-form loss must contain a live term")
    if (
        args.trajectory_mode != "continuous-policy"
        and args.continuous_collocation != "nodes"
    ):
        raise ValueError(
            "--continuous-collocation nodes-midpoints requires "
            "--trajectory-mode continuous-policy"
        )
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    # Use the same Transformer arithmetic path as training and the formal
    # off-grid evaluator.  The fused inference MHA path can shift fixed-support
    # query logits enough to alter this sensitive trajectory.
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint_path = args.checkpoint.expanduser().resolve()
    model, base_cfg, source_args = load_feedback_checkpoint(
        checkpoint_path
    )
    if args.reset_state_branch:
        reset_state_branch_reproducibly(
            model,
            args.state_branch_init_seed,
        )
    if (
        args.state_sampling == "balanced-mixed"
        and args.interval_end > base_cfg.T
    ):
        raise ValueError("balanced mixed interior support must lie within [0,T]")
    cfg = fine_problem(base_cfg, args.multiplier)
    model.to(device=device, dtype=torch.float64)
    params = build_params(cfg, device, torch.float64)
    fixed_validation_params = build_params(
        base_cfg, device, torch.float64
    )
    model.set_feature_vectors(params["r"], params["phi"])
    dense_time, dense_raw = fixed_support_dense_logits(
        model,
        base_cfg,
        args.multiplier,
        query_batch_size=args.query_batch_size,
    )
    dense_midpoint_raw = (
        pchip_midpoint_logits(dense_time, dense_raw)
        if args.trajectory_mode == "continuous-policy"
        else None
    )

    if args.refresh_nominal_reference:
        reference_multiplier = (
            args.multiplier
            if args.reference_multiplier == 0
            else args.reference_multiplier
        )
        if reference_multiplier == args.multiplier:
            reference_cfg = cfg
            reference_time = dense_time
            reference_raw = dense_raw
        else:
            reference_cfg = fine_problem(base_cfg, reference_multiplier)
            reference_time, reference_raw = fixed_support_dense_logits(
                model,
                base_cfg,
                reference_multiplier,
                query_batch_size=args.query_batch_size,
            )
        reference_midpoint_raw = (
            pchip_midpoint_logits(reference_time, reference_raw)
            if args.trajectory_mode == "continuous-policy"
            else None
        )
        nominal = torch.full(
            (1, reference_cfg.m),
            reference_cfg.n0,
            device=device,
            dtype=torch.float64,
        )
        if args.reference_method == "zoh":
            with torch.no_grad():
                if args.trajectory_mode == "continuous-policy":
                    if reference_midpoint_raw is None:
                        raise RuntimeError(
                            "continuous reference midpoint logits are missing"
                        )
                    reference_states = continuous_feedback_pmp_pack(
                        model,
                        nominal,
                        reference_cfg,
                        reference_time,
                        reference_raw,
                        reference_midpoint_raw,
                        params,
                        state_mode="w_zero",
                    ).states
                else:
                    reference_states, _, _ = fine_feedback_rollout(
                        model,
                        nominal,
                        reference_cfg,
                        reference_time,
                        reference_raw,
                        params,
                        state_mode="w_zero",
                    )
            model.set_nominal_reference(reference_states[0])
        else:
            if model.action_parameterization == "linear-raw-box":
                time_wrapper: dict[str, Any] = {
                    "class": "LinearRawBoxProjection"
                }
            elif model.action_offset:
                if not math.isclose(model.action_temperature, 1.0):
                    raise ValueError(
                        "the DOP853 reference does not support a simultaneous "
                        "nonunit action temperature and nonzero offset"
                    )
                time_wrapper = {
                    "class": "AffineBoundaryProjectedControl",
                    "scale": model.action_scale,
                    "offset": model.action_offset,
                }
            else:
                time_wrapper = {
                    "class": "FixedBoxProjection",
                    "scale": model.action_scale,
                    "temperature": model.action_temperature,
                }
            reference_policy = DensePolicy(
                "time_only_reference",
                base_cfg,
                (
                    reference_time.detach().cpu().numpy()
                    * base_cfg.T
                ),
                reference_raw.detach().cpu().numpy(),
                raw_time_logits(
                    model.time_branch,
                    torch.linspace(
                        0.0,
                        1.0,
                        base_cfg.n + 1,
                        device=device,
                        dtype=torch.float64,
                    ),
                )
                .detach()
                .cpu()
                .numpy(),
                checkpoint_path,
                time_wrapper=time_wrapper,
            )
            reference_result = integrate_trajectory(
                reference_policy,
                np.full(base_cfg.m, base_cfg.n0, dtype=np.float64),
                reference_policy.dense_time,
                rtol=args.reference_rtol,
                atol=args.reference_atol,
                max_step=(
                    base_cfg.T / (base_cfg.n * reference_multiplier)
                ),
            )
            model.set_nominal_reference(
                torch.from_numpy(reference_result.state).to(
                    device=device, dtype=torch.float64
                )
            )

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    linears = [
        module
        for module in model.state_branch.modules()
        if isinstance(module, torch.nn.Linear)
    ]
    if args.train_scope == "last_layer":
        trainable_modules = [linears[-1]]
    else:
        trainable_modules = [model.state_branch]
    trainable: list[torch.nn.Parameter] = []
    for module in trainable_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            trainable.append(parameter)

    if args.state_sampling == "balanced-mixed":
        train_states = balanced_mixed_states(
            args.train_random_states,
            args.train_seed,
            args.radius,
            args.structured_radius,
            args.mixed_composition_fraction,
            cfg,
            device,
            torch.float64,
        )
        validation_states = balanced_mixed_states(
            args.validation_random_states,
            args.validation_seed,
            args.radius,
            args.structured_radius,
            args.mixed_composition_fraction,
            cfg,
            device,
            torch.float64,
        )
        train_componentwise_count, train_composition_count = (
            _balanced_mixed_counts(
                args.train_random_states,
                args.mixed_composition_fraction,
            )
        )
        validation_componentwise_count, validation_composition_count = (
            _balanced_mixed_counts(
                args.validation_random_states,
                args.mixed_composition_fraction,
            )
        )
        if args.state_group_weighting == "group-balanced":
            train_state_weights = group_balanced_state_weights(
                train_componentwise_count,
                train_composition_count,
                device,
                torch.float64,
                args.structured_group_weight,
            )
            validation_state_weights = group_balanced_state_weights(
                validation_componentwise_count,
                validation_composition_count,
                device,
                torch.float64,
                args.structured_group_weight,
            )
        else:
            train_state_weights = None
            validation_state_weights = None
    else:
        train_states = anchored_states(
            args.train_random_states,
            args.train_seed,
            args.radius,
            cfg,
            device,
            torch.float64,
        )
        if args.exclude_nominal_from_training:
            train_states = train_states[1:]
        validation_states = anchored_states(
            args.validation_random_states,
            args.validation_seed,
            args.radius,
            cfg,
            device,
            torch.float64,
        )
        train_state_weights = None
        validation_state_weights = None

    def realized_controls(states: torch.Tensor) -> torch.Tensor:
        if args.trajectory_mode == "continuous-policy":
            if dense_midpoint_raw is None:
                raise RuntimeError(
                    "continuous-policy midpoint logits are missing"
                )
            continuous = continuous_feedback_pmp_pack(
                model,
                states,
                cfg,
                dense_time,
                dense_raw,
                dense_midpoint_raw,
                params,
                state_mode="feedback",
            )
            node_controls = continuous.node_controls[:, :-1]
            if args.continuous_collocation == "nodes-midpoints":
                return torch.stack(
                    [node_controls, continuous.midpoint_controls],
                    dim=-1,
                )
            return node_controls
        return fine_feedback_rollout(
            model,
            states,
            cfg,
            dense_time,
            dense_raw,
            params,
            state_mode="feedback",
        )[1]

    with torch.no_grad():
        train_anchor_controls = realized_controls(train_states)
        validation_anchor_controls = realized_controls(validation_states)
    train_anchor_controls = train_anchor_controls.detach()
    validation_anchor_controls = validation_anchor_controls.detach()
    if args.state_sampling == "balanced-mixed":
        train_state_names = [
            "nominal",
            "structured_r0p10",
            *[
                f"componentwise_{index:03d}"
                for index in range(train_componentwise_count)
            ],
            *[
                f"composition_{index:03d}"
                for index in range(train_composition_count)
            ],
        ]
        validation_state_names = [
            "nominal",
            "structured_r0p10",
            *[
                f"componentwise_{index:03d}"
                for index in range(validation_componentwise_count)
            ],
            *[
                f"composition_{index:03d}"
                for index in range(validation_composition_count)
            ],
        ]
    else:
        train_state_names = [
            "resistant_heavy",
            *[
                f"random_{index:03d}"
                for index in range(args.train_random_states)
            ],
        ]
        if not args.exclude_nominal_from_training:
            train_state_names.insert(0, "nominal")
        validation_state_names = [
            "nominal",
            "resistant_heavy",
            *[
                f"random_{index:03d}"
                for index in range(args.validation_random_states)
            ],
        ]
    if args.protected_control_trust_weight < 0.0:
        raise ValueError("protected control trust weight must be nonnegative")
    if args.protected_state_count < 0:
        raise ValueError("protected state count must be nonnegative")
    if args.protected_control_trust_weight and not args.protected_state_count:
        raise ValueError(
            "a protected control trust weight requires protected states"
        )
    if args.protected_state_count > min(
        train_states.shape[0], validation_states.shape[0]
    ):
        raise ValueError(
            "protected state count exceeds the training or validation batch"
        )
    optimizer = torch.optim.LBFGS(
        trainable,
        lr=args.lr,
        max_iter=args.inner_iterations,
        max_eval=args.max_eval,
        tolerance_grad=args.tolerance_grad,
        tolerance_change=args.tolerance_change,
        history_size=args.history_size,
        line_search_fn="strong_wolfe",
    )

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    initial_metrics_by_state: dict[str, dict[str, float]] | None = None
    best_loss = math.inf
    best_step = 0
    best_state = copy.deepcopy(model.state_dict())
    best_reference = model.nominal_reference.detach().cpu().clone()
    baseline_validation_group_lopt: dict[str, torch.Tensor] | None = None
    best_validation_group_lopt: dict[str, float] | None = None
    best_selection_feasible = False

    def evaluate(step: int) -> float:
        nonlocal best_loss, best_step, best_state, best_reference
        nonlocal initial_metrics_by_state
        nonlocal baseline_validation_group_lopt
        nonlocal best_validation_group_lopt, best_selection_feasible
        model.eval()
        with torch.no_grad():
            train_pack = scalar_pack(
                model,
                train_states,
                cfg,
                dense_time,
                dense_raw,
                params,
                trajectory_mode=args.trajectory_mode,
                midpoint_raw_logits=dense_midpoint_raw,
                continuous_collocation=args.continuous_collocation,
                option=args.option,
                interval_start=args.interval_start,
                interval_end=args.interval_end,
                w0=args.w0,
                w1=args.w1,
                w2=args.w2,
                residual_p=args.residual_p,
                cf_weight=args.cf_weight,
                cf_scalar_weight=args.cf_scalar_weight,
                control_trust_weight=args.control_trust_weight,
                anchor_controls=train_anchor_controls,
                state_weights=train_state_weights,
            )
            validation_pack = scalar_pack(
                model,
                validation_states,
                cfg,
                dense_time,
                dense_raw,
                params,
                trajectory_mode=args.trajectory_mode,
                midpoint_raw_logits=dense_midpoint_raw,
                continuous_collocation=args.continuous_collocation,
                option=args.option,
                interval_start=args.interval_start,
                interval_end=args.interval_end,
                w0=args.w0,
                w1=args.w1,
                w2=args.w2,
                residual_p=args.residual_p,
                cf_weight=args.cf_weight,
                cf_scalar_weight=args.cf_scalar_weight,
                control_trust_weight=args.control_trust_weight,
                anchor_controls=validation_anchor_controls,
                state_weights=validation_state_weights,
            )
            if args.exclude_nominal_from_training:
                selection_pack = scalar_pack(
                    model,
                    validation_states[1:],
                    cfg,
                    dense_time,
                    dense_raw,
                    params,
                    trajectory_mode=args.trajectory_mode,
                    midpoint_raw_logits=dense_midpoint_raw,
                    continuous_collocation=args.continuous_collocation,
                    option=args.option,
                    interval_start=args.interval_start,
                    interval_end=args.interval_end,
                    w0=args.w0,
                    w1=args.w1,
                    w2=args.w2,
                    residual_p=args.residual_p,
                    cf_weight=args.cf_weight,
                    cf_scalar_weight=args.cf_scalar_weight,
                    control_trust_weight=args.control_trust_weight,
                    anchor_controls=validation_anchor_controls[1:],
                    state_weights=(
                        None
                        if validation_state_weights is None
                        else validation_state_weights[1:]
                    ),
                )
            else:
                selection_pack = validation_pack
        validation_group_lopt: dict[str, torch.Tensor] | None = None
        feasibility = None
        feasibility_score = None
        feasibility_ratios: dict[str, torch.Tensor] | None = None
        if args.state_sampling == "balanced-mixed":
            validation_group_lopt = scalar_lopt_by_group(
                validation_pack,
                validation_componentwise_count,
                validation_composition_count,
                w0=args.w0,
                w1=args.w1,
                w2=args.w2,
            )
            if baseline_validation_group_lopt is None:
                baseline_validation_group_lopt = {
                    name: value.detach().clone()
                    for name, value in validation_group_lopt.items()
                }
            if args.group_feasible_selection:
                (
                    feasibility,
                    feasibility_score,
                    feasibility_ratios,
                ) = group_feasible_selection(
                    validation_group_lopt,
                    baseline_validation_group_lopt,
                    nominal_ratio_limit=args.feasible_nominal_ratio,
                    structured_ratio_limit=args.feasible_structured_ratio,
                    iid_ratio_limit=args.feasible_iid_ratio,
                    composition_ratio_limit=(
                        args.feasible_composition_ratio
                    ),
                )
        train_protected_trust = protected_control_trust(
            train_pack["sample_controls"],
            train_anchor_controls,
            args.protected_state_count,
        )
        validation_protected_trust = protected_control_trust(
            validation_pack["sample_controls"],
            validation_anchor_controls,
            args.protected_state_count,
        )
        if args.group_feasible_selection:
            assert feasibility_score is not None
            selection_value = feasibility_score
        elif args.selection_metric == "scalar":
            selection_value = (
                float(args.w0)
                * selection_pack["component_losses"]["H_u"]
                + float(args.w1)
                * selection_pack["component_losses"]["dH_u_dt"]
                + float(args.w2)
                * selection_pack["component_losses"]["d2H_u_dt2"]
            )
        else:
            selection_value = selection_pack["loss"]
        selection_value = selection_value + (
            float(args.protected_control_trust_weight)
            * validation_protected_trust
        )
        metrics = physical_metrics(
            validation_pack, scale=args.report_scale_factor
        )
        fixed_validation = evaluate_fixed_nominal_der(
            model,
            base_cfg,
            fixed_validation_params,
            state_mode="feedback",
        )
        row = {
            "phase": "state_time_feedback_adaptation",
            "subphase": "scalar_optimality_gap_refinement",
            "phase_step": step,
            "optimizer_step": step,
            "outer_step": step,
            "train_loss": float(
                (
                    train_pack["loss"]
                    + float(args.protected_control_trust_weight)
                    * train_protected_trust
                ).cpu()
            ),
            "validation_loss": float(
                (
                    validation_pack["loss"]
                    + float(args.protected_control_trust_weight)
                    * validation_protected_trust
                ).cpu()
            ),
            "selection_loss": float(selection_value.cpu()),
            "train_protected_control_trust": float(
                train_protected_trust.cpu()
            ),
            "validation_protected_control_trust": float(
                validation_protected_trust.cpu()
            ),
            **fixed_validation,
            **metrics,
        }
        if validation_group_lopt is not None:
            row.update(
                {
                    f"validation_lopt_{name}": float(value.cpu())
                    for name, value in validation_group_lopt.items()
                }
            )
        if feasibility_ratios is not None:
            row.update(
                {
                    f"validation_lopt_{name}_ratio": float(value.cpu())
                    for name, value in feasibility_ratios.items()
                }
            )
            row["selection_group_feasible"] = bool(feasibility.cpu())
        history.append(row)
        if step == 0:
            initial_metrics_by_state = physical_metrics_by_sample(
                validation_pack,
                scale=args.report_scale_factor,
                names=validation_state_names,
            )
        if args.group_feasible_selection:
            is_feasible = bool(feasibility.cpu())
            should_select = (
                step == 0
                or (
                    is_feasible
                    and (
                        not best_selection_feasible
                        or row["selection_loss"] < best_loss
                    )
                )
            )
        else:
            is_feasible = True
            should_select = row["selection_loss"] < best_loss
        if should_select:
            best_loss = row["selection_loss"]
            best_step = step
            best_state = copy.deepcopy(model.state_dict())
            best_reference = model.nominal_reference.detach().cpu().clone()
            best_selection_feasible = is_feasible
            if validation_group_lopt is not None:
                best_validation_group_lopt = {
                    name: float(value.cpu())
                    for name, value in validation_group_lopt.items()
                }
        print(
            f"[{step:03d}] train={row['train_loss']:.8g} "
            f"val={row['validation_loss']:.8g} "
            f"select={row['selection_loss']:.8g} "
            f"physical=({row['H_u_physical_rms']:.4g},"
            f"{row['dH_u_dt_physical_rms']:.4g},"
            f"{row['d2H_u_dt2_physical_rms']:.4g})",
            flush=True,
        )
        return row["validation_loss"]

    evaluate(0)
    for step in range(1, args.outer_steps + 1):
        model.train()

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            pack = scalar_pack(
                model,
                train_states,
                cfg,
                dense_time,
                dense_raw,
                params,
                trajectory_mode=args.trajectory_mode,
                midpoint_raw_logits=dense_midpoint_raw,
                continuous_collocation=args.continuous_collocation,
                option=args.option,
                interval_start=args.interval_start,
                interval_end=args.interval_end,
                w0=args.w0,
                w1=args.w1,
                w2=args.w2,
                residual_p=args.residual_p,
                cf_weight=args.cf_weight,
                cf_scalar_weight=args.cf_scalar_weight,
                control_trust_weight=args.control_trust_weight,
                anchor_controls=train_anchor_controls,
                state_weights=train_state_weights,
            )
            protected_trust = protected_control_trust(
                pack["sample_controls"],
                train_anchor_controls,
                args.protected_state_count,
            )
            loss = pack["loss"] + (
                float(args.protected_control_trust_weight)
                * protected_trust
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        evaluate(step)

    model.load_state_dict(best_state)
    model.set_nominal_reference(
        best_reference.to(device=device, dtype=torch.float64)
    )
    model.eval()
    with torch.no_grad():
        selected_train_pack = scalar_pack(
            model,
            train_states,
            cfg,
            dense_time,
            dense_raw,
            params,
            trajectory_mode=args.trajectory_mode,
            midpoint_raw_logits=dense_midpoint_raw,
            continuous_collocation=args.continuous_collocation,
            option=args.option,
            interval_start=args.interval_start,
            interval_end=args.interval_end,
            w0=args.w0,
            w1=args.w1,
            w2=args.w2,
            residual_p=args.residual_p,
            cf_weight=args.cf_weight,
            cf_scalar_weight=args.cf_scalar_weight,
            control_trust_weight=args.control_trust_weight,
            anchor_controls=train_anchor_controls,
            state_weights=train_state_weights,
        )
        selected_validation_pack = scalar_pack(
            model,
            validation_states,
            cfg,
            dense_time,
            dense_raw,
            params,
            trajectory_mode=args.trajectory_mode,
            midpoint_raw_logits=dense_midpoint_raw,
            continuous_collocation=args.continuous_collocation,
            option=args.option,
            interval_start=args.interval_start,
            interval_end=args.interval_end,
            w0=args.w0,
            w1=args.w1,
            w2=args.w2,
            residual_p=args.residual_p,
            cf_weight=args.cf_weight,
            cf_scalar_weight=args.cf_scalar_weight,
            control_trust_weight=args.control_trust_weight,
            anchor_controls=validation_anchor_controls,
            state_weights=validation_state_weights,
        )
    selected_train_metrics_by_state = physical_metrics_by_sample(
        selected_train_pack,
        scale=args.report_scale_factor,
        names=train_state_names,
    )
    selected_validation_metrics_by_state = physical_metrics_by_sample(
        selected_validation_pack,
        scale=args.report_scale_factor,
        names=validation_state_names,
    )
    output_args = vars(source_args).copy()
    output_args.update(
        {
            "option": args.option,
            "loss_variant": "lc_live" if args.option == "der" else "literal",
            "state_feature_mode": model.state_feature_mode,
            "center_state_correction": model.center_state_correction,
            "state_mode": "feedback",
            "training_integrator": (
                "continuous-policy-rk4"
                if args.trajectory_mode == "continuous-policy"
                else "rk4-zoh"
            ),
            "continuous_collocation": args.continuous_collocation,
            "offgrid_refinement_multiplier": args.multiplier,
            "offgrid_refinement_interval": [
                args.interval_start,
                args.interval_end,
            ],
            "random_state_radius": args.radius,
            "train_random_states": args.train_random_states,
            "validation_random_states": args.validation_random_states,
            "train_random_state_seed": args.train_seed,
            "validation_random_state_seed": args.validation_seed,
        }
    )
    if args.state_sampling == "balanced-mixed":
        output_args.update(
            {
                "state_sampling": args.state_sampling,
                "structured_state_radius": args.structured_radius,
                "mixed_composition_fraction": (
                    args.mixed_composition_fraction
                ),
                "state_group_weighting": args.state_group_weighting,
                "structured_group_weight": args.structured_group_weight,
                "group_feasible_selection": args.group_feasible_selection,
            }
        )
    payload = {
        "model_state": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "args": output_args,
        "problem": base_cfg.__dict__,
        "best_validation_loss": best_loss,
        "best_selection_loss": best_loss,
        "best_epoch": best_step,
        "selection_metric": (
            "fixed_support_offgrid_group_feasible_relative_iid_composition"
            if args.group_feasible_selection
            else (
                "fixed_support_offgrid_"
                + (
                    "non_nominal_"
                    if args.exclude_nominal_from_training
                    else ""
                )
                + (
                    "scalar_residual_loss"
                    if args.selection_metric == "scalar"
                    else "combined_training_loss"
                )
            )
        ),
        "nominal_reference": model.nominal_reference.detach().cpu(),
        "initialization_checkpoint": str(checkpoint_path),
        "refinement": {
            "optimizer": "LBFGS",
            "train_scope": args.train_scope,
            "direct_supervision_used": False,
            "physical_objective_used": False,
            "multiplier": args.multiplier,
            "nominal_reference_multiplier": (
                args.multiplier
                if args.reference_multiplier == 0
                else args.reference_multiplier
            ),
            "nominal_reference_method": args.reference_method,
            "time_context_tokens": base_cfg.n + 1,
            "trajectory_mode": args.trajectory_mode,
            "continuous_collocation": args.continuous_collocation,
            "state_requery": (
                "every RK4 stage"
                if args.trajectory_mode == "continuous-policy"
                else "every fine subinterval"
            ),
            "nominal_in_scalar_training": (
                not args.exclude_nominal_from_training
            ),
            "interval": [args.interval_start, args.interval_end],
            "residual_weights": [args.w0, args.w1, args.w2],
            "residual_p": args.residual_p,
            "closed_form_weight": args.cf_weight,
            "closed_form_scalar_weight": args.cf_scalar_weight,
            "control_trust_weight": args.control_trust_weight,
            "protected_control_trust_weight": (
                args.protected_control_trust_weight
            ),
            "protected_state_count": args.protected_state_count,
            "random_state_radius": args.radius,
            "train_random_states": args.train_random_states,
            "validation_random_states": args.validation_random_states,
            "train_random_state_seed": args.train_seed,
            "validation_random_state_seed": args.validation_seed,
            "checkpoint_selection": args.selection_metric,
            "state_branch_initialization": (
                "zero-output restart"
                if args.reset_state_branch
                else "incoming checkpoint"
            ),
            "state_branch_initialization_seed": (
                args.state_branch_init_seed
                if args.reset_state_branch
                else None
            ),
        },
    }
    if args.state_sampling == "balanced-mixed":
        payload["refinement"].update(
            {
                "state_sampling": args.state_sampling,
                "structured_state_radius": args.structured_radius,
                "mixed_composition_fraction": (
                    args.mixed_composition_fraction
                ),
                "validation_is_independent": True,
                "state_group_weighting": args.state_group_weighting,
                "group_balanced_scalar_objective": (
                    args.state_group_weighting == "group-balanced"
                ),
                "structured_group_weight": args.structured_group_weight,
                "group_feasible_selection": args.group_feasible_selection,
                "candidate_gate_used": False,
                "direct_target_used": False,
            }
        )
        if args.state_group_weighting == "group-balanced":
            payload["refinement"]["state_group_weights"] = {
                "nominal": 1.0,
                "structured": 1.0,
                "componentwise_total": 1.0,
                "composition_total": 1.0,
            }
            payload["refinement"]["state_group_weights"][
                "structured"
            ] = args.structured_group_weight
        if args.group_feasible_selection:
            payload["refinement"]["group_feasibility_limits"] = {
                "nominal": args.feasible_nominal_ratio,
                "structured": args.feasible_structured_ratio,
                "iid": args.feasible_iid_ratio,
                "composition": args.feasible_composition_ratio,
            }
    torch.save(payload, out_dir / "best_feedback_section5.pt")
    write_history(out_dir / "history.csv", history)
    write_history(
        out_dir / "fixed_validation_opt_gap.csv",
        [
            {
                key: row[key]
                for key in (
                    "phase",
                    "subphase",
                    "phase_step",
                    "optimizer_step",
                    "outer_step",
                    "fixed_nominal_lopt_der",
                    "fixed_nominal_singular_component",
                    "fixed_nominal_boundary_component",
                    "fixed_nominal_invalid_component",
                    "fixed_nominal_q_mean",
                )
            }
            for row in history
        ],
    )
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "best_validation_loss": best_loss,
                "best_selection_loss": best_loss,
                "best_outer_step": best_step,
                "best_selection_feasible": best_selection_feasible,
                "best_validation_group_lopt": best_validation_group_lopt,
                "initialization_checkpoint": str(checkpoint_path),
                "initial_validation_metrics_by_state": (
                    initial_metrics_by_state
                ),
                "selected_train_metrics_by_state": (
                    selected_train_metrics_by_state
                ),
                "selected_validation_metrics_by_state": (
                    selected_validation_metrics_by_state
                ),
                "history": history,
                "fixed_validation_metric": metric_metadata(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--reset-state-branch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "discard the incoming feedback-branch weights and restart its "
            "zero-output nested correction while preserving the time branch"
        ),
    )
    parser.add_argument(
        "--state-branch-init-seed",
        type=int,
        default=20260730,
        help="reproducible hidden-layer seed used with --reset-state-branch",
    )
    parser.add_argument("--option", choices=["cf", "der"], default="der")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--multiplier", type=int, default=4)
    parser.add_argument(
        "--trajectory-mode",
        choices=["zoh", "continuous-policy"],
        default="zoh",
        help=(
            "hold each left-endpoint action through an RK4 interval, or "
            "re-query u(N,t) at every RK4 stage and integrate the continuous "
            "PMP costate equation"
        ),
    )
    parser.add_argument(
        "--continuous-collocation",
        choices=["nodes", "nodes-midpoints"],
        default="nodes",
        help=(
            "in continuous-policy mode, enforce scalar conditions at left "
            "RK4 nodes only (the historical default), or jointly at left "
            "nodes and RK4 midpoints"
        ),
    )
    parser.add_argument(
        "--reference-multiplier",
        type=int,
        default=0,
        help=(
            "fine-grid multiplier used for the stored nominal reference; "
            "0 reuses --multiplier"
        ),
    )
    parser.add_argument(
        "--reference-method",
        choices=["zoh", "dop853"],
        default="zoh",
    )
    parser.add_argument("--reference-rtol", type=float, default=1.0e-10)
    parser.add_argument("--reference-atol", type=float, default=1.0e-12)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--radius", type=float, default=0.10)
    parser.add_argument(
        "--state-sampling",
        choices=["anchored-componentwise", "balanced-mixed"],
        default="anchored-componentwise",
        help=(
            "use the historical nominal/structured/iid states, or a balanced "
            "iid and fixed-total composition mixture"
        ),
    )
    parser.add_argument(
        "--structured-radius",
        type=float,
        default=0.10,
        help="radius of the fixed structured state in balanced-mixed mode",
    )
    parser.add_argument(
        "--mixed-composition-fraction",
        type=float,
        default=0.5,
        help=(
            "fraction of stochastic states assigned to zero-sum composition "
            "directions in balanced-mixed mode"
        ),
    )
    parser.add_argument(
        "--state-group-weighting",
        choices=["sample-mean", "group-balanced"],
        default="sample-mean",
        help=(
            "pool scalar residuals over all states uniformly, or give nominal, "
            "structured, iid, and composition groups equal total weight"
        ),
    )
    parser.add_argument(
        "--structured-group-weight",
        type=float,
        default=1.0,
        help=(
            "structured-group mass s in empirical-mean group weights "
            "(1,s,1,1)"
        ),
    )
    parser.add_argument(
        "--group-feasible-selection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "select feasible checkpoints by the mean relative iid/composition "
            "validation L_opt instead of the pooled scalar loss"
        ),
    )
    parser.add_argument("--feasible-nominal-ratio", type=float, default=1.25)
    parser.add_argument("--feasible-structured-ratio", type=float, default=2.0)
    parser.add_argument("--feasible-iid-ratio", type=float, default=0.8)
    parser.add_argument(
        "--feasible-composition-ratio",
        type=float,
        default=0.8,
    )
    parser.add_argument("--interval-start", type=float, default=1.5)
    parser.add_argument("--interval-end", type=float, default=8.0)
    parser.add_argument("--w0", type=float, default=1.0)
    parser.add_argument("--w1", type=float, default=1.0)
    parser.add_argument("--w2", type=float, default=1.0)
    parser.add_argument(
        "--residual-p",
        type=float,
        default=2.0,
        help=(
            "weighted p-norm used for each scalar residual, represented on "
            "a squared scale; 2 recovers the RMS loss"
        ),
    )
    parser.add_argument("--cf-weight", type=float, default=1.0)
    parser.add_argument("--cf-scalar-weight", type=float, default=0.0)
    parser.add_argument("--control-trust-weight", type=float, default=0.0)
    parser.add_argument(
        "--protected-control-trust-weight",
        type=float,
        default=0.0,
        help=(
            "penalize control drift only on the first protected initial "
            "states, leaving the remaining random states free to adapt"
        ),
    )
    parser.add_argument(
        "--protected-state-count",
        type=int,
        default=0,
        help=(
            "number of leading states protected by the checkpoint-control "
            "trust term (nominal then structured when nominal is included)"
        ),
    )
    parser.add_argument(
        "--selection-metric",
        choices=["combined", "scalar"],
        default="combined",
        help=(
            "select the checkpoint by the complete training loss or by the "
            "three scalar Hamiltonian residuals only"
        ),
    )
    parser.add_argument("--train-random-states", type=int, default=0)
    parser.add_argument("--validation-random-states", type=int, default=0)
    parser.add_argument("--train-seed", type=int, default=20260721)
    parser.add_argument("--validation-seed", type=int, default=20260719)
    parser.add_argument(
        "--exclude-nominal-from-training",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "train the scalar feedback correction on resistant/random states "
            "only; nominal remains in validation as a centering guard"
        ),
    )
    parser.add_argument(
        "--refresh-nominal-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--train-scope", choices=["last_layer", "all"], default="last_layer"
    )
    parser.add_argument("--outer-steps", type=int, default=6)
    parser.add_argument("--inner-iterations", type=int, default=3)
    parser.add_argument("--max-eval", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--history-size", type=int, default=30)
    parser.add_argument("--tolerance-grad", type=float, default=1.0e-12)
    parser.add_argument("--tolerance-change", type=float, default=1.0e-14)
    parser.add_argument("--report-scale-factor", type=float, default=400.0)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
