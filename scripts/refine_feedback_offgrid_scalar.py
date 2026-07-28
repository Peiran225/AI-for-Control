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
    denominator = (
        weighted_mask.sum() * controls.shape[0]
    ).clamp_min(torch.finfo(controls.dtype).eps)

    def mean_square(value: torch.Tensor) -> torch.Tensor:
        return (weighted_mask * value.square()).sum() / denominator

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
            weighted_mask * (value.abs() / scale).pow(float(residual_p))
        ).sum() / denominator
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
        "denominator": denominator,
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


def run(args: argparse.Namespace) -> None:
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
    cfg = fine_problem(base_cfg, args.multiplier)
    model.to(device=device, dtype=torch.float64)
    params = build_params(cfg, device, torch.float64)
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

    def evaluate(step: int) -> float:
        nonlocal best_loss, best_step, best_state, best_reference
        nonlocal initial_metrics_by_state
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
                )
            else:
                selection_pack = validation_pack
        if args.selection_metric == "scalar":
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
        metrics = physical_metrics(
            validation_pack, scale=args.report_scale_factor
        )
        row = {
            "outer_step": step,
            "train_loss": float(train_pack["loss"].cpu()),
            "validation_loss": float(validation_pack["loss"].cpu()),
            "selection_loss": float(selection_value.cpu()),
            **metrics,
        }
        history.append(row)
        if step == 0:
            initial_metrics_by_state = physical_metrics_by_sample(
                validation_pack,
                scale=args.report_scale_factor,
                names=validation_state_names,
            )
        if row["selection_loss"] < best_loss:
            best_loss = row["selection_loss"]
            best_step = step
            best_state = copy.deepcopy(model.state_dict())
            best_reference = model.nominal_reference.detach().cpu().clone()
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
            )
            pack["loss"].backward()
            return pack["loss"]

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
            "checkpoint_selection": args.selection_metric,
        },
    }
    torch.save(payload, out_dir / "best_feedback_section5.pt")
    write_history(out_dir / "history.csv", history)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "best_validation_loss": best_loss,
                "best_selection_loss": best_loss,
                "best_outer_step": best_step,
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
