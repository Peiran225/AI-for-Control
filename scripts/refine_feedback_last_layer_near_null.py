#!/usr/bin/env python3
"""Near-nullspace last-layer refinement for a locked feedback checkpoint.

This is an optional, conservative experiment.  It freezes the time policy and
the nonlinear feedback features, then restricts the final feedback-layer
weight update to a numerical near-nullspace of the feature differences traced
by one protected structured trajectory.  The remaining degrees of freedom are
trained only with the continuous-policy DER scalar PMP residual on independent
iid and fixed-total composition perturbations.

No physical objective, direct-control target, control-trust penalty, or
candidate gate is used.
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

import torch
import torch.nn as nn
from torch.nn.utils import parametrize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.feedback_continuous_policy_rk4 import pchip_midpoint_logits
from scripts.constrained_scalar_lm_step import (
    constrained_lm_step,
    der_residual_vector_from_pack,
    forward_jacobian_columns,
    nonlinear_feasible_backtracking,
)
from scripts.refine_feedback_offgrid_scalar import (
    _balanced_mixed_counts,
    fine_problem,
    fixed_support_dense_logits,
    physical_metrics,
    scalar_pack,
)
from scripts.train_feedback_section5 import dynamics
from scripts.evaluate_feedback_section5 import load_feedback_checkpoint
from train_paper_pmp_kkt import ProblemConfig, build_params


class NearNullWeightParametrization(nn.Module):
    """Represent ``weight = base_weight + theta @ basis.T``."""

    def __init__(
        self,
        base_weight: torch.Tensor,
        basis: torch.Tensor,
    ) -> None:
        super().__init__()
        if base_weight.ndim != 2 or base_weight.shape[0] != 1:
            raise ValueError("the final feedback weight must have shape (1, hidden)")
        if basis.ndim != 2 or basis.shape[0] != base_weight.shape[1]:
            raise ValueError("basis must have shape (hidden, nullity)")
        if basis.shape[1] < 1:
            raise ValueError("the numerical near-nullspace is empty")
        self.register_buffer("base_weight", base_weight.detach().clone())
        self.register_buffer("basis", basis.detach().clone())
        self.theta = nn.Parameter(
            base_weight.new_zeros((base_weight.shape[0], basis.shape[1]))
        )

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        # ``original`` is kept only for PyTorch parametrization compatibility.
        del original
        return self.base_weight + self.theta @ self.basis.T


def numerical_near_nullspace(
    matrix: torch.Tensor,
    relative_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return right singular vectors below a relative singular-value cutoff."""

    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError("feature matrix must have shape (queries, hidden)")
    if not math.isfinite(relative_tolerance) or not (
        0.0 < relative_tolerance < 1.0
    ):
        raise ValueError("relative tolerance must lie strictly between 0 and 1")
    _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=True)
    if singular_values.numel() == 0 or not bool(
        torch.isfinite(singular_values).all()
    ):
        raise ValueError("feature-matrix SVD failed")
    cutoff = singular_values[0] * float(relative_tolerance)
    rank = int((singular_values > cutoff).sum().item())
    basis = vh[rank:].T.contiguous()
    return basis, singular_values, rank


def parse_structured_radii(
    values: str,
    legacy_radius: float,
) -> tuple[float, ...]:
    """Parse optional comma-separated radii with legacy single-value fallback."""

    if values.strip():
        radii = tuple(
            float(item.strip())
            for item in values.split(",")
            if item.strip()
        )
        if not radii:
            raise ValueError("structured radii must not be empty")
    else:
        radii = (float(legacy_radius),)
    if any(
        not math.isfinite(radius) or not 0.0 <= radius <= 1.0
        for radius in radii
    ):
        raise ValueError("structured radii must be finite and lie in [0,1]")
    if len(set(radii)) != len(radii):
        raise ValueError("structured radii must be unique")
    return radii


def radius_tag(radius: float) -> str:
    return f"{radius:.6g}".replace("-", "m").replace(".", "p")


def hidden_feature_differences(
    model: nn.Module,
    normalized_time: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    """Return hidden features relative to the centered nominal state."""

    if not bool(model.center_state_correction):
        raise ValueError(
            "near-null refinement requires center_state_correction=True"
        )
    hidden = model.state_branch[:-1]
    reference = model.nominal_state_at(normalized_time)
    return hidden(model.state_features(normalized_time, state)) - hidden(
        model.state_features(normalized_time, reference)
    )


def continuous_rk4_stage_feature_matrix(
    model: nn.Module,
    initial_state: torch.Tensor,
    cfg: ProblemConfig,
    normalized_time: torch.Tensor,
    raw_logits: torch.Tensor,
    midpoint_raw_logits: torch.Tensor,
    params: dict[str, torch.Tensor],
    *,
    interval_start: float,
    interval_end: float,
) -> torch.Tensor:
    """Collect feature differences at every RK4 stage in a time interval."""

    if not 0.0 <= interval_start < interval_end <= cfg.T:
        raise ValueError("constraint interval must lie inside [0,T]")
    if initial_state.ndim != 2 or initial_state.shape[1] != cfg.m:
        raise ValueError("initial states must have shape (batch, m)")
    step = cfg.T / cfg.n
    batch = initial_state.shape[0]
    state = initial_state
    rows: list[torch.Tensor] = []

    def action(
        raw: torch.Tensor,
        query_time: torch.Tensor,
        query_state: torch.Tensor,
    ) -> torch.Tensor:
        return model.interval_action(
            raw,
            query_time,
            query_state,
            state_mode="feedback",
        )

    def record(query_time: torch.Tensor, query_state: torch.Tensor) -> None:
        physical_time = float(query_time[0].detach().cpu()) * cfg.T
        if interval_start <= physical_time <= interval_end:
            rows.append(
                hidden_feature_differences(
                    model,
                    query_time,
                    query_state,
                )
            )

    for index in range(cfg.n):
        left_time = normalized_time[index].expand(batch)
        middle_time = (
            0.5 * (normalized_time[index] + normalized_time[index + 1])
        ).expand(batch)
        right_time = normalized_time[index + 1].expand(batch)

        record(left_time, state)
        control1 = action(raw_logits[index], left_time, state)
        slope1 = dynamics(state, control1, params)

        stage2 = state + 0.5 * step * slope1
        record(middle_time, stage2)
        control2 = action(midpoint_raw_logits[index], middle_time, stage2)
        slope2 = dynamics(stage2, control2, params)

        stage3 = state + 0.5 * step * slope2
        record(middle_time, stage3)
        control3 = action(midpoint_raw_logits[index], middle_time, stage3)
        slope3 = dynamics(stage3, control3, params)

        stage4 = state + step * slope3
        record(right_time, stage4)
        control4 = action(raw_logits[index + 1], right_time, stage4)
        slope4 = dynamics(stage4, control4, params)

        state = state + (step / 6.0) * (
            slope1 + 2.0 * slope2 + 2.0 * slope3 + slope4
        )
        if not bool(torch.isfinite(state).all()):
            raise FloatingPointError("structured trajectory became nonfinite")
    if not rows:
        raise ValueError("constraint interval contains no RK4 stage queries")
    return torch.stack(rows, dim=1).reshape(-1, rows[0].shape[-1])


def iid_and_composition_states(
    count: int,
    seed: int,
    radius: float,
    cfg: ProblemConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """Return stochastic states and equal-total-mass group weights."""

    componentwise_count, composition_count = _balanced_mixed_counts(count, 0.5)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    componentwise_direction = 2.0 * torch.rand(
        componentwise_count,
        cfg.m,
        generator=generator,
        dtype=torch.float64,
    ) - 1.0
    componentwise = cfg.n0 * (1.0 + radius * componentwise_direction)

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
    composition_direction = composition_direction / (
        composition_direction.abs()
        .amax(dim=-1, keepdim=True)
        .clamp_min(torch.finfo(torch.float64).eps)
    )
    amplitude = torch.rand(
        composition_count,
        1,
        generator=generator,
        dtype=torch.float64,
    )
    composition = cfg.n0 * (
        1.0 + radius * amplitude * composition_direction
    )
    states = torch.cat((componentwise, composition), dim=0).to(
        device=device,
        dtype=dtype,
    )
    weights = torch.cat(
        (
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
    return states, weights, (componentwise_count, composition_count)


def install_near_null_parametrization(
    final_layer: nn.Linear,
    basis: torch.Tensor,
) -> NearNullWeightParametrization:
    """Install a differentiable near-null update on a standard final Linear."""

    module = NearNullWeightParametrization(final_layer.weight, basis)
    parametrize.register_parametrization(final_layer, "weight", module)
    final_layer.parametrizations.weight.original.requires_grad_(False)
    if final_layer.bias is not None:
        final_layer.bias.requires_grad_(False)
    return module


def materialize_near_null_weight(final_layer: nn.Linear) -> None:
    """Remove the parametrization while retaining its current standard weight."""

    parametrize.remove_parametrizations(
        final_layer,
        "weight",
        leave_parametrized=True,
    )


def _scalar_loss(
    model: nn.Module,
    states: torch.Tensor,
    state_weights: torch.Tensor,
    cfg: ProblemConfig,
    dense_time: torch.Tensor,
    dense_raw: torch.Tensor,
    midpoint_raw: torch.Tensor,
    params: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, Any]]:
    pack = scalar_pack(
        model,
        states,
        cfg,
        dense_time,
        dense_raw,
        params,
        trajectory_mode="continuous-policy",
        midpoint_raw_logits=midpoint_raw,
        continuous_collocation="nodes",
        option="der",
        interval_start=args.interval_start,
        interval_end=args.interval_end,
        w0=1.0,
        w1=1.0,
        w2=4.0,
        residual_p=2.0,
        cf_weight=0.0,
        cf_scalar_weight=0.0,
        control_trust_weight=0.0,
        anchor_controls=None,
        state_weights=state_weights,
    )
    return pack["loss"], pack


def scalar_lopt_by_sample(pack: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return the unpooled DER scalar loss for every initial state."""

    values = pack["quantities"]["psi"]
    weighted_mask = pack["weighted_mask"].expand_as(values)
    reduction_dims = tuple(range(1, weighted_mask.ndim))
    denominator = weighted_mask.sum(dim=reduction_dims).clamp_min(
        torch.finfo(weighted_mask.dtype).eps
    )
    return sum(
        weight
        * (
            weighted_mask * pack["quantities"][name].square()
        ).sum(dim=reduction_dims)
        / denominator
        for weight, name in (
            (1.0, "psi"),
            (1.0, "dot_psi"),
            (4.0, "ddot_psi"),
        )
    )


class ScalarPMPFunctionalModule(nn.Module):
    """Expose the near-null coordinate through ``functional_call``.

    The module lets the constrained-LM branch evaluate residuals at a trial
    coordinate without mutating the live checkpoint.  All returned quantities
    come from the same continuous-policy scalar PMP pack used by LBFGS.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: ProblemConfig,
        dense_time: torch.Tensor,
        dense_raw: torch.Tensor,
        midpoint_raw: torch.Tensor,
        params: dict[str, torch.Tensor],
        args: argparse.Namespace,
    ) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.register_buffer("dense_time", dense_time)
        self.register_buffer("dense_raw", dense_raw)
        self.register_buffer("midpoint_raw", midpoint_raw)
        self.params = params
        self.args = args

    def forward(
        self,
        states: torch.Tensor,
        state_weights: torch.Tensor,
        output: str,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        _, pack = _scalar_loss(
            self.model,
            states,
            state_weights,
            self.cfg,
            self.dense_time,
            self.dense_raw,
            self.midpoint_raw,
            self.params,
            self.args,
        )
        if output == "residual":
            return der_residual_vector_from_pack(
                pack,
                weights=(1.0, 1.0, 4.0),
            )
        if output == "lopt":
            return scalar_lopt_by_sample(pack)
        if output == "protected":
            return scalar_lopt_by_sample(pack), pack["sample_controls"]
        raise ValueError(f"unknown scalar PMP functional output: {output}")


def protected_metrics_pass(
    structured_prelogit_drift: torch.Tensor,
    protected_control_drift: torch.Tensor,
    protected_lopt: torch.Tensor,
    protected_lopt_limit: torch.Tensor,
    *,
    max_nominal_control_drift: float,
    max_structured_control_drift: float,
    max_structured_prelogit_drift: float,
) -> bool:
    """Apply simultaneous nominal and per-structured-trajectory hard gates."""

    if protected_control_drift.ndim != 1 or protected_control_drift.numel() < 2:
        raise ValueError("protected control drift requires nominal and structured rows")
    if protected_lopt.shape != protected_control_drift.shape:
        raise ValueError("protected Lopt and control drift must have matching rows")
    if protected_lopt_limit.shape != protected_lopt.shape:
        raise ValueError("protected Lopt limits must match protected Lopt")
    return (
        float(structured_prelogit_drift)
        <= max_structured_prelogit_drift
        and float(protected_control_drift[0])
        <= max_nominal_control_drift
        and bool(
            (
                protected_control_drift[1:]
                <= max_structured_control_drift
            ).all()
        )
        and bool((protected_lopt <= protected_lopt_limit).all())
    )


def run(args: argparse.Namespace) -> None:
    if args.train_seed == args.validation_seed:
        raise ValueError("training and validation seeds must differ")
    if not 0.0 <= args.radius <= 1.0:
        raise ValueError("random-state radius must lie in [0,1]")
    if args.optimizer == "constrained-lm":
        if not (
            args.lm_initial_damping > 0.0
            and args.lm_minimum_damping > 0.0
            and args.lm_maximum_damping >= args.lm_initial_damping
            and 0.0 < args.lm_damping_decrease < 1.0
            and args.lm_damping_increase > 1.0
            and args.lm_maximum_attempts > 0
            and args.lm_backtracking_steps > 0
            and 0.0 < args.lm_backtracking_factor < 1.0
            and args.lm_maximum_step_norm > 0.0
            and args.lm_acceptance_tolerance >= 0.0
        ):
            raise ValueError("invalid constrained-LM configuration")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint_path = args.checkpoint.expanduser().resolve()
    source_checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model, base_cfg, _ = load_feedback_checkpoint(checkpoint_path)
    if not bool(model.center_state_correction):
        raise ValueError(
            "the incoming checkpoint must use centered state correction"
        )
    cfg = fine_problem(base_cfg, args.multiplier)
    model.to(device=device, dtype=torch.float64).eval()
    params = build_params(cfg, device, torch.float64)
    model.set_feature_vectors(params["r"], params["phi"])
    dense_time, dense_raw = fixed_support_dense_logits(
        model,
        base_cfg,
        args.multiplier,
        query_batch_size=args.query_batch_size,
    )
    midpoint_raw = pchip_midpoint_logits(dense_time, dense_raw)

    structured_radii = parse_structured_radii(
        args.structured_radii,
        args.structured_radius,
    )
    structured_direction = torch.linspace(
        -1.0,
        1.0,
        base_cfg.m,
        device=device,
        dtype=torch.float64,
    )
    structured_initials = torch.stack(
        [
            base_cfg.n0 * (1.0 + radius * structured_direction)
            for radius in structured_radii
        ]
    )
    structured_matrices: list[torch.Tensor] = []
    with torch.no_grad():
        for structured_initial in structured_initials:
            structured_matrices.append(
                continuous_rk4_stage_feature_matrix(
                    model,
                    structured_initial.unsqueeze(0),
                    cfg,
                    dense_time,
                    dense_raw,
                    midpoint_raw,
                    params,
                    interval_start=args.constraint_interval_start,
                    interval_end=args.constraint_interval_end,
                )
            )
    structured_matrix = torch.cat(structured_matrices, dim=0)
    basis, singular_values, rank = numerical_near_nullspace(
        structured_matrix,
        args.svd_rtol,
    )
    nullity = int(basis.shape[1])
    if nullity < args.minimum_nullity:
        raise ValueError(
            f"near-nullity {nullity} is below required minimum "
            f"{args.minimum_nullity}; relaxing the tolerance changes the "
            "preservation guarantee and must be explicit"
        )

    train_states, train_weights, train_counts = iid_and_composition_states(
        args.train_random_states,
        args.train_seed,
        args.radius,
        cfg,
        device,
        torch.float64,
    )
    validation_states, validation_weights, validation_counts = (
        iid_and_composition_states(
            args.validation_random_states,
            args.validation_seed,
            args.radius,
            cfg,
            device,
            torch.float64,
        )
    )
    protected_states = torch.stack(
        [
            torch.full(
                (cfg.m,),
                cfg.n0,
                device=device,
                dtype=torch.float64,
            ),
            *structured_initials,
        ]
    )
    protected_weights = torch.ones(
        1 + len(structured_radii),
        device=device,
        dtype=torch.float64,
    )
    with torch.no_grad():
        _, protected_initial_pack = _scalar_loss(
            model,
            protected_states,
            protected_weights,
            cfg,
            dense_time,
            dense_raw,
            midpoint_raw,
            params,
            args,
        )
        protected_initial_controls = (
            protected_initial_pack["sample_controls"].detach().clone()
        )
        protected_initial_lopt = scalar_lopt_by_sample(
            protected_initial_pack
        ).detach()
        protected_lopt_limit = (
            protected_initial_lopt
            * (1.0 + args.max_protected_lopt_relative_increase)
            + args.max_protected_lopt_absolute_increase
        )
    with torch.no_grad():
        random_matrix = continuous_rk4_stage_feature_matrix(
            model,
            train_states,
            cfg,
            dense_time,
            dense_raw,
            midpoint_raw,
            params,
            interval_start=args.interval_start,
            interval_end=args.interval_end,
        )
        projected_random_energy = float(
            (random_matrix @ basis).norm()
            / random_matrix.norm().clamp_min(torch.finfo(torch.float64).eps)
        )
        basis_leakage_max = float(
            (structured_matrix @ basis).abs().max()
        )
        basis_leakage_relative = float(
            (structured_matrix @ basis).norm()
            / structured_matrix.norm().clamp_min(
                torch.finfo(torch.float64).eps
            )
        )

    incoming_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    final_layer = model.state_branch[-1]
    if not isinstance(final_layer, nn.Linear):
        raise TypeError("the final feedback module must be nn.Linear")
    base_weight = final_layer.weight.detach().clone()
    base_bias = (
        None
        if final_layer.bias is None
        else final_layer.bias.detach().clone()
    )
    near_null = install_near_null_parametrization(final_layer, basis)
    optimizer: torch.optim.LBFGS | None = None
    if args.optimizer == "lbfgs":
        optimizer = torch.optim.LBFGS(
            [near_null.theta],
            lr=args.lr,
            max_iter=args.inner_iterations,
            max_eval=args.max_eval,
            tolerance_grad=args.tolerance_grad,
            tolerance_change=args.tolerance_change,
            history_size=args.history_size,
            line_search_fn="strong_wolfe",
        )
    lm_functional: ScalarPMPFunctionalModule | None = None
    lm_theta_name: str | None = None
    if args.optimizer == "constrained-lm":
        lm_functional = ScalarPMPFunctionalModule(
            model,
            cfg,
            dense_time,
            dense_raw,
            midpoint_raw,
            params,
            args,
        )
        lm_theta_name = next(
            name
            for name, parameter in lm_functional.named_parameters()
            if parameter is near_null.theta
        )

    history: list[dict[str, float | int]] = []
    best_loss = math.inf
    best_step = 0
    best_theta = near_null.theta.detach().clone()

    def evaluate(
        step: int,
        optimizer_metadata: dict[str, float | int | str] | None = None,
    ) -> bool:
        nonlocal best_loss, best_step, best_theta
        with torch.no_grad():
            train_loss, train_pack = _scalar_loss(
                model,
                train_states,
                train_weights,
                cfg,
                dense_time,
                dense_raw,
                midpoint_raw,
                params,
                args,
            )
            validation_loss, validation_pack = _scalar_loss(
                model,
                validation_states,
                validation_weights,
                cfg,
                dense_time,
                dense_raw,
                midpoint_raw,
                params,
                args,
            )
            _, protected_pack = _scalar_loss(
                model,
                protected_states,
                protected_weights,
                cfg,
                dense_time,
                dense_raw,
                midpoint_raw,
                params,
                args,
            )
            protected_control_drift = (
                protected_pack["sample_controls"]
                - protected_initial_controls
            ).abs()
            protected_control_drift = protected_control_drift.amax(
                dim=tuple(range(1, protected_control_drift.ndim))
            )
            protected_lopt = scalar_lopt_by_sample(protected_pack)
            delta_weight = final_layer.weight - base_weight
            structured_prelogit_drift = (
                structured_matrix @ delta_weight.T
            ).abs().max()
            row: dict[str, float | int] = {
                "outer_step": step,
                "train_loss": float(train_loss),
                "validation_loss": float(validation_loss),
                "structured_prelogit_drift_max": float(
                    structured_prelogit_drift
                ),
                "nominal_control_drift_max": float(
                    protected_control_drift[0]
                ),
                "structured_control_drift_max": float(
                    protected_control_drift[1:].max()
                ),
                "nominal_scalar_lopt": float(protected_lopt[0]),
                "structured_scalar_lopt": float(
                    protected_lopt[1:].max()
                ),
                "nominal_scalar_lopt_limit": float(
                    protected_lopt_limit[0]
                ),
                "structured_scalar_lopt_limit": float(
                    protected_lopt_limit[1:].max()
                ),
                "theta_l2": float(near_null.theta.norm()),
            }
            for index, radius in enumerate(structured_radii, start=1):
                tag = radius_tag(radius)
                row[f"structured_r{tag}_control_drift_max"] = float(
                    protected_control_drift[index]
                )
                row[f"structured_r{tag}_scalar_lopt"] = float(
                    protected_lopt[index]
                )
                row[f"structured_r{tag}_scalar_lopt_limit"] = float(
                    protected_lopt_limit[index]
                )
            row.update(
                {
                    f"validation_{key}": value
                    for key, value in physical_metrics(
                        validation_pack,
                        scale=args.report_scale_factor,
                    ).items()
                }
            )
            if optimizer_metadata is not None:
                row.update(optimizer_metadata)
                if step == 0 and row.get("optimizer") == "constrained-lm":
                    row["lm_train_objective_before"] = row["train_loss"]
                    row["lm_train_objective_after"] = row["train_loss"]
            history.append(row)
            accepted = protected_metrics_pass(
                structured_prelogit_drift,
                protected_control_drift,
                protected_lopt,
                protected_lopt_limit,
                max_nominal_control_drift=args.max_nominal_control_drift,
                max_structured_control_drift=(
                    args.max_structured_control_drift
                ),
                max_structured_prelogit_drift=(
                    args.max_structured_prelogit_drift
                ),
            )
            row["preservation_gate_pass"] = int(accepted)
            if accepted and float(validation_loss) < best_loss:
                best_loss = float(validation_loss)
                best_step = step
                best_theta = near_null.theta.detach().clone()
        print(json.dumps(history[-1], sort_keys=True), flush=True)
        return accepted

    initial_optimizer_metadata = (
        None
        if args.optimizer == "lbfgs"
        else {
            "optimizer": "constrained-lm",
            "lm_step_accepted": 0,
            "lm_attempts": 0,
            "lm_backtracking_step": 0,
            "lm_step_scale": 0.0,
            "lm_damping": args.lm_initial_damping,
            "lm_active_constraints": "",
            "lm_quadratic_value": 0.0,
            "lm_train_objective_before": 0.0,
            "lm_train_objective_after": 0.0,
        }
    )
    if not evaluate(0, initial_optimizer_metadata):
        raise RuntimeError("the incoming checkpoint fails its own preservation gate")
    if args.optimizer == "lbfgs":
        if optimizer is None:
            raise RuntimeError("LBFGS optimizer was not initialized")
        for outer_step in range(1, args.outer_steps + 1):

            def closure() -> torch.Tensor:
                optimizer.zero_grad(set_to_none=True)
                loss, _ = _scalar_loss(
                    model,
                    train_states,
                    train_weights,
                    cfg,
                    dense_time,
                    dense_raw,
                    midpoint_raw,
                    params,
                    args,
                )
                loss.backward()
                return loss

            optimizer.step(closure)
            if not evaluate(outer_step):
                # Treat the nominal/structured limits as hard constraints
                # rather than merely checkpoint-selection criteria.
                with torch.no_grad():
                    near_null.theta.copy_(best_theta)
                optimizer.state.clear()
    else:
        if lm_functional is None or lm_theta_name is None:
            raise RuntimeError("constrained-LM functional model is missing")

        def functional_output(
            flat_theta: torch.Tensor,
            states: torch.Tensor,
            weights: torch.Tensor,
            output: str,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            return torch.func.functional_call(
                lm_functional,
                {
                    lm_theta_name: flat_theta.reshape_as(
                        near_null.theta
                    )
                },
                (states, weights, output),
            )

        def train_residual(flat_theta: torch.Tensor) -> torch.Tensor:
            value = functional_output(
                flat_theta,
                train_states,
                train_weights,
                "residual",
            )
            if not isinstance(value, torch.Tensor):
                raise TypeError("scalar residual output must be a tensor")
            return value

        def protected_losses(flat_theta: torch.Tensor) -> torch.Tensor:
            value = functional_output(
                flat_theta,
                protected_states,
                protected_weights,
                "lopt",
            )
            if not isinstance(value, torch.Tensor):
                raise TypeError("protected Lopt output must be a tensor")
            return value

        damping = args.lm_initial_damping
        current_theta = near_null.theta.detach().reshape(-1).clone()
        for outer_step in range(1, args.outer_steps + 1):
            theta_for_jacobian = (
                current_theta.detach().requires_grad_(True)
            )
            current_residual, jacobian = forward_jacobian_columns(
                train_residual,
                theta_for_jacobian,
            )
            current_residual = current_residual.detach()
            jacobian = jacobian.detach()
            current_objective = float(
                current_residual.square().sum().detach()
            )
            current_protected = protected_losses(theta_for_jacobian)
            protected_jacobian = torch.autograd.functional.jacobian(
                protected_losses,
                theta_for_jacobian,
                vectorize=True,
                strategy="reverse-mode",
            )
            current_protected = current_protected.detach()
            protected_jacobian = protected_jacobian.detach()
            budgets = protected_lopt_limit - current_protected
            budget_tolerance = max(
                args.lm_acceptance_tolerance,
                32.0 * torch.finfo(torch.float64).eps,
            )
            if float(budgets.min()) < -budget_tolerance:
                raise RuntimeError(
                    "the current constrained-LM iterate violates a "
                    "protected scalar-Lopt limit"
                )
            budgets = budgets.clamp_min(0.0)

            accepted = False
            accepted_scale = 0.0
            accepted_backtracking = -1
            attempts_used = 0
            active_constraints: tuple[int, ...] = ()
            quadratic_value = 0.0
            trial_objective = current_objective
            for attempt in range(1, args.lm_maximum_attempts + 1):
                attempts_used = attempt
                step_result = constrained_lm_step(
                    current_residual,
                    jacobian,
                    protected_jacobian,
                    budgets,
                    damping=damping,
                    maximum_step_norm=args.lm_maximum_step_norm,
                )
                active_constraints = (
                    step_result.active_constraints
                )
                quadratic_value = step_result.quadratic_value
                if (
                    not step_result.linearized_feasible
                    or step_result.step_norm
                    <= torch.finfo(torch.float64).eps
                ):
                    damping = min(
                        args.lm_maximum_damping,
                        damping * args.lm_damping_increase,
                    )
                    continue

                def evaluate_trial(
                    candidate_theta: torch.Tensor,
                ) -> tuple[float, bool]:
                    with torch.no_grad():
                        candidate_residual = train_residual(
                            candidate_theta
                        )
                        candidate_objective = float(
                            candidate_residual.square().sum()
                        )
                        protected_output = functional_output(
                            candidate_theta,
                            protected_states,
                            protected_weights,
                            "protected",
                        )
                        if not isinstance(protected_output, tuple):
                            raise TypeError(
                                "protected output must be a tensor tuple"
                            )
                        candidate_protected, candidate_controls = (
                            protected_output
                        )
                        protected_control_drift = (
                            candidate_controls
                            - protected_initial_controls
                        ).abs()
                        protected_control_drift = (
                            protected_control_drift.amax(
                                dim=tuple(
                                    range(
                                        1,
                                        protected_control_drift.ndim,
                                    )
                                )
                            )
                        )
                        candidate_delta_weight = (
                            candidate_theta.reshape_as(
                                near_null.theta
                            )
                            @ basis.T
                        )
                        structured_prelogit_drift = (
                            structured_matrix
                            @ candidate_delta_weight.T
                        ).abs().max()
                    gate_pass = protected_metrics_pass(
                        structured_prelogit_drift,
                        protected_control_drift,
                        candidate_protected,
                        protected_lopt_limit,
                        max_nominal_control_drift=(
                            args.max_nominal_control_drift
                        ),
                        max_structured_control_drift=(
                            args.max_structured_control_drift
                        ),
                        max_structured_prelogit_drift=(
                            args.max_structured_prelogit_drift
                        ),
                    )
                    return candidate_objective, gate_pass

                backtracking = nonlinear_feasible_backtracking(
                    current_theta,
                    step_result.step,
                    current_objective,
                    evaluate_trial,
                    maximum_steps=args.lm_backtracking_steps,
                    factor=args.lm_backtracking_factor,
                    acceptance_tolerance=args.lm_acceptance_tolerance,
                )
                if backtracking.accepted:
                    accepted = True
                    accepted_scale = backtracking.scale
                    accepted_backtracking = (
                        backtracking.backtracking_step
                    )
                    current_theta = backtracking.point
                    trial_objective = backtracking.objective
                    with torch.no_grad():
                        near_null.theta.copy_(
                            current_theta.reshape_as(
                                near_null.theta
                            )
                        )
                    damping = max(
                        args.lm_minimum_damping,
                        damping * args.lm_damping_decrease,
                    )
                    break
                damping = min(
                    args.lm_maximum_damping,
                    damping * args.lm_damping_increase,
                )

            metadata: dict[str, float | int | str] = {
                "optimizer": "constrained-lm",
                "lm_step_accepted": int(accepted),
                "lm_attempts": attempts_used,
                "lm_backtracking_step": accepted_backtracking,
                "lm_step_scale": accepted_scale,
                "lm_damping": damping,
                "lm_active_constraints": ",".join(
                    str(index) for index in active_constraints
                ),
                "lm_quadratic_value": quadratic_value,
                "lm_train_objective_before": current_objective,
                "lm_train_objective_after": trial_objective,
            }
            if not evaluate(outer_step, metadata):
                raise RuntimeError(
                    "nonlinear constrained-LM acceptance disagrees with "
                    "the common preservation gate"
                )

    with torch.no_grad():
        near_null.theta.copy_(best_theta)
    materialize_near_null_weight(final_layer)
    if base_bias is not None:
        torch.testing.assert_close(final_layer.bias, base_bias)
    final_state = model.state_dict()
    changed_keys = [
        key
        for key, value in final_state.items()
        if not torch.equal(value.detach().cpu(), incoming_state[key])
    ]
    expected_weight_key = next(
        key
        for key in incoming_state
        if key.startswith("state_branch.")
        and key.endswith(".weight")
        and incoming_state[key].shape == base_weight.shape
    )
    if any(key != expected_weight_key for key in changed_keys):
        raise RuntimeError(
            "near-null refinement changed parameters outside the final weight: "
            f"{changed_keys}"
        )
    final_delta = final_layer.weight.detach() - base_weight
    final_structured_drift = float(
        (structured_matrix @ final_delta.T).abs().max()
    )
    if final_structured_drift > args.max_structured_prelogit_drift:
        raise RuntimeError(
            "selected near-null update violates the structured drift gate"
        )
    with torch.no_grad():
        _, final_protected_pack = _scalar_loss(
            model,
            protected_states,
            protected_weights,
            cfg,
            dense_time,
            dense_raw,
            midpoint_raw,
            params,
            args,
        )
        final_protected_control_drift = (
            final_protected_pack["sample_controls"]
            - protected_initial_controls
        ).abs()
        final_protected_control_drift = final_protected_control_drift.amax(
            dim=tuple(range(1, final_protected_control_drift.ndim))
        )
        final_protected_lopt = scalar_lopt_by_sample(
            final_protected_pack
        )
        final_protected_lopt_limit = (
            protected_initial_lopt
            * (1.0 + args.max_protected_lopt_relative_increase)
            + args.max_protected_lopt_absolute_increase
        )
    if not protected_metrics_pass(
        torch.as_tensor(
            final_structured_drift,
            device=device,
            dtype=torch.float64,
        ),
        final_protected_control_drift,
        final_protected_lopt,
        final_protected_lopt_limit,
        max_nominal_control_drift=args.max_nominal_control_drift,
        max_structured_control_drift=args.max_structured_control_drift,
        max_structured_prelogit_drift=(
            args.max_structured_prelogit_drift
        ),
    ):
        raise RuntimeError("selected checkpoint violates a preservation gate")

    output_args = dict(source_checkpoint["args"])
    output_args.update(
        {
            "near_null_last_layer_refinement": True,
            "near_null_svd_rtol": args.svd_rtol,
            "near_null_optimizer": args.optimizer,
        }
    )
    payload = copy.deepcopy(source_checkpoint)
    payload.update(
        {
            "model_state": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "args": output_args,
            "initialization_checkpoint": str(checkpoint_path),
            "best_validation_loss": best_loss,
            "best_selection_loss": best_loss,
            "best_epoch": best_step,
            "selection_metric": "independent_validation_scalar_Lopt",
            "refinement": {
                "optimizer": (
                    "LBFGS"
                    if args.optimizer == "lbfgs"
                    else "constrained-LM"
                ),
                "train_scope": "near-nullspace final state-layer weight",
                "physical_objective_used": False,
                "direct_supervision_used": False,
                "candidate_gate_used": False,
                "trajectory_mode": "continuous-policy",
                "continuous_collocation": "nodes",
                "state_requery": "every RK4 stage",
                "scalar_residual_weights": [1.0, 1.0, 4.0],
                "structured_radius": structured_radii[0],
                "structured_radii": list(structured_radii),
                "constraint_interval": [
                    args.constraint_interval_start,
                    args.constraint_interval_end,
                ],
                "svd_relative_tolerance": args.svd_rtol,
                "feature_matrix_shape": list(structured_matrix.shape),
                "feature_rows_by_structured_radius": {
                    str(radius): int(matrix.shape[0])
                    for radius, matrix in zip(
                        structured_radii,
                        structured_matrices,
                    )
                },
                "feature_matrix_rank": rank,
                "feature_matrix_nullity": nullity,
                "singular_value_max": float(singular_values[0]),
                "singular_value_min": float(singular_values[-1]),
                "basis_leakage_max": basis_leakage_max,
                "basis_leakage_relative_frobenius": (
                    basis_leakage_relative
                ),
                "random_feature_energy_retained": projected_random_energy,
                "final_structured_prelogit_drift_max": (
                    final_structured_drift
                ),
                "protected_initial_scalar_lopt": [
                    float(value) for value in protected_initial_lopt
                ],
                "protected_final_scalar_lopt": [
                    float(value) for value in final_protected_lopt
                ],
                "protected_final_control_drift_max": [
                    float(value)
                    for value in final_protected_control_drift
                ],
                "protected_state_order": [
                    "nominal",
                    *[
                        f"structured_r={radius:g}"
                        for radius in structured_radii
                    ],
                ],
                "preservation_gates": {
                    "max_nominal_control_drift": (
                        args.max_nominal_control_drift
                    ),
                    "max_structured_control_drift": (
                        args.max_structured_control_drift
                    ),
                    "max_structured_prelogit_drift": (
                        args.max_structured_prelogit_drift
                    ),
                    "max_scalar_lopt_relative_increase": (
                        args.max_protected_lopt_relative_increase
                    ),
                    "max_scalar_lopt_absolute_increase": (
                        args.max_protected_lopt_absolute_increase
                    ),
                },
                "train_state_groups": {
                    "iid_componentwise": train_counts[0],
                    "zero_sum_composition": train_counts[1],
                },
                "validation_state_groups": {
                    "iid_componentwise": validation_counts[0],
                    "zero_sum_composition": validation_counts[1],
                },
                "train_seed": args.train_seed,
                "validation_seed": args.validation_seed,
                "constrained_lm": (
                    None
                    if args.optimizer == "lbfgs"
                    else {
                        "initial_damping": args.lm_initial_damping,
                        "minimum_damping": args.lm_minimum_damping,
                        "maximum_damping": args.lm_maximum_damping,
                        "damping_decrease": args.lm_damping_decrease,
                        "damping_increase": args.lm_damping_increase,
                        "maximum_attempts": args.lm_maximum_attempts,
                        "backtracking_steps": (
                            args.lm_backtracking_steps
                        ),
                        "backtracking_factor": (
                            args.lm_backtracking_factor
                        ),
                        "maximum_step_norm": (
                            args.lm_maximum_step_norm
                        ),
                        "acceptance_tolerance": (
                            args.lm_acceptance_tolerance
                        ),
                        "linearized_constraints": (
                            "per-protected-state continuous scalar Lopt"
                        ),
                        "nonlinear_acceptance": (
                            "strict decrease in random-state scalar Lopt "
                            "and all common preservation gates"
                        ),
                    }
                ),
            },
        }
    )
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_dir / "best_feedback_section5.pt")
    with (out_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "best_validation_loss": best_loss,
                "best_outer_step": best_step,
                "optimizer": args.optimizer,
                "feature_matrix_shape": list(structured_matrix.shape),
                "structured_radii": list(structured_radii),
                "feature_rows_by_structured_radius": {
                    str(radius): int(matrix.shape[0])
                    for radius, matrix in zip(
                        structured_radii,
                        structured_matrices,
                    )
                },
                "feature_matrix_rank": rank,
                "feature_matrix_nullity": nullity,
                "singular_values": [
                    float(value) for value in singular_values.detach().cpu()
                ],
                "basis_leakage_max": basis_leakage_max,
                "basis_leakage_relative_frobenius": basis_leakage_relative,
                "random_feature_energy_retained": projected_random_energy,
                "final_structured_prelogit_drift_max": (
                    final_structured_drift
                ),
                "protected_initial_scalar_lopt": [
                    float(value) for value in protected_initial_lopt
                ],
                "protected_final_scalar_lopt": [
                    float(value) for value in final_protected_lopt
                ],
                "protected_final_control_drift_max": [
                    float(value)
                    for value in final_protected_control_drift
                ],
                "protected_state_order": [
                    "nominal",
                    *[
                        f"structured_r={radius:g}"
                        for radius in structured_radii
                    ],
                ],
                "changed_parameter_keys": changed_keys,
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
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--multiplier", type=int, default=4)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--radius", type=float, default=0.10)
    parser.add_argument("--structured-radius", type=float, default=0.10)
    parser.add_argument(
        "--structured-radii",
        default="",
        help=(
            "optional comma-separated protected radii; when omitted, "
            "--structured-radius retains the historical single-radius behavior"
        ),
    )
    parser.add_argument("--interval-start", type=float, default=1.5)
    parser.add_argument("--interval-end", type=float, default=8.0)
    parser.add_argument(
        "--constraint-interval-start",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--constraint-interval-end",
        type=float,
        default=8.0,
    )
    parser.add_argument("--svd-rtol", type=float, default=1.0e-10)
    parser.add_argument("--minimum-nullity", type=int, default=1)
    parser.add_argument(
        "--max-structured-prelogit-drift",
        type=float,
        default=1.0e-8,
    )
    parser.add_argument(
        "--max-nominal-control-drift",
        type=float,
        default=1.0e-12,
    )
    parser.add_argument(
        "--max-structured-control-drift",
        type=float,
        default=1.0e-8,
    )
    parser.add_argument(
        "--max-protected-lopt-relative-increase",
        type=float,
        default=1.0e-6,
    )
    parser.add_argument(
        "--max-protected-lopt-absolute-increase",
        type=float,
        default=1.0e-12,
    )
    parser.add_argument("--train-random-states", type=int, default=16)
    parser.add_argument("--validation-random-states", type=int, default=32)
    parser.add_argument("--train-seed", type=int, default=20261520)
    parser.add_argument("--validation-seed", type=int, default=20261521)
    parser.add_argument("--outer-steps", type=int, default=6)
    parser.add_argument(
        "--optimizer",
        choices=("lbfgs", "constrained-lm"),
        default="lbfgs",
        help=(
            "LBFGS preserves the historical behavior. constrained-lm uses "
            "a damped Gauss--Newton step with linearized protected scalar-"
            "Lopt constraints and nonlinear backtracking."
        ),
    )
    parser.add_argument("--inner-iterations", type=int, default=2)
    parser.add_argument("--max-eval", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--history-size", type=int, default=30)
    parser.add_argument("--tolerance-grad", type=float, default=1.0e-12)
    parser.add_argument("--tolerance-change", type=float, default=1.0e-14)
    parser.add_argument("--lm-initial-damping", type=float, default=1.0e-2)
    parser.add_argument("--lm-minimum-damping", type=float, default=1.0e-10)
    parser.add_argument("--lm-maximum-damping", type=float, default=1.0e10)
    parser.add_argument("--lm-damping-decrease", type=float, default=0.3)
    parser.add_argument("--lm-damping-increase", type=float, default=10.0)
    parser.add_argument("--lm-maximum-attempts", type=int, default=6)
    parser.add_argument("--lm-backtracking-steps", type=int, default=8)
    parser.add_argument("--lm-backtracking-factor", type=float, default=0.5)
    parser.add_argument("--lm-maximum-step-norm", type=float, default=0.5)
    parser.add_argument(
        "--lm-acceptance-tolerance",
        type=float,
        default=1.0e-12,
    )
    parser.add_argument("--report-scale-factor", type=float, default=400.0)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
