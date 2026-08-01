#!/usr/bin/env python3
"""Refine only a Transformer's output head with scalar optimality residuals.

The input checkpoint may have been initialized by fitting a direct
transcription.  After loading that checkpoint, this script uses neither the
direct control nor the physical objective.  It freezes the time-feature
embedding and Transformer encoder, and applies a damped Gauss--Newton
(Levenberg--Marquardt) iteration to the original linear output head.

All support nodes and all strict midpoint queries are evaluated together with
the exact fixed-support attention mask.  By default, the residual vector
contains trajectory-wise ``H_u``,
``d H_u / dt``, and ``d^2 H_u / dt^2`` on the stated interior interval.  A
controlled ablation can instead match the state-only closed-form (CF)
singular-control target on the same interval.  Both modes add projected
box-KKT residuals only where the starting control is near a box boundary.  No
auxiliary control correction is added.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT, ROOT / "scripts"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from continue_direct_initialized_offgrid_scalar_transformer import (  # noqa: E402
    compute_continuous_costate_rk4,
    simulate_continuous_control_rk4,
)
from continue_direct_initialized_scalar_transformer import (  # noqa: E402
    clone_state,
    load_model,
    sha256,
)
from refine_direct_offgrid_query_fit import (  # noqa: E402
    fixed_support_query_control,
)
from scripts.fixed_nominal_opt_gap import (  # noqa: E402
    TimeOnlyPolicyAdapter,
    evaluate_fixed_nominal_der,
    metric_metadata,
)
from train_feedback_section5 import singular_quantities  # noqa: E402
from train_paper_pmp_kkt import build_params, set_seed, time_features  # noqa: E402
from train_teacher_free_resolution_curriculum import evaluate  # noqa: E402


def resolve(path: str | Path) -> Path:
    expanded = Path(path).expanduser()
    return (expanded if expanded.is_absolute() else ROOT / expanded).resolve()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2607)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--trajectory-multiplier",
        type=int,
        default=2,
        help="Uniform RK4 intervals per original Transformer interval.",
    )
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument(
        "--trainable-scope",
        choices=("output-head", "input-output"),
        default="output-head",
        help=(
            "Refine only the final projection, or the original input and "
            "output projections while keeping the Transformer encoder frozen."
        ),
    )
    parser.add_argument("--interior-start", type=float, default=1.5)
    parser.add_argument("--interior-end", type=float, default=8.0)
    parser.add_argument("--boundary-margin", type=float, default=0.05)
    parser.add_argument("--w0", type=float, default=1.0)
    parser.add_argument("--w1", type=float, default=1.0)
    parser.add_argument("--w2", type=float, default=1.0)
    parser.add_argument(
        "--singular-loss",
        choices=("derivative", "cf-state"),
        default="derivative",
        help=(
            "Use scalar PMP time-derivative residuals, or match the "
            "state-only closed-form singular-control target on the same "
            "fixed interior interval."
        ),
    )
    parser.add_argument("--cf-weight", type=float, default=1.0)
    parser.add_argument("--cf-scale", type=float, default=1.0)
    parser.add_argument("--b-min", type=float, default=1.0e-8)
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--psi-scale", type=float, default=1.0)
    parser.add_argument("--dot-scale", type=float, default=1.0)
    parser.add_argument("--ddot-scale", type=float, default=1.0)
    parser.add_argument("--boundary-scale", type=float, default=1.0)
    parser.add_argument("--report-scale-factor", type=float, default=400.0)
    parser.add_argument("--initial-damping", type=float, default=1.0e-2)
    parser.add_argument("--minimum-damping", type=float, default=1.0e-10)
    parser.add_argument("--maximum-damping", type=float, default=1.0e12)
    parser.add_argument("--damping-decrease", type=float, default=0.3)
    parser.add_argument("--damping-increase", type=float, default=10.0)
    parser.add_argument("--maximum-attempts", type=int, default=8)
    parser.add_argument("--maximum-step-norm", type=float, default=2.0)
    parser.add_argument("--acceptance-tolerance", type=float, default=1.0e-12)
    parser.add_argument(
        "--selection-metric",
        choices=("physical-all", "training-objective"),
        default="physical-all",
        help=(
            "Select the retained checkpoint by all three reported physical "
            "RMS components (legacy behavior), or by the residual objective "
            "actually optimized.  The latter avoids using zero-weight "
            "diagnostics for first-order-only controls."
        ),
    )
    parser.add_argument(
        "--jacobian-mode",
        choices=("forward", "reverse"),
        default="forward",
    )
    parser.add_argument(
        "--linear-solver",
        choices=("explicit", "matrix-free"),
        default="explicit",
        help=(
            "Form the small output-head Jacobian explicitly, or apply "
            "Gauss--Newton matrix products with JVP/VJP and conjugate gradients."
        ),
    )
    parser.add_argument("--cg-iterations", type=int, default=16)
    parser.add_argument("--cg-relative-tolerance", type=float, default=1.0e-4)
    args = parser.parse_args()

    if not (
        args.iterations >= 0
        and args.trajectory_multiplier >= 2
        and args.query_batch_size > 0
        and 0.0 <= args.interior_start < args.interior_end <= 10.0
        and args.boundary_margin >= 0.0
        and args.maximum_attempts > 0
        and args.maximum_step_norm > 0.0
        and args.cg_iterations > 0
        and args.cg_relative_tolerance > 0.0
    ):
        raise ValueError("invalid iteration, interval, or trust-region setting")
    if args.trainable_scope == "input-output" and args.linear_solver != "matrix-free":
        raise ValueError(
            "--trainable-scope=input-output requires --linear-solver=matrix-free"
        )
    if (
        args.trainable_scope == "input-output"
        and args.trajectory_multiplier != 2
    ):
        raise ValueError(
            "input-output scope currently requires --trajectory-multiplier=2"
        )
    for name in (
        "psi_scale",
        "dot_scale",
        "ddot_scale",
        "cf_scale",
        "b_min",
        "boundary_scale",
        "initial_damping",
        "minimum_damping",
        "maximum_damping",
        "damping_decrease",
        "damping_increase",
    ):
        if getattr(args, name) <= 0.0:
            raise ValueError(f"{name} must be positive")
    for name in ("w0", "w1", "w2", "cf_weight", "boundary_weight"):
        if getattr(args, name) < 0.0:
            raise ValueError(f"{name} must be nonnegative")
    if (
        args.singular_loss == "cf-state"
        and args.selection_metric != "training-objective"
    ):
        raise ValueError(
            "CF state-target runs must be selected by their actual training "
            "objective"
        )

    set_seed(args.seed)
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    start_checkpoint = resolve(args.start_checkpoint)
    out_dir = resolve(args.out_dir)
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True)

    model, cfg, source = load_model(start_checkpoint, device)
    model.eval()
    fixed_validation_model = TimeOnlyPolicyAdapter(model, cfg)
    fixed_validation_params = build_params(
        cfg, device, torch.float64
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if not isinstance(model.base.output, torch.nn.Linear):
        raise TypeError("the frozen Transformer must have a linear output head")

    dense_cfg = replace(cfg, n=args.trajectory_multiplier * cfg.n)
    support_time = torch.linspace(
        0.0, 1.0, cfg.n + 1, device=device, dtype=torch.float64
    )
    dense_normalized = torch.linspace(
        0.0,
        1.0,
        dense_cfg.n + 1,
        device=device,
        dtype=torch.float64,
    )
    dense_indices = torch.arange(
        dense_cfg.n + 1, device=device, dtype=torch.long
    )
    on_support = dense_indices.remainder(args.trajectory_multiplier) == 0
    query_time = dense_normalized[~on_support]
    combined_time = torch.cat((support_time, query_time))
    support_count = int(support_time.numel())
    dense_lookup = torch.empty_like(dense_indices)
    dense_lookup[on_support] = dense_indices[on_support] // (
        args.trajectory_multiplier
    )
    dense_lookup[~on_support] = support_count + torch.arange(
        query_time.numel(), device=device, dtype=torch.long
    )
    combined_features = time_features(combined_time)
    attention_mask: torch.Tensor | None = None
    with torch.no_grad():
        if args.trainable_scope == "output-head":
            support_features = time_features(support_time)
            support_encoded = model.base.encoder(
                model.base.input(support_features).unsqueeze(0)
            ).squeeze(0)
            pointwise_mask = torch.zeros(
                (support_count + 1, support_count + 1),
                device=device,
                dtype=torch.bool,
            )
            pointwise_mask[:support_count, support_count] = True
            pointwise_mask[support_count, support_count] = True
            query_encoded_batches = []
            for start in range(0, int(query_time.numel()), args.query_batch_size):
                current = query_time[start : start + args.query_batch_size]
                batch = int(current.numel())
                features = torch.cat(
                    (
                        support_features.unsqueeze(0).expand(
                            batch, -1, -1
                        ),
                        time_features(current).unsqueeze(1),
                    ),
                    dim=1,
                )
                hidden = model.base.input(features)
                encoded_batch = model.base.encoder(
                    hidden,
                    mask=pointwise_mask,
                )
                query_encoded_batches.append(
                    encoded_batch[:, -1, :].detach()
                )
            frozen_encoded = torch.cat(
                (support_encoded, *query_encoded_batches),
                dim=0,
            ).detach()
        else:
            sequence_length = int(combined_time.numel())
            attention_mask = torch.zeros(
                (sequence_length, sequence_length),
                device=device,
                dtype=torch.bool,
            )
            attention_mask[:support_count, support_count:] = True
            attention_mask[support_count:, support_count:] = True
            embedded = model.base.input(combined_features).unsqueeze(0)
            frozen_encoded = model.base.encoder(
                embedded,
                mask=attention_mask,
            ).squeeze(0).detach()
    head_width = int(model.base.output.in_features)
    input_shape = tuple(model.base.input.weight.shape)
    input_weight_count = int(model.base.input.weight.numel())
    input_bias_count = int(model.base.input.bias.numel())
    output_weight_count = int(model.base.output.weight.numel())
    output_bias_count = int(model.base.output.bias.numel())
    if args.trainable_scope == "output-head":
        initial_theta = torch.cat(
            (
                model.base.output.weight.detach().reshape(-1),
                model.base.output.bias.detach().reshape(-1),
            )
        )
    else:
        initial_theta = torch.cat(
            (
                model.base.input.weight.detach().reshape(-1),
                model.base.input.bias.detach().reshape(-1),
                model.base.output.weight.detach().reshape(-1),
                model.base.output.bias.detach().reshape(-1),
            )
        )
    initial_theta = initial_theta.to(device=device, dtype=torch.float64)

    def control_from_theta(theta: torch.Tensor) -> torch.Tensor:
        if args.trainable_scope == "output-head":
            output_weight = theta[:output_weight_count].reshape(
                1, head_width
            )
            output_bias = theta[
                output_weight_count : output_weight_count + output_bias_count
            ]
            encoded = frozen_encoded
        else:
            offset = 0
            input_weight = theta[
                offset : offset + input_weight_count
            ].reshape(input_shape)
            offset += input_weight_count
            input_bias = theta[
                offset : offset + input_bias_count
            ]
            offset += input_bias_count
            output_weight = theta[
                offset : offset + output_weight_count
            ].reshape(1, head_width)
            offset += output_weight_count
            output_bias = theta[
                offset : offset + output_bias_count
            ]
            embedded_value = torch.nn.functional.linear(
                combined_features,
                input_weight,
                input_bias,
            ).unsqueeze(0)
            if attention_mask is None:
                raise RuntimeError("input-output attention mask was not built")
            encoded = model.base.encoder(
                embedded_value,
                mask=attention_mask,
            ).squeeze(0)
        raw = torch.nn.functional.linear(
            encoded,
            output_weight,
            output_bias,
        ).squeeze(-1)
        base_control = model.umax * torch.sigmoid(raw)
        combined_control = torch.clamp(
            model.scale * base_control - model.offset,
            0.0,
            model.umax,
        )
        return combined_control[dense_lookup]

    with torch.no_grad():
        starting_dense = control_from_theta(initial_theta).detach()
        reference_combined = torch.cat(
            (
                model(support_time),
                fixed_support_query_control(
                    model,
                    support_time,
                    query_time,
                    batch_size=args.query_batch_size,
                ),
            )
        )
        reference_dense = reference_combined[dense_lookup]
        construction_error = float(
            (starting_dense - reference_dense).abs()
            .max()
            .cpu()
        )
    if construction_error > 2.0e-11:
        raise RuntimeError(
            "frozen-feature output-head construction changed the policy: "
            f"{construction_error:.3e}"
        )

    physical_time = torch.linspace(
        0.0, dense_cfg.T, dense_cfg.n + 1,
        device=device, dtype=torch.float64,
    )
    interval_time = physical_time[:-1]
    interior = (
        (interval_time >= args.interior_start)
        & (interval_time < args.interior_end)
    ).unsqueeze(0)
    source_interval_control = starting_dense[:-1].unsqueeze(0)
    boundary_gate = (
        (source_interval_control <= args.boundary_margin)
        | (
            source_interval_control
            >= dense_cfg.umax - args.boundary_margin
        )
    )
    if not bool(interior.any()) or not bool(boundary_gate.any()):
        raise RuntimeError("the scalar interval or source-box gate is empty")

    params = build_params(dense_cfg, device, torch.float64)
    original_params = build_params(cfg, device, torch.float64)
    initial_state = torch.full(
        (1, dense_cfg.m),
        dense_cfg.n0,
        device=device,
        dtype=torch.float64,
    )

    def raw_pack(theta: torch.Tensor) -> dict[str, torch.Tensor]:
        all_control = control_from_theta(theta)
        node_control = all_control.unsqueeze(0)
        controls = node_control[:, :-1]
        states, midpoint_states = simulate_continuous_control_rk4(
            node_control,
            initial_state,
            dense_cfg,
            params,
        )
        costates = compute_continuous_costate_rk4(
            states,
            midpoint_states,
            node_control,
            dense_cfg,
            params,
        )
        quantities = singular_quantities(
            states,
            controls,
            costates,
            dense_cfg,
            params,
            costate_pairing="current",
        )
        psi_all = quantities["psi"]
        psi = args.report_scale_factor * psi_all[interior]
        dot = args.report_scale_factor * quantities["dot_psi"][interior]
        ddot = args.report_scale_factor * quantities["ddot_psi"][interior]
        cf_candidate_all = quantities["u_state"]
        cf = (controls - cf_candidate_all)[interior]
        cf_candidate = cf_candidate_all[interior]
        cf_B = quantities["B"][interior]
        cf_in_box = (
            (cf_candidate >= 0.0) & (cf_candidate <= dense_cfg.umax)
        )
        cf_B_valid = cf_B.abs() >= args.b_min
        cf_B_nonpositive = cf_B <= 0.0
        cf_admissible = cf_in_box & cf_B_valid & cf_B_nonpositive
        projected = controls - torch.clamp(
            controls - psi_all, 0.0, dense_cfg.umax
        )
        boundary = (
            args.report_scale_factor * projected[boundary_gate]
        )
        return {
            "all_control": all_control,
            "psi": psi,
            "dot": dot,
            "ddot": ddot,
            "cf": cf,
            "cf_candidate": cf_candidate,
            "cf_B": cf_B,
            "cf_in_box": cf_in_box,
            "cf_B_valid": cf_B_valid,
            "cf_B_nonpositive": cf_B_nonpositive,
            "cf_admissible": cf_admissible,
            "boundary": boundary,
        }

    if args.singular_loss == "derivative":
        component_specs = (
            ("psi", args.w0, args.psi_scale),
            ("dot", args.w1, args.dot_scale),
            ("ddot", args.w2, args.ddot_scale),
            ("boundary", args.boundary_weight, args.boundary_scale),
        )
    else:
        component_specs = (
            ("cf", args.cf_weight, args.cf_scale),
            ("boundary", args.boundary_weight, args.boundary_scale),
        )

    def residual_vector(theta: torch.Tensor) -> torch.Tensor:
        pack = raw_pack(theta)
        residuals = []
        for name, weight, scale in component_specs:
            value = pack[name].reshape(-1)
            if weight > 0.0:
                residuals.append(
                    math.sqrt(weight / value.numel()) * value / scale
                )
        if not residuals:
            raise RuntimeError("at least one residual weight must be positive")
        return torch.cat(residuals)

    def metrics(theta: torch.Tensor) -> dict[str, float]:
        with torch.no_grad():
            pack = raw_pack(theta)
            values = {
                f"{name}_rms": float(pack[name].square().mean().sqrt().cpu())
                for name in ("psi", "dot", "ddot", "cf", "boundary")
            }
            values.update(
                {
                    f"{name}_linf": float(pack[name].abs().max().cpu())
                    for name in ("psi", "dot", "ddot", "cf", "boundary")
                }
            )
            values.update(
                {
                    "cf_candidate_min": float(pack["cf_candidate"].min().cpu()),
                    "cf_candidate_max": float(pack["cf_candidate"].max().cpu()),
                    "cf_B_min": float(pack["cf_B"].min().cpu()),
                    "cf_B_max": float(pack["cf_B"].max().cpu()),
                    "cf_in_box_fraction": float(
                        pack["cf_in_box"].to(torch.float64).mean().cpu()
                    ),
                    "cf_B_valid_fraction": float(
                        pack["cf_B_valid"].to(torch.float64).mean().cpu()
                    ),
                    "cf_B_nonpositive_fraction": float(
                        pack["cf_B_nonpositive"].to(torch.float64).mean().cpu()
                    ),
                    "cf_admissible_fraction": float(
                        pack["cf_admissible"].to(torch.float64).mean().cpu()
                    ),
                }
            )
            values["control_drift_rms"] = float(
                (
                    pack["all_control"] - starting_dense
                ).square().mean().sqrt().cpu()
            )
            values["control_drift_linf"] = float(
                (pack["all_control"] - starting_dense).abs().max().cpu()
            )
            values["physical_component_max"] = max(
                values["psi_rms"],
                values["dot_rms"],
                values["ddot_rms"],
            )
            values["physical_joint_rms"] = math.sqrt(
                values["psi_rms"] ** 2
                + values["dot_rms"] ** 2
                + values["ddot_rms"] ** 2
            )
        return values

    def set_head(theta: torch.Tensor) -> None:
        with torch.no_grad():
            offset = 0
            if args.trainable_scope == "input-output":
                model.base.input.weight.copy_(
                    theta[
                        offset : offset + input_weight_count
                    ].reshape_as(model.base.input.weight)
                )
                offset += input_weight_count
                model.base.input.bias.copy_(
                    theta[offset : offset + input_bias_count].reshape_as(
                        model.base.input.bias
                    )
                )
                offset += input_bias_count
            model.base.output.weight.copy_(
                theta[
                    offset : offset + output_weight_count
                ].reshape_as(model.base.output.weight)
            )
            offset += output_weight_count
            model.base.output.bias.copy_(
                theta[
                    offset : offset + output_bias_count
                ].reshape_as(model.base.output.bias)
            )

    def make_payload(
        theta: torch.Tensor,
        iteration: int,
        reported_metrics: dict[str, float],
    ) -> dict[str, Any]:
        set_head(theta)
        return {
            **source,
            "model_state": clone_state(model),
            "source_checkpoint": str(start_checkpoint),
            "source_checkpoint_sha256": sha256(start_checkpoint),
            "teacher_free": False,
            "direct_or_manual_solution_used": True,
            "continuation_reads_direct_or_manual_solution": False,
            "continuation_uses_objective_value_as_loss_or_selection": False,
            "external_correction_head": False,
            "method": (
                "direct-initialized Transformer followed by "
                f"{args.trainable_scope} damped Gauss-Newton refinement "
                + (
                    "of scalar PMP/KKT residuals"
                    if args.singular_loss == "derivative"
                    else "of the state-only closed-form singular target"
                )
            ),
            "selected_lm_iteration": iteration,
            "scalar_dense_grid_metrics": reported_metrics,
            "lm_args": vars(args),
        }

    theta = initial_theta.clone()
    damping = args.initial_damping
    history: list[dict[str, Any]] = []
    candidates: list[
        tuple[float, float, torch.Tensor, int, dict[str, float]]
    ] = []
    started = time.perf_counter()

    def record(iteration: int, accepted: bool, objective: float) -> None:
        # The residual path evaluates a detached parameter vector.  Synchronize
        # the live Transformer before applying the common fixed validator.
        set_head(theta)
        fixed_validation = evaluate_fixed_nominal_der(
            fixed_validation_model,
            cfg,
            fixed_validation_params,
            state_mode="w_zero",
        )
        current_metrics = metrics(theta)
        row: dict[str, Any] = {
            "phase": "time_only_optimality_gap_refinement",
            "phase_step": iteration,
            "optimizer_step": iteration,
            "iteration": iteration,
            "accepted": accepted,
            "weighted_objective": objective,
            "damping": damping,
            "elapsed_seconds": time.perf_counter() - started,
            **fixed_validation,
            **current_metrics,
        }
        history.append(row)
        candidates.append(
            (
                current_metrics["physical_component_max"],
                current_metrics["physical_joint_rms"],
                theta.detach().cpu().clone(),
                iteration,
                current_metrics,
            )
        )
        print(
            f"[LM {iteration:03d}] RMS "
            f"{current_metrics['psi_rms']:.6e}/"
            f"{current_metrics['dot_rms']:.6e}/"
            f"{current_metrics['ddot_rms']:.6e} "
            f"max={current_metrics['physical_component_max']:.6e} "
            f"objective={objective:.6e} damping={damping:.3e} "
            f"accepted={accepted}",
            flush=True,
        )

    initial_residual = residual_vector(theta)
    objective = 0.5 * float(initial_residual.square().sum().detach().cpu())
    record(0, True, objective)

    def conjugate_gradient(
        matvec: Any,
        right_hand_side: torch.Tensor,
    ) -> torch.Tensor:
        solution = torch.zeros_like(right_hand_side)
        residual = right_hand_side.clone()
        direction = residual.clone()
        residual_norm_squared = residual.dot(residual)
        initial_norm = residual_norm_squared.sqrt().clamp_min(
            torch.finfo(residual.dtype).eps
        )
        for _ in range(args.cg_iterations):
            matrix_direction = matvec(direction)
            denominator = direction.dot(matrix_direction)
            if float(denominator.detach().cpu()) <= 0.0:
                break
            alpha = residual_norm_squared / denominator
            solution = solution + alpha * direction
            residual = residual - alpha * matrix_direction
            next_norm_squared = residual.dot(residual)
            if float(
                (next_norm_squared.sqrt() / initial_norm).detach().cpu()
            ) <= args.cg_relative_tolerance:
                break
            beta = next_norm_squared / residual_norm_squared
            direction = residual + beta * direction
            residual_norm_squared = next_norm_squared
        return solution

    for iteration in range(1, args.iterations + 1):
        theta_for_jacobian = theta.detach().requires_grad_(True)
        if args.linear_solver == "explicit":
            residual = residual_vector(theta_for_jacobian)
            if args.jacobian_mode == "forward":
                jacobian = torch.autograd.functional.jacobian(
                    residual_vector,
                    theta_for_jacobian,
                    vectorize=True,
                    strategy="forward-mode",
                )
            else:
                jacobian = torch.autograd.functional.jacobian(
                    residual_vector,
                    theta_for_jacobian,
                    vectorize=True,
                    strategy="reverse-mode",
                )
            residual = residual.detach()
            jacobian = jacobian.detach()
            gradient = jacobian.T @ residual
            normal = jacobian.T @ jacobian
            diagonal = normal.diagonal().clamp_min(
                torch.finfo(normal.dtype).eps
            )
        else:
            residual, vjp_function = torch.func.vjp(
                residual_vector,
                theta_for_jacobian,
            )
            residual = residual.detach()
            gradient = vjp_function(residual)[0].detach()
        accepted = False
        best_trial_theta = theta
        best_trial_objective = objective
        attempts = 0
        while attempts < args.maximum_attempts:
            attempts += 1
            if args.linear_solver == "explicit":
                system = normal + damping * torch.diag(diagonal)
                try:
                    step = torch.linalg.solve(system, -gradient)
                except torch.linalg.LinAlgError:
                    damping = min(
                        args.maximum_damping,
                        damping * args.damping_increase,
                    )
                    continue
            else:
                def matrix_vector_product(
                    vector: torch.Tensor,
                ) -> torch.Tensor:
                    _, jvp_value = torch.func.jvp(
                        residual_vector,
                        (theta_for_jacobian,),
                        (vector,),
                    )
                    return (
                        vjp_function(jvp_value)[0].detach()
                        + damping * vector
                    )

                step = conjugate_gradient(
                    matrix_vector_product,
                    -gradient,
                )
            step_norm = float(step.norm().detach().cpu())
            if step_norm > args.maximum_step_norm:
                step = step * (args.maximum_step_norm / step_norm)
            trial_theta = theta + step
            with torch.no_grad():
                trial_residual = residual_vector(trial_theta)
                trial_objective = 0.5 * float(
                    trial_residual.square().sum().cpu()
                )
            if trial_objective + args.acceptance_tolerance < objective:
                accepted = True
                best_trial_theta = trial_theta.detach()
                best_trial_objective = trial_objective
                damping = max(
                    args.minimum_damping,
                    damping * args.damping_decrease,
                )
                break
            damping = min(
                args.maximum_damping,
                damping * args.damping_increase,
            )
        if accepted:
            theta = best_trial_theta
            objective = best_trial_objective
        record(iteration, accepted, objective)
        if not accepted and damping >= args.maximum_damping:
            break

    if args.selection_metric == "training-objective":
        selected_iteration = min(
            range(len(history)),
            key=lambda index: float(history[index]["weighted_objective"]),
        )
        selected = candidates[selected_iteration]
    else:
        selected = min(candidates, key=lambda item: item[:2])
    selected_theta = selected[2].to(device=device, dtype=torch.float64)
    selected_iteration = selected[3]
    selected_metrics = selected[4]
    set_head(selected_theta)
    selected_control = control_from_theta(selected_theta).detach()
    # ``evaluate`` differentiates the reduced objective with respect to the
    # produced control.  Keep the encoder frozen, but temporarily mark the
    # output head trainable so that this diagnostic graph is constructed.
    for parameter in model.base.output.parameters():
        parameter.requires_grad_(True)
    grid_metrics, selected_support = evaluate(
        model,
        cfg,
        support_time,
        original_params,
        high_accuracy=True,
    )
    for parameter in model.base.output.parameters():
        parameter.requires_grad_(False)
    payload = make_payload(
        selected_theta,
        selected_iteration,
        selected_metrics,
    )
    payload["metrics"] = grid_metrics
    torch.save(payload, out_dir / "selected_checkpoint.pt")
    np.savez_compressed(
        out_dir / "solution.npz",
        t=physical_time.detach().cpu().numpy(),
        u=selected_control.cpu().numpy(),
        support_t=physical_time[on_support].detach().cpu().numpy(),
        support_u=selected_support,
        starting_u=starting_dense.detach().cpu().numpy(),
    )
    write_csv(out_dir / "history.csv", history)
    write_csv(
        out_dir / "fixed_validation_opt_gap.csv",
        [
            {
                key: row[key]
                for key in (
                    "phase",
                    "phase_step",
                    "optimizer_step",
                    "iteration",
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
    summary = {
        "status": "completed",
        "start_checkpoint": str(start_checkpoint),
        "start_checkpoint_sha256": sha256(start_checkpoint),
        "device": str(device),
        "trainable_parameter_count": int(initial_theta.numel()),
        "continuation_reads_direct_or_manual_solution": False,
        "continuation_uses_objective_value_as_loss_or_selection": False,
        "external_correction_head": False,
        "selected_iteration": selected_iteration,
        "selected_scalar_dense_grid_metrics": selected_metrics,
        "grid_metrics": grid_metrics,
        "construction_error": construction_error,
        "wall_seconds": time.perf_counter() - started,
        "training": vars(args),
        "fixed_validation_metric": metric_metadata(),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
