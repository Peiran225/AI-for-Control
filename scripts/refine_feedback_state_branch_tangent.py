#!/usr/bin/env python3
"""Full-state-branch scalar-PMP refinement with anchor-tangent projection.

The incoming feedback checkpoint and its final network architecture are kept
unchanged.  The time branch is frozen.  Every parameter in the existing state
branch may move, but the stochastic-state scalar-PMP gradient is projected
away from a deterministic sketch of the full-horizon nominal and structured
control-response Jacobian.  The nominal and structured scalar ``L_opt``
gradients are added to the tangent constraints explicitly.

Candidate steps are accepted only when the complete RK4-stage control
responses and the protected scalar ``L_opt`` values remain within hard
feasibility gates.  No physical objective, direct-control target, control
trust penalty, or candidate mask is used.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_feedback_section5 import load_feedback_checkpoint
from scripts.feedback_continuous_policy_rk4 import pchip_midpoint_logits
from scripts.refine_feedback_last_layer_near_null import (
    _scalar_loss,
    iid_and_composition_states,
    scalar_lopt_by_sample,
)
from scripts.refine_feedback_offgrid_scalar import (
    fine_problem,
    fixed_support_dense_logits,
    physical_metrics,
)
from scripts.train_feedback_section5 import dynamics
from train_paper_pmp_kkt import ProblemConfig, build_params


def flatten_optional_gradients(
    gradients: Sequence[torch.Tensor | None],
    parameters: Sequence[torch.nn.Parameter],
) -> torch.Tensor:
    """Flatten autograd results, replacing unused entries by exact zeros."""

    if len(gradients) != len(parameters):
        raise ValueError("gradient and parameter lists must have equal length")
    values: list[torch.Tensor] = []
    for gradient, parameter in zip(gradients, parameters):
        if gradient is None:
            values.append(torch.zeros_like(parameter).reshape(-1))
        else:
            if gradient.shape != parameter.shape:
                raise ValueError("gradient shape does not match its parameter")
            values.append(gradient.reshape(-1))
    if not values:
        raise ValueError("at least one trainable parameter is required")
    return torch.cat(values)


def parameter_vector(parameters: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    """Return a detached flat copy of a parameter sequence."""

    if not parameters:
        raise ValueError("at least one parameter is required")
    return torch.cat([parameter.detach().reshape(-1) for parameter in parameters])


def assign_parameter_vector(
    parameters: Sequence[torch.nn.Parameter],
    vector: torch.Tensor,
) -> None:
    """Assign a flat vector to an existing parameter sequence."""

    expected = sum(parameter.numel() for parameter in parameters)
    if vector.ndim != 1 or vector.numel() != expected:
        raise ValueError(
            f"expected a flat vector with {expected} values, got "
            f"{tuple(vector.shape)}"
        )
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            size = parameter.numel()
            parameter.copy_(
                vector[offset : offset + size].reshape_as(parameter)
            )
            offset += size


def project_onto_constraint_tangent(
    gradient: torch.Tensor,
    constraint_gradients: torch.Tensor,
    relative_tolerance: float,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """Project ``gradient`` onto the nullspace of constraint-gradient rows."""

    if gradient.ndim != 1:
        raise ValueError("the optimization gradient must be one-dimensional")
    if (
        constraint_gradients.ndim != 2
        or constraint_gradients.shape[1] != gradient.numel()
    ):
        raise ValueError(
            "constraint gradients must have shape (constraints, parameters)"
        )
    if not 0.0 < relative_tolerance < 1.0:
        raise ValueError("projection tolerance must lie strictly in (0,1)")
    row_norm = constraint_gradients.norm(dim=1)
    nonzero = row_norm > torch.finfo(gradient.dtype).eps
    if not bool(nonzero.any()):
        return gradient.clone(), 0, gradient.new_empty(0)
    normalized = (
        constraint_gradients[nonzero]
        / row_norm[nonzero].unsqueeze(1)
    )
    _, singular_values, vh = torch.linalg.svd(
        normalized,
        full_matrices=False,
    )
    cutoff = singular_values[0] * float(relative_tolerance)
    rank = int((singular_values > cutoff).sum().item())
    if rank == 0:
        return gradient.clone(), 0, singular_values
    row_basis = vh[:rank]
    projected = gradient - row_basis.T @ (row_basis @ gradient)
    return projected, rank, singular_values


def parse_structured_radii(text: str) -> tuple[float, ...]:
    """Parse positive, distinct structured-shift radii."""

    values = tuple(float(value.strip()) for value in text.split(",") if value.strip())
    if not values:
        raise ValueError("at least one protected structured radius is required")
    if any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in values):
        raise ValueError("structured radii must be finite and lie in (0,1)")
    if len(set(values)) != len(values):
        raise ValueError("structured radii must be distinct")
    return values


def protected_initial_states(
    cfg: ProblemConfig,
    structured_radii: Iterable[float],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return nominal followed by fixed-total structured shifts."""

    direction = torch.linspace(
        -1.0,
        1.0,
        cfg.m,
        device=device,
        dtype=dtype,
    )
    return torch.stack(
        (
            torch.full(
                (cfg.m,),
                cfg.n0,
                device=device,
                dtype=dtype,
            ),
            *[
                cfg.n0 * (1.0 + float(radius) * direction)
                for radius in structured_radii
            ],
        )
    )


def continuous_rk4_stage_controls(
    model: torch.nn.Module,
    initial_state: torch.Tensor,
    cfg: ProblemConfig,
    normalized_time: torch.Tensor,
    raw_logits: torch.Tensor,
    midpoint_raw_logits: torch.Tensor,
    params: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return all four continuously queried policy actions per RK4 interval."""

    if initial_state.ndim != 2 or initial_state.shape[1] != cfg.m:
        raise ValueError("initial states must have shape (batch, m)")
    if normalized_time.shape != (cfg.n + 1,):
        raise ValueError("normalized time does not match the RK4 grid")
    if raw_logits.shape != (cfg.n + 1,):
        raise ValueError("raw node logits do not match the RK4 grid")
    if midpoint_raw_logits.shape != (cfg.n,):
        raise ValueError("midpoint logits do not match the RK4 intervals")
    step = cfg.T / cfg.n
    batch = initial_state.shape[0]
    state = initial_state
    intervals: list[torch.Tensor] = []

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

    for index in range(cfg.n):
        left_time = normalized_time[index].expand(batch)
        middle_time = (
            0.5 * (normalized_time[index] + normalized_time[index + 1])
        ).expand(batch)
        right_time = normalized_time[index + 1].expand(batch)

        control1 = action(raw_logits[index], left_time, state)
        slope1 = dynamics(state, control1, params)

        stage2 = state + 0.5 * step * slope1
        control2 = action(
            midpoint_raw_logits[index],
            middle_time,
            stage2,
        )
        slope2 = dynamics(stage2, control2, params)

        stage3 = state + 0.5 * step * slope2
        control3 = action(
            midpoint_raw_logits[index],
            middle_time,
            stage3,
        )
        slope3 = dynamics(stage3, control3, params)

        stage4 = state + step * slope3
        control4 = action(raw_logits[index + 1], right_time, stage4)
        slope4 = dynamics(stage4, control4, params)

        intervals.append(
            torch.stack((control1, control2, control3, control4), dim=-1)
        )
        state = state + (step / 6.0) * (
            slope1 + 2.0 * slope2 + 2.0 * slope3 + slope4
        )
        if not bool(torch.isfinite(state).all()):
            raise FloatingPointError("protected RK4 trajectory became nonfinite")
    return torch.stack(intervals, dim=1)


def deterministic_rademacher_sketch(
    rows: int,
    columns: int,
    seed: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return normalized fixed-sign sketch rows."""

    if rows < 1 or columns < 1:
        raise ValueError("sketch dimensions must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    signs = torch.randint(
        0,
        2,
        (rows, columns),
        generator=generator,
        dtype=torch.int64,
    )
    return (
        (2.0 * signs.to(dtype=torch.float64) - 1.0)
        / math.sqrt(columns)
    ).to(device=device, dtype=dtype)


def autograd_flat(
    value: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> torch.Tensor:
    """Differentiate one scalar and flatten the result."""

    gradients = torch.autograd.grad(
        value,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return flatten_optional_gradients(gradients, parameters)


def protected_metrics(
    model: torch.nn.Module,
    protected_states: torch.Tensor,
    cfg: ProblemConfig,
    dense_time: torch.Tensor,
    dense_raw: torch.Tensor,
    midpoint_raw: torch.Tensor,
    params: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return full-horizon stage controls and per-state scalar ``L_opt``."""

    stage_controls = continuous_rk4_stage_controls(
        model,
        protected_states,
        cfg,
        dense_time,
        dense_raw,
        midpoint_raw,
        params,
    )
    _, scalar_pack = _scalar_loss(
        model,
        protected_states,
        torch.ones(
            protected_states.shape[0],
            device=protected_states.device,
            dtype=protected_states.dtype,
        ),
        cfg,
        dense_time,
        dense_raw,
        midpoint_raw,
        params,
        args,
    )
    return stage_controls, scalar_lopt_by_sample(scalar_pack)


def preservation_gate(
    control_drift: torch.Tensor,
    scalar_lopt: torch.Tensor,
    scalar_limit: torch.Tensor,
    *,
    max_nominal_control_drift: float,
    max_structured_control_drift: float,
) -> bool:
    """Check complete protected control trajectories and scalar residuals."""

    if control_drift.ndim != 1 or control_drift.numel() < 2:
        raise ValueError("protected drift must include nominal and structured rows")
    if scalar_lopt.shape != control_drift.shape:
        raise ValueError("protected scalar values do not match control rows")
    control_drift = control_drift.detach()
    scalar_lopt = scalar_lopt.detach()
    scalar_limit = scalar_limit.detach()
    return (
        float(control_drift[0]) <= max_nominal_control_drift
        and bool(
            (control_drift[1:] <= max_structured_control_drift).all()
        )
        and bool((scalar_lopt <= scalar_limit).all())
    )


def run(args: argparse.Namespace) -> None:
    if args.train_seed == args.validation_seed:
        raise ValueError("training and validation seeds must differ")
    if not 0.0 < args.radius < 1.0:
        raise ValueError("random-state radius must lie in (0,1)")
    if args.sketch_rows < 1:
        raise ValueError("at least one anchor sketch row is required")
    if args.max_backtracks < 0:
        raise ValueError("max backtracks must be nonnegative")
    structured_radii = parse_structured_radii(args.structured_radii)

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
            "anchor-tangent refinement requires centered state correction"
        )
    cfg = fine_problem(base_cfg, args.multiplier)
    model.to(device=device, dtype=torch.float64).eval()
    incoming_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    params = build_params(cfg, device, torch.float64)
    model.set_feature_vectors(params["r"], params["phi"])
    dense_time, dense_raw = fixed_support_dense_logits(
        model,
        base_cfg,
        args.multiplier,
        query_batch_size=args.query_batch_size,
    )
    midpoint_raw = pchip_midpoint_logits(dense_time, dense_raw)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = list(model.state_branch.parameters())
    for parameter in trainable:
        parameter.requires_grad_(True)

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
    protected_states = protected_initial_states(
        cfg,
        structured_radii,
        device=device,
        dtype=torch.float64,
    )
    with torch.no_grad():
        baseline_controls, baseline_lopt = protected_metrics(
            model,
            protected_states,
            cfg,
            dense_time,
            dense_raw,
            midpoint_raw,
            params,
            args,
        )
    baseline_controls = baseline_controls.detach()
    baseline_lopt = baseline_lopt.detach()
    scalar_limit = (
        baseline_lopt * (1.0 + args.max_protected_lopt_relative_increase)
        + args.max_protected_lopt_absolute_increase
    )

    history: list[dict[str, float | int]] = []
    best_validation = math.inf
    best_step = 0
    best_state = copy.deepcopy(model.state_dict())

    def evaluate(step: int, accepted_step_size: float) -> bool:
        nonlocal best_validation, best_step, best_state
        with torch.no_grad():
            train_loss, _ = _scalar_loss(
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
            controls, scalar_lopt = protected_metrics(
                model,
                protected_states,
                cfg,
                dense_time,
                dense_raw,
                midpoint_raw,
                params,
                args,
            )
            drift = (controls - baseline_controls).abs().amax(dim=(1, 2))
            feasible = preservation_gate(
                drift,
                scalar_lopt,
                scalar_limit,
                max_nominal_control_drift=args.max_nominal_control_drift,
                max_structured_control_drift=(
                    args.max_structured_control_drift
                ),
            )
            row: dict[str, float | int] = {
                "outer_step": step,
                "accepted_step_size": accepted_step_size,
                "train_loss": float(train_loss),
                "validation_loss": float(validation_loss),
                "preservation_gate_pass": int(feasible),
                "nominal_control_drift_max": float(drift[0]),
                "nominal_scalar_lopt": float(scalar_lopt[0]),
            }
            for index, radius in enumerate(structured_radii, start=1):
                key = f"structured_r{radius:g}".replace(".", "p")
                row[f"{key}_control_drift_max"] = float(drift[index])
                row[f"{key}_scalar_lopt"] = float(scalar_lopt[index])
            row.update(
                {
                    f"validation_{name}": value
                    for name, value in physical_metrics(
                        validation_pack,
                        scale=args.report_scale_factor,
                    ).items()
                }
            )
            history.append(row)
            if feasible and float(validation_loss) < best_validation:
                best_validation = float(validation_loss)
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
        print(json.dumps(history[-1], sort_keys=True), flush=True)
        return feasible

    evaluate(0, 0.0)
    for outer_step in range(1, args.outer_steps + 1):
        model.train()
        train_loss, _ = _scalar_loss(
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
        random_gradient = autograd_flat(
            train_loss,
            trainable,
            retain_graph=False,
        )

        protected_controls = continuous_rk4_stage_controls(
            model,
            protected_states,
            cfg,
            dense_time,
            dense_raw,
            midpoint_raw,
            params,
        )
        constraint_rows: list[torch.Tensor] = []
        for protected_index in range(protected_controls.shape[0]):
            flat_controls = protected_controls[protected_index].reshape(-1)
            sketch = deterministic_rademacher_sketch(
                args.sketch_rows,
                flat_controls.numel(),
                (
                    args.sketch_seed
                    + 1009 * outer_step
                    + 9176 * protected_index
                ),
                device=device,
                dtype=torch.float64,
            )
            sketch_values = sketch @ flat_controls
            for sketch_index in range(args.sketch_rows):
                constraint_rows.append(
                    autograd_flat(
                        sketch_values[sketch_index],
                        trainable,
                        retain_graph=True,
                    )
                )

        _, protected_pack = _scalar_loss(
            model,
            protected_states,
            torch.ones(
                protected_states.shape[0],
                device=device,
                dtype=torch.float64,
            ),
            cfg,
            dense_time,
            dense_raw,
            midpoint_raw,
            params,
            args,
        )
        protected_scalar = scalar_lopt_by_sample(protected_pack)
        for index in range(protected_scalar.numel()):
            constraint_rows.append(
                autograd_flat(
                    protected_scalar[index],
                    trainable,
                    retain_graph=(
                        index + 1 < protected_scalar.numel()
                    ),
                )
            )
        constraints = torch.stack(constraint_rows)
        projected_gradient, tangent_rank, singular_values = (
            project_onto_constraint_tangent(
                random_gradient,
                constraints,
                args.projection_rtol,
            )
        )
        gradient_norm = random_gradient.norm()
        projected_norm = projected_gradient.norm()
        if not bool(torch.isfinite(projected_gradient).all()):
            raise FloatingPointError("projected gradient became nonfinite")
        if float(projected_norm) <= torch.finfo(torch.float64).eps:
            print("Projected gradient is numerically zero; stopping.", flush=True)
            break

        current_vector = parameter_vector(trainable)
        accepted = False
        accepted_size = 0.0
        current_train = float(train_loss.detach())
        for backtrack in range(args.max_backtracks + 1):
            step_size = args.lr * (args.backtrack_factor**backtrack)
            candidate = current_vector - step_size * projected_gradient
            assign_parameter_vector(trainable, candidate)
            with torch.no_grad():
                candidate_train, _ = _scalar_loss(
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
                candidate_controls, candidate_lopt = protected_metrics(
                    model,
                    protected_states,
                    cfg,
                    dense_time,
                    dense_raw,
                    midpoint_raw,
                    params,
                    args,
                )
                candidate_drift = (
                    candidate_controls - baseline_controls
                ).abs().amax(dim=(1, 2))
                feasible = preservation_gate(
                    candidate_drift,
                    candidate_lopt,
                    scalar_limit,
                    max_nominal_control_drift=(
                        args.max_nominal_control_drift
                    ),
                    max_structured_control_drift=(
                        args.max_structured_control_drift
                    ),
                )
                decreases = float(candidate_train) < current_train
            if feasible and decreases:
                accepted = True
                accepted_size = step_size
                break
        if not accepted:
            assign_parameter_vector(trainable, current_vector)
            print(
                "No feasible decreasing tangent step after backtracking; "
                "stopping.",
                flush=True,
            )
            break
        feasible_after = evaluate(outer_step, accepted_size)
        history[-1].update(
            {
                "raw_gradient_l2": float(gradient_norm),
                "projected_gradient_l2": float(projected_norm),
                "tangent_constraint_rank": tangent_rank,
                "tangent_singular_value_max": (
                    float(singular_values[0])
                    if singular_values.numel()
                    else 0.0
                ),
                "tangent_singular_value_min": (
                    float(singular_values[-1])
                    if singular_values.numel()
                    else 0.0
                ),
            }
        )
        if not feasible_after:
            raise RuntimeError("accepted tangent step failed its repeated gate")

    model.load_state_dict(best_state)
    model.eval()
    selected_state = model.state_dict()
    changed_keys = [
        key
        for key, value in selected_state.items()
        if not torch.equal(value.detach().cpu(), incoming_state[key])
    ]
    if any(not key.startswith("state_branch.") for key in changed_keys):
        raise RuntimeError(
            "anchor-tangent refinement changed parameters outside the "
            f"existing state branch: {changed_keys}"
        )
    with torch.no_grad():
        final_controls, final_lopt = protected_metrics(
            model,
            protected_states,
            cfg,
            dense_time,
            dense_raw,
            midpoint_raw,
            params,
            args,
        )
    final_drift = (final_controls - baseline_controls).abs().amax(dim=(1, 2))
    if not preservation_gate(
        final_drift,
        final_lopt,
        scalar_limit,
        max_nominal_control_drift=args.max_nominal_control_drift,
        max_structured_control_drift=args.max_structured_control_drift,
    ):
        raise RuntimeError("selected checkpoint violates a preservation gate")

    payload = copy.deepcopy(source_checkpoint)
    output_args = dict(source_checkpoint["args"])
    output_args.update(
        {
            "state_branch_tangent_refinement": True,
            "random_state_radius": args.radius,
            "protected_structured_radii": list(structured_radii),
        }
    )
    payload.update(
        {
            "model_state": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "args": output_args,
            "initialization_checkpoint": str(checkpoint_path),
            "best_validation_loss": best_validation,
            "best_selection_loss": best_validation,
            "best_epoch": best_step,
            "selection_metric": (
                "independent_validation_scalar_Lopt_under_"
                "full_horizon_anchor_gates"
            ),
            "refinement": {
                "optimizer": "anchor-tangent projected gradient",
                "train_scope": "all existing state-branch parameters",
                "physical_objective_used": False,
                "direct_supervision_used": False,
                "control_trust_penalty_used": False,
                "trajectory_mode": "continuous-policy",
                "state_requery": "every RK4 stage",
                "scalar_residual_weights": [1.0, 1.0, 4.0],
                "random_state_radius": args.radius,
                "protected_structured_radii": list(structured_radii),
                "anchor_sketch_rows_per_protected_trajectory": (
                    args.sketch_rows
                ),
                "anchor_sketch_seed": args.sketch_seed,
                "projection_relative_tolerance": args.projection_rtol,
                "protected_initial_scalar_lopt": [
                    float(value) for value in baseline_lopt
                ],
                "protected_final_scalar_lopt": [
                    float(value) for value in final_lopt
                ],
                "protected_final_control_drift_max": [
                    float(value) for value in final_drift
                ],
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
                "changed_parameter_keys": changed_keys,
            },
        }
    )
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_dir / "best_feedback_section5.pt")
    with (out_dir / "history.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        fieldnames = sorted({key for row in history for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "best_validation_loss": best_validation,
                "best_outer_step": best_step,
                "protected_structured_radii": list(structured_radii),
                "protected_initial_scalar_lopt": [
                    float(value) for value in baseline_lopt
                ],
                "protected_final_scalar_lopt": [
                    float(value) for value in final_lopt
                ],
                "protected_final_control_drift_max": [
                    float(value) for value in final_drift
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
    parser.add_argument("--multiplier", type=int, default=1)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--radius", type=float, default=0.20)
    parser.add_argument("--structured-radii", default="0.20")
    parser.add_argument("--interval-start", type=float, default=1.5)
    parser.add_argument("--interval-end", type=float, default=8.0)
    parser.add_argument("--train-random-states", type=int, default=8)
    parser.add_argument("--validation-random-states", type=int, default=16)
    parser.add_argument("--train-seed", type=int, default=20261520)
    parser.add_argument("--validation-seed", type=int, default=20261521)
    parser.add_argument("--outer-steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1.0e-2)
    parser.add_argument("--max-backtracks", type=int, default=8)
    parser.add_argument("--backtrack-factor", type=float, default=0.5)
    parser.add_argument("--sketch-rows", type=int, default=12)
    parser.add_argument("--sketch-seed", type=int, default=817263)
    parser.add_argument("--projection-rtol", type=float, default=1.0e-10)
    parser.add_argument(
        "--max-nominal-control-drift",
        type=float,
        default=1.0e-4,
    )
    parser.add_argument(
        "--max-structured-control-drift",
        type=float,
        default=1.0e-4,
    )
    parser.add_argument(
        "--max-protected-lopt-relative-increase",
        type=float,
        default=1.0e-2,
    )
    parser.add_argument(
        "--max-protected-lopt-absolute-increase",
        type=float,
        default=1.0e-12,
    )
    parser.add_argument("--report-scale-factor", type=float, default=400.0)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
