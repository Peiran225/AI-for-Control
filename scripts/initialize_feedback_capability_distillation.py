#!/usr/bin/env python3
"""Distill gated feedback capability into the existing DER state branch.

This is an initialization stage, not a final model-selection stage.  A
validated gated fallback supplies only its state-dependent capability
increment

    delta_u = u_gated(N,t) - u_locked(N,t).

The target for the ordinary DER architecture is its own incoming action plus
``delta_u``.  Consequently, differences between the teacher's locked
time/state policy and the DER source policy are not copied.  The student keeps
the original ``NestedFeedbackTransformer`` architecture; only its existing
``state_branch`` is trainable.

Nominal and structured r=.10/.20 trajectories are replayed from the incoming
DER checkpoint.  A candidate is selectable only if the complete four-stage
RK4 control response for every protected initial state remains inside the
configured hard drift limits.  The saved artifact is explicitly marked as an
initialization checkpoint.  Final cleanup and selection must be performed by
``refine_feedback_state_branch_tangent.py` using scalar PMP/DER residuals.

Neither the physical objective nor a direct-control target is used here.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.continue_feedback_full_state_gated_fallback import (  # noqa: E402
    FlatTubeProbeFeedbackTransformer,
    continuous_policy_stage_trace,
    load_gated_feedback_checkpoint,
)
from scripts.evaluate_feedback_section5 import (  # noqa: E402
    load_feedback_checkpoint,
)
from scripts.feedback_continuous_policy_rk4 import (  # noqa: E402
    pchip_midpoint_logits,
)
from scripts.refine_feedback_last_layer_near_null import (  # noqa: E402
    iid_and_composition_states,
)
from scripts.refine_feedback_state_branch_tangent import (  # noqa: E402
    continuous_rk4_stage_controls,
    parse_structured_radii,
    protected_initial_states,
)
from train_paper_pmp_kkt import ProblemConfig, build_params  # noqa: E402


INITIALIZATION_FORMAT = "existing_state_branch_capability_initialization_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_positive_teacher_audit(
    audit_path: Path,
    teacher_path: Path,
) -> dict[str, Any]:
    """Require an independent, positive audit for the capability teacher."""

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    expected_sha = sha256(teacher_path)
    recorded_sha = audit.get("artifact_protocol", {}).get("artifact_sha256")
    if recorded_sha != expected_sha:
        raise ValueError("teacher audit does not identify the supplied artifact")
    if audit.get("used_for_checkpoint_selection") is not False:
        raise ValueError("teacher audit was used for checkpoint selection")
    if audit.get("used_for_hyperparameter_tuning") is not False:
        raise ValueError("teacher audit was used for hyperparameter tuning")
    if (
        audit.get("physical_objective_used_only_for_posthoc_evaluation")
        is not True
    ):
        raise ValueError("teacher audit lacks post-hoc-only objective provenance")
    advantage = audit.get("candidate_advantage_over_frozen_time", {})
    interval = audit.get("paired_bootstrap", {}).get(
        "candidate_advantage_over_frozen_time", {}
    )
    if float(advantage.get("mean", -math.inf)) <= 0.0:
        raise ValueError("capability teacher lacks positive mean held-out advantage")
    if float(interval.get("lower_95", -math.inf)) <= 0.0:
        raise ValueError(
            "capability teacher lacks a positive paired-bootstrap lower bound"
        )
    return audit


def stage_raw_logits(
    node_logits: torch.Tensor,
    midpoint_logits: torch.Tensor,
) -> torch.Tensor:
    """Return the four base logits queried in every RK4 interval."""

    if node_logits.ndim != 1:
        raise ValueError("node logits must be one-dimensional")
    if midpoint_logits.shape != (node_logits.numel() - 1,):
        raise ValueError("midpoint logits do not match node intervals")
    return torch.stack(
        (
            node_logits[:-1],
            midpoint_logits,
            midpoint_logits,
            node_logits[1:],
        ),
        dim=-1,
    )


def _flat_stage_tensor(value: torch.Tensor) -> torch.Tensor:
    if value.ndim < 3:
        raise ValueError("an RK4 stage tensor must include batch, interval, stage")
    return value.reshape(-1, *value.shape[3:])


def capability_transfer_targets(
    source_student: torch.nn.Module,
    teacher: FlatTubeProbeFeedbackTransformer,
    query_states: torch.Tensor,
    normalized_time: torch.Tensor,
    teacher_base_logits: torch.Tensor,
    student_base_logits: torch.Tensor,
    teacher_actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return source-relative teacher targets and the transferred increment.

    All arguments describe the *same* pointwise policy queries.  The teacher's
    locked action is evaluated at those states rather than on a separately
    rolled trajectory, so the transferred quantity is a genuine local policy
    increment.
    """

    if query_states.ndim != 2:
        raise ValueError("query states must have shape (queries, m)")
    count = query_states.shape[0]
    expected = (count,)
    for name, value in (
        ("normalized_time", normalized_time),
        ("teacher_base_logits", teacher_base_logits),
        ("student_base_logits", student_base_logits),
        ("teacher_actions", teacher_actions),
    ):
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    with torch.no_grad():
        locked_actions = teacher.interval_action(
            teacher_base_logits,
            normalized_time,
            query_states,
            state_mode="locked_feedback",
        )
        source_actions = source_student.interval_action(
            student_base_logits,
            normalized_time,
            query_states,
            state_mode="feedback",
        )
        increment = teacher_actions - locked_actions
        targets = (source_actions + increment).clamp(
            0.0, float(source_student.umax)
        )
    return targets.detach(), increment.detach()


def capability_query_batch(
    source_student: torch.nn.Module,
    teacher: FlatTubeProbeFeedbackTransformer,
    initial_states: torch.Tensor,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Roll the teacher and build pointwise source-relative targets."""

    device = initial_states.device
    dtype = initial_states.dtype
    normalized_time = torch.linspace(
        0.0, 1.0, cfg.n + 1, device=device, dtype=dtype
    )
    with torch.no_grad():
        teacher_nodes = teacher.time_logits(normalized_time)
        teacher_midpoints = pchip_midpoint_logits(
            normalized_time, teacher_nodes
        )
        student_nodes = source_student.time_logits(normalized_time)
        student_midpoints = pchip_midpoint_logits(
            normalized_time, student_nodes
        )
        trace = continuous_policy_stage_trace(
            teacher,
            initial_states,
            cfg,
            normalized_time,
            teacher_nodes,
            teacher_midpoints,
            params,
            state_mode="feedback",
        )
        batch = initial_states.shape[0]
        teacher_stage_raw = stage_raw_logits(
            teacher_nodes, teacher_midpoints
        ).unsqueeze(0).expand(batch, -1, -1)
        student_stage_raw = stage_raw_logits(
            student_nodes, student_midpoints
        ).unsqueeze(0).expand(batch, -1, -1)
        stage_time = trace["normalized_time"].unsqueeze(0).expand(
            batch, -1, -1
        )
        flat_states = _flat_stage_tensor(trace["states"])
        flat_time = stage_time.reshape(-1)
        flat_teacher_raw = teacher_stage_raw.reshape(-1)
        flat_student_raw = student_stage_raw.reshape(-1)
        flat_teacher_actions = trace["controls"].reshape(-1)
        targets, increment = capability_transfer_targets(
            source_student,
            teacher,
            flat_states,
            flat_time,
            flat_teacher_raw,
            flat_student_raw,
            flat_teacher_actions,
        )
    return {
        "state": flat_states.detach(),
        "normalized_time": flat_time.detach(),
        "student_base_logit": flat_student_raw.detach(),
        "target_action": targets,
        "teacher_action": flat_teacher_actions.detach(),
        "source_action": source_student.interval_action(
            flat_student_raw,
            flat_time,
            flat_states,
            state_mode="feedback",
        ).detach(),
        "teacher_increment": increment,
    }


def anchor_query_batch(
    source_student: torch.nn.Module,
    initial_states: torch.Tensor,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return exact incoming-policy targets on protected trajectories."""

    device = initial_states.device
    dtype = initial_states.dtype
    normalized_time = torch.linspace(
        0.0, 1.0, cfg.n + 1, device=device, dtype=dtype
    )
    with torch.no_grad():
        nodes = source_student.time_logits(normalized_time)
        midpoints = pchip_midpoint_logits(normalized_time, nodes)
        trace = continuous_policy_stage_trace(
            source_student,
            initial_states,
            cfg,
            normalized_time,
            nodes,
            midpoints,
            params,
            state_mode="feedback",
        )
        batch = initial_states.shape[0]
        stage_raw = stage_raw_logits(nodes, midpoints).unsqueeze(0).expand(
            batch, -1, -1
        )
        stage_time = trace["normalized_time"].unsqueeze(0).expand(
            batch, -1, -1
        )
    return {
        "state": _flat_stage_tensor(trace["states"]).detach(),
        "normalized_time": stage_time.reshape(-1).detach(),
        "student_base_logit": stage_raw.reshape(-1).detach(),
        "target_action": trace["controls"].reshape(-1).detach(),
    }


def pointwise_action_mse(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return action-space MSE on a complete or indexed query batch."""

    if indices is None:
        state = batch["state"]
        time = batch["normalized_time"]
        raw = batch["student_base_logit"]
        target = batch["target_action"]
    else:
        state = batch["state"][indices]
        time = batch["normalized_time"][indices]
        raw = batch["student_base_logit"][indices]
        target = batch["target_action"][indices]
    prediction = model.interval_action(
        raw, time, state, state_mode="feedback"
    )
    return (prediction - target).square().mean()


def hard_anchor_gate(
    drift: torch.Tensor,
    *,
    nominal_limit: float,
    structured_limit: float,
) -> bool:
    """Require nominal and every structured trajectory to pass independently."""

    if drift.ndim != 1 or drift.numel() < 2:
        raise ValueError("anchor drift must contain nominal and structured rows")
    if min(nominal_limit, structured_limit) < 0.0:
        raise ValueError("anchor drift limits must be nonnegative")
    return bool(
        torch.isfinite(drift).all()
        and drift[0] <= nominal_limit
        and (drift[1:] <= structured_limit).all()
    )


def changed_parameter_keys(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> list[str]:
    """Return changed keys and reject incompatible state dictionaries."""

    if set(before) != set(after):
        raise ValueError("before/after state dictionaries have different keys")
    return [
        key
        for key in before
        if not torch.equal(before[key].detach().cpu(), after[key].detach().cpu())
    ]


def assert_existing_state_branch_only(keys: Iterable[str]) -> None:
    invalid = [key for key in keys if not key.startswith("state_branch.")]
    if invalid:
        raise RuntimeError(
            "capability initialization changed parameters outside the existing "
            f"state branch: {invalid}"
        )


def _sample_indices(
    count: int,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    if count < 1 or batch_size < 1:
        raise ValueError("dataset and batch sizes must be positive")
    return torch.randint(
        count,
        (min(batch_size, count),),
        generator=generator,
        device="cpu",
    ).to(device)


def _validate_compatibility(
    student_cfg: ProblemConfig,
    teacher_cfg: ProblemConfig,
    source_args: argparse.Namespace,
) -> None:
    for name in ("T", "n", "m", "umax", "n0"):
        if not math.isclose(
            float(getattr(student_cfg, name)),
            float(getattr(teacher_cfg, name)),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(f"student/teacher problem mismatch in {name}")
    if str(getattr(source_args, "option", "")).lower() != "der":
        raise ValueError("the student checkpoint must be a DER policy")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.train_seed == args.validation_seed:
        raise ValueError("training and validation seeds must differ")
    if not 0.0 < args.radius < 1.0:
        raise ValueError("random-state radius must lie in (0,1)")
    if args.steps < 0:
        raise ValueError("steps must be nonnegative")
    structured_radii = parse_structured_radii(args.structured_radii)
    if tuple(structured_radii) != (0.10, 0.20):
        raise ValueError("this protocol requires structured radii .10,.20")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device(args.device)
    if device.type == "cuda" and not args.allow_gpu:
        raise ValueError("GPU use requires the explicit --allow-gpu flag")
    dtype = torch.float64

    student_path = args.student_checkpoint.expanduser().resolve()
    teacher_path = args.teacher_checkpoint.expanduser().resolve()
    teacher_audit_path = args.teacher_blind_evaluation.expanduser().resolve()
    teacher_audit = validate_positive_teacher_audit(
        teacher_audit_path, teacher_path
    )
    source_payload = torch.load(
        student_path, map_location="cpu", weights_only=False
    )
    student, cfg, source_args = load_feedback_checkpoint(student_path)
    teacher, teacher_cfg, _, teacher_payload = (
        load_gated_feedback_checkpoint(teacher_path)
    )
    _validate_compatibility(cfg, teacher_cfg, source_args)
    student.to(device=device, dtype=dtype).eval()
    teacher.to(device=device, dtype=dtype).eval()
    source_student = copy.deepcopy(student).eval()
    for parameter in source_student.parameters():
        parameter.requires_grad_(False)
    params = build_params(cfg, device, dtype)
    student.set_feature_vectors(params["r"], params["phi"])
    source_student.set_feature_vectors(params["r"], params["phi"])
    teacher.set_feature_vectors(params["r"], params["phi"])

    for parameter in student.parameters():
        parameter.requires_grad_(False)
    trainable = list(student.state_branch.parameters())
    for parameter in trainable:
        parameter.requires_grad_(True)
    incoming_state = {
        key: value.detach().cpu().clone()
        for key, value in student.state_dict().items()
    }

    train_states, _, _ = iid_and_composition_states(
        args.train_states,
        args.train_seed,
        args.radius,
        cfg,
        device,
        dtype,
    )
    validation_states, _, _ = iid_and_composition_states(
        args.validation_states,
        args.validation_seed,
        args.radius,
        cfg,
        device,
        dtype,
    )
    protected_states = protected_initial_states(
        cfg,
        structured_radii,
        device=device,
        dtype=dtype,
    )
    train_queries = capability_query_batch(
        source_student, teacher, train_states, cfg, params
    )
    validation_queries = capability_query_batch(
        source_student, teacher, validation_states, cfg, params
    )
    anchor_queries = anchor_query_batch(
        source_student, protected_states, cfg, params
    )

    normalized_time = torch.linspace(
        0.0, 1.0, cfg.n + 1, device=device, dtype=dtype
    )
    with torch.no_grad():
        student_nodes = source_student.time_logits(normalized_time)
        student_midpoints = pchip_midpoint_logits(
            normalized_time, student_nodes
        )
        baseline_anchor_controls = continuous_rk4_stage_controls(
            source_student,
            protected_states,
            cfg,
            normalized_time,
            student_nodes,
            student_midpoints,
            params,
        ).detach()

    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(args.optimizer_seed)
    history: list[dict[str, float | int]] = []
    best_validation = math.inf
    best_step = 0
    best_state = copy.deepcopy(student.state_dict())

    def evaluate(step: int) -> None:
        nonlocal best_validation, best_step, best_state
        student.eval()
        with torch.no_grad():
            train_mse = pointwise_action_mse(student, train_queries)
            validation_mse = pointwise_action_mse(student, validation_queries)
            anchor_mse = pointwise_action_mse(student, anchor_queries)
            current_anchor_controls = continuous_rk4_stage_controls(
                student,
                protected_states,
                cfg,
                normalized_time,
                student_nodes,
                student_midpoints,
                params,
            )
            drift = (
                current_anchor_controls - baseline_anchor_controls
            ).abs().amax(dim=(1, 2))
            feasible = hard_anchor_gate(
                drift,
                nominal_limit=args.max_nominal_control_drift,
                structured_limit=args.max_structured_control_drift,
            )
        row: dict[str, float | int] = {
            "step": step,
            "train_teacher_mse": float(train_mse),
            "validation_teacher_mse": float(validation_mse),
            "anchor_query_mse": float(anchor_mse),
            "anchor_gate_pass": int(feasible),
            "nominal_control_drift_max": float(drift[0]),
            "structured_r0p10_control_drift_max": float(drift[1]),
            "structured_r0p20_control_drift_max": float(drift[2]),
        }
        history.append(row)
        if feasible and float(validation_mse) < best_validation:
            best_validation = float(validation_mse)
            best_step = step
            best_state = copy.deepcopy(student.state_dict())
        print(json.dumps(row, sort_keys=True), flush=True)

    evaluate(0)
    for step in range(1, args.steps + 1):
        student.train()
        optimizer.zero_grad(set_to_none=True)
        train_indices = _sample_indices(
            train_queries["state"].shape[0],
            args.batch_size,
            generator,
            device,
        )
        anchor_indices = _sample_indices(
            anchor_queries["state"].shape[0],
            args.anchor_batch_size,
            generator,
            device,
        )
        teacher_loss = pointwise_action_mse(
            student, train_queries, train_indices
        )
        anchor_loss = pointwise_action_mse(
            student, anchor_queries, anchor_indices
        )
        loss = teacher_loss + args.anchor_weight * anchor_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        if step % args.evaluate_every == 0 or step == args.steps:
            evaluate(step)

    student.load_state_dict(best_state)
    student.eval()
    changed = changed_parameter_keys(incoming_state, student.state_dict())
    assert_existing_state_branch_only(changed)
    with torch.no_grad():
        final_anchor_controls = continuous_rk4_stage_controls(
            student,
            protected_states,
            cfg,
            normalized_time,
            student_nodes,
            student_midpoints,
            params,
        )
        final_drift = (
            final_anchor_controls - baseline_anchor_controls
        ).abs().amax(dim=(1, 2))
    if not hard_anchor_gate(
        final_drift,
        nominal_limit=args.max_nominal_control_drift,
        structured_limit=args.max_structured_control_drift,
    ):
        raise RuntimeError("selected initialization violates a hard anchor")

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = copy.deepcopy(source_payload)
    output_args = dict(source_payload["args"])
    output_args.update(
        {
            "capability_initialization": True,
            "capability_initialization_format": INITIALIZATION_FORMAT,
        }
    )
    payload.update(
        {
            "model_state": {
                key: value.detach().cpu()
                for key, value in student.state_dict().items()
            },
            "args": output_args,
            "initialization_checkpoint": str(student_path),
            "best_epoch": best_step,
            "best_validation_loss": best_validation,
            "best_selection_loss": best_validation,
            "selection_metric": (
                "teacher_action_MSE_under_full_horizon_control_anchor_gates"
            ),
            "capability_initialization": {
                "format": INITIALIZATION_FORMAT,
                "student_source": {
                    "path": str(student_path),
                    "sha256": sha256(student_path),
                },
                "capability_teacher": {
                    "path": str(teacher_path),
                    "sha256": sha256(teacher_path),
                    "checkpoint_format": teacher_payload["checkpoint_format"],
                    "role": "initialization_only",
                    "positive_audit": {
                        "path": str(teacher_audit_path),
                        "sha256": sha256(teacher_audit_path),
                        "mean_advantage_over_frozen_time": teacher_audit[
                            "candidate_advantage_over_frozen_time"
                        ]["mean"],
                        "paired_bootstrap_lower_95": teacher_audit[
                            "paired_bootstrap"
                        ]["candidate_advantage_over_frozen_time"]["lower_95"],
                        "used_for_student_training_or_selection": False,
                    },
                },
                "target": "student_source_action_plus_gated_minus_locked_action",
                "train_scope": "existing_state_branch_only",
                "physical_objective_used": False,
                "direct_control_target_used": False,
                "protected_structured_radii": list(structured_radii),
                "protected_final_control_drift_max": [
                    float(value) for value in final_drift
                ],
                "train_seed": args.train_seed,
                "validation_seed": args.validation_seed,
                "optimizer_seed": args.optimizer_seed,
                "changed_parameter_keys": changed,
                "final_cleanup_required": True,
                "required_cleanup": (
                    "scalar_PMP_DER_with_residual_only_checkpoint_selection"
                ),
            },
        }
    )
    checkpoint = out_dir / "capability_initialized_feedback_section5.pt"
    torch.save(payload, checkpoint)
    with (out_dir / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "format": INITIALIZATION_FORMAT,
        "best_step": best_step,
        "best_validation_teacher_mse": best_validation,
        "protected_structured_radii": list(structured_radii),
        "protected_final_control_drift_max": [
            float(value) for value in final_drift
        ],
        "changed_parameter_keys": changed,
        "next_stage": {
            "script": "scripts/refine_feedback_state_branch_tangent.py",
            "loss": "scalar PMP/DER only",
            "selection": "independent validation scalar PMP residual",
            "structured_radii": ".10,.20",
        },
        "problem": asdict(cfg),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--teacher-blind-evaluation",
        type=Path,
        required=True,
        help=(
            "independent positive audit of the gated capability teacher; "
            "used only to validate teacher provenance"
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--allow-gpu",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--radius", type=float, default=0.20)
    parser.add_argument("--structured-radii", default=".10,.20")
    parser.add_argument("--train-states", type=int, default=24)
    parser.add_argument("--validation-states", type=int, default=48)
    parser.add_argument("--train-seed", type=int, default=20261610)
    parser.add_argument("--validation-seed", type=int, default=20261611)
    parser.add_argument("--optimizer-seed", type=int, default=20261612)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--anchor-batch-size", type=int, default=512)
    parser.add_argument("--anchor-weight", type=float, default=100.0)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--evaluate-every", type=int, default=25)
    parser.add_argument("--max-nominal-control-drift", type=float, default=1.0e-4)
    parser.add_argument(
        "--max-structured-control-drift", type=float, default=1.0e-4
    )
    return parser


if __name__ == "__main__":
    print(
        json.dumps(
            run(build_parser().parse_args()),
            indent=2,
            sort_keys=True,
        )
    )
