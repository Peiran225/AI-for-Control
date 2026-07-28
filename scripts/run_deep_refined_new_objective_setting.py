#!/usr/bin/env python3
"""Rebuild one new-weight three-case result from a deep-refined time Transformer.

The time-only checkpoint is selected before this script and must be a
teacher-free, source-inclusive full projected-gradient result.  The existing
feedback state branch is transferred onto that new frozen time branch and
retrained under the same Section-5 losses.  All outputs use new versioned
paths, leaving the original experiment tree intact.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "outputs/new_objective_alpha1_grid_20260721_v2"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_new_objective_weight_setting import (  # noqa: E402
    PYTHON,
    feedback_command,
    normalized_problem,
    physical_problem,
    rebind_feedback_checkpoint,
    rebind_time_checkpoint,
    run_command,
    scalar_tag,
    sha256,
    write_json,
)


def validate_time_checkpoint(
    path: Path,
    normalized: dict[str, Any],
    maximum_pg_linf: float,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("teacher_free") is not True:
        raise ValueError("deep-refined time checkpoint is not teacher-free")
    for key in (
        "direct_or_manual_solution_used",
        "switching_time_or_mask_used",
        "objective_value_used_as_loss_or_selection",
    ):
        if payload.get(key) is not False:
            raise ValueError(f"deep-refined time checkpoint does not certify {key}=False")
    if payload.get("problem") != normalized:
        raise ValueError("deep-refined time checkpoint problem mismatch")
    if payload.get("wrapper", {}).get("class") != "LinearRawBoxProjection":
        raise ValueError("expected the standard LinearRawBoxProjection Transformer")
    metrics = payload.get("selection_residual_metrics", {})
    pg_linf = float(metrics.get("projected_gradient_linf", math.inf))
    if not math.isfinite(pg_linf) or pg_linf > maximum_pg_linf:
        raise ValueError(
            f"deep-refined time checkpoint PG Linf {pg_linf:.6g} exceeds "
            f"the required {maximum_pg_linf:.6g}"
        )
    return payload


def resolved_recorded_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(f"missing recorded path for {label}")
    return Path(value).expanduser().resolve()


def require_recorded_path(value: Any, expected: Path, *, label: str) -> None:
    observed = resolved_recorded_path(value, label=label)
    if observed != expected.expanduser().resolve():
        raise ValueError(
            f"{label} mismatch: recorded {observed}, expected {expected.resolve()}"
        )


def require_same_model_state(
    observed: dict[str, Any],
    expected: dict[str, Any],
    *,
    label: str,
    allow_dtype_conversion: bool = False,
) -> None:
    if observed.keys() != expected.keys():
        missing = sorted(expected.keys() - observed.keys())
        extra = sorted(observed.keys() - expected.keys())
        raise ValueError(
            f"{label} state-dict keys differ (missing={missing}, extra={extra})"
        )
    for key in observed:
        left = observed[key]
        right = expected[key]
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise TypeError(f"{label}.{key} is not a tensor")
        if allow_dtype_conversion:
            right = right.to(dtype=left.dtype)
        if not torch.equal(left.detach().cpu(), right.detach().cpu()):
            raise ValueError(f"{label}.{key} differs from its current source")


def save_checkpoint_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.provenance.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def ensure_time_rebind(
    source: Path,
    destination: Path,
    normalized: dict[str, Any],
    physical: dict[str, Any],
    scale: float,
) -> None:
    source_sha = sha256(source)
    rebind_time_checkpoint(source, destination, normalized, physical, scale)
    source_payload = torch.load(source, map_location="cpu", weights_only=False)
    rebound = torch.load(destination, map_location="cpu", weights_only=False)
    if rebound.get("problem") != physical:
        raise ValueError(f"{destination}: physical problem mismatch")
    require_recorded_path(
        rebound.get("normalized_source_checkpoint"),
        source,
        label=f"{destination}: normalized source",
    )
    require_same_model_state(
        rebound.get("model_state", {}),
        source_payload.get("model_state", {}),
        label=f"{destination}: rebound time model",
    )
    recorded_sha = rebound.get("normalized_source_checkpoint_sha256")
    if recorded_sha is not None and recorded_sha != source_sha:
        raise ValueError(f"{destination}: normalized source SHA256 mismatch")
    if recorded_sha is None:
        rebound["normalized_source_checkpoint_sha256"] = source_sha
        save_checkpoint_atomic(destination, rebound)


def feedback_provenance(
    *,
    time_checkpoint: Path,
    state_checkpoint: Path,
    selected: Path,
    completion: Path,
) -> dict[str, Any]:
    return {
        "time_checkpoint": str(time_checkpoint.resolve()),
        "time_checkpoint_sha256": sha256(time_checkpoint),
        "state_checkpoint": str(state_checkpoint.resolve()),
        "state_checkpoint_sha256": sha256(state_checkpoint),
        "selected_checkpoint": str(selected.resolve()),
        "selected_checkpoint_sha256": sha256(selected),
        "training_completion": str(completion.resolve()),
        "training_completion_sha256": sha256(completion),
    }


def validate_feedback_selected(
    selected: Path,
    *,
    time_checkpoint: Path,
    state_checkpoint: Path,
    normalized: dict[str, Any],
) -> None:
    payload = torch.load(selected, map_location="cpu", weights_only=False)
    if payload.get("problem") != normalized:
        raise ValueError(f"{selected}: normalized problem mismatch")
    require_recorded_path(
        payload.get("time_checkpoint"),
        time_checkpoint,
        label=f"{selected}: time checkpoint",
    )
    require_recorded_path(
        payload.get("state_checkpoint"),
        state_checkpoint,
        label=f"{selected}: state checkpoint",
    )
    checkpoint_args = payload.get("args", {})
    if checkpoint_args.get("freeze_time_branch") is not True:
        raise ValueError(f"{selected}: time branch was not frozen")
    require_recorded_path(
        checkpoint_args.get("time_checkpoint"),
        time_checkpoint,
        label=f"{selected}: args.time_checkpoint",
    )
    require_recorded_path(
        checkpoint_args.get("state_checkpoint"),
        state_checkpoint,
        label=f"{selected}: args.state_checkpoint",
    )

    time_payload = torch.load(time_checkpoint, map_location="cpu", weights_only=False)
    expected_time_state = {
        key.removeprefix("base."): value
        for key, value in time_payload.get("model_state", {}).items()
        if key.startswith("base.")
    }
    observed_time_state = {
        key.removeprefix("time_branch."): value
        for key, value in payload.get("model_state", {}).items()
        if key.startswith("time_branch.")
    }
    if not expected_time_state:
        raise ValueError(f"{time_checkpoint}: wrapped base state is missing")
    require_same_model_state(
        observed_time_state,
        expected_time_state,
        label=f"{selected}: frozen time branch",
        # Feedback training is intentionally float32, whereas the exact
        # full-gradient time-only refinement is stored in float64.  Loading
        # the frozen branch performs this one deterministic dtype cast.
        allow_dtype_conversion=True,
    )


def ensure_feedback_training(
    command: list[str],
    *,
    output: Path,
    selected: Path,
    time_checkpoint: Path,
    state_checkpoint: Path,
    normalized: dict[str, Any],
    command_log: list[dict[str, Any]],
) -> None:
    completion = output / "test_summary.json"
    provenance_path = output / "deep_refine_provenance.json"
    if output.is_dir() and any(output.iterdir()) and not completion.is_file():
        raise RuntimeError(
            f"refusing to reuse incomplete feedback directory: {output}"
        )
    if completion.is_file():
        if not selected.is_file():
            raise RuntimeError(
                f"completed feedback directory lacks selected checkpoint: {output}"
            )
        validate_feedback_selected(
            selected,
            time_checkpoint=time_checkpoint,
            state_checkpoint=state_checkpoint,
            normalized=normalized,
        )
        if provenance_path.is_file():
            observed = json.loads(provenance_path.read_text(encoding="utf-8"))
            expected = feedback_provenance(
                time_checkpoint=time_checkpoint,
                state_checkpoint=state_checkpoint,
                selected=selected,
                completion=completion,
            )
            if observed != expected:
                raise ValueError(f"{output}: existing feedback provenance mismatch")

    run_command(command, expected=completion, command_log=command_log)
    if not selected.is_file():
        raise FileNotFoundError(
            f"feedback training completed without selected checkpoint: {selected}"
        )
    validate_feedback_selected(
        selected,
        time_checkpoint=time_checkpoint,
        state_checkpoint=state_checkpoint,
        normalized=normalized,
    )
    write_json(
        provenance_path,
        feedback_provenance(
            time_checkpoint=time_checkpoint,
            state_checkpoint=state_checkpoint,
            selected=selected,
            completion=completion,
        ),
    )


def ensure_feedback_rebind(
    source: Path,
    destination: Path,
    physical_time_checkpoint: Path,
    normalized: dict[str, Any],
    physical: dict[str, Any],
    scale: float,
) -> None:
    source_sha = sha256(source)
    time_sha = sha256(physical_time_checkpoint)
    rebind_feedback_checkpoint(
        source,
        destination,
        physical_time_checkpoint,
        normalized,
        physical,
        scale,
    )
    source_payload = torch.load(source, map_location="cpu", weights_only=False)
    rebound = torch.load(destination, map_location="cpu", weights_only=False)
    if rebound.get("problem") != physical:
        raise ValueError(f"{destination}: physical problem mismatch")
    require_recorded_path(
        rebound.get("normalized_source_checkpoint"),
        source,
        label=f"{destination}: normalized source",
    )
    require_recorded_path(
        rebound.get("time_checkpoint"),
        physical_time_checkpoint,
        label=f"{destination}: physical time checkpoint",
    )
    require_same_model_state(
        rebound.get("model_state", {}),
        source_payload.get("model_state", {}),
        label=f"{destination}: rebound feedback model",
    )
    recorded_source_sha = rebound.get("normalized_source_checkpoint_sha256")
    recorded_time_sha = rebound.get("physical_time_checkpoint_sha256")
    if recorded_source_sha is not None and recorded_source_sha != source_sha:
        raise ValueError(f"{destination}: normalized source SHA256 mismatch")
    if recorded_time_sha is not None and recorded_time_sha != time_sha:
        raise ValueError(f"{destination}: physical time SHA256 mismatch")
    if recorded_source_sha is None or recorded_time_sha is None:
        rebound["normalized_source_checkpoint_sha256"] = source_sha
        rebound["physical_time_checkpoint_sha256"] = time_sha
        save_checkpoint_atomic(destination, rebound)


def diagnostics_provenance(
    *, time_checkpoint: Path, cf_checkpoint: Path, der_checkpoint: Path, metadata: Path
) -> dict[str, Any]:
    return {
        "time_checkpoint": str(time_checkpoint.resolve()),
        "time_checkpoint_sha256": sha256(time_checkpoint),
        "case1_checkpoint": str(cf_checkpoint.resolve()),
        "case1_checkpoint_sha256": sha256(cf_checkpoint),
        "case2_checkpoint": str(der_checkpoint.resolve()),
        "case2_checkpoint_sha256": sha256(der_checkpoint),
        "metadata": str(metadata.resolve()),
        "metadata_sha256": sha256(metadata),
    }


def ensure_diagnostics(
    command: list[str],
    *,
    diagnostics: Path,
    time_checkpoint: Path,
    cf_checkpoint: Path,
    der_checkpoint: Path,
    command_log: list[dict[str, Any]],
) -> None:
    metadata_path = diagnostics / "metadata.json"
    provenance_path = diagnostics / "deep_refine_provenance.json"
    if diagnostics.is_dir() and any(diagnostics.iterdir()) and not metadata_path.is_file():
        raise RuntimeError(
            f"refusing to reuse incomplete diagnostics directory: {diagnostics}"
        )
    if metadata_path.is_file():
        if not provenance_path.is_file():
            raise RuntimeError(
                f"completed diagnostics lack deep-refinement provenance: {diagnostics}"
            )
        observed = json.loads(provenance_path.read_text(encoding="utf-8"))
        expected = diagnostics_provenance(
            time_checkpoint=time_checkpoint,
            cf_checkpoint=cf_checkpoint,
            der_checkpoint=der_checkpoint,
            metadata=metadata_path,
        )
        if observed != expected:
            raise ValueError(f"{diagnostics}: existing diagnostics provenance mismatch")

    run_command(command, expected=metadata_path, command_log=command_log)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    recorded = metadata.get("checkpoints", {})
    for key, expected_path in (
        ("time_only", time_checkpoint),
        ("feedback_cf", cf_checkpoint),
        ("feedback_der", der_checkpoint),
    ):
        require_recorded_path(
            recorded.get(key), expected_path, label=f"{metadata_path}: {key}"
        )
    write_json(
        provenance_path,
        diagnostics_provenance(
            time_checkpoint=time_checkpoint,
            cf_checkpoint=cf_checkpoint,
            der_checkpoint=der_checkpoint,
            metadata=metadata_path,
        ),
    )


def build(args: argparse.Namespace) -> None:
    for name in ("alpha", "beta", "gamma"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    physical = physical_problem(args.beta, args.gamma, alpha=args.alpha)
    normalized, scale = normalized_problem(
        args.beta,
        args.gamma,
        alpha=args.alpha,
    )
    tag = (
        f"a{scalar_tag(args.alpha)}_"
        f"b{scalar_tag(args.beta)}_g{scalar_tag(args.gamma)}"
    )
    setting = args.root.expanduser().resolve() / tag
    time_normalized = (
        setting
        / "time_only"
        / args.time_refinement_dirname
        / "selected_checkpoint.pt"
    )
    time_payload = validate_time_checkpoint(
        time_normalized, normalized, args.maximum_pg_linf
    )

    command_log_path = setting / "deep_refine_pipeline_commands.json"
    command_log: list[dict[str, Any]] = []
    if command_log_path.is_file():
        previous = json.loads(command_log_path.read_text(encoding="utf-8"))
        if isinstance(previous, list):
            command_log.extend(previous)

    time_physical = (
        setting
        / "time_only"
        / args.time_refinement_dirname
        / "selected_checkpoint_physical.pt"
    )
    ensure_time_rebind(
        time_normalized, time_physical, normalized, physical, scale
    )

    feedback = setting / "feedback"
    cases = (
        (
            "cf",
            feedback / "case1_cf_deep_v1_train",
            feedback / "case1_cf_exact_train/best_feedback_section5_full_gradient.pt",
            feedback / "case1_cf_deep_v1_physical.pt",
        ),
        (
            "der",
            feedback / "case2_der_deep_v1_train",
            feedback / "case2_der_exact_train/best_feedback_section5_full_gradient.pt",
            feedback / "case2_der_deep_v1_physical.pt",
        ),
    )
    physical_feedback: dict[str, Path] = {}
    for option, output, state_checkpoint, rebound in cases:
        command = feedback_command(
            option=option,
            time_checkpoint=time_normalized,
            output=output,
            normalized=normalized,
            smoke=False,
        )
        command.extend(["--state_checkpoint", str(state_checkpoint)])
        selected = output / "best_feedback_section5_full_gradient.pt"
        ensure_feedback_training(
            command,
            output=output,
            selected=selected,
            time_checkpoint=time_normalized,
            state_checkpoint=state_checkpoint,
            normalized=normalized,
            command_log=command_log,
        )
        ensure_feedback_rebind(
            selected,
            rebound,
            time_physical,
            normalized,
            physical,
            scale,
        )
        physical_feedback[option] = rebound

    diagnostics = setting / args.diagnostics_dirname
    diagnostics_command = [
            str(PYTHON),
            "scripts/generate_two_state_three_case_main_figures.py",
            "--time-checkpoint",
            str(time_physical),
            "--cf-checkpoint",
            str(physical_feedback["cf"]),
            "--der-checkpoint",
            str(physical_feedback["der"]),
            "--out-dir",
            str(diagnostics),
            "--n",
            "800",
            "--kkt-tolerance",
            str(scale * 1.0e-4),
        ]
    ensure_diagnostics(
        diagnostics_command,
        diagnostics=diagnostics,
        time_checkpoint=time_physical,
        cf_checkpoint=physical_feedback["cf"],
        der_checkpoint=physical_feedback["der"],
        command_log=command_log,
    )
    write_json(command_log_path, command_log)
    write_json(
        setting / "DEEP_REFINEMENT_COMPLETED.json",
        {
            "status": "completed",
            "alpha": args.alpha,
            "beta": args.beta,
            "gamma": args.gamma,
            "time_checkpoint_normalized": str(time_normalized),
            "time_checkpoint_normalized_sha256": sha256(time_normalized),
            "time_checkpoint_physical": str(time_physical),
            "time_checkpoint_physical_sha256": sha256(time_physical),
            "time_selected_origin": time_payload.get("selected_origin"),
            "time_refinement_dirname": args.time_refinement_dirname,
            "time_normalized_projected_gradient_linf": float(
                time_payload["selection_residual_metrics"]["projected_gradient_linf"]
            ),
            "case1_checkpoint_physical": str(physical_feedback["cf"]),
            "case1_checkpoint_normalized": str(
                feedback / "case1_cf_deep_v1_train/best_feedback_section5_full_gradient.pt"
            ),
            "case1_checkpoint_normalized_sha256": sha256(
                feedback / "case1_cf_deep_v1_train/best_feedback_section5_full_gradient.pt"
            ),
            "case1_checkpoint_physical_sha256": sha256(physical_feedback["cf"]),
            "case2_checkpoint_physical": str(physical_feedback["der"]),
            "case2_checkpoint_normalized": str(
                feedback / "case2_der_deep_v1_train/best_feedback_section5_full_gradient.pt"
            ),
            "case2_checkpoint_normalized_sha256": sha256(
                feedback / "case2_der_deep_v1_train/best_feedback_section5_full_gradient.pt"
            ),
            "case2_checkpoint_physical_sha256": sha256(physical_feedback["der"]),
            "diagnostics": str(diagnostics),
            "diagnostics_dirname": args.diagnostics_dirname,
            "diagnostics_metadata_sha256": sha256(diagnostics / "metadata.json"),
        },
    )
    print(f"completed deep-refined setting {setting}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--maximum-pg-linf", type=float, default=1.0e-4)
    parser.add_argument(
        "--diagnostics-dirname", default="two_state_diagnostics_deep_v1"
    )
    parser.add_argument(
        "--time-refinement-dirname",
        default="deep_refine_kkt_head_ols_exact_v3",
    )
    args = parser.parse_args()
    if args.diagnostics_dirname != Path(args.diagnostics_dirname).name:
        raise ValueError("diagnostics dirname must be one plain directory name")
    if args.time_refinement_dirname != Path(args.time_refinement_dirname).name:
        raise ValueError("time refinement dirname must be one plain directory name")
    build(args)


if __name__ == "__main__":
    main()
