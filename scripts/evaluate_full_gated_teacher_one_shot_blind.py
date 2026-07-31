#!/usr/bin/env python3
"""Formal one-shot blind evaluator for the frozen full-gated CF teacher.

The protocol is not configurable: 128 iid componentwise-random states at
radius 0.20 are generated with newly reserved seed 20261701, and paired mean
differences use 20,000 bootstrap repeats.  The primary comparison is
``frozen time - teacher`` and the secondary comparison is
``locked CF source - teacher``; positive differences favor the teacher.

Artifact, locked-source, evaluator, and dependency hashes must be supplied.
An exclusive repository-wide seed ledger is created before the reserved
states are materialized.  A failed run therefore still consumes the seed.
Existing result files and ledgers are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import scipy
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for search_path in (ROOT, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts.continue_feedback_full_state_gated_fallback import (  # noqa: E402
    ANCHOR_FAMILY_PROTOCOL,
    CHECKPOINT_FORMAT,
    GATE_PROTOCOL,
    OPTIMIZER_SEED,
    PMP_TRAIN_SEED,
    PMP_VALIDATION_SEED,
    REQUIRED_PROTECTED_RADII,
    RESERVED_BLIND_SEED as ARTIFACT_LEGACY_RESERVED_SEED,
    STANDARD_BASE_GRID_PROTOCOL,
    STRUCTURAL_GUARD_PROTOCOL,
    load_gated_feedback_checkpoint,
)
from scripts.evaluate_compact_exact_gated_one_shot_blind import (  # noqa: E402
    BOOTSTRAP_REPEATS,
    BOOTSTRAP_SEED,
    CANONICAL_CONSUMPTION_LEDGER,
    FORMAL_COUNT,
    FORMAL_RADIUS,
    FORMAL_SEED,
    ConsumptionLedger,
    canonical_continuous_objective,
    canonical_continuous_rollout,
    canonical_continuous_trace,
    manifest_sha256,
    paired_bootstrap_mean,
    require_sha256,
    sha256_file,
    tensor_commitment,
    write_json_exclusive,
)
from scripts.evaluate_feedback_section5 import (  # noqa: E402
    load_feedback_checkpoint,
)
from train_paper_pmp_kkt import build_params  # noqa: E402


PROTOCOL = "full_gated_teacher_one_shot_blind_v1"
IDENTITY_RADII = (0.0, 0.10, 0.20)

IDENTITY_KEYS = (
    "source_stage_gate_max_abs",
    "teacher_stage_gate_max_abs",
    "source_node_gate_max_abs",
    "teacher_node_gate_max_abs",
    "external_source_vs_internal_locked_time_logit_max_abs",
    "external_source_vs_internal_locked_stage_control_max_abs",
    "external_source_vs_internal_locked_stage_state_max_abs",
    "external_source_vs_internal_locked_terminal_state_max_abs",
    "external_source_vs_internal_locked_node_state_max_abs",
    "external_source_vs_internal_locked_objective_max_abs",
    "internal_locked_vs_teacher_time_logit_max_abs",
    "internal_locked_vs_teacher_stage_control_max_abs",
    "internal_locked_vs_teacher_stage_state_max_abs",
    "internal_locked_vs_teacher_terminal_state_max_abs",
    "internal_locked_vs_teacher_node_state_max_abs",
    "internal_locked_vs_teacher_objective_max_abs",
    "external_source_vs_teacher_stage_control_max_abs",
    "external_source_vs_teacher_stage_state_max_abs",
    "external_source_vs_teacher_terminal_state_max_abs",
    "external_source_vs_teacher_node_state_max_abs",
    "external_source_vs_teacher_objective_max_abs",
)

EVALUATOR_DEPENDENCIES = (
    "scripts/evaluate_compact_exact_gated_one_shot_blind.py",
    "scripts/continue_feedback_full_state_gated_fallback.py",
    "scripts/evaluate_feedback_section5.py",
    "scripts/feedback_continuous_policy_rk4.py",
    "scripts/feedback_section5_rk4_reference.py",
    "scripts/refine_feedback_offgrid_scalar.py",
    "scripts/train_feedback_section5.py",
    "train_paper_pmp_kkt.py",
)


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def evaluator_code_manifest() -> dict[str, str]:
    relative_paths = (
        "scripts/evaluate_full_gated_teacher_one_shot_blind.py",
        *EVALUATOR_DEPENDENCIES,
    )
    return {
        relative: sha256_file(ROOT / relative)
        for relative in relative_paths
    }


def _absolute_without_resolving_symlinks(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _validate_paths(
    artifact: Path,
    locked: Path,
    output: Path,
    ledger: Path,
) -> None:
    if not artifact.is_file():
        raise FileNotFoundError(f"artifact is not a regular file: {artifact}")
    if not locked.is_file():
        raise FileNotFoundError(
            f"locked source is not a regular file: {locked}"
        )
    if len({artifact, locked, output, ledger}) != 4:
        raise ValueError(
            "artifact, locked source, output, and ledger paths must be distinct"
        )
    if _path_exists(output):
        raise FileExistsError(
            f"refusing to overwrite existing output: {output}"
        )
    if _path_exists(ledger):
        raise FileExistsError(
            f"reserved seed already has a consumption ledger: {ledger}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    ledger.parent.mkdir(parents=True, exist_ok=True)


def _load_gated_snapshot(
    data: bytes,
) -> tuple[torch.nn.Module, Any, Any, dict[str, Any]]:
    with tempfile.TemporaryDirectory(
        prefix="full_gated_blind_artifact_snapshot_"
    ) as directory:
        snapshot = Path(directory) / "artifact.pt"
        with snapshot.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        return load_gated_feedback_checkpoint(snapshot)


def _load_locked_snapshot(
    data: bytes,
) -> tuple[torch.nn.Module, Any, Any]:
    with tempfile.TemporaryDirectory(
        prefix="full_gated_blind_locked_snapshot_"
    ) as directory:
        snapshot = Path(directory) / "locked.pt"
        with snapshot.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        return load_feedback_checkpoint(snapshot)


def validate_formal_artifact_metadata(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if payload.get("checkpoint_format") != CHECKPOINT_FORMAT:
        raise RuntimeError("unsupported full-gated checkpoint format")
    gate = payload.get("gated_fallback")
    if not isinstance(gate, Mapping):
        raise RuntimeError("full-gated protocol metadata is missing")

    expected = {
        "centered_state_logits": True,
        "shared_time_branch": True,
        "gate": GATE_PROTOCOL,
        "structural_guard": STRUCTURAL_GUARD_PROTOCOL,
        "anchor_families": ANCHOR_FAMILY_PROTOCOL,
        "base_n": 800,
        "standard_base_grid": STANDARD_BASE_GRID_PROTOCOL,
        "formal_protocol": True,
        "protected_tolerance": 0.0,
        "source_weights_loaded_without_dtype_roundtrip": True,
        "train_scope": "probe_state_branch_only",
        "direct_supervision_used_in_cleanup": False,
        "physical_objective_used_in_loss": False,
        "physical_objective_used_in_selection": False,
        "posthoc_identity_audit": True,
        "posthoc_identity_audit_completed": True,
        "train_seed": PMP_TRAIN_SEED,
        "validation_seed": PMP_VALIDATION_SEED,
        "optimizer_seed": OPTIMIZER_SEED,
        "reserved_blind_seed": ARTIFACT_LEGACY_RESERVED_SEED,
        "reserved_blind_used_for_training_or_selection": False,
    }
    wrong = {
        key: (gate.get(key), value)
        for key, value in expected.items()
        if gate.get(key) != value
    }
    if wrong:
        raise RuntimeError(f"full-gated artifact protocol mismatch: {wrong}")

    raw_radii = gate.get("protected_radii")
    if not isinstance(raw_radii, (list, tuple)):
        raise RuntimeError("artifact protected radii are missing")
    if tuple(float(value) for value in raw_radii) != REQUIRED_PROTECTED_RADII:
        raise RuntimeError(
            "artifact protected radii do not match the full-gated contract"
        )
    posthoc = gate.get("posthoc_protected_identity")
    if not isinstance(posthoc, Mapping) or not posthoc:
        raise RuntimeError("artifact posthoc identity audit is missing")
    nonzero = {
        str(key): value
        for key, value in posthoc.items()
        if type(value) not in (int, float) or float(value) != 0.0
    }
    if nonzero:
        raise RuntimeError(
            f"artifact posthoc identity audit is not exact: {nonzero}"
        )

    locked = gate.get("locked_checkpoint")
    if not isinstance(locked, Mapping):
        raise RuntimeError("artifact locked-source provenance is missing")
    locked_hash = require_sha256(
        locked.get("sha256"),
        label="artifact locked-source SHA256",
    )
    locked_path = locked.get("path")
    if not isinstance(locked_path, str) or not locked_path:
        raise RuntimeError("artifact locked-source path is missing")
    reference = gate.get("posthoc_identity_reference")
    if (
        not isinstance(reference, Mapping)
        or reference.get("sha256") != locked_hash
        or reference.get("evaluator") != STANDARD_BASE_GRID_PROTOCOL
    ):
        raise RuntimeError(
            "artifact posthoc identity reference is inconsistent"
        )

    prior_seeds = {
        "train_seed": gate.get("train_seed"),
        "validation_seed": gate.get("validation_seed"),
        "optimizer_seed": gate.get("optimizer_seed"),
        "artifact_legacy_reserved_seed": gate.get("reserved_blind_seed"),
    }
    reused = sorted(
        key
        for key, value in prior_seeds.items()
        if type(value) is int and value == FORMAL_SEED
    )
    if reused:
        raise RuntimeError(
            f"new reserved seed {FORMAL_SEED} appears in prior artifact "
            f"fields: {reused}"
        )

    return {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "formal_protocol": True,
        "protected_radii": list(REQUIRED_PROTECTED_RADII),
        "posthoc_protected_identity": dict(posthoc),
        "locked_checkpoint_recorded_path": locked_path,
        "locked_checkpoint_sha256": locked_hash,
        "artifact_legacy_reserved_seed": ARTIFACT_LEGACY_RESERVED_SEED,
        "artifact_legacy_reserved_seed_used": False,
        "new_one_shot_reserved_seed": FORMAL_SEED,
        "new_one_shot_reserved_seed_absent_from_prior_artifact_seeds": True,
    }


def _assert_equal_tensor_dicts(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
    *,
    label: str,
) -> None:
    if set(left) != set(right):
        raise RuntimeError(f"{label} tensor keys differ")
    unequal = [
        key
        for key in sorted(left)
        if (
            left[key].dtype != right[key].dtype
            or left[key].device != right[key].device
            or not torch.equal(left[key], right[key])
        )
    ]
    if unequal:
        raise RuntimeError(f"{label} tensors differ: {unequal[:5]}")


def verify_locked_source(
    teacher: torch.nn.Module,
    teacher_cfg: Any,
    locked: torch.nn.Module,
    locked_cfg: Any,
) -> None:
    if teacher_cfg != locked_cfg:
        raise RuntimeError("locked source problem differs from teacher")
    scalar_attributes = (
        "m",
        "umax",
        "state_log_scale",
        "correction_gain",
        "center_state_correction",
        "action_temperature",
        "action_scale",
        "action_offset",
        "action_parameterization",
        "state_feature_mode",
    )
    wrong_attributes = {
        name: (getattr(teacher, name), getattr(locked, name))
        for name in scalar_attributes
        if getattr(teacher, name) != getattr(locked, name)
    }
    if wrong_attributes:
        raise RuntimeError(
            "locked source architecture/action metadata differs from teacher: "
            f"{wrong_attributes}"
        )
    _assert_equal_tensor_dicts(
        teacher.time_branch.state_dict(),
        locked.time_branch.state_dict(),
        label="locked time branch",
    )
    _assert_equal_tensor_dicts(
        teacher.state_branch.state_dict(),
        locked.state_branch.state_dict(),
        label="locked CF state branch",
    )
    if not torch.equal(
        teacher.nominal_reference,
        locked.nominal_reference,
    ):
        raise RuntimeError("locked nominal reference differs from teacher")


def _node_states(trace: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        (
            trace["states"][:, :, 0, :],
            trace["terminal_state"].unsqueeze(1),
        ),
        dim=1,
    )


def _gate_values(
    teacher: torch.nn.Module,
    trace: Mapping[str, torch.Tensor],
    cfg: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = trace["states"].shape[0]
    stage_time = (
        trace["normalized_time"]
        .unsqueeze(0)
        .expand(batch, -1, -1)
        .reshape(-1)
    )
    stage_gate = teacher.gate_values(
        stage_time,
        trace["states"].reshape(-1, int(cfg.m)),
    )
    nodes = _node_states(trace)
    node_time = torch.linspace(
        0.0,
        1.0,
        int(cfg.n) + 1,
        device=nodes.device,
        dtype=nodes.dtype,
    )
    node_gate = teacher.gate_values(
        node_time.repeat(batch),
        nodes.reshape(-1, int(cfg.m)),
    )
    return stage_gate, node_gate


def _comparison_metrics(
    prefix: str,
    left_trace: Mapping[str, torch.Tensor],
    right_trace: Mapping[str, torch.Tensor],
    left_logits: torch.Tensor,
    right_logits: torch.Tensor,
    cfg: Any,
    params: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    left_nodes = _node_states(left_trace)
    right_nodes = _node_states(right_trace)
    left_objective = canonical_continuous_objective(
        left_trace, cfg, params
    )
    right_objective = canonical_continuous_objective(
        right_trace, cfg, params
    )
    return {
        f"{prefix}_time_logit_max_abs": float(
            (left_logits - right_logits).abs().max().cpu()
        ),
        f"{prefix}_stage_control_max_abs": float(
            (left_trace["controls"] - right_trace["controls"])
            .abs()
            .max()
            .cpu()
        ),
        f"{prefix}_stage_state_max_abs": float(
            (left_trace["states"] - right_trace["states"])
            .abs()
            .max()
            .cpu()
        ),
        f"{prefix}_terminal_state_max_abs": float(
            (
                left_trace["terminal_state"]
                - right_trace["terminal_state"]
            )
            .abs()
            .max()
            .cpu()
        ),
        f"{prefix}_node_state_max_abs": float(
            (left_nodes - right_nodes).abs().max().cpu()
        ),
        f"{prefix}_objective_max_abs": float(
            (left_objective - right_objective).abs().max().cpu()
        ),
    }


@torch.no_grad()
def protected_identity_metrics(
    locked: torch.nn.Module,
    teacher: torch.nn.Module,
    initial: torch.Tensor,
    cfg: Any,
    params: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    source_trace = canonical_continuous_trace(
        locked,
        initial,
        cfg,
        params,
        state_mode="feedback",
    )
    internal_trace = canonical_continuous_trace(
        teacher,
        initial,
        cfg,
        params,
        state_mode="locked_feedback",
    )
    teacher_trace = canonical_continuous_trace(
        teacher,
        initial,
        cfg,
        params,
        state_mode="feedback",
    )
    source_stage_gate, source_node_gate = _gate_values(
        teacher, source_trace, cfg
    )
    teacher_stage_gate, teacher_node_gate = _gate_values(
        teacher, teacher_trace, cfg
    )
    normalized_time = torch.linspace(
        0.0,
        1.0,
        int(cfg.n) + 1,
        device=initial.device,
        dtype=initial.dtype,
    )
    source_logits = locked.time_logits(normalized_time)
    internal_logits = teacher.time_logits(normalized_time)
    teacher_logits = teacher.time_logits(normalized_time)
    result = {
        "source_stage_gate_max_abs": float(
            source_stage_gate.abs().max().cpu()
        ),
        "teacher_stage_gate_max_abs": float(
            teacher_stage_gate.abs().max().cpu()
        ),
        "source_node_gate_max_abs": float(
            source_node_gate.abs().max().cpu()
        ),
        "teacher_node_gate_max_abs": float(
            teacher_node_gate.abs().max().cpu()
        ),
        **_comparison_metrics(
            "external_source_vs_internal_locked",
            source_trace,
            internal_trace,
            source_logits,
            internal_logits,
            cfg,
            params,
        ),
        **_comparison_metrics(
            "internal_locked_vs_teacher",
            internal_trace,
            teacher_trace,
            internal_logits,
            teacher_logits,
            cfg,
            params,
        ),
    }
    direct = _comparison_metrics(
        "external_source_vs_teacher",
        source_trace,
        teacher_trace,
        source_logits,
        teacher_logits,
        cfg,
        params,
    )
    direct.pop("external_source_vs_teacher_time_logit_max_abs")
    result.update(direct)
    return result


def assert_exact_protected_identity(
    identity: Mapping[str, Any],
) -> None:
    wrong = {
        name: identity.get(name)
        for name in IDENTITY_KEYS
        if (
            type(identity.get(name)) not in (int, float)
            or float(identity[name]) != 0.0
        )
    }
    if wrong:
        raise RuntimeError(
            f"protected identity is not bitwise exact: {wrong}"
        )


def _formal_initial_states(cfg: Any) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(FORMAL_SEED)
    directions = (
        2.0
        * torch.rand(
            FORMAL_COUNT,
            int(cfg.m),
            generator=generator,
            device="cpu",
            dtype=torch.float64,
        )
        - 1.0
    )
    return float(cfg.n0) * (1.0 + FORMAL_RADIUS * directions)


def _as_finite_vector(
    value: torch.Tensor,
    *,
    label: str,
) -> torch.Tensor:
    detached = value.detach().cpu().to(dtype=torch.float64)
    if detached.ndim != 1 or detached.numel() == 0:
        raise ValueError(f"{label} must be a nonempty vector")
    if not torch.isfinite(detached).all():
        raise ValueError(f"{label} must be finite")
    return detached


def _summary(value: torch.Tensor) -> dict[str, float]:
    value = _as_finite_vector(value, label="summary vector")
    sample_sd = (
        value.std(unbiased=True)
        if value.numel() > 1
        else torch.zeros((), dtype=value.dtype)
    )
    return {
        "mean": float(value.mean()),
        "sample_sd": float(sample_sd),
        "minimum": float(value.min()),
        "maximum": float(value.max()),
    }


def per_sample_records(
    teacher: torch.Tensor,
    locked_source: torch.Tensor,
    frozen_time: torch.Tensor,
) -> list[dict[str, float | int]]:
    teacher = _as_finite_vector(teacher, label="teacher J")
    locked_source = _as_finite_vector(
        locked_source, label="locked-source J"
    )
    frozen_time = _as_finite_vector(
        frozen_time, label="frozen-time J"
    )
    if not (
        teacher.shape == locked_source.shape == frozen_time.shape
    ):
        raise ValueError("physical-J vectors must have identical shapes")
    rows: list[dict[str, float | int]] = []
    for index in range(teacher.numel()):
        teacher_j = float(teacher[index])
        locked_j = float(locked_source[index])
        frozen_j = float(frozen_time[index])
        rows.append(
            {
                "index": index,
                "teacher_physical_J": teacher_j,
                "locked_cf_source_physical_J": locked_j,
                "frozen_time_physical_J": frozen_j,
                "primary_delta_frozen_time_minus_teacher": (
                    frozen_j - teacher_j
                ),
                "secondary_delta_locked_cf_source_minus_teacher": (
                    locked_j - teacher_j
                ),
                "delta_locked_cf_source_minus_frozen_time": (
                    locked_j - frozen_j
                ),
            }
        )
    return rows


def _formal_protocol() -> dict[str, Any]:
    return {
        "seed": FORMAL_SEED,
        "count": FORMAL_COUNT,
        "radius": FORMAL_RADIUS,
        "sampler": (
            "iid componentwise Uniform[-1,1] directions generated on CPU "
            "with torch.float64 and torch.Generator.manual_seed; "
            "initial=cfg.n0*(1+radius*direction)"
        ),
        "antithetic": False,
        "canonical_evaluator": (
            "continuous_policy_RK4_four_stage_requery_with_RK4_running_cost"
        ),
        "teacher_state_mode": "feedback",
        "locked_cf_source_state_mode": "feedback",
        "frozen_time_state_mode": "w_zero",
        "primary_delta": "frozen_time_minus_teacher",
        "secondary_delta": "locked_cf_source_minus_teacher",
        "physical_scale_factor": "1/cfg.alpha",
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "bootstrap_seeds": {
            "primary_frozen_time_minus_teacher": BOOTSTRAP_SEED,
            "secondary_locked_cf_source_minus_teacher": BOOTSTRAP_SEED + 1,
        },
        "acceptance_rule": (
            "both paired-bootstrap lower_95 bounds must be strictly positive"
        ),
    }


def _runtime_environment(
    device: torch.device,
    *,
    threads: int,
) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "device": str(device),
        "threads": threads,
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False

    artifact_path = args.artifact.expanduser().resolve()
    locked_path = args.locked_checkpoint.expanduser().resolve()
    output_path = _absolute_without_resolving_symlinks(args.output)
    ledger_path = _absolute_without_resolving_symlinks(
        CANONICAL_CONSUMPTION_LEDGER
    )
    _validate_paths(
        artifact_path,
        locked_path,
        output_path,
        ledger_path,
    )

    expected_artifact_hash = require_sha256(
        args.expected_artifact_sha256,
        label="expected artifact SHA256",
    )
    expected_locked_hash = require_sha256(
        args.expected_locked_sha256,
        label="expected locked-source SHA256",
    )
    expected_evaluator_hash = require_sha256(
        args.expected_evaluator_sha256,
        label="expected evaluator SHA256",
    )
    expected_manifest_hash = require_sha256(
        args.expected_code_manifest_sha256,
        label="expected evaluator code-manifest SHA256",
    )
    code_manifest_before = evaluator_code_manifest()
    code_manifest_hash = manifest_sha256(code_manifest_before)
    evaluator_relative_path = (
        "scripts/evaluate_full_gated_teacher_one_shot_blind.py"
    )
    if code_manifest_before[evaluator_relative_path] != expected_evaluator_hash:
        raise RuntimeError("one-shot evaluator SHA256 mismatch")
    if code_manifest_hash != expected_manifest_hash:
        raise RuntimeError("one-shot evaluator code-manifest SHA256 mismatch")

    artifact_bytes = artifact_path.read_bytes()
    artifact_hash_before = hashlib.sha256(artifact_bytes).hexdigest()
    if artifact_hash_before != expected_artifact_hash:
        raise RuntimeError("full-gated teacher artifact SHA256 mismatch")
    locked_bytes = locked_path.read_bytes()
    locked_hash_before = hashlib.sha256(locked_bytes).hexdigest()
    if locked_hash_before != expected_locked_hash:
        raise RuntimeError("locked CF source SHA256 mismatch")

    teacher, cfg, _, payload = _load_gated_snapshot(artifact_bytes)
    locked, locked_cfg, _ = _load_locked_snapshot(locked_bytes)
    del artifact_bytes
    del locked_bytes
    artifact_protocol = validate_formal_artifact_metadata(payload)
    if artifact_protocol["locked_checkpoint_sha256"] != expected_locked_hash:
        raise RuntimeError(
            "CLI locked-source SHA256 differs from artifact provenance"
        )
    verify_locked_source(teacher, cfg, locked, locked_cfg)
    if sha256_file(artifact_path) != artifact_hash_before:
        raise RuntimeError("teacher artifact changed while it was loaded")
    if sha256_file(locked_path) != locked_hash_before:
        raise RuntimeError("locked source changed while it was loaded")

    device = torch.device(args.device)
    teacher.to(device=device, dtype=torch.float64).eval()
    locked.to(device=device, dtype=torch.float64).eval()
    params = build_params(cfg, device, torch.float64)
    teacher.set_feature_vectors(params["r"], params["phi"])
    locked.set_feature_vectors(params["r"], params["phi"])

    direction = torch.linspace(
        -1.0,
        1.0,
        int(cfg.m),
        device=device,
        dtype=torch.float64,
    )
    protected_initial = torch.stack(
        [
            float(cfg.n0) * (1.0 + radius * direction)
            for radius in IDENTITY_RADII
        ],
        dim=0,
    )
    identity = protected_identity_metrics(
        locked,
        teacher,
        protected_initial,
        cfg,
        params,
    )
    assert_exact_protected_identity(identity)

    if sha256_file(artifact_path) != artifact_hash_before:
        raise RuntimeError("teacher artifact changed during identity audit")
    if sha256_file(locked_path) != locked_hash_before:
        raise RuntimeError("locked source changed during identity audit")
    if evaluator_code_manifest() != code_manifest_before:
        raise RuntimeError("evaluator code changed during preflight")

    ledger: ConsumptionLedger | None = None
    try:
        ledger = ConsumptionLedger.reserve(
            ledger_path,
            {
                "event": "started",
                "protocol": PROTOCOL,
                "seed": FORMAL_SEED,
                "teacher_artifact": str(artifact_path),
                "teacher_artifact_sha256": artifact_hash_before,
                "locked_cf_source": str(locked_path),
                "locked_cf_source_sha256": locked_hash_before,
                "evaluator_sha256": expected_evaluator_hash,
                "evaluator_code_manifest_sha256": code_manifest_hash,
                "evaluator_code_manifest": code_manifest_before,
                "output": str(output_path),
                "status": "reserved_seed_consumed_on_ledger_creation",
            },
        )

        initial_cpu = _formal_initial_states(cfg)
        commitment = tensor_commitment(initial_cpu)
        ledger.append(
            {
                "event": "initial_states_committed",
                "seed": FORMAL_SEED,
                "initial_state_commitment": commitment,
            }
        )
        initial = initial_cpu.to(device=device, dtype=torch.float64)
        if tensor_commitment(initial)["sha256"] != commitment["sha256"]:
            raise RuntimeError("initial states changed during device transfer")

        with torch.no_grad():
            teacher_rollout = canonical_continuous_rollout(
                teacher,
                initial,
                cfg,
                params,
                state_mode="feedback",
            )
            locked_rollout = canonical_continuous_rollout(
                locked,
                initial,
                cfg,
                params,
                state_mode="feedback",
            )
            frozen_rollout = canonical_continuous_rollout(
                locked,
                initial,
                cfg,
                params,
                state_mode="w_zero",
            )

        physical_scale = 1.0 / float(cfg.alpha)
        teacher_j = _as_finite_vector(
            teacher_rollout.objectives * physical_scale,
            label="teacher J",
        )
        locked_j = _as_finite_vector(
            locked_rollout.objectives * physical_scale,
            label="locked-source J",
        )
        frozen_j = _as_finite_vector(
            frozen_rollout.objectives * physical_scale,
            label="frozen-time J",
        )
        if not (
            teacher_j.numel()
            == locked_j.numel()
            == frozen_j.numel()
            == FORMAL_COUNT
        ):
            raise RuntimeError("formal evaluator returned the wrong sample count")

        primary_delta = frozen_j - teacher_j
        secondary_delta = locked_j - teacher_j
        primary_bootstrap = paired_bootstrap_mean(
            primary_delta,
            seed=BOOTSTRAP_SEED,
            repeats=BOOTSTRAP_REPEATS,
        )
        secondary_bootstrap = paired_bootstrap_mean(
            secondary_delta,
            seed=BOOTSTRAP_SEED + 1,
            repeats=BOOTSTRAP_REPEATS,
        )

        artifact_hash_after = sha256_file(artifact_path)
        locked_hash_after = sha256_file(locked_path)
        if artifact_hash_after != artifact_hash_before:
            raise RuntimeError(
                "teacher artifact changed during formal evaluation"
            )
        if locked_hash_after != locked_hash_before:
            raise RuntimeError(
                "locked source changed during formal evaluation"
            )
        code_manifest_after = evaluator_code_manifest()
        if code_manifest_after != code_manifest_before:
            raise RuntimeError("evaluator code changed during formal evaluation")

        acceptance = {
            "primary_frozen_time_minus_teacher_lower_95_positive": (
                primary_bootstrap["lower_95"] > 0.0
            ),
            "secondary_locked_cf_source_minus_teacher_lower_95_positive": (
                secondary_bootstrap["lower_95"] > 0.0
            ),
        }
        acceptance["passed"] = all(acceptance.values())
        result = {
            "protocol": PROTOCOL,
            "mode": "formal_one_shot_blind",
            "created_utc": utc_now(),
            "teacher_artifact": {
                "path": str(artifact_path),
                "expected_sha256": expected_artifact_hash,
                "sha256_before": artifact_hash_before,
                "sha256_after": artifact_hash_after,
                **artifact_protocol,
            },
            "locked_cf_source": {
                "path": str(locked_path),
                "expected_sha256": expected_locked_hash,
                "sha256_before": locked_hash_before,
                "sha256_after": locked_hash_after,
                "matches_teacher_embedded_locked_branch": True,
            },
            "evaluator": {
                "path": str(Path(__file__).resolve()),
                "expected_sha256": expected_evaluator_hash,
                "sha256": code_manifest_before[evaluator_relative_path],
                "expected_code_manifest_sha256": expected_manifest_hash,
                "code_manifest_sha256": code_manifest_hash,
                "code_manifest_files": code_manifest_before,
                "runtime": _runtime_environment(
                    device,
                    threads=args.threads,
                ),
            },
            "evaluation": _formal_protocol(),
            "consumption": {
                "ledger": str(ledger_path),
                "reservation_precedes_initial_state_materialization": True,
                "reserved_seed_consumed_by_this_evaluation": True,
                "failed_runs_remain_consumed": True,
            },
            "initial_state_commitment": commitment,
            "used_for_training": False,
            "used_for_checkpoint_selection": False,
            "used_for_hyperparameter_tuning": False,
            "protected_identity": identity,
            "aggregate": {
                "teacher_physical_J": _summary(teacher_j),
                "locked_cf_source_physical_J": _summary(locked_j),
                "frozen_time_physical_J": _summary(frozen_j),
                "primary_frozen_time_minus_teacher": {
                    **_summary(primary_delta),
                    "positive_fraction": float(
                        (primary_delta > 0.0).double().mean()
                    ),
                },
                "secondary_locked_cf_source_minus_teacher": {
                    **_summary(secondary_delta),
                    "positive_fraction": float(
                        (secondary_delta > 0.0).double().mean()
                    ),
                },
            },
            "paired_bootstrap": {
                "primary_frozen_time_minus_teacher": primary_bootstrap,
                "secondary_locked_cf_source_minus_teacher": (
                    secondary_bootstrap
                ),
            },
            "acceptance": acceptance,
            "per_sample": per_sample_records(
                teacher_j,
                locked_j,
                frozen_j,
            ),
        }
        output_hash = write_json_exclusive(output_path, result)
        ledger.append(
            {
                "event": "completed",
                "seed": FORMAL_SEED,
                "teacher_artifact_sha256": artifact_hash_after,
                "locked_cf_source_sha256": locked_hash_after,
                "evaluator_sha256": expected_evaluator_hash,
                "evaluator_code_manifest_sha256": code_manifest_hash,
                "initial_state_sha256": commitment["sha256"],
                "output": str(output_path),
                "output_sha256": output_hash,
                "acceptance_passed": acceptance["passed"],
            }
        )
        ledger.close_read_only()
        ledger = None
    except BaseException as error:
        if ledger is not None:
            try:
                ledger.append(
                    {
                        "event": "failed",
                        "seed": FORMAL_SEED,
                        "exception_type": type(error).__name__,
                        "exception": str(error),
                        "status": (
                            "reserved seed remains permanently consumed"
                        ),
                    }
                )
            finally:
                ledger.close_read_only()
        raise

    print(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "teacher_artifact_sha256": artifact_hash_before,
                "locked_cf_source_sha256": locked_hash_before,
                "evaluator_sha256": expected_evaluator_hash,
                "evaluator_code_manifest_sha256": code_manifest_hash,
                "output": str(output_path),
                "output_sha256": output_hash,
                "consumption_ledger": str(ledger_path),
                "acceptance_passed": result["acceptance"]["passed"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Canonical consumption ledger (not CLI-adjustable): "
            f"{CANONICAL_CONSUMPTION_LEDGER}"
        ),
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--locked-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-artifact-sha256", required=True)
    parser.add_argument("--expected-locked-sha256", required=True)
    parser.add_argument("--expected-evaluator-sha256", required=True)
    parser.add_argument("--expected-code-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device",
        required=True,
        help="Explicit torch device for the one-shot run, e.g. cuda:0 or cpu.",
    )
    parser.add_argument("--threads", type=int, default=8)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
