#!/usr/bin/env python3
"""Formal one-shot blind evaluator for a frozen compact exact-gated adapter.

The formal protocol is intentionally not configurable: it uses 128 iid
componentwise-random states at radius 0.20 with reserved seed 20261701 and a
20,000-repeat paired bootstrap.  Frozen artifact, evaluator, and direct code-
dependency SHA256 commitments are required.  Before the reserved states are
generated, an exclusive consumption ledger is created; a failed run therefore
still consumes the reserved seed.  Existing ledgers and result files are never
overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import tempfile
from dataclasses import dataclass
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
    continuous_policy_stage_trace,
)
from scripts.feedback_continuous_policy_rk4 import (  # noqa: E402
    pchip_midpoint_logits,
)
from train_paper_pmp_kkt import build_params  # noqa: E402


PROTOCOL = "compact_exact_gated_one_shot_blind_v1"
EXPECTED_ARTIFACT_FORMAT = "exact_gated_compact_residual_adapter_v1"
REQUIRED_PROTECTED_RADII = (0.0, 0.10, 0.20)
FORMAL_SEED = 202_617_01
FORMAL_COUNT = 128
FORMAL_RADIUS = 0.20
BOOTSTRAP_REPEATS = 20_000
BOOTSTRAP_SEED = 202_617_02
INITIAL_STATE_HASH_PROTOCOL = (
    "sha256_shape_float64_le_c_contiguous_bytes_v1"
)
CANONICAL_CONSUMPTION_LEDGER = (
    ROOT
    / "outputs"
    / "formal_blind_consumption"
    / f"one_shot_blind_seed{FORMAL_SEED}.jsonl"
)

IDENTITY_KEYS = (
    "source_stage_gate_max_abs",
    "candidate_stage_gate_max_abs",
    "source_node_gate_max_abs",
    "candidate_node_gate_max_abs",
    "node_time_logit_max_abs",
    "stage_control_max_abs",
    "stage_state_max_abs",
    "terminal_state_max_abs",
    "node_state_max_abs",
    "objective_max_abs",
)

EVALUATOR_DEPENDENCIES = (
    "scripts/compact_exact_gated_residual_adapter.py",
    "scripts/continue_feedback_full_state_gated_fallback.py",
    "scripts/feedback_continuous_policy_rk4.py",
    "scripts/feedback_section5_rk4_reference.py",
    "scripts/refine_feedback_gated_residual_adapter.py",
    "scripts/refine_feedback_svd_null_projected_kkt.py",
    "scripts/train_feedback_section5.py",
    "train_paper_pmp_kkt.py",
)


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: str, *, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _path_exists(path: Path) -> bool:
    """Return true for ordinary files, directories, and broken symlinks."""

    return os.path.lexists(path)


def _absolute_without_resolving_symlinks(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    """Create a JSON file atomically with respect to competing creators."""

    data = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    expected_hash = hashlib.sha256(data).hexdigest()
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise RuntimeError("exclusive result file failed SHA256 readback")
    return actual_hash


class ConsumptionLedger:
    """An append-only-in-process, exclusive one-shot consumption record."""

    def __init__(self, path: Path, stream: Any) -> None:
        self.path = path
        self._stream = stream
        self._closed = False

    @classmethod
    def reserve(
        cls,
        path: Path,
        record: Mapping[str, Any],
    ) -> "ConsumptionLedger":
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"utc": utc_now(), **dict(record)}
        data = (
            json.dumps(entry, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        stream = os.fdopen(descriptor, "ab", buffering=0)
        try:
            stream.write(data)
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            # Never remove a claimed ledger: even reservation failure blocks
            # accidental reuse until a human audit resolves it.
            raise
        return cls(path, stream)

    def append(self, record: Mapping[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("consumption ledger is already closed")
        entry = {"utc": utc_now(), **dict(record)}
        data = (
            json.dumps(entry, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        self._stream.write(data)
        os.fsync(self._stream.fileno())

    def close_read_only(self) -> None:
        if self._closed:
            return
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stream.close()
        self._closed = True
        os.chmod(self.path, 0o444)


def evaluator_code_manifest() -> dict[str, str]:
    relative_paths = (
        "scripts/evaluate_compact_exact_gated_one_shot_blind.py",
        *EVALUATOR_DEPENDENCIES,
    )
    return {
        relative: sha256_file(ROOT / relative)
        for relative in relative_paths
    }


def manifest_sha256(manifest: Mapping[str, str]) -> str:
    encoded = json.dumps(
        dict(manifest),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def validate_formal_artifact_metadata(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if payload.get("format") != EXPECTED_ARTIFACT_FORMAT:
        raise RuntimeError("unsupported compact exact-gated artifact format")

    raw_radii = payload.get("protected_radii")
    if not isinstance(raw_radii, (list, tuple)):
        raise RuntimeError("artifact protected radii are missing")
    if any(type(value) not in (int, float) for value in raw_radii):
        raise RuntimeError("artifact protected radii must be numeric")
    radii = tuple(float(value) for value in raw_radii)
    if radii != REQUIRED_PROTECTED_RADII:
        raise RuntimeError("artifact protected radii must be exactly 0,.10,.20")

    stored_identity = payload.get("protected_identity")
    required_stored_identity = (
        "gate_max_abs",
        "control_max_abs",
        "state_max_abs",
        "objective_max_abs",
    )
    if not isinstance(stored_identity, Mapping):
        raise RuntimeError("artifact protected identity audit is missing")
    wrong_identity = {
        name: stored_identity.get(name)
        for name in required_stored_identity
        if (
            type(stored_identity.get(name)) not in (int, float)
            or float(stored_identity[name]) != 0.0
        )
    }
    if wrong_identity:
        raise RuntimeError(
            "artifact protected identity audit is not exact: "
            f"{wrong_identity}"
        )

    seeds = payload.get("seeds")
    if not isinstance(seeds, Mapping):
        raise RuntimeError("artifact seed provenance is missing")
    reserved = seeds.get("reserved_final_blind")
    if type(reserved) is not int or reserved != FORMAL_SEED:
        raise RuntimeError(
            f"artifact must reserve final blind seed {FORMAL_SEED}"
        )
    if seeds.get("reserved_final_blind_used") is not False:
        raise RuntimeError("artifact reserved final blind seed was already used")
    reused = sorted(
        str(name)
        for name, value in seeds.items()
        if name
        not in ("reserved_final_blind", "reserved_final_blind_used")
        and type(value) is int
        and value == FORMAL_SEED
    )
    if reused:
        raise RuntimeError(
            "reserved final blind seed also appears in prior protocol fields: "
            f"{reused}"
        )

    source_hash = require_sha256(
        payload.get("source_checkpoint_sha256"),
        label="artifact source checkpoint SHA256",
    )
    source_path = payload.get("source_checkpoint")
    if not isinstance(source_path, str) or not source_path:
        raise RuntimeError("artifact source checkpoint provenance is missing")

    return {
        "format": EXPECTED_ARTIFACT_FORMAT,
        "protected_radii": list(REQUIRED_PROTECTED_RADII),
        "source_checkpoint": source_path,
        "source_checkpoint_sha256": source_hash,
        "stored_protected_identity": {
            name: float(stored_identity[name])
            for name in required_stored_identity
        },
        "reserved_final_blind_seed": FORMAL_SEED,
        "reserved_final_blind_used_before_evaluation": False,
    }


def tensor_commitment(tensor: torch.Tensor) -> dict[str, Any]:
    if tensor.dtype != torch.float64:
        raise ValueError("tensor commitment requires float64 input")
    if tensor.ndim < 1:
        raise ValueError("tensor commitment requires a non-scalar tensor")
    array = tensor.detach().cpu().contiguous().numpy()
    array = array.astype(np.dtype("<f8"), copy=False)
    metadata = {
        "shape": list(array.shape),
        "dtype": "float64-le",
        "order": "C",
    }
    digest = hashlib.sha256()
    digest.update(INITIAL_STATE_HASH_PROTOCOL.encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\0")
    raw = array.tobytes(order="C")
    digest.update(raw)
    return {
        "protocol": INITIAL_STATE_HASH_PROTOCOL,
        **metadata,
        "byte_count": len(raw),
        "sha256": digest.hexdigest(),
    }


def _formal_initial_states(cfg: Any) -> torch.Tensor:
    """Materialize the reserved iid componentwise states on CPU exactly once."""

    generator = torch.Generator(device="cpu").manual_seed(FORMAL_SEED)
    direction = (
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
    return float(cfg.n0) * (1.0 + FORMAL_RADIUS * direction)


@dataclass(frozen=True)
class ContinuousRollout:
    states: torch.Tensor
    controls: torch.Tensor
    stage_states: torch.Tensor
    terminal_state: torch.Tensor
    objectives: torch.Tensor


def canonical_continuous_trace(
    model: torch.nn.Module,
    initial: torch.Tensor,
    cfg: Any,
    params: Mapping[str, torch.Tensor],
    *,
    state_mode: str,
) -> dict[str, torch.Tensor]:
    normalized_time = torch.linspace(
        0.0,
        1.0,
        int(cfg.n) + 1,
        device=initial.device,
        dtype=initial.dtype,
    )
    node_logits = model.time_logits(normalized_time)
    midpoint_logits = pchip_midpoint_logits(normalized_time, node_logits)
    return continuous_policy_stage_trace(
        model,
        initial,
        cfg,
        normalized_time,
        node_logits,
        midpoint_logits,
        params,
        state_mode=state_mode,
    )


def _node_states(trace: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        (
            trace["states"][:, :, 0, :],
            trace["terminal_state"].unsqueeze(1),
        ),
        dim=1,
    )


def canonical_continuous_objective(
    trace: Mapping[str, torch.Tensor],
    cfg: Any,
    params: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    stage_running = (
        (trace["states"] * params["beta"]).sum(dim=-1)
        + params["gamma"] * trace["controls"]
    )
    weights = torch.tensor(
        (1.0, 2.0, 2.0, 1.0),
        device=stage_running.device,
        dtype=stage_running.dtype,
    )
    running = (
        (float(cfg.T) / int(cfg.n))
        * (stage_running * weights).sum(dim=(1, 2))
        / 6.0
    )
    terminal = (
        trace["terminal_state"] * params["alpha"]
    ).sum(dim=-1)
    return terminal + running


@torch.no_grad()
def canonical_continuous_rollout(
    model: torch.nn.Module,
    initial: torch.Tensor,
    cfg: Any,
    params: Mapping[str, torch.Tensor],
    *,
    state_mode: str,
) -> ContinuousRollout:
    trace = canonical_continuous_trace(
        model,
        initial,
        cfg,
        params,
        state_mode=state_mode,
    )
    return ContinuousRollout(
        states=_node_states(trace),
        controls=trace["controls"],
        stage_states=trace["states"],
        terminal_state=trace["terminal_state"],
        objectives=canonical_continuous_objective(trace, cfg, params),
    )


def _gate_values(
    adapted: torch.nn.Module,
    trace: Mapping[str, torch.Tensor],
    cfg: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = trace["states"].shape[0]
    stage_time = trace["normalized_time"]
    flat_stage_time = (
        stage_time.unsqueeze(0).expand(batch, -1, -1).reshape(-1)
    )
    stage_gate = adapted.gate(
        flat_stage_time,
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
    node_gate = adapted.gate(
        node_time.repeat(batch),
        nodes.reshape(-1, int(cfg.m)),
    )
    return stage_gate, node_gate


@torch.no_grad()
def protected_identity_metrics(
    source: torch.nn.Module,
    adapted: torch.nn.Module,
    initial: torch.Tensor,
    cfg: Any,
    params: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    source_trace = canonical_continuous_trace(
        source, initial, cfg, params, state_mode="feedback"
    )
    candidate_trace = canonical_continuous_trace(
        adapted, initial, cfg, params, state_mode="feedback"
    )
    source_stage_gate, source_node_gate = _gate_values(
        adapted, source_trace, cfg
    )
    candidate_stage_gate, candidate_node_gate = _gate_values(
        adapted, candidate_trace, cfg
    )
    source_nodes = _node_states(source_trace)
    candidate_nodes = _node_states(candidate_trace)
    normalized_time = torch.linspace(
        0.0,
        1.0,
        int(cfg.n) + 1,
        device=initial.device,
        dtype=initial.dtype,
    )
    source_logits = source.time_logits(normalized_time)
    candidate_logits = adapted.time_logits(normalized_time)
    source_objective = canonical_continuous_objective(
        source_trace, cfg, params
    )
    candidate_objective = canonical_continuous_objective(
        candidate_trace, cfg, params
    )
    return {
        "source_stage_gate_max_abs": float(
            source_stage_gate.abs().max().cpu()
        ),
        "candidate_stage_gate_max_abs": float(
            candidate_stage_gate.abs().max().cpu()
        ),
        "source_node_gate_max_abs": float(
            source_node_gate.abs().max().cpu()
        ),
        "candidate_node_gate_max_abs": float(
            candidate_node_gate.abs().max().cpu()
        ),
        "node_time_logit_max_abs": float(
            (source_logits - candidate_logits).abs().max().cpu()
        ),
        "stage_control_max_abs": float(
            (
                source_trace["controls"] - candidate_trace["controls"]
            )
            .abs()
            .max()
            .cpu()
        ),
        "stage_state_max_abs": float(
            (source_trace["states"] - candidate_trace["states"])
            .abs()
            .max()
            .cpu()
        ),
        "terminal_state_max_abs": float(
            (
                source_trace["terminal_state"]
                - candidate_trace["terminal_state"]
            )
            .abs()
            .max()
            .cpu()
        ),
        "node_state_max_abs": float(
            (source_nodes - candidate_nodes).abs().max().cpu()
        ),
        "objective_max_abs": float(
            (source_objective - candidate_objective).abs().max().cpu()
        ),
    }


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


def paired_bootstrap_mean(
    values: torch.Tensor,
    *,
    seed: int,
    repeats: int,
) -> dict[str, Any]:
    if repeats <= 0:
        raise ValueError("bootstrap repeats must be positive")
    array = values.detach().cpu().numpy().astype(np.float64, copy=False)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("paired bootstrap requires a nonempty vector")
    if not np.isfinite(array).all():
        raise ValueError("paired bootstrap values must be finite")
    generator = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=np.float64)
    block = 1000
    for start in range(0, repeats, block):
        stop = min(start + block, repeats)
        indices = generator.integers(
            0,
            array.size,
            size=(stop - start, array.size),
        )
        means[start:stop] = array[indices].mean(axis=1)
    return {
        "method": "iid_paired_percentile_bootstrap_of_sample_mean",
        "confidence_level": 0.95,
        "quantile_method": "linear",
        "repeats": int(repeats),
        "seed": int(seed),
        "lower_95": float(
            np.quantile(means, 0.025, method="linear")
        ),
        "upper_95": float(
            np.quantile(means, 0.975, method="linear")
        ),
    }


def _as_finite_vector(value: torch.Tensor, *, label: str) -> torch.Tensor:
    detached = value.detach().cpu().to(dtype=torch.float64)
    if detached.ndim != 1 or detached.numel() == 0:
        raise ValueError(f"{label} must be a nonempty vector")
    if not torch.isfinite(detached).all():
        raise ValueError(f"{label} must be finite")
    return detached


def per_sample_records(
    candidate: torch.Tensor,
    source_feedback: torch.Tensor,
    frozen_time: torch.Tensor,
) -> list[dict[str, float | int]]:
    candidate = _as_finite_vector(candidate, label="candidate J")
    source_feedback = _as_finite_vector(
        source_feedback, label="source-feedback J"
    )
    frozen_time = _as_finite_vector(frozen_time, label="frozen-time J")
    if not (
        candidate.shape == source_feedback.shape == frozen_time.shape
    ):
        raise ValueError("physical-J vectors must have identical shapes")
    rows: list[dict[str, float | int]] = []
    for index in range(candidate.numel()):
        candidate_j = float(candidate[index])
        source_j = float(source_feedback[index])
        frozen_j = float(frozen_time[index])
        rows.append(
            {
                "index": index,
                "candidate_physical_J": candidate_j,
                "source_feedback_physical_J": source_j,
                "frozen_time_physical_J": frozen_j,
                "delta_frozen_time_minus_candidate": (
                    frozen_j - candidate_j
                ),
                "delta_source_feedback_minus_candidate": (
                    source_j - candidate_j
                ),
                "delta_source_feedback_minus_frozen_time": (
                    source_j - frozen_j
                ),
            }
        )
    return rows


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


def _formal_evaluation_protocol() -> dict[str, Any]:
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
        "candidate_state_mode": "feedback",
        "source_feedback_state_mode": "feedback",
        "frozen_time_state_mode": "w_zero",
        "physical_scale_factor": "1/cfg.alpha",
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "bootstrap_seeds": {
            "frozen_time_minus_candidate": BOOTSTRAP_SEED,
            "source_feedback_minus_candidate": BOOTSTRAP_SEED + 1,
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


def _validate_paths(
    artifact: Path,
    output: Path,
    ledger: Path,
) -> None:
    if not artifact.is_file():
        raise FileNotFoundError(f"artifact is not a regular file: {artifact}")
    if len({artifact, output, ledger}) != 3:
        raise ValueError("artifact, output, and ledger paths must be distinct")
    if _path_exists(output):
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if _path_exists(ledger):
        raise FileExistsError(
            f"reserved seed already has a consumption ledger: {ledger}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    ledger.parent.mkdir(parents=True, exist_ok=True)


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
    output_path = _absolute_without_resolving_symlinks(args.output)
    # Repository-scoped and seed-scoped by construction.  Letting callers
    # choose another path would permit the same reserved seed to be evaluated
    # repeatedly simply by changing a CLI argument.
    ledger_path = _absolute_without_resolving_symlinks(
        CANONICAL_CONSUMPTION_LEDGER
    )
    _validate_paths(artifact_path, output_path, ledger_path)

    expected_artifact_hash = require_sha256(
        args.expected_artifact_sha256,
        label="expected artifact SHA256",
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
        "scripts/evaluate_compact_exact_gated_one_shot_blind.py"
    )
    if code_manifest_before[evaluator_relative_path] != expected_evaluator_hash:
        raise RuntimeError("one-shot evaluator SHA256 mismatch")
    if code_manifest_hash != expected_manifest_hash:
        raise RuntimeError("one-shot evaluator code-manifest SHA256 mismatch")
    artifact_bytes = artifact_path.read_bytes()
    artifact_hash_before = hashlib.sha256(artifact_bytes).hexdigest()
    if artifact_hash_before != expected_artifact_hash:
        raise RuntimeError("compact adapter artifact SHA256 mismatch")

    device = torch.device(args.device)
    from scripts.compact_exact_gated_residual_adapter import (  # noqa: PLC0415
        load_compact_adapter_artifact,
    )

    # Load the exact byte string that was hashed.  Reopening the caller's path
    # inside the artifact loader would otherwise leave a hash/load TOCTOU gap.
    with tempfile.TemporaryDirectory(
        prefix="compact_blind_artifact_snapshot_"
    ) as snapshot_directory:
        snapshot_path = Path(snapshot_directory) / "artifact.pt"
        with snapshot_path.open("xb") as snapshot:
            snapshot.write(artifact_bytes)
            snapshot.flush()
            os.fsync(snapshot.fileno())
        payload, source, candidate_model, cfg = (
            load_compact_adapter_artifact(
                snapshot_path,
                device=device,
            )
        )
    del artifact_bytes
    artifact_protocol = validate_formal_artifact_metadata(payload)
    if sha256_file(artifact_path) != artifact_hash_before:
        raise RuntimeError("artifact changed while it was being loaded")

    params = build_params(cfg, device, torch.float64)
    source.eval()
    candidate_model.eval()
    protected_direction = torch.linspace(
        -1.0,
        1.0,
        int(cfg.m),
        device=device,
        dtype=torch.float64,
    )
    protected_initial = torch.stack(
        [
            float(cfg.n0) * (1.0 + radius * protected_direction)
            for radius in REQUIRED_PROTECTED_RADII
        ],
        dim=0,
    )
    identity = protected_identity_metrics(
        source,
        candidate_model,
        protected_initial,
        cfg,
        params,
    )
    assert_exact_protected_identity(identity)

    if sha256_file(artifact_path) != artifact_hash_before:
        raise RuntimeError("artifact changed during protected identity audit")
    if evaluator_code_manifest() != code_manifest_before:
        raise RuntimeError("evaluator code changed during preflight")

    ledger: ConsumptionLedger | None = None
    try:
        # This reservation is deliberately before the first call that
        # materializes the reserved seed.  Any later failure consumes it.
        ledger = ConsumptionLedger.reserve(
            ledger_path,
            {
                "event": "started",
                "protocol": PROTOCOL,
                "seed": FORMAL_SEED,
                "artifact": str(artifact_path),
                "artifact_sha256": artifact_hash_before,
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
        initial = initial_cpu.to(
            device=device,
            dtype=torch.float64,
        )
        if tensor_commitment(initial)["sha256"] != commitment["sha256"]:
            raise RuntimeError("initial states changed during device transfer")

        with torch.no_grad():
            candidate = canonical_continuous_rollout(
                candidate_model,
                initial,
                cfg,
                params,
                state_mode="feedback",
            )
            source_feedback = canonical_continuous_rollout(
                source,
                initial,
                cfg,
                params,
                state_mode="feedback",
            )
            frozen_time = canonical_continuous_rollout(
                source,
                initial,
                cfg,
                params,
                state_mode="w_zero",
            )

        physical_scale = 1.0 / float(cfg.alpha)
        candidate_j = candidate.objectives * physical_scale
        source_j = source_feedback.objectives * physical_scale
        frozen_j = frozen_time.objectives * physical_scale
        candidate_j = _as_finite_vector(candidate_j, label="candidate J")
        source_j = _as_finite_vector(source_j, label="source-feedback J")
        frozen_j = _as_finite_vector(frozen_j, label="frozen-time J")
        if not (
            candidate_j.numel()
            == source_j.numel()
            == frozen_j.numel()
            == FORMAL_COUNT
        ):
            raise RuntimeError("formal evaluator returned the wrong sample count")

        frozen_delta = frozen_j - candidate_j
        source_delta = source_j - candidate_j
        frozen_bootstrap = paired_bootstrap_mean(
            frozen_delta,
            seed=BOOTSTRAP_SEED,
            repeats=BOOTSTRAP_REPEATS,
        )
        source_bootstrap = paired_bootstrap_mean(
            source_delta,
            seed=BOOTSTRAP_SEED + 1,
            repeats=BOOTSTRAP_REPEATS,
        )

        artifact_hash_after = sha256_file(artifact_path)
        if artifact_hash_after != artifact_hash_before:
            raise RuntimeError("artifact changed during formal blind evaluation")
        code_manifest_after = evaluator_code_manifest()
        if code_manifest_after != code_manifest_before:
            raise RuntimeError("evaluator code changed during formal evaluation")

        acceptance = {
            "candidate_advantage_over_frozen_time_lower_95_positive": (
                frozen_bootstrap["lower_95"] > 0.0
            ),
            "candidate_advantage_over_source_feedback_lower_95_positive": (
                source_bootstrap["lower_95"] > 0.0
            ),
        }
        acceptance["passed"] = all(acceptance.values())
        result = {
            "protocol": PROTOCOL,
            "mode": "formal_one_shot_blind",
            "created_utc": utc_now(),
            "artifact": {
                "path": str(artifact_path),
                "expected_sha256": expected_artifact_hash,
                "sha256_before": artifact_hash_before,
                "sha256_after": artifact_hash_after,
                **artifact_protocol,
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
            "evaluation": _formal_evaluation_protocol(),
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
                "candidate_physical_J": _summary(candidate_j),
                "source_feedback_physical_J": _summary(source_j),
                "frozen_time_physical_J": _summary(frozen_j),
                "candidate_advantage_over_frozen_time": {
                    **_summary(frozen_delta),
                    "positive_fraction": float(
                        (frozen_delta > 0.0).double().mean()
                    ),
                },
                "candidate_advantage_over_source_feedback": {
                    **_summary(source_delta),
                    "positive_fraction": float(
                        (source_delta > 0.0).double().mean()
                    ),
                },
            },
            "paired_bootstrap": {
                "candidate_advantage_over_frozen_time": frozen_bootstrap,
                "candidate_advantage_over_source_feedback": source_bootstrap,
            },
            "acceptance": acceptance,
            "per_sample": per_sample_records(
                candidate_j,
                source_j,
                frozen_j,
            ),
        }
        output_hash = write_json_exclusive(output_path, result)
        ledger.append(
            {
                "event": "completed",
                "seed": FORMAL_SEED,
                "artifact_sha256": artifact_hash_after,
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
                "artifact_sha256": artifact_hash_before,
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
    parser.add_argument("--expected-artifact-sha256", required=True)
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
