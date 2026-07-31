#!/usr/bin/env python3
"""Common-random post-training audit for three full-gated CF candidates.

This evaluator is deliberately separate from the formal one-shot blind
evaluators.  It never creates, reads, or updates a blind-seed consumption
ledger.  The protocol is fixed to 128 iid componentwise perturbations at
radius 0.20 generated from development seed 2026073101.  All three candidate
lanes are evaluated on the exact same initial-state tensor.

For each lane, the candidate is compared with (i) its hash-locked CF source
and (ii) the frozen temporal branch of that source.  Objectives use the
continuous-policy four-stage RK4 evaluator: the policy is queried at every RK4
stage, and the running cost uses the matching RK4 quadrature.

The manifest schema is::

    {
      "schema": "gated-three-seed-common-r020-manifest-v1",
      "lanes": [
        {
          "lane": "base4_fb31",
          "candidate": {"path": "...", "sha256": "<64 hex>"},
          "locked_cf": {"path": "...", "sha256": "<64 hex>"}
        },
        ...
      ]
    }

Exactly three lanes are required.  Outputs are exclusive-created in a new
directory.  A failed run leaves that directory in place so that it cannot be
mistaken for a clean rerun.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import sys
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for search_path in (ROOT, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts.continue_feedback_full_state_gated_fallback import (  # noqa: E402
    REQUIRED_PROTECTED_RADII,
    RESERVED_BLIND_SEED,
    load_gated_feedback_checkpoint,
)
from scripts.evaluate_compact_exact_gated_one_shot_blind import (  # noqa: E402
    FORMAL_SEED,
    canonical_continuous_rollout,
    paired_bootstrap_mean,
    tensor_commitment,
)
from scripts.evaluate_feedback_section5 import (  # noqa: E402
    load_feedback_checkpoint,
)
from scripts.evaluate_full_gated_teacher_one_shot_blind import (  # noqa: E402
    assert_exact_protected_identity,
    protected_identity_metrics,
    verify_locked_source,
)
from train_paper_pmp_kkt import build_params  # noqa: E402


PROTOCOL = "gated_three_seed_common_r020_v1"
MANIFEST_SCHEMA = "gated-three-seed-common-r020-manifest-v1"
COMMON_EVALUATION_SEED = 2026073101
COMMON_COUNT = 128
COMMON_RADIUS = 0.20
BOOTSTRAP_REPEATS = 20_000
BOOTSTRAP_SEED_FROZEN = 2026073102
BOOTSTRAP_SEED_LOCKED = 2026073103
KNOWN_FORMAL_OR_RESERVED_SEEDS = frozenset(
    {
        int(RESERVED_BLIND_SEED),
        int(FORMAL_SEED),
        20260820,
    }
)

EVALUATOR_DEPENDENCIES = (
    "scripts/evaluate_gated_three_seed_common_r020.py",
    "scripts/continue_feedback_full_state_gated_fallback.py",
    "scripts/evaluate_compact_exact_gated_one_shot_blind.py",
    "scripts/evaluate_feedback_section5.py",
    "scripts/evaluate_full_gated_teacher_one_shot_blind.py",
    "scripts/feedback_continuous_policy_rk4.py",
    "scripts/feedback_section5_rk4_reference.py",
    "scripts/train_feedback_section5.py",
    "train_paper_pmp_kkt.py",
)


@dataclass(frozen=True)
class ArtifactSpec:
    path: Path
    sha256: str


@dataclass(frozen=True)
class LaneSpec:
    lane: str
    candidate: ArtifactSpec
    locked_cf: ArtifactSpec


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON output contains a non-finite float")
    return value


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    data = (
        json.dumps(
            json_safe(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _resolve_input_path(raw: Any, *, manifest_parent: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} path must be a nonempty string")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = manifest_parent / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} is not a regular file: {candidate}")
    return candidate


def _artifact_spec(
    raw: Any,
    *,
    manifest_parent: Path,
    label: str,
) -> ArtifactSpec:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} must be an object")
    if set(raw) != {"path", "sha256"}:
        raise ValueError(f"{label} must contain exactly path and sha256")
    return ArtifactSpec(
        path=_resolve_input_path(
            raw["path"],
            manifest_parent=manifest_parent,
            label=label,
        ),
        sha256=require_sha256(raw["sha256"], label=f"{label} SHA256"),
    )


def load_manifest(path: Path) -> tuple[list[LaneSpec], str]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"manifest is not a regular file: {path}")
    data = path.read_bytes()
    manifest_hash = sha256_bytes(data)
    payload = json.loads(data)
    if not isinstance(payload, Mapping):
        raise ValueError("manifest root must be an object")
    if set(payload) != {"schema", "lanes"}:
        raise ValueError("manifest must contain exactly schema and lanes")
    if payload["schema"] != MANIFEST_SCHEMA:
        raise ValueError(f"manifest schema must be {MANIFEST_SCHEMA}")
    raw_lanes = payload["lanes"]
    if not isinstance(raw_lanes, list) or len(raw_lanes) != 3:
        raise ValueError("manifest must contain exactly three lanes")
    lanes: list[LaneSpec] = []
    for index, raw_lane in enumerate(raw_lanes):
        label = f"lane[{index}]"
        if not isinstance(raw_lane, Mapping):
            raise ValueError(f"{label} must be an object")
        if set(raw_lane) != {"lane", "candidate", "locked_cf"}:
            raise ValueError(
                f"{label} must contain exactly lane, candidate, and locked_cf"
            )
        lane_name = raw_lane["lane"]
        if (
            not isinstance(lane_name, str)
            or re.fullmatch(r"[A-Za-z0-9_.-]+", lane_name) is None
        ):
            raise ValueError(
                f"{label}.lane must use only letters, digits, '.', '_', or '-'"
            )
        candidate = _artifact_spec(
            raw_lane["candidate"],
            manifest_parent=path.parent,
            label=f"{label}.candidate",
        )
        locked_cf = _artifact_spec(
            raw_lane["locked_cf"],
            manifest_parent=path.parent,
            label=f"{label}.locked_cf",
        )
        if candidate.path == locked_cf.path:
            raise ValueError(f"{label} candidate and locked CF must differ")
        lanes.append(
            LaneSpec(
                lane=lane_name,
                candidate=candidate,
                locked_cf=locked_cf,
            )
        )
    lane_names = [lane.lane for lane in lanes]
    if len(set(lane_names)) != len(lane_names):
        raise ValueError("manifest lane names must be unique")
    candidate_paths = [lane.candidate.path for lane in lanes]
    if len(set(candidate_paths)) != len(candidate_paths):
        raise ValueError("manifest candidate paths must be unique")
    return lanes, manifest_hash


def evaluator_code_manifest() -> dict[str, str]:
    return {
        relative: sha256_file(ROOT / relative)
        for relative in EVALUATOR_DEPENDENCIES
    }


def _snapshot_load_gated(
    path: Path,
    expected_hash: str,
) -> tuple[torch.nn.Module, Any, Any, dict[str, Any], str]:
    data = path.read_bytes()
    actual_hash = sha256_bytes(data)
    if actual_hash != expected_hash:
        raise RuntimeError(f"candidate SHA256 mismatch: {path}")
    with tempfile.TemporaryDirectory(prefix="common_r020_gated_") as directory:
        snapshot = Path(directory) / "candidate.pt"
        with snapshot.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        model, cfg, args, payload = load_gated_feedback_checkpoint(snapshot)
    return model, cfg, args, payload, actual_hash


def _snapshot_load_locked(
    path: Path,
    expected_hash: str,
) -> tuple[torch.nn.Module, Any, Any, str]:
    data = path.read_bytes()
    actual_hash = sha256_bytes(data)
    if actual_hash != expected_hash:
        raise RuntimeError(f"locked-CF SHA256 mismatch: {path}")
    with tempfile.TemporaryDirectory(prefix="common_r020_locked_") as directory:
        snapshot = Path(directory) / "locked.pt"
        with snapshot.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        model, cfg, args = load_feedback_checkpoint(snapshot)
    return model, cfg, args, actual_hash


def common_initial_states(cfg: Any) -> torch.Tensor:
    if COMMON_EVALUATION_SEED in KNOWN_FORMAL_OR_RESERVED_SEEDS:
        raise RuntimeError("common evaluation seed collides with a formal seed")
    generator = torch.Generator(device="cpu").manual_seed(
        COMMON_EVALUATION_SEED
    )
    directions = (
        2.0
        * torch.rand(
            COMMON_COUNT,
            int(cfg.m),
            generator=generator,
            device="cpu",
            dtype=torch.float64,
        )
        - 1.0
    )
    return float(cfg.n0) * (1.0 + COMMON_RADIUS * directions)


def _config_record(cfg: Any) -> dict[str, Any]:
    if is_dataclass(cfg):
        return asdict(cfg)
    return {
        key: value
        for key, value in vars(cfg).items()
        if not key.startswith("_")
    }


def _finite_vector(value: torch.Tensor, *, label: str) -> torch.Tensor:
    result = value.detach().cpu().to(dtype=torch.float64)
    if result.ndim != 1 or result.numel() == 0:
        raise ValueError(f"{label} must be a nonempty vector")
    if not torch.isfinite(result).all():
        raise ValueError(f"{label} contains non-finite values")
    return result


def vector_summary(value: torch.Tensor) -> dict[str, float | int]:
    value = _finite_vector(value, label="summary value")
    sample_sd = (
        value.std(unbiased=True)
        if value.numel() > 1
        else torch.zeros((), dtype=value.dtype)
    )
    return {
        "count": int(value.numel()),
        "mean": float(value.mean()),
        "sample_sd": float(sample_sd),
        "median": float(torch.quantile(value, 0.50)),
        "minimum": float(value.min()),
        "maximum": float(value.max()),
        "q05": float(torch.quantile(value, 0.05)),
        "q95": float(torch.quantile(value, 0.95)),
    }


def delta_summary(
    value: torch.Tensor,
    *,
    bootstrap_seed: int,
) -> dict[str, Any]:
    value = _finite_vector(value, label="paired delta")
    return {
        **vector_summary(value),
        "positive_fraction": float((value > 0.0).double().mean()),
        "paired_bootstrap_mean_95ci": paired_bootstrap_mean(
            value,
            seed=bootstrap_seed,
            repeats=BOOTSTRAP_REPEATS,
        ),
    }


def _per_sample_metrics(
    candidate: Any,
    locked: Any,
    frozen: Any,
    params: Mapping[str, torch.Tensor],
    *,
    physical_scale: float,
) -> dict[str, torch.Tensor]:
    candidate_j = _finite_vector(
        candidate.objectives * physical_scale,
        label="candidate physical J",
    )
    locked_j = _finite_vector(
        locked.objectives * physical_scale,
        label="locked-CF physical J",
    )
    frozen_j = _finite_vector(
        frozen.objectives * physical_scale,
        label="frozen-time physical J",
    )
    if not (
        candidate_j.shape == locked_j.shape == frozen_j.shape == (COMMON_COUNT,)
    ):
        raise RuntimeError("canonical evaluator returned an unexpected batch size")

    def stage_drift(left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        difference = (left - right).detach().cpu().to(dtype=torch.float64)
        flattened = difference.reshape(difference.shape[0], -1)
        return (
            flattened.square().mean(dim=1).sqrt(),
            flattened.abs().max(dim=1).values,
        )

    candidate_locked_rms, candidate_locked_max = stage_drift(
        candidate.controls, locked.controls
    )
    candidate_frozen_rms, candidate_frozen_max = stage_drift(
        candidate.controls, frozen.controls
    )
    resistant_mask = (params["x"] >= 0.7).detach().cpu()
    candidate_resistant = (
        candidate.terminal_state.detach().cpu()[:, resistant_mask].sum(dim=1)
    )
    locked_resistant = (
        locked.terminal_state.detach().cpu()[:, resistant_mask].sum(dim=1)
    )
    frozen_resistant = (
        frozen.terminal_state.detach().cpu()[:, resistant_mask].sum(dim=1)
    )
    return {
        "candidate_J": candidate_j,
        "locked_CF_J": locked_j,
        "frozen_time_J": frozen_j,
        "frozen_time_minus_candidate_J": frozen_j - candidate_j,
        "locked_CF_minus_candidate_J": locked_j - candidate_j,
        "frozen_time_minus_locked_CF_J": frozen_j - locked_j,
        "candidate_vs_locked_control_stage_rms": candidate_locked_rms,
        "candidate_vs_locked_control_stage_max_abs": candidate_locked_max,
        "candidate_vs_frozen_control_stage_rms": candidate_frozen_rms,
        "candidate_vs_frozen_control_stage_max_abs": candidate_frozen_max,
        "candidate_terminal_resistant_burden": candidate_resistant,
        "locked_CF_terminal_resistant_burden": locked_resistant,
        "frozen_time_terminal_resistant_burden": frozen_resistant,
        "locked_CF_minus_candidate_resistant_burden": (
            locked_resistant - candidate_resistant
        ),
        "frozen_time_minus_candidate_resistant_burden": (
            frozen_resistant - candidate_resistant
        ),
    }


def _lane_summary(metrics: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    return {
        "physical_objective": {
            "candidate": vector_summary(metrics["candidate_J"]),
            "locked_CF": vector_summary(metrics["locked_CF_J"]),
            "frozen_time": vector_summary(metrics["frozen_time_J"]),
        },
        "paired_physical_objective_deltas": {
            "frozen_time_minus_candidate": delta_summary(
                metrics["frozen_time_minus_candidate_J"],
                bootstrap_seed=BOOTSTRAP_SEED_FROZEN,
            ),
            "locked_CF_minus_candidate": delta_summary(
                metrics["locked_CF_minus_candidate_J"],
                bootstrap_seed=BOOTSTRAP_SEED_LOCKED,
            ),
            "frozen_time_minus_locked_CF": vector_summary(
                metrics["frozen_time_minus_locked_CF_J"]
            ),
        },
        "control_response": {
            "candidate_vs_locked_stage_rms": vector_summary(
                metrics["candidate_vs_locked_control_stage_rms"]
            ),
            "candidate_vs_locked_stage_max_abs": vector_summary(
                metrics["candidate_vs_locked_control_stage_max_abs"]
            ),
            "candidate_vs_frozen_stage_rms": vector_summary(
                metrics["candidate_vs_frozen_control_stage_rms"]
            ),
            "candidate_vs_frozen_stage_max_abs": vector_summary(
                metrics["candidate_vs_frozen_control_stage_max_abs"]
            ),
        },
        "terminal_resistant_burden_x_ge_0p7": {
            "candidate": vector_summary(
                metrics["candidate_terminal_resistant_burden"]
            ),
            "locked_CF": vector_summary(
                metrics["locked_CF_terminal_resistant_burden"]
            ),
            "frozen_time": vector_summary(
                metrics["frozen_time_terminal_resistant_burden"]
            ),
            "locked_CF_minus_candidate": vector_summary(
                metrics["locked_CF_minus_candidate_resistant_burden"]
            ),
            "frozen_time_minus_candidate": vector_summary(
                metrics["frozen_time_minus_candidate_resistant_burden"]
            ),
        },
    }


def _output_rows(
    lane: str,
    initial_states: torch.Tensor,
    metrics: Mapping[str, torch.Tensor],
) -> list[dict[str, Any]]:
    initial = initial_states.detach().cpu()
    rows: list[dict[str, Any]] = []
    for index in range(COMMON_COUNT):
        row: dict[str, Any] = {
            "lane": lane,
            "sample": index,
            "initial_total": float(initial[index].sum()),
            "initial_min": float(initial[index].min()),
            "initial_max": float(initial[index].max()),
        }
        for key, values in metrics.items():
            row[key] = float(values[index])
        rows.append(row)
    return rows


def _write_csv_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fieldnames = list(rows[0])
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())


def _write_initial_states(path: Path, states: torch.Tensor) -> None:
    array = (
        states.detach()
        .cpu()
        .contiguous()
        .numpy()
        .astype(np.dtype("<f8"), copy=False)
    )
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def _runtime(device: torch.device, *, threads: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "torch_threads": int(threads),
    }
    if device.type == "cuda":
        result["cuda"] = torch.version.cuda
        result["device_name"] = torch.cuda.get_device_name(device)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    manifest_path = args.manifest.expanduser().resolve()
    lanes, manifest_hash_before = load_manifest(manifest_path)
    code_manifest_before = evaluator_code_manifest()
    output_dir = args.out_dir.expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    first_cfg_record: dict[str, Any] | None = None
    initial_cpu: torch.Tensor | None = None
    initial_commitment: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    lane_summaries: list[dict[str, Any]] = []
    frozen_delta_vectors: list[torch.Tensor] = []
    locked_delta_vectors: list[torch.Tensor] = []
    input_hashes_after: dict[str, dict[str, str]] = {}

    for lane_index, lane in enumerate(lanes):
        candidate, cfg, _, payload, candidate_hash = _snapshot_load_gated(
            lane.candidate.path,
            lane.candidate.sha256,
        )
        locked, locked_cfg, _, locked_hash = _snapshot_load_locked(
            lane.locked_cf.path,
            lane.locked_cf.sha256,
        )
        gate = payload.get("gated_fallback", {})
        embedded_locked = gate.get("locked_checkpoint", {})
        if embedded_locked.get("sha256") != lane.locked_cf.sha256:
            raise RuntimeError(
                f"{lane.lane}: embedded locked-source SHA256 differs from manifest"
            )
        prior_seeds = {
            key: gate.get(key)
            for key in (
                "train_seed",
                "validation_seed",
                "optimizer_seed",
                "reserved_blind_seed",
            )
        }
        reused = [
            key
            for key, value in prior_seeds.items()
            if type(value) is int and value == COMMON_EVALUATION_SEED
        ]
        if reused:
            raise RuntimeError(
                f"{lane.lane}: common evaluation seed appears in artifact "
                f"training metadata: {reused}"
            )
        verify_locked_source(candidate, cfg, locked, locked_cfg)

        cfg_record = _config_record(cfg)
        if first_cfg_record is None:
            first_cfg_record = cfg_record
            initial_cpu = common_initial_states(cfg)
            initial_commitment = tensor_commitment(initial_cpu)
            _write_initial_states(
                output_dir / "initial_states.npy",
                initial_cpu,
            )
        elif cfg_record != first_cfg_record:
            raise RuntimeError(
                f"{lane.lane}: problem configuration differs from the first lane"
            )
        assert initial_cpu is not None
        assert initial_commitment is not None

        candidate.to(device=device, dtype=torch.float64).eval()
        locked.to(device=device, dtype=torch.float64).eval()
        params = build_params(cfg, device, torch.float64)
        candidate.set_feature_vectors(params["r"], params["phi"])
        locked.set_feature_vectors(params["r"], params["phi"])
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
            locked,
            candidate,
            protected_initial,
            cfg,
            params,
        )
        assert_exact_protected_identity(identity)

        initial = initial_cpu.to(device=device, dtype=torch.float64)
        if tensor_commitment(initial)["sha256"] != initial_commitment["sha256"]:
            raise RuntimeError("shared initial states changed during device transfer")
        with torch.inference_mode():
            candidate_rollout = canonical_continuous_rollout(
                candidate,
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
        if not math.isfinite(physical_scale) or physical_scale <= 0.0:
            raise RuntimeError("physical objective scale must be finite and positive")
        metrics = _per_sample_metrics(
            candidate_rollout,
            locked_rollout,
            frozen_rollout,
            params,
            physical_scale=physical_scale,
        )
        rows.extend(_output_rows(lane.lane, initial_cpu, metrics))
        frozen_delta_vectors.append(
            metrics["frozen_time_minus_candidate_J"]
        )
        locked_delta_vectors.append(
            metrics["locked_CF_minus_candidate_J"]
        )
        lane_summaries.append(
            {
                "lane": lane.lane,
                "candidate": {
                    "path": str(lane.candidate.path),
                    "sha256": candidate_hash,
                },
                "locked_CF": {
                    "path": str(lane.locked_cf.path),
                    "sha256": locked_hash,
                    "matches_candidate_embedded_locked_branch": True,
                },
                "protected_identity": identity,
                "metrics": _lane_summary(metrics),
            }
        )
        input_hashes_after[lane.lane] = {
            "candidate": sha256_file(lane.candidate.path),
            "locked_CF": sha256_file(lane.locked_cf.path),
        }
        if input_hashes_after[lane.lane]["candidate"] != lane.candidate.sha256:
            raise RuntimeError(f"{lane.lane}: candidate changed during evaluation")
        if input_hashes_after[lane.lane]["locked_CF"] != lane.locked_cf.sha256:
            raise RuntimeError(f"{lane.lane}: locked CF changed during evaluation")

        del candidate_rollout
        del locked_rollout
        del frozen_rollout
        del metrics
        del candidate
        del locked
        del params
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert first_cfg_record is not None
    assert initial_cpu is not None
    assert initial_commitment is not None
    frozen_matrix = torch.stack(frozen_delta_vectors, dim=0)
    locked_matrix = torch.stack(locked_delta_vectors, dim=0)
    state_averaged_frozen = frozen_matrix.mean(dim=0)
    state_averaged_locked = locked_matrix.mean(dim=0)
    lane_mean_frozen = frozen_matrix.mean(dim=1)
    lane_mean_locked = locked_matrix.mean(dim=1)
    summary = {
        "protocol": PROTOCOL,
        "created_utc": utc_now(),
        "evaluation": {
            "seed": COMMON_EVALUATION_SEED,
            "count": COMMON_COUNT,
            "radius": COMMON_RADIUS,
            "sampler": (
                "iid componentwise Uniform[-1,1] generated once on CPU in "
                "torch.float64; initial=cfg.n0*(1+radius*direction)"
            ),
            "antithetic": False,
            "shared_initial_tensor_across_all_lanes": True,
            "canonical_evaluator": (
                "continuous_policy_RK4_four_stage_requery_with_RK4_running_cost"
            ),
            "candidate_state_mode": "feedback",
            "locked_CF_state_mode": "feedback",
            "frozen_time_state_mode": "w_zero",
            "positive_delta_favors_candidate": True,
            "used_for_training": False,
            "used_for_checkpoint_selection": False,
            "used_for_hyperparameter_tuning": False,
            "formal_blind_protocol": False,
            "formal_seed_or_ledger_used": False,
        },
        "problem": first_cfg_record,
        "initial_state_commitment": initial_commitment,
        "lanes": lane_summaries,
        "across_lanes": {
            "lane_level_mean_deltas": {
                "frozen_time_minus_candidate": vector_summary(
                    lane_mean_frozen
                ),
                "locked_CF_minus_candidate": vector_summary(
                    lane_mean_locked
                ),
                "note": (
                    "sample SD is across the three lane-level means; pooled "
                    "3x128 rows are not treated as independent"
                ),
            },
            "statewise_seed_averaged_deltas": {
                "frozen_time_minus_candidate": delta_summary(
                    state_averaged_frozen,
                    bootstrap_seed=BOOTSTRAP_SEED_FROZEN,
                ),
                "locked_CF_minus_candidate": delta_summary(
                    state_averaged_locked,
                    bootstrap_seed=BOOTSTRAP_SEED_LOCKED,
                ),
                "note": (
                    "each of 128 common states is first averaged over the "
                    "three lanes, then bootstrapped over states"
                ),
            },
        },
    }

    if sha256_file(manifest_path) != manifest_hash_before:
        raise RuntimeError("manifest changed during evaluation")
    if evaluator_code_manifest() != code_manifest_before:
        raise RuntimeError("evaluator code changed during evaluation")

    per_sample_path = output_dir / "per_sample.csv"
    summary_path = output_dir / "summary.json"
    provenance_path = output_dir / "provenance.json"
    initial_path = output_dir / "initial_states.npy"
    _write_csv_exclusive(per_sample_path, rows)
    write_json_exclusive(summary_path, summary)
    output_hashes = {
        "initial_states.npy": sha256_file(initial_path),
        "per_sample.csv": sha256_file(per_sample_path),
        "summary.json": sha256_file(summary_path),
    }
    provenance = {
        "protocol": PROTOCOL,
        "created_utc": utc_now(),
        "manifest": {
            "path": str(manifest_path),
            "sha256_before": manifest_hash_before,
            "sha256_after": sha256_file(manifest_path),
        },
        "evaluator_code_manifest": code_manifest_before,
        "runtime": _runtime(device, threads=args.threads),
        "inputs_after_evaluation": input_hashes_after,
        "initial_state_commitment": initial_commitment,
        "output_file_sha256": output_hashes,
        "formal_seed_or_ledger_used": False,
    }
    write_json_exclusive(provenance_path, provenance)
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True), flush=True)
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, required=True)
    result.add_argument("--out-dir", type=Path, required=True)
    result.add_argument("--device", default="cpu")
    result.add_argument("--threads", type=int, default=8)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
