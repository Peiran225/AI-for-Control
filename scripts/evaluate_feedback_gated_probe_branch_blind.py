#!/usr/bin/env python3
"""One-shot blind evaluator for a selected full-state gated checkpoint.

The evaluator is read-only with respect to model parameters: it performs no
training, checkpoint selection, hyperparameter tuning, or direct supervision.
Formal mode is hard-locked to 128 componentwise-random states generated with
the never-used seed 20260901.  Smoke mode uses a separate development seed and
therefore cannot consume the formal blind set.

Before evaluating physical objectives, the script verifies the complete
artifact protocol, the embedded locked branch against its source checkpoint,
and exact identity on the protected trajectories.  The identity audit includes
continuous-feedback RK4 node, midpoint, and internal-stage policy queries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for search_path in (ROOT, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from continue_feedback_full_state_gated_fallback import (  # noqa: E402
    ANCHOR_FAMILY_PROTOCOL,
    CHECKPOINT_FORMAT,
    GATE_PROTOCOL,
    OPTIMIZER_SEED,
    PMP_TRAIN_SEED,
    PMP_VALIDATION_SEED,
    REQUIRED_PROTECTED_RADII,
    RESERVED_BLIND_SEED,
    STANDARD_BASE_GRID_PROTOCOL,
    STRUCTURAL_GUARD_PROTOCOL,
    FlatTubeProbeFeedbackTransformer,
    load_gated_feedback_checkpoint,
    require_protocol_radii,
    structural_guard_metrics,
)
from evaluate_feedback_section5 import (  # noqa: E402
    load_feedback_checkpoint as load_standard_feedback_checkpoint,
)
from feedback_continuous_policy_rk4 import (  # noqa: E402
    pchip_midpoint_logits,
)
from feedback_section5_rk4_reference import dynamics  # noqa: E402
from refine_feedback_offgrid_scalar import (  # noqa: E402
    fine_problem,
    fixed_support_dense_logits,
)
from refine_feedback_svd_null_projected_kkt import (  # noqa: E402
    antithetic_initial_states,
    policy_rollout,
    posthoc_test_metrics,
    structured_initial_states,
)
from train_paper_pmp_kkt import build_params  # noqa: E402


FORMAL_COUNT = 128
FORMAL_RADIUS = 0.20
FORMAL_BOOTSTRAP_REPEATS = 20_000
FORMAL_BOOTSTRAP_SEED = 20260902
DEVELOPMENT_SMOKE_SEED = 20260890
DEVELOPMENT_BOOTSTRAP_SEED = 20260891
DEVELOPMENT_SMOKE_COUNT = 4
DEVELOPMENT_BOOTSTRAP_REPEATS = 200


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def paired_bootstrap_mean(
    values: torch.Tensor,
    *,
    seed: int,
    repeats: int,
) -> dict[str, float | int]:
    if repeats <= 0:
        raise ValueError("bootstrap repeats must be positive")
    array = values.detach().cpu().numpy().astype(np.float64, copy=False)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("paired bootstrap requires a nonempty vector")
    generator = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=np.float64)
    block = 1000
    for start in range(0, repeats, block):
        stop = min(start + block, repeats)
        indices = generator.integers(
            0, array.size, size=(stop - start, array.size)
        )
        means[start:stop] = array[indices].mean(axis=1)
    return {
        "repeats": int(repeats),
        "seed": int(seed),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
    }


def evaluation_protocol(smoke: bool) -> dict[str, int | float | bool]:
    if smoke:
        return {
            "smoke": True,
            "seed": DEVELOPMENT_SMOKE_SEED,
            "count": DEVELOPMENT_SMOKE_COUNT,
            "radius": FORMAL_RADIUS,
            "bootstrap_repeats": DEVELOPMENT_BOOTSTRAP_REPEATS,
            "bootstrap_seed": DEVELOPMENT_BOOTSTRAP_SEED,
            "identity_multiplier": 1,
        }
    return {
        "smoke": False,
        "seed": RESERVED_BLIND_SEED,
        "count": FORMAL_COUNT,
        "radius": FORMAL_RADIUS,
        "bootstrap_repeats": FORMAL_BOOTSTRAP_REPEATS,
        "bootstrap_seed": FORMAL_BOOTSTRAP_SEED,
        "identity_multiplier": 2,
    }


def _assert_equal_tensor_dicts(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    *,
    label: str,
) -> None:
    if set(left) != set(right):
        raise RuntimeError(f"{label} tensor keys differ")
    dtype_or_device_mismatch = [
        key
        for key in sorted(left)
        if (
            left[key].dtype != right[key].dtype
            or left[key].device != right[key].device
        )
    ]
    if dtype_or_device_mismatch:
        raise RuntimeError(
            f"{label} tensor dtype/device differs: "
            f"{dtype_or_device_mismatch[:5]}"
        )
    unequal = [
        key for key in sorted(left) if not torch.equal(left[key], right[key])
    ]
    if unequal:
        raise RuntimeError(f"{label} tensors differ: {unequal[:5]}")


def verify_artifact_protocol(
    payload: dict[str, Any],
    artifact_path: Path,
    expected_artifact_sha256: str,
    *,
    development_smoke: bool,
) -> dict[str, Any]:
    actual_hash = sha256(artifact_path)
    if actual_hash != expected_artifact_sha256:
        raise RuntimeError("gated artifact SHA256 mismatch")
    if payload.get("checkpoint_format") != CHECKPOINT_FORMAT:
        raise RuntimeError("unsupported gated checkpoint format")
    gate = payload.get("gated_fallback", {})
    expected = {
        "centered_state_logits": True,
        "shared_time_branch": True,
        "gate": GATE_PROTOCOL,
        "structural_guard": STRUCTURAL_GUARD_PROTOCOL,
        "anchor_families": ANCHOR_FAMILY_PROTOCOL,
        "base_n": 800,
        "standard_base_grid": STANDARD_BASE_GRID_PROTOCOL,
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
        "reserved_blind_seed": RESERVED_BLIND_SEED,
        "reserved_blind_used_for_training_or_selection": False,
    }
    wrong = {
        key: (gate.get(key), value)
        for key, value in expected.items()
        if gate.get(key) != value
    }
    if wrong:
        raise RuntimeError(f"artifact protocol mismatch: {wrong}")
    if type(gate.get("formal_protocol")) is not bool:
        raise RuntimeError("artifact formal_protocol flag is missing")
    if not development_smoke and gate["formal_protocol"] is not True:
        raise RuntimeError(
            "formal blind evaluation cannot use a development artifact"
        )
    posthoc = gate.get("posthoc_protected_identity", {})
    if not posthoc or any(float(value) != 0.0 for value in posthoc.values()):
        raise RuntimeError(
            "artifact is missing an exact completed posthoc identity audit"
        )
    require_protocol_radii(
        tuple(float(value) for value in gate.get("protected_radii", ()))
    )
    return {
        "artifact_sha256": actual_hash,
        "gate": gate["gate"],
        "anchor_families": gate["anchor_families"],
        "standard_base_grid": gate["standard_base_grid"],
        "posthoc_identity_audit_completed": True,
        "formal_protocol": gate["formal_protocol"],
        "protected_radii": list(gate["protected_radii"]),
        "training_seeds": {
            "train": gate["train_seed"],
            "validation": gate["validation_seed"],
            "optimizer": gate["optimizer_seed"],
        },
        "reserved_blind_seed": gate["reserved_blind_seed"],
        "reserved_blind_used_before_evaluation": False,
    }


def verify_locked_source(
    model: FlatTubeProbeFeedbackTransformer,
    cfg: Any,
    payload: dict[str, Any],
    locked_path: Path,
) -> tuple[torch.nn.Module, str]:
    gate = payload["gated_fallback"]
    expected_hash = gate["locked_checkpoint"]["sha256"]
    actual_hash = sha256(locked_path)
    if actual_hash != expected_hash:
        raise RuntimeError("locked source checkpoint SHA256 mismatch")
    source, source_cfg, _ = load_standard_feedback_checkpoint(locked_path)
    if source_cfg != cfg:
        raise RuntimeError("locked source problem differs from gated artifact")
    _assert_equal_tensor_dicts(
        source.time_branch.state_dict(),
        model.time_branch.state_dict(),
        label="locked time branch",
    )
    _assert_equal_tensor_dicts(
        source.state_branch.state_dict(),
        model.state_branch.state_dict(),
        label="locked state branch",
    )
    if not torch.equal(source.nominal_reference, model.nominal_reference):
        raise RuntimeError("locked nominal reference differs from artifact")
    return source, actual_hash


@torch.no_grad()
def continuous_stage_identity_metrics(
    model: FlatTubeProbeFeedbackTransformer,
    initial: torch.Tensor,
    cfg: Any,
    normalized_time: torch.Tensor,
    raw_logits: torch.Tensor,
    midpoint_raw_logits: torch.Tensor,
    params: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Compare locked/candidate queries at every continuous RK4 stage."""

    def rollout(
        state_mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        step = cfg.T / cfg.n
        batch = initial.shape[0]
        state = initial
        nodes = [state]
        stage_states: list[torch.Tensor] = []
        stage_controls: list[torch.Tensor] = []
        for index in range(cfg.n):
            left = normalized_time[index].expand(batch)
            middle = (
                0.5 * (normalized_time[index] + normalized_time[index + 1])
            ).expand(batch)
            right = normalized_time[index + 1].expand(batch)
            control1 = model.interval_action(
                raw_logits[index],
                left,
                state,
                state_mode=state_mode,
            )
            slope1 = dynamics(state, control1, params)
            stage2 = state + 0.5 * step * slope1
            control2 = model.interval_action(
                midpoint_raw_logits[index],
                middle,
                stage2,
                state_mode=state_mode,
            )
            slope2 = dynamics(stage2, control2, params)
            stage3 = state + 0.5 * step * slope2
            control3 = model.interval_action(
                midpoint_raw_logits[index],
                middle,
                stage3,
                state_mode=state_mode,
            )
            slope3 = dynamics(stage3, control3, params)
            stage4 = state + step * slope3
            control4 = model.interval_action(
                raw_logits[index + 1],
                right,
                stage4,
                state_mode=state_mode,
            )
            slope4 = dynamics(stage4, control4, params)
            state = state + (step / 6.0) * (
                slope1 + 2.0 * slope2 + 2.0 * slope3 + slope4
            )
            nodes.append(state)
            stage_states.append(
                torch.stack((nodes[-2], stage2, stage3, stage4), dim=1)
            )
            stage_controls.append(
                torch.stack(
                    (control1, control2, control3, control4), dim=1
                )
            )
        return (
            torch.stack(nodes, dim=1),
            torch.stack(stage_states, dim=1),
            torch.stack(stage_controls, dim=1),
        )

    locked_nodes, locked_stages, locked_controls = rollout("locked_feedback")
    candidate_nodes, candidate_stages, candidate_controls = rollout("feedback")
    batch = initial.shape[0]
    stage_time = torch.stack(
        (
            normalized_time[:-1],
            0.5 * (normalized_time[:-1] + normalized_time[1:]),
            0.5 * (normalized_time[:-1] + normalized_time[1:]),
            normalized_time[1:],
        ),
        dim=1,
    )
    flat_time = (
        stage_time.unsqueeze(0)
        .expand(batch, -1, -1)
        .reshape(-1)
    )
    flat_state = candidate_stages.reshape(-1, candidate_stages.shape[-1])
    stage_gate = model.gate_values(flat_time, flat_state)
    return {
        "protected_stage_gate_max": float(stage_gate.max().cpu()),
        "protected_stage_control_max_abs": float(
            (candidate_controls - locked_controls).abs().max().cpu()
        ),
        "protected_stage_state_max_abs": float(
            (candidate_stages - locked_stages).abs().max().cpu()
        ),
        "protected_stage_node_state_max_abs": float(
            (candidate_nodes - locked_nodes).abs().max().cpu()
        ),
    }


def protected_identity_audit(
    model: FlatTubeProbeFeedbackTransformer,
    source: torch.nn.Module,
    cfg: Any,
    params: dict[str, torch.Tensor],
    *,
    multiplier: int,
) -> dict[str, float]:
    protected = structured_initial_states(
        REQUIRED_PROTECTED_RADII,
        cfg,
        model.nominal_reference.device,
        torch.float64,
    )
    evaluation_cfg = fine_problem(cfg, multiplier)
    normalized_time, raw_logits = fixed_support_dense_logits(
        model, cfg, multiplier, query_batch_size=16
    )
    midpoint_raw_logits = pchip_midpoint_logits(
        normalized_time, raw_logits
    )
    common = structural_guard_metrics(
        model,
        protected,
        evaluation_cfg,
        normalized_time,
        raw_logits,
        midpoint_raw_logits,
        params,
    )
    stage = continuous_stage_identity_metrics(
        model,
        protected,
        evaluation_cfg,
        normalized_time,
        raw_logits,
        midpoint_raw_logits,
        params,
    )
    with torch.no_grad():
        candidate = policy_rollout(
            model, protected, cfg, params, state_mode="feedback"
        )
        source_rollout = policy_rollout(
            source, protected, cfg, params, state_mode="feedback"
        )
    objective_scale = 1.0 / cfg.alpha
    candidate_stage_controls = candidate.controls.unsqueeze(-1).expand(
        -1, -1, 4
    )
    source_stage_controls = source_rollout.controls.unsqueeze(-1).expand(
        -1, -1, 4
    )
    task = {
        "protected_task_control_max_abs": float(
            (candidate.controls - source_rollout.controls).abs().max().cpu()
        ),
        "protected_task_stage_control_max_abs": float(
            (
                candidate_stage_controls - source_stage_controls
            ).abs().max().cpu()
        ),
        "protected_task_state_max_abs": float(
            (candidate.states - source_rollout.states).abs().max().cpu()
        ),
        "protected_task_stage_state_max_abs": float(
            (
                candidate.stage_states - source_rollout.stage_states
            ).abs().max().cpu()
        ),
        "protected_task_physical_objective_max_abs": float(
            (
                (candidate.objectives - source_rollout.objectives)
                * objective_scale
            ).abs().max().cpu()
        ),
    }
    result = {**common, **stage, **task}
    nonzero = {key: value for key, value in result.items() if value != 0.0}
    if nonzero:
        raise RuntimeError(
            f"protected identity is not bitwise exact: {nonzero}"
        )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if args.smoke and args.device != "cpu":
        raise ValueError("development smoke is CPU-only")
    protocol = evaluation_protocol(args.smoke)
    if not args.smoke and args.device == "auto":
        raise ValueError("formal blind evaluation requires an explicit device")
    output = args.out_dir.expanduser().resolve()
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(f"refusing to overwrite nonempty {output}")

    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    artifact_path = args.artifact.expanduser().resolve()
    model, cfg, _, payload = load_gated_feedback_checkpoint(artifact_path)
    artifact_protocol = verify_artifact_protocol(
        payload,
        artifact_path,
        args.expected_artifact_sha256,
        development_smoke=args.smoke,
    )
    locked_path = args.locked_checkpoint.expanduser().resolve()
    source, source_hash = verify_locked_source(
        model, cfg, payload, locked_path
    )
    model.to(device=device, dtype=torch.float64).eval()
    source.to(device=device, dtype=torch.float64).eval()
    params = build_params(cfg, device, torch.float64)
    model.set_feature_vectors(params["r"], params["phi"])
    source.set_feature_vectors(params["r"], params["phi"])
    identity = protected_identity_audit(
        model,
        source,
        cfg,
        params,
        multiplier=int(protocol["identity_multiplier"]),
    )

    initial = antithetic_initial_states(
        int(protocol["count"]),
        int(protocol["seed"]),
        float(protocol["radius"]),
        cfg,
        device,
        torch.float64,
        antithetic=False,
    )
    with torch.no_grad():
        candidate = policy_rollout(
            model, initial, cfg, params, state_mode="feedback"
        )
        source_feedback = policy_rollout(
            source, initial, cfg, params, state_mode="feedback"
        )
        frozen_time = policy_rollout(
            source, initial, cfg, params, state_mode="w_zero"
        )
    physical_scale = 1.0 / cfg.alpha
    time_advantage = (
        frozen_time.objectives - candidate.objectives
    ) * physical_scale
    source_advantage = (
        source_feedback.objectives - candidate.objectives
    ) * physical_scale
    bootstrap_seed = int(protocol["bootstrap_seed"])
    result = {
        "protocol": (
            "full_state_gated_blind_v1"
            if not args.smoke
            else "full_state_gated_development_smoke_v1"
        ),
        "mode": "formal_blind" if not args.smoke else "development_smoke",
        "artifact": str(artifact_path),
        "locked_checkpoint": str(locked_path),
        "locked_checkpoint_sha256": source_hash,
        "evaluation": protocol,
        "artifact_protocol": artifact_protocol,
        "used_for_training": False,
        "used_for_checkpoint_selection": False,
        "used_for_hyperparameter_tuning": False,
        "physical_objective_used_only_for_posthoc_evaluation": True,
        "protected_identity": identity,
        **posthoc_test_metrics(
            candidate,
            source_feedback,
            frozen_time,
            physical_scale_factor=physical_scale,
        ),
        "paired_bootstrap": {
            "candidate_advantage_over_frozen_time": paired_bootstrap_mean(
                time_advantage,
                seed=bootstrap_seed,
                repeats=int(protocol["bootstrap_repeats"]),
            ),
            "candidate_advantage_over_source_feedback": paired_bootstrap_mean(
                source_advantage,
                seed=bootstrap_seed + 1,
                repeats=int(protocol["bootstrap_repeats"]),
            ),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "blind_evaluation.json"
    result_path.write_text(
        json.dumps(safe(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(safe(result), indent=2, sort_keys=True), flush=True)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--artifact", type=Path, required=True)
    result.add_argument("--locked-checkpoint", type=Path, required=True)
    result.add_argument("--expected-artifact-sha256", required=True)
    result.add_argument("--out-dir", type=Path, required=True)
    result.add_argument("--device", default="cpu")
    result.add_argument("--threads", type=int, default=8)
    result.add_argument(
        "--smoke", action=argparse.BooleanOptionalAction, default=False
    )
    return result


if __name__ == "__main__":
    run(parser().parse_args())
