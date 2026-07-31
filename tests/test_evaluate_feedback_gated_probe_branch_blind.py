from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
TESTS = ROOT / "tests"
for search_path in (ROOT, SCRIPTS, TESTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from continue_feedback_full_state_gated_fallback import (  # noqa: E402
    ANCHOR_FAMILY_PROTOCOL,
    CHECKPOINT_FORMAT,
    GATE_PROTOCOL,
    OPTIMIZER_SEED,
    PMP_TRAIN_SEED,
    PMP_VALIDATION_SEED,
    RESERVED_BLIND_SEED,
    STANDARD_BASE_GRID_PROTOCOL,
    STRUCTURAL_GUARD_PROTOCOL,
    construct_gated_model,
    cpu_state_dict,
)
from evaluate_feedback_gated_probe_branch_blind import (  # noqa: E402
    DEVELOPMENT_SMOKE_SEED,
    FORMAL_COUNT,
    FORMAL_RADIUS,
    continuous_stage_identity_metrics,
    evaluation_protocol,
    paired_bootstrap_mean,
    parser,
    run,
    sha256,
    verify_artifact_protocol,
)
from feedback_continuous_policy_rk4 import (  # noqa: E402
    pchip_midpoint_logits,
)
from refine_feedback_offgrid_scalar import (  # noqa: E402
    fixed_support_dense_logits,
)
from test_continue_feedback_full_state_gated_fallback import (  # noqa: E402
    formal_tiny_standard_models,
    tiny_standard_models,
)
from train_paper_pmp_kkt import build_params  # noqa: E402


def _write_tiny_artifact(tmp_path: Path) -> tuple[Path, Path]:
    locked, probe, cfg, args = formal_tiny_standard_models()
    locked_path = tmp_path / "locked.pt"
    common = {
        "args": vars(args),
        "problem": asdict(cfg),
        "nominal_reference": locked.nominal_reference.detach().clone(),
    }
    torch.save(
        {**common, "model_state": cpu_state_dict(locked)}, locked_path
    )
    anchor_time = torch.linspace(
        0.0, 1.0, cfg.n + 1, dtype=torch.float64
    )
    anchor_states = torch.full(
        (
            int(ANCHOR_FAMILY_PROTOCOL["total"]),
            cfg.n + 1,
            cfg.m,
        ),
        cfg.n0,
        dtype=torch.float64,
    )
    model = construct_gated_model(
        locked,
        probe,
        cfg,
        args,
        anchor_time=anchor_time,
        anchor_states=anchor_states,
        gate_tube=1.0,
        gate_transition=0.1,
    ).double()
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    model.set_feature_vectors(params["r"], params["phi"])
    payload = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "model_state": cpu_state_dict(model),
        "args": vars(args),
        "problem": asdict(cfg),
        "nominal_reference": model.nominal_reference.detach().clone(),
        "gated_fallback": {
            "centered_state_logits": True,
            "shared_time_branch": True,
            "gate": GATE_PROTOCOL,
            "structural_guard": STRUCTURAL_GUARD_PROTOCOL,
            "anchor_families": ANCHOR_FAMILY_PROTOCOL,
            "base_n": cfg.n,
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
            "posthoc_protected_identity": {
                "protected_standard_gate_max": 0.0,
                "protected_standard_control_max_abs": 0.0,
                "protected_standard_stage_control_max_abs": 0.0,
                "protected_standard_node_state_max_abs": 0.0,
                "protected_standard_stage_state_max_abs": 0.0,
                "protected_standard_objective_max_abs": 0.0,
            },
            "posthoc_identity_reference": {
                "path": str(locked_path),
                "sha256": sha256(locked_path),
                "evaluator": STANDARD_BASE_GRID_PROTOCOL,
            },
            "train_seed": PMP_TRAIN_SEED,
            "validation_seed": PMP_VALIDATION_SEED,
            "optimizer_seed": OPTIMIZER_SEED,
            "reserved_blind_seed": RESERVED_BLIND_SEED,
            "reserved_blind_used_for_training_or_selection": False,
            "protected_radii": [0.0, 0.1, 0.2, 0.4, 0.6],
            "gate_normalization": cfg.n0,
            "gate_tube": model.gate_tube,
            "gate_transition": model.gate_transition,
            "locked_checkpoint": {
                "path": str(locked_path),
                "sha256": sha256(locked_path),
            },
        },
    }
    artifact = tmp_path / "gated.pt"
    torch.save(payload, artifact)
    return artifact, locked_path


def test_protocol_hard_locks_formal_and_separates_smoke_seed() -> None:
    formal = evaluation_protocol(False)
    smoke = evaluation_protocol(True)
    assert formal["seed"] == RESERVED_BLIND_SEED
    assert formal["count"] == FORMAL_COUNT
    assert formal["radius"] == FORMAL_RADIUS
    assert smoke["seed"] == DEVELOPMENT_SMOKE_SEED
    assert smoke["seed"] != RESERVED_BLIND_SEED
    assert smoke["count"] < formal["count"]


def test_formal_blind_rejects_development_artifact(tmp_path: Path) -> None:
    artifact, _ = _write_tiny_artifact(tmp_path)
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    payload["gated_fallback"]["formal_protocol"] = False
    development_artifact = tmp_path / "development_gated.pt"
    torch.save(payload, development_artifact)
    artifact_hash = sha256(development_artifact)

    verify_artifact_protocol(
        payload,
        development_artifact,
        artifact_hash,
        development_smoke=True,
    )
    with pytest.raises(RuntimeError, match="formal blind.*development"):
        verify_artifact_protocol(
            payload,
            development_artifact,
            artifact_hash,
            development_smoke=False,
        )


def test_paired_bootstrap_is_deterministic() -> None:
    values = torch.tensor([1.0, -2.0, 3.0, 4.0], dtype=torch.float64)
    first = paired_bootstrap_mean(values, seed=91, repeats=1000)
    second = paired_bootstrap_mean(values, seed=91, repeats=1000)
    assert first == second
    assert first["lower_95"] <= values.mean() <= first["upper_95"]


def test_continuous_stage_guard_is_exact_inside_flat_tube() -> None:
    locked, probe, cfg, args = tiny_standard_models()
    anchor_time = torch.linspace(
        0.0, 1.0, cfg.n + 1, dtype=torch.float64
    )
    anchors = torch.full(
        (5, cfg.n + 1, cfg.m), cfg.n0, dtype=torch.float64
    )
    model = construct_gated_model(
        locked,
        probe,
        cfg,
        args,
        anchor_time=anchor_time,
        anchor_states=anchors,
        gate_tube=1.0,
        gate_transition=0.1,
    ).double()
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    model.set_feature_vectors(params["r"], params["phi"])
    normalized_time, raw_logits = fixed_support_dense_logits(
        model, cfg, 1, query_batch_size=2
    )
    midpoint = pchip_midpoint_logits(normalized_time, raw_logits)
    initial = torch.full((5, cfg.m), cfg.n0, dtype=torch.float64)
    metrics = continuous_stage_identity_metrics(
        model,
        initial,
        cfg,
        normalized_time,
        raw_logits,
        midpoint,
        params,
    )
    assert metrics
    assert all(value == 0.0 for value in metrics.values())


def test_development_smoke_never_consumes_formal_blind_seed(
    tmp_path: Path,
) -> None:
    artifact, locked = _write_tiny_artifact(tmp_path)
    output = tmp_path / "blind_smoke"
    args = parser().parse_args(
        [
            "--artifact",
            str(artifact),
            "--locked-checkpoint",
            str(locked),
            "--expected-artifact-sha256",
            sha256(artifact),
            "--out-dir",
            str(output),
            "--device",
            "cpu",
            "--threads",
            "1",
            "--smoke",
        ]
    )
    result = run(args)
    assert result["mode"] == "development_smoke"
    assert result["evaluation"]["seed"] == DEVELOPMENT_SMOKE_SEED
    assert result["evaluation"]["seed"] != RESERVED_BLIND_SEED
    assert result["used_for_training"] is False
    assert result["used_for_checkpoint_selection"] is False
    assert result["used_for_hyperparameter_tuning"] is False
    assert (
        result["artifact_protocol"]["anchor_families"]
        == ANCHOR_FAMILY_PROTOCOL
    )
    assert (
        result["artifact_protocol"]["standard_base_grid"]
        == STANDARD_BASE_GRID_PROTOCOL
    )
    assert (
        result["artifact_protocol"]["posthoc_identity_audit_completed"]
        is True
    )
    assert all(
        value == 0.0 for value in result["protected_identity"].values()
    )
    assert (output / "blind_evaluation.json").is_file()
    with pytest.raises(FileExistsError, match="overwrite"):
        run(args)


def test_formal_cli_exposes_no_seed_or_count_override(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parser().parse_args(
            [
                "--artifact",
                str(tmp_path / "a.pt"),
                "--locked-checkpoint",
                str(tmp_path / "b.pt"),
                "--expected-artifact-sha256",
                "0" * 64,
                "--out-dir",
                str(tmp_path / "out"),
                "--seed",
                "1",
            ]
        )
