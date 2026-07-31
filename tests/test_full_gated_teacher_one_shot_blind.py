from __future__ import annotations

import argparse
import copy
import inspect
import sys
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
    REQUIRED_PROTECTED_RADII,
    RESERVED_BLIND_SEED,
    STANDARD_BASE_GRID_PROTOCOL,
    STRUCTURAL_GUARD_PROTOCOL,
    construct_gated_model,
)
from evaluate_full_gated_teacher_one_shot_blind import (  # noqa: E402
    BOOTSTRAP_REPEATS,
    FORMAL_COUNT,
    FORMAL_RADIUS,
    FORMAL_SEED,
    IDENTITY_RADII,
    PROTOCOL,
    assert_exact_protected_identity,
    build_parser,
    per_sample_records,
    protected_identity_metrics,
    run,
    validate_formal_artifact_metadata,
    verify_locked_source,
)
from train_feedback_section5 import NestedFeedbackTransformer  # noqa: E402
from train_paper_pmp_kkt import (  # noqa: E402
    ProblemConfig,
    build_params,
)


def tiny_standard_models() -> tuple[
    NestedFeedbackTransformer,
    NestedFeedbackTransformer,
    ProblemConfig,
    argparse.Namespace,
]:
    torch.manual_seed(7)
    cfg = ProblemConfig(
        T=1.0,
        n=4,
        m=3,
        umax=3.0,
        beta=0.1,
        alpha=1.0,
        gamma=20.0,
        n0=10.0,
        m_suppression=0.5,
    )
    args = argparse.Namespace(
        state_scale=15.0,
        state_hidden="8",
        d_model=8,
        heads=2,
        layers=1,
        init_u=1.5,
        correction_gain=0.5,
        state_feature_mode="relative_nominal",
        center_state_correction=True,
        action_temperature=1.0,
        action_scale=1.0,
        action_parameterization="logit-temperature",
        action_offset=0.0,
        option="der",
        initialization="direct-trajectory supervised state-branch fit",
    )
    locked = NestedFeedbackTransformer(
        cfg.m,
        cfg.umax,
        args.state_scale,
        (8,),
        args.d_model,
        args.heads,
        args.layers,
        args.init_u,
        args.correction_gain,
        args.state_feature_mode,
        args.center_state_correction,
        args.action_temperature,
        args.action_scale,
        args.action_parameterization,
        args.action_offset,
    ).double()
    reference = torch.full(
        (cfg.n + 1, cfg.m),
        cfg.n0,
        dtype=torch.float64,
    )
    locked.set_nominal_reference(reference)
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    locked.set_feature_vectors(params["r"], params["phi"])
    with torch.no_grad():
        final = [
            module
            for module in locked.state_branch.modules()
            if isinstance(module, torch.nn.Linear)
        ][-1]
        final.weight.fill_(0.04)
        final.bias.fill_(0.01)
    probe = copy.deepcopy(locked)
    with torch.no_grad():
        probe_linears = [
            module
            for module in probe.state_branch.modules()
            if isinstance(module, torch.nn.Linear)
        ]
        probe_linears[0].weight.add_(0.02)
        probe_linears[-1].weight.mul_(1.7)
        probe_linears[-1].bias.add_(0.03)
    locked.eval()
    probe.eval()
    return locked, probe, cfg, args


def valid_payload() -> dict[str, object]:
    return {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "gated_fallback": {
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
            "posthoc_protected_identity": {
                "control_max_abs": 0.0,
                "state_max_abs": 0.0,
                "objective_max_abs": 0.0,
            },
            "posthoc_identity_reference": {
                "path": "/frozen/locked.pt",
                "sha256": "a" * 64,
                "evaluator": STANDARD_BASE_GRID_PROTOCOL,
            },
            "train_seed": PMP_TRAIN_SEED,
            "validation_seed": PMP_VALIDATION_SEED,
            "optimizer_seed": OPTIMIZER_SEED,
            "reserved_blind_seed": RESERVED_BLIND_SEED,
            "reserved_blind_used_for_training_or_selection": False,
            "protected_radii": list(REQUIRED_PROTECTED_RADII),
            "locked_checkpoint": {
                "path": "/frozen/locked.pt",
                "sha256": "a" * 64,
            },
        },
    }


def test_formal_protocol_is_hard_locked_and_not_cli_adjustable() -> None:
    assert PROTOCOL == "full_gated_teacher_one_shot_blind_v1"
    assert FORMAL_SEED == 20261701
    assert FORMAL_COUNT == 128
    assert FORMAL_RADIUS == 0.20
    assert BOOTSTRAP_REPEATS == 20_000
    assert IDENTITY_RADII == (0.0, 0.10, 0.20)

    parser = build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    for forbidden in (
        "--seed",
        "--count",
        "--radius",
        "--bootstrap-seed",
        "--bootstrap-repeats",
        "--consumption-ledger",
    ):
        assert forbidden not in option_strings


def test_reserved_states_are_materialized_only_after_ledger_reservation() -> None:
    source = inspect.getsource(run)
    reserve = source.index("ConsumptionLedger.reserve")
    materialize = source.index("_formal_initial_states")
    physical_j = source.index("canonical_continuous_rollout")
    assert source.count("_formal_initial_states") == 1
    assert reserve < materialize < physical_j


def test_metadata_preserves_legacy_seed_and_reserves_new_seed() -> None:
    result = validate_formal_artifact_metadata(valid_payload())
    assert result["artifact_legacy_reserved_seed"] == RESERVED_BLIND_SEED
    assert result["artifact_legacy_reserved_seed_used"] is False
    assert result["new_one_shot_reserved_seed"] == FORMAL_SEED
    assert (
        result[
            "new_one_shot_reserved_seed_absent_from_prior_artifact_seeds"
        ]
        is True
    )


def test_metadata_rejects_new_seed_reuse_and_nonexact_identity() -> None:
    payload = valid_payload()
    payload["gated_fallback"]["train_seed"] = FORMAL_SEED  # type: ignore[index]
    with pytest.raises(RuntimeError, match="protocol mismatch"):
        validate_formal_artifact_metadata(payload)

    payload = valid_payload()
    payload["gated_fallback"]["posthoc_protected_identity"][  # type: ignore[index]
        "state_max_abs"
    ] = 1.0e-16
    with pytest.raises(RuntimeError, match="not exact"):
        validate_formal_artifact_metadata(payload)


def test_per_sample_records_use_prespecified_primary_secondary_signs() -> None:
    teacher = torch.tensor([3.0, 7.0], dtype=torch.float64)
    locked = torch.tensor([5.0, 6.0], dtype=torch.float64)
    frozen = torch.tensor([4.0, 9.0], dtype=torch.float64)
    assert per_sample_records(teacher, locked, frozen) == [
        {
            "index": 0,
            "teacher_physical_J": 3.0,
            "locked_cf_source_physical_J": 5.0,
            "frozen_time_physical_J": 4.0,
            "primary_delta_frozen_time_minus_teacher": 1.0,
            "secondary_delta_locked_cf_source_minus_teacher": 2.0,
            "delta_locked_cf_source_minus_frozen_time": 1.0,
        },
        {
            "index": 1,
            "teacher_physical_J": 7.0,
            "locked_cf_source_physical_J": 6.0,
            "frozen_time_physical_J": 9.0,
            "primary_delta_frozen_time_minus_teacher": 2.0,
            "secondary_delta_locked_cf_source_minus_teacher": -1.0,
            "delta_locked_cf_source_minus_frozen_time": -3.0,
        },
    ]


def test_exact_continuous_identity_uses_external_and_internal_locked_source() -> None:
    locked, probe, cfg, args = tiny_standard_models()
    anchor_time = torch.linspace(
        0.0,
        1.0,
        cfg.n + 1,
        dtype=torch.float64,
    )
    anchors = torch.full(
        (
            int(ANCHOR_FAMILY_PROTOCOL["total"]),
            cfg.n + 1,
            cfg.m,
        ),
        cfg.n0,
        dtype=torch.float64,
    )
    teacher = construct_gated_model(
        locked,
        probe,
        cfg,
        args,
        anchor_time=anchor_time,
        anchor_states=anchors,
        gate_tube=100.0,
        gate_transition=0.1,
    ).double()
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    teacher.set_feature_vectors(params["r"], params["phi"])
    teacher.eval()
    verify_locked_source(teacher, cfg, locked, cfg)
    direction = torch.linspace(-1.0, 1.0, cfg.m, dtype=torch.float64)
    initial = torch.stack(
        [
            cfg.n0 * (1.0 + radius * direction)
            for radius in IDENTITY_RADII
        ]
    )
    identity = protected_identity_metrics(
        locked,
        teacher,
        initial,
        cfg,
        params,
    )
    assert_exact_protected_identity(identity)

    teacher.gate_tube = 0.0
    teacher.gate_transition = 1.0e-6
    broken = protected_identity_metrics(
        locked,
        teacher,
        initial,
        cfg,
        params,
    )
    with pytest.raises(RuntimeError, match="not bitwise exact"):
        assert_exact_protected_identity(broken)
