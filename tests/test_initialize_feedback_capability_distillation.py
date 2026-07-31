from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from scripts.continue_feedback_full_state_gated_fallback import (
    FlatTubeProbeFeedbackTransformer,
    construct_gated_model,
)
from scripts.initialize_feedback_capability_distillation import (
    assert_existing_state_branch_only,
    capability_transfer_targets,
    changed_parameter_keys,
    hard_anchor_gate,
    sha256,
    stage_raw_logits,
    validate_positive_teacher_audit,
)
from scripts.train_feedback_section5 import NestedFeedbackTransformer
from train_paper_pmp_kkt import ProblemConfig, build_params


def _models():
    cfg = ProblemConfig(
        T=1.0,
        n=4,
        m=3,
        umax=3.0,
        beta=0.1,
        alpha=0.0025,
        gamma=20.0,
        n0=10.0,
        m_suppression=0.5,
    )
    constructor = dict(
        m=cfg.m,
        umax=cfg.umax,
        state_scale=15.0,
        state_hidden=(8,),
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
    )
    student = NestedFeedbackTransformer(**constructor).double()
    locked = copy.deepcopy(student)
    probe = copy.deepcopy(student)
    reference = torch.full((cfg.n + 1, cfg.m), 10.0, dtype=torch.float64)
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    for model in (student, locked, probe):
        model.set_nominal_reference(reference)
        model.set_feature_vectors(params["r"], params["phi"])
    with torch.no_grad():
        probe_linears = [
            module
            for module in probe.state_branch.modules()
            if isinstance(module, torch.nn.Linear)
        ]
        probe_linears[-1].weight.fill_(0.4)
        probe_linears[-1].bias.fill_(0.1)
    source_args = type(
        "Args",
        (),
        {
            "state_scale": 15.0,
            "state_hidden": "8",
            "d_model": 8,
            "heads": 2,
            "layers": 1,
            "init_u": 1.5,
            "correction_gain": 0.5,
            "state_feature_mode": "relative_nominal",
            "center_state_correction": True,
            "action_temperature": 1.0,
            "action_scale": 1.0,
            "action_parameterization": "logit-temperature",
            "action_offset": 0.0,
        },
    )()
    teacher = construct_gated_model(
        locked,
        probe,
        cfg,
        source_args,
        anchor_time=torch.linspace(0.0, 1.0, cfg.n + 1),
        anchor_states=torch.full(
            (1, cfg.n + 1, cfg.m), 10.0, dtype=torch.float64
        ),
        gate_tube=0.0,
        gate_transition=0.01,
    ).double()
    teacher.set_feature_vectors(params["r"], params["phi"])
    return student.eval(), teacher.eval()


def test_stage_raw_logits_has_rk4_order() -> None:
    nodes = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    middle = torch.tensor([1.5, 2.5], dtype=torch.float64)
    result = stage_raw_logits(nodes, middle)
    torch.testing.assert_close(
        result,
        torch.tensor(
            [[1.0, 1.5, 1.5, 2.0], [2.0, 2.5, 2.5, 3.0]],
            dtype=torch.float64,
        ),
    )


def test_transfer_uses_only_gated_minus_locked_increment() -> None:
    student, teacher = _models()
    state = torch.tensor(
        [[13.0, 9.0, 7.0], [8.0, 10.0, 12.0]],
        dtype=torch.float64,
    )
    time = torch.tensor([0.3, 0.7], dtype=torch.float64)
    teacher_raw = torch.tensor([-0.4, 0.2], dtype=torch.float64)
    student_raw = torch.tensor([0.1, -0.2], dtype=torch.float64)
    with torch.no_grad():
        teacher_action = teacher.interval_action(
            teacher_raw, time, state, state_mode="feedback"
        )
        locked_action = teacher.interval_action(
            teacher_raw, time, state, state_mode="locked_feedback"
        )
        student_source = student.interval_action(
            student_raw, time, state, state_mode="feedback"
        )
    target, increment = capability_transfer_targets(
        student,
        teacher,
        state,
        time,
        teacher_raw,
        student_raw,
        teacher_action,
    )
    torch.testing.assert_close(increment, teacher_action - locked_action)
    torch.testing.assert_close(
        target,
        (student_source + teacher_action - locked_action).clamp(0.0, 3.0),
    )
    assert not torch.equal(target, teacher_action)


def test_transfer_is_identity_when_teacher_gate_is_zero() -> None:
    student, teacher = _models()
    state = torch.full((2, 3), 10.0, dtype=torch.float64)
    time = torch.tensor([0.25, 0.75], dtype=torch.float64)
    teacher_raw = torch.tensor([0.1, 0.2], dtype=torch.float64)
    student_raw = torch.tensor([-0.3, 0.4], dtype=torch.float64)
    teacher_action = teacher.interval_action(
        teacher_raw, time, state, state_mode="feedback"
    )
    target, increment = capability_transfer_targets(
        student,
        teacher,
        state,
        time,
        teacher_raw,
        student_raw,
        teacher_action,
    )
    source = student.interval_action(
        student_raw, time, state, state_mode="feedback"
    )
    torch.testing.assert_close(increment, torch.zeros_like(increment))
    torch.testing.assert_close(target, source)


def test_hard_anchor_gate_checks_all_structured_rows() -> None:
    assert hard_anchor_gate(
        torch.tensor([1.0e-5, 2.0e-5, 3.0e-5]),
        nominal_limit=1.0e-4,
        structured_limit=1.0e-4,
    )
    assert not hard_anchor_gate(
        torch.tensor([1.0e-5, 2.0e-5, 2.0e-4]),
        nominal_limit=1.0e-4,
        structured_limit=1.0e-4,
    )


def test_changed_keys_must_stay_in_existing_state_branch() -> None:
    student, _ = _models()
    before = copy.deepcopy(student.state_dict())
    with torch.no_grad():
        next(student.state_branch.parameters()).reshape(-1)[0].add_(1.0e-6)
    changed = changed_parameter_keys(before, student.state_dict())
    assert changed
    assert_existing_state_branch_only(changed)
    with pytest.raises(RuntimeError, match="outside the existing state branch"):
        assert_existing_state_branch_only(["time_branch.output.weight"])


def test_teacher_requires_an_independent_positive_audit(
    tmp_path: Path,
) -> None:
    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"validated teacher")
    audit = {
        "artifact_protocol": {"artifact_sha256": sha256(teacher)},
        "used_for_checkpoint_selection": False,
        "used_for_hyperparameter_tuning": False,
        "physical_objective_used_only_for_posthoc_evaluation": True,
        "candidate_advantage_over_frozen_time": {"mean": 0.75},
        "paired_bootstrap": {
            "candidate_advantage_over_frozen_time": {"lower_95": 0.5}
        },
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    assert validate_positive_teacher_audit(audit_path, teacher) == audit

    audit["paired_bootstrap"]["candidate_advantage_over_frozen_time"][
        "lower_95"
    ] = -0.1
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="positive paired-bootstrap"):
        validate_positive_teacher_audit(audit_path, teacher)
