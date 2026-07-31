from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scripts.compact_exact_gated_residual_adapter import (
    COMPACT_ADAPTER_FORMAT,
    ExactGatedCompactFeedback,
    compact_artifact_payload,
    compact_parameter_count,
    continuous_policy_identity_metrics,
    continuous_policy_trace,
    load_compact_adapter_artifact,
    parameter_inventory_for_paper_model,
    source_feature_dimension,
    _node_states_from_trace,
)
from scripts.refine_feedback_gated_residual_adapter import (
    ProtectedTrajectoryGate,
    calibrated_zero_tube_squared,
    load_adapter_artifact,
)
from scripts.refine_feedback_svd_null_projected_kkt import (
    structured_initial_states,
)
from scripts.train_feedback_section5 import NestedFeedbackTransformer
from train_paper_pmp_kkt import ProblemConfig, build_params


def _source():
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
    args = argparse.Namespace(
        state_scale=15.0,
        state_hidden="8",
        d_model=8,
        heads=2,
        layers=1,
        init_u=1.5,
        correction_gain=0.5,
        state_feature_mode="burden_composition",
        center_state_correction=True,
        action_temperature=1.0,
        action_scale=1.0,
        action_parameterization="logit-temperature",
        action_offset=0.0,
        option="der",
    )
    source = NestedFeedbackTransformer(
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
        (cfg.n + 1, cfg.m), cfg.n0, dtype=torch.float64
    )
    source.set_nominal_reference(reference)
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    source.set_feature_vectors(params["r"], params["phi"])
    source.eval()
    protected_initial = structured_initial_states(
        (0.0, 0.10, 0.20),
        cfg,
        torch.device("cpu"),
        torch.float64,
    )
    trace = continuous_policy_trace(
        source,
        protected_initial,
        cfg,
        params,
    )
    protected = _node_states_from_trace(trace)
    zero_tube_squared, _ = calibrated_zero_tube_squared(
        protected,
        trace["states"],
        distance_scale=0.05,
        safety_factor=2.0,
    )
    gate = ProtectedTrajectoryGate(
        protected,
        distance_scale=0.05,
        zero_tube_squared=zero_tube_squared,
    )
    return source, gate, cfg, args


def test_parameter_counts_are_far_below_full_state_branch() -> None:
    inventory = parameter_inventory_for_paper_model()
    assert inventory[8]["parameters"] == 257
    assert inventory[16]["parameters"] == 513
    assert inventory[32]["parameters"] == 1025
    assert inventory[32]["percent_of_20609"] < 5.0


def test_adapter_is_zero_initialized_and_source_is_frozen() -> None:
    source, gate, _, _ = _source()
    model = ExactGatedCompactFeedback(source, gate, width=8)
    assert model.adapter_parameter_count == compact_parameter_count(
        source_feature_dimension(source), 8
    )
    assert all(not parameter.requires_grad for parameter in source.parameters())
    time = torch.tensor([0.2, 0.6], dtype=torch.float64)
    state = torch.tensor(
        [[8.0, 10.0, 12.0], [9.0, 10.0, 11.0]],
        dtype=torch.float64,
    )
    base = torch.tensor([0.1, 0.2], dtype=torch.float64)
    torch.testing.assert_close(
        model.interval_action(base, time, state, state_mode="feedback"),
        source.interval_action(base, time, state, state_mode="feedback"),
    )


def test_flat_tube_gate_keeps_all_three_protected_families_exact() -> None:
    source, gate, _, _ = _source()
    model = ExactGatedCompactFeedback(source, gate, width=16)
    with torch.no_grad():
        model.adapter.output.weight.fill_(1.0)
        model.adapter.output.bias.fill_(0.5)
    time = torch.linspace(0.0, 1.0, gate.protected_states.shape[1])
    states = gate.protected_states
    flat_time = time.repeat(states.shape[0])
    flat_state = states.reshape(-1, source.m)
    correction, gate_value = model.residual_logit(flat_time, flat_state)
    assert torch.equal(gate_value, torch.zeros_like(gate_value))
    assert torch.equal(correction, torch.zeros_like(correction))


def test_compact_artifact_round_trip_and_legacy_evaluator_dispatch(
    tmp_path: Path,
) -> None:
    source, gate, cfg, args = _source()
    model = ExactGatedCompactFeedback(source, gate, width=8).double()
    with torch.no_grad():
        model.adapter.output.weight.fill_(0.01)
    payload = compact_artifact_payload(
        source_payload={"args": vars(args)},
        source=source,
        model=model,
        cfg=cfg,
        protected_radii=(0.0, 0.10, 0.20),
        gate_zero_tube_squared=gate.zero_tube_squared,
        teacher_provenance={
            "role": "initialization_only",
            "artifact_sha256": "teacher-sha",
        },
        cleanup_provenance={
            "loss": "scalar_PMP_DER",
            "selection": "validation_PMP_residual",
            "selected_train_PMP_gate_distribution": {
                "transition_fraction": 0.0
            },
            "selected_validation_PMP_gate_distribution": {
                "transition_fraction": 0.0
            },
        },
    )
    assert payload["format"] == COMPACT_ADAPTER_FORMAT
    path = tmp_path / "compact.pt"
    torch.save(payload, path)
    loaded, _, native_model, loaded_cfg = load_compact_adapter_artifact(
        path, device=torch.device("cpu")
    )
    assert loaded_cfg == cfg
    expected = compact_parameter_count(source_feature_dimension(source), 8)
    assert loaded["adapter_spec"]["parameter_count"] == expected
    assert native_model.adapter_parameter_count == expected
    dispatched, _, dispatched_model, _ = load_adapter_artifact(
        path, device=torch.device("cpu")
    )
    assert dispatched["format"] == COMPACT_ADAPTER_FORMAT
    assert isinstance(dispatched_model, ExactGatedCompactFeedback)
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    protected_initial = structured_initial_states(
        (0.0, 0.10, 0.20),
        cfg,
        torch.device("cpu"),
        torch.float64,
    )
    identity = continuous_policy_identity_metrics(
        dispatched_model.source,
        dispatched_model,
        protected_initial,
        cfg,
        params,
    )
    assert identity["gate_max_abs"] == 0.0
    assert identity["control_max_abs"] == 0.0
    assert identity["state_max_abs"] == 0.0
    assert identity["objective_max_abs"] == 0.0
