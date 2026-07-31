from __future__ import annotations

import argparse
import copy
import math
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
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
    assert_only_probe_trainable,
    build_locked_anchors,
    construct_gated_model,
    continuous_policy_stage_trace,
    cpu_state_dict,
    load_gated_feedback_checkpoint,
    parameter_inventory,
    parser,
    pmp_residual_pack,
    require_protocol_radii,
    run,
    validate_checkpoint_pair,
)
from evaluate_feedback_section5 import (  # noqa: E402
    load_feedback_checkpoint as load_standard_feedback_checkpoint,
)
from feedback_continuous_policy_rk4 import (  # noqa: E402
    continuous_feedback_state_rk4,
    pchip_midpoint_logits,
)
from refine_feedback_offgrid_scalar import (  # noqa: E402
    fine_problem,
    fixed_support_dense_logits,
)
from refine_feedback_svd_null_projected_kkt import (  # noqa: E402
    structured_initial_states,
)
from natural_cubic_anchor import (  # noqa: E402
    natural_cubic_second_derivatives,
    natural_cubic_uniform_value,
)
from train_feedback_section5 import (  # noqa: E402
    NestedFeedbackTransformer,
    simulate_feedback_rk4_stagewise,
)
from train_paper_pmp_kkt import (  # noqa: E402
    ProblemConfig,
    build_params,
)


def tiny_cfg() -> ProblemConfig:
    return ProblemConfig(
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


def tiny_args() -> argparse.Namespace:
    return argparse.Namespace(
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


def tiny_standard_models() -> tuple[
    NestedFeedbackTransformer,
    NestedFeedbackTransformer,
    ProblemConfig,
    argparse.Namespace,
]:
    torch.manual_seed(7)
    cfg = tiny_cfg()
    args = tiny_args()
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
        (cfg.n + 1, cfg.m), cfg.n0, dtype=torch.float64
    )
    locked.set_nominal_reference(reference)
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    locked.set_feature_vectors(params["r"], params["phi"])
    with torch.no_grad():
        time_parameter = next(locked.time_branch.parameters())
        time_parameter.reshape(-1)[0].add_(1.234567890123e-10)
        linears = [
            module
            for module in locked.state_branch.modules()
            if isinstance(module, torch.nn.Linear)
        ]
        linears[0].weight[0, 0].add_(9.876543210987e-10)
        linears[-1].weight.fill_(0.04)
        linears[-1].bias.fill_(0.01)
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


def formal_tiny_standard_models() -> tuple[
    NestedFeedbackTransformer,
    NestedFeedbackTransformer,
    ProblemConfig,
    argparse.Namespace,
]:
    locked, probe, tiny, args = tiny_standard_models()
    cfg = ProblemConfig(**{**asdict(tiny), "n": 800})
    reference = torch.full(
        (cfg.n + 1, cfg.m), cfg.n0, dtype=torch.float64
    )
    locked.set_nominal_reference(reference)
    probe.set_nominal_reference(reference)
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    locked.set_feature_vectors(params["r"], params["phi"])
    probe.set_feature_vectors(params["r"], params["phi"])
    return locked, probe, cfg, args


def simple_gated_model() -> FlatTubeProbeFeedbackTransformer:
    locked, probe, cfg, args = tiny_standard_models()
    anchor_time = torch.linspace(
        0.0, 1.0, cfg.n + 1, dtype=torch.float64
    )
    anchor_states = torch.full(
        (1, cfg.n + 1, cfg.m), cfg.n0, dtype=torch.float64
    )
    model = construct_gated_model(
        locked,
        probe,
        cfg,
        args,
        anchor_time=anchor_time,
        anchor_states=anchor_states,
        gate_tube=0.10,
        gate_transition=0.10,
    ).double()
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    model.set_feature_vectors(params["r"], params["phi"])
    return model


def formal_gated_model() -> tuple[
    FlatTubeProbeFeedbackTransformer,
    ProblemConfig,
    argparse.Namespace,
]:
    locked, probe, cfg, args = formal_tiny_standard_models()
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
    return model, cfg, args


def save_standard_pair(
    tmp_path: Path,
) -> tuple[Path, Path, ProblemConfig, argparse.Namespace]:
    locked, probe, cfg, args = formal_tiny_standard_models()
    common = {
        "args": vars(args),
        "problem": asdict(cfg),
        "nominal_reference": locked.nominal_reference.detach().clone(),
    }
    locked_path = tmp_path / "locked.pt"
    probe_path = tmp_path / "capability_probe.pt"
    torch.save(
        {**common, "model_state": cpu_state_dict(locked)}, locked_path
    )
    torch.save(
        {**common, "model_state": cpu_state_dict(probe)}, probe_path
    )
    return locked_path, probe_path, cfg, args


def test_c2_gate_has_exact_flat_regions_and_zero_join_derivatives() -> None:
    model = FlatTubeProbeFeedbackTransformer(
        m=1,
        umax=3.0,
        state_scale=15.0,
        state_hidden=(4,),
        d_model=8,
        heads=2,
        layers=1,
        init_u=1.5,
        correction_gain=1.0,
        state_feature_mode="log_absolute",
        center_state_correction=False,
        action_temperature=1.0,
        action_scale=1.0,
        action_parameterization="logit-temperature",
        action_offset=0.0,
        anchor_time=torch.tensor([0.0, 1.0], dtype=torch.float64),
        anchor_states=torch.ones((1, 2, 1), dtype=torch.float64),
        gate_normalization=1.0,
        gate_tube=0.2,
        gate_transition=0.2,
    ).double()
    time = torch.zeros(1, dtype=torch.float64)
    inside = torch.tensor([[1.1]], dtype=torch.float64)
    outside = torch.tensor([[1.5]], dtype=torch.float64)
    assert torch.equal(model.gate_values(time, inside), torch.zeros(1))
    assert torch.equal(model.gate_values(time, outside), torch.ones(1))

    for state_value in (1.2, 1.4):
        state = torch.tensor(
            [[state_value]], dtype=torch.float64, requires_grad=True
        )
        gate = model.gate_values(time, state)
        first = torch.autograd.grad(
            gate.sum(), state, create_graph=True
        )[0]
        second = torch.autograd.grad(first.sum(), state)[0]
        assert abs(float(first.detach())) < 1.0e-10
        assert abs(float(second.detach())) < 1.0e-8


def test_natural_cubic_anchor_makes_gate_c2_across_time_knots() -> None:
    anchors = torch.tensor(
        [[1.0, 1.2, 0.9, 1.1, 1.0]], dtype=torch.float64
    ).reshape(1, 5, 1)
    model = FlatTubeProbeFeedbackTransformer(
        m=1,
        umax=3.0,
        state_scale=15.0,
        state_hidden=(4,),
        d_model=8,
        heads=2,
        layers=1,
        init_u=1.5,
        correction_gain=1.0,
        state_feature_mode="log_absolute",
        center_state_correction=False,
        action_temperature=1.0,
        action_scale=1.0,
        action_parameterization="logit-temperature",
        action_offset=0.0,
        anchor_time=torch.linspace(0.0, 1.0, 5, dtype=torch.float64),
        anchor_states=anchors,
        gate_normalization=1.0,
        gate_tube=0.1,
        gate_transition=0.6,
    ).double()
    at_nodes = model._anchor_states_at(
        torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    ).squeeze(1)
    assert torch.allclose(
        at_nodes, anchors.squeeze(0), rtol=0.0, atol=2.0e-15
    )

    derivatives: list[tuple[float, float]] = []
    fixed_state = torch.tensor([[1.35]], dtype=torch.float64)
    for offset in (-1.0e-7, 1.0e-7):
        time = torch.tensor(
            [0.5 + offset], dtype=torch.float64, requires_grad=True
        )
        gate = model.gate_values(time, fixed_state)
        first = torch.autograd.grad(
            gate.sum(), time, create_graph=True
        )[0]
        second = torch.autograd.grad(first.sum(), time)[0]
        derivatives.append(
            (float(first.detach()), float(second.detach()))
        )
    assert math.isclose(
        derivatives[0][0], derivatives[1][0], rel_tol=0.0, abs_tol=2.0e-5
    )
    assert math.isclose(
        derivatives[0][1], derivatives[1][1], rel_tol=0.0, abs_tol=2.0e-4
    )


def test_centered_logits_are_composed_before_single_action_map() -> None:
    model = simple_gated_model()
    time = torch.tensor([0.35, 0.35], dtype=torch.float64)
    state = torch.tensor(
        [[10.0, 10.0, 10.0], [15.0, 7.0, 12.0]],
        dtype=torch.float64,
    )
    features = model.state_features(time, state)
    reference = model.nominal_state_at(time)
    reference_features = model.state_features(time, reference)
    locked = model._centered_branch_logits(
        model.state_branch,
        time,
        state,
        features=features,
        reference_features=reference_features,
    )
    probe = model._centered_branch_logits(
        model.probe_state_branch,
        time,
        state,
        features=features,
        reference_features=reference_features,
    )
    gate = model.gate_values(time, state)
    expected_logits = locked + gate * (probe - locked)
    assert torch.equal(model.state_logits(time, state), expected_logits)
    assert gate[0] == 0.0
    assert gate[1] == 1.0

    base = torch.tensor([0.1, -0.2], dtype=torch.float64)
    expected_action = model._action_from_correction(base, expected_logits)
    actual_action = model.interval_action(
        base, time, state, state_mode="feedback"
    )
    assert torch.equal(actual_action, expected_action)


def test_only_probe_branch_trains_while_state_gradient_is_preserved() -> None:
    model = simple_gated_model().train()
    assert_only_probe_trainable(model)
    state = torch.tensor(
        [[15.0, 7.0, 12.0]], dtype=torch.float64, requires_grad=True
    )
    action = model.interval_action(
        torch.tensor([0.0], dtype=torch.float64),
        torch.tensor([0.4], dtype=torch.float64),
        state,
        state_mode="feedback",
    )
    action.sum().backward()
    assert state.grad is not None
    assert torch.isfinite(state.grad).all()
    assert all(
        parameter.grad is None
        for parameter in model.time_branch.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in model.state_branch.parameters()
    )
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in model.probe_state_branch.parameters()
    )


def test_real_architecture_parameter_budget_is_full_state_branch_only() -> None:
    m = 21
    model = FlatTubeProbeFeedbackTransformer(
        m=m,
        umax=3.0,
        state_scale=15.0,
        state_hidden=(128, 128),
        d_model=64,
        heads=8,
        layers=2,
        init_u=1.5,
        correction_gain=1.0,
        state_feature_mode="log_absolute",
        center_state_correction=False,
        action_temperature=1.0,
        action_scale=1.0,
        action_parameterization="logit-temperature",
        action_offset=0.0,
        anchor_time=torch.tensor([0.0, 1.0]),
        anchor_states=torch.ones((5, 2, m)),
        gate_normalization=10.0,
        gate_tube=0.005,
        gate_transition=0.010,
    )
    inventory = parameter_inventory(model)
    assert inventory["time_branch_parameters"] == 100_481
    assert inventory["locked_state_branch_parameters"] == 20_609
    assert inventory["probe_state_branch_parameters"] == 20_609
    assert inventory["total_parameters"] == 141_699
    assert inventory["trainable_parameters"] == 20_609


def test_construct_preserves_external_float64_branches_and_source_rollout() -> None:
    locked, probe, cfg, args = tiny_standard_models()
    locked_time = {
        key: value.detach().clone()
        for key, value in locked.time_branch.state_dict().items()
    }
    locked_state = {
        key: value.detach().clone()
        for key, value in locked.state_branch.state_dict().items()
    }
    probe_state = {
        key: value.detach().clone()
        for key, value in probe.state_branch.state_dict().items()
    }
    anchor_time = torch.linspace(
        0.0, 1.0, cfg.n + 1, dtype=torch.float64
    )
    anchor_states = torch.full(
        (1, cfg.n + 1, cfg.m), cfg.n0, dtype=torch.float64
    )
    model = construct_gated_model(
        locked,
        probe,
        cfg,
        args,
        anchor_time=anchor_time,
        anchor_states=anchor_states,
        gate_tube=0.1,
        gate_transition=0.1,
    )
    assert next(model.parameters()).dtype == torch.float64
    for expected, actual in (
        (locked_time, model.time_branch.state_dict()),
        (locked_state, model.state_branch.state_dict()),
        (probe_state, model.probe_state_branch.state_dict()),
    ):
        assert set(expected) == set(actual)
        for key in expected:
            assert expected[key].dtype == actual[key].dtype
            assert torch.equal(expected[key], actual[key])
    for expected, actual in (
        (locked_time, locked.time_branch.state_dict()),
        (locked_state, locked.state_branch.state_dict()),
        (probe_state, probe.state_branch.state_dict()),
    ):
        for key in expected:
            assert torch.equal(expected[key], actual[key])

    params = build_params(cfg, torch.device("cpu"), torch.float64)
    model.set_feature_vectors(params["r"], params["phi"])
    time = torch.tensor([0.2, 0.7], dtype=torch.float64)
    state = torch.tensor(
        [[15.0, 7.0, 12.0], [12.0, 14.0, 7.0]],
        dtype=torch.float64,
    )
    base = locked.time_logits(time)
    assert torch.equal(base, model.time_logits(time))
    assert torch.equal(
        locked.interval_action(
            base, time, state, state_mode="feedback"
        ),
        model.interval_action(
            base, time, state, state_mode="locked_feedback"
        ),
    )

    normalized_time, raw = fixed_support_dense_logits(
        locked, cfg, 1, query_batch_size=2
    )
    midpoint_raw = pchip_midpoint_logits(normalized_time, raw)
    locked_states, locked_midpoints = continuous_feedback_state_rk4(
        locked,
        state,
        cfg,
        normalized_time,
        raw,
        midpoint_raw,
        params,
        state_mode="feedback",
    )
    model_states, model_midpoints = continuous_feedback_state_rk4(
        model,
        state,
        cfg,
        normalized_time,
        raw,
        midpoint_raw,
        params,
        state_mode="locked_feedback",
    )
    assert torch.equal(locked_states, model_states)
    assert torch.equal(locked_midpoints, model_midpoints)


def test_standard_anchor_family_is_coarse_node_cubic_not_fine_reintegration() -> None:
    locked, _, cfg, _ = tiny_standard_models()
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    protected = structured_initial_states(
        REQUIRED_PROTECTED_RADII,
        cfg,
        torch.device("cpu"),
        torch.float64,
    )
    multiplier = 2
    anchor_time, anchors = build_locked_anchors(
        locked,
        cfg,
        params,
        protected,
        multiplier=multiplier,
        query_batch_size=2,
    )
    family_count = len(REQUIRED_PROTECTED_RADII)
    assert anchors.shape[0] == ANCHOR_FAMILY_PROTOCOL["total"]
    standard_anchors = anchors[family_count:]

    base_states, _, _, _ = simulate_feedback_rk4_stagewise(
        locked,
        protected,
        cfg,
        params,
        state_mode="feedback",
    )
    base_second = natural_cubic_second_derivatives(base_states)
    expected = natural_cubic_uniform_value(
        base_states,
        base_second,
        anchor_time,
    ).permute(1, 0, 2).contiguous()
    expected[:, ::multiplier, :] = base_states
    assert torch.equal(standard_anchors, expected)
    assert torch.equal(standard_anchors[:, ::multiplier, :], base_states)

    fine_cfg = fine_problem(cfg, multiplier)
    fine_reintegration, _, _, _ = simulate_feedback_rk4_stagewise(
        locked,
        protected,
        fine_cfg,
        params,
        state_mode="feedback",
    )
    assert standard_anchors.shape == fine_reintegration.shape
    assert not torch.equal(standard_anchors, fine_reintegration)


def test_checkpoint_roundtrip_is_self_contained_and_legacy_fails_closed(
    tmp_path: Path,
) -> None:
    model, cfg, args = formal_gated_model()
    model.eval()
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
                "protected_standard_node_state_max_abs": 0.0,
                "protected_standard_stage_state_max_abs": 0.0,
                "protected_standard_objective_max_abs": 0.0,
            },
            "posthoc_identity_reference": {
                "path": "unused",
                "sha256": "a" * 64,
                "evaluator": STANDARD_BASE_GRID_PROTOCOL,
            },
            "train_seed": PMP_TRAIN_SEED,
            "validation_seed": PMP_VALIDATION_SEED,
            "optimizer_seed": OPTIMIZER_SEED,
            "reserved_blind_seed": RESERVED_BLIND_SEED,
            "reserved_blind_used_for_training_or_selection": False,
            "locked_checkpoint": {
                "path": "unused",
                "sha256": "a" * 64,
            },
            "protected_radii": [0.0, 0.1, 0.2, 0.4, 0.6],
            "gate_normalization": cfg.n0,
            "gate_tube": model.gate_tube,
            "gate_transition": model.gate_transition,
        },
    }
    checkpoint = tmp_path / "gated.pt"
    torch.save(payload, checkpoint)
    reloaded, reload_cfg, _, reload_payload = (
        load_gated_feedback_checkpoint(checkpoint)
    )
    assert reload_cfg == cfg
    assert reload_payload["checkpoint_format"] == CHECKPOINT_FORMAT
    assert all(
        torch.equal(value, reloaded.state_dict()[key])
        for key, value in model.state_dict().items()
    )
    time = torch.tensor([0.2, 0.8], dtype=torch.float64)
    state = torch.tensor(
        [[10.0, 10.0, 10.0], [14.0, 7.0, 12.0]],
        dtype=torch.float64,
    )
    base = torch.tensor([0.2, -0.1], dtype=torch.float64)
    assert torch.equal(
        model.interval_action(base, time, state, state_mode="feedback"),
        reloaded.interval_action(
            base, time, state, state_mode="feedback"
        ),
    )
    with pytest.raises((KeyError, RuntimeError, TypeError, ValueError)):
        load_standard_feedback_checkpoint(checkpoint)

    incomplete = copy.deepcopy(payload)
    incomplete["gated_fallback"].pop("posthoc_identity_audit_completed")
    incomplete["gated_fallback"].pop("posthoc_protected_identity")
    incomplete_checkpoint = tmp_path / "provisional.pt"
    torch.save(incomplete, incomplete_checkpoint)
    with pytest.raises(ValueError, match="posthoc identity audit"):
        load_gated_feedback_checkpoint(incomplete_checkpoint)
    load_gated_feedback_checkpoint(
        incomplete_checkpoint,
        _allow_incomplete_posthoc=True,
    )

    for field, bad_value in (
        (
            "anchor_families",
            {
                **ANCHOR_FAMILY_PROTOCOL,
                "total": int(ANCHOR_FAMILY_PROTOCOL["total"]) - 1,
            },
        ),
        (
            "standard_base_grid",
            {**STANDARD_BASE_GRID_PROTOCOL, "n": 400},
        ),
        ("base_n", 400),
    ):
        malformed = copy.deepcopy(payload)
        malformed["gated_fallback"][field] = bad_value
        malformed_checkpoint = tmp_path / f"bad_{field}.pt"
        torch.save(malformed, malformed_checkpoint)
        with pytest.raises(ValueError, match="protocol|base_n|base-grid"):
            load_gated_feedback_checkpoint(malformed_checkpoint)


def test_pmp_cleanup_pack_has_no_physical_objective_and_backpropagates() -> None:
    model = simple_gated_model().train()
    cfg = tiny_cfg()
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    normalized_time, raw = fixed_support_dense_logits(
        model, cfg, 1, query_batch_size=2
    )
    midpoint_raw = pchip_midpoint_logits(normalized_time, raw)
    initial = torch.tensor(
        [[15.0, 7.0, 12.0], [12.0, 14.0, 7.0]],
        dtype=torch.float64,
    )
    pack = pmp_residual_pack(
        model,
        initial,
        cfg,
        normalized_time,
        raw,
        midpoint_raw,
        params,
        continuous_collocation="nodes",
        option="der",
        interval_start=0.0,
        interval_end=cfg.T,
        w0=1.0,
        w1=1.0,
        w2=4.0,
        cf_weight=0.0,
        cf_scalar_weight=0.0,
    )
    assert "objective" not in pack
    assert math.isfinite(float(pack["loss"].detach()))
    pack["loss"].backward()
    assert any(
        parameter.grad is not None
        for parameter in model.probe_state_branch.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in model.time_branch.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in model.state_branch.parameters()
    )
    with pytest.raises(ValueError, match="no collocation samples"):
        pmp_residual_pack(
            model,
            initial,
            cfg,
            normalized_time,
            raw,
            midpoint_raw,
            params,
            continuous_collocation="nodes",
            option="der",
            interval_start=0.11,
            interval_end=0.12,
            w0=1.0,
            w1=1.0,
            w2=4.0,
            cf_weight=0.0,
            cf_scalar_weight=0.0,
        )


def test_stage_trace_covers_all_actual_rk4_policy_queries() -> None:
    model = simple_gated_model().eval()
    model.gate_tube = 10.0
    model.gate_transition = 0.1
    cfg = tiny_cfg()
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    normalized_time, raw = fixed_support_dense_logits(
        model, cfg, 1, query_batch_size=2
    )
    midpoint_raw = pchip_midpoint_logits(normalized_time, raw)
    initial = torch.tensor(
        [[15.0, 7.0, 12.0], [12.0, 14.0, 7.0]],
        dtype=torch.float64,
    )
    locked = continuous_policy_stage_trace(
        model,
        initial,
        cfg,
        normalized_time,
        raw,
        midpoint_raw,
        params,
        state_mode="locked_feedback",
    )
    candidate = continuous_policy_stage_trace(
        model,
        initial,
        cfg,
        normalized_time,
        raw,
        midpoint_raw,
        params,
        state_mode="feedback",
    )
    reference_states, _ = continuous_feedback_state_rk4(
        model,
        initial,
        cfg,
        normalized_time,
        raw,
        midpoint_raw,
        params,
        state_mode="feedback",
    )
    assert candidate["states"].shape == (2, cfg.n, 4, cfg.m)
    assert candidate["controls"].shape == (2, cfg.n, 4)
    assert candidate["normalized_time"].shape == (cfg.n, 4)
    assert torch.equal(candidate["states"], locked["states"])
    assert torch.equal(candidate["controls"], locked["controls"])
    assert torch.equal(
        candidate["terminal_state"], locked["terminal_state"]
    )
    assert torch.equal(candidate["states"][:, :, 0], reference_states[:, :-1])
    assert torch.equal(candidate["terminal_state"], reference_states[:, -1])
    flattened_gate = model.gate_values(
        candidate["normalized_time"]
        .unsqueeze(0)
        .expand(initial.shape[0], -1, -1)
        .reshape(-1),
        candidate["states"].reshape(-1, cfg.m),
    )
    assert torch.equal(flattened_gate, torch.zeros_like(flattened_gate))


def test_protocol_rejects_extra_radius_and_any_blind_seed_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="exactly"):
        require_protocol_radii((0.0, 0.1, 0.2, 0.4, 0.6, 0.8))

    nonzero_tolerance = parser().parse_args(
        [
            "--locked-checkpoint",
            str(tmp_path / "missing_locked.pt"),
            "--capability-probe-checkpoint",
            str(tmp_path / "missing_probe.pt"),
            "--out-dir",
            str(tmp_path / "unused"),
            "--protected-tolerance",
            "1e-12",
        ]
    )
    with pytest.raises(ValueError, match="exact zero"):
        run(nonzero_tolerance)

    seed_calls: list[int] = []
    monkeypatch.setattr(
        torch,
        "manual_seed",
        lambda seed: seed_calls.append(int(seed)),
    )
    args = parser().parse_args(
        [
            "--locked-checkpoint",
            str(tmp_path / "missing_locked.pt"),
            "--capability-probe-checkpoint",
            str(tmp_path / "missing_probe.pt"),
            "--out-dir",
            str(tmp_path / "unused"),
            "--optimizer-seed",
            str(RESERVED_BLIND_SEED),
        ]
    )
    with pytest.raises(ValueError, match="seed protocol mismatch"):
        run(args)
    assert seed_calls == []


def test_pair_validation_requires_centered_phenotype_resolved_probe() -> None:
    locked, probe, cfg, args = tiny_standard_models()
    uncentered = copy.deepcopy(args)
    uncentered.center_state_correction = False
    with pytest.raises(ValueError, match="centered"):
        validate_checkpoint_pair(
            locked, probe, cfg, cfg, uncentered, uncentered
        )
    total_only = copy.deepcopy(args)
    total_only.state_feature_mode = "total_burden"
    with pytest.raises(ValueError, match="phenotype-resolved"):
        validate_checkpoint_pair(
            locked, probe, cfg, cfg, total_only, total_only
        )


def test_cpu_smoke_reports_step0_pmp_and_never_touches_blind_seed(
    tmp_path: Path,
) -> None:
    locked, probe, _, _ = save_standard_pair(tmp_path)
    output = tmp_path / "smoke_output"
    args = parser().parse_args(
        [
            "--locked-checkpoint",
            str(locked),
            "--capability-probe-checkpoint",
            str(probe),
            "--out-dir",
            str(output),
            "--device",
            "cpu",
            "--threads",
            "1",
            "--option",
            "der",
            "--smoke",
            "--smoke-epochs",
            "1",
            "--smoke-random-states",
            "2",
            "--smoke-pmp-multiplier",
            "1",
            "--smoke-anchor-multiplier",
            "1",
            "--interval-start",
            "0",
            "--interval-end",
            "1",
            "--gate-tube",
            "1.0",
            "--gate-transition",
            "0.1",
            "--protected-tolerance",
            "0",
            "--train-seed",
            "20260810",
            "--validation-seed",
            "20260811",
            "--blind-seed",
            str(RESERVED_BLIND_SEED),
        ]
    )
    summary = run(args)
    assert summary["mode"] == "smoke"
    assert summary["step0"]["step"] == 0
    assert math.isfinite(summary["step0"]["train_loss"])
    assert math.isfinite(summary["step0"]["validation_loss"])
    assert math.isfinite(summary["step0"]["train_selection_pmp"])
    assert math.isfinite(summary["step0"]["validation_selection_pmp"])
    assert summary["step0"]["guard_passed"] is True
    for key in (
        "protected_state_max_abs",
        "protected_node_control_max_abs",
        "protected_midpoint_control_max_abs",
        "protected_gate_max",
        "protected_stage_state_max_abs",
        "protected_stage_control_max_abs",
        "protected_stage_gate_max",
        "protected_standard_gate_max",
        "protected_standard_control_max_abs",
        "protected_standard_stage_control_max_abs",
        "protected_standard_node_state_max_abs",
        "protected_standard_stage_state_max_abs",
    ):
        assert summary["step0"][key] == 0.0
    assert summary["selected_step"] == 0
    assert summary["native_gated_reload_passed"] is True
    assert summary["legacy_loader_failed_closed"] is True
    assert summary["physical_objective_used_in_loss"] is False
    assert summary["physical_objective_used_in_selection"] is False
    assert summary["physical_objective_computed_posthoc"] is True
    assert all(
        value == 0.0
        for value in summary["posthoc_protected_identity"].values()
    )
    assert summary["reserved_blind_seed"] == RESERVED_BLIND_SEED
    assert summary["reserved_blind_used"] is False
    assert str(RESERVED_BLIND_SEED) not in {
        str(args.train_seed),
        str(args.validation_seed),
        str(args.optimizer_seed),
    }
