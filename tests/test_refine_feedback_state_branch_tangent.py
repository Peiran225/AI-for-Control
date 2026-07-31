from __future__ import annotations

import torch

from scripts.feedback_continuous_policy_rk4 import pchip_midpoint_logits
from scripts.refine_feedback_state_branch_tangent import (
    assign_parameter_vector,
    continuous_rk4_stage_controls,
    parameter_vector,
    parse_structured_radii,
    preservation_gate,
    project_onto_constraint_tangent,
    protected_initial_states,
)
from scripts.train_feedback_section5 import NestedFeedbackTransformer
from train_paper_pmp_kkt import ProblemConfig, build_params


def make_model_and_problem():
    cfg = ProblemConfig(
        T=2.0,
        n=4,
        m=5,
        umax=3.0,
        beta=0.1,
        alpha=0.0025,
        gamma=20.0,
        n0=10.0,
        m_suppression=0.5,
    )
    model = NestedFeedbackTransformer(
        m=cfg.m,
        umax=cfg.umax,
        state_scale=15.0,
        state_hidden=(8,),
        d_model=8,
        heads=2,
        layers=1,
        init_u=1.5,
        state_feature_mode="burden_composition",
        center_state_correction=True,
    ).double()
    model.set_nominal_reference(
        torch.full((cfg.n + 1, cfg.m), 10.0, dtype=torch.float64)
    )
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    model.set_feature_vectors(params["r"], params["phi"])
    return model, cfg, params


def test_projected_gradient_is_orthogonal_to_constraint_rows() -> None:
    torch.manual_seed(7)
    gradient = torch.randn(11, dtype=torch.float64)
    constraints = torch.randn(4, 11, dtype=torch.float64)
    projected, rank, singular_values = project_onto_constraint_tangent(
        gradient,
        constraints,
        1.0e-12,
    )
    assert rank == 4
    assert singular_values.shape == (4,)
    torch.testing.assert_close(
        constraints @ projected,
        torch.zeros(4, dtype=torch.float64),
        atol=1.0e-12,
        rtol=0.0,
    )
    assert projected.norm() <= gradient.norm()


def test_parameter_vector_round_trip() -> None:
    model, _, _ = make_model_and_problem()
    parameters = list(model.state_branch.parameters())
    before = parameter_vector(parameters)
    update = torch.linspace(
        -1.0e-4,
        1.0e-4,
        before.numel(),
        dtype=torch.float64,
    )
    assign_parameter_vector(parameters, before + update)
    torch.testing.assert_close(
        parameter_vector(parameters),
        before + update,
    )
    assign_parameter_vector(parameters, before)
    torch.testing.assert_close(parameter_vector(parameters), before)


def test_full_horizon_stage_controls_have_four_rk4_queries() -> None:
    model, cfg, params = make_model_and_problem()
    states = protected_initial_states(
        cfg,
        (0.20,),
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    time = torch.linspace(0.0, 1.0, cfg.n + 1, dtype=torch.float64)
    raw = model.time_logits(time)
    midpoint_raw = pchip_midpoint_logits(time, raw)
    controls = continuous_rk4_stage_controls(
        model,
        states,
        cfg,
        time,
        raw,
        midpoint_raw,
        params,
    )
    assert controls.shape == (2, cfg.n, 4)
    assert torch.isfinite(controls).all()
    assert ((controls >= 0.0) & (controls <= cfg.umax)).all()


def test_structured_radius_parser_and_states() -> None:
    model, cfg, _ = make_model_and_problem()
    del model
    radii = parse_structured_radii("0.10, 0.20")
    assert radii == (0.10, 0.20)
    states = protected_initial_states(
        cfg,
        radii,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert states.shape == (3, cfg.m)
    torch.testing.assert_close(
        states.sum(dim=1),
        torch.full((3,), cfg.n0 * cfg.m, dtype=torch.float64),
    )


def test_preservation_gate_checks_every_protected_row() -> None:
    drift = torch.tensor([1.0e-5, 2.0e-5, 3.0e-5], dtype=torch.float64)
    lopt = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    limits = torch.tensor([1.01, 2.02, 3.03], dtype=torch.float64)
    assert preservation_gate(
        drift,
        lopt,
        limits,
        max_nominal_control_drift=1.0e-4,
        max_structured_control_drift=1.0e-4,
    )
    lopt[2] = 3.04
    assert not preservation_gate(
        drift,
        lopt,
        limits,
        max_nominal_control_drift=1.0e-4,
        max_structured_control_drift=1.0e-4,
    )
