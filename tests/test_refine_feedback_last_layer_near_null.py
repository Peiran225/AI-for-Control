from __future__ import annotations

import torch

from scripts.refine_feedback_last_layer_near_null import (
    ScalarPMPFunctionalModule,
    build_parser,
    continuous_rk4_stage_feature_matrix,
    hidden_feature_differences,
    iid_and_composition_states,
    install_near_null_parametrization,
    materialize_near_null_weight,
    numerical_near_nullspace,
    parse_structured_radii,
    protected_metrics_pass,
)
from scripts.constrained_scalar_lm_step import forward_jacobian_columns
from scripts.feedback_continuous_policy_rk4 import pchip_midpoint_logits
from scripts.train_feedback_section5 import NestedFeedbackTransformer
from train_paper_pmp_kkt import ProblemConfig, build_params


def make_model() -> NestedFeedbackTransformer:
    model = NestedFeedbackTransformer(
        m=5,
        umax=3.0,
        state_scale=15.0,
        state_hidden=(8,),
        d_model=8,
        heads=2,
        layers=1,
        init_u=1.5,
        state_feature_mode="relative_nominal",
        center_state_correction=True,
    ).double()
    model.set_nominal_reference(
        torch.full((9, 5), 10.0, dtype=torch.float64)
    )
    return model


def test_numerical_near_nullspace_includes_exact_wide_nullspace() -> None:
    matrix = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    basis, singular_values, rank = numerical_near_nullspace(
        matrix,
        1.0e-12,
    )
    assert rank == 2
    assert singular_values.shape == (2,)
    assert basis.shape == (4, 2)
    torch.testing.assert_close(
        matrix @ basis,
        torch.zeros((2, 2), dtype=torch.float64),
    )


def test_parser_keeps_lbfgs_default_and_accepts_constrained_lm() -> None:
    parser = build_parser()
    default_args = parser.parse_args(
        ["--checkpoint", "in.pt", "--out-dir", "out"]
    )
    assert default_args.optimizer == "lbfgs"
    lm_args = parser.parse_args(
        [
            "--checkpoint",
            "in.pt",
            "--out-dir",
            "out",
            "--optimizer",
            "constrained-lm",
        ]
    )
    assert lm_args.optimizer == "constrained-lm"


def test_scalar_pmp_functional_module_supports_near_null_jvp() -> None:
    model = make_model()
    cfg = ProblemConfig(
        T=2.0,
        n=8,
        m=5,
        umax=3.0,
        beta=0.1,
        alpha=0.0025,
        gamma=20.0,
        n0=10.0,
        m_suppression=0.5,
    )
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    time = torch.linspace(0.0, 1.0, cfg.n + 1, dtype=torch.float64)
    raw = model.time_logits(time).detach()
    midpoint_raw = pchip_midpoint_logits(time, raw)
    final_layer = model.state_branch[-1]
    assert isinstance(final_layer, torch.nn.Linear)
    parametrization = install_near_null_parametrization(
        final_layer,
        torch.eye(final_layer.in_features, dtype=torch.float64),
    )
    args = type(
        "Args",
        (),
        {"interval_start": 0.5, "interval_end": 1.5},
    )()
    functional = ScalarPMPFunctionalModule(
        model,
        cfg,
        time,
        raw,
        midpoint_raw,
        params,
        args,
    )
    theta_name = next(
        name
        for name, parameter in functional.named_parameters()
        if parameter is parametrization.theta
    )
    states = torch.stack(
        (
            torch.full((cfg.m,), 10.0, dtype=torch.float64),
            torch.linspace(9.0, 11.0, cfg.m, dtype=torch.float64),
        )
    )
    weights = torch.ones(2, dtype=torch.float64)
    initial = parametrization.theta.detach().reshape(-1)

    def residual(theta: torch.Tensor) -> torch.Tensor:
        value = torch.func.functional_call(
            functional,
            {theta_name: theta.reshape_as(parametrization.theta)},
            (states, weights, "residual"),
        )
        assert isinstance(value, torch.Tensor)
        return value

    value, jacobian = forward_jacobian_columns(residual, initial)
    assert value.ndim == 1
    assert jacobian.shape == (value.numel(), initial.numel())
    assert bool(torch.isfinite(jacobian).all())

    def protected(theta: torch.Tensor) -> torch.Tensor:
        value = torch.func.functional_call(
            functional,
            {theta_name: theta.reshape_as(parametrization.theta)},
            (states, weights, "lopt"),
        )
        assert isinstance(value, torch.Tensor)
        return value

    protected_jacobian = torch.autograd.functional.jacobian(
        protected,
        initial.requires_grad_(True),
        vectorize=True,
        strategy="reverse-mode",
    )
    assert protected_jacobian.shape == (states.shape[0], initial.numel())
    assert bool(torch.isfinite(protected_jacobian).all())
    # Functional trial evaluations must not mutate the live checkpoint.
    torch.testing.assert_close(
        parametrization.theta,
        torch.zeros_like(parametrization.theta),
    )


def test_structured_radii_parser_preserves_legacy_and_accepts_multiple() -> None:
    assert parse_structured_radii("", 0.10) == (0.10,)
    assert parse_structured_radii("0.10, 0.20", 0.05) == (0.10, 0.20)
    try:
        parse_structured_radii("0.10,0.10", 0.10)
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate protected radii must be rejected")


def test_random_state_sampler_supports_heldout_radius_point_two() -> None:
    cfg = ProblemConfig(
        T=2.0,
        n=8,
        m=5,
        umax=3.0,
        beta=0.1,
        alpha=0.0025,
        gamma=20.0,
        n0=10.0,
        m_suppression=0.5,
    )
    states, weights, counts = iid_and_composition_states(
        8,
        123,
        0.20,
        cfg,
        torch.device("cpu"),
        torch.float64,
    )
    assert states.shape == (8, 5)
    assert counts == (4, 4)
    assert float(states.min()) >= 8.0
    assert float(states.max()) <= 12.0
    one = torch.tensor(1.0, dtype=torch.float64)
    torch.testing.assert_close(weights[:4].sum(), one)
    torch.testing.assert_close(weights[4:].sum(), one)
    torch.testing.assert_close(
        states[4:].sum(dim=1),
        torch.full((4,), 50.0, dtype=torch.float64),
    )


def test_multiple_structured_trajectories_stack_stage_feature_rows() -> None:
    model = make_model()
    cfg = ProblemConfig(
        T=2.0,
        n=8,
        m=5,
        umax=3.0,
        beta=0.1,
        alpha=0.0025,
        gamma=20.0,
        n0=10.0,
        m_suppression=0.5,
    )
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    time = torch.linspace(0.0, 1.0, cfg.n + 1, dtype=torch.float64)
    raw = model.time_logits(time).detach()
    midpoint_raw = pchip_midpoint_logits(time, raw)
    direction = torch.linspace(-1.0, 1.0, cfg.m, dtype=torch.float64)
    matrices = []
    for radius in (0.10, 0.20):
        matrices.append(
            continuous_rk4_stage_feature_matrix(
                model,
                (cfg.n0 * (1.0 + radius * direction)).unsqueeze(0),
                cfg,
                time,
                raw,
                midpoint_raw,
                params,
                interval_start=0.5,
                interval_end=1.5,
            )
        )
    stacked = torch.cat(matrices)
    assert matrices[0].shape == matrices[1].shape
    assert stacked.shape[0] == 2 * matrices[0].shape[0]
    assert stacked.shape[1] == 8


def test_protected_gate_checks_every_structured_radius() -> None:
    control_drift = torch.tensor(
        [0.0, 1.0e-9, 2.0e-8],
        dtype=torch.float64,
    )
    lopt = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    limits = lopt + 1.0e-6
    assert not protected_metrics_pass(
        torch.tensor(1.0e-12, dtype=torch.float64),
        control_drift,
        lopt,
        limits,
        max_nominal_control_drift=1.0e-12,
        max_structured_control_drift=1.0e-8,
        max_structured_prelogit_drift=1.0e-8,
    )
    control_drift[-1] = 2.0e-9
    assert protected_metrics_pass(
        torch.tensor(1.0e-12, dtype=torch.float64),
        control_drift,
        lopt,
        limits,
        max_nominal_control_drift=1.0e-12,
        max_structured_control_drift=1.0e-8,
        max_structured_prelogit_drift=1.0e-8,
    )


def test_materialized_update_preserves_protected_logits_and_only_changes_weight() -> None:
    torch.manual_seed(42)
    model = make_model()
    final_layer = model.state_branch[-1]
    assert isinstance(final_layer, torch.nn.Linear)
    before = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }

    protected_time = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)
    protected_state = torch.stack(
        [
            torch.linspace(9.0, 11.0, 5, dtype=torch.float64),
            torch.linspace(9.1, 10.9, 5, dtype=torch.float64),
            torch.linspace(9.2, 10.8, 5, dtype=torch.float64),
        ]
    )
    protected_matrix = hidden_feature_differences(
        model,
        protected_time,
        protected_state,
    ).detach()
    basis, _, _ = numerical_near_nullspace(
        protected_matrix,
        1.0e-12,
    )
    assert basis.shape[1] > 0

    random_time = torch.tensor([0.3, 0.6], dtype=torch.float64)
    random_state = torch.tensor(
        [
            [8.7, 10.9, 9.4, 10.5, 10.1],
            [10.8, 8.9, 10.4, 9.6, 10.3],
        ],
        dtype=torch.float64,
    )
    random_matrix = hidden_feature_differences(
        model,
        random_time,
        random_state,
    ).detach()
    retained_energy = (random_matrix @ basis).norm()
    assert retained_energy > 0.0

    parametrization = install_near_null_parametrization(
        final_layer,
        basis,
    )
    with torch.no_grad():
        parametrization.theta.normal_(mean=0.0, std=0.05)
    materialize_near_null_weight(final_layer)

    weight_key = "state_branch.2.weight"
    delta_weight = model.state_dict()[weight_key] - before[weight_key]
    assert delta_weight.norm() > 0.0
    torch.testing.assert_close(
        protected_matrix @ delta_weight.T,
        torch.zeros(
            protected_matrix.shape[0],
            1,
            dtype=torch.float64,
        ),
        atol=1.0e-12,
        rtol=0.0,
    )
    changed = [
        key
        for key, value in model.state_dict().items()
        if not torch.equal(value, before[key])
    ]
    assert changed == [weight_key]
    # The checkpoint is standard again: no parametrization-only state keys.
    assert all("parametrizations" not in key for key in model.state_dict())
