from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from train_feedback_section5 import (
    NestedFeedbackTransformer,
    compose_section5_optimality_loss,
    compute_costate,
    compute_costate_rk4,
    load_operational_time_control,
    objective_per_sample,
    persistence_gate,
    sample_componentwise_initial_states,
    rk4_objective_per_sample,
    section5_loss,
    simulate_feedback,
    simulate_open_loop,
    smoothness_components,
    singular_quantities,
)
from train_paper_pmp_kkt import ProblemConfig, build_params


def make_cfg(n: int = 12) -> ProblemConfig:
    return ProblemConfig(
        T=2.0,
        n=n,
        m=21,
        umax=3.0,
        beta=0.1,
        alpha=1.0,
        gamma=20.0,
        n0=10.0,
        m_suppression=0.5,
    )


def make_model(cfg: ProblemConfig) -> NestedFeedbackTransformer:
    return NestedFeedbackTransformer(
        cfg.m, cfg.umax, 15.0, (16,), 16, 4, 1, 1.5
    ).double()


def test_componentwise_sampler_is_bounded_and_not_a_common_scale() -> None:
    cfg = make_cfg()
    states = sample_componentwise_initial_states(
        128, cfg, 0.1, torch.device("cpu"), torch.float64
    )
    assert torch.all(states >= 9.0)
    assert torch.all(states <= 11.0)
    assert torch.any(states.std(dim=-1) > 0.1)


def test_zero_state_head_reproduces_time_branch() -> None:
    cfg = make_cfg()
    model = make_model(cfg).eval()
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    initial = torch.full((3, cfg.m), cfg.n0, dtype=torch.float64)
    with torch.no_grad():
        _, controls, _ = simulate_feedback(model, initial, cfg, params)
        grid = torch.linspace(0.0, 1.0, cfg.n + 1, dtype=torch.float64)
        expected = model.time_branch(grid)[: cfg.n]
    assert torch.allclose(controls, expected.expand_as(controls), atol=1e-12, rtol=1e-12)


def test_w_zero_bypasses_a_nonzero_state_branch_exactly() -> None:
    cfg = make_cfg()
    model = make_model(cfg).eval()
    final = [
        module
        for module in model.state_branch.modules()
        if isinstance(module, torch.nn.Linear)
    ][-1]
    with torch.no_grad():
        final.weight.fill_(0.25)
        final.bias.fill_(0.4)
    state = torch.stack(
        [torch.full((cfg.m,), 7.0), torch.full((cfg.m,), 13.0)]
    ).double()
    time = torch.tensor([0.25, 0.25], dtype=torch.float64)
    base = torch.tensor(0.3, dtype=torch.float64)
    action = model.interval_action(base, time, state, state_mode="w_zero")
    expected = cfg.umax * torch.sigmoid(base)
    assert torch.equal(action, expected.expand_as(action))


def test_lower_action_temperature_sharpens_smooth_bound_approach() -> None:
    cfg = make_cfg()
    regular = make_model(cfg).eval()
    sharpened = NestedFeedbackTransformer(
        cfg.m,
        cfg.umax,
        15.0,
        (16,),
        16,
        4,
        1,
        1.5,
        action_temperature=0.75,
    ).double().eval()
    state = torch.full((2, cfg.m), cfg.n0, dtype=torch.float64)
    time = torch.tensor([0.25, 0.25], dtype=torch.float64)
    logits = torch.tensor([-2.0, 2.0], dtype=torch.float64)
    regular_action = regular.interval_action(
        logits, time, state, state_mode="w_zero"
    )
    sharpened_action = sharpened.interval_action(
        logits, time, state, state_mode="w_zero"
    )
    assert torch.all((sharpened_action > 0.0) & (sharpened_action < cfg.umax))
    assert sharpened_action[0] < regular_action[0]
    assert sharpened_action[1] > regular_action[1]


def test_relative_nominal_features_center_the_reference_trajectory() -> None:
    cfg = make_cfg(n=4)
    model = NestedFeedbackTransformer(
        cfg.m,
        cfg.umax,
        15.0,
        (16,),
        16,
        4,
        1,
        1.5,
        state_feature_mode="relative_nominal",
    ).double()
    reference = torch.stack(
        [torch.full((cfg.m,), 10.0 + index, dtype=torch.float64) for index in range(5)]
    )
    model.set_nominal_reference(reference)
    time = torch.tensor([0.5], dtype=torch.float64)
    centered = model.state_features(time, reference[2:3])
    shifted = model.state_features(time, 1.1 * reference[2:3])
    assert torch.equal(centered[:, 6:], torch.zeros_like(centered[:, 6:]))
    assert torch.allclose(shifted[:, 6 : 6 + cfg.m], torch.full((1, cfg.m), 0.1, dtype=torch.float64))


def test_burden_composition_features_separate_scale_and_composition() -> None:
    cfg = make_cfg(n=4)
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    model = NestedFeedbackTransformer(
        cfg.m,
        cfg.umax,
        15.0,
        (16,),
        16,
        4,
        1,
        1.5,
        state_feature_mode="burden_composition",
    ).double()
    reference = torch.stack(
        [torch.linspace(8.0, 12.0, cfg.m, dtype=torch.float64) for _ in range(5)]
    )
    model.set_nominal_reference(reference)
    model.set_feature_vectors(params["r"], params["phi"])
    time = torch.tensor([0.5], dtype=torch.float64)
    centered = model.state_features(time, reference[2:3])[:, 6:]
    scaled = model.state_features(time, 1.1 * reference[2:3])[:, 6:]
    assert torch.allclose(centered, torch.zeros_like(centered), atol=1e-14, rtol=0.0)
    assert torch.allclose(
        scaled[:, : cfg.m], torch.zeros_like(scaled[:, : cfg.m]), atol=1e-14, rtol=0.0
    )
    assert torch.allclose(
        scaled[:, cfg.m], torch.tensor(math.log(1.1), dtype=torch.float64), atol=1e-14
    )
    assert torch.allclose(
        scaled[:, cfg.m + 1 :], torch.zeros_like(scaled[:, cfg.m + 1 :]), atol=1e-14
    )


def test_centered_state_correction_is_exactly_zero_on_the_nominal_path() -> None:
    cfg = make_cfg(n=4)
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    model = NestedFeedbackTransformer(
        cfg.m,
        cfg.umax,
        15.0,
        (16,),
        16,
        4,
        1,
        1.5,
        state_feature_mode="burden_composition",
        center_state_correction=True,
    ).double()
    reference = torch.stack(
        [torch.linspace(8.0, 12.0, cfg.m, dtype=torch.float64) for _ in range(5)]
    )
    model.set_nominal_reference(reference)
    model.set_feature_vectors(params["r"], params["phi"])
    with torch.no_grad():
        for parameter in model.state_branch.parameters():
            parameter.normal_(mean=0.2, std=0.1)
    time = torch.tensor([0.5], dtype=torch.float64)
    base = torch.tensor(0.3, dtype=torch.float64)
    nominal_action = model.interval_action(base, time, reference[2:3])
    expected = cfg.umax * torch.sigmoid(base)
    assert torch.equal(nominal_action, expected.expand_as(nominal_action))
    shifted_action = model.interval_action(base, time, 1.1 * reference[2:3])
    assert not torch.equal(shifted_action, nominal_action)


def test_nested_feedback_checkpoint_can_supply_a_time_only_warm_start(tmp_path: Path) -> None:
    cfg = make_cfg(n=4)
    model = make_model(cfg)
    checkpoint = tmp_path / "nested.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "args": {
                "d_model": 16,
                "heads": 4,
                "layers": 1,
                "init_u": 1.5,
            },
            "problem": cfg.__dict__,
        },
        checkpoint,
    )
    loaded = load_operational_time_control(
        checkpoint, cfg, torch.device("cpu"), torch.float64
    )
    grid = torch.linspace(0.0, 1.0, cfg.n + 1, dtype=torch.float64)
    expected = model.time_branch(grid)[: cfg.n]
    assert torch.equal(loaded, expected)


def test_paired_training_sampler_uses_an_independent_reproducible_stream() -> None:
    cfg = make_cfg()
    first = sample_componentwise_initial_states(
        8,
        cfg,
        0.1,
        torch.device("cpu"),
        torch.float64,
        generator=torch.Generator().manual_seed(1234),
    )
    second = sample_componentwise_initial_states(
        8,
        cfg,
        0.1,
        torch.device("cpu"),
        torch.float64,
        generator=torch.Generator().manual_seed(1234),
    )
    third = sample_componentwise_initial_states(
        8,
        cfg,
        0.1,
        torch.device("cpu"),
        torch.float64,
        generator=torch.Generator().manual_seed(1235),
    )
    assert torch.equal(first, second)
    assert not torch.equal(first, third)


def test_frozen_time_branch_accumulates_no_gradient_or_adam_state() -> None:
    cfg = make_cfg(n=4)
    model = make_model(cfg)
    time_parameters = list(model.time_branch.parameters())
    for parameter in time_parameters:
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    grid = torch.linspace(0.0, 1.0, cfg.n + 1, dtype=torch.float64)
    state = torch.full((2, cfg.m), cfg.n0, dtype=torch.float64)
    action = model.interval_action(
        model.time_logits(grid)[0],
        torch.zeros(2, dtype=torch.float64),
        state,
    )
    action.sum().backward()
    optimizer.step()
    assert all(parameter.grad is None for parameter in time_parameters)
    assert all(parameter not in optimizer.state for parameter in time_parameters)
    for parameter in time_parameters:
        parameter.requires_grad_(True)
    optimizer.zero_grad(set_to_none=True)
    action = model.interval_action(
        model.time_logits(grid)[0],
        torch.zeros(2, dtype=torch.float64),
        state,
    )
    action.sum().backward()
    optimizer.step()
    assert any(parameter in optimizer.state for parameter in time_parameters)


def test_w_zero_has_no_gradient_to_the_state_branch() -> None:
    cfg = make_cfg(n=4)
    model = make_model(cfg)
    grid = torch.linspace(0.0, 1.0, cfg.n + 1, dtype=torch.float64)
    state = torch.full((2, cfg.m), cfg.n0, dtype=torch.float64)
    action = model.interval_action(
        model.time_logits(grid)[0],
        torch.zeros(2, dtype=torch.float64),
        state,
        state_mode="w_zero",
    )
    gradients = torch.autograd.grad(
        action.sum(), list(model.state_branch.parameters()), allow_unused=True
    )
    assert all(gradient is None for gradient in gradients)


def test_batched_euler_costate_matches_full_objective_gradient() -> None:
    cfg = make_cfg(n=10)
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    initial = torch.stack(
        [
            torch.full((cfg.m,), 9.5, dtype=torch.float64),
            torch.linspace(9.0, 11.0, cfg.m, dtype=torch.float64),
        ]
    )
    controls = torch.linspace(
        0.8, 2.2, cfg.n, dtype=torch.float64
    ).expand(initial.shape[0], -1).clone().requires_grad_(True)
    states = simulate_open_loop(controls, initial, cfg, params)
    objective = objective_per_sample(states, controls, cfg, params).sum()
    gradient = torch.autograd.grad(objective, controls, retain_graph=True)[0]
    costates = compute_costate(states, controls, cfg, params)
    expected = (cfg.T / cfg.n) * (
        params["gamma"]
        - (costates[:, 1:] * params["phi"] * states[:, :-1]).sum(dim=-1)
    )
    assert torch.allclose(gradient, expected, atol=1e-10, rtol=1e-10)


def test_rk4_discrete_adjoint_matches_full_objective_gradient() -> None:
    from run_direct_openloop_cost import rk4_objective_interval_controls

    cfg = make_cfg(n=6)
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    initial = torch.linspace(9.0, 11.0, cfg.m, dtype=torch.float64)
    controls = torch.linspace(0.7, 2.1, cfg.n, dtype=torch.float64).requires_grad_(True)
    states = simulate_open_loop(
        controls, initial.unsqueeze(0), cfg, params, integrator="rk4"
    )
    _, discrete_gradient = compute_costate_rk4(
        states, controls.unsqueeze(0), cfg, params
    )
    objective = rk4_objective_interval_controls(controls, initial, cfg, params)
    gradient = torch.autograd.grad(objective, controls, retain_graph=True)[0]
    assert torch.allclose(
        discrete_gradient[0], gradient, atol=2e-11, rtol=2e-11
    )


def test_rk4_training_diagnostic_uses_matching_stage_quadrature() -> None:
    from feedback_section5_rk4_reference import simulate_open_loop_rk4
    from evaluate_feedback_section5 import rk4_zoh_open_loop

    cfg = make_cfg(n=6)
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    initial = torch.linspace(9.0, 11.0, cfg.m, dtype=torch.float64).view(1, -1)
    controls = torch.linspace(0.7, 2.1, cfg.n, dtype=torch.float64).view(1, -1)
    states, realized_controls, stages = simulate_open_loop_rk4(
        controls, initial, cfg, params
    )
    training_value = rk4_objective_per_sample(
        states, realized_controls, stages, cfg, params
    )
    evaluator_value, _ = rk4_zoh_open_loop(
        controls, initial, cfg, params, substeps=1
    )
    assert torch.allclose(training_value, evaluator_value, atol=1e-12, rtol=1e-12)


def test_uniform_beta_singular_formula_has_expected_sign_and_quotient() -> None:
    cfg = make_cfg(n=4)
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    batch = 3
    state = 8.0 + 4.0 * torch.rand(batch, cfg.m, dtype=torch.float64)
    states = torch.stack([state] * (cfg.n + 1), dim=1)
    controls = torch.ones(batch, cfg.n, dtype=torch.float64)

    # Choose costates so rho=beta and psi=0 simultaneously.  The two linear
    # constraints leave many degrees of freedom; solve the minimum-norm pair.
    interval_costates = []
    for row in state:
        matrix = torch.stack([params["M"] * row, params["phi"] * row])
        target = torch.tensor(
            [cfg.beta * (cfg.m + row.sum()), cfg.gamma], dtype=torch.float64
        )
        interval_costates.append(matrix.T @ torch.linalg.solve(matrix @ matrix.T, target))
    interval_costate = torch.stack(interval_costates)
    costates = torch.stack([interval_costate] * (cfg.n + 1), dim=1)
    values = singular_quantities(states, controls, costates, cfg, params)
    assert torch.all(values["B"] < 0.0)
    assert torch.allclose(values["u_strict"], values["u_state"], atol=1e-11, rtol=1e-11)
    assert torch.allclose(values["dot_psi"], torch.zeros_like(values["dot_psi"]), atol=1e-11, rtol=0.0)


def test_persistence_gate_rejects_an_isolated_candidate() -> None:
    point = torch.zeros(1, 9, dtype=torch.float64)
    point[0, 4] = 1.0
    persistent = persistence_gate(point, 3)
    assert persistent[0, 4] == 0.0
    point[0, 3:6] = 0.8
    persistent = persistence_gate(point, 3)
    assert torch.isclose(persistent[0, 4], torch.tensor(0.8, dtype=torch.float64))


def test_legendre_clebsch_penalty_is_live_for_positive_B() -> None:
    B = torch.tensor([0.25], dtype=torch.float64, requires_grad=True)
    detached_gate = torch.tensor([0.9], dtype=torch.float64)
    penalty = (detached_gate * torch.relu(B).square()).sum()
    penalty.backward()
    assert B.grad is not None
    assert float(B.grad) > 0.0


def make_loss_args(option: str, variant: str) -> argparse.Namespace:
    return argparse.Namespace(
        option=option,
        loss_variant=variant,
        singular_eps=0.1,
        singular_tau=0.03,
        dot_eps=0.1,
        dot_tau=0.03,
        b_min=1e-8,
        persistence_window=1,
        detach_gate=True,
        w_lc=1.0,
        w0=1.0,
        w1=1.0,
        w2=1.0,
    )


def synthetic_quantities(
    psi: torch.Tensor,
    dot_psi: torch.Tensor,
    B: torch.Tensor,
) -> dict[str, torch.Tensor]:
    shape = psi.shape
    return {
        "psi": psi,
        "dot_psi": dot_psi,
        "B": B,
        "ddot_psi": torch.full(shape, 0.2, dtype=psi.dtype),
        "u_state": torch.full(shape, 1.25, dtype=psi.dtype),
        "u_strict": torch.full(shape, 1.50, dtype=psi.dtype),
    }


def test_literal_closed_form_gate_uses_psi_only_and_stated_indicator() -> None:
    cfg = make_cfg()
    controls = torch.full((1, 2), 1.0, dtype=torch.float64)
    psi = torch.zeros_like(controls)
    dot_psi = torch.full_like(controls, 100.0)
    B = torch.tensor([[-0.5, 0.5]], dtype=torch.float64)
    pack = compose_section5_optimality_loss(
        controls,
        synthetic_quantities(psi, dot_psi, B),
        cfg,
        make_loss_args("cf", "literal"),
    )
    expected_gate = torch.sigmoid(torch.tensor(0.1 / 0.03, dtype=torch.float64))
    assert torch.allclose(pack["point_gate"], expected_gate.expand_as(controls))
    assert torch.isclose(pack["q"][0, 0], expected_gate)
    assert pack["q"][0, 1] == 0.0


def test_literal_boundary_loss_uses_unscaled_control() -> None:
    cfg = make_cfg()
    controls = torch.tensor([[1.5]], dtype=torch.float64)
    psi = torch.tensor([[2.0]], dtype=torch.float64)
    quantities = synthetic_quantities(
        psi, torch.tensor([[3.0]], dtype=torch.float64), torch.tensor([[0.5]], dtype=torch.float64)
    )
    pack = compose_section5_optimality_loss(
        controls, quantities, cfg, make_loss_args("cf", "literal")
    )
    assert torch.isclose(pack["nonsingular_component"], torch.tensor(9.0, dtype=torch.float64))


def test_literal_derivative_gate_makes_displayed_lc_term_inactive() -> None:
    cfg = make_cfg()
    controls = torch.ones(1, 1, dtype=torch.float64)
    B = torch.tensor([[0.25]], dtype=torch.float64, requires_grad=True)
    pack = compose_section5_optimality_loss(
        controls,
        synthetic_quantities(torch.zeros_like(B), torch.zeros_like(B), B),
        cfg,
        make_loss_args("der", "literal"),
    )
    assert pack["q"].item() == 0.0
    assert pack["singular_component"].item() == 0.0


def test_lc_live_derivative_variant_keeps_lc_gradient_live() -> None:
    cfg = make_cfg()
    controls = torch.ones(1, 1, dtype=torch.float64)
    B = torch.tensor([[0.25]], dtype=torch.float64, requires_grad=True)
    pack = compose_section5_optimality_loss(
        controls,
        synthetic_quantities(torch.zeros_like(B), torch.zeros_like(B), B),
        cfg,
        make_loss_args("der", "lc_live"),
    )
    gradient = torch.autograd.grad(pack["singular_component"], B)[0]
    assert pack["q"].item() > 0.0
    assert gradient.item() > 0.0


def test_lc_live_closed_form_variant_is_identical_to_literal() -> None:
    cfg = make_cfg()
    controls = torch.tensor([[0.5, 2.0]], dtype=torch.float64)
    quantities = synthetic_quantities(
        torch.tensor([[0.02, -0.03]], dtype=torch.float64),
        torch.tensor([[4.0, -5.0]], dtype=torch.float64),
        torch.tensor([[-0.2, -0.3]], dtype=torch.float64),
    )
    literal = compose_section5_optimality_loss(
        controls, quantities, cfg, make_loss_args("cf", "literal")
    )
    lc_live = compose_section5_optimality_loss(
        controls, quantities, cfg, make_loss_args("cf", "lc_live")
    )
    for key in literal:
        assert torch.equal(literal[key], lc_live[key])


def test_live_and_detached_gates_have_identical_values_but_different_gradients() -> None:
    cfg = make_cfg()
    controls = torch.ones(1, 1, dtype=torch.float64)
    gradients = []
    losses = []
    for mode in ("live", "detached"):
        psi = torch.tensor([[0.05]], dtype=torch.float64, requires_grad=True)
        quantities = synthetic_quantities(
            psi,
            torch.zeros_like(psi),
            -torch.ones_like(psi),
        )
        quantities["boundary_psi"] = 0.0 * psi
        args = make_loss_args("cf", "literal")
        args.gate_gradient_mode = mode
        pack = compose_section5_optimality_loss(controls, quantities, cfg, args)
        losses.append(pack["opt_gap"].detach())
        gradients.append(torch.autograd.grad(pack["opt_gap"], psi)[0])
    assert torch.equal(losses[0], losses[1])
    assert not torch.equal(gradients[0], gradients[1])


def test_literal_derivative_indicator_truth_table() -> None:
    cfg = make_cfg()
    controls = torch.ones(1, 6, dtype=torch.float64)
    psi = torch.zeros_like(controls)
    dot_psi = torch.zeros_like(controls)
    B = torch.tensor(
        [[-0.5, -0.5, -0.5, -1.0e-10, 0.5, -0.5]],
        dtype=torch.float64,
    )
    quantities = synthetic_quantities(psi, dot_psi, B)
    quantities["u_strict"] = torch.tensor(
        [[1.5, 0.0, cfg.umax, 1.5, 1.5, cfg.umax + 0.1]],
        dtype=torch.float64,
    )
    pack = compose_section5_optimality_loss(
        controls, quantities, cfg, make_loss_args("der", "literal")
    )
    assert pack["q"][0, 0] > 0.0
    assert torch.equal(pack["q"][0, 1:], torch.zeros(5, dtype=torch.float64))


def test_soft_max_smoothness_focuses_on_an_isolated_jump() -> None:
    controls = torch.zeros(1, 10, dtype=torch.float64)
    controls[:, 5:] = 1.0
    args = argparse.Namespace(
        smooth_weight=0.0,
        smooth_second_weight=0.0,
        smooth_max_weight=1.0,
        smooth_max_tau=0.02,
    )
    pack = smoothness_components(controls, args)
    assert pack["smooth_max_like"] > pack["smooth"]
    assert torch.equal(pack["smooth_total"], pack["smooth_max_like"])


def test_all_smoothness_components_vanish_for_a_constant_control() -> None:
    controls = torch.full((3, 10), 1.25, dtype=torch.float64)
    args = argparse.Namespace(
        smooth_weight=3.0,
        smooth_second_weight=15.0,
        smooth_max_weight=0.11,
        smooth_max_tau=0.02,
    )
    pack = smoothness_components(controls, args)
    for name in ("smooth", "smooth_second", "smooth_max_like", "smooth_total"):
        assert torch.allclose(
            pack[name], torch.zeros_like(pack[name]), atol=1e-30, rtol=0.0
        )


def test_full_feedback_loss_gradient_matches_central_difference() -> None:
    cfg = make_cfg(n=5)
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    model = make_model(cfg)
    initial = torch.linspace(9.0, 11.0, cfg.m, dtype=torch.float64).view(1, -1)
    args = argparse.Namespace(
        singular_eps=0.1,
        singular_tau=0.03,
        dot_eps=0.1,
        dot_tau=0.03,
        persistence_window=1,
        detach_gate=False,
        option="cf",
        w_lc=1.0,
        w0=1.0,
        w1=1.0,
        w2=1.0,
        smooth_weight=0.3,
    )
    bias = [
        module.bias
        for module in model.state_branch.modules()
        if isinstance(module, torch.nn.Linear)
    ][-1]
    loss = section5_loss(model, initial, cfg, params, args)["loss"]
    analytic = torch.autograd.grad(loss, bias)[0].item()

    epsilon = 1e-6
    original = bias.detach().clone()
    with torch.no_grad():
        bias.copy_(original + epsilon)
    plus = float(section5_loss(model, initial, cfg, params, args)["loss"].detach())
    with torch.no_grad():
        bias.copy_(original - epsilon)
    minus = float(section5_loss(model, initial, cfg, params, args)["loss"].detach())
    with torch.no_grad():
        bias.copy_(original)
    finite_difference = (plus - minus) / (2.0 * epsilon)
    assert abs(analytic - finite_difference) <= 1e-6 * max(1.0, abs(analytic))


def test_main_rk4_feedback_loss_gradient_matches_central_difference() -> None:
    cfg = make_cfg(n=4)
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    model = make_model(cfg)
    initial = torch.linspace(9.0, 11.0, cfg.m, dtype=torch.float64).view(1, -1)
    args = make_loss_args("cf", "literal")
    args.gate_gradient_mode = "live"
    args.training_integrator = "rk4"
    args.state_mode = "feedback"
    args.smooth_weight = 0.3
    args.smooth_second_weight = 0.2
    args.smooth_max_weight = 0.1
    args.smooth_max_tau = 0.02
    bias = [
        module.bias
        for module in model.state_branch.modules()
        if isinstance(module, torch.nn.Linear)
    ][-1]
    loss = section5_loss(model, initial, cfg, params, args)["loss"]
    analytic = torch.autograd.grad(loss, bias)[0].item()

    epsilon = 1e-6
    original = bias.detach().clone()
    with torch.no_grad():
        bias.copy_(original + epsilon)
    plus = float(section5_loss(model, initial, cfg, params, args)["loss"].detach())
    with torch.no_grad():
        bias.copy_(original - epsilon)
    minus = float(section5_loss(model, initial, cfg, params, args)["loss"].detach())
    with torch.no_grad():
        bias.copy_(original)
    finite_difference = (plus - minus) / (2.0 * epsilon)
    assert abs(analytic - finite_difference) <= 2e-6 * max(1.0, abs(analytic))
