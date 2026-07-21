from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from feedback_section5_rk4_reference import (
    RK4_B,
    discrete_rk4_adjoint,
    rk4_running_integral,
    simulate_feedback_rk4,
    simulate_open_loop_rk4,
    singular_quantities_at_points,
    switching_function,
)
from train_paper_pmp_kkt import ProblemConfig, build_params


def make_cfg(n: int = 6) -> ProblemConfig:
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


def physical_rk4_objective(
    states: torch.Tensor,
    controls: torch.Tensor,
    stages: torch.Tensor,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
) -> torch.Tensor:
    terminal = (params["alpha"] * states[:, -1]).sum(dim=-1)
    return terminal + rk4_running_integral(stages, controls, cfg, params)


def test_rk4_stage_adjoint_matches_full_open_loop_autodiff() -> None:
    cfg = make_cfg(n=7)
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    initial = torch.stack(
        (
            torch.linspace(9.0, 11.0, cfg.m, dtype=torch.float64),
            torch.linspace(10.8, 9.2, cfg.m, dtype=torch.float64),
        )
    )
    controls = torch.stack(
        (
            torch.linspace(0.6, 2.4, cfg.n, dtype=torch.float64),
            torch.linspace(2.1, 0.9, cfg.n, dtype=torch.float64),
        )
    ).requires_grad_(True)
    states, realized, stages = simulate_open_loop_rk4(
        controls, initial, cfg, params
    )
    objective = physical_rk4_objective(
        states, realized, stages, cfg, params
    ).sum()
    autodiff = torch.autograd.grad(objective, controls, retain_graph=True)[0]
    adjoint = discrete_rk4_adjoint(states, realized, stages, cfg, params)

    torch.testing.assert_close(
        adjoint.control_gradient, autodiff, rtol=2e-12, atol=2e-12
    )
    weights = torch.tensor(RK4_B, dtype=torch.float64)
    weighted_stage_switching = (
        adjoint.stage_switching * weights.view(1, 1, 4)
    ).sum(dim=-1)
    torch.testing.assert_close(
        adjoint.interval_switching,
        weighted_stage_switching,
        rtol=1e-14,
        atol=1e-14,
    )
    stage_controls = realized.unsqueeze(-1).expand(-1, -1, 4)
    singular = singular_quantities_at_points(
        stages, stage_controls, adjoint.stage_costates, params
    )
    torch.testing.assert_close(
        singular["psi"], adjoint.stage_switching, rtol=1e-14, atol=1e-14
    )
    torch.testing.assert_close(
        singular["ddot_psi"],
        singular["A"] + singular["B"] * stage_controls,
        rtol=1e-14,
        atol=1e-14,
    )


def test_naive_euler_pair_is_not_the_rk4_switching() -> None:
    cfg = make_cfg(n=4)
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    initial = torch.linspace(8.5, 11.5, cfg.m, dtype=torch.float64).unsqueeze(0)
    controls = torch.tensor(
        [[0.35, 1.2, 2.65, 0.8]], dtype=torch.float64
    )
    states, realized, stages = simulate_open_loop_rk4(
        controls, initial, cfg, params
    )
    adjoint = discrete_rk4_adjoint(states, realized, stages, cfg, params)
    naive = switching_function(
        states[:, :-1], adjoint.node_costates[:, 1:], params
    )

    # The Euler pair is a different discretization.  At this deliberately
    # coarse step it cannot be presented as the matching RK4 derivative.
    assert float((naive - adjoint.interval_switching).abs().max()) > 1e-3


class ToyFeedback(torch.nn.Module):
    def __init__(self, umax: float) -> None:
        super().__init__()
        self.umax = umax
        self.bias = torch.nn.Parameter(torch.tensor(0.15, dtype=torch.float64))

    def time_logits(self, grid: torch.Tensor) -> torch.Tensor:
        return 0.2 * torch.sin(2.0 * torch.pi * grid)

    def interval_action(
        self,
        base_logit: torch.Tensor,
        normalized_time: torch.Tensor,
        state: torch.Tensor,
        *,
        state_blind: bool = False,
    ) -> torch.Tensor:
        query = torch.full_like(state, 10.0) if state_blind else state
        state_feature = (query.mean(dim=-1) - 10.0) / 10.0
        return self.umax * torch.sigmoid(
            base_logit + self.bias + 0.4 * state_feature + 0.1 * normalized_time
        )


def feedback_pack(
    model: ToyFeedback,
    initial: torch.Tensor,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
):
    states, controls, stages = simulate_feedback_rk4(
        model, initial, cfg, params
    )
    adjoint = discrete_rk4_adjoint(states, controls, stages, cfg, params)
    return states, controls, stages, adjoint


def test_open_loop_control_gradient_chains_through_feedback_policy() -> None:
    cfg = make_cfg(n=5)
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    initial = torch.stack(
        (
            torch.linspace(9.0, 11.0, cfg.m, dtype=torch.float64),
            torch.linspace(10.5, 9.5, cfg.m, dtype=torch.float64),
        )
    )
    model = ToyFeedback(cfg.umax)
    states, controls, stages, adjoint = feedback_pack(
        model, initial, cfg, params
    )
    objective = physical_rk4_objective(
        states, controls, stages, cfg, params
    ).sum()
    direct_parameter_gradient = torch.autograd.grad(
        objective, model.bias, retain_graph=True
    )[0]
    chained_parameter_gradient = torch.autograd.grad(
        controls,
        model.bias,
        grad_outputs=adjoint.control_gradient.detach(),
        retain_graph=True,
    )[0]
    torch.testing.assert_close(
        chained_parameter_gradient,
        direct_parameter_gradient,
        rtol=2e-11,
        atol=2e-11,
    )


def test_pmp_residual_keeps_the_full_outer_autograd_graph() -> None:
    cfg = make_cfg(n=5)
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    initial = torch.linspace(9.0, 11.0, cfg.m, dtype=torch.float64).unsqueeze(0)
    model = ToyFeedback(cfg.umax)

    def residual_value() -> torch.Tensor:
        _, controls, stages, adjoint = feedback_pack(model, initial, cfg, params)
        smooth = (controls[:, 1:] - controls[:, :-1]).square().mean()
        stage_controls = controls.unsqueeze(-1).expand(-1, -1, 4)
        singular = singular_quantities_at_points(
            stages, stage_controls, adjoint.stage_costates, params
        )
        weights = torch.tensor(RK4_B, dtype=controls.dtype).view(1, 1, 4)
        higher_order = (
            weights
            * (
                singular["psi"].square()
                + singular["dot_psi"].square()
                + singular["ddot_psi"].square()
                + torch.relu(singular["B"]).square()
            )
        ).sum(dim=-1).mean()
        # This is deliberately a Section-5 optimality residual, not J.
        return higher_order + 0.3 * smooth

    residual = residual_value()
    analytic = torch.autograd.grad(residual, model.bias)[0]

    epsilon = 1e-6
    original = model.bias.detach().clone()
    with torch.no_grad():
        model.bias.copy_(original + epsilon)
    plus = residual_value().detach()
    with torch.no_grad():
        model.bias.copy_(original - epsilon)
    minus = residual_value().detach()
    with torch.no_grad():
        model.bias.copy_(original)
    finite_difference = (plus - minus) / (2.0 * epsilon)

    torch.testing.assert_close(
        analytic, finite_difference, rtol=2e-7, atol=2e-7
    )
