from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for search_path in (ROOT, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from direct_rk4_discrete_adjoint import (  # noqa: E402
    RK4ZOHDiscreteAdjoint,
    TumorObjective,
)
from evaluate_feedback_section5 import rk4_zoh_open_loop  # noqa: E402
from train_paper_pmp_kkt import ProblemConfig, build_params  # noqa: E402


def test_numpy_discrete_adjoint_matches_float64_autograd() -> None:
    cfg = ProblemConfig(
        T=2.0,
        n=17,
        m=21,
        umax=3.0,
        beta=40.0,
        alpha=1.0,
        gamma=8000.0,
        n0=10.0,
        m_suppression=0.5,
    )
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    problem = TumorObjective.from_mapping(params, T=cfg.T, umax=cfg.umax)
    evaluator = RK4ZOHDiscreteAdjoint(problem, cfg.n, substeps=4)

    rng = np.random.default_rng(20260728)
    control_numpy = rng.uniform(0.05, 2.95, size=cfg.n)
    initial_numpy = cfg.n0 * (1.0 + 0.2 * rng.uniform(-1.0, 1.0, cfg.m))
    value, gradient = evaluator(control_numpy, initial_numpy)

    control = torch.tensor(
        control_numpy, dtype=torch.float64, requires_grad=True
    )
    initial = torch.tensor(initial_numpy[None, :], dtype=torch.float64)
    objective, _ = rk4_zoh_open_loop(
        control, initial, cfg, params, substeps=4
    )
    autodiff = torch.autograd.grad(objective.sum(), control)[0]

    np.testing.assert_allclose(
        value,
        float(objective.detach()),
        rtol=2.0e-13,
        atol=2.0e-10,
    )
    np.testing.assert_allclose(
        gradient,
        autodiff.detach().numpy(),
        rtol=2.0e-12,
        atol=2.0e-11,
    )
