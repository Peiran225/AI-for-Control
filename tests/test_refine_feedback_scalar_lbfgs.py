from __future__ import annotations

import argparse

import torch

from scripts.refine_feedback_scalar_lbfgs import (
    anchored_states,
    configure_scalar_loss,
)
from train_paper_pmp_kkt import ProblemConfig


def make_cfg() -> ProblemConfig:
    return ProblemConfig(
        T=10.0,
        n=8,
        m=5,
        umax=3.0,
        beta=0.1,
        alpha=0.0025,
        gamma=20.0,
        n0=10.0,
        m_suppression=0.5,
    )


def test_anchored_states_start_with_named_states() -> None:
    cfg = make_cfg()
    states = anchored_states(
        3,
        7,
        0.1,
        cfg,
        torch.device("cpu"),
        torch.float64,
    )
    assert states.shape == (5, cfg.m)
    torch.testing.assert_close(
        states[0], torch.full((cfg.m,), 10.0, dtype=torch.float64)
    )
    torch.testing.assert_close(
        states[1],
        torch.tensor([9.0, 9.5, 10.0, 10.5, 11.0], dtype=torch.float64),
    )


def test_scalar_configuration_contains_only_three_der_residuals() -> None:
    cfg = make_cfg()
    configured = configure_scalar_loss(
        argparse.Namespace(),
        cfg,
        argparse.Namespace(interval_start=1.5, interval_end=8.0),
        torch.device("cpu"),
        torch.float64,
    )
    assert configured.w0 == configured.w1 == configured.w2 == 1.0
    assert configured.w_lc == 0.0
    assert configured.nonsingular_loss_weight == 0.0
    assert configured.smooth_weight == 0.0
    assert configured.full_gradient_weight == 0.0
    assert configured._fixed_candidate_mask_tensor.shape == (1, cfg.n, 4)
