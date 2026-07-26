from __future__ import annotations

import numpy as np
import torch

from scripts.generate_offgrid_policy_switching_diagnostics import (
    build_grid_flags,
    fixed_support_query_logits,
    raw_time_logits,
)
from train_paper_pmp_kkt import ProblemConfig, TimeTransformer


def test_fixed_support_query_reconstructs_training_coordinates() -> None:
    torch.manual_seed(20260726)
    model = TimeTransformer(
        d_model=16,
        heads=4,
        layers=2,
        umax=3.0,
        init_u=1.0,
    ).double()
    model.eval()
    support = torch.linspace(0.0, 1.0, 9, dtype=torch.float64)

    expected = raw_time_logits(model, support)
    actual = fixed_support_query_logits(
        model,
        support,
        support,
        batch_size=3,
    )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=2.0e-12)


def test_midpoint_grid_contains_training_nodes_and_off_grid_queries() -> None:
    cfg = ProblemConfig(
        T=10.0,
        n=8,
        m=3,
        umax=3.0,
        beta=40.0,
        alpha=1.0,
        gamma=8000.0,
        n0=10.0,
        m_suppression=0.5,
    )
    evaluation_time = np.linspace(0.0, cfg.T, 2 * cfg.n + 1)

    on_grid, nearest_index, distance = build_grid_flags(evaluation_time, cfg)

    assert int(on_grid.sum()) == cfg.n + 1
    assert int((~on_grid).sum()) == cfg.n
    np.testing.assert_array_equal(nearest_index[::2], np.arange(cfg.n + 1))
    np.testing.assert_allclose(distance[::2], 0.0, rtol=0.0, atol=1.0e-14)
    assert np.all(distance[1::2] > 0.0)
