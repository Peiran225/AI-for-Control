from __future__ import annotations

import numpy as np
import torch

from scripts.generate_offgrid_policy_switching_diagnostics import (
    CASE_ORDER,
    QUANTITY_ORDER,
    STATE_ORDER,
    TrajectoryResult,
    build_grid_flags,
    build_refinement_flags,
    build_summary_rows,
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


def test_dense_grid_separates_refinement_queries_from_held_out_times() -> None:
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
    evaluation_time = np.linspace(0.0, cfg.T, 16 * cfg.n + 1)
    refinement, queries, held_out = build_refinement_flags(
        evaluation_time,
        cfg,
        refinement_multiplier=8,
    )
    support, _, _ = build_grid_flags(evaluation_time, cfg)

    assert int(refinement.sum()) == 8 * cfg.n + 1
    assert int(queries.sum()) == 7 * cfg.n
    assert int(held_out.sum()) == 8 * cfg.n
    assert np.all(refinement == (support | queries))
    assert not np.any(refinement & held_out)


def test_summary_rows_record_empty_subsets_without_reduction_error() -> None:
    time = np.linspace(0.0, 1.0, 5)
    template = TrajectoryResult(
        time=time,
        state=np.ones((time.size, 3)),
        costate=np.ones((time.size, 3)),
        control=np.ones(time.size),
        hamiltonian=np.ones(time.size),
        quantities={
            quantity: np.arange(time.size, dtype=np.float64)
            for quantity in QUANTITY_ORDER
        },
        identity_errors={},
        normalized_running_cost=0.0,
        normalized_objective=0.0,
    )
    results = {
        case_id: {state_id: template for state_id in STATE_ORDER}
        for case_id in CASE_ORDER
    }
    on_grid = np.ones(time.size, dtype=bool)
    refinement_query = np.zeros(time.size, dtype=bool)
    rows = build_summary_rows(
        results,
        on_grid,
        on_grid.copy(),
        refinement_query,
        np.zeros(time.size, dtype=bool),
        interior_start=0.25,
        interior_end=0.75,
    )

    empty = [
        row for row in rows if row["subset"] == "refinement_queries"
    ]
    assert empty
    assert all(row["count"] == 0 for row in empty)
    assert all(row["rms"] is None for row in empty)
    assert all(row["mean_abs"] is None for row in empty)
    assert all(row["max_abs"] is None for row in empty)
