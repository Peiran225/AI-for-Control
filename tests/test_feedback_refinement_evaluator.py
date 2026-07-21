import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_feedback_refinement import (
    advantage_stats,
    make_held_out_direction_families,
    paired_summary_block,
)


def test_held_out_direction_families_have_expected_geometry() -> None:
    directions, labels, names = make_held_out_direction_families(
        8,
        7,
        20260723,
        torch.float64,
        ["random", "total", "composition"],
    )
    assert directions.shape == (24, 7)
    assert len(labels) == len(names) == 24
    assert labels[:8] == ["random"] * 8
    assert labels[8:16] == ["total"] * 8
    assert labels[16:] == ["composition"] * 8

    total = directions[8:16]
    torch.testing.assert_close(total, total[:, :1].expand_as(total))
    torch.testing.assert_close(total[0], torch.ones(7, dtype=torch.float64))
    torch.testing.assert_close(total[1], -torch.ones(7, dtype=torch.float64))

    composition = directions[16:]
    torch.testing.assert_close(
        composition.sum(dim=1), torch.zeros(8, dtype=torch.float64), atol=1e-14, rtol=0
    )
    torch.testing.assert_close(
        composition.abs().amax(dim=1),
        torch.ones(8, dtype=torch.float64),
        atol=1e-14,
        rtol=0,
    )


def test_held_out_direction_families_are_reproducible() -> None:
    first = make_held_out_direction_families(
        5, 4, 91, torch.float64, ["composition", "random"]
    )
    second = make_held_out_direction_families(
        5, 4, 91, torch.float64, ["composition", "random"]
    )
    torch.testing.assert_close(first[0], second[0])
    assert first[1:] == second[1:]

    total_only = make_held_out_direction_families(
        5, 4, 91, torch.float64, ["total"]
    )[0]
    all_families = make_held_out_direction_families(
        5, 4, 91, torch.float64, ["random", "total", "composition"]
    )[0]
    torch.testing.assert_close(total_only, all_families[5:10])


def test_advantage_summary_reports_win_worst_and_bootstrap_ci() -> None:
    summary = advantage_stats(
        np.array([-2.0, 1.0, 3.0]), seed=17, repeats=500
    )
    assert summary["mean"] == 2.0 / 3.0
    assert summary["median"] == 1.0
    assert summary["win_fraction"] == 2.0 / 3.0
    assert summary["worst"] == -2.0
    assert summary["best"] == 3.0
    assert summary["mean_bootstrap_95ci"][0] <= summary["mean"]
    assert summary["mean_bootstrap_95ci"][1] >= summary["mean"]


def test_paired_summary_decomposition_is_exact() -> None:
    pre = {
        "feedback_J": np.array([10.0, 12.0]),
        "frozen_nominal_J": np.array([11.0, 13.0]),
        "feedback_advantage": np.array([1.0, 1.0]),
    }
    post = {
        "feedback_J": np.array([8.0, 11.0]),
        "frozen_nominal_J": np.array([10.5, 12.5]),
        "feedback_advantage": np.array([2.5, 1.5]),
    }
    summary = paired_summary_block(
        pre,
        post,
        np.array([0, 1]),
        bootstrap_seed=7,
        bootstrap_repeats=100,
    )
    assert summary["total_improvement"]["mean"] == 1.5
    assert summary["nominal_schedule_improvement"]["mean"] == 0.5
    assert summary["feedback_specific_improvement"]["mean"] == 1.0
    assert summary["maximum_decomposition_error"] == 0.0
