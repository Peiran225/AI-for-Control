from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts.run_final_paper_recompute import (
    aggregate_task,
    diagnostic_tasks,
    figure_config,
)


def base_args() -> SimpleNamespace:
    return SimpleNamespace(
        python="python",
        time_checkpoint=Path("/checkpoints/time.pt"),
        cf_checkpoint=Path("/checkpoints/cf.pt"),
        der_checkpoint=Path("/checkpoints/der.pt"),
        direct_nominal=Path("/references/nominal.npz"),
        direct_resistant=Path("/references/resistant.npz"),
        time_refinement_multiplier=8,
        feedback_refinement_multiplier=16,
        query_batch_size=16,
        report_scale_factor=400.0,
        torch_threads=8,
        device="cpu",
    )


def test_aggregate_scale_argument_occurs_once() -> None:
    task = aggregate_task(base_args(), Path("/stage"))
    assert task.command.count("--report-scale-factor") == 1
    index = task.command.index("--report-scale-factor")
    assert task.command[index + 1] == "400.0"
    multiplier_index = task.command.index(
        "--feedback-refinement-multiplier"
    )
    assert task.command[multiplier_index + 1] == "16"
    assert len(task.command) == multiplier_index + 2


def test_feedback_diagnostics_bind_m16_refinement_metadata() -> None:
    tasks = diagnostic_tasks(base_args(), Path("/stage"))
    feedback = [task for task in tasks if "feedback_" in task.name]
    assert len(feedback) == 4
    for task in feedback:
        index = task.command.index("--refinement-multiplier")
        assert task.command[index + 1] == "16"


def test_figure_config_uses_fresh_m32_artifacts() -> None:
    config = figure_config(Path("/stage"))
    assert len(config["cases"]) == 3
    for case in config["cases"]:
        assert "/diagnostics/m32/" in case["path"]
        assert case["nominal_prefix"] == "nominal"
        assert case["resistant_heavy_prefix"] == "resistant_heavy"
