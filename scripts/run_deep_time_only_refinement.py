#!/usr/bin/env python3
"""Run the reproducible deep time-only optimality-refinement chain.

The input setting must already contain the formal output of
``run_new_objective_weight_setting.py``.  This wrapper reuses only the
teacher-free Transformer checkpoint and successively applies exact
projected-gradient refinement, an active-set Newton solve derived from the
same discretized model, and an affine-head transfer followed by an exact
residual finish.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_new_objective_weight_setting import (  # noqa: E402
    PYTHON,
    run_command,
    write_json,
)


def build(args: argparse.Namespace) -> None:
    setting = args.setting_dir.expanduser().resolve()
    time_root = setting / "time_only"
    source = time_root / "linear_box_exact/selected_checkpoint.pt"
    if not source.is_file():
        raise FileNotFoundError(source)

    log_path = time_root / "deep_refinement_commands.json"
    command_log: list[dict[str, Any]] = (
        json.loads(log_path.read_text(encoding="utf-8"))
        if log_path.is_file()
        else []
    )

    p24 = time_root / "deep_refine_lbfgs_p24_stage1"
    run_command(
        [
            str(PYTHON),
            "scripts/continue_teacher_free_linear_box_projection.py",
            "--checkpoint",
            str(source),
            "--out-dir",
            str(p24),
            "--seed",
            "1940",
            "--adam-epochs",
            "0",
            "--lbfgs-outer-steps",
            "30",
            "--lbfgs-inner-iterations",
            "10",
            "--lbfgs-learning-rate",
            "0.12",
            "--lbfgs-history-size",
            "50",
            "--lbfgs-train-scope",
            "all",
            "--p",
            "24",
            "--high-p-weight",
            "4",
            "--solver-mode",
            "exact",
            "--selection-guard",
            "residual-only",
        ],
        expected=p24 / "COMPLETED.json",
        command_log=command_log,
    )

    p32 = time_root / "deep_refine_lbfgs_p32_stage2"
    run_command(
        [
            str(PYTHON),
            "scripts/continue_teacher_free_linear_box_projection.py",
            "--checkpoint",
            str(p24 / "selected_checkpoint.pt"),
            "--out-dir",
            str(p32),
            "--seed",
            "1941",
            "--adam-epochs",
            "0",
            "--lbfgs-outer-steps",
            "30",
            "--lbfgs-inner-iterations",
            "10",
            "--lbfgs-learning-rate",
            "0.08",
            "--lbfgs-history-size",
            "50",
            "--lbfgs-train-scope",
            "all",
            "--p",
            "32",
            "--high-p-weight",
            "6",
            "--solver-mode",
            "exact",
            "--selection-guard",
            "residual-only",
        ],
        expected=p32 / "COMPLETED.json",
        command_log=command_log,
    )

    target = time_root / "deep_refine_active_set_kkt_target"
    run_command(
        [
            str(PYTHON),
            "scripts/run_teacher_free_active_set_newton.py",
            "--checkpoint",
            str(p32 / "selected_checkpoint.pt"),
            "--out-dir",
            str(target),
            "--seed",
            "2002",
            "--max-newton-steps",
            "8",
            "--target-pg-linf",
            "1e-5",
        ],
        expected=target / "COMPLETED.json",
        command_log=command_log,
    )

    target_round2 = time_root / "deep_refine_active_set_kkt_target_round2_retry"
    run_command(
        [
            str(PYTHON),
            "scripts/run_teacher_free_active_set_newton.py",
            "--checkpoint",
            str(p32 / "selected_checkpoint.pt"),
            "--out-dir",
            str(target_round2),
            "--bootstrap-self-target",
            str(target / "solution.npz"),
            "--seed",
            "2102",
            "--max-newton-steps",
            "8",
            "--target-pg-linf",
            "1e-6",
            "--dampings",
            "0,1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1",
            "--alphas",
            "1,0.5,0.25,0.125,0.0625,0.03125",
        ],
        expected=target_round2 / "COMPLETED.json",
        command_log=command_log,
    )

    final = time_root / "deep_refine_kkt_head_ols_exact_v3"
    run_command(
        [
            str(PYTHON),
            "scripts/fit_teacher_free_kkt_self_target.py",
            "--checkpoint",
            str(p32 / "selected_checkpoint.pt"),
            "--out-dir",
            str(final),
            "--seed",
            "2703",
            "--target-npz",
            str(target_round2 / "solution.npz"),
            "--fit-adam-epochs",
            "0",
            "--target-head-ols-bound-margins",
            "0.05,0.075,0.1,0.125,0.15,0.2,0.25,0.5,1.0",
            "--target-head-ols-ridges",
            "1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2",
            "--fit-lbfgs-outer",
            "0",
            "--finish-outer",
            "8",
            "--finish-inner",
            "8",
            "--finish-lr",
            "0.05",
            "--finish-p",
            "32",
            "--finish-p-weight",
            "6",
            "--finish-stop-pg-linf",
            "7.5e-5",
            "--finish-stagnation-patience",
            "5",
            "--selection-guard",
            "residual-only",
        ],
        expected=final / "COMPLETED.json",
        command_log=command_log,
    )

    write_json(log_path, command_log)
    write_json(
        time_root / "DEEP_REFINEMENT_COMPLETED.json",
        {
            "status": "completed",
            "source_checkpoint": str(source),
            "selected_checkpoint": str(final / "selected_checkpoint.pt"),
            "selected_solution": str(final / "solution.npz"),
        },
    )
    print(f"completed deep time-only refinement: {final}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting-dir", type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
