#!/usr/bin/env python3
"""Small-budget original-paper Section 4.2.1 LQR reproduction smoke."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from faithful_related_work.pi_deeponet.core import (  # noqa: E402
    TrainConfig,
    required_viscosity_constant,
)
from faithful_related_work.pi_deeponet.experiment import ExperimentConfig, run_experiment  # noqa: E402
from faithful_related_work.pi_deeponet.problems import PaperLQR5D  # noqa: E402


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-dir", type=Path, default=ROOT / "faithful_related_work" / "pi_deeponet" / "runs" / "lqr_smoke")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--outer", type=int, default=3, help="paper uses M=3")
    result.add_argument("--steps-per-outer", type=int, default=20, help="smoke budget, not a converged paper run")
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--sensors", type=int, default=32)
    result.add_argument("--width", type=int, default=48)
    result.add_argument("--h", type=float, default=0.005, help="paper Section 4.2 uses h=0.005")
    result.add_argument("--evaluation-intervals", type=int, default=100)
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    problem = PaperLQR5D()
    viscosity_N = math.ceil(required_viscosity_constant(problem.dynamics_sup_bound()) * 1000.0) / 1000.0
    config = TrainConfig(
        h=args.h,
        viscosity_N=viscosity_N,
        outer_iterations=args.outer,
        steps_per_outer=args.steps_per_outer,
        batch_size=args.batch_size,
        terminal_batch_size=args.batch_size,
        sensors=args.sensors,
        width=args.width,
        latent_dim=args.width,
        log_every=max(1, args.steps_per_outer // 4),
        value_scale=1.0,
        branch_scale=4.0,
    )
    experiment = ExperimentConfig(
        seeds=(args.seed,),
        terminal_parameter_family=(1.0, 2.0, 3.0),
        target_terminal_parameter=0.57,
        initial_control=(0.0, 0.0, 0.0),
        initial_state=(0.3, -0.3, 0.2, -0.2, 0.1),
        evaluation_intervals=args.evaluation_intervals,
        output_dir=args.output_dir,
    )
    paper_pdf = ROOT / "external" / "pi_deeponet_paper" / "2406.10920.pdf"
    run_experiment(problem, config, experiment, paper_pdf=paper_pdf if paper_pdf.exists() else None)


if __name__ == "__main__":
    main()
