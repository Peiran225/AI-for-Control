#!/usr/bin/env python3
"""Multi-seed bounded-domain tumor adaptation of paper Algorithm 1."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from faithful_related_work.pi_deeponet.core import TrainConfig, required_viscosity_constant  # noqa: E402
from faithful_related_work.pi_deeponet.experiment import ExperimentConfig, run_experiment  # noqa: E402
from faithful_related_work.pi_deeponet.problems import TumorAdaptation  # noqa: E402


def _ints(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("provide at least one seed")
    return values


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-dir", type=Path, default=ROOT / "faithful_related_work" / "pi_deeponet" / "runs" / "tumor")
    result.add_argument("--seeds", type=_ints, default=(0, 1, 2))
    result.add_argument("--outer", type=int, default=3)
    result.add_argument("--steps-per-outer", type=int, default=300)
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("--sensors", type=int, default=64)
    result.add_argument("--width", type=int, default=64)
    result.add_argument("--h", type=float, default=0.02, help="finite difference in x=N/state_scale")
    result.add_argument("--viscosity-N", type=float, default=0.0, help="0 chooses the smallest 0.001-grid value satisfying Theorem 1")
    result.add_argument("--evaluation-intervals", type=int, default=200)
    result.add_argument("--float32", action="store_true")
    result.add_argument("--alpha", type=float, default=1.0)
    result.add_argument("--beta", type=float, default=0.1)
    result.add_argument("--gamma", type=float, default=20.0)
    result.add_argument(
        "--value-scale",
        type=float,
        default=500.0,
        help="positive output scale for the value network",
    )
    result.add_argument(
        "--branch-scale",
        type=float,
        default=None,
        help=(
            "positive terminal-function sensor scale; by default 5000*alpha, "
            "which preserves the original dimensionless branch-input scale"
        ),
    )
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.alpha <= 0.0 or args.beta <= 0.0 or args.gamma <= 0.0:
        raise ValueError("alpha, beta, and gamma must be positive")
    if args.value_scale <= 0.0:
        raise ValueError("--value-scale must be positive")
    branch_scale = (
        float(args.branch_scale)
        if args.branch_scale is not None
        else 5000.0 * float(args.alpha)
    )
    if branch_scale <= 0.0:
        raise ValueError("--branch-scale must be positive")
    problem = TumorAdaptation(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
    )
    required = required_viscosity_constant(problem.dynamics_sup_bound())
    viscosity_N = args.viscosity_N if args.viscosity_N > 0.0 else math.ceil(required * 1000.0) / 1000.0
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
        log_every=max(1, args.steps_per_outer // 20),
        value_scale=args.value_scale,
        branch_scale=branch_scale,
        dtype="float32" if args.float32 else "float64",
        tie_tolerance=0.0,
    )
    experiment = ExperimentConfig(
        seeds=args.seeds,
        terminal_parameter_family=(0.8, 1.0, 1.2),
        target_terminal_parameter=1.0,
        initial_control=(0.0,),
        initial_state=tuple(problem.initial_state.tolist()),
        evaluation_intervals=args.evaluation_intervals,
        output_dir=args.output_dir,
    )
    paper_pdf = ROOT / "external" / "pi_deeponet_paper" / "2406.10920.pdf"
    run_experiment(problem, config, experiment, paper_pdf=paper_pdf if paper_pdf.exists() else None)


if __name__ == "__main__":
    main()
