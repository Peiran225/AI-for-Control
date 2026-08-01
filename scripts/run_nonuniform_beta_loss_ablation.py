#!/usr/bin/env python3
"""Run output-head LM refinement with the nonuniform-beta pilot objective."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    sys.path.insert(0, str(path))

import refine_direct_initialized_output_head_lm as lm  # noqa: E402


_original_build_params = lm.build_params


def nonuniform_build_params(cfg, device, dtype):
    params = _original_build_params(cfg, device, dtype)
    params["beta"] = 0.1 * (0.75 + 0.5 * params["x"])
    return params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--trainable-scope", default="output-head")
    parser.add_argument("--linear-solver", default="explicit")
    parser.add_argument("--psi-scale", type=float, default=1.0)
    parser.add_argument("--dot-scale", type=float, default=1.0)
    parser.add_argument("--ddot-scale", type=float, default=1.0)
    parser.add_argument("--w0", type=float, default=1.0)
    parser.add_argument("--w1", type=float, default=1.0)
    parser.add_argument("--w2", type=float, default=1.0)
    parser.add_argument(
        "--singular-loss",
        choices=("derivative", "cf-state"),
        default="derivative",
    )
    parser.add_argument("--cf-weight", type=float, default=1.0)
    parser.add_argument("--cf-scale", type=float, default=1.0)
    parser.add_argument("--b-min", type=float, default=1.0e-8)
    parser.add_argument(
        "--selection-metric",
        choices=("physical-all", "training-objective"),
        default="physical-all",
    )
    parser.add_argument("--max-step-norm", type=float, default=0.2)
    parser.add_argument("--initial-damping", type=float, default=0.01)
    cli = parser.parse_args()

    lm.build_params = nonuniform_build_params
    sys.argv = [
        sys.argv[0],
        "--start-checkpoint", cli.start_checkpoint,
        "--out-dir", cli.out_dir,
        "--device", cli.device,
        "--iterations", str(cli.iterations),
        "--trajectory-multiplier", "2",
        "--trainable-scope", cli.trainable_scope,
        "--linear-solver", cli.linear_solver,
        "--interior-start", "1.5",
        "--interior-end", "8.0",
        "--boundary-margin", "0.05",
        "--w0", str(cli.w0),
        "--w1", str(cli.w1),
        "--w2", str(cli.w2),
        "--singular-loss", cli.singular_loss,
        "--cf-weight", str(cli.cf_weight),
        "--cf-scale", str(cli.cf_scale),
        "--b-min", str(cli.b_min),
        "--selection-metric", cli.selection_metric,
        "--boundary-weight", "1",
        "--psi-scale", str(cli.psi_scale),
        "--dot-scale", str(cli.dot_scale),
        "--ddot-scale", str(cli.ddot_scale),
        "--boundary-scale", "1",
        "--report-scale-factor", "400",
        "--initial-damping", str(cli.initial_damping),
        "--maximum-step-norm", str(cli.max_step_norm),
    ]
    lm.main()


if __name__ == "__main__":
    main()
