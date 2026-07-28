#!/usr/bin/env python3
"""Recompute nominal and resistant-heavy Table-1 entries on one evaluator.

This script reuses the held-out evaluator's numerical implementation so the
fixed-state and held-out table blocks cannot silently use different objective
or singular-residual definitions.  Direct transcription accepts distinct
nominal and resistant-heavy control artifacts; all other time-only methods
replay one retained schedule on both states, while feedback policies are
queried closed loop from each state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_table1_heldout_common import (
    METHOD_CHOICES,
    Problem,
    load_control,
    load_policy,
    resolve,
    rollout_controls,
    sha256,
    singular_residual,
    write_csv,
    write_json,
)


METHODS = tuple(
    "direct_statewise" if method == "direct_nominal_replay" else method
    for method in METHOD_CHOICES
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        help="shared time-only control or feedback checkpoint",
    )
    parser.add_argument(
        "--nominal-source",
        type=Path,
        help="nominal direct-transcription control",
    )
    parser.add_argument(
        "--resistant-source",
        type=Path,
        help="resistant-heavy direct-transcription control",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-label", default="selected")
    parser.add_argument("--objective-substeps", type=int, default=4)
    parser.add_argument("--diagnostic-refinement", type=int, default=32)
    parser.add_argument("--interior-start", type=float, default=1.5)
    parser.add_argument("--interior-end", type=float, default=8.0)
    parser.add_argument("--hjb-tau", type=float, default=10.0)
    return parser.parse_args()


def source_record(path: Path) -> dict[str, Any]:
    if path.is_file():
        return {
            "path": str(path),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    if path.is_dir():
        files = sorted(item for item in path.rglob("*") if item.is_file())
        if not files:
            raise ValueError(f"source directory is empty: {path}")
        return {
            "path": str(path),
            "files": [
                {
                    "path": str(item),
                    "sha256": sha256(item),
                    "bytes": item.stat().st_size,
                }
                for item in files
            ],
        }
    raise FileNotFoundError(path)


def main() -> None:
    args = parse_args()
    problem = Problem()
    vectors = problem.vectors()
    output_dir = resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} already exists; use a new immutable output directory"
        )
    output_dir.mkdir(parents=True)

    nominal = np.full(problem.m, problem.n0, dtype=np.float64)
    resistant = np.linspace(9.0, 11.0, problem.m, dtype=np.float64)
    states = {
        "nominal": nominal,
        "resistant_heavy": resistant,
    }

    time_only = args.method in {
        "direct_statewise",
        "neural_pmp_learned",
        "neural_pmp_exact",
        "pmp_kkt_time",
    }
    sources: dict[str, Path] = {}
    query = None
    close = lambda: None
    policy_metadata: dict[str, Any] = {}
    if args.method == "direct_statewise":
        if args.source is not None:
            raise ValueError("direct_statewise uses --nominal-source/--resistant-source")
        if args.nominal_source is None or args.resistant_source is None:
            raise ValueError(
                "direct_statewise requires both --nominal-source and "
                "--resistant-source"
            )
        sources = {
            "nominal": resolve(args.nominal_source),
            "resistant_heavy": resolve(args.resistant_source),
        }
    else:
        if args.source is None:
            raise ValueError(f"{args.method} requires --source")
        shared_source = resolve(args.source)
        sources = {state_id: shared_source for state_id in states}
        if not time_only:
            query, close, policy_metadata = load_policy(
                args.method, shared_source, hjb_tau=args.hjb_tau
            )

    rows: list[dict[str, Any]] = []
    try:
        for state_id, initial_state in states.items():
            source = sources[state_id]
            if not source.exists():
                raise FileNotFoundError(source)
            fixed_control = (
                load_control(source, problem) if time_only else None
            )
            controls, objective = rollout_controls(
                initial_state[None, :],
                problem,
                vectors,
                fixed_control=fixed_control,
                query=query,
                objective_substeps=args.objective_substeps,
            )
            rms_psi, rms_dot, rms_ddot, residual = singular_residual(
                initial_state[None, :],
                controls,
                problem,
                vectors,
                refinement=args.diagnostic_refinement,
                interior_start=args.interior_start,
                interior_end=args.interior_end,
            )
            rows.append(
                {
                    "method": args.method,
                    "seed_label": args.seed_label,
                    "state_id": state_id,
                    "J": f"{objective[0]:.12g}",
                    "RMS_H_u": f"{rms_psi[0]:.12g}",
                    "RMS_dH_u_dt": f"{rms_dot[0]:.12g}",
                    "RMS_d2H_u_dt2": f"{rms_ddot[0]:.12g}",
                    "R_sing": f"{residual[0]:.12g}",
                    "initial_state": json.dumps(
                        initial_state.tolist(), separators=(",", ":")
                    ),
                    "source": str(source),
                    "source_sha256": sha256(source) if source.is_file() else "",
                    "control_semantics": (
                        "state-specific independently optimized direct control"
                        if args.method == "direct_statewise"
                        else (
                            "fixed learned time-only schedule replayed unchanged"
                            if time_only
                            else "closed-loop policy queried at each n=800 left endpoint"
                        )
                    ),
                }
            )
            print(
                f"[{args.method}/{args.seed_label}/{state_id}] "
                f"J={objective[0]:.9f}, R_sing={residual[0]:.9g}",
                flush=True,
            )
    finally:
        close()

    write_csv(output_dir / "fixed_states.csv", rows)
    unique_sources = {
        str(path): source_record(path) for path in sorted(set(sources.values()))
    }
    write_json(
        output_dir / "summary.json",
        {
            "schema": "table1-fixed-states-common-v1",
            "method": args.method,
            "seed_label": args.seed_label,
            "problem": problem.__dict__,
            "protocol": {
                "control_grid": "n=800 left-endpoint zero-order hold",
                "objective_integrator": (
                    f"float64 classical RK4, {args.objective_substeps} substeps "
                    "per control interval, matching RK4 running-cost quadrature"
                ),
                "scalar_diagnostic": (
                    "analytic continuous-PMP H_u, dH_u/dt, d2H_u/dt2 "
                    f"on q={args.diagnostic_refinement} refined ZOH nodes"
                ),
                "diagnostic_interval": [
                    args.interior_start,
                    args.interior_end,
                ],
                "diagnostic_interval_semantics": "half-open",
                "R_sing": (
                    "sqrt(RMS(H_u)^2+RMS(dH_u/dt)^2+"
                    "RMS(d2H_u/dt2)^2)"
                ),
            },
            "policy_metadata": policy_metadata,
            "sources": unique_sources,
            "rows": rows,
            "outputs": {
                "fixed_states_csv": str(
                    (output_dir / "fixed_states.csv").resolve()
                )
            },
        },
    )


if __name__ == "__main__":
    main()
