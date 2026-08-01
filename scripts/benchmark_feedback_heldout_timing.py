#!/usr/bin/env python3
"""Time loaded feedback rollouts on the Direct held-out state set.

The state generator, seed, radius, physical objective, execution grid, and
single-thread DOP853 rollout semantics match the paper's held-out evaluation.
This script times deployment only: checkpoints are loaded before timing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch

from compare_feedback_related_work import load_ours, rollout_feedback
from train_feedback_section5 import make_fixed_directions, test_states_from_directions
from train_paper_pmp_kkt import ProblemConfig
from tumor_problem import TumorProblem


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)) if array.size > 1 else math.nan,
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "q05": float(np.quantile(array, 0.05)),
        "q95": float(np.quantile(array, 0.95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def affinity() -> list[int] | None:
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cf", type=Path, required=True)
    parser.add_argument("--der", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--radius", type=float, default=0.20)
    parser.add_argument("--intervals", type=int, default=800)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument(
        "--direct-state-dir",
        type=Path,
        help="Optional canonical Direct sample directory used to verify N(0).",
    )
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = ProblemConfig(
        T=10.0,
        n=args.intervals,
        m=21,
        umax=3.0,
        alpha=0.0025,
        beta=0.1,
        gamma=20.0,
        n0=10.0,
        m_suppression=0.5,
    )
    directions = make_fixed_directions(args.samples, cfg.m, args.seed, torch.float64)
    states = test_states_from_directions(
        directions, args.radius, cfg, torch.device("cpu"), torch.float64
    ).cpu().numpy()

    direct_state_dir = (
        args.direct_state_dir.expanduser().resolve() if args.direct_state_dir else None
    )
    max_state_mismatch = 0.0
    if direct_state_dir:
        for index, state in enumerate(states):
            with np.load(direct_state_dir / f"sample_{index:04d}.npz") as pack:
                max_state_mismatch = max(
                    max_state_mismatch,
                    float(np.max(np.abs(np.asarray(pack["N0"]) - state))),
                )
        if max_state_mismatch > 1.0e-12:
            raise RuntimeError(
                f"held-out states do not match Direct artifacts: {max_state_mismatch}"
            )

    problem = TumorProblem(
        T=10.0,
        m=21,
        umax=3.0,
        alpha=1.0,
        beta=40.0,
        gamma=8000.0,
        n0=10.0,
        m_suppression=0.5,
    )
    checkpoints = {"PMP-CF": args.cf.resolve(), "PMP-DER": args.der.resolve()}
    rows: list[dict[str, Any]] = []
    method_stats: dict[str, Any] = {}

    for method, checkpoint in checkpoints.items():
        policy, native_intervals, load_seconds = load_ours(checkpoint)
        if native_intervals != args.intervals:
            raise RuntimeError(
                f"{method} checkpoint uses n={native_intervals}, expected {args.intervals}"
            )
        for _ in range(args.warmups):
            rollout_feedback(
                policy,
                states[0],
                args.intervals,
                problem,
                rtol=args.rtol,
                atol=args.atol,
            )

        seconds: list[float] = []
        objectives: list[float] = []
        for index, state in enumerate(states):
            started = time.perf_counter()
            trajectory = rollout_feedback(
                policy,
                state,
                args.intervals,
                problem,
                rtol=args.rtol,
                atol=args.atol,
            )
            elapsed = time.perf_counter() - started
            seconds.append(elapsed)
            objectives.append(float(trajectory["J"]))
            rows.append(
                {
                    "method": method,
                    "sample": index,
                    "seconds": elapsed,
                    "J": float(trajectory["J"]),
                    "N0_mean": float(np.mean(state)),
                    "N0_std": float(np.std(state)),
                }
            )
            print(
                f"[{method}] {index + 1:03d}/{args.samples:03d} "
                f"time={elapsed:.4f}s J={trajectory['J']:.6f}",
                flush=True,
            )
        method_stats[method] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "load_seconds_excluded": load_seconds,
            "rollout_seconds": stats(seconds),
            "objective": stats(objectives),
        }

    with (out_dir / "per_state.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "schema": "feedback-heldout-deployment-timing-v1",
        "protocol": {
            "samples": args.samples,
            "seed": args.seed,
            "radius": args.radius,
            "state_distribution": "N_i(0)=10(1+r Z_i), Z_i iid Uniform[-1,1]",
            "intervals": args.intervals,
            "policy_queries_per_rollout": args.intervals,
            "integrator": "segmented DOP853 with left-query/ZOH policy execution",
            "rtol": args.rtol,
            "atol": args.atol,
            "warmup_rollouts_per_method": args.warmups,
            "checkpoint_loading_excluded": True,
            "physical_weights": {"alpha": 1.0, "beta": 40.0, "gamma": 8000.0},
            "direct_state_dir": str(direct_state_dir) if direct_state_dir else None,
            "max_abs_state_mismatch_vs_direct": max_state_mismatch,
        },
        "methods": method_stats,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cpu_affinity": affinity(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
