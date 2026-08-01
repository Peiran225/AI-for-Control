#!/usr/bin/env python3
"""Package the predeclared fresh-direct timing experiment for paper use."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    REPO / "output/table1_direct_heldout_r020_exactadjoint_20260728_v3"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "sample_sd": float(values.std(ddof=1)) if values.size > 1 else math.nan,
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


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


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--pinned-summary", type=Path)
    parser.add_argument("--feedback-summary", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--direct-replay-ms", type=float, default=600.53)
    parser.add_argument("--cf-rollout-ms", type=float, default=1033.76)
    parser.add_argument("--der-rollout-ms", type=float, default=1037.28)
    args = parser.parse_args()

    source = args.source.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    source_summary_path = source / "summary.json"
    source_csv_path = source / "per_sample.csv"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    rows = load_rows(source_csv_path)

    optimizer_seconds = np.asarray(
        [float(row["optimizer_seconds"]) for row in rows], dtype=np.float64
    )
    elapsed_seconds = np.asarray(
        [float(row["elapsed_seconds"]) for row in rows], dtype=np.float64
    )
    projected_gradient = np.asarray(
        [float(row["projected_gradient_linf"]) for row in rows], dtype=np.float64
    )
    if len(rows) != 128:
        raise RuntimeError(f"expected 128 predeclared states, found {len(rows)}")
    if np.max(projected_gradient) > 1.0e-5:
        raise RuntimeError("at least one direct solve fails the projected-gradient check")

    protocol = source_summary["protocol"]
    expected_weights = {"alpha": 1.0, "beta": 40.0, "gamma": 8000.0}
    if protocol["physical_weights"] != expected_weights:
        raise RuntimeError("source result does not use the paper objective weights")
    if protocol["problem_normalized"]["n"] != 800:
        raise RuntimeError("source result does not use n=800")
    if protocol["radius"] != 0.20 or protocol["direction_seed"] != 20260720:
        raise RuntimeError("source result does not use the predeclared held-out states")

    optimizer_stats = stats(optimizer_seconds)
    elapsed_stats = stats(elapsed_seconds)
    cf_seconds = args.cf_rollout_ms / 1000.0
    der_seconds = args.der_rollout_ms / 1000.0
    feedback = None
    if args.feedback_summary:
        feedback_path = args.feedback_summary.resolve()
        feedback_payload = json.loads(feedback_path.read_text(encoding="utf-8"))
        if feedback_payload["protocol"]["samples"] != 128:
            raise RuntimeError("feedback timing does not contain 128 states")
        if feedback_payload["protocol"]["seed"] != 20260720:
            raise RuntimeError("feedback timing uses a different held-out seed")
        if feedback_payload["protocol"]["radius"] != 0.20:
            raise RuntimeError("feedback timing uses a different held-out radius")
        if feedback_payload["protocol"]["max_abs_state_mismatch_vs_direct"] > 1e-12:
            raise RuntimeError("feedback and Direct timing use different states")
        feedback = {
            "path": str(feedback_path),
            "sha256": sha256(feedback_path),
            "PMP-CF": feedback_payload["methods"]["PMP-CF"]["rollout_seconds"],
            "PMP-DER": feedback_payload["methods"]["PMP-DER"]["rollout_seconds"],
            "environment": feedback_payload["environment"],
            "protocol": feedback_payload["protocol"],
        }
        cf_seconds = float(feedback["PMP-CF"]["median"])
        der_seconds = float(feedback["PMP-DER"]["median"])
    comparison = {
        "cached_direct_replay_seconds": args.direct_replay_ms / 1000.0,
        "loaded_pmp_cf_rollout_median_seconds": cf_seconds,
        "loaded_pmp_der_rollout_median_seconds": der_seconds,
        "fresh_direct_optimizer_median_seconds": optimizer_stats["median"],
        "fresh_direct_with_diagnostic_median_seconds": elapsed_stats["median"],
        "fresh_direct_over_cf_rollout_ratio": optimizer_stats["median"] / cf_seconds,
        "fresh_direct_over_der_rollout_ratio": optimizer_stats["median"] / der_seconds,
    }

    pinned = None
    if args.pinned_summary:
        pinned_path = args.pinned_summary.resolve()
        pinned = json.loads(pinned_path.read_text(encoding="utf-8"))
        pinned = {
            "path": str(pinned_path),
            "sha256": sha256(pinned_path),
            "sample_count": pinned["sample_count"],
            "optimizer_seconds": pinned["optimizer_seconds"],
            "elapsed_seconds": pinned["elapsed_seconds"],
            "projected_gradient_linf": pinned["projected_gradient_linf"],
        }

    payload = {
        "schema": "fresh-direct-resolve-timing-v1",
        "purpose": (
            "Fresh state-specific direct-transcription construction time versus "
            "loaded feedback-policy rollout time"
        ),
        "source": {
            "summary": str(source_summary_path),
            "summary_sha256": sha256(source_summary_path),
            "per_sample": str(source_csv_path),
            "per_sample_sha256": sha256(source_csv_path),
            "protocol_signature": source_summary["protocol_signature"],
        },
        "protocol": protocol,
        "timing_scope": {
            "fresh_direct": (
                "two-stage warm-started L-BFGS-B optimizer only; model loading, "
                "plotting, serialization, and post-hoc dense diagnostic excluded"
            ),
            "learned_policy": (
                "preloaded policy plus 800 online state-conditioned queries and "
                "the matched closed-loop rollout"
            ),
            "cached_direct_replay": (
                "precomputed schedule replay only; not a new-state optimization"
            ),
        },
        "optimizer_seconds": optimizer_stats,
        "sample_pipeline_seconds": elapsed_stats,
        "projected_gradient_linf": stats(projected_gradient),
        "projected_gradient_pass_threshold": 1.0e-5,
        "projected_gradient_pass_count": int(np.sum(projected_gradient <= 1.0e-5)),
        "comparison": comparison,
        "matched_feedback_timing": feedback,
        "pinned_sequential_confirmation": pinned,
        "packaging_environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    selected_columns = [
        "sample",
        "optimizer_seconds",
        "elapsed_seconds",
        "projected_gradient_linf",
        "optimizer_nit",
        "optimizer_nfev",
    ]
    with (out_dir / "per_sample_timing.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=selected_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in selected_columns})

    if feedback is not None:
        cf_iqr = f"{feedback['PMP-CF']['q25']:.3f}--{feedback['PMP-CF']['q75']:.3f}"
        der_iqr = f"{feedback['PMP-DER']['q25']:.3f}--{feedback['PMP-DER']['q75']:.3f}"
    else:
        cf_iqr = "--"
        der_iqr = "--"

    table = rf"""% Generated by aggregate_fresh_direct_timing.py
\begin{{tabular}}{{lcc}}
\toprule
Per-state operation & Median (s) & IQR (s) \\
\midrule
Warm-started Direct solve & {optimizer_stats['median']:.3f} & {optimizer_stats['q25']:.3f}--{optimizer_stats['q75']:.3f} \\
Loaded \textsc{{PMP-CF}} rollout & {cf_seconds:.3f} & {cf_iqr} \\
Loaded \textsc{{PMP-DER}} rollout & {der_seconds:.3f} & {der_iqr} \\
\bottomrule
\end{{tabular}}
"""
    (out_dir / "supplement_table_snippet.tex").write_text(table, encoding="utf-8")


if __name__ == "__main__":
    main()
