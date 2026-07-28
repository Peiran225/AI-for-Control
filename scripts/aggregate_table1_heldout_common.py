#!/usr/bin/env python3
"""Aggregate auditable Table-1 held-out runs across seeds and states.

For a multi-seed method, each held-out state's metric is first averaged across
the declared seeds.  The table statistic is then the mean and sample standard
deviation across the 128 statewise seed means.  Pooled and seed-level
statistics are retained alongside the primary statewise aggregation so the
reporting convention is explicit and reproducible.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def source_artifacts(encoded: str) -> list[dict[str, Any]]:
    path = Path(encoded).expanduser().resolve()
    if path.is_file():
        return [
            {
                "path": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        ]
    if path.is_dir():
        files = sorted(item for item in path.rglob("*") if item.is_file())
        if not files:
            raise ValueError(f"source directory is empty: {path}")
        return [
            {
                "path": str(item),
                "sha256": sha256(item),
                "bytes": item.stat().st_size,
            }
            for item in files
        ]
    raise FileNotFoundError(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty aggregate table")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def stats(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "sample_std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q95": float(np.quantile(array, 0.95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=128)
    args = parser.parse_args()

    run_root = args.run_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} already exists; aggregation outputs are immutable"
        )
    output_dir.mkdir(parents=True)
    summaries = sorted(run_root.glob("*/summary.json"))
    if not summaries:
        raise FileNotFoundError(f"no run summaries under {run_root}")

    groups: dict[str, list[dict[str, Any]]] = {}
    direction_hash: str | None = None
    protocol_reference: dict[str, Any] | None = None
    for summary_path in summaries:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("schema") != "table1-heldout-common-v1":
            continue
        observed_direction_hash = str(summary["directions_sha256"])
        if direction_hash is None:
            direction_hash = observed_direction_hash
        elif observed_direction_hash != direction_hash:
            raise ValueError("run summaries use different held-out directions")
        protocol = dict(summary["protocol"])
        if protocol_reference is None:
            protocol_reference = protocol
        elif protocol != protocol_reference:
            raise ValueError("run summaries use different evaluation protocols")
        per_sample = summary_path.with_name("per_sample.csv")
        rows = read_csv(per_sample)
        if len(rows) != args.expected_samples:
            raise ValueError(
                f"{per_sample}: expected {args.expected_samples} rows, found {len(rows)}"
            )
        record = {
            "summary_path": summary_path,
            "summary_sha256": sha256(summary_path),
            "per_sample_path": per_sample,
            "per_sample_sha256": sha256(per_sample),
            "summary": summary,
            "rows": rows,
        }
        groups.setdefault(str(summary["method"]), []).append(record)

    aggregate_rows: list[dict[str, Any]] = []
    aggregate_json: dict[str, Any] = {}
    for method, records in sorted(groups.items()):
        seeds = [str(record["summary"]["seed_label"]) for record in records]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"{method}: repeated seed labels {seeds}")
        by_seed_sample: list[dict[int, dict[str, str]]] = []
        initial_by_sample: dict[int, str] = {}
        for record in records:
            indexed = {int(row["sample"]): row for row in record["rows"]}
            if sorted(indexed) != list(range(args.expected_samples)):
                raise ValueError(f"{record['per_sample_path']}: noncanonical samples")
            for sample, row in indexed.items():
                encoded = row["initial_state"]
                if sample in initial_by_sample and initial_by_sample[sample] != encoded:
                    raise ValueError(
                        f"{method}: initial state differs across seeds at sample {sample}"
                    )
                initial_by_sample[sample] = encoded
            by_seed_sample.append(indexed)

        metrics = ("J", "R_sing")
        method_payload: dict[str, Any] = {
            "seeds": seeds,
            "runs": [
                {
                    "summary": str(record["summary_path"]),
                    "summary_sha256": record["summary_sha256"],
                    "per_sample": str(record["per_sample_path"]),
                    "per_sample_sha256": record["per_sample_sha256"],
                    "source": record["summary"]["source"],
                    "source_sha256": record["summary"]["source_sha256"],
                    "source_artifacts": source_artifacts(
                        record["summary"]["source"]
                    ),
                }
                for record in records
            ],
            "metrics": {},
        }
        flat_row: dict[str, Any] = {
            "method": method,
            "seeds": ";".join(seeds),
            "seed_count": len(seeds),
            "state_count": args.expected_samples,
        }
        for metric in metrics:
            matrix = np.asarray(
                [
                    [
                        float(indexed[sample][metric])
                        for sample in range(args.expected_samples)
                    ]
                    for indexed in by_seed_sample
                ],
                dtype=np.float64,
            )
            statewise_seed_mean = matrix.mean(axis=0)
            primary = stats(statewise_seed_mean)
            pooled = stats(matrix.reshape(-1))
            per_seed_means = matrix.mean(axis=1)
            seed_mean_stats = stats(per_seed_means)
            method_payload["metrics"][metric] = {
                "primary_statewise_seed_mean": primary,
                "pooled_seed_state": pooled,
                "per_seed_means": per_seed_means.tolist(),
                "per_seed_mean_statistics": seed_mean_stats,
            }
            flat_row[f"{metric}_mean"] = f"{primary['mean']:.12g}"
            flat_row[f"{metric}_sample_std"] = f"{primary['sample_std']:.12g}"
            flat_row[f"{metric}_median"] = f"{primary['median']:.12g}"
            flat_row[f"{metric}_q05"] = f"{primary['q05']:.12g}"
            flat_row[f"{metric}_q95"] = f"{primary['q95']:.12g}"
            flat_row[f"{metric}_table"] = (
                f"{primary['mean']:.6g} +/- {primary['sample_std']:.3g}"
            )
        aggregate_rows.append(flat_row)
        aggregate_json[method] = method_payload

    write_csv(output_dir / "table1_heldout_aggregate.csv", aggregate_rows)
    payload = {
        "schema": "table1-heldout-aggregate-v1",
        "aggregation": (
            "For each sample, average the metric across method seeds; then "
            "report mean and sample standard deviation across the 128 samples."
        ),
        "directions_sha256": direction_hash,
        "protocol": protocol_reference,
        "methods": aggregate_json,
        "table_csv": str((output_dir / "table1_heldout_aggregate.csv").resolve()),
    }
    (output_dir / "table1_heldout_aggregate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate_rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
