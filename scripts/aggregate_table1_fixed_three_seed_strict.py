#!/usr/bin/env python3
"""Aggregate fixed-state Table-1 results across three training seeds.

``J`` is read only from the objective-only RK4/ZOH evaluator.  ``R_sing`` is
read only from the strict continuous-policy/off-grid diagnostics on
``[1.5,8.0)``.  This separation prevents the q-refined ZOH diagnostic from
being mixed into the paper table.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


REPLICATES = ("base4_fb31_locked", "base2_fb32", "base3_fb33")
STATES = ("nominal", "resistant_heavy")
QUANTITIES = ("H_u", "dH_u_dt", "d2H_u_dt2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def mean_sd(values: list[float]) -> dict[str, Any]:
    if len(values) != 3:
        raise ValueError(f"expected three training seeds, found {len(values)}")
    return {
        "count": 3,
        "values": values,
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
    }


def expected_state(state: str) -> list[float]:
    if state == "nominal":
        return [10.0] * 21
    return [8.0 + 0.2 * index for index in range(21)]


def objective_path(root: Path, replicate: str, method: str) -> Path:
    base = root / "runs" / replicate / method
    if method == "time" and replicate != "base4_fb31_locked":
        preferred = base / "fixed_objective_common_v2/summary.json"
        if preferred.is_file():
            return preferred
    return base / "fixed_objective_common/summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--methods",
        default="time,cf,der",
        help="comma-separated subset of time,cf,der",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.evaluation_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    if not methods or any(method not in {"time", "cf", "der"} for method in methods):
        raise ValueError("--methods must be a subset of time,cf,der")
    if output.exists():
        raise FileExistsError(output)

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "schema": "table1-fixed-three-seed-strict-v1",
        "replicates": list(REPLICATES),
        "methods": {},
        "uncertainty_definition": (
            "mean and sample standard deviation across three independently "
            "trained seeds"
        ),
        "metric_provenance": {
            "J": (
                "objective-only n=800 left-endpoint ZOH, float64 RK4 with "
                "four substeps per interval"
            ),
            "R_sing": (
                "strict continuous-policy off-grid scalar diagnostics on "
                "[1.5,8.0): sqrt(RMS(psi)^2+RMS(dot psi)^2+"
                "RMS(ddot psi)^2)"
            ),
        },
    }
    source_files: list[Path] = []
    for method in methods:
        method_payload: dict[str, Any] = {}
        for state in STATES:
            values = {"J": [], "R_sing": []}
            for replicate in REPLICATES:
                objective_file = objective_path(root, replicate, method)
                strict_file = (
                    root
                    / "runs"
                    / replicate
                    / method
                    / "fixed_continuous_m32"
                    / "summary.json"
                )
                if not objective_file.is_file() or not strict_file.is_file():
                    raise FileNotFoundError(
                        f"missing {replicate}/{method}: "
                        f"{objective_file} or {strict_file}"
                    )
                source_files.extend((objective_file, strict_file))
                objective = load(objective_file)
                strict = load(strict_file)
                objective_row = next(
                    row
                    for row in objective["rows"]
                    if row["sample"] == state
                )
                strict_state = strict["states"][state]
                actual_state = [float(value) for value in strict_state["initial_state"]]
                expected = expected_state(state)
                if len(actual_state) != 21 or any(
                    not math.isclose(actual, target, rel_tol=0.0, abs_tol=1.0e-10)
                    for actual, target in zip(actual_state, expected)
                ):
                    raise ValueError(
                        f"{replicate}/{method}/{state}: wrong initial state"
                    )
                metrics = strict_state["metrics"][
                    "singular_interior_strict_off_grid"
                ]
                components = {
                    quantity: float(metrics[quantity]["rms"])
                    for quantity in QUANTITIES
                }
                residual = math.sqrt(
                    sum(value * value for value in components.values())
                )
                record = {
                    "method": method,
                    "replicate": replicate,
                    "state": state,
                    "J": float(objective_row["J"]),
                    "R_sing": residual,
                    **{f"RMS_{key}": value for key, value in components.items()},
                    "objective_source_sha256": objective_row["source_sha256"],
                    "strict_checkpoint_sha256": strict["checkpoint_sha256"],
                }
                rows.append(record)
                values["J"].append(record["J"])
                values["R_sing"].append(record["R_sing"])
            method_payload[state] = {
                metric: mean_sd(metric_values)
                for metric, metric_values in values.items()
            }
        summary["methods"][method] = method_payload

    output.mkdir(parents=True)
    fields = list(rows[0])
    with (output / "per_seed.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    audit = {
        "schema": "table1-fixed-three-seed-strict-audit-v1",
        "evaluation_root": str(root),
        "structured_resistant_state": expected_state("resistant_heavy"),
        "structured_resistant_total": sum(expected_state("resistant_heavy")),
        "inputs": [
            {
                "path": str(path),
                "sha256": sha256(path),
            }
            for path in sorted(set(source_files))
        ],
    }
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
