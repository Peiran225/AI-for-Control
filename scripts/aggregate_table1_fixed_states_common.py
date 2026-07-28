#!/usr/bin/env python3
"""Aggregate and audit the common-evaluator fixed-state Table-1 runs.

The primary table convention is a mean and sample standard deviation across
the available independent seeds for each fixed initial state.  Single-run
methods retain a zero numerical standard deviation in the machine-readable
aggregate but are marked as selected/locked runs rather than as replicated
estimates.

The comparison artifact is deliberately separate from the paper source.  It
compares the recomputed values with the numbers currently typeset in the main
Table 1 and in the detailed supplementary fixed-state table; it never edits
either document.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


METHOD_ORDER = (
    "direct_statewise",
    "neural_pmp_learned",
    "neural_pmp_exact",
    "adaptive_hjb_nn",
    "pi_deeponet",
    "deepbsde",
    "pmp_kkt_time",
    "pmp_kkt_cf",
    "pmp_kkt_der",
)
STATE_ORDER = ("nominal", "resistant_heavy")
METRICS = (
    "J",
    "RMS_H_u",
    "RMS_dH_u_dt",
    "RMS_d2H_u_dt2",
    "R_sing",
)
EXPECTED_SEEDS = {
    "direct_statewise": 1,
    "neural_pmp_learned": 3,
    "neural_pmp_exact": 1,
    "adaptive_hjb_nn": 3,
    "pi_deeponet": 3,
    "deepbsde": 3,
    "pmp_kkt_time": 1,
    "pmp_kkt_cf": 1,
    "pmp_kkt_der": 1,
}


def reported(
    mean: float,
    mean_tolerance: float,
    sample_std: float | None = None,
    std_tolerance: float | None = None,
) -> dict[str, float | None]:
    return {
        "mean": mean,
        "mean_tolerance": mean_tolerance,
        "sample_std": sample_std,
        "std_tolerance": std_tolerance,
    }


# Values currently typeset in experiments_template.tex, Table 1.  Tolerances
# are one half-unit in the last displayed digit, not statistical tolerances.
CURRENT_MAIN_TABLE: dict[str, dict[str, dict[str, dict[str, float | None]]]] = {
    "direct_statewise": {
        "nominal": {
            "J": reported(124866.877, 0.0005),
            "R_sing": reported(0.294, 0.0005),
        },
        "resistant_heavy": {
            "J": reported(124408.936, 0.0005),
            "R_sing": reported(0.317, 0.0005),
        },
    },
    "neural_pmp_learned": {
        "nominal": {
            "J": reported(147775.0, 0.5, 6382.0, 0.5),
            "R_sing": reported(1.261e4, 5.0, 0.227e4, 5.0),
        },
        "resistant_heavy": {
            "J": reported(146747.0, 0.5, 6250.0, 0.5),
            "R_sing": reported(1.242e4, 5.0, 0.224e4, 5.0),
        },
    },
    "neural_pmp_exact": {
        "nominal": {
            "J": reported(124871.919, 0.0005),
            "R_sing": reported(60.83, 0.005),
        },
        "resistant_heavy": {
            "J": reported(124416.306, 0.0005),
            "R_sing": reported(92.14, 0.005),
        },
    },
    "adaptive_hjb_nn": {
        "nominal": {
            "J": reported(129111.7, 0.05, 37.3, 0.05),
            "R_sing": reported(1637.0, 0.5, 12.0, 0.5),
        },
        "resistant_heavy": {
            "J": reported(128764.0, 0.05, 70.6, 0.05),
            "R_sing": reported(1678.0, 0.5, 20.0, 0.5),
        },
    },
    "pi_deeponet": {
        "nominal": {
            "J": reported(214415.0, 0.5, 8362.0, 0.5),
            "R_sing": reported(3.397e4, 5.0, 0.227e4, 5.0),
        },
        "resistant_heavy": {
            "J": reported(213351.0, 0.5, 8230.0, 0.5),
            "R_sing": reported(3.382e4, 5.0, 0.224e4, 5.0),
        },
    },
    "deepbsde": {
        "nominal": {
            "J": reported(219172.0, 0.5, 71896.0, 0.5),
            "R_sing": reported(3.165e4, 5.0, 2.352e4, 5.0),
        },
        "resistant_heavy": {
            "J": reported(178600.0, 0.5, 69809.0, 0.5),
            "R_sing": reported(2.048e4, 5.0, 2.163e4, 5.0),
        },
    },
    "pmp_kkt_time": {
        "nominal": {
            "J": reported(124867.181, 0.0005),
            "R_sing": reported(3.078e-3, 0.5e-6),
        },
        "resistant_heavy": {
            "J": reported(124413.929, 0.0005),
            "R_sing": reported(91.14, 0.005),
        },
    },
    "pmp_kkt_cf": {
        "nominal": {
            "J": reported(124867.159, 0.0005),
            "R_sing": reported(2.722e-3, 0.5e-6),
        },
        "resistant_heavy": {
            "J": reported(124409.196, 0.0005),
            "R_sing": reported(9.834e-2, 0.5e-5),
        },
    },
    "pmp_kkt_der": {
        "nominal": {
            "J": reported(124867.159, 0.0005),
            "R_sing": reported(2.728e-3, 0.5e-6),
        },
        "resistant_heavy": {
            "J": reported(124409.196, 0.0005),
            "R_sing": reported(6.733e-2, 0.5e-5),
        },
    },
}


# Values currently typeset in supplementary Table S2 (the strict off-q8
# diagnostic table).  The objective is the same n=800 ZOH objective, while the
# three derivative columns use a different continuous/off-grid query protocol.
CURRENT_SUPPLEMENT_DETAIL = {
    "pmp_kkt_time": {
        "nominal": {
            "J": reported(124867.180821, 0.5e-6),
            "RMS_H_u": reported(1.47033e-3, 0.5e-8),
            "RMS_dH_u_dt": reported(4.49437e-4, 0.5e-9),
            "RMS_d2H_u_dt2": reported(2.68034e-3, 0.5e-8),
        },
        "resistant_heavy": {
            "J": reported(124413.928984, 0.5e-6),
            "RMS_H_u": reported(90.4714, 0.5e-4),
            "RMS_dH_u_dt": reported(9.21195, 0.5e-5),
            "RMS_d2H_u_dt2": reported(6.07223, 0.5e-5),
        },
    },
    "pmp_kkt_cf": {
        "nominal": {
            "J": reported(124867.158714, 0.5e-6),
            "RMS_H_u": reported(4.95392e-4, 0.5e-9),
            "RMS_dH_u_dt": reported(2.99322e-4, 0.5e-9),
            "RMS_d2H_u_dt2": reported(2.67373e-3, 0.5e-8),
        },
        "resistant_heavy": {
            "J": reported(124409.195709, 0.5e-6),
            "RMS_H_u": reported(1.77150e-2, 0.5e-7),
            "RMS_dH_u_dt": reported(1.64451e-2, 0.5e-7),
            "RMS_d2H_u_dt2": reported(9.53027e-2, 0.5e-7),
        },
    },
    "pmp_kkt_der": {
        "nominal": {
            "J": reported(124867.158718, 0.5e-6),
            "RMS_H_u": reported(5.17958e-4, 0.5e-9),
            "RMS_dH_u_dt": reported(3.11787e-4, 0.5e-9),
            "RMS_d2H_u_dt2": reported(2.67377e-3, 0.5e-8),
        },
        "resistant_heavy": {
            "J": reported(124409.195727, 0.5e-6),
            "RMS_H_u": reported(2.12425e-2, 0.5e-7),
            "RMS_dH_u_dt": reported(1.33235e-2, 0.5e-7),
            "RMS_d2H_u_dt2": reported(6.24706e-2, 0.5e-7),
        },
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "sample_std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def difference_row(
    *,
    source_table: str,
    source_path: str,
    method: str,
    state_id: str,
    metric: str,
    statistic: str,
    recomputed: float,
    report: float,
    tolerance: float,
) -> dict[str, Any]:
    absolute = abs(recomputed - report)
    relative = absolute / abs(report) if report != 0.0 else float("nan")
    return {
        "source_table": source_table,
        "source_path": source_path,
        "method": method,
        "state_id": state_id,
        "metric": metric,
        "statistic": statistic,
        "recomputed": f"{recomputed:.12g}",
        "currently_reported": f"{report:.12g}",
        "absolute_difference": f"{absolute:.12g}",
        "relative_difference": f"{relative:.12g}",
        "display_rounding_tolerance": f"{tolerance:.12g}",
        "matches_current_display_rounding": absolute <= tolerance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--main-table-source", type=Path, required=True)
    parser.add_argument("--supplement-source", type=Path, required=True)
    args = parser.parse_args()

    run_root = args.run_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    main_source = args.main_table_source.expanduser().resolve()
    supplement_source = args.supplement_source.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} already exists; aggregation outputs are immutable"
        )
    if not main_source.is_file() or not supplement_source.is_file():
        raise FileNotFoundError("current report sources must both exist")
    output_dir.mkdir(parents=True)

    records: dict[str, list[dict[str, Any]]] = {}
    protocol_reference: dict[str, Any] | None = None
    problem_reference: dict[str, Any] | None = None
    initial_states: dict[str, str] = {}
    for summary_path in sorted(run_root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("schema") != "table1-fixed-states-common-v1":
            continue
        method = str(summary["method"])
        if method not in EXPECTED_SEEDS:
            raise ValueError(f"unexpected method in fixed-state runs: {method}")
        protocol = dict(summary["protocol"])
        problem = dict(summary["problem"])
        if protocol_reference is None:
            protocol_reference = protocol
            problem_reference = problem
        elif protocol != protocol_reference or problem != problem_reference:
            raise ValueError("fixed-state summaries use different protocols/problems")

        csv_path = summary_path.with_name("fixed_states.csv")
        rows = read_csv(csv_path)
        if len(rows) != len(STATE_ORDER):
            raise ValueError(f"{csv_path}: expected two fixed-state rows")
        indexed = {row["state_id"]: row for row in rows}
        if set(indexed) != set(STATE_ORDER):
            raise ValueError(f"{csv_path}: wrong fixed-state identifiers")
        for state_id, row in indexed.items():
            encoded = row["initial_state"]
            if state_id in initial_states and initial_states[state_id] != encoded:
                raise ValueError(f"{state_id}: initial vector differs across runs")
            initial_states[state_id] = encoded
        records.setdefault(method, []).append(
            {
                "summary": summary,
                "summary_path": summary_path,
                "summary_sha256": sha256(summary_path),
                "csv_path": csv_path,
                "csv_sha256": sha256(csv_path),
                "rows": indexed,
            }
        )

    if set(records) != set(EXPECTED_SEEDS):
        missing = sorted(set(EXPECTED_SEEDS) - set(records))
        extra = sorted(set(records) - set(EXPECTED_SEEDS))
        raise ValueError(f"fixed-state method mismatch: missing={missing}, extra={extra}")
    for method, expected in EXPECTED_SEEDS.items():
        observed = len(records[method])
        labels = [str(record["summary"]["seed_label"]) for record in records[method]]
        if observed != expected or len(labels) != len(set(labels)):
            raise ValueError(
                f"{method}: expected {expected} unique runs, observed {labels}"
            )

    aggregate_rows: list[dict[str, Any]] = []
    aggregate_payload: dict[str, Any] = {}
    direct_objective: dict[str, float] = {}
    for method in METHOD_ORDER:
        method_records = records[method]
        seed_labels = [str(record["summary"]["seed_label"]) for record in method_records]
        method_payload: dict[str, Any] = {
            "seed_labels": seed_labels,
            "runs": [
                {
                    "summary": str(record["summary_path"]),
                    "summary_sha256": record["summary_sha256"],
                    "fixed_states_csv": str(record["csv_path"]),
                    "fixed_states_csv_sha256": record["csv_sha256"],
                    "sources": record["summary"]["sources"],
                }
                for record in method_records
            ],
            "states": {},
        }
        for state_id in STATE_ORDER:
            metric_stats = {
                metric: stats(
                    [
                        float(record["rows"][state_id][metric])
                        for record in method_records
                    ]
                )
                for metric in METRICS
            }
            if method == "direct_statewise":
                direct_objective[state_id] = metric_stats["J"]["mean"]
            method_payload["states"][state_id] = metric_stats
            row: dict[str, Any] = {
                "method": method,
                "state_id": state_id,
                "seed_labels": ";".join(seed_labels),
                "seed_count": len(seed_labels),
                "replication_semantics": (
                    "mean and sample SD across independent seeds"
                    if len(seed_labels) > 1
                    else "selected/locked run; no across-seed uncertainty"
                ),
            }
            for metric in METRICS:
                row[f"{metric}_mean"] = f"{metric_stats[metric]['mean']:.12g}"
                row[f"{metric}_sample_std"] = (
                    f"{metric_stats[metric]['sample_std']:.12g}"
                )
            aggregate_rows.append(row)
        aggregate_payload[method] = method_payload

    for row in aggregate_rows:
        state_id = str(row["state_id"])
        objective = float(row["J_mean"])
        gap = 100.0 * (objective - direct_objective[state_id]) / direct_objective[state_id]
        row["Delta_J_percent_mean"] = f"{gap:.12g}"
        row["Delta_J_percent_sample_std"] = (
            f"{100.0 * float(row['J_sample_std']) / direct_objective[state_id]:.12g}"
        )
        aggregate_payload[str(row["method"])]["states"][state_id][
            "Delta_J_percent"
        ] = {
            "mean": gap,
            "sample_std": (
                100.0
                * float(row["J_sample_std"])
                / direct_objective[state_id]
            ),
        }

    comparison_rows: list[dict[str, Any]] = []
    aggregate_index = {
        (str(row["method"]), str(row["state_id"])): row
        for row in aggregate_rows
    }
    main_path_string = str(main_source)
    supplement_path_string = str(supplement_source)
    for method, state_payload in CURRENT_MAIN_TABLE.items():
        for state_id, metric_payload in state_payload.items():
            aggregate = aggregate_index[(method, state_id)]
            for metric, report_payload in metric_payload.items():
                comparison_rows.append(
                    difference_row(
                        source_table="main Table 1",
                        source_path=main_path_string,
                        method=method,
                        state_id=state_id,
                        metric=metric,
                        statistic="mean",
                        recomputed=float(aggregate[f"{metric}_mean"]),
                        report=float(report_payload["mean"]),
                        tolerance=float(report_payload["mean_tolerance"]),
                    )
                )
                if report_payload["sample_std"] is not None:
                    comparison_rows.append(
                        difference_row(
                            source_table="main Table 1",
                            source_path=main_path_string,
                            method=method,
                            state_id=state_id,
                            metric=metric,
                            statistic="sample_std",
                            recomputed=float(aggregate[f"{metric}_sample_std"]),
                            report=float(report_payload["sample_std"]),
                            tolerance=float(report_payload["std_tolerance"]),
                        )
                    )

    for method, state_payload in CURRENT_SUPPLEMENT_DETAIL.items():
        for state_id, metric_payload in state_payload.items():
            aggregate = aggregate_index[(method, state_id)]
            for metric, report_payload in metric_payload.items():
                comparison_rows.append(
                    difference_row(
                        source_table="supplement fixed-state derivative table",
                        source_path=supplement_path_string,
                        method=method,
                        state_id=state_id,
                        metric=metric,
                        statistic="mean",
                        recomputed=float(aggregate[f"{metric}_mean"]),
                        report=float(report_payload["mean"]),
                        tolerance=float(report_payload["mean_tolerance"]),
                    )
                )

    write_csv(output_dir / "table1_fixed_aggregate.csv", aggregate_rows)
    write_csv(output_dir / "table1_fixed_comparison.csv", comparison_rows)
    mismatch_rows = [
        row
        for row in comparison_rows
        if not bool(row["matches_current_display_rounding"])
    ]
    payload = {
        "schema": "table1-fixed-state-audit-v1",
        "aggregation": (
            "For each fixed state and metric, report the mean and sample "
            "standard deviation across available independent seeds. "
            "Single-run methods are locked/selected results, not uncertainty "
            "estimates."
        ),
        "problem": problem_reference,
        "protocol": protocol_reference,
        "initial_states": {
            state_id: json.loads(encoded)
            for state_id, encoded in initial_states.items()
        },
        "methods": aggregate_payload,
        "current_report_sources": {
            "main_table": {
                "path": main_path_string,
                "sha256": sha256(main_source),
            },
            "supplement": {
                "path": supplement_path_string,
                "sha256": sha256(supplement_source),
            },
        },
        "comparison": {
            "row_count": len(comparison_rows),
            "mismatch_count": len(mismatch_rows),
            "mismatch_definition": (
                "absolute recomputed-minus-reported difference exceeds one "
                "half-unit in the last currently displayed digit"
            ),
            "mismatches": mismatch_rows,
        },
        "outputs": {
            "aggregate_csv": str(
                (output_dir / "table1_fixed_aggregate.csv").resolve()
            ),
            "comparison_csv": str(
                (output_dir / "table1_fixed_comparison.csv").resolve()
            ),
        },
    }
    (output_dir / "table1_fixed_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "methods": len(aggregate_payload),
                "aggregate_rows": len(aggregate_rows),
                "comparison_rows": len(comparison_rows),
                "mismatches": len(mismatch_rows),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
