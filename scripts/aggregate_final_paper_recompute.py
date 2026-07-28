#!/usr/bin/env python3
"""Aggregate one staged final-paper recomputation into stable CSV/JSON fields.

The script consumes only artifacts produced by ``run_final_paper_recompute.py``.
It does not train models or edit LaTeX.  Every table row remains linked to the
checkpoint and raw file from which it was computed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_new_objective_related_work_paper_results import (  # noqa: E402
    NOMINAL,
    RESISTANT,
    common_schedule,
    evaluate_schedule,
    load_schedule,
)


CASES = ("time_only", "feedback_cf", "feedback_der")
STATES = ("nominal", "resistant_heavy")
QUANTITIES = ("H_u", "dH_u_dt", "d2H_u_dt2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {result}")
    return result


def diagnostic_dir(stage: Path, resolution: str, case: str) -> Path:
    return stage / "diagnostics" / resolution / case


def validate_diagnostics(
    stage: Path,
    resolutions: tuple[str, ...],
    *,
    feedback_refinement_multiplier: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    scalar_rows: list[dict[str, Any]] = []
    objective_rows: list[dict[str, Any]] = []
    checkpoint_hashes: dict[str, str] = {}
    expected_points = {"m16": 12801, "m32": 25601}

    for resolution in resolutions:
        for case in CASES:
            directory = diagnostic_dir(stage, resolution, case)
            summary_path = directory / "summary.json"
            timeseries_path = directory / "timeseries.npz"
            if not summary_path.is_file() or not timeseries_path.is_file():
                raise FileNotFoundError(
                    f"missing {resolution}/{case} diagnostic artifacts"
                )
            summary = load_json(summary_path)
            if int(summary["dense_points"]) != expected_points[resolution]:
                raise ValueError(
                    f"{summary_path}: dense_points={summary['dense_points']}, "
                    f"expected {expected_points[resolution]}"
                )
            checkpoint_hash = str(summary["checkpoint_sha256"])
            previous = checkpoint_hashes.setdefault(case, checkpoint_hash)
            if previous != checkpoint_hash:
                raise ValueError(
                    f"{case} changed checkpoint between diagnostic resolutions"
                )
            if tuple(float(v) for v in summary["interior"]) != (1.5, 8.0):
                raise ValueError(f"{summary_path}: expected interior [1.5, 8.0]")
            if case != "time_only" and int(
                summary.get("refinement_multiplier", -1)
            ) != feedback_refinement_multiplier:
                raise ValueError(
                    f"{summary_path}: final feedback diagnostics must declare "
                    "refinement_multiplier="
                    f"{feedback_refinement_multiplier}"
                )
            if case != "time_only" and int(
                summary.get("checkpoint_refinement_multiplier", -1)
            ) != feedback_refinement_multiplier:
                raise ValueError(
                    f"{summary_path}: checkpoint metadata does not record "
                    "refinement multiplier "
                    f"{feedback_refinement_multiplier}"
                )

            states = summary.get("states", {})
            if set(states) != set(STATES):
                raise ValueError(
                    f"{summary_path}: expected states {STATES}, found {sorted(states)}"
                )
            for state in STATES:
                state_payload = states[state]
                objective_rows.append(
                    {
                        "evaluation": "continuous_policy",
                        "resolution": resolution,
                        "case": case,
                        "state": state,
                        "physical_objective": finite(
                            state_payload["physical_objective"],
                            f"{resolution}/{case}/{state}/physical_objective",
                        ),
                        "normalized_objective": finite(
                            state_payload["normalized_objective"],
                            f"{resolution}/{case}/{state}/normalized_objective",
                        ),
                        "checkpoint_sha256": checkpoint_hash,
                        "source_summary": str(summary_path.resolve()),
                    }
                )
                region_keys = {
                    "singular_interior": "all_dense_points",
                    "singular_interior_refinement_grid": (
                        "checkpoint_refinement_grid"
                    ),
                    "singular_interior_held_out_from_refinement": (
                        "held_out_from_checkpoint_refinement"
                    ),
                }
                for metric_key, region_label in region_keys.items():
                    metrics = state_payload["metrics"][metric_key]
                    for quantity in QUANTITIES:
                        values = metrics[quantity]
                        count = int(values["count"])
                        if count == 0:
                            continue
                        scalar_rows.append(
                            {
                                "resolution": resolution,
                                "dense_points": expected_points[resolution],
                                "case": case,
                                "state": state,
                                "time_window": "1.5<=t<8.0",
                                "sample_class": region_label,
                                "quantity": quantity,
                                "count": count,
                                "rms": finite(
                                    values["rms"],
                                    f"{resolution}/{case}/{state}/"
                                    f"{metric_key}/{quantity}/rms",
                                ),
                                "mean_abs": finite(
                                    values["mean_abs"],
                                    f"{resolution}/{case}/{state}/"
                                    f"{metric_key}/{quantity}/mean_abs",
                                ),
                                "max_abs": finite(
                                    values["max_abs"],
                                    f"{resolution}/{case}/{state}/"
                                    f"{metric_key}/{quantity}/max_abs",
                                ),
                                "checkpoint_sha256": checkpoint_hash,
                                "source_summary": str(
                                    summary_path.resolve()
                                ),
                            }
                        )
    return scalar_rows, objective_rows, checkpoint_hashes


def convergence_rows(
    scalar_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {
        (row["resolution"], row["case"], row["state"], row["quantity"]): row
        for row in scalar_rows
        if row["sample_class"] == "all_dense_points"
    }
    rows: list[dict[str, Any]] = []
    for case in CASES:
        for state in STATES:
            for quantity in QUANTITIES:
                low = lookup[("m16", case, state, quantity)]
                high = lookup[("m32", case, state, quantity)]
                for statistic in ("rms", "mean_abs", "max_abs"):
                    low_value = float(low[statistic])
                    high_value = float(high[statistic])
                    denominator = max(abs(high_value), 1.0e-300)
                    rows.append(
                        {
                            "case": case,
                            "state": state,
                            "quantity": quantity,
                            "statistic": statistic,
                            "m16": low_value,
                            "m32": high_value,
                            "absolute_difference": abs(low_value - high_value),
                            "relative_difference_to_m32": (
                                abs(low_value - high_value) / denominator
                            ),
                        }
                    )
    return rows


def extract_control_and_adaptation(
    stage: Path,
    primary_resolution: str,
    tables: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    time_solution = tables / "time_only_solution_n800.npz"
    for case in CASES:
        path = diagnostic_dir(stage, primary_resolution, case) / "timeseries.npz"
        with np.load(path) as source:
            time = np.asarray(source["t"], dtype=np.float64)
            nominal = np.asarray(source["nominal__u"], dtype=np.float64)
            resistant = np.asarray(
                source["resistant_heavy__u"], dtype=np.float64
            )
            if time.shape != nominal.shape or nominal.shape != resistant.shape:
                raise ValueError(f"{path}: incompatible t/u shapes")
            difference = resistant - nominal
            rows.append(
                {
                    "resolution": primary_resolution,
                    "case": case,
                    "max_abs_delta_u": float(np.max(np.abs(difference))),
                    "mean_abs_delta_u": float(np.mean(np.abs(difference))),
                    "rms_delta_u": float(np.sqrt(np.mean(difference**2))),
                    "source_timeseries": str(path.resolve()),
                }
            )
            if case == "time_only":
                support = np.asarray(
                    source["is_transformer_support"], dtype=bool
                )
                support_t = time[support]
                support_u = nominal[support]
                if support_t.size != 801 or support_u.size != 801:
                    raise ValueError(
                        f"{path}: expected 801 Transformer support points"
                    )
                np.savez_compressed(
                    time_solution,
                    t=support_t,
                    u=support_u[:-1],
                    source_timeseries=np.asarray(str(path.resolve())),
                )
    return time_solution, rows


def evaluate_n800_related_rows(
    *,
    stage: Path,
    time_solution: Path,
    direct_nominal: Path,
    direct_resistant: Path,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    direct_paths = {
        "nominal": direct_nominal,
        "resistant_heavy": direct_resistant,
    }
    initial_states = {"nominal": NOMINAL, "resistant_heavy": RESISTANT}
    references: dict[str, float] = {}
    for state, path in direct_paths.items():
        t, u = load_schedule(path)
        common_t, common_u = common_schedule(t, u, 800)
        references[state] = float(
            evaluate_schedule(common_t, common_u, initial_states[state])["J"]
        )

    time_t, time_u = load_schedule(time_solution)
    time_t, time_u = common_schedule(time_t, time_u, 800)
    rows: list[dict[str, Any]] = []
    for state in STATES:
        objective = float(
            evaluate_schedule(time_t, time_u, initial_states[state])["J"]
        )
        reference = references[state]
        rows.append(
            {
                "case": "time_only",
                "state": state,
                "J": objective,
                "direct_reference_J": reference,
                "objective_gap": objective - reference,
                "relative_gap_percent": 100.0
                * (objective - reference)
                / reference,
                "source": str(time_solution.resolve()),
            }
        )

    feedback_csv = stage / "timing" / "feedback" / "per_run_ours.csv"
    with feedback_csv.open(encoding="utf-8", newline="") as stream:
        feedback = list(csv.DictReader(stream))
    mapping = {"case1": "feedback_cf", "case2": "feedback_der"}
    for raw in feedback:
        case = mapping.get(raw["variant"])
        if case is None:
            continue
        state = raw["state_id"]
        reference = references[state]
        objective = float(raw["J_unregularized"])
        rows.append(
            {
                "case": case,
                "state": state,
                "J": objective,
                "direct_reference_J": reference,
                "objective_gap": objective - reference,
                "relative_gap_percent": 100.0
                * (objective - reference)
                / reference,
                "query_median_ms": float(raw["query_median_ms"]),
                "closed_loop_median_ms": float(
                    raw["closed_loop_median_ms"]
                ),
                "checkpoint_sha256": raw["checkpoint_sha256"],
                "source": str(feedback_csv.resolve()),
            }
        )
    expected = {(case, state) for case in CASES for state in STATES}
    found = {(row["case"], row["state"]) for row in rows}
    if found != expected:
        raise ValueError(
            f"related-work ours rows incomplete: missing={sorted(expected-found)}"
        )
    return rows, references


def wilson_interval(successes: int, count: int) -> tuple[float, float]:
    if count <= 0:
        raise ValueError("Wilson interval requires a positive count")
    z = 1.959963984540054
    fraction = successes / count
    denominator = 1.0 + z * z / count
    center = (fraction + z * z / (2.0 * count)) / denominator
    radius = (
        z
        * math.sqrt(
            fraction * (1.0 - fraction) / count
            + z * z / (4.0 * count * count)
        )
        / denominator
    )
    return center - radius, center + radius


def heldout_rows(
    stage: Path, *, report_scale_factor: float
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case, short in (("feedback_cf", "cf"), ("feedback_der", "der")):
        path = stage / "heldout" / short / "summary.json"
        payload = load_json(path)
        protocol = payload["protocol"]
        if int(protocol["total_samples_per_radius"]) != 128:
            raise ValueError(f"{path}: held-out evaluation is not 128 states")
        if protocol["direction_families"] != ["random"]:
            raise ValueError(f"{path}: expected the shared random direction family")
        for radius, block in payload["radii"].items():
            advantage = block["post"]["feedback_advantage"]
            controls = block["post"].get("feedback_control_cross_state_sd")
            count = int(advantage["count"])
            successes = int(round(float(advantage["win_fraction"]) * count))
            wilson_low, wilson_high = wilson_interval(successes, count)
            rows.append(
                {
                    "case": case,
                    "radius": float(radius),
                    "count": count,
                    "mean_delta_J": report_scale_factor * finite(
                        advantage["mean"], f"{case}/{radius}/mean"
                    ),
                    "median_delta_J": report_scale_factor * finite(
                        advantage["median"], f"{case}/{radius}/median"
                    ),
                    "min_delta_J": report_scale_factor * finite(
                        advantage["min"], f"{case}/{radius}/min"
                    ),
                    "max_delta_J": report_scale_factor * finite(
                        advantage["max"], f"{case}/{radius}/max"
                    ),
                    "fraction_improved": finite(
                        advantage["win_fraction"],
                        f"{case}/{radius}/win_fraction",
                    ),
                    "states_improved": successes,
                    "fraction_improved_wilson_low": wilson_low,
                    "fraction_improved_wilson_high": wilson_high,
                    "mean_ci_low": report_scale_factor * finite(
                        advantage["mean_bootstrap_95ci"][0],
                        f"{case}/{radius}/mean_ci_low",
                    ),
                    "mean_ci_high": report_scale_factor * finite(
                        advantage["mean_bootstrap_95ci"][1],
                        f"{case}/{radius}/mean_ci_high",
                    ),
                    "control_cross_state_sd": (
                        finite(controls, f"{case}/{radius}/control_sd")
                        if controls is not None
                        else ""
                    ),
                    "source_summary": str(path.resolve()),
                }
            )
    return rows


def timing_rows(stage: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    time_path = stage / "timing" / "time_only.json"
    time_payload = load_json(time_path)
    rows.append(
        {
            "case": "time_only",
            "state": "not_applicable",
            "metric": "801-node loaded-network forward",
            "median_ms": float(time_payload["forward_time_ms"]["median"]),
            "warmups": int(time_payload["warmup_runs"]),
            "repeats": int(time_payload["timed_runs"]),
            "threads": int(time_payload["torch_threads"]),
            "source": str(time_path.resolve()),
        }
    )
    feedback_path = stage / "timing" / "feedback" / "per_run_ours.csv"
    with feedback_path.open(encoding="utf-8", newline="") as stream:
        for raw in csv.DictReader(stream):
            case = {
                "case1": "feedback_cf",
                "case2": "feedback_der",
            }[raw["variant"]]
            rows.append(
                {
                    "case": case,
                    "state": raw["state_id"],
                    "metric": "single policy evaluation",
                    "median_ms": float(raw["query_median_ms"]),
                    "warmups": "",
                    "repeats": int(raw["query_repeats"]),
                    "threads": 1,
                    "source": str(feedback_path.resolve()),
                }
            )
            rows.append(
                {
                    "case": case,
                    "state": raw["state_id"],
                    "metric": "800-step closed-loop rollout",
                    "median_ms": float(raw["closed_loop_median_ms"]),
                    "warmups": int(raw["closed_loop_warmups"]),
                    "repeats": int(raw["closed_loop_repeats"]),
                    "threads": 1,
                    "source": str(feedback_path.resolve()),
                }
            )
    return rows


def output_manifest(stage: Path, tables: Path) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for path in sorted(tables.iterdir()):
        if path.is_file():
            artifacts[path.name] = {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
    return {
        "schema": "final-paper-recompute-tables-v1",
        "stage": str(stage.resolve()),
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--direct-nominal", type=Path, required=True)
    parser.add_argument("--direct-resistant", type=Path, required=True)
    parser.add_argument("--report-scale-factor", type=float, default=400.0)
    parser.add_argument(
        "--feedback-refinement-multiplier", type=int, required=True
    )
    args = parser.parse_args()

    stage = args.stage_dir.expanduser().resolve()
    tables = stage / "tables"
    if tables.exists():
        raise FileExistsError(f"refusing to overwrite existing {tables}")
    tables.mkdir(parents=True)
    direct_nominal = args.direct_nominal.expanduser().resolve()
    direct_resistant = args.direct_resistant.expanduser().resolve()

    if args.feedback_refinement_multiplier < 1:
        raise ValueError(
            "--feedback-refinement-multiplier must be positive"
        )
    scalar, continuous_objectives, checkpoint_hashes = validate_diagnostics(
        stage,
        ("m16", "m32"),
        feedback_refinement_multiplier=args.feedback_refinement_multiplier,
    )
    write_csv(tables / "continuous_scalar_metrics.csv", scalar)
    write_csv(
        tables / "continuous_resolution_convergence.csv",
        convergence_rows(scalar),
    )
    write_csv(tables / "continuous_objectives.csv", continuous_objectives)

    time_solution, adaptation = extract_control_and_adaptation(
        stage, "m32", tables
    )
    write_csv(tables / "control_adaptation.csv", adaptation)
    related_rows, references = evaluate_n800_related_rows(
        stage=stage,
        time_solution=time_solution,
        direct_nominal=direct_nominal,
        direct_resistant=direct_resistant,
    )
    write_csv(tables / "related_work_ours_rows.csv", related_rows)
    if args.report_scale_factor <= 0.0:
        raise ValueError("--report-scale-factor must be positive")
    write_csv(
        tables / "heldout_128_summary.csv",
        heldout_rows(
            stage, report_scale_factor=float(args.report_scale_factor)
        ),
    )
    write_csv(tables / "timing.csv", timing_rows(stage))

    summary = {
        "schema": "final-paper-recompute-summary-v1",
        "primary_continuous_resolution": "m32",
        "verification_continuous_resolution": "m16",
        "interior": [1.5, 8.0],
        "physical_objective_weights": [1.0, 40.0, 8000.0],
        "training_to_physical_scale": float(args.report_scale_factor),
        "feedback_refinement_multiplier": (
            args.feedback_refinement_multiplier
        ),
        "checkpoint_sha256": checkpoint_hashes,
        "direct_reference_J_n800": references,
        "time_only_solution_n800": str(time_solution.resolve()),
        "field_files": {
            "scalar_metrics": "continuous_scalar_metrics.csv",
            "resolution_check": "continuous_resolution_convergence.csv",
            "continuous_objectives": "continuous_objectives.csv",
            "control_adaptation": "control_adaptation.csv",
            "heldout": "heldout_128_summary.csv",
            "related_work_ours": "related_work_ours_rows.csv",
            "timing": "timing.csv",
        },
    }
    (tables / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    manifest = output_manifest(stage, tables)
    (tables / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
