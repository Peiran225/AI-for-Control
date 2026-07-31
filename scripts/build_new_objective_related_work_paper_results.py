#!/usr/bin/env python3
"""Assemble matched-weight related-work results for the AAAI experiment paper.

All policies are evaluated under the physical objective

    (alpha, beta, gamma) = (1, 40, 8000)

on a common 800-interval left-endpoint/ZOH execution grid.  Time-only
controls are resampled as piecewise-constant schedules.  Feedback results are
read from ``compare_feedback_related_work.py``, which re-queries every policy
at every common-grid left endpoint.  The two direct-transcription controls
are numerical references, not learned baselines.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter
from scipy.integrate import solve_ivp


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tumor_problem import TumorProblem, dynamics_numpy  # noqa: E402


DEFAULT_OUTPUT = ROOT / "outputs/related_work_a1_b40_g8000_20260723/paper_results"
DEFAULT_FIGURE_DIR = ROOT / "reports/aaai2027_experiments/figures"
DEFAULT_FEEDBACK = (
    ROOT / "outputs/related_work_a1_b40_g8000_20260723/feedback_raw/per_run.csv"
)
DEFAULT_FEEDBACK_PROTOCOL = (
    ROOT / "outputs/related_work_a1_b40_g8000_20260723/feedback_raw/protocol.json"
)
DEFAULT_DIRECT = (
    ROOT
    / "outputs/new_objective_alpha1_grid_20260721_v2/a1_b40_g8000/"
    "direct_nominal/n800_strict_final/scale_1_direct_solution.npz"
)
DEFAULT_RESISTANT_DIRECT = (
    ROOT
    / "outputs/related_work_a1_b40_g8000_20260723/"
    "resistant_direct_refined/solution.npz"
)
DEFAULT_OURS = (
    ROOT
    / "outputs/new_objective_alpha1_grid_20260721_v2/a1_b40_g8000/"
    "time_only/deep_refine_kkt_head_ols_exact_v3/solution.npz"
)
DEFAULT_EXACT_NEURAL_PMP = (
    ROOT
    / "paper_runs/faithful_related_work/"
    "neural_pmp_exact_a1_b40_g8000_n800_20260723/"
    "best_neural_pmp_solution.npz"
)
DEFAULT_LEARNED_NEURAL_PMP = (
    ROOT
    / "paper_runs/faithful_related_work/"
    "neural_pmp_tumor_a1_b40_g8000_budgeted_3seed_20260723"
)

PROBLEM = TumorProblem(
    T=10.0,
    m=21,
    umax=3.0,
    alpha=1.0,
    beta=40.0,
    gamma=8000.0,
    n0=10.0,
    m_suppression=0.5,
)
NOMINAL = np.full(PROBLEM.m, 10.0, dtype=np.float64)
RESISTANT = np.linspace(9.0, 11.0, PROBLEM.m, dtype=np.float64)

BLUE = "#2F6B9A"
ORANGE = "#C65A2E"
GREEN = "#3F7F5F"
PURPLE = "#7B5A93"
REFERENCE_GRAY = "#4D4D4D"
GRID_GRAY = "#D8D8D8"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_schedule(path: Path, *, control_key: str = "u") -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as pack:
        t = np.asarray(pack["t"], dtype=np.float64).reshape(-1)
        u = np.asarray(pack[control_key], dtype=np.float64).reshape(-1)
    if u.size == t.size:
        u = u[:-1]
    if t.size != u.size + 1:
        raise ValueError(f"{path}: expected len(t)=len(u)+1, found {t.size}/{u.size}")
    if (
        not np.all(np.diff(t) > 0.0)
        or not np.isclose(t[0], 0.0)
        or not np.isclose(t[-1], PROBLEM.T)
    ):
        raise ValueError(f"{path}: invalid time grid")
    if float(u.min()) < -1.0e-9 or float(u.max()) > PROBLEM.umax + 1.0e-9:
        raise ValueError(f"{path}: infeasible control")
    return t, np.clip(u, 0.0, PROBLEM.umax)


def common_schedule(t: np.ndarray, u: np.ndarray, intervals: int) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(0.0, PROBLEM.T, intervals + 1, dtype=np.float64)
    left = grid[:-1]
    indices = np.searchsorted(t, left, side="right") - 1
    indices = np.clip(indices, 0, u.size - 1)
    return grid, np.asarray(u[indices], dtype=np.float64)


def evaluate_schedule(
    t: np.ndarray,
    u: np.ndarray,
    initial_state: np.ndarray,
    *,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-12,
) -> dict[str, float]:
    """Evaluate a fixed schedule with the same segmented physical integrator."""

    state0 = np.asarray(initial_state, dtype=np.float64).reshape(PROBLEM.m)
    if not np.all(state0 > 0.0):
        raise ValueError("initial state must be positive")
    params = PROBLEM.vectors()
    augmented = np.concatenate([state0, np.zeros(1, dtype=np.float64)])
    for index, control in enumerate(u):
        left = float(t[index])
        right = float(t[index + 1])
        control_value = float(control)

        def rhs(_time: float, value: np.ndarray) -> np.ndarray:
            state = np.maximum(value[: PROBLEM.m], 1.0e-12)
            running = float(params["beta"] @ state + PROBLEM.gamma * control_value)
            return np.concatenate(
                [
                    dynamics_numpy(state, control_value, PROBLEM, params),
                    np.asarray([running], dtype=np.float64),
                ]
            )

        solution = solve_ivp(
            rhs,
            (left, right),
            augmented,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            max_step=max((right - left) / 4.0, 1.0e-8),
        )
        if not solution.success:
            raise RuntimeError(
                f"physical rollout failed on [{left}, {right}]: {solution.message}"
            )
        augmented = np.asarray(solution.y[:, -1], dtype=np.float64)
    terminal_cost = float(params["alpha"] @ augmented[: PROBLEM.m])
    running_cost = float(augmented[-1])
    return {
        "J": terminal_cost + running_cost,
        "terminal_cost": terminal_cost,
        "running_cost": running_cost,
        "final_total_population": float(augmented[: PROBLEM.m].sum()),
        "u_min": float(np.min(u)),
        "u_max": float(np.max(u)),
        "u_mean": float(np.sum(u * np.diff(t)) / PROBLEM.T),
    }


def selected_learned_neural_pmp(root: Path) -> list[tuple[int, Path]]:
    rows = read_csv(root / "all_runs.csv")
    selected: list[tuple[int, Path]] = []
    for row in rows:
        if str(row.get("selected_within_seed", "")).strip().lower() == "true":
            selected.append((int(row["seed"]), root / row["control_checkpoint"]))
    selected.sort()
    if [seed for seed, _ in selected] != [0, 1, 2]:
        raise ValueError("learned-dynamics Neural-PMP must contain selected seeds 0,1,2")
    return selected


def time_only_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, float]]:
    specifications: list[tuple[str, str, int | None, Path]] = [
        ("direct", "Direct transcription", None, args.direct),
        ("ours", r"PMP/KKT time-only \(u_\theta(t)\)", None, args.ours),
        (
            "exact_neural_pmp",
            "Neural-PMP (exact dynamics)",
            0,
            args.exact_neural_pmp,
        ),
    ]
    for seed, path in selected_learned_neural_pmp(args.learned_neural_pmp):
        specifications.append(
            (
                "learned_neural_pmp",
            "Neural-PMP (learned dynamics)",
                seed,
                path,
            )
        )

    rows: list[dict[str, Any]] = []
    trajectory_dir = args.out_dir / "time_only_trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    for method_id, method, seed, path in specifications:
        source_t, source_u = load_schedule(path)
        common_t, common_u = common_schedule(source_t, source_u, args.intervals)
        for state_id, state in (("nominal", NOMINAL), ("resistant_heavy", RESISTANT)):
            metrics = evaluate_schedule(common_t, common_u, state)
            seed_tag = "selected" if seed is None else str(seed)
            trajectory_path = (
                trajectory_dir / f"{method_id}__seed_{seed_tag}__{state_id}.npz"
            )
            np.savez_compressed(
                trajectory_path,
                t=common_t,
                u=common_u,
                initial_state=state,
            )
            rows.append(
                {
                    "scope": "time_only",
                    "method_id": method_id,
                    "method": method,
                    "seed": "" if seed is None else seed,
                    "state_id": state_id,
                    "native_intervals": int(source_u.size),
                    "evaluation_intervals": args.intervals,
                    "control_artifact": str(trajectory_path),
                    "source_control_artifact": str(path),
                    **metrics,
                }
            )

    direct_nominal = next(
        float(row["J"])
        for row in rows
        if row["method_id"] == "direct" and row["state_id"] == "nominal"
    )
    with np.load(args.resistant_direct, allow_pickle=True) as pack:
        direct_t = np.asarray(pack["t"], dtype=np.float64).reshape(-1)
        control_key = "state_direct_control" if "state_direct_control" in pack else "u"
        direct_u = np.asarray(pack[control_key], dtype=np.float64).reshape(-1)
        direct_n0 = np.asarray(pack["N0"], dtype=np.float64).reshape(-1)
    if not np.allclose(direct_n0, RESISTANT, rtol=0.0, atol=1.0e-12):
        raise ValueError("the state-specific direct artifact is not the declared resistant-heavy state")
    if direct_t.size != args.intervals + 1 or direct_u.size != args.intervals:
        raise ValueError("resistant-heavy direct artifact is not on the common grid")
    direct_resistant_metrics = evaluate_schedule(direct_t, direct_u, RESISTANT)
    direct_resistant = float(direct_resistant_metrics["J"])
    direct_resistant_row = next(
        row
        for row in rows
        if row["method_id"] == "direct" and row["state_id"] == "resistant_heavy"
    )
    resistant_trajectory_path = (
        trajectory_dir / "direct__seed_selected__resistant_heavy.npz"
    )
    np.savez_compressed(
        resistant_trajectory_path,
        t=direct_t,
        u=direct_u,
        initial_state=RESISTANT,
    )
    direct_resistant_row.update(
        {
            "native_intervals": int(direct_u.size),
            "control_artifact": str(resistant_trajectory_path),
            "source_control_artifact": str(args.resistant_direct),
            **direct_resistant_metrics,
        }
    )
    references = {"nominal": direct_nominal, "resistant_heavy": direct_resistant}

    for row in rows:
        reference = references[str(row["state_id"])]
        gap = float(row["J"]) - reference
        row["direct_reference_J"] = reference
        row["objective_gap"] = gap
        row["relative_gap_percent"] = 100.0 * gap / reference
    return rows, references


def validate_feedback_protocol(path: Path, intervals: int) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    problem = protocol["physical_problem"]
    expected = {"alpha": 1.0, "beta": 40.0, "gamma": 8000.0}
    for key, value in expected.items():
        if not np.isclose(float(problem[key]), value, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"feedback protocol has {key}={problem[key]}, expected {value}")
    if int(protocol["execution"]["evaluation_intervals"]) != intervals:
        raise ValueError("feedback protocol does not use the declared common grid")
    return protocol


def feedback_rows(
    args: argparse.Namespace, references: dict[str, float]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validate_feedback_protocol(args.feedback_protocol, args.intervals)
    raw = read_csv(args.feedback)
    if any(int(row["evaluation_intervals"]) != args.intervals for row in raw):
        raise ValueError("feedback row found off the common execution grid")

    retained = []
    for row in raw:
        method = row["method"]
        variant = row["variant"]
        keep = (
            method == "Section-5 Transformer feedback"
            or (method == "Adaptive HJB-NN" and variant == "tau_norm_10")
            or (method == "PI-DeepONet" and variant == "final_predeclared_outer_2")
            or (
                method == "DeepBSDE log-state HJB"
                and variant == "sigma_0.025"
            )
        )
        if keep:
            retained.append(row)
    expected_counts = {
        ("Section-5 Transformer feedback", "case1"): 2,
        ("Section-5 Transformer feedback", "case2"): 2,
        ("Adaptive HJB-NN", "tau_norm_10"): 6,
        ("PI-DeepONet", "final_predeclared_outer_2"): 6,
        ("DeepBSDE log-state HJB", "sigma_0.025"): 6,
    }
    for key, count in expected_counts.items():
        found = sum((row["method"], row["variant"]) == key for row in retained)
        if found != count:
            raise ValueError(f"expected {count} feedback rows for {key}, found {found}")

    labels = {
        ("Section-5 Transformer feedback", "case1"): ("cf", "PMP/KKT-CF"),
        ("Section-5 Transformer feedback", "case2"): ("der", "PMP/KKT-DER"),
        ("Adaptive HJB-NN", "tau_norm_10"): ("hjb", "Adaptive HJB-NN"),
        ("PI-DeepONet", "final_predeclared_outer_2"): ("pi", "PI-DeepONet"),
        ("DeepBSDE log-state HJB", "sigma_0.025"): ("deepbsde", "DeepBSDE"),
    }
    per_run: list[dict[str, Any]] = []
    for row in retained:
        method_id, label = labels[(row["method"], row["variant"])]
        state_id = row["state_id"]
        reference = references[state_id]
        objective = float(row["J_unregularized"])
        gap = objective - reference
        per_run.append(
            {
                "scope": "feedback",
                "method_id": method_id,
                "method": label,
                "variant": row["variant"],
                "seed": row["seed"],
                "state_id": state_id,
                "native_intervals": int(row["native_intervals"]),
                "evaluation_intervals": int(row["evaluation_intervals"]),
                "J": objective,
                "direct_reference_J": reference,
                "objective_gap": gap,
                "relative_gap_percent": 100.0 * gap / reference,
                "query_median_ms": float(row["query_median_ms"]),
                "closed_loop_median_ms": float(row["closed_loop_median_ms"]),
                "trajectory_npz": row["trajectory_npz"],
            }
        )

    aggregate: list[dict[str, Any]] = []
    for method_id in ("cf", "der", "hjb", "pi", "deepbsde"):
        for state_id in ("nominal", "resistant_heavy"):
            group = [
                row
                for row in per_run
                if row["method_id"] == method_id and row["state_id"] == state_id
            ]
            values = np.asarray([float(row["J"]) for row in group], dtype=np.float64)
            gaps = np.asarray(
                [float(row["objective_gap"]) for row in group], dtype=np.float64
            )
            relative = np.asarray(
                [float(row["relative_gap_percent"]) for row in group], dtype=np.float64
            )
            aggregate.append(
                {
                    "method_id": method_id,
                    "method": group[0]["method"],
                    "state_id": state_id,
                    "runs": len(group),
                    "native_intervals": group[0]["native_intervals"],
                    "evaluation_intervals": args.intervals,
                    "J_mean": float(values.mean()),
                    "J_sample_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "objective_gap_mean": float(gaps.mean()),
                    "objective_gap_sample_sd": (
                        float(gaps.std(ddof=1)) if len(gaps) > 1 else 0.0
                    ),
                    "relative_gap_percent_mean": float(relative.mean()),
                    "relative_gap_percent_sample_sd": (
                        float(relative.std(ddof=1)) if len(relative) > 1 else 0.0
                    ),
                    "query_median_ms_across_runs": float(
                        np.median([float(row["query_median_ms"]) for row in group])
                    ),
                    "closed_loop_median_ms_across_runs": float(
                        np.median(
                            [float(row["closed_loop_median_ms"]) for row in group]
                        )
                    ),
                }
            )
    return per_run, aggregate


def aggregate_time_only(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for method_id in ("direct", "ours", "learned_neural_pmp", "exact_neural_pmp"):
        for state_id in ("nominal", "resistant_heavy"):
            group = [
                row
                for row in rows
                if row["method_id"] == method_id and row["state_id"] == state_id
            ]
            values = np.asarray([float(row["J"]) for row in group], dtype=np.float64)
            gaps = np.asarray(
                [float(row["objective_gap"]) for row in group], dtype=np.float64
            )
            relative = np.asarray(
                [float(row["relative_gap_percent"]) for row in group], dtype=np.float64
            )
            result.append(
                {
                    "method_id": method_id,
                    "method": group[0]["method"],
                    "state_id": state_id,
                    "runs": len(group),
                    "native_intervals": ",".join(
                        str(value)
                        for value in sorted(
                            {int(row["native_intervals"]) for row in group}
                        )
                    ),
                    "evaluation_intervals": group[0]["evaluation_intervals"],
                    "J_mean": float(values.mean()),
                    "J_sample_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "objective_gap_mean": float(gaps.mean()),
                    "objective_gap_sample_sd": (
                        float(gaps.std(ddof=1)) if len(gaps) > 1 else 0.0
                    ),
                    "relative_gap_percent_mean": float(relative.mean()),
                    "relative_gap_percent_sample_sd": (
                        float(relative.std(ddof=1)) if len(relative) > 1 else 0.0
                    ),
                }
            )
    return result


def paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.70,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.70,
            "ytick.major.width": 0.70,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def positive_errorbar(mean: float, sd: float) -> np.ndarray:
    lower = min(sd, 0.8 * mean)
    return np.asarray([[lower], [sd]], dtype=np.float64)


def plot_objective_gap(
    path: Path,
    time_summary: list[dict[str, Any]],
    feedback_summary: list[dict[str, Any]],
) -> None:
    paper_style()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.0, 2.05),
        gridspec_kw={"width_ratios": (0.82, 1.42), "wspace": 0.30},
    )

    time_axis = axes[0]
    time_ids = ("ours", "learned_neural_pmp", "exact_neural_pmp")
    time_labels = (
        "PMP/KKT\ntime-only",
        "Neural-PMP\n(learned dynamics)",
        "Neural-PMP\n(exact dynamics)",
    )
    colors = (BLUE, ORANGE, PURPLE)
    for index, (method_id, label, color) in enumerate(
        zip(time_ids, time_labels, colors)
    ):
        row = next(
            item
            for item in time_summary
            if item["method_id"] == method_id and item["state_id"] == "nominal"
        )
        mean = float(row["relative_gap_percent_mean"])
        sd = float(row["relative_gap_percent_sample_sd"])
        time_axis.scatter(
            [index],
            [mean],
            s=29,
            marker="o" if method_id != "exact_neural_pmp" else "D",
            color=color,
            edgecolor="white",
            linewidth=0.65,
            zorder=4,
        )
        if sd > 0.0:
            time_axis.errorbar(
                [index],
                [mean],
                yerr=positive_errorbar(mean, sd),
                fmt="none",
                ecolor=color,
                elinewidth=1.15,
                capsize=3.4,
                capthick=1.0,
                barsabove=True,
                zorder=6,
            )
    time_axis.set_yscale("log")
    time_axis.yaxis.set_major_locator(
        LogLocator(base=10.0, subs=(1.0,), numticks=12)
    )
    time_axis.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
    )
    time_axis.yaxis.set_minor_formatter(NullFormatter())
    time_axis.set_xticks(range(len(time_labels)), time_labels)
    time_axis.set_ylabel(
        "relative objective gap to\n"
        "direct transcription (%)"
    )
    time_axis.set_title("(a) Time-only policies: nominal state", loc="left")
    time_axis.grid(
        True, axis="y", which="major", color=GRID_GRAY, lw=0.40, alpha=0.72
    )
    feedback_axis = axes[1]
    feedback_ids = ("cf", "der", "hjb", "pi", "deepbsde")
    feedback_labels = (
        "PMP/KKT-CF",
        "PMP/KKT-DER",
        "Adaptive\nHJB-NN",
        "PI-DeepONet",
        "DeepBSDE",
    )
    x = np.arange(len(feedback_ids), dtype=np.float64)
    offsets = {"nominal": -0.11, "resistant_heavy": 0.11}
    state_style = {
        "nominal": ("o", BLUE, "Nominal"),
        "resistant_heavy": ("s", ORANGE, "Resistant-heavy"),
    }
    for state_id in ("nominal", "resistant_heavy"):
        marker, color, state_label = state_style[state_id]
        for index, method_id in enumerate(feedback_ids):
            row = next(
                item
                for item in feedback_summary
                if item["method_id"] == method_id and item["state_id"] == state_id
            )
            mean = float(row["relative_gap_percent_mean"])
            sd = float(row["relative_gap_percent_sample_sd"])
            position = x[index] + offsets[state_id]
            feedback_axis.scatter(
                [position],
                [mean],
                s=27,
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.60,
                zorder=4,
                label=state_label if index == 0 else None,
            )
            if sd > 0.0:
                feedback_axis.errorbar(
                    [position],
                    [mean],
                    yerr=positive_errorbar(mean, sd),
                    fmt="none",
                    ecolor=color,
                    elinewidth=1.10,
                    capsize=3.2,
                    capthick=0.95,
                    barsabove=True,
                    zorder=6,
                )
    feedback_axis.set_yscale("log")
    feedback_axis.yaxis.set_major_locator(
        LogLocator(base=10.0, subs=(1.0,), numticks=12)
    )
    feedback_axis.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
    )
    feedback_axis.yaxis.set_minor_formatter(NullFormatter())
    feedback_axis.set_xticks(x, feedback_labels)
    feedback_axis.set_title(
        "(b) State-time feedback policies", loc="left"
    )
    feedback_axis.grid(
        True, axis="y", which="major", color=GRID_GRAY, lw=0.40, alpha=0.72
    )
    feedback_axis.legend(frameon=False, loc="upper left", ncol=2)

    figure.subplots_adjust(left=0.08, right=0.995, top=0.90, bottom=0.27)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), dpi=260, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    parser.add_argument(
        "--feedback-protocol", type=Path, default=DEFAULT_FEEDBACK_PROTOCOL
    )
    parser.add_argument("--direct", type=Path, default=DEFAULT_DIRECT)
    parser.add_argument(
        "--resistant-direct", type=Path, default=DEFAULT_RESISTANT_DIRECT
    )
    parser.add_argument("--ours", type=Path, default=DEFAULT_OURS)
    parser.add_argument(
        "--exact-neural-pmp", type=Path, default=DEFAULT_EXACT_NEURAL_PMP
    )
    parser.add_argument(
        "--learned-neural-pmp", type=Path, default=DEFAULT_LEARNED_NEURAL_PMP
    )
    parser.add_argument("--intervals", type=int, default=800)
    args = parser.parse_args()

    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.expanduser().resolve())
    if args.intervals != 800:
        raise ValueError("the paper comparison is fixed to the common n=800 grid")
    required = [
        args.feedback,
        args.feedback_protocol,
        args.direct,
        args.resistant_direct,
        args.ours,
        args.exact_neural_pmp,
        args.learned_neural_pmp / "all_runs.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required result artifact(s): " + ", ".join(missing))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    time_rows, references = time_only_rows(args)
    time_summary = aggregate_time_only(time_rows)
    feedback_per_run, feedback_summary = feedback_rows(args, references)

    write_csv(args.out_dir / "time_only_per_run.csv", time_rows)
    write_csv(args.out_dir / "time_only_summary.csv", time_summary)
    write_csv(args.out_dir / "feedback_per_run.csv", feedback_per_run)
    write_csv(args.out_dir / "feedback_summary.csv", feedback_summary)

    figure_path = args.out_dir / "related_work_objective_gap.pdf"
    plot_objective_gap(figure_path, time_summary, feedback_summary)
    plot_objective_gap(
        args.figure_dir / "related_work_objective_gap.pdf",
        time_summary,
        feedback_summary,
    )

    payload = {
        "schema_version": 1,
        "physical_problem": PROBLEM.to_dict(),
        "training_scale": {
            "factor": 400.0,
            "equivalent_training_weights": {
                "alpha": 0.0025,
                "beta": 0.1,
                "gamma": 20.0,
            },
        },
        "common_evaluation": {
            "intervals": args.intervals,
            "control_hold": "left-endpoint zero-order hold",
            "integrator": "segmented scipy.integrate.solve_ivp DOP853",
            "rtol": 1.0e-10,
            "atol": 1.0e-12,
        },
        "direct_references": references,
        "time_only": time_summary,
        "feedback": feedback_summary,
        "interpretation": {
            "direct": "state-specific numerical reference, not a learned baseline",
            "neural_pmp": "tumor-model adaptation with learned dynamics; three fixed random seeds",
            "exact_neural_pmp": "exact-dynamics algorithmic ablation; not an original-paper result",
            "hjb": "entropy-regularized tumor-model adaptation, normalized tau=10",
            "pi_deeponet": "paper-specification tumor-model adaptation",
            "deepbsde": "viscous log-state tumor-model adaptation, sigma=0.025",
            "error_bars": "sample standard deviation over three seeds where available",
        },
        "artifacts": {
            "figure": str(figure_path),
            "time_only_per_run": str(args.out_dir / "time_only_per_run.csv"),
            "feedback_per_run": str(args.out_dir / "feedback_per_run.csv"),
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(safe(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
