#!/usr/bin/env python3
"""Build the fixed-policy robustness trajectory used as main-paper Figure 3.

The three reported controllers are evaluated without retraining or
state-specific re-optimization:

* the nominal direct-transcription schedule is replayed unchanged;
* the learned PMP-Time schedule is replayed unchanged;
* the PMP-CF feedback policy is queried along each closed-loop trajectory.

The nominal and resistant-heavy initial states are deterministic.  The
held-out set is the same fixed set of 128 radius-0.20 perturbations used by
the common Table-1 evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for search_path in (ROOT, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from evaluate_table1_heldout_common import (  # noqa: E402
    Problem,
    load_control,
    load_directions,
    load_policy,
)


METHODS = (
    ("direct", "Direct", "#555555"),
    ("time", "PMP-Time", "#0072B2"),
    ("feedback", "PMP-CF", "#009E73"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dynamics(
    state: np.ndarray,
    control: np.ndarray,
    problem: Problem,
    vectors: dict[str, np.ndarray],
) -> np.ndarray:
    safe_state = np.maximum(np.asarray(state, dtype=np.float64), 1.0e-12)
    growth = (
        vectors["r"][None, :]
        - vectors["phi"][None, :] * control[:, None]
        - vectors["M"][None, :] * np.log1p(safe_state.mean(axis=1))[:, None]
    )
    return growth * safe_state


def rollout_total_population(
    initial: np.ndarray,
    problem: Problem,
    vectors: dict[str, np.ndarray],
    *,
    fixed_control: np.ndarray | None,
    query: Callable[[int, float, np.ndarray], float] | None,
    substeps: int = 4,
) -> np.ndarray:
    """Return total-population trajectories at all 801 common-grid nodes."""

    if (fixed_control is None) == (query is None):
        raise ValueError("supply exactly one of fixed_control or query")
    state = np.asarray(initial, dtype=np.float64).copy()
    batch = state.shape[0]
    totals = np.empty((batch, problem.intervals + 1), dtype=np.float64)
    totals[:, 0] = state.sum(axis=1)
    interval_step = problem.T / problem.intervals
    step = interval_step / substeps

    for index in range(problem.intervals):
        if fixed_control is not None:
            control = np.full(batch, fixed_control[index], dtype=np.float64)
        else:
            assert query is not None
            physical_time = index * interval_step
            control = np.asarray(
                [
                    query(index, physical_time, state[sample].copy())
                    for sample in range(batch)
                ],
                dtype=np.float64,
            )
        if not np.all(np.isfinite(control)):
            raise RuntimeError(f"non-finite control at interval {index}")
        if float(np.min(control)) < -1.0e-8 or float(np.max(control)) > 3.0 + 1.0e-8:
            raise RuntimeError(f"control outside [0,3] at interval {index}")
        control = np.clip(control, 0.0, 3.0)
        for _ in range(substeps):
            k1 = dynamics(state, control, problem, vectors)
            k2 = dynamics(state + 0.5 * step * k1, control, problem, vectors)
            k3 = dynamics(state + 0.5 * step * k2, control, problem, vectors)
            k4 = dynamics(state + step * k3, control, problem, vectors)
            state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            if not np.all(np.isfinite(state)) or float(np.min(state)) <= 0.0:
                raise RuntimeError("RK4 rollout produced a nonpositive/non-finite state")
        totals[:, index + 1] = state.sum(axis=1)
    return totals


def normalized_total(
    initial: np.ndarray,
    problem: Problem,
    vectors: dict[str, np.ndarray],
    *,
    fixed_control: np.ndarray | None,
    query: Callable[[int, float, np.ndarray], float] | None,
) -> np.ndarray:
    totals = rollout_total_population(
        initial,
        problem,
        vectors,
        fixed_control=fixed_control,
        query=query,
    )
    return totals / totals[:, [0]]


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.4,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.3,
            "mathtext.fontset": "stixsans",
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_figure(
    time: np.ndarray,
    trajectories: dict[str, dict[str, np.ndarray]],
    output_pdf: Path,
    output_png: Path,
) -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(3.25, 2.60))
    ax.set_facecolor("#FAFBFC")

    for key, _label, color in METHODS:
        values = trajectories[key]
        heldout = values["heldout"]
        mean = heldout.mean(axis=0)
        q10, q90 = np.quantile(heldout, [0.10, 0.90], axis=0)
        ax.fill_between(
            time,
            q10,
            q90,
            color=color,
            alpha=0.095,
            linewidth=0.0,
            zorder=1,
        )
        ax.plot(
            time,
            mean,
            color=color,
            linewidth=1.45,
            linestyle="-",
            zorder=3,
        )
        ax.plot(
            time,
            values["resistant"][0],
            color=color,
            linewidth=1.10,
            linestyle=(0, (4.0, 2.0)),
            zorder=4,
        )
        ax.plot(
            time,
            values["nominal"][0],
            color=color,
            linewidth=1.05,
            linestyle=(0, (1.0, 1.55)),
            zorder=5,
        )

    ax.set_xlim(0.0, 10.0)
    ax.set_xticks(np.arange(0.0, 10.1, 2.0))
    all_values = np.concatenate(
        [
            state_values.reshape(-1)
            for method_values in trajectories.values()
            for state_values in method_values.values()
        ]
    )
    lower = max(0.0, float(np.min(all_values)) - 0.04)
    upper = float(np.max(all_values)) + 0.06
    ax.set_ylim(lower, upper)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(
        r"$\sum_i N_i(t)\,/\,\sum_i N_i(0)$",
        labelpad=2.0,
    )
    ax.grid(True, color="#D7DEE5", linewidth=0.55, alpha=0.72)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    method_handles = [
        Line2D([0], [0], color=color, linewidth=1.6, label=label)
        for _key, label, color in (
            ("time", "PMP-Time", "#0072B2"),
            ("feedback", "PMP-CF", "#009E73"),
            ("direct", "Direct transcription", "#555555"),
        )
    ]
    state_handles = [
        Line2D(
            [0],
            [0],
            color="#30363B",
            linewidth=1.2,
            linestyle=(0, (1.0, 1.55)),
            label="Nominal",
        ),
        Line2D(
            [0],
            [0],
            color="#30363B",
            linewidth=1.2,
            linestyle=(0, (4.0, 2.0)),
            label="Resistant-heavy",
        ),
        Line2D(
            [0],
            [0],
            color="#30363B",
            linewidth=1.4,
            linestyle="-",
            label="Held-out mean",
        ),
    ]
    method_legend = ax.legend(
        handles=method_handles,
        ncol=3,
        loc="lower left",
        bbox_to_anchor=(-0.01, 1.055),
        frameon=False,
        handlelength=2.25,
        columnspacing=1.05,
        handletextpad=0.45,
        borderaxespad=0.0,
    )
    ax.add_artist(method_legend)
    ax.legend(
        handles=state_handles,
        ncol=3,
        loc="lower left",
        bbox_to_anchor=(-0.01, 1.005),
        frameon=False,
        handlelength=2.25,
        columnspacing=0.90,
        handletextpad=0.45,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(left=0.185, right=0.985, bottom=0.165, top=0.835)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.075)
    fig.savefig(output_png, dpi=320, bbox_inches="tight", pad_inches=0.075)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--direct",
        type=Path,
        default=ROOT
        / "outputs/related_work_a1_b40_g8000_20260723/paper_results/"
        "time_only_trajectories/direct__seed_selected__nominal.npz",
    )
    parser.add_argument(
        "--time",
        type=Path,
        default=ROOT
        / "output/final_paper_recompute_20260727_clean_v2/tables/"
        "time_only_solution_n800.npz",
    )
    parser.add_argument(
        "--feedback",
        type=Path,
        default=ROOT / "tmp/table1_checkpoints_20260728/cf_final.pt",
    )
    parser.add_argument(
        "--directions",
        type=Path,
        default=ROOT
        / "output/final_paper_recompute_20260727_clean_v2/heldout/cf/"
        "directions.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output/figure3_robustness_20260728",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {
        "direct": args.direct.resolve(),
        "time": args.time.resolve(),
        "feedback": args.feedback.resolve(),
        "directions": args.directions.resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    problem = Problem()
    vectors = problem.vectors()
    directions = load_directions(paths["directions"], 128, problem.m)
    initial_states = {
        "nominal": np.full((1, problem.m), 10.0, dtype=np.float64),
        "resistant": np.arange(9.0, 11.0 + 0.05, 0.1, dtype=np.float64)[None, :],
        "heldout": 10.0 * (1.0 + 0.20 * directions),
    }
    if initial_states["resistant"].shape != (1, problem.m):
        raise RuntimeError("resistant-heavy initial state does not have 21 components")

    direct_control = load_control(paths["direct"], problem)
    time_control = load_control(paths["time"], problem)
    feedback_query, feedback_close, feedback_metadata = load_policy(
        "pmp_kkt_cf",
        paths["feedback"],
        hjb_tau=10.0,
    )

    controllers: dict[
        str,
        tuple[np.ndarray | None, Callable[[int, float, np.ndarray], float] | None],
    ] = {
        "direct": (direct_control, None),
        "time": (time_control, None),
        "feedback": (None, feedback_query),
    }
    trajectories: dict[str, dict[str, np.ndarray]] = {}
    try:
        for method, (fixed_control, query) in controllers.items():
            trajectories[method] = {}
            for state_name, initial in initial_states.items():
                trajectories[method][state_name] = normalized_total(
                    initial,
                    problem,
                    vectors,
                    fixed_control=fixed_control,
                    query=query,
                )
    finally:
        feedback_close()

    time = np.linspace(0.0, problem.T, problem.intervals + 1)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "robustness_trajectories.npz",
        time=time,
        **{
            f"{method}_{state}": values
            for method, method_values in trajectories.items()
            for state, values in method_values.items()
        },
    )
    plot_figure(
        time,
        trajectories,
        output_dir / "robustness_normalized_population.pdf",
        output_dir / "robustness_normalized_population.png",
    )

    summary: dict[str, Any] = {
        "schema": "figure3-robustness-v1",
        "problem": {
            "T": problem.T,
            "intervals": problem.intervals,
            "m": problem.m,
            "objective_weights": [problem.alpha, problem.beta, problem.gamma],
        },
        "evaluation": {
            "policy_update": "none",
            "direct_semantics": "fixed nominal schedule replayed unchanged",
            "time_semantics": "fixed learned schedule replayed unchanged",
            "feedback_semantics": "closed-loop PMP-CF query at every left endpoint",
            "heldout_samples": 128,
            "heldout_radius": 0.20,
            "heldout_summary": "mean and pointwise 10th--90th percentiles",
            "normalization": "each trajectory divided by its own initial total population",
        },
        "sources": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "feedback_metadata": feedback_metadata,
        "terminal_normalized_population": {
            method: {
                state: {
                    "mean": float(values[:, -1].mean()),
                    "q10": float(np.quantile(values[:, -1], 0.10)),
                    "q90": float(np.quantile(values[:, -1], 0.90)),
                }
                for state, values in method_values.items()
            }
            for method, method_values in trajectories.items()
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
