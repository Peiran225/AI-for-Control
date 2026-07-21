#!/usr/bin/env python3
"""Create report figures from the canonical result manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tumor_problem import NOMINAL_TUMOR_PROBLEM, evaluate_zoh_control  # noqa: E402


RESULTS = ROOT / "paper_runs" / "canonical_results"
FIGURES = RESULTS / "figures"
REFINED_TIME_ONLY = (
    ROOT
    / "outputs"
    / "time_only_singular_plateau_refinement_20260720"
    / "base_then_rk4_full_gradient_sm0_500"
    / "solution.npz"
)

BLUE = "#2F6690"
ORANGE = "#D97706"
GREEN = "#2A7F62"
RED = "#B4473A"
GRAY = "#5B6573"


def setup() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 300,
        }
    )


def selected_solution(manifest: dict, family: str) -> tuple[Path, dict, dict]:
    record = manifest["selected"][family]
    source = ROOT / record["source"]
    data = np.load(source)
    result = evaluate_zoh_control(data["t"], data["u"], diagnostic_points=4001)
    # Retain the exported grid for diagnostics that distinguish interval interiors
    # from the jump points of the right-continuous, left-endpoint-held ZOH control.
    result["control_breakpoints"] = np.asarray(data["t"], dtype=np.float64)
    return source, record, result


def evaluated_solution(path: Path) -> dict:
    """Load one saved ZOH schedule and evaluate it on the common dense grid."""
    data = np.load(path)
    result = evaluate_zoh_control(data["t"], data["u"], diagnostic_points=4001)
    result["control_breakpoints"] = np.asarray(data["t"], dtype=np.float64)
    return result


def control_transition_centers(path: Path) -> tuple[float, float]:
    """Return the largest downward/upward control-jump breakpoints."""
    data = np.load(path)
    time = np.asarray(data["t"], dtype=np.float64)
    control = np.asarray(data["u"], dtype=np.float64)
    jumps = np.diff(control)
    early_candidates = np.flatnonzero(time[:-1] < 2.0)
    late_candidates = np.flatnonzero(time[:-1] >= 7.0)
    early_index = early_candidates[np.argmin(jumps[early_candidates])]
    late_index = late_candidates[np.argmax(jumps[late_candidates])]
    return float(time[early_index + 1]), float(time[late_index + 1])


def refined_trajectory_figure() -> None:
    """Trajectory view for the baseline-curriculum/full-gradient refinement."""
    result = evaluated_solution(REFINED_TIME_ONLY)
    time = result["diagnostic_t"]
    control = result["diagnostic_u"]
    population = result["diagnostic_N"]
    singular_control = result["diagnostic_u_singular"]
    early, late = control_transition_centers(REFINED_TIME_ONLY)

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(7.1, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.5, 1.0]},
    )
    axes[0].step(
        time,
        control,
        where="post",
        color=BLUE,
        lw=1.8,
        label=r"refined Transformer control $u_\theta(t)$",
    )
    axes[0].plot(
        time,
        singular_control,
        color=ORANGE,
        lw=1.35,
        ls="--",
        label="state-only singular-control candidate",
    )
    axes[0].set_ylabel(r"control $u$")
    axes[0].set_ylim(-0.08, 3.08)
    axes[0].legend(frameon=False, ncol=2, loc="best")

    colors = plt.cm.viridis(np.linspace(0.08, 0.92, population.shape[1]))
    for index, color in enumerate(colors):
        axes[1].plot(time, population[:, index], color=color, lw=0.9, alpha=0.9)
    axes[1].set_ylabel(r"subpopulation $N_i$")

    total = population.sum(axis=1)
    plateau = (time >= 1.0) & (time <= 8.5)
    plateau_range = float(np.ptp(total[plateau]))
    plateau_relative = 100.0 * plateau_range / float(np.mean(total[plateau]))
    axes[2].axvspan(1.0, 8.5, color="#F3F4F6", alpha=0.8, zorder=0)
    axes[2].plot(time, total, color=GREEN, lw=2.0)
    axes[2].text(
        0.50,
        0.85,
        rf"$t\in[1,8.5]$: range $={plateau_range:.3f}$ ({plateau_relative:.3f}\%)",
        transform=axes[2].transAxes,
        ha="center",
        va="top",
        color=GRAY,
        fontsize=8,
    )
    axes[2].set_ylabel(r"total population $\sum_i N_i$")
    axes[2].set_xlabel(r"time $t$")

    for axis in axes:
        axis.axvline(early, color=ORANGE, ls=":", lw=0.9, alpha=0.9)
        axis.axvline(late, color=RED, ls=":", lw=0.9, alpha=0.9)
        axis.grid(True, color="#E5E7EB", lw=0.6)
    axes[0].text(early + 0.08, 2.45, rf"$t={early:.2f}$", color=ORANGE, fontsize=7.5)
    axes[0].text(late - 0.08, 2.45, rf"$t={late:.2f}$", color=RED, fontsize=7.5, ha="right")
    figure.tight_layout()
    figure.savefig(FIGURES / "refined_time_only_trajectory.png", bbox_inches="tight")
    plt.close(figure)


def refined_phenotype_transition_figure() -> None:
    """Show N(t,x) and fixed-time slices at the rapid control transitions."""
    result = evaluated_solution(REFINED_TIME_ONLY)
    time = np.asarray(result["diagnostic_t"], dtype=np.float64)
    population = np.asarray(result["diagnostic_N"], dtype=np.float64)
    trait = np.linspace(0.0, 1.0, population.shape[1])
    early, late = control_transition_centers(REFINED_TIME_ONLY)
    slice_times = (early, 2.0, 8.5, late)
    slice_rows = [int(np.argmin(np.abs(time - value))) for value in slice_times]

    figure = plt.figure(figsize=(10.4, 5.7))
    grid = figure.add_gridspec(
        2,
        3,
        width_ratios=(1.42, 1.0, 1.0),
        wspace=0.28,
        hspace=0.34,
    )
    surface_axis = figure.add_subplot(grid[:, 0], projection="3d")
    surface_rows = np.arange(0, time.size, 40, dtype=int)
    if surface_rows[-1] != time.size - 1:
        surface_rows = np.append(surface_rows, time.size - 1)
    trait_mesh, time_mesh = np.meshgrid(trait, time[surface_rows])
    surface_axis.plot_surface(
        trait_mesh,
        time_mesh,
        population[surface_rows],
        cmap="viridis",
        linewidth=0.0,
        antialiased=True,
        alpha=0.92,
    )
    for value, color in ((early, ORANGE), (late, RED)):
        row = int(np.argmin(np.abs(time - value)))
        surface_axis.plot(
            trait,
            np.full_like(trait, time[row]),
            population[row],
            color=color,
            lw=2.5,
        )
    surface_axis.set_title(r"Full phenotype evolution $N(t,x)$", pad=8)
    surface_axis.set_xlabel(r"phenotype $x$", labelpad=5)
    surface_axis.set_ylabel(r"time $t$", labelpad=5)
    # The adjacent slice panels already carry the N(t,x) label; omitting a
    # second vertical label keeps the compact report layout legible.
    surface_axis.view_init(elev=27, azim=-54)

    slice_axes = [
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[0, 2]),
        figure.add_subplot(grid[1, 1]),
        figure.add_subplot(grid[1, 2]),
    ]
    shared_upper = 1.08 * float(np.max(population[slice_rows]))
    for axis, target, row in zip(slice_axes, slice_times, slice_rows):
        is_transition = target in (early, late)
        color = ORANGE if target == early else RED if target == late else BLUE
        axis.plot(
            trait,
            population[row],
            color=color,
            marker="o",
            ms=2.6,
            lw=1.55,
        )
        label = "control transition" if is_transition else "comparison time"
        title_kwargs = {"color": color} if is_transition else {}
        axis.set_title(rf"$t={time[row]:.2f}$ ({label})", **title_kwargs)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, shared_upper)
        axis.grid(True, color="#E5E7EB", lw=0.55)
        peak = int(np.argmax(population[row]))
        axis.text(
            0.04,
            0.91,
            rf"peak $x={trait[peak]:.2f}$",
            transform=axis.transAxes,
            color=GRAY,
            fontsize=7.5,
        )
    for axis in slice_axes[2:]:
        axis.set_xlabel(r"phenotype $x$")
    for axis in (slice_axes[0], slice_axes[2]):
        axis.set_ylabel(r"$N(t,x)$")
    for axis in (slice_axes[1], slice_axes[3]):
        axis.tick_params(labelleft=False)
    figure.suptitle(
        "Phenotype evolution with slices through the two rapid control transitions",
        fontsize=11,
        y=0.995,
    )
    figure.subplots_adjust(
        left=0.035, right=0.985, bottom=0.09, top=0.88, wspace=0.30, hspace=0.38
    )
    figure.savefig(
        FIGURES / "refined_time_only_phenotype_transition_slices.png",
        bbox_inches="tight",
    )
    plt.close(figure)


def trajectory_figure(manifest: dict) -> None:
    _, _, result = selected_solution(manifest, "PMP/KKT Transformer")
    t = result["diagnostic_t"]
    u = result["diagnostic_u"]
    N = result["diagnostic_N"]
    u_sing = result["diagnostic_u_singular"]

    fig, axes = plt.subplots(3, 1, figsize=(7.1, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.0, 1.5, 1.0]})
    axes[0].plot(t, u, color=BLUE, lw=2.0, label=r"Transformer control $u_\theta(t)$")
    axes[0].plot(
        t,
        u_sing,
        color=ORANGE,
        lw=1.5,
        ls="--",
        label="state-only singular-control candidate",
    )
    axes[0].set_ylabel(r"control $u$")
    axes[0].set_ylim(-0.08, 3.08)
    axes[0].legend(frameon=False, ncol=2, loc="best")

    colors = plt.cm.viridis(np.linspace(0.08, 0.92, N.shape[1]))
    for index, color in enumerate(colors, start=1):
        axes[1].plot(t, N[:, index - 1], color=color, lw=0.9, alpha=0.9)
    axes[1].set_ylabel(r"subpopulation $N_i$")

    axes[2].plot(t, N.sum(axis=1), color=GREEN, lw=2.0)
    axes[2].set_ylabel(r"total population $\sum_i N_i$")
    axes[2].set_xlabel(r"time $t$")
    for ax in axes:
        ax.grid(True, color="#E5E7EB", lw=0.6)
    fig.tight_layout()
    fig.savefig(FIGURES / "selected_pmp_trajectory.png", bbox_inches="tight")
    plt.close(fig)


def phase_figure(manifest: dict) -> None:
    _, _, result = selected_solution(manifest, "PMP/KKT Transformer")
    u = result["diagnostic_u"]
    total = result["diagnostic_N"].sum(axis=1)
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.plot(u, total, color=BLUE, lw=1.8)
    ax.scatter([u[0]], [total[0]], color=GREEN, s=28, zorder=3, label="initial point")
    ax.scatter([u[-1]], [total[-1]], color=RED, s=28, zorder=3, label="terminal point")
    ax.set_xlabel(r"control $u$")
    ax.set_ylabel(r"total population $\sum_{i=1}^{21} N_i$")
    ax.grid(True, color="#E5E7EB", lw=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "selected_pmp_N_u_phase.png", bbox_inches="tight")
    plt.close(fig)


def phenotype_snapshot_figure(manifest: dict) -> None:
    """Plot N(t, x) across the 21 phenotype nodes at fixed times."""
    _, _, result = selected_solution(manifest, "PMP/KKT Transformer")
    t = result["diagnostic_t"]
    N = result["diagnostic_N"]
    x = np.linspace(0.0, 1.0, N.shape[1])
    snapshot_times = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0)
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(snapshot_times)))

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    for target, color in zip(snapshot_times, colors):
        index = int(np.argmin(np.abs(t - target)))
        state = N[index]
        label = rf"$t={t[index]:.0f}$"
        axes[0].plot(x, state, marker="o", ms=2.6, lw=1.45, color=color, label=label)
        axes[1].plot(
            x,
            100.0 * state / state.sum(),
            marker="o",
            ms=2.6,
            lw=1.45,
            color=color,
            label=label,
        )

    axes[0].set_title("(a) Absolute subpopulation profile")
    axes[0].set_xlabel(r"phenotypic trait $x$")
    axes[0].set_ylabel(r"population $N(t,x)$")
    axes[1].set_title("(b) Composition of total population")
    axes[1].set_xlabel(r"phenotypic trait $x$")
    axes[1].set_ylabel("share of total population (%)")
    axes[1].legend(frameon=False, ncol=2, loc="best")
    for ax in axes:
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, color="#E5E7EB", lw=0.6)
    fig.tight_layout()
    fig.savefig(FIGURES / "selected_pmp_phenotype_snapshots.png", bbox_inches="tight")
    plt.close(fig)


def phenotype_plot_options(manifest: dict) -> None:
    """Generate literal and complementary versions of the requested N-of-x view."""
    _, _, result = selected_solution(manifest, "PMP/KKT Transformer")
    diagnostic_t = result["diagnostic_t"]
    population = result["diagnostic_N"]
    phenotype_index = np.arange(1, population.shape[1] + 1, dtype=int)
    normalized_trait = (phenotype_index - 1) / (population.shape[1] - 1)
    snapshot_times = np.asarray((0.0, 2.0, 4.0, 6.0, 8.0, 10.0))
    snapshot_rows = np.asarray(
        [int(np.argmin(np.abs(diagnostic_t - target))) for target in snapshot_times],
        dtype=int,
    )
    realized_times = diagnostic_t[snapshot_rows]
    snapshot_population = population[snapshot_rows]
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, snapshot_times.size))
    index_ticks = (1, 5, 9, 13, 17, 21)

    # Option A: the most literal reading of the advisor's x=1,...,m request.
    figure, axis = plt.subplots(figsize=(7.4, 4.35))
    for time_value, state, color in zip(realized_times, snapshot_population, colors):
        axis.plot(
            phenotype_index,
            state,
            color=color,
            marker="o",
            markersize=3.2,
            linewidth=1.65,
            label=rf"$t={time_value:.0f}$",
        )
    axis.set_xlabel(r"phenotype / subpopulation index $i$ ($1,\ldots,21$)")
    axis.set_ylabel(r"population $N_i(t)$")
    axis.set_xticks(index_ticks)
    axis.set_xlim(1, population.shape[1])
    axis.grid(True, color="#E5E7EB", linewidth=0.6)
    axis.legend(frameon=False, ncol=2, loc="best")
    figure.tight_layout()
    figure.savefig(FIGURES / "phenotype_teacher_literal_overlay.png", bbox_inches="tight")
    plt.close(figure)

    # Option B: one fixed-time N-of-x curve per panel, avoiding overlap.
    figure, axes = plt.subplots(2, 3, figsize=(9.3, 5.45), sharex=True, sharey=True)
    common_upper = 1.08 * float(np.max(snapshot_population))
    for axis, time_value, state, color in zip(
        axes.flat, realized_times, snapshot_population, colors
    ):
        axis.plot(
            phenotype_index,
            state,
            color=color,
            marker="o",
            markersize=2.8,
            linewidth=1.55,
        )
        axis.set_title(rf"$t={time_value:.0f}$")
        axis.set_xticks(index_ticks)
        axis.set_xlim(1, population.shape[1])
        axis.set_ylim(0.0, common_upper)
        axis.grid(True, color="#E5E7EB", linewidth=0.55)
    for axis in axes[-1, :]:
        axis.set_xlabel(r"phenotype index $i$")
    for axis in axes[:, 0]:
        axis.set_ylabel(r"$N_i(t)$")
    figure.tight_layout()
    figure.savefig(FIGURES / "phenotype_teacher_small_multiples.png", bbox_inches="tight")
    plt.close(figure)

    # Option C: the same six fixed-time profiles as a discrete heatmap.
    figure, axis = plt.subplots(figsize=(7.4, 3.7))
    image = axis.imshow(
        snapshot_population,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=(0.5, population.shape[1] + 0.5, snapshot_times.size - 0.5, -0.5),
        cmap="viridis",
    )
    axis.set_xlabel(r"phenotype / subpopulation index $i$")
    axis.set_ylabel(r"fixed time $t$")
    axis.set_xticks(index_ticks)
    axis.set_yticks(np.arange(snapshot_times.size), labels=[f"{value:g}" for value in realized_times])
    peak_indices = 1 + np.argmax(snapshot_population[1:], axis=1)
    axis.scatter(
        peak_indices,
        np.arange(1, snapshot_times.size),
        s=32,
        facecolors="none",
        edgecolors="white",
        linewidths=0.9,
    )
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label(r"population $N_i(t)$")
    figure.tight_layout()
    figure.savefig(FIGURES / "phenotype_fixed_time_index_heatmap.png", bbox_inches="tight")
    plt.close(figure)

    # Option D: a continuous-time companion view; this supplements rather than
    # replaces the fixed-time curves requested by the advisor.
    figure, axis = plt.subplots(figsize=(7.6, 4.25))
    image = axis.pcolormesh(
        diagnostic_t,
        phenotype_index,
        population.T,
        shading="auto",
        cmap="viridis",
    )
    for time_value in snapshot_times[1:-1]:
        axis.axvline(time_value, color="white", linewidth=0.65, alpha=0.55)
    axis.set_xlabel(r"time $t$")
    axis.set_ylabel(r"phenotype / subpopulation index $i$")
    axis.set_yticks(index_ticks)
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label(r"population $N_i(t)$")
    figure.tight_layout()
    figure.savefig(FIGURES / "phenotype_time_index_heatmap.png", bbox_inches="tight")
    plt.close(figure)

    records = []
    for time_value, state in zip(realized_times, snapshot_population):
        for index, trait, value in zip(phenotype_index, normalized_trait, state):
            records.append(
                {
                    "time": float(time_value),
                    "phenotype_index": int(index),
                    "normalized_trait": float(trait),
                    "population": float(value),
                }
            )
    pd.DataFrame.from_records(records).to_csv(
        RESULTS / "structure_diagnostics" / "phenotype_fixed_time_profiles.csv",
        index=False,
    )


def training_figure(manifest: dict) -> None:
    selected_seed = int(manifest["selected"]["PMP/KKT Transformer"]["seed"])
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    root = RESULTS / "training_histories"
    for history_path in sorted(root.glob("pmp_kkt_seed_*.csv")):
        seed = int(history_path.stem.split("_")[-1])
        history = pd.read_csv(history_path)
        best = np.minimum.accumulate(history["loss"].to_numpy())
        selected = seed == selected_seed
        ax.plot(
            history["epoch"],
            best,
            lw=2.2 if selected else 1.1,
            alpha=1.0 if selected else 0.62,
            label=f"seed {seed}" + (" (reported trajectory)" if selected else ""),
        )
    ax.set_yscale("log")
    ax.set_xlabel("training epoch")
    ax.set_ylabel("minimum regularized training loss")
    ax.grid(True, which="both", color="#E5E7EB", lw=0.6)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "pmp_training_trajectories.png", bbox_inches="tight")
    plt.close(fig)


def component_training_figure(manifest: dict) -> None:
    selected = manifest["selected"]["PMP/KKT Transformer"]
    selected_seed = int(selected["seed"])
    smoothness_weight = float(selected["smoothness_weight"])
    history_path = RESULTS / "training_histories" / f"pmp_kkt_seed_{selected_seed}.csv"
    if not history_path.exists():
        return
    history = pd.read_csv(history_path)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    curves = [
        (history["loss"].to_numpy(dtype=float), "regularized training loss", BLUE, 2.0),
        (
            history["singular_component"].to_numpy(dtype=float),
            "singular-condition contribution",
            ORANGE,
            1.5,
        ),
        (
            history["nonsingular_component"].to_numpy(dtype=float),
            "boundary-KKT contribution",
            GREEN,
            1.5,
        ),
        (
            smoothness_weight * history["smooth"].to_numpy(dtype=float),
            "weighted smoothness contribution",
            "#7C3AED",
            1.4,
        ),
    ]
    for raw_values, label, color, width in curves:
        values = np.maximum(raw_values, 1e-12)
        smoothed = pd.Series(values).rolling(window=15, min_periods=1).median().to_numpy()
        ax.plot(history["epoch"], values, color=color, lw=0.45, alpha=0.16)
        ax.plot(history["epoch"], smoothed, color=color, lw=width, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("training epoch")
    ax.set_ylabel("training-loss contribution")
    ax.grid(True, which="both", color="#E5E7EB", lw=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "pmp_training_components.png", bbox_inches="tight")
    plt.close(fig)


def optimality_figure(manifest: dict) -> None:
    _, _, result = selected_solution(manifest, "PMP/KKT Transformer")
    t = result["diagnostic_t"]
    u = result["diagnostic_u"]
    u_sing = result["diagnostic_u_singular"]
    psi_relative = result["diagnostic_psi"] / 20.0
    projected = np.abs(u / 3.0 - np.clip(u / 3.0 - psi_relative, 0.0, 1.0))
    singular_error = np.abs(u - u_sing) / 3.0
    weight = result["diagnostic_singular_weight"]

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.2), sharex=True)
    axes[0].plot(t, psi_relative, color=GRAY, lw=1.5, label=r"normalized switching function $\psi/\gamma$")
    axes[0].axhline(0.0, color="#111827", lw=0.8)
    axes[0].set_ylabel(r"$\psi/\gamma$")
    axes[0].legend(frameon=False)

    axes[1].plot(t, projected, color=BLUE, lw=1.5, label="projected KKT residual")
    axes[1].plot(t, weight * singular_error, color=ORANGE, lw=1.4, label="weighted singular-control residual")
    axes[1].set_xlabel(r"time $t$")
    axes[1].set_ylabel("dimensionless residual")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(True, color="#E5E7EB", lw=0.6)
    fig.tight_layout()
    fig.savefig(FIGURES / "selected_pmp_optimality_conditions.png", bbox_inches="tight")
    plt.close(fig)


def refined_optimality_figure() -> None:
    result = evaluated_solution(REFINED_TIME_ONLY)
    time = result["diagnostic_t"]
    control = result["diagnostic_u"]
    singular_control = result["diagnostic_u_singular"]
    psi_relative = result["diagnostic_psi"] / NOMINAL_TUMOR_PROBLEM.gamma
    projected = np.abs(
        control / NOMINAL_TUMOR_PROBLEM.umax
        - np.clip(
            control / NOMINAL_TUMOR_PROBLEM.umax - psi_relative,
            0.0,
            1.0,
        )
    )
    singular_error = np.abs(control - singular_control) / NOMINAL_TUMOR_PROBLEM.umax
    weight = result["diagnostic_singular_weight"]

    figure, axes = plt.subplots(2, 1, figsize=(7.0, 5.2), sharex=True)
    axes[0].plot(
        time,
        psi_relative,
        color=GRAY,
        lw=1.5,
        label=r"normalized switching function $\psi/\gamma$",
    )
    axes[0].axhspan(-0.005, 0.005, color="#D1FAE5", alpha=0.7)
    axes[0].axhline(0.0, color="#111827", lw=0.8)
    axes[0].set_ylabel(r"$\psi/\gamma$")
    axes[0].legend(frameon=False)

    axes[1].plot(
        time, projected, color=BLUE, lw=1.5, label="projected KKT residual"
    )
    axes[1].plot(
        time,
        weight * singular_error,
        color=ORANGE,
        lw=1.4,
        label="weighted singular-control residual",
    )
    axes[1].set_xlabel(r"time $t$")
    axes[1].set_ylabel("dimensionless residual")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(True, color="#E5E7EB", lw=0.6)
    figure.tight_layout()
    figure.savefig(
        FIGURES / "refined_time_only_optimality_conditions.png",
        bbox_inches="tight",
    )
    plt.close(figure)


def comparison_figure() -> None:
    summary = pd.read_csv(RESULTS / "main_method_summary.csv").set_index("family")
    order = ["direct time-mesh", "direct-J Transformer", "Neural-PMP [3]", "PMP/KKT Transformer", "constant control"]
    labels = [
        "Direct time mesh\n($n=800$)",
        "Direct-$J$\nTransformer",
        "Known-dynamics\nNeural-PMP",
        "PMP/KKT\nTransformer",
        "Constant control\n$u=1.5$",
    ]
    values = np.array([summary.loc[item, "difference_from_lowest_mean"] for item in order])
    errors = np.array([summary.loc[item, "J_std"] for item in order])
    colors = [GREEN, ORANGE, RED, BLUE, GRAY]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), gridspec_kw={"width_ratios": [1.0, 1.35]})
    axes[0].bar(labels, values, yerr=errors, color=colors, capsize=3)
    axes[0].set_title("(a) All methods")
    axes[0].set_ylabel(r"$J-J_{\rm ref}$")
    axes[0].grid(True, axis="y", color="#E5E7EB", lw=0.6)
    axes[0].tick_params(axis="x", labelsize=7.5)

    near_order = order[:-1]
    near_labels = labels[:-1]
    near_values = values[:-1]
    near_errors = errors[:-1]
    axes[1].bar(near_labels, near_values, yerr=near_errors, color=colors[:-1], capsize=3)
    axes[1].set_title("(b) Near-reference methods")
    axes[1].set_ylabel(r"$J-J_{\rm ref}$")
    axes[1].grid(True, axis="y", color="#E5E7EB", lw=0.6)
    axes[1].tick_params(axis="x", labelsize=7.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "main_method_objective_comparison.png", bbox_inches="tight")
    plt.close(fig)


def refined_comparison_figure() -> None:
    """Add the single refined checkpoint to the common-objective comparison."""
    summary = pd.read_csv(RESULTS / "main_method_summary.csv").set_index("family")
    order = [
        "direct time-mesh",
        "direct-J Transformer",
        "Neural-PMP [3]",
        "PMP/KKT Transformer",
        "refined time-only",
        "constant control",
    ]
    labels = [
        "Direct time mesh\n($n=800$)",
        "Direct-$J$\nTransformer",
        "Known-dynamics\nNeural-PMP",
        "Base PMP/KKT\nTransformer",
        "Refined time-only\nTransformer",
        "Constant control\n$u=1.5$",
    ]
    reference = float(summary.loc["direct time-mesh", "J_mean"])
    values = []
    errors = []
    for item in order:
        if item == "refined time-only":
            refined = evaluated_solution(REFINED_TIME_ONLY)
            values.append(float(refined["J"]) - reference)
            errors.append(0.0)
        else:
            values.append(float(summary.loc[item, "difference_from_lowest_mean"]))
            errors.append(float(summary.loc[item, "J_std"]))
    values_array = np.asarray(values)
    errors_array = np.asarray(errors)
    colors = [GREEN, ORANGE, RED, BLUE, "#7C3AED", GRAY]

    figure, axes = plt.subplots(
        1, 2, figsize=(10.8, 4.0), gridspec_kw={"width_ratios": [1.0, 1.45]}
    )
    axes[0].bar(labels, values_array, yerr=errors_array, color=colors, capsize=3)
    axes[0].set_title("(a) All methods")
    axes[0].set_ylabel(r"$J-J_{\rm ref}$")
    axes[0].grid(True, axis="y", color="#E5E7EB", lw=0.6)
    axes[0].tick_params(axis="x", labelsize=6.4, labelrotation=24)
    for label in axes[0].get_xticklabels():
        label.set_horizontalalignment("right")

    near = slice(0, -1)
    axes[1].bar(
        labels[near],
        values_array[near],
        yerr=errors_array[near],
        color=colors[near],
        capsize=3,
    )
    axes[1].set_title("(b) Near-reference methods")
    axes[1].set_ylabel(r"$J-J_{\rm ref}$")
    axes[1].grid(True, axis="y", color="#E5E7EB", lw=0.6)
    axes[1].tick_params(axis="x", labelsize=7.2)
    figure.tight_layout()
    figure.savefig(
        FIGURES / "refined_method_objective_comparison.png", bbox_inches="tight"
    )
    plt.close(figure)


def mesh_figure() -> None:
    runs = pd.read_csv(RESULTS / "main_method_runs.csv")
    mesh = runs[runs["family"] == "direct time-mesh"].copy()
    mesh["n"] = mesh["method"].str.extract(r"n=(\d+)")[0].astype(int)
    mesh = mesh.sort_values("n")
    reference = mesh["J"].min()
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(mesh["n"], mesh["J"] - reference, marker="o", color=GREEN, lw=1.8)
    ax.set_xlabel("number of control intervals n")
    ax.set_ylabel(r"$J_n-J_{800}$")
    ax.set_xticks(mesh["n"])
    ax.grid(True, color="#E5E7EB", lw=0.6)
    fig.tight_layout()
    fig.savefig(FIGURES / "direct_mesh_convergence.png", bbox_inches="tight")
    plt.close(fig)


def singular_figure() -> None:
    data = pd.read_csv(RESULTS / "structure_diagnostics" / "singular_diagnostics.csv")
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.1), sharex=True)
    axes[0].plot(data["t"], data["u"], color=GREEN, lw=1.8, label="direct control")
    axes[0].plot(
        data["t"],
        data["u_singular_state"],
        color=ORANGE,
        ls="--",
        lw=1.5,
        label="state-only singular-control candidate",
    )
    axes[0].set_ylabel(r"control $u$")
    axes[0].legend(frameon=False)
    axes[1].plot(data["t"], data["psi_relative"], color=GRAY, lw=1.4, label=r"$\psi/\gamma$")
    axes[1].axhline(0.0, color="#111827", lw=0.8)
    axes[1].set_ylabel(r"$\psi/\gamma$")
    axes[1].set_xlabel(r"time $t$")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(True, color="#E5E7EB", lw=0.6)
    fig.tight_layout()
    fig.savefig(FIGURES / "direct_singular_structure.png", bbox_inches="tight")
    plt.close(fig)


def strict_singular_quantities(result: dict) -> dict[str, np.ndarray]:
    """Evaluate the analytic order-one singular conditions on one trajectory."""
    problem = NOMINAL_TUMOR_PROBLEM
    parameters = problem.vectors()
    N = np.asarray(result["diagnostic_N"], dtype=np.float64)
    costate = np.asarray(result["diagnostic_lambda"], dtype=np.float64)
    control = np.asarray(result["diagnostic_u"], dtype=np.float64)

    G = np.log1p(N.mean(axis=1))
    denominator = problem.m + N.sum(axis=1)
    coupled_costate = (parameters["M"][None, :] * costate * N).sum(axis=1)
    density_ratio = coupled_costate / denominator
    phi_N = (parameters["phi"][None, :] * N).sum(axis=1)

    psi = problem.gamma - (parameters["phi"][None, :] * costate * N).sum(axis=1)
    dot_psi = (
        (parameters["phi"] * parameters["beta"] * N).sum(axis=1)
        - density_ratio * phi_N
    )

    drift_without_control = (
        parameters["r"][None, :] - parameters["M"][None, :] * G[:, None]
    )
    weighted_beta_dot = (
        parameters["phi"]
        * parameters["beta"]
        * drift_without_control
        * N
    ).sum(axis=1)
    phi_dot = (parameters["phi"] * drift_without_control * N).sum(axis=1)
    coupled_costate_dot = (
        parameters["M"] * N * (-parameters["beta"] + density_ratio[:, None])
    ).sum(axis=1)
    denominator_dot = (drift_without_control * N).sum(axis=1)
    density_ratio_dot = (
        coupled_costate_dot / denominator
        - density_ratio * denominator_dot / denominator
    )
    A = weighted_beta_dot - density_ratio_dot * phi_N - density_ratio * phi_dot
    B = (
        -(parameters["phi"] ** 2 * parameters["beta"] * N).sum(axis=1)
        + density_ratio * (parameters["phi"] ** 2 * N).sum(axis=1)
        - density_ratio * phi_N**2 / denominator
    )
    strict_control = np.divide(
        -A,
        B,
        out=np.full_like(A, np.nan),
        where=np.abs(B) > 1.0e-12,
    )
    ddot_psi = A + B * control
    return {
        "psi": psi,
        "dot_psi": dot_psi,
        "ddot_psi": ddot_psi,
        "A": A,
        "B": B,
        "strict_control": strict_control,
    }


def strict_singular_summary(result: dict, quantities: dict[str, np.ndarray]) -> dict:
    problem = NOMINAL_TUMOR_PROBLEM
    tolerance = 0.005
    psi_relative = quantities["psi"] / problem.gamma
    near = np.abs(psi_relative) <= tolerance
    strict_control = quantities["strict_control"]
    admissible = (
        np.isfinite(strict_control)
        & (strict_control >= 0.0)
        & (strict_control <= problem.umax)
    )
    interior = (result["diagnostic_u"] > 0.0) & (result["diagnostic_u"] < problem.umax)
    screening_mask = near & admissible & interior

    diagnostic_t = np.asarray(result["diagnostic_t"], dtype=np.float64)
    breakpoint_mask = np.zeros(diagnostic_t.size, dtype=bool)
    for breakpoint in np.asarray(result.get("control_breakpoints", []), dtype=np.float64)[1:-1]:
        position = int(np.searchsorted(diagnostic_t, breakpoint))
        for candidate in (position - 1, position):
            if 0 <= candidate < diagnostic_t.size and np.isclose(
                diagnostic_t[candidate], breakpoint, rtol=0.0, atol=1.0e-12
            ):
                breakpoint_mask[candidate] = True
    interval_interior = ~breakpoint_mask

    summary = {
        "near_singular_fraction": float(np.mean(near)),
        "minimum_abs_psi_relative": float(np.min(np.abs(psi_relative))),
        "near_singular_point_count": int(np.sum(near)),
        "diagnostic_point_count": int(near.size),
        "interior_admissible_screening_fraction": float(np.mean(screening_mask)),
        "interior_admissible_screening_point_count": int(np.sum(screening_mask)),
        "interior_breakpoint_count_on_diagnostic_grid": int(np.sum(breakpoint_mask)),
    }
    if np.any(near):
        summary.update(
            {
                "mean_abs_dot_psi_near": float(np.mean(np.abs(quantities["dot_psi"][near]))),
                "mean_abs_ddot_psi_near": float(np.mean(np.abs(quantities["ddot_psi"][near]))),
                "B_nonpositive_fraction_near": float(np.mean(quantities["B"][near] <= 0.0)),
                "strict_control_admissible_fraction_near": float(np.mean(admissible[near])),
                "control_vs_strict_mae_near": float(
                    np.nanmean(np.abs(result["diagnostic_u"][near] - strict_control[near]))
                ),
            }
        )
    else:
        summary.update(
            {
                "mean_abs_dot_psi_near": None,
                "mean_abs_ddot_psi_near": None,
                "B_nonpositive_fraction_near": None,
                "strict_control_admissible_fraction_near": None,
                "control_vs_strict_mae_near": None,
            }
        )
    if np.any(screening_mask):
        summary.update(
            {
                "mean_abs_psi_relative_screening": float(
                    np.mean(np.abs(psi_relative[screening_mask]))
                ),
                "mean_abs_dot_psi_screening": float(
                    np.mean(np.abs(quantities["dot_psi"][screening_mask]))
                ),
                "mean_abs_ddot_psi_screening": float(
                    np.mean(np.abs(quantities["ddot_psi"][screening_mask]))
                ),
                "B_nonpositive_fraction_screening": float(
                    np.mean(quantities["B"][screening_mask] <= 0.0)
                ),
                "control_vs_strict_mae_screening": float(
                    np.nanmean(
                        np.abs(
                            result["diagnostic_u"][screening_mask]
                            - strict_control[screening_mask]
                        )
                    )
                ),
            }
        )
    else:
        summary.update(
            {
                "mean_abs_psi_relative_screening": None,
                "mean_abs_dot_psi_screening": None,
                "mean_abs_ddot_psi_screening": None,
                "B_nonpositive_fraction_screening": None,
                "control_vs_strict_mae_screening": None,
            }
        )

    near_without_breakpoints = near & interval_interior
    screening_without_breakpoints = screening_mask & interval_interior

    def masked_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
        return float(np.mean(values[mask])) if np.any(mask) else None

    summary["breakpoint_excluded_diagnostics"] = {
        "threshold_point_count": int(np.sum(near_without_breakpoints)),
        "screening_point_count": int(np.sum(screening_without_breakpoints)),
        "mean_abs_dot_psi_threshold": masked_mean(
            np.abs(quantities["dot_psi"]), near_without_breakpoints
        ),
        "mean_abs_ddot_psi_threshold": masked_mean(
            np.abs(quantities["ddot_psi"]), near_without_breakpoints
        ),
        "control_vs_strict_mae_threshold": masked_mean(
            np.abs(result["diagnostic_u"] - strict_control), near_without_breakpoints
        ),
        "mean_abs_dot_psi_screening": masked_mean(
            np.abs(quantities["dot_psi"]), screening_without_breakpoints
        ),
        "mean_abs_ddot_psi_screening": masked_mean(
            np.abs(quantities["ddot_psi"]), screening_without_breakpoints
        ),
        "control_vs_strict_mae_screening": masked_mean(
            np.abs(result["diagnostic_u"] - strict_control), screening_without_breakpoints
        ),
    }
    return summary


def strict_singular_comparison_figure(manifest: dict) -> None:
    """Compare all order-one singular conditions under one continuous evaluator."""
    methods = [
        ("direct time-mesh", "Direct time-mesh, n=800", GREEN),
        ("PMP/KKT Transformer", "Selected PMP/KKT Transformer", BLUE),
    ]
    evaluated = []
    summaries = {}
    for family, label, color in methods:
        _, _, result = selected_solution(manifest, family)
        quantities = strict_singular_quantities(result)
        summaries[family] = strict_singular_summary(result, quantities)
        evaluated.append((label, color, result, quantities))

    figure, axes = plt.subplots(4, 2, figsize=(9.1, 8.2), sharex="col")
    for column, (label, color, result, quantities) in enumerate(evaluated):
        t = result["diagnostic_t"]
        axes[0, column].plot(
            t,
            result["diagnostic_u"],
            color=color,
            lw=1.7,
            label=r"control $u(t)$",
        )
        psi_relative = quantities["psi"] / NOMINAL_TUMOR_PROBLEM.gamma
        strict_control = quantities["strict_control"]
        screening_mask = (
            (np.abs(psi_relative) <= 0.005)
            & np.isfinite(strict_control)
            & (strict_control >= 0.0)
            & (strict_control <= NOMINAL_TUMOR_PROBLEM.umax)
            & (result["diagnostic_u"] > 0.0)
            & (result["diagnostic_u"] < NOMINAL_TUMOR_PROBLEM.umax)
        )
        axes[0, column].plot(
            t,
            np.where(screening_mask, strict_control, np.nan),
            color=ORANGE,
            lw=1.15,
            ls="--",
            label=r"PMP singular-control candidate $-A/B$",
        )
        if not np.any(screening_mask):
            axes[0, column].text(
                0.5,
                0.08,
                "no point satisfies\n" r"$|\psi/\gamma|\leq 0.005$",
                transform=axes[0, column].transAxes,
                ha="center",
                va="bottom",
                color=GRAY,
                fontsize=7.5,
            )
        axes[0, column].set_ylim(-0.12, 3.12)
        axes[0, column].set_title(label)

        axes[1, column].axhspan(-0.005, 0.005, color="#D1FAE5", alpha=0.7)
        axes[1, column].plot(t, psi_relative, color=GRAY, lw=1.25)
        axes[1, column].axhline(0.0, color="#111827", lw=0.7)

        axes[2, column].plot(
            t,
            quantities["dot_psi"] / NOMINAL_TUMOR_PROBLEM.gamma,
            color=RED,
            lw=1.15,
        )
        axes[2, column].axhline(0.0, color="#111827", lw=0.7)

        axes[3, column].plot(
            t,
            quantities["ddot_psi"] / NOMINAL_TUMOR_PROBLEM.gamma,
            color=BLUE,
            lw=1.05,
        )
        axes[3, column].axhline(0.0, color="#111827", lw=0.7)
        axes[3, column].set_xlabel(r"time $t$")

        for row in range(4):
            axes[row, column].grid(True, color="#E5E7EB", lw=0.55)

    axes[0, 0].set_ylabel(r"control $u$")
    axes[1, 0].set_ylabel(r"$\psi/\gamma$")
    axes[2, 0].set_ylabel(r"$\dot{\psi}/\gamma$")
    axes[3, 0].set_ylabel(r"$\ddot{\psi}/\gamma$")
    axes[0, 0].legend(frameon=False, ncol=2, loc="lower center")
    figure.tight_layout()
    figure.savefig(FIGURES / "strict_singular_comparison.png", bbox_inches="tight")
    plt.close(figure)

    output = RESULTS / "structure_diagnostics" / "strict_comparison_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "psi_relative_tolerance": 0.005,
                "continuous_diagnostic_points": 4001,
                "screening_mask_definition": (
                    "|psi/gamma|<=0.005, 0<u<umax, |B|>1e-12, and admissible -A/B; "
                    "the mask does not impose additional dot(psi), ddot(psi), or B<=0 thresholds"
                ),
                "zoh_breakpoint_convention": (
                    "right-continuous, left-endpoint-held: u(t_k)=u_k and "
                    "u(t)=u_k on [t_k,t_{k+1})"
                ),
                "methods": summaries,
            },
            indent=2,
        )
        + "\n"
    )


def refined_strict_singular_comparison_figure(manifest: dict) -> None:
    """Compare the direct reference with the transition-refined schedule."""
    _, _, direct_result = selected_solution(manifest, "direct time-mesh")
    refined_result = evaluated_solution(REFINED_TIME_ONLY)
    _, _, original_result = selected_solution(manifest, "PMP/KKT Transformer")
    methods = [
        ("Direct numerical reference, n=800", GREEN, direct_result),
        ("Full-gradient-refined time-only Transformer", BLUE, refined_result),
    ]
    evaluated = []
    summaries = {}
    for label, color, result in methods:
        quantities = strict_singular_quantities(result)
        summaries[label] = strict_singular_summary(result, quantities)
        evaluated.append((label, color, result, quantities))

    figure, axes = plt.subplots(4, 2, figsize=(9.1, 8.2), sharex="col")
    for column, (label, color, result, quantities) in enumerate(evaluated):
        time = result["diagnostic_t"]
        axes[0, column].step(
            time,
            result["diagnostic_u"],
            where="post",
            color=color,
            lw=1.65,
            label=r"reported control $u(t)$",
        )
        psi_relative = quantities["psi"] / NOMINAL_TUMOR_PROBLEM.gamma
        strict_control = quantities["strict_control"]
        screening_mask = (
            (np.abs(psi_relative) <= 0.005)
            & np.isfinite(strict_control)
            & (strict_control >= 0.0)
            & (strict_control <= NOMINAL_TUMOR_PROBLEM.umax)
            & (result["diagnostic_u"] > 0.0)
            & (result["diagnostic_u"] < NOMINAL_TUMOR_PROBLEM.umax)
        )
        axes[0, column].plot(
            time,
            np.where(screening_mask, strict_control, np.nan),
            color=ORANGE,
            lw=1.1,
            ls="--",
            label=r"singular-control value $-A/B$",
        )
        if column == 1:
            axes[0, column].step(
                original_result["diagnostic_t"],
                original_result["diagnostic_u"],
                where="post",
                color=GRAY,
                lw=1.0,
                ls=":",
                alpha=0.9,
                label="pre-refinement control",
            )
            early, late = control_transition_centers(REFINED_TIME_ONLY)
            for center, marker_color in ((early, ORANGE), (late, RED)):
                axes[0, column].axvline(
                    center, color=marker_color, lw=0.75, ls=":", alpha=0.8
                )
        axes[0, column].set_ylim(-0.12, 3.12)
        axes[0, column].set_title(label)

        axes[1, column].axhspan(-0.005, 0.005, color="#D1FAE5", alpha=0.7)
        axes[1, column].plot(time, psi_relative, color=GRAY, lw=1.25)
        axes[1, column].axhline(0.0, color="#111827", lw=0.7)

        axes[2, column].plot(
            time,
            quantities["dot_psi"] / NOMINAL_TUMOR_PROBLEM.gamma,
            color=RED,
            lw=1.15,
        )
        axes[2, column].axhline(0.0, color="#111827", lw=0.7)

        axes[3, column].plot(
            time,
            quantities["ddot_psi"] / NOMINAL_TUMOR_PROBLEM.gamma,
            color=BLUE,
            lw=1.05,
        )
        axes[3, column].axhline(0.0, color="#111827", lw=0.7)
        axes[3, column].set_xlabel(r"time $t$")
        for row in range(4):
            axes[row, column].grid(True, color="#E5E7EB", lw=0.55)

    axes[0, 0].set_ylabel(r"control $u$")
    axes[1, 0].set_ylabel(r"$\psi/\gamma$")
    axes[2, 0].set_ylabel(r"$\dot{\psi}/\gamma$")
    axes[3, 0].set_ylabel(r"$\ddot{\psi}/\gamma$")
    axes[0, 0].legend(frameon=False, fontsize=7.1, loc="lower center")
    axes[0, 1].legend(frameon=False, fontsize=6.8, loc="lower center")
    figure.tight_layout()
    figure.savefig(
        FIGURES / "refined_strict_singular_comparison.png", bbox_inches="tight"
    )
    plt.close(figure)

    output = (
        RESULTS
        / "structure_diagnostics"
        / "refined_strict_comparison_summary.json"
    )
    output.write_text(
        json.dumps(
            {
                "psi_relative_tolerance": 0.005,
                "continuous_diagnostic_points": 4001,
                "refined_source": str(REFINED_TIME_ONLY.relative_to(ROOT)),
                "methods": summaries,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    setup()
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    trajectory_figure(manifest)
    phase_figure(manifest)
    phenotype_snapshot_figure(manifest)
    phenotype_plot_options(manifest)
    training_figure(manifest)
    component_training_figure(manifest)
    optimality_figure(manifest)
    refined_optimality_figure()
    comparison_figure()
    refined_comparison_figure()
    mesh_figure()
    singular_figure()
    strict_singular_comparison_figure(manifest)
    refined_trajectory_figure()
    refined_phenotype_transition_figure()
    refined_strict_singular_comparison_figure(manifest)
    print(FIGURES)


if __name__ == "__main__":
    main()
