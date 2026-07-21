#!/usr/bin/env python3
"""Plot the compact teacher-facing figures from the unified n=800 bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

from train_teacher_free_resolution_curriculum import interval_crossing_width


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "outputs/teacher_three_case_time_only_transformer_results_20260720"
DEFAULT_DIRECT = ROOT / "paper_runs/canonical_results/solutions/direct_time_mesh_n800.npz"
CASES = ("time_only", "feedback_cf", "feedback_der")
LABELS = (
    r"Transformer $u(t)$",
    r"Case 1 $u(N,t)$",
    r"Case 2 $u(N,t)$",
)
COLORS = ("#266D63", "#A65348", "#4B6382")


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_cases(data_dir: Path) -> list[dict[str, np.ndarray]]:
    loaded = []
    for case in CASES:
        path = data_dir / case / "teacher_facing_diagnostics.npz"
        with np.load(path) as data:
            loaded.append({key: np.asarray(data[key]) for key in data.files})
    return loaded


def save_both(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_control_derivatives(data_dir: Path, cases: list[dict[str, np.ndarray]]) -> None:
    style()
    figure, axes = plt.subplots(
        4,
        3,
        figsize=(15.0, 6.35),
        dpi=220,
        sharex="col",
        gridspec_kw={"height_ratios": (1.0, 1.0, 1.0, 1.0)},
    )
    h_low = min(float(d["continuous_hamiltonian"].min()) for d in cases)
    h_high = max(float(d["continuous_hamiltonian"].max()) for d in cases)
    h_pad = 0.05 * (h_high - h_low)
    interval_hu = [
        data["full_gradient"] / float(np.diff(data["breakpoints"][:2])[0])
        for data in cases
    ]
    g_bound = 1.06 * max(float(np.abs(values).max()) for values in interval_hu)
    hessian_column_norms = [
        np.linalg.norm(data["full_hessian"], axis=0) for data in cases
    ]
    r_high = 1.06 * max(float(values.max()) for values in hessian_column_norms)
    grid_color = "#E2E5E8"

    for column, (data, label, color, column_norm, hu_values) in enumerate(
        zip(cases, LABELS, COLORS, hessian_column_norms, interval_hu, strict=True)
    ):
        t = data["breakpoints"]
        ti = t[:-1]
        axes[0, column].step(ti, data["interval_control"], where="post", color=color, lw=1.25)
        axes[0, column].set_ylim(-0.05, 3.08)
        axes[0, column].set_title(label, fontweight="semibold", pad=4)

        axes[1, column].plot(
            data["continuous_time"], data["continuous_hamiltonian"], color=color, lw=1.2
        )
        axes[1, column].set_ylim(h_low - h_pad, h_high + h_pad)

        axes[2, column].plot(ti, hu_values, color=color, lw=1.05)
        axes[2, column].axhline(0.0, color="#20242A", lw=0.55)
        axes[2, column].set_ylim(-g_bound, g_bound)

        axes[3, column].plot(ti, column_norm, color=color, lw=1.15)
        axes[3, column].set_ylim(0.0, r_high)
        axes[3, column].set_xlabel(r"interval time $t_j$")
        axes[3, column].grid(True, color=grid_color, lw=0.45)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        axes[3, column].yaxis.set_major_formatter(formatter)

        for row in range(3):
            axis = axes[row, column]
            axis.set_xlim(0.0, 10.0)
            axis.grid(True, color=grid_color, lw=0.45)
            if row == 2:
                formatter = ScalarFormatter(useMathText=True)
                formatter.set_powerlimits((-2, 2))
                axis.yaxis.set_major_formatter(formatter)
        axes[3, column].set_xlim(0.0, 10.0)

    axes[0, 0].set_ylabel(r"control $u_k$")
    axes[1, 0].set_ylabel(r"instantaneous $H(t)$")
    axes[2, 0].set_ylabel(r"interval-averaged full $H_u=g_k/\Delta t$")
    axes[3, 0].set_ylabel(r"Hessian-column $\ell_2$ norm $\|R_{:,j}\|_2$")
    figure.subplots_adjust(left=0.07, right=0.99, top=0.94, bottom=0.09, wspace=0.24, hspace=0.18)
    save_both(figure, data_dir / "three_case_control_derivatives")


def plot_state_evolution(data_dir: Path, cases: list[dict[str, np.ndarray]]) -> None:
    style()
    figure, axes = plt.subplots(
        3,
        3,
        figsize=(13.7, 7.35),
        dpi=220,
        gridspec_kw={"height_ratios": (0.82, 0.72, 1.08)},
    )
    state_max = max(float(d["continuous_state"].max()) for d in cases)
    images = []

    for column, (data, label, color) in enumerate(zip(cases, LABELS, COLORS, strict=True)):
        t = data["continuous_time"]
        total = data["total_population"]
        top = axes[0, column]
        top.plot(t, total, color=color, lw=1.5)
        top.set_xlim(0.0, 10.0)
        top.set_ylim(0.0, 218.0)
        top.set_title(label, fontweight="semibold", pad=5)
        top.grid(True, color="#E2E5E8", lw=0.5)
        top.tick_params(labelsize=6.5, pad=1.5)
        top.text(
            0.97,
            0.90,
            "full horizon",
            transform=top.transAxes,
            ha="right",
            va="top",
            fontsize=6.4,
            color="#4B5563",
        )
        if column == 0:
            top.set_ylabel(r"total population $\sum_i N_i(t)$")

        mask = (t >= 1.0) & (t <= 8.5)
        relative_range = 100.0 * np.ptp(total[mask]) / np.mean(total[mask])
        detail = axes[1, column]
        detail.plot(t[mask], total[mask], color=color, lw=1.25)
        detail.set_xlim(1.0, 8.5)
        local_span = max(float(np.ptp(total[mask])), 0.05)
        local_mid = 0.5 * float(total[mask].min() + total[mask].max())
        detail.set_ylim(local_mid - 0.62 * local_span, local_mid + 0.62 * local_span)
        detail.grid(True, color="#E5E7EB", lw=0.4)
        detail.tick_params(labelsize=6.5, pad=1.5)
        detail.set_xlabel(r"time $t$", labelpad=1)
        if column == 0:
            detail.set_ylabel("mid-horizon detail")
        detail.text(
            0.97,
            0.88,
            rf"$t\in[1,8.5]$: {relative_range:.3f}%",
            transform=detail.transAxes,
            ha="right",
            va="top",
            fontsize=6.4,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0},
        )

        bottom = axes[2, column]
        image = bottom.imshow(
            data["continuous_state"].T,
            origin="lower",
            extent=(0.0, 10.0, 0.0, 1.0),
            aspect="auto",
            cmap="viridis",
            vmin=0.0,
            vmax=state_max,
            interpolation="nearest",
        )
        images.append(image)
        bottom.set_xlim(0.0, 10.0)
        bottom.set_xlabel(r"time $t$", labelpad=1)
        bottom.tick_params(labelsize=6.5, pad=1.5)
        if column == 0:
            bottom.set_ylabel(r"phenotype $x$")

    figure.subplots_adjust(
        left=0.065,
        right=0.925,
        top=0.95,
        bottom=0.075,
        wspace=0.19,
        hspace=0.35,
    )
    colorbar_axis = figure.add_axes([0.944, 0.075, 0.012, 0.34])
    colorbar = figure.colorbar(images[0], cax=colorbar_axis)
    colorbar.set_label(r"population density $N(t,x)$")
    save_both(figure, data_dir / "three_case_state_evolution")


def plot_time_only_sharpness(
    data_dir: Path, cases: list[dict[str, np.ndarray]], direct_path: Path
) -> None:
    style()
    learned = cases[0]
    with np.load(direct_path) as direct_data:
        direct_t = np.asarray(direct_data["t"], dtype=np.float64)
        direct_u = np.asarray(direct_data["u"], dtype=np.float64)
    learned_t = learned["breakpoints"]
    learned_u = learned["interval_control"]
    if learned_u.size == learned_t.size - 1:
        learned_u_plot = np.r_[learned_u, learned_u[-1]]
    else:
        learned_u_plot = learned_u

    figure = plt.figure(figsize=(10.0, 7.05), dpi=220)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.05, 1.0), hspace=0.27, wspace=0.20)
    full = figure.add_subplot(grid[0, :])
    early = figure.add_subplot(grid[1, 0])
    late = figure.add_subplot(grid[1, 1])
    learned_color = "#266D63"
    direct_color = "#D1792F"

    learned_early = interval_crossing_width(learned_t, learned_u_plot, "early")
    learned_late = interval_crossing_width(learned_t, learned_u_plot, "late")
    direct_early = interval_crossing_width(direct_t, direct_u, "early")
    direct_late = interval_crossing_width(direct_t, direct_u, "late")

    for axis in (full, early, late):
        axis.step(
            learned_t,
            learned_u_plot,
            where="post",
            color=learned_color,
            lw=1.7,
            label=r"Transformer $u(t)$",
        )
        axis.step(direct_t, direct_u, where="post", color=direct_color, lw=1.25, ls="--", label=r"Direct $n=800$ reference")
        axis.set_ylim(-0.05, 3.08)
        axis.grid(True, color="#E2E5E8", lw=0.5)
        axis.set_xlabel(r"time $t$")
        axis.set_ylabel(r"control $u$")
    full.set_xlim(0.0, 10.0)
    full.set_title("Full-horizon schedules", fontweight="semibold", pad=5)
    full.legend(loc="lower center", ncol=2, frameon=False, fontsize=8)
    early.set_xlim(0.35, 0.58)
    early.set_title("Early transition", fontweight="semibold", pad=5)
    early.text(
        0.04,
        0.09,
        f"10-90% width: {learned_early:.5f} (Transformer)\n"
        f"{direct_early:.5f} (direct)",
        transform=early.transAxes,
        fontsize=7.0,
    )
    late.set_xlim(8.94, 9.22)
    late.set_title("Late transition", fontweight="semibold", pad=5)
    late.text(
        0.04,
        0.09,
        f"10-90% width: {learned_late:.5f} (Transformer)\n"
        f"{direct_late:.5f} (direct)",
        transform=late.transAxes,
        fontsize=7.0,
    )
    figure.subplots_adjust(left=0.085, right=0.985, top=0.955, bottom=0.08)
    save_both(figure, data_dir / "time_only_sharpness")

    metrics = {
        "time_only_transformer": {
            "early_width_10_90": learned_early,
            "late_width_10_90": learned_late,
        },
        "direct_n800_reference": {
            "early_width_10_90": direct_early,
            "late_width_10_90": direct_late,
        },
    }
    (data_dir / "time_only_sharpness_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )


def transition_breakpoint(
    breakpoints: np.ndarray, control: np.ndarray, side: str
) -> float:
    interval_t = np.asarray(breakpoints[:-1], dtype=np.float64)
    differences = np.diff(np.asarray(control, dtype=np.float64))
    if side == "early":
        candidates = np.flatnonzero(interval_t[:-1] < 0.5 * interval_t[-1])
        index = int(candidates[np.argmin(differences[candidates])])
    elif side == "late":
        candidates = np.flatnonzero(interval_t[:-1] >= 0.5 * interval_t[-1])
        index = int(candidates[np.argmax(differences[candidates])])
    else:
        raise ValueError(side)
    return float(breakpoints[index + 1])


def plot_time_only_phenotype_profiles(
    data_dir: Path, cases: list[dict[str, np.ndarray]]
) -> None:
    style()
    data = cases[0]
    time = data["continuous_time"]
    state = data["continuous_state"]
    trait = np.linspace(0.0, 1.0, state.shape[1], dtype=np.float64)
    early = transition_breakpoint(
        data["breakpoints"], data["interval_control"], "early"
    )
    late = transition_breakpoint(
        data["breakpoints"], data["interval_control"], "late"
    )
    profile_times = (early, 2.0, 8.5, late)
    profile_rows = [int(np.argmin(np.abs(time - value))) for value in profile_times]

    figure = plt.figure(figsize=(13.4, 6.7), dpi=220)
    grid = figure.add_gridspec(
        2,
        3,
        width_ratios=(1.55, 1.0, 1.0),
        wspace=0.27,
        hspace=0.34,
    )
    surface = figure.add_subplot(grid[:, 0], projection="3d")
    surface_rows = np.arange(0, time.size, 32, dtype=int)
    if surface_rows[-1] != time.size - 1:
        surface_rows = np.append(surface_rows, time.size - 1)
    trait_mesh, time_mesh = np.meshgrid(trait, time[surface_rows])
    surface.plot_surface(
        trait_mesh,
        time_mesh,
        state[surface_rows],
        cmap="viridis",
        linewidth=0.0,
        antialiased=True,
        alpha=0.94,
    )
    transition_colors = ("#C56A24", "#A44237")
    for value, color in zip((early, late), transition_colors, strict=True):
        row = int(np.argmin(np.abs(time - value)))
        surface.plot(
            trait,
            np.full_like(trait, time[row]),
            state[row],
            color=color,
            lw=2.5,
        )
    surface.set_title(r"Complete phenotype evolution $N(t,x)$", pad=8)
    surface.set_xlabel(r"phenotype $x$", labelpad=5)
    surface.set_ylabel(r"time $t$", labelpad=5)
    surface.set_zlabel(r"population $N$", labelpad=4)
    surface.view_init(elev=27, azim=-54)

    axes = [
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[0, 2]),
        figure.add_subplot(grid[1, 1]),
        figure.add_subplot(grid[1, 2]),
    ]
    y_upper = 1.08 * float(np.max(state[profile_rows]))
    for axis, target, row in zip(axes, profile_times, profile_rows, strict=True):
        if np.isclose(target, early):
            color = transition_colors[0]
            label = "early control transition"
        elif np.isclose(target, late):
            color = transition_colors[1]
            label = "late control transition"
        else:
            color = "#4B6382"
            label = "intermediate profile"
        axis.plot(trait, state[row], color=color, marker="o", ms=2.5, lw=1.45)
        axis.set_title(rf"$t={time[row]:.3f}$ ({label})", color=color, pad=4)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, y_upper)
        axis.grid(True, color="#E2E5E8", lw=0.45)
    for axis in axes[2:]:
        axis.set_xlabel(r"phenotype $x$")
    for axis in (axes[0], axes[2]):
        axis.set_ylabel(r"population $N(t,x)$")
    for axis in (axes[1], axes[3]):
        axis.tick_params(labelleft=False)
    figure.subplots_adjust(left=0.035, right=0.985, bottom=0.09, top=0.93)
    save_both(figure, data_dir / "time_only_phenotype_profiles")

    metrics = {
        "early_transition_breakpoint": early,
        "late_transition_breakpoint": late,
        "profile_times": [float(time[row]) for row in profile_rows],
        "early_adjacent_control_change": float(
            np.min(np.diff(data["interval_control"])[data["breakpoints"][:-2] < 5.0])
        ),
        "late_adjacent_control_change": float(
            np.max(np.diff(data["interval_control"])[data["breakpoints"][:-2] >= 5.0])
        ),
    }
    (data_dir / "time_only_phenotype_profiles_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )


def write_metrics_table(data_dir: Path, cases: list[dict[str, np.ndarray]]) -> None:
    objectives = []
    hamiltonian_ranges = []
    raw_linf = []
    pg_linf = []
    hessian_eigenvalue_l2 = []
    curvature = []
    for data in cases:
        control = data["interval_control"]
        gradient = data["full_gradient"]
        raw_linf.append(float(np.max(np.abs(gradient))))
        projected = control - np.clip(control - gradient, 0.0, 3.0)
        pg_linf.append(float(np.max(np.abs(projected))))
        hessian_eigenvalue_l2.append(
            float(np.linalg.norm(data["hessian_eigenvalues"]))
        )
        # F_h is not stored in the NPZ; read it from the case summary below.
        curvature.append(
            float(data["tested_critical_subspace_hessian_eigenvalues"].min())
        )
    for case in CASES:
        summary = json.loads((data_dir / case / "summary.json").read_text())
        objectives.append(float(summary["reduced_objective_F_h"]))
        hamiltonian_ranges.append(
            (float(summary["hamiltonian_min"]), float(summary["hamiltonian_max"]))
        )

    def tex_scientific(value: float, digits: int = 3, show_plus: bool = False) -> str:
        exponent = int(np.floor(np.log10(abs(value))))
        mantissa = value / (10.0**exponent)
        sign = "+" if show_plus and value > 0.0 else ""
        return rf"{sign}{mantissa:.{digits}f}\times10^{{{exponent}}}"

    lines = [
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"Metric & Transformer $u(t)$ & Case 1 $u(N,t)$ & Case 2 $u(N,t)$ \\",
        r"\midrule",
        (
            rf"Hamiltonian range $H(t)$ & "
            rf"{hamiltonian_ranges[0][0]:.4f}--{hamiltonian_ranges[0][1]:.4f} & "
            rf"{hamiltonian_ranges[1][0]:.4f}--{hamiltonian_ranges[1][1]:.4f} & "
            rf"{hamiltonian_ranges[2][0]:.4f}--{hamiltonian_ranges[2][1]:.4f} \\"
        ),
        (
            rf"Reduced objective $\widehat J_h$ & {objectives[0]:.6f} & "
            rf"{objectives[1]:.6f} & {objectives[2]:.6f} \\"
        ),
        (
            rf"Full reduced-gradient $\lVert g\rVert_\infty$ & "
            rf"${tex_scientific(raw_linf[0])}$ & "
            rf"${tex_scientific(raw_linf[1])}$ & "
            rf"${tex_scientific(raw_linf[2])}$ \\"
        ),
        (
            rf"First-order KKT residual $\|\mathcal G\|_\infty$ & "
            rf"${tex_scientific(pg_linf[0])}$ & "
            rf"${tex_scientific(pg_linf[1])}$ & "
            rf"${tex_scientific(pg_linf[2])}$ \\"
        ),
        (
            rf"Hessian eigenvalue $\ell_2$ norm $\|\lambda(R)\|_2$ & "
            rf"${hessian_eigenvalue_l2[0]:.6f}$ & "
            rf"${hessian_eigenvalue_l2[1]:.6f}$ & "
            rf"${hessian_eigenvalue_l2[2]:.6f}$ \\"
        ),
        (
            rf"Minimum eigenvalue of restricted Hessian $\lambda_{{\min}}$ & "
            rf"${tex_scientific(curvature[0], show_plus=True)}$ & "
            rf"${tex_scientific(curvature[1], show_plus=True)}$ & "
            rf"${tex_scientific(curvature[2], show_plus=True)}$ \\"
        ),
        (
            r"First- and second-order optimality checks & "
            + " & ".join(
                "Pass" if pg <= 1.0e-4 and value > 0.0 else "Fail"
                for pg, value in zip(pg_linf, curvature, strict=True)
            )
            + r" \\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    (data_dir / "metrics_table.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--direct", type=Path, default=DEFAULT_DIRECT)
    args = parser.parse_args()
    data_dir = resolve(args.data_dir)
    cases = load_cases(data_dir)
    plot_control_derivatives(data_dir, cases)
    plot_state_evolution(data_dir, cases)
    plot_time_only_sharpness(data_dir, cases, resolve(args.direct))
    plot_time_only_phenotype_profiles(data_dir, cases)
    write_metrics_table(data_dir, cases)
    print(data_dir)


if __name__ == "__main__":
    main()
