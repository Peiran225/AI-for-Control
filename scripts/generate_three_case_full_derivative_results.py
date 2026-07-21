#!/usr/bin/env python3
"""Evaluate H(t) and full control-space stationarity derivatives for three cases.

The teacher-requested derivatives are with respect to the complete physical
interval-control vector, not with respect to time.  For each realized
200-interval control vector this script differentiates

    F_h(u) = J_h(N_h(u), u),    N_{k+1} = Phi_k(N_k, u_k),

through the complete RK4 state recursion.  The resulting gradient is the exact
RK4 control-stationarity residual (its per-unit-time counterpart corresponds to
H_u), and the Hessian is the full derivative of that residual with respect to
all interval controls.  Both therefore include the indirect state dependence
N=N(u).  Feedback networks are used only to realize the comparison control
vector on one common initial state; the derivatives are not taken with respect
to network parameters.

The instantaneous Hamiltonian H(N(t), lambda(t), u(t)) is also shown along the
same high-accuracy continuous trajectory.  It is kept conceptually separate
from the scalar reduced physical objective F_h whose derivatives are reported.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import ScalarFormatter
import numpy as np
import torch
from scipy.integrate import simpson


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for search_path in (ROOT, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from diagnose_reduced_objective_hessian import (  # noqa: E402
    discrete_adjoint_gradient,
    projected_kkt,
    rk4_extended_step,
)
from generate_three_case_hamiltonian_results import (  # noqa: E402
    EvaluatedCase,
    RealizedCase,
    assert_common_problem,
    evaluate_case,
    load_time_only_case,
    realize_feedback_case,
    resistant_heavy_initial_state,
)
from train_paper_pmp_kkt import build_params  # noqa: E402
from tumor_problem import dynamics_numpy  # noqa: E402


DEFAULT_TIME_CHECKPOINT = (
    ROOT / "paper_runs/smoothness_weight_sweep/w3/seed_4/best_pmp_kkt.pt"
)
DEFAULT_CF_CHECKPOINT = (
    ROOT
    / "outputs/feedback_teacher_followup_20260719/cf_seed1/train/best_feedback_section5.pt"
)
DEFAULT_DER_CHECKPOINT = (
    ROOT
    / "outputs/feedback_teacher_followup_20260719/der_seed2/train/best_feedback_section5.pt"
)
DEFAULT_OUT_DIR = ROOT / "outputs/three_case_full_derivative_results"


@dataclass
class FullDerivativeResult:
    realized: RealizedCase
    continuous: EvaluatedCase
    reduced_value: float
    hamiltonian: np.ndarray
    gradient: np.ndarray
    projected_gradient: np.ndarray
    hessian: np.ndarray
    hessian_diagonal: np.ndarray
    eigenvalues: np.ndarray
    free_eigenvalues: np.ndarray
    free_indices: np.ndarray
    summary: dict[str, Any]


def resolve_path(path: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate.resolve()


def make_reduced_objective(
    realized: RealizedCase,
    initial_state: np.ndarray,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], dict[str, torch.Tensor]]:
    cfg = realized.cfg
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    params["N0"] = torch.tensor(initial_state, dtype=torch.float64)

    def reduced_objective(control: torch.Tensor) -> torch.Tensor:
        if control.ndim != 1 or control.numel() != cfg.n:
            raise ValueError(
                f"expected {cfg.n} interval controls, got {tuple(control.shape)}"
            )
        state_and_cost = torch.cat(
            (
                params["N0"],
                torch.zeros(1, dtype=control.dtype, device=control.device),
            )
        )
        for index in range(cfg.n):
            state_and_cost = rk4_extended_step(
                state_and_cost, control[index], cfg, params
            )
        return (
            params["alpha"] * state_and_cost[:-1]
        ).sum() + state_and_cost[-1]

    return reduced_objective, params


def instantaneous_hamiltonian(
    evaluated: EvaluatedCase,
    problem: Any,
) -> np.ndarray:
    vectors = problem.vectors()
    values = np.empty(evaluated.time.size, dtype=np.float64)
    for index, (state, costate, control) in enumerate(
        zip(
            evaluated.state,
            evaluated.costate,
            evaluated.control,
            strict=True,
        )
    ):
        drift = dynamics_numpy(state, float(control), problem)
        values[index] = (
            float(vectors["beta"] @ state)
            + float(problem.gamma * control)
            + float(costate @ drift)
        )
    return values


def continuous_interval_gradient(
    evaluated: EvaluatedCase,
) -> np.ndarray:
    n = evaluated.specification.cfg.n
    samples = evaluated.time.size - 1
    if samples % n != 0:
        raise ValueError(
            "diagnostic grid must contain an integer number of samples per interval"
        )
    samples_per_interval = samples // n
    psi = np.asarray(evaluated.quantities["psi"], dtype=np.float64)
    gradient = np.empty(n, dtype=np.float64)
    for index in range(n):
        left = index * samples_per_interval
        right = (index + 1) * samples_per_interval + 1
        gradient[index] = simpson(
            psi[left:right], x=evaluated.time[left:right]
        )
    return gradient


def directional_checks(
    reduced_objective: Callable[[torch.Tensor], torch.Tensor],
    control: np.ndarray,
    gradient: np.ndarray,
    hessian: np.ndarray,
    *,
    epsilon: float,
) -> dict[str, float]:
    generator = np.random.default_rng(20260719)
    direction = generator.normal(size=control.size)
    direction /= np.linalg.norm(direction)
    plus = control + epsilon * direction
    minus = control - epsilon * direction
    if np.any(plus < 0.0) or np.any(plus > 3.0):
        raise RuntimeError("positive finite-difference point left the control box")
    if np.any(minus < 0.0) or np.any(minus > 3.0):
        raise RuntimeError("negative finite-difference point left the control box")

    def value(values: np.ndarray) -> float:
        tensor = torch.tensor(values, dtype=torch.float64)
        return float(reduced_objective(tensor).detach().cpu())

    gradient_function = torch.func.grad(reduced_objective)

    def autodiff_gradient(values: np.ndarray) -> np.ndarray:
        tensor = torch.tensor(values, dtype=torch.float64)
        return gradient_function(tensor).detach().cpu().numpy()

    directional_exact = float(gradient @ direction)
    directional_fd = (value(plus) - value(minus)) / (2.0 * epsilon)
    hvp_exact = hessian @ direction
    hvp_fd = (
        autodiff_gradient(plus) - autodiff_gradient(minus)
    ) / (2.0 * epsilon)
    hvp_error = float(np.linalg.norm(hvp_fd - hvp_exact))
    hvp_reference = max(float(np.linalg.norm(hvp_exact)), 1.0e-14)
    return {
        "finite_difference_epsilon": epsilon,
        "directional_derivative_exact": directional_exact,
        "directional_derivative_finite_difference": directional_fd,
        "directional_derivative_absolute_error": abs(
            directional_fd - directional_exact
        ),
        "hessian_vector_relative_error": hvp_error / hvp_reference,
    }


def evaluate_full_derivatives(
    realized: RealizedCase,
    evaluated: EvaluatedCase,
    initial_state: np.ndarray,
    problem: Any,
    *,
    bound_tolerance: float,
    kkt_tolerance: float,
    fd_epsilon: float,
) -> FullDerivativeResult:
    reduced_objective, params = make_reduced_objective(realized, initial_state)
    control_tensor = torch.tensor(realized.controls, dtype=torch.float64)
    gradient_function = torch.func.grad(reduced_objective)
    gradient_tensor = gradient_function(control_tensor)
    hessian_raw_tensor = torch.func.hessian(reduced_objective)(control_tensor)
    reduced_value = float(reduced_objective(control_tensor).detach().cpu())
    gradient = gradient_tensor.detach().cpu().numpy()
    hessian_raw = hessian_raw_tensor.detach().cpu().numpy()
    hessian = 0.5 * (hessian_raw + hessian_raw.T)
    hessian_diagonal = np.diag(hessian).copy()

    projected, signed_stationarity, free, active = projected_kkt(
        realized.controls,
        gradient,
        0.0,
        float(realized.cfg.umax),
        bound_tolerance,
    )
    eigenvalues = np.linalg.eigvalsh(hessian)
    free_indices = np.flatnonzero(free)
    if free_indices.size:
        free_hessian = hessian[np.ix_(free, free)]
        free_eigenvalues = np.linalg.eigvalsh(free_hessian)
    else:
        free_eigenvalues = np.empty(0, dtype=np.float64)

    discrete_gradient = discrete_adjoint_gradient(
        control_tensor, realized.cfg, params
    )
    continuous_gradient = continuous_interval_gradient(evaluated)
    checks = directional_checks(
        reduced_objective,
        realized.controls,
        gradient,
        hessian,
        epsilon=fd_epsilon,
    )
    symmetry_relative = float(
        np.linalg.norm(hessian_raw - hessian_raw.T)
        / max(np.linalg.norm(hessian_raw), 1.0e-14)
    )
    hamiltonian = instantaneous_hamiltonian(evaluated, problem)
    projected_linf = float(np.max(np.abs(projected)))
    first_order_pass = projected_linf <= kkt_tolerance
    eigenvalue_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    eigenvalue_tolerance = 1.0e-8 * eigenvalue_scale
    free_minimum = (
        float(np.min(free_eigenvalues))
        if free_eigenvalues.size
        else float("nan")
    )
    second_order_pass = bool(
        first_order_pass
        and (
            not free_eigenvalues.size
            or free_minimum >= -eigenvalue_tolerance
        )
    )

    summary: dict[str, Any] = {
        "case_id": realized.case_id,
        "case_label": realized.label.replace("$", ""),
        "checkpoint": str(realized.checkpoint),
        "n_intervals": int(realized.cfg.n),
        "dt": float(realized.cfg.T / realized.cfg.n),
        "reduced_objective_F_h": reduced_value,
        "hamiltonian_min": float(np.min(hamiltonian)),
        "hamiltonian_max": float(np.max(hamiltonian)),
        "hamiltonian_rms": float(np.sqrt(np.mean(hamiltonian**2))),
        "full_gradient_l2": float(np.linalg.norm(gradient)),
        "full_gradient_linf": float(np.max(np.abs(gradient))),
        "projected_gradient_linf": projected_linf,
        "signed_box_stationarity_linf": float(
            np.max(np.abs(signed_stationarity))
        ),
        "free_variables": int(np.sum(free)),
        "active_variables": int(np.sum(active)),
        "hessian_diagonal_min": float(np.min(hessian_diagonal)),
        "hessian_diagonal_max": float(np.max(hessian_diagonal)),
        "hessian_min_eigenvalue_full": float(np.min(eigenvalues)),
        "hessian_max_eigenvalue_full": float(np.max(eigenvalues)),
        "hessian_min_eigenvalue_free": (
            free_minimum if free_eigenvalues.size else None
        ),
        "hessian_negative_eigenvalues_full": int(
            np.sum(eigenvalues < -eigenvalue_tolerance)
        ),
        "hessian_symmetry_relative_error": symmetry_relative,
        "autodiff_vs_discrete_adjoint_linf": float(
            np.max(np.abs(gradient - discrete_gradient))
        ),
        "rk4_vs_continuous_adjoint_linf": float(
            np.max(np.abs(gradient - continuous_gradient))
        ),
        "kkt_tolerance": kkt_tolerance,
        "first_order_pass": bool(first_order_pass),
        "second_order_test_applicable": bool(first_order_pass),
        "second_order_pass": bool(second_order_pass),
        **checks,
    }
    return FullDerivativeResult(
        realized=realized,
        continuous=evaluated,
        reduced_value=reduced_value,
        hamiltonian=hamiltonian,
        gradient=gradient,
        projected_gradient=projected,
        hessian=hessian,
        hessian_diagonal=hessian_diagonal,
        eigenvalues=eigenvalues,
        free_eigenvalues=free_eigenvalues,
        free_indices=free_indices,
        summary=summary,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0].keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def save_case(result: FullDerivativeResult, out_dir: Path) -> None:
    case_dir = out_dir / result.realized.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    time_intervals = result.realized.breakpoints[:-1]
    gradient_rows = [
        {
            "k": index,
            "t_left": float(time_intervals[index]),
            "u_k": float(result.realized.controls[index]),
            "full_gradient": float(result.gradient[index]),
            "projected_gradient": float(result.projected_gradient[index]),
            "hessian_diagonal": float(result.hessian_diagonal[index]),
        }
        for index in range(result.realized.cfg.n)
    ]
    hamiltonian_rows = [
        {
            "t": float(time),
            "u": float(control),
            "H": float(value),
        }
        for time, control, value in zip(
            result.continuous.time,
            result.continuous.control,
            result.hamiltonian,
            strict=True,
        )
    ]
    eigenvalue_rows = [
        {"index": index, "hessian_eigenvalue": float(value)}
        for index, value in enumerate(result.eigenvalues)
    ]
    write_csv(case_dir / "gradient_and_diagonal_by_interval.csv", gradient_rows)
    write_csv(case_dir / "hamiltonian_by_time.csv", hamiltonian_rows)
    write_csv(case_dir / "hessian_eigenvalues.csv", eigenvalue_rows)
    np.savez_compressed(
        case_dir / "full_derivatives.npz",
        time=result.realized.breakpoints,
        interval_control=result.realized.controls,
        hamiltonian_time=result.continuous.time,
        hamiltonian=result.hamiltonian,
        full_gradient=result.gradient,
        projected_gradient=result.projected_gradient,
        full_hessian=result.hessian,
        hessian_diagonal=result.hessian_diagonal,
        hessian_eigenvalues=result.eigenvalues,
        free_indices=result.free_indices,
        free_hessian_eigenvalues=result.free_eigenvalues,
    )
    (case_dir / "summary.json").write_text(
        json.dumps(result.summary, indent=2) + "\n", encoding="utf-8"
    )


def common_limits(arrays: list[np.ndarray], *, symmetric: bool) -> tuple[float, float]:
    low = min(float(np.min(array)) for array in arrays)
    high = max(float(np.max(array)) for array in arrays)
    if symmetric:
        bound = max(abs(low), abs(high), 1.0e-12) * 1.08
        return -bound, bound
    span = max(high - low, 1.0e-12)
    return low - 0.06 * span, high + 0.06 * span


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 9.4,
            "axes.labelsize": 8.3,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def plot_time_profiles(
    results: list[FullDerivativeResult],
    pdf_path: Path,
    png_path: Path,
) -> None:
    configure_plotting()
    colors = ("#5B6573", "#9E4F45", "#28766B")
    figure, axes = plt.subplots(3, 3, figsize=(13.8, 5.9), dpi=220)
    h_limits = common_limits([result.hamiltonian for result in results], symmetric=False)
    g_limits = common_limits([result.gradient for result in results], symmetric=True)
    d_limits = common_limits(
        [result.hessian_diagonal for result in results], symmetric=True
    )
    zero_color = "#1F2937"
    grid_color = "#E5E7EB"

    for column, (result, color) in enumerate(zip(results, colors, strict=True)):
        label = result.realized.label.replace("$", "")
        time = result.continuous.time
        interval_time = result.realized.breakpoints[:-1]

        top = axes[0, column]
        top.plot(time, result.hamiltonian, color=color, linewidth=1.55)
        top.set_ylim(*h_limits)
        top.set_title(label, pad=7, fontweight="semibold")
        top.text(
            0.97,
            0.94,
            rf"range [{result.summary['hamiltonian_min']:.2f}, {result.summary['hamiltonian_max']:.2f}]",
            transform=top.transAxes,
            ha="right",
            va="top",
            fontsize=6.8,
            color="#374151",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
        )

        middle = axes[1, column]
        middle.plot(interval_time, result.gradient, color=color, linewidth=1.45)
        middle.axhline(0.0, color=zero_color, linewidth=0.7)
        middle.set_ylim(*g_limits)
        middle.text(
            0.97,
            0.94,
            rf"$\|g\|_\infty={result.summary['full_gradient_linf']:.3g}$",
            transform=middle.transAxes,
            ha="right",
            va="top",
            fontsize=6.8,
            color="#374151",
        )

        bottom = axes[2, column]
        bottom.plot(
            interval_time,
            result.hessian_diagonal,
            color=color,
            linewidth=1.45,
        )
        bottom.axhline(0.0, color=zero_color, linewidth=0.7)
        bottom.set_ylim(*d_limits)
        bottom.text(
            0.97,
            0.94,
            (
                rf"diag range [{result.summary['hessian_diagonal_min']:.2e}, "
                rf"{result.summary['hessian_diagonal_max']:.2e}]"
            ),
            transform=bottom.transAxes,
            ha="right",
            va="top",
            fontsize=6.8,
            color="#374151",
        )

        for row, axis in enumerate(axes[:, column]):
            axis.set_xlim(0.0, result.realized.cfg.T)
            axis.grid(True, color=grid_color, linewidth=0.5)
            if row < 2:
                axis.tick_params(labelbottom=False)
            else:
                axis.set_xlabel(r"control interval start time $t_k$")
            axis.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

    axes[0, 0].set_ylabel(r"Hamiltonian $H(t)$")
    axes[1, 0].set_ylabel(r"full first derivative $g_k$")
    axes[2, 0].set_ylabel(r"full second derivative $R_{kk}$")
    figure.subplots_adjust(
        left=0.07, right=0.985, top=0.93, bottom=0.085, wspace=0.20, hspace=0.15
    )
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_hessian_results(
    results: list[FullDerivativeResult],
    pdf_path: Path,
    png_path: Path,
) -> None:
    configure_plotting()
    colors = ("#5B6573", "#9E4F45", "#28766B")
    figure, axes = plt.subplots(2, 3, figsize=(13.8, 5.9), dpi=220)
    absolute_bound = max(
        float(np.max(np.abs(result.hessian))) for result in results
    )
    absolute_bound = max(absolute_bound, 1.0e-12)
    norm = TwoSlopeNorm(vmin=-absolute_bound, vcenter=0.0, vmax=absolute_bound)
    all_eigenvalues = [result.eigenvalues for result in results]
    eigen_minimum = min(float(np.min(values)) for values in all_eigenvalues)
    eigen_maximum = max(float(np.max(values)) for values in all_eigenvalues)
    eigen_limits = (
        min(1.8 * eigen_minimum, -1.0e-4),
        1.08 * eigen_maximum,
    )
    images = []

    for column, (result, color) in enumerate(zip(results, colors, strict=True)):
        label = result.realized.label.replace("$", "")
        heatmap = axes[0, column]
        image = heatmap.imshow(
            result.hessian,
            origin="lower",
            extent=(0.0, result.realized.cfg.T, 0.0, result.realized.cfg.T),
            aspect="auto",
            cmap="RdBu_r",
            norm=norm,
            interpolation="nearest",
        )
        images.append(image)
        heatmap.set_title(label, pad=7, fontweight="semibold")
        heatmap.set_xlabel(r"control-interval start time $t_j$")
        if column == 0:
            heatmap.set_ylabel(r"control-interval start time $t_i$")

        spectrum = axes[1, column]
        sorted_eigenvalues = np.sort(result.eigenvalues)
        rank = np.linspace(0.0, 1.0, sorted_eigenvalues.size)
        spectrum.plot(rank, sorted_eigenvalues, color=color, linewidth=1.6)
        spectrum.axhline(0.0, color="#1F2937", linewidth=0.7)
        spectrum.set_xlim(0.0, 1.0)
        spectrum.set_yscale("symlog", linthresh=1.0e-4, linscale=1.0)
        spectrum.set_ylim(*eigen_limits)
        spectrum.grid(True, color="#E5E7EB", linewidth=0.5)
        spectrum.set_xlabel("normalized eigenvalue rank")
        if column == 0:
            spectrum.set_ylabel(r"eigenvalue of $R$")
        spectrum.text(
            0.04,
            0.94,
            (
                rf"$\lambda_{{\min}}={result.summary['hessian_min_eigenvalue_full']:.2e}$"
                "\n"
                rf"$\lambda_{{\max}}={result.summary['hessian_max_eigenvalue_full']:.2e}$"
            ),
            transform=spectrum.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            color="#374151",
        )
    figure.subplots_adjust(
        left=0.07, right=0.915, top=0.92, bottom=0.09, wspace=0.22, hspace=0.32
    )
    colorbar_axis = figure.add_axes([0.935, 0.565, 0.012, 0.31])
    colorbar = figure.colorbar(images[0], cax=colorbar_axis, orientation="vertical")
    colorbar.set_label(r"full second-derivative entry $R_{ij}$")
    colorbar.formatter = ScalarFormatter(useMathText=True)
    colorbar.formatter.set_powerlimits((-3, 3))
    colorbar.update_ticks()
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def tex_number(value: float) -> str:
    if value == 0.0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    if -2 <= exponent <= 2:
        return f"{value:.5g}"
    mantissa = value / (10.0**exponent)
    return rf"{mantissa:.3f}\times10^{{{exponent}}}"


def write_latex_table(results: list[FullDerivativeResult], path: Path) -> None:
    case_order = [result.summary for result in results]
    lines = [
        "% Generated by generate_three_case_full_derivative_results.py.",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        (
            r"Quantity & Time-only $u(t)$ & Case 1 $u(N,t)$ schedule "
            r"& Case 2 $u(N,t)$ schedule \\"
        ),
        r"\midrule",
    ]
    hamiltonian_ranges = [
        rf"$[{float(summary['hamiltonian_min']):.2f},\,{float(summary['hamiltonian_max']):.2f}]$"
        for summary in case_order
    ]
    lines.append(
        r"Hamiltonian along trajectory $H(t)$ range & "
        + " & ".join(hamiltonian_ranges)
        + r" \\"
    )
    lines.append(r"\addlinespace[1pt]")
    gradient_cells = [
        rf"${tex_number(float(summary['full_gradient_linf']))}$"
        for summary in case_order
    ]
    lines.append(
        r"Full first-derivative norm $\|g\|_\infty$ & "
        + " & ".join(gradient_cells)
        + r" \\"
    )
    spectrum_cells = [
        (
            rf"$[{tex_number(float(summary['hessian_min_eigenvalue_full']))},\,"
            rf"{tex_number(float(summary['hessian_max_eigenvalue_full']))}]$"
        )
        for summary in case_order
    ]
    lines.append(
        r"Full second-derivative spectrum $[\lambda_{\min},\lambda_{\max}]$ & "
        + " & ".join(spectrum_cells)
        + r" \\"
    )
    lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--time_checkpoint", type=Path, default=DEFAULT_TIME_CHECKPOINT
    )
    parser.add_argument(
        "--cf_checkpoint", type=Path, default=DEFAULT_CF_CHECKPOINT
    )
    parser.add_argument(
        "--der_checkpoint", type=Path, default=DEFAULT_DER_CHECKPOINT
    )
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--diagnostic_points", type=int, default=4001)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--bound_tolerance", type=float, default=1.0e-6)
    parser.add_argument("--kkt_tolerance", type=float, default=1.0e-4)
    parser.add_argument("--fd_epsilon", type=float, default=1.0e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    time_checkpoint = resolve_path(args.time_checkpoint)
    cf_checkpoint = resolve_path(args.cf_checkpoint)
    der_checkpoint = resolve_path(args.der_checkpoint)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    time_case = load_time_only_case(time_checkpoint)
    provisional_problem = assert_common_problem([time_case])
    initial_state = resistant_heavy_initial_state(provisional_problem)
    cf_case = realize_feedback_case(
        cf_checkpoint,
        "cf",
        "feedback_cf",
        r"Case 1 $u(N,t)$ schedule (CF)",
        initial_state,
        rtol=args.rtol,
        atol=args.atol,
    )
    der_case = realize_feedback_case(
        der_checkpoint,
        "der",
        "feedback_der",
        r"Case 2 $u(N,t)$ schedule (DER)",
        initial_state,
        rtol=args.rtol,
        atol=args.atol,
    )
    cases = [time_case, cf_case, der_case]
    problem = assert_common_problem(cases)
    shared_time = np.linspace(
        0.0, problem.T, args.diagnostic_points, dtype=np.float64
    )

    results: list[FullDerivativeResult] = []
    for realized in cases:
        print(f"Evaluating {realized.case_id} ...", flush=True)
        continuous = evaluate_case(
            realized,
            shared_time,
            initial_state,
            problem,
            rtol=args.rtol,
            atol=args.atol,
        )
        result = evaluate_full_derivatives(
            realized,
            continuous,
            initial_state,
            problem,
            bound_tolerance=args.bound_tolerance,
            kkt_tolerance=args.kkt_tolerance,
            fd_epsilon=args.fd_epsilon,
        )
        save_case(result, out_dir)
        results.append(result)
        print(
            f"  F={result.reduced_value:.9f} "
            f"||G||inf={result.summary['projected_gradient_linf']:.6g} "
            f"lambda_min={result.summary['hessian_min_eigenvalue_full']:.6g}",
            flush=True,
        )

    summary_rows = [result.summary for result in results]
    write_csv(out_dir / "three_case_full_derivative_summary.csv", summary_rows)
    write_latex_table(results, out_dir / "three_case_full_derivative_table.tex")
    plot_time_profiles(
        results,
        out_dir / "three_case_H_gradient_hessian_diagonal.pdf",
        out_dir / "three_case_H_gradient_hessian_diagonal.png",
    )
    plot_hessian_results(
        results,
        out_dir / "three_case_full_hessian.pdf",
        out_dir / "three_case_full_hessian.png",
    )
    metadata = {
        "definition": {
            "instantaneous_H": "H(N(t), lambda(t), u(t)) on the common DOP853 trajectory",
            "reduced_functional": "J_hat_h(u) = J_h(N_h(u), u) under RK4/ZOH",
            "full_gradient": "g = nabla_u J_hat_h(u), the RK4 control-stationarity residual",
            "full_hessian": "R = nabla_u^2 J_hat_h(u), the full derivative of g",
            "feedback_scope": (
                "the network realizes the base control vector; derivatives are "
                "with respect to the 200 physical interval controls"
            ),
        },
        "initial_state": initial_state.tolist(),
        "diagnostic_points": args.diagnostic_points,
        "rtol": args.rtol,
        "atol": args.atol,
        "kkt_tolerance": args.kkt_tolerance,
        "cases": summary_rows,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
