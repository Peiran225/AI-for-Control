#!/usr/bin/env python3
"""Generate the teacher-requested two-state comparison for all three cases.

The two fixed initial states are evaluated under the selected native-n=800
time-only and feedback checkpoints.  For every realized control vector, this
script recomputes the state/costate trajectory, the full reduced gradient, and
the dense reduced Hessian.  Each case receives the same six-panel figure:

1. control,
2. total population,
3. phenotype profiles at the early and late control transitions,
4. instantaneous Hamiltonian,
5. full reduced first derivative versus control time, and
6. Hessian-column ell-2 norm versus control time.

The final panel uses ||R[:,j]||_2 for R = d^2 Jhat_h / d u^2.  The scalar
Hessian summary is ||lambda(R)||_2, which equals ||R||_F for the symmetrized
Hessian.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import ScalarFormatter  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for search_path in (ROOT, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from generate_teacher_facing_three_case_n800 import (  # noqa: E402
    evaluate_reduced_derivatives,
    load_time_case,
    realize_feedback_n800,
)
from generate_three_case_full_derivative_results import (  # noqa: E402
    continuous_interval_gradient,
    instantaneous_hamiltonian,
)
from generate_three_case_hamiltonian_results import (  # noqa: E402
    RealizedCase,
    assert_common_problem,
    evaluate_case,
    resistant_heavy_initial_state,
)


DEFAULT_TIME = (
    ROOT
    / "outputs/teacher_free_n800_strict_20260720/"
    "network_lbfgs_after_learn_tau/selected_checkpoint.pt"
)
DEFAULT_CF = (
    ROOT
    / "outputs/feedback_n800_state_refinement_20260720/"
    "cf_zero_strong/best_feedback_section5_full_gradient.pt"
)
DEFAULT_DER = (
    ROOT
    / "outputs/feedback_n800_state_refinement_20260720/"
    "der_moderate/best_feedback_section5_full_gradient.pt"
)
DEFAULT_OUT = ROOT / "outputs/two_state_three_case_main_20260721"

CASE_ORDER = ("time_only", "feedback_cf", "feedback_der")
CASE_LABELS = {
    "time_only": r"Transformer $u(t)$",
    "feedback_cf": r"Case 1 $u(N,t)$",
    "feedback_der": r"Case 2 $u(N,t)$",
}
CASE_STEMS = {
    "time_only": "time_only_two_state_main",
    "feedback_cf": "case1_two_state_main",
    "feedback_der": "case2_two_state_main",
}
STATE_ORDER = ("nominal", "resistant_heavy")
STATE_LABELS = {
    "nominal": "Nominal",
    "resistant_heavy": "Resistant-heavy",
}
STATE_STYLE = {
    "nominal": {"color": "#244A74", "linestyle": "-", "linewidth": 1.55},
    "resistant_heavy": {
        "color": "#C15B32",
        "linestyle": "--",
        "linewidth": 1.55,
    },
}


def resolve(path: Path) -> Path:
    candidate = path.expanduser()
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "stixsans",
            "font.size": 8.4,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "axes.grid": True,
            "grid.color": "#D8DDE2",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.78,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def switching_indices(case: RealizedCase) -> tuple[int, int]:
    changes = np.diff(case.controls)
    transition_times = case.breakpoints[1:-1]
    early_candidates = np.flatnonzero(transition_times < 0.5 * case.cfg.T)
    late_candidates = np.flatnonzero(transition_times >= 0.5 * case.cfg.T)
    if not early_candidates.size or not late_candidates.size:
        raise RuntimeError(f"{case.case_id}: no early/late switching candidates")
    early = int(early_candidates[np.argmin(changes[early_candidates])])
    late = int(late_candidates[np.argmax(changes[late_candidates])])
    return early, late


def realize_all_cases(
    time_checkpoint: Path,
    cf_checkpoint: Path,
    der_checkpoint: Path,
    n: int,
    states: dict[str, np.ndarray],
    *,
    rtol: float,
    atol: float,
) -> tuple[dict[str, dict[str, RealizedCase]], Any]:
    time_case, _ = load_time_case(time_checkpoint, n)
    provisional_problem = assert_common_problem([time_case])
    realized: dict[str, dict[str, RealizedCase]] = {
        "time_only": {state_id: time_case for state_id in STATE_ORDER}
    }
    realized["feedback_cf"] = {}
    realized["feedback_der"] = {}
    for state_id in STATE_ORDER:
        cf_case, _ = realize_feedback_n800(
            cf_checkpoint,
            "cf",
            "feedback_cf",
            CASE_LABELS["feedback_cf"],
            states[state_id],
            n,
            rtol=rtol,
            atol=atol,
        )
        der_case, _ = realize_feedback_n800(
            der_checkpoint,
            "der",
            "feedback_der",
            CASE_LABELS["feedback_der"],
            states[state_id],
            n,
            rtol=rtol,
            atol=atol,
        )
        realized["feedback_cf"][state_id] = cf_case
        realized["feedback_der"][state_id] = der_case
    all_cases = [realized[case_id][state_id] for case_id in CASE_ORDER for state_id in STATE_ORDER]
    problem = assert_common_problem(all_cases)
    if problem.m != provisional_problem.m:
        raise RuntimeError("case dimensions changed during realization")
    return realized, problem


def evaluate_all(
    realized: dict[str, dict[str, RealizedCase]],
    states: dict[str, np.ndarray],
    problem: Any,
    out_dir: Path,
    *,
    diagnostic_points: int,
    rtol: float,
    atol: float,
    bound_tolerance: float,
    kkt_tolerance: float,
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], dict[str, Any]]:
    shared_time = np.linspace(0.0, problem.T, diagnostic_points, dtype=np.float64)
    results: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    summaries: dict[str, Any] = {}
    for case_id in CASE_ORDER:
        results[case_id] = {}
        summaries[case_id] = {}
        for state_id in STATE_ORDER:
            case = realized[case_id][state_id]
            print(f"[{case_id}/{state_id}] state, costate, H", flush=True)
            evaluated = evaluate_case(
                case,
                shared_time,
                states[state_id],
                problem,
                rtol=rtol,
                atol=atol,
            )
            hamiltonian = instantaneous_hamiltonian(evaluated, problem)
            print(f"[{case_id}/{state_id}] full gradient and dense Hessian", flush=True)
            derivative_summary, derivative_arrays = evaluate_reduced_derivatives(
                case,
                states[state_id],
                bound_tolerance,
                kkt_tolerance,
            )
            hessian = derivative_arrays["full_hessian"]
            eigenvalues = derivative_arrays["hessian_eigenvalues"]
            dt = float(case.cfg.T / case.cfg.n)
            interval_hu = derivative_arrays["full_gradient"] / dt
            continuous_hu_integral = continuous_interval_gradient(evaluated)
            hu_integral_error = float(
                np.max(
                    np.abs(
                        continuous_hu_integral - derivative_arrays["full_gradient"]
                    )
                )
            )
            if hu_integral_error > 1.0e-7:
                raise RuntimeError(
                    f"{case_id}/{state_id}: full H_u integral check failed "
                    f"({hu_integral_error:.3e})"
                )
            column_l2 = np.linalg.norm(hessian, axis=0)
            eigenvalue_l2 = float(np.linalg.norm(eigenvalues))
            if not np.isclose(eigenvalue_l2, np.linalg.norm(hessian), rtol=2.0e-12, atol=2.0e-12):
                raise RuntimeError(f"{case_id}/{state_id}: Hessian norm identity failed")
            early, late = switching_indices(case)
            early_node = early + 1
            late_node = late + 1
            data = {
                "breakpoints": case.breakpoints,
                "interval_time": 0.5 * (case.breakpoints[:-1] + case.breakpoints[1:]),
                "interval_control": case.controls,
                "continuous_time": evaluated.time,
                "continuous_state": evaluated.state,
                "node_states": evaluated.node_states,
                "total_population": evaluated.state.sum(axis=1),
                "continuous_hamiltonian": hamiltonian,
                "continuous_Hu_partial": np.asarray(
                    evaluated.quantities["psi"], dtype=np.float64
                ),
                "full_gradient": derivative_arrays["full_gradient"],
                "interval_averaged_full_Hu": interval_hu,
                "full_hessian": hessian,
                "hessian_column_l2": column_l2,
                "hessian_eigenvalues": eigenvalues,
                "early_profile": evaluated.node_states[early_node],
                "late_profile": evaluated.node_states[late_node],
            }
            results[case_id][state_id] = data
            switch_times = {
                "early": float(case.breakpoints[early_node]),
                "late": float(case.breakpoints[late_node]),
                "early_adjacent_control_change": float(np.diff(case.controls)[early]),
                "late_adjacent_control_change": float(np.diff(case.controls)[late]),
            }
            summary = {
                "case_id": case_id,
                "case_label": CASE_LABELS[case_id].replace("$", ""),
                "state_id": state_id,
                "initial_state": states[state_id],
                "switching": switch_times,
                "hamiltonian_min": float(hamiltonian.min()),
                "hamiltonian_max": float(hamiltonian.max()),
                "hessian_eigenvalue_l2": eigenvalue_l2,
                "hessian_column_l2_min": float(column_l2.min()),
                "hessian_column_l2_max": float(column_l2.max()),
                "full_Hu_integral_check_max_abs": hu_integral_error,
                **derivative_summary,
            }
            summaries[case_id][state_id] = summary
            case_dir = out_dir / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(case_dir / f"{state_id}.npz", **data)
            (case_dir / f"{state_id}.json").write_text(
                json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"[{case_id}/{state_id}] ||lambda(R)||2={eigenvalue_l2:.9f} "
                f"column-L2=[{column_l2.min():.6g},{column_l2.max():.6g}]",
                flush=True,
            )
    return results, summaries


def padded_limits(low: float, high: float, fraction: float = 0.06) -> tuple[float, float]:
    span = max(high - low, 1.0e-12)
    return low - fraction * span, high + fraction * span


def global_limits(results: dict[str, dict[str, dict[str, np.ndarray]]]) -> dict[str, tuple[float, float]]:
    blocks = [results[case_id][state_id] for case_id in CASE_ORDER for state_id in STATE_ORDER]
    total_high = max(float(block["total_population"].max()) for block in blocks)
    profile_high = max(
        float(max(block["early_profile"].max(), block["late_profile"].max()))
        for block in blocks
    )
    h_low = min(float(block["continuous_hamiltonian"].min()) for block in blocks)
    h_high = max(float(block["continuous_hamiltonian"].max()) for block in blocks)
    g_bound = 1.06 * max(
        float(np.abs(block["interval_averaged_full_Hu"]).max()) for block in blocks
    )
    r_high = 1.06 * max(float(block["hessian_column_l2"].max()) for block in blocks)
    return {
        "control": (-0.05, 3.08),
        "total": (0.0, 1.025 * total_high),
        "profile": (0.0, 1.06 * profile_high),
        "hamiltonian": padded_limits(h_low, h_high),
        "gradient": (-g_bound, g_bound),
        "hessian_column": (0.0, r_high),
    }


def state_vector_text(label: str, state: np.ndarray) -> str:
    return f"{label} N(0) = (" + ", ".join(f"{value:.1f}" for value in state) + ")"


def plot_case(
    case_id: str,
    results: dict[str, dict[str, dict[str, np.ndarray]]],
    summaries: dict[str, Any],
    states: dict[str, np.ndarray],
    limits: dict[str, tuple[float, float]],
    out_dir: Path,
) -> None:
    configure_style()
    figure, axes = plt.subplots(3, 2, figsize=(15.25, 7.05), dpi=220)
    control_axis, total_axis = axes[0]
    profile_axis, h_axis = axes[1]
    gradient_axis, hessian_axis = axes[2]
    figure.suptitle(
        CASE_LABELS[case_id] + ": comparison of two fixed initial states",
        x=0.5,
        y=0.987,
        fontsize=13.0,
        fontweight="semibold",
    )
    figure.text(
        0.5,
        0.948,
        state_vector_text("Nominal", states["nominal"]),
        ha="center",
        va="center",
        fontsize=7.5,
        color="#26313B",
    )
    figure.text(
        0.5,
        0.921,
        state_vector_text("Resistant-heavy", states["resistant_heavy"]),
        ha="center",
        va="center",
        fontsize=7.5,
        color="#26313B",
    )

    for state_id in STATE_ORDER:
        data = results[case_id][state_id]
        style = STATE_STYLE[state_id]
        control_axis.step(
            data["breakpoints"][:-1],
            data["interval_control"],
            where="post",
            label=STATE_LABELS[state_id],
            zorder=3 if state_id == "resistant_heavy" else 2,
            **style,
        )
        total_axis.plot(
            data["continuous_time"], data["total_population"], **style
        )
        h_axis.plot(
            data["continuous_time"], data["continuous_hamiltonian"], **style
        )
        gradient_axis.plot(
            data["interval_time"], data["interval_averaged_full_Hu"], **style
        )
        hessian_axis.plot(
            data["interval_time"], data["hessian_column_l2"], **style
        )

        switch = summaries[case_id][state_id]["switching"]
        profile_axis.plot(
            np.arange(1, data["early_profile"].size + 1),
            data["early_profile"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            marker="o",
            markersize=2.7,
            markevery=2,
            label=f"{STATE_LABELS[state_id]}, early t={switch['early']:.4f}",
        )
        profile_axis.plot(
            np.arange(1, data["late_profile"].size + 1),
            data["late_profile"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            marker="s",
            markersize=2.6,
            markevery=2,
            alpha=0.82,
            label=f"{STATE_LABELS[state_id]}, late t={switch['late']:.4f}",
        )

    nominal_switch = summaries[case_id]["nominal"]["switching"]
    for time_value in (nominal_switch["early"], nominal_switch["late"]):
        control_axis.axvline(time_value, color="#747B82", lw=0.7, ls=":", zorder=1)
    control_difference = float(
        np.max(
            np.abs(
                results[case_id]["nominal"]["interval_control"]
                - results[case_id]["resistant_heavy"]["interval_control"]
            )
        )
    )
    control_axis.text(
        0.985,
        0.06,
        rf"$\max |u_R(t)-u_N(t)|={control_difference:.4g}$",
        transform=control_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        color="#4B5560",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
    )

    control_label = r"control $u(t)$" if case_id == "time_only" else r"control $u(N(t),t)$"
    control_axis.set_title("(a) Control")
    control_axis.set_ylabel(control_label)
    control_axis.set_ylim(*limits["control"])
    control_axis.legend(frameon=False, loc="lower center", ncol=2)

    total_axis.set_title(r"(b) Total population $\sum_i N_i(t)$")
    total_axis.set_ylabel(r"$\sum_i N_i(t)$")
    total_axis.set_ylim(*limits["total"])

    profile_axis.set_title("(c) Phenotype profiles at the two switching points")
    profile_axis.set_xlabel("phenotype index i")
    profile_axis.set_ylabel(r"$N_i(t_s)$")
    profile_axis.set_xlim(1.0, float(states["nominal"].size))
    profile_axis.set_ylim(*limits["profile"])
    profile_axis.legend(frameon=False, loc="upper right", ncol=2, columnspacing=1.0)

    h_axis.set_title(r"(d) Instantaneous Hamiltonian $H(t)$")
    h_axis.set_ylabel(r"$H(t)$")
    h_axis.set_ylim(*limits["hamiltonian"])

    gradient_axis.set_title(r"(e) Interval-averaged full first derivative $H_u$")
    gradient_axis.set_ylabel(r"$H_{u,j}=g_j/\Delta t$")
    gradient_axis.set_ylim(*limits["gradient"])
    gradient_axis.axhline(0.0, color="#262B30", lw=0.65)

    hessian_axis.set_title(r"(f) Reduced-Hessian column $\ell_2$ norm")
    hessian_axis.set_ylabel(r"$\|R_{:,j}\|_2$")
    hessian_axis.set_ylim(*limits["hessian_column"])
    eigen_lines = []
    for state_id in STATE_ORDER:
        value = summaries[case_id][state_id]["hessian_eigenvalue_l2"]
        eigen_lines.append(
            rf"{STATE_LABELS[state_id]}: $\|\lambda(R)\|_2={value:.6f}$"
        )
    hessian_axis.text(
        0.985,
        0.94,
        "\n".join(eigen_lines),
        transform=hessian_axis.transAxes,
        ha="right",
        va="top",
        fontsize=7.1,
        color="#343B43",
        bbox={"facecolor": "white", "edgecolor": "#D3D7DB", "alpha": 0.90, "pad": 2.0},
    )

    time_axes = (control_axis, total_axis, h_axis, gradient_axis, hessian_axis)
    for axis in time_axes:
        axis.set_xlim(0.0, 10.0)
        axis.set_xlabel("time t")
    for axis in (gradient_axis, hessian_axis):
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        axis.yaxis.set_major_formatter(formatter)

    state_handles = [
        Line2D(
            [0],
            [0],
            color=STATE_STYLE[state_id]["color"],
            linestyle=STATE_STYLE[state_id]["linestyle"],
            linewidth=1.7,
            label=STATE_LABELS[state_id],
        )
        for state_id in STATE_ORDER
    ]
    figure.legend(
        handles=state_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=2,
        frameon=False,
        fontsize=8.0,
    )
    figure.text(
        0.985,
        0.017,
        "Switching points: largest downward/upward adjacent control changes",
        ha="right",
        va="bottom",
        fontsize=7.0,
        color="#59616A",
    )
    figure.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.105,
        top=0.875,
        wspace=0.20,
        hspace=0.43,
    )
    stem = out_dir / CASE_STEMS[case_id]
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-checkpoint", type=Path, default=DEFAULT_TIME)
    parser.add_argument("--cf-checkpoint", type=Path, default=DEFAULT_CF)
    parser.add_argument("--der-checkpoint", type=Path, default=DEFAULT_DER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--diagnostic-points", type=int, default=4001)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--bound-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--kkt-tolerance", type=float, default=1.0e-4)
    args = parser.parse_args()

    out_dir = resolve(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    time_checkpoint = resolve(args.time_checkpoint)
    cf_checkpoint = resolve(args.cf_checkpoint)
    der_checkpoint = resolve(args.der_checkpoint)
    time_case, _ = load_time_case(time_checkpoint, args.n)
    problem = assert_common_problem([time_case])
    nominal = np.full(problem.m, problem.n0, dtype=np.float64)
    resistant = resistant_heavy_initial_state(problem)
    states = {"nominal": nominal, "resistant_heavy": resistant}
    realized, problem = realize_all_cases(
        time_checkpoint,
        cf_checkpoint,
        der_checkpoint,
        args.n,
        states,
        rtol=args.rtol,
        atol=args.atol,
    )
    results, summaries = evaluate_all(
        realized,
        states,
        problem,
        out_dir,
        diagnostic_points=args.diagnostic_points,
        rtol=args.rtol,
        atol=args.atol,
        bound_tolerance=args.bound_tolerance,
        kkt_tolerance=args.kkt_tolerance,
    )
    limits = global_limits(results)
    for case_id in CASE_ORDER:
        plot_case(case_id, results, summaries, states, limits, out_dir)

    metadata = {
        "definition": {
            "nominal_initial_state": nominal,
            "resistant_heavy_initial_state": resistant,
            "resistant_heavy_formula": "N_i(0)=10[1+0.10(2(i-1)/20-1)]",
            "switching_points": "largest downward and upward adjacent control changes in the first and second halves of the horizon",
            "instantaneous_hamiltonian": "H(N(t),lambda(t),u(t)) from segmented DOP853 state/costate trajectories",
            "full_first_derivative": "g_j=d Jhat_h/d u_j with the complete RK4 N_h=N_h(u) dependence; the time curve is interval-averaged H_u=g_j/dt",
            "full_second_derivative": "R=d^2 Jhat_h/d u^2 on the n=800 RK4/ZOH transcription",
            "hessian_eigenvalue_l2": "||lambda(R)||_2=||R||_F for the symmetrized Hessian",
            "hessian_time_curve": "r_j=||R[:,j]||_2 at interval midpoint t_j",
        },
        "checkpoints": {
            "time_only": time_checkpoint,
            "feedback_cf": cf_checkpoint,
            "feedback_der": der_checkpoint,
        },
        "common_problem": {
            "n": args.n,
            "dt": problem.T / args.n,
            "diagnostic_points": args.diagnostic_points,
            "rtol": args.rtol,
            "atol": args.atol,
        },
        "plot_limits": limits,
        "summaries": summaries,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(json_safe(metadata), indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
