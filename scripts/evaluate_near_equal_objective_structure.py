#!/usr/bin/env python3
"""Near-equal-objective perturbation study for the locked PMP-Time policy.

The locked continuous control is not retrained.  A pre-registered family of
smooth, compactly supported oscillations is added on the declared singular
core.  The preferred protocol uses no compensating control: at a stationary
policy these signed perturbations change the objective only to second order.
An optional common-target protocol is retained for auditing, but is rejected
if its compensation materially changes the core diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for search_path in (ROOT, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts.generate_offgrid_policy_switching_diagnostics import (  # noqa: E402
    integrate_trajectory,
)
from scripts.generate_three_case_hamiltonian_results import (  # noqa: E402
    problem_from_config,
)
from train_paper_pmp_kkt import ProblemConfig  # noqa: E402
from tumor_problem import dynamics_numpy  # noqa: E402


PHYSICAL_SCALE = 400.0
CORE_START = 1.5
CORE_END = 8.0
PERTURB_START = 2.0
PERTURB_END = 7.5
COMPENSATION_START = 8.65
COMPENSATION_END = 9.55


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class CurvePolicy:
    cfg: ProblemConfig
    interpolator: PchipInterpolator

    def action(self, time: float, state: np.ndarray) -> float:
        del state
        value = float(self.interpolator(float(np.clip(time, 0.0, self.cfg.T))))
        return float(np.clip(value, 0.0, self.cfg.umax))


def compact_envelope(
    time: np.ndarray | float,
    start: float,
    end: float,
    *,
    power: int,
) -> np.ndarray:
    values = np.asarray(time, dtype=np.float64)
    result = np.zeros_like(values)
    mask = (values >= start) & (values <= end)
    s = (values[mask] - start) / (end - start)
    result[mask] = np.sin(math.pi * s) ** power
    return result


def oscillation(time: np.ndarray | float, cycles: int) -> np.ndarray:
    values = np.asarray(time, dtype=np.float64)
    result = np.zeros_like(values)
    mask = (values >= PERTURB_START) & (values <= PERTURB_END)
    s = (values[mask] - PERTURB_START) / (PERTURB_END - PERTURB_START)
    result[mask] = np.sin(math.pi * s) ** 4 * np.sin(
        2.0 * math.pi * float(cycles) * s
    )
    return result


def compensation_envelope(time: np.ndarray | float) -> np.ndarray:
    return compact_envelope(
        time,
        COMPENSATION_START,
        COMPENSATION_END,
        power=4,
    )


def build_policy(
    cfg: ProblemConfig,
    source_time: np.ndarray,
    source_control: np.ndarray,
    *,
    amplitude: float,
    cycles: int,
    compensation: float,
) -> CurvePolicy:
    control = (
        np.asarray(source_control, dtype=np.float64)
        + amplitude * oscillation(source_time, cycles)
        + compensation * compensation_envelope(source_time)
    )
    if float(control.min()) < -1.0e-12 or float(control.max()) > cfg.umax + 1.0e-12:
        raise ValueError("candidate leaves the control box before clipping")
    return CurvePolicy(
        cfg=cfg,
        interpolator=PchipInterpolator(source_time, control, extrapolate=False),
    )


def objective_only(
    policy: CurvePolicy,
    initial_state: np.ndarray,
    *,
    max_step: float,
    rtol: float,
    atol: float,
) -> float:
    problem = problem_from_config(policy.cfg)
    params = problem.vectors()

    def rhs(time: float, augmented: np.ndarray) -> np.ndarray:
        state = augmented[: problem.m]
        control = policy.action(time, state)
        running = float(params["beta"] @ state + problem.gamma * control)
        return np.concatenate(
            [
                dynamics_numpy(state, control, problem, params),
                np.asarray([running], dtype=np.float64),
            ]
        )

    solution = solve_ivp(
        rhs,
        (0.0, problem.T),
        np.concatenate([initial_state, np.zeros(1, dtype=np.float64)]),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    final = np.asarray(solution.y[:, -1], dtype=np.float64)
    normalized = float(params["alpha"] @ final[: problem.m] + final[problem.m])
    return PHYSICAL_SCALE * normalized


def solve_compensation(
    cfg: ProblemConfig,
    source_time: np.ndarray,
    source_control: np.ndarray,
    initial_state: np.ndarray,
    target_objective: float,
    *,
    amplitude: float,
    cycles: int,
    max_step: float,
    rtol: float,
    atol: float,
) -> tuple[float, float, int]:
    evaluations = 0

    def residual(compensation: float) -> float:
        nonlocal evaluations
        policy = build_policy(
            cfg,
            source_time,
            source_control,
            amplitude=amplitude,
            cycles=cycles,
            compensation=compensation,
        )
        evaluations += 1
        return objective_only(
            policy,
            initial_state,
            max_step=max_step,
            rtol=rtol,
            atol=atol,
        ) - target_objective

    lower, upper = 0.0, 1.0e-4
    f_lower, f_upper = residual(lower), residual(upper)
    for _ in range(10):
        if f_lower == 0.0:
            return lower, target_objective, evaluations
        if f_upper == 0.0:
            return upper, target_objective, evaluations
        if f_lower * f_upper < 0.0:
            break
        upper *= 2.0
        if upper > 0.05:
            break
        f_upper = residual(upper)
    else:
        raise RuntimeError(
            f"failed to bracket compensation for amplitude={amplitude}, "
            f"cycles={cycles}: residuals=({f_lower},{f_upper})"
        )
    coefficient = float(
        brentq(
            residual,
            lower,
            upper,
            xtol=1.0e-12,
            rtol=1.0e-12,
            maxiter=32,
        )
    )
    achieved = target_objective + residual(coefficient)
    return coefficient, achieved, evaluations


def summarize_trajectory(
    result: Any,
    *,
    strict_mask: np.ndarray,
    source_time: np.ndarray,
    source_control: np.ndarray,
    amplitude: float,
    cycles: int,
    compensation: float,
) -> dict[str, float]:
    core = (result.time >= CORE_START) & (result.time < CORE_END)
    core_control = np.asarray(result.control[core], dtype=np.float64)
    row: dict[str, float] = {
        "objective": PHYSICAL_SCALE * float(result.normalized_objective),
        "control_min": float(np.min(result.control)),
        "control_max": float(np.max(result.control)),
        "control_total_variation": float(np.sum(np.abs(np.diff(result.control)))),
        "control_rms_change": float(
            np.sqrt(
                np.mean(
                    (
                        result.control
                        - np.interp(result.time, source_time, source_control)
                    )
                    ** 2
                )
            )
        ),
        "amplitude": float(amplitude),
        "cycles": float(cycles),
        "compensation": float(compensation),
    }
    row["core_box_clipping_fraction"] = float(
        np.mean((core_control <= 1.0e-12) | (core_control >= 3.0 - 1.0e-12))
    )
    diagnostic_names = {
        "H_u": "psi",
        "dH_u_dt": "dot_psi",
        "d2H_u_dt2": "ddot_psi",
    }
    for key, values in result.quantities.items():
        scaled = PHYSICAL_SCALE * np.asarray(values[strict_mask], dtype=np.float64)
        name = diagnostic_names.get(key, key)
        row[f"{name}_rms"] = float(np.sqrt(np.mean(scaled**2)))
        row[f"{name}_max_abs"] = float(np.max(np.abs(scaled)))
    return row


def write_outputs(
    out_dir: Path,
    rows: list[dict[str, Any]],
    trajectories: dict[str, Any],
    baseline_objective: float,
    metadata: dict[str, Any],
) -> None:
    fieldnames = list(rows[0].keys())
    with (out_dir / "matched_j_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        **metadata,
        "baseline_objective": baseline_objective,
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(rows)))
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.2), constrained_layout=True)
    t = trajectories[rows[0]["candidate"]].time
    core = (t >= CORE_START) & (t < CORE_END)
    for color, row in zip(colors, rows):
        result = trajectories[row["candidate"]]
        label = (
            "locked policy"
            if row["candidate"] == "locked baseline"
            else rf"$a={row['amplitude']:+.4f}$"
        )
        axes[0, 0].plot(t[core], result.control[core], color=color, lw=1.3, label=label)
    axes[0, 0].set_title("Control perturbations on the singular core")
    axes[0, 0].set_xlabel(r"time $t$")
    axes[0, 0].set_ylabel(r"$u(t)$")
    axes[0, 0].legend(fontsize=8, ncol=2)

    matched_rows = rows[1:]
    candidates = np.arange(len(matched_rows))
    delta_j = np.asarray(
        [row["objective_relative_difference"] for row in matched_rows]
    )
    axes[0, 1].bar(candidates, delta_j, color=colors[1:])
    axes[0, 1].axhline(0.0, color="0.35", lw=0.8)
    axes[0, 1].set_title("Near-equal physical objective")
    axes[0, 1].set_ylabel(r"$(J-J_0)/J_0$")
    axes[0, 1].set_xticks(
        candidates,
        [rf"$a={row['amplitude']:+.4f}$" for row in matched_rows],
        rotation=30,
        ha="right",
    )

    metric_keys = ["psi_rms", "dot_psi_rms", "ddot_psi_rms"]
    metric_labels = [r"$\psi$", r"$\dot\psi$", r"$\ddot\psi$"]
    width = 0.22
    all_candidates = np.arange(len(rows))
    for offset, (key, label) in enumerate(zip(metric_keys, metric_labels)):
        axes[1, 0].bar(
            all_candidates + (offset - 1) * width,
            [row[key] for row in rows],
            width=width,
            label=label,
        )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Strict off-grid residual RMS")
    axes[1, 0].set_ylabel("physical magnitude")
    axes[1, 0].set_xticks(
        all_candidates,
        [
            "locked" if row["candidate"] == "locked baseline" else rf"$a={row['amplitude']:+.4f}$"
            for row in rows
        ],
        rotation=30,
        ha="right",
    )
    axes[1, 0].legend()

    ordered_rows = sorted(rows, key=lambda row: row["amplitude"])
    axes[1, 1].plot(
        [row["amplitude"] for row in ordered_rows],
        [row["ddot_psi_rms"] for row in ordered_rows],
        marker="o",
        color="#c44e13",
    )
    axes[1, 1].axhline(rows[0]["ddot_psi_rms"], color="0.35", ls="--", lw=1.0)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title(r"Near-equal objectives, different $\ddot\psi$")
    axes[1, 1].set_xlabel("signed perturbation amplitude")
    axes[1, 1].set_ylabel(r"RMS $|\ddot\psi|$")
    fig.savefig(out_dir / "matched_j_structure.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "matched_j_structure.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    protocol_description = (
        "No compensation is applied; the pre-registered perturbations remain "
        "within 5e-4 physical objective units of the locked policy."
        if metadata["protocol"] == "uncompensated_near_objective"
        else "A terminal compensation matches every perturbation to J0+1."
    )
    lines = [
        "# Near-equal-objective structure study",
        "",
        "The locked PMP-Time checkpoint is not retrained. A pre-registered signed family of smooth oscillations is supported inside [2.0,7.5]. "
        + protocol_description,
        "",
        f"- Baseline physical objective: {baseline_objective:.12f}",
        f"- Reference objective: {metadata['target_objective']:.12f}",
        f"- Final grid: T_{metadata['final_q']} ({metadata['dense_points']} points)",
        "- Strict diagnostic set: final grid with every T8 point excluded",
        "",
        "| Candidate | J-J0 | relative J change | control RMS change | psi RMS | psi-dot RMS | psi-ddot RMS | psi-ddot ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {row['objective']-metadata['target_objective']:.3e} | "
            f"{row['objective_relative_difference']:.3e} | "
            f"{row['control_rms_change']:.3e} | {row['psi_rms']:.3e} | "
            f"{row['dot_psi_rms']:.3e} | {row['ddot_psi_rms']:.3e} | "
            f"{row['ddot_psi_rms_ratio_to_locked']:.3e} |"
        )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-npz", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--final-q", type=int, default=32)
    parser.add_argument("--screen-steps", type=int, default=3200)
    parser.add_argument(
        "--amplitudes",
        type=float,
        nargs="+",
        default=[0.0, 0.0025, -0.0025, 0.005, -0.005],
    )
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--no-compensation", action="store_true")
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    args = parser.parse_args()

    baseline_npz = args.baseline_npz.expanduser().resolve()
    baseline_summary = args.baseline_summary.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    summary = json.loads(baseline_summary.read_text(encoding="utf-8"))
    cfg = ProblemConfig(**summary["problem"])
    loaded = np.load(baseline_npz)
    source_time = np.asarray(loaded["t"], dtype=np.float64)
    source_control = np.asarray(loaded["nominal__u"], dtype=np.float64)
    initial_state = np.full(cfg.m, cfg.n0, dtype=np.float64)
    baseline_policy = build_policy(
        cfg,
        source_time,
        source_control,
        amplitude=0.0,
        cycles=args.cycles,
        compensation=0.0,
    )
    objective_step = cfg.T / float(args.screen_steps)
    baseline_objective = objective_only(
        baseline_policy,
        initial_state,
        max_step=objective_step,
        rtol=args.rtol,
        atol=args.atol,
    )
    screen_baseline_objective = baseline_objective

    target_objective = (
        baseline_objective if args.no_compensation else baseline_objective + 1.0
    )
    coefficients: dict[float, tuple[float, float, int]] = {}
    for amplitude in args.amplitudes:
        if args.no_compensation:
            candidate = build_policy(
                cfg,
                source_time,
                source_control,
                amplitude=amplitude,
                cycles=args.cycles,
                compensation=0.0,
            )
            achieved = objective_only(
                candidate,
                initial_state,
                max_step=objective_step,
                rtol=args.rtol,
                atol=args.atol,
            )
            coefficients[amplitude] = (0.0, achieved, 1)
        else:
            coefficients[amplitude] = solve_compensation(
                cfg,
                source_time,
                source_control,
                initial_state,
                target_objective,
                amplitude=amplitude,
                cycles=args.cycles,
                max_step=objective_step,
                rtol=args.rtol,
                atol=args.atol,
            )
        print(
            json.dumps(
                {
                    "amplitude": amplitude,
                    "compensation": coefficients[amplitude][0],
                    "objective": coefficients[amplitude][1],
                    "objective_evaluations": coefficients[amplitude][2],
                }
            ),
            flush=True,
        )

    dense_points = 800 * args.final_q + 1
    dense_time = np.linspace(0.0, cfg.T, dense_points, dtype=np.float64)
    strict_mask = (
        (dense_time >= CORE_START)
        & (dense_time < CORE_END)
        & (np.arange(dense_points) % (args.final_q // 8) != 0)
    )
    if args.final_q % 8:
        raise ValueError("final-q must be divisible by 8")
    if args.final_q == 32 and int(strict_mask.sum()) != 12_480:
        raise RuntimeError(
            f"strict T32\\T8 core must contain 12,480 points, found {strict_mask.sum()}"
        )

    specifications = [("locked baseline", 0.0, args.cycles, 0.0)] + [
        (
            (
                "matched reference"
                if amplitude == 0.0 and not args.no_compensation
                else f"amplitude {amplitude:+.4f}"
            ),
            amplitude,
            args.cycles,
            coefficients[amplitude][0],
        )
        for amplitude in args.amplitudes
    ]
    rows: list[dict[str, Any]] = []
    trajectories: dict[str, Any] = {}
    for name, amplitude, cycles, compensation in specifications:
        policy = build_policy(
            cfg,
            source_time,
            source_control,
            amplitude=amplitude,
            cycles=cycles,
            compensation=compensation,
        )
        result = integrate_trajectory(
            policy,
            initial_state,
            dense_time,
            rtol=args.rtol,
            atol=args.atol,
            max_step=cfg.T / (dense_points - 1),
        )
        row: dict[str, Any] = {"candidate": name}
        row.update(
            summarize_trajectory(
                result,
                strict_mask=strict_mask,
                source_time=source_time,
                source_control=source_control,
                amplitude=amplitude,
                cycles=cycles,
                compensation=compensation,
            )
        )
        row["objective_difference"] = row["objective"] - baseline_objective
        rows.append(row)
        trajectories[name] = result
        print(json.dumps(row), flush=True)

    baseline_row = rows[0]
    baseline_objective = float(baseline_row["objective"])
    if args.no_compensation:
        target_objective = baseline_objective
    for row in rows:
        row["objective_difference"] = float(
            row["objective"] - baseline_objective
        )
        row["objective_relative_difference"] = float(
            (row["objective"] - baseline_objective) / baseline_objective
        )
        for key in ("psi_rms", "dot_psi_rms", "ddot_psi_rms"):
            row[f"{key}_ratio_to_locked"] = float(
                row[key] / baseline_row[key]
            )

    for row in rows[1:]:
        mismatch = abs(float(row["objective"]) - target_objective)
        tolerance = 5.0e-4 if args.no_compensation else 1.0e-4
        if mismatch > tolerance:
            raise RuntimeError(
                f"{row['candidate']} objective mismatch {mismatch:.3e} exceeds {tolerance:.1e}"
            )
        if float(row["core_box_clipping_fraction"]) != 0.0:
            raise RuntimeError(f"{row['candidate']} clips inside the diagnostic core")

    metadata = {
        "baseline_npz": str(baseline_npz),
        "baseline_npz_sha256": sha256(baseline_npz),
        "baseline_summary": str(baseline_summary),
        "baseline_summary_sha256": sha256(baseline_summary),
        "locked_checkpoint": summary.get("checkpoint"),
        "locked_checkpoint_sha256": summary.get("checkpoint_sha256"),
        "problem": summary["problem"],
        "core_window": [CORE_START, CORE_END],
        "physical_scale": PHYSICAL_SCALE,
        "amplitudes": args.amplitudes,
        "cycles": args.cycles,
        "target_objective": target_objective,
        "screen_reference_objective": screen_baseline_objective,
        "protocol": (
            "uncompensated_near_objective"
            if args.no_compensation
            else "terminal_compensated_common_target"
        ),
        "perturbation_support": [PERTURB_START, PERTURB_END],
        "compensation_support": (
            None
            if args.no_compensation
            else [COMPENSATION_START, COMPENSATION_END]
        ),
        "final_q": args.final_q,
        "dense_points": dense_points,
        "strict_count": int(strict_mask.sum()),
        "screen_steps": args.screen_steps,
        "strict_exclusion_grid": "T8",
    }
    write_outputs(out_dir, rows, trajectories, baseline_objective, metadata)


if __name__ == "__main__":
    main()
