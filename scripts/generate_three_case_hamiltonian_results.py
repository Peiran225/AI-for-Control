#!/usr/bin/env python3
"""Generate uniform continuous-PMP diagnostics for the three requested cases.

The three cases are evaluated on the same resistant-heavy initial state

    N_i(0) = 10 [1 + 0.10 (2 x_i - 1)],  x_i = (i-1)/(m-1).

For each realized zero-order-hold control, this script integrates the physical
state forward and the standard PMP costate backward with DOP853.  It then
evaluates, on one shared time grid,

    H_u = psi,       d H_u / dt = dot(psi),
    d^2 H_u / dt^2 = ddot(psi) = A + B u.

In particular, the RK4 interval derivative ``hat(Psi)`` used by the training
code is not substituted for the continuous switching function ``psi`` here.
"""

from __future__ import annotations

import argparse
import csv
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
import torch
from scipy.integrate import solve_ivp


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for search_path in (ROOT, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from evaluate_feedback_section5 import (  # noqa: E402
    configured_state_mode,
    load_feedback_checkpoint,
)
from make_canonical_report_figures import strict_singular_quantities  # noqa: E402
from train_feedback_section5 import load_operational_time_control  # noqa: E402
from train_paper_pmp_kkt import ProblemConfig  # noqa: E402
from tumor_problem import TumorProblem, dH_dN_numpy, dynamics_numpy  # noqa: E402


DEFAULT_TIME_CHECKPOINT = (
    ROOT / "paper_runs/smoothness_weight_sweep/w3/seed_4/best_pmp_kkt.pt"
)
DEFAULT_FORMAL_ROOT = (
    ROOT / "outputs/feedback_section5_round3_formal_finalists_20260718"
)
DEFAULT_CF_CHECKPOINT = (
    DEFAULT_FORMAL_ROOT
    / "cf_feedback_w_one_seed0/train/best_feedback_section5.pt"
)
DEFAULT_DER_CHECKPOINT = (
    DEFAULT_FORMAL_ROOT
    / "der_feedback_w_one_seed0/train/best_feedback_section5.pt"
)
DEFAULT_OUT_DIR = ROOT / "outputs/three_case_hamiltonian_results"


QUANTITIES: tuple[tuple[str, str, str], ...] = (
    ("H_u", r"$H_u=\psi$", r"$H_u$"),
    ("dH_u_dt", r"$dH_u/dt=\dot{\psi}$", r"$dH_u/dt$"),
    ("d2H_u_dt2", r"$d^2H_u/dt^2=\ddot{\psi}$", r"$d^2H_u/dt^2$"),
)


@dataclass
class RealizedCase:
    case_id: str
    label: str
    checkpoint: Path
    cfg: ProblemConfig
    breakpoints: np.ndarray
    controls: np.ndarray
    feedback_model: torch.nn.Module | None = None


@dataclass
class EvaluatedCase:
    specification: RealizedCase
    time: np.ndarray
    state: np.ndarray
    costate: np.ndarray
    control: np.ndarray
    node_states: np.ndarray
    quantities: dict[str, np.ndarray]
    identity_errors: dict[str, float]
    feedback_closure_error: float | None


def resolve_path(path: Path) -> Path:
    """Resolve a CLI path relative to the repository root when needed."""

    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate.resolve()


def problem_from_config(cfg: ProblemConfig) -> TumorProblem:
    return TumorProblem(
        T=float(cfg.T),
        m=int(cfg.m),
        umax=float(cfg.umax),
        beta=float(cfg.beta),
        alpha=float(cfg.alpha),
        gamma=float(cfg.gamma),
        n0=float(cfg.n0),
        m_suppression=float(cfg.m_suppression),
    )


def load_time_only_case(path: Path) -> RealizedCase:
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ProblemConfig(**checkpoint["problem"])
    controls = (
        load_operational_time_control(
            path, cfg, torch.device("cpu"), torch.float64
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64, copy=False)
    )
    breakpoints = np.linspace(0.0, cfg.T, cfg.n + 1, dtype=np.float64)
    return RealizedCase(
        case_id="time_only",
        label=r"Time-only $u(t)$",
        checkpoint=path,
        cfg=cfg,
        breakpoints=breakpoints,
        controls=controls,
    )


def solve_state_interval(
    left: float,
    right: float,
    initial_state: np.ndarray,
    control: float,
    problem: TumorProblem,
    *,
    rtol: float,
    atol: float,
    dense_output: bool,
) -> Any:
    solution = solve_ivp(
        lambda _time, state: dynamics_numpy(state, control, problem),
        (left, right),
        np.asarray(initial_state, dtype=np.float64),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=dense_output,
        max_step=max((right - left) / 4.0, 1.0e-8),
    )
    if not solution.success:
        raise RuntimeError(
            f"state integration failed on [{left}, {right}]: {solution.message}"
        )
    return solution


def realize_feedback_case(
    path: Path,
    expected_option: str,
    case_id: str,
    label: str,
    initial_state: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> RealizedCase:
    if not path.is_file():
        raise FileNotFoundError(path)
    model, cfg, checkpoint_args = load_feedback_checkpoint(path)
    actual_option = str(getattr(checkpoint_args, "option", ""))
    if actual_option != expected_option:
        raise ValueError(
            f"{path}: expected option={expected_option}, found {actual_option!r}"
        )
    mode = configured_state_mode(checkpoint_args)
    if mode != "feedback":
        raise ValueError(f"{path}: expected a feedback checkpoint, found mode={mode}")

    problem = problem_from_config(cfg)
    breakpoints = np.linspace(0.0, cfg.T, cfg.n + 1, dtype=np.float64)
    grid = torch.linspace(0.0, 1.0, cfg.n + 1, dtype=torch.float64)
    with torch.inference_mode():
        base_logits = model.time_logits(grid)[: cfg.n]

    state = np.asarray(initial_state, dtype=np.float64).copy()
    controls = np.empty(cfg.n, dtype=np.float64)
    for index in range(cfg.n):
        normalized_time = torch.tensor(
            [index / cfg.n], dtype=torch.float64
        )
        state_tensor = torch.from_numpy(state.copy()).to(torch.float64).unsqueeze(0)
        with torch.inference_mode():
            action = model.interval_action(
                base_logits[index],
                normalized_time,
                state_tensor,
                state_mode="feedback",
            )
        control = float(action.item())
        if not -1.0e-10 <= control <= cfg.umax + 1.0e-10:
            raise RuntimeError(f"{case_id}: policy action outside the control box")
        controls[index] = np.clip(control, 0.0, cfg.umax)
        solution = solve_state_interval(
            float(breakpoints[index]),
            float(breakpoints[index + 1]),
            state,
            controls[index],
            problem,
            rtol=rtol,
            atol=atol,
            dense_output=False,
        )
        state = np.asarray(solution.y[:, -1], dtype=np.float64)
        if not np.all(np.isfinite(state)) or float(state.min()) <= 0.0:
            raise RuntimeError(f"{case_id}: nonpositive or nonfinite state encountered")

    return RealizedCase(
        case_id=case_id,
        label=label,
        checkpoint=path,
        cfg=cfg,
        breakpoints=breakpoints,
        controls=controls,
        feedback_model=model,
    )


def assert_common_problem(cases: list[RealizedCase]) -> TumorProblem:
    if not cases:
        raise ValueError("at least one case is required")
    fields = (
        "T",
        "m",
        "umax",
        "beta",
        "alpha",
        "gamma",
        "n0",
        "m_suppression",
    )
    reference = cases[0].cfg
    for case in cases[1:]:
        differences = {
            field: (getattr(reference, field), getattr(case.cfg, field))
            for field in fields
            if getattr(reference, field) != getattr(case.cfg, field)
        }
        if differences:
            raise ValueError(
                f"{case.case_id} uses a different physical problem: {differences}"
            )
    return problem_from_config(reference)


def resistant_heavy_initial_state(problem: TumorProblem) -> np.ndarray:
    x = np.linspace(0.0, 1.0, problem.m, dtype=np.float64)
    state = problem.n0 * (1.0 + 0.10 * (2.0 * x - 1.0))
    if not math.isclose(float(state[0]), 9.0, abs_tol=1.0e-12):
        raise RuntimeError("unexpected first resistant-heavy state component")
    if not math.isclose(float(state[-1]), 11.0, abs_tol=1.0e-12):
        raise RuntimeError("unexpected last resistant-heavy state component")
    if not math.isclose(
        float(state.sum()), problem.m * problem.n0, abs_tol=1.0e-10
    ):
        raise RuntimeError("resistant-heavy state should preserve total burden")
    return state


def forward_zoh_segments(
    breakpoints: np.ndarray,
    controls: np.ndarray,
    initial_state: np.ndarray,
    problem: TumorProblem,
    *,
    rtol: float,
    atol: float,
) -> tuple[list[Any], np.ndarray]:
    state = np.asarray(initial_state, dtype=np.float64).copy()
    segments: list[Any] = []
    node_states = [state.copy()]
    for index, control in enumerate(controls):
        solution = solve_state_interval(
            float(breakpoints[index]),
            float(breakpoints[index + 1]),
            state,
            float(control),
            problem,
            rtol=rtol,
            atol=atol,
            dense_output=True,
        )
        segments.append(solution.sol)
        state = np.asarray(solution.y[:, -1], dtype=np.float64)
        if not np.all(np.isfinite(state)) or float(state.min()) <= 0.0:
            raise RuntimeError("continuous state integration lost positivity")
        node_states.append(state.copy())
    return segments, np.asarray(node_states, dtype=np.float64)


def backward_costate_segments(
    breakpoints: np.ndarray,
    controls: np.ndarray,
    forward_segments: list[Any],
    problem: TumorProblem,
    *,
    rtol: float,
    atol: float,
) -> list[Any]:
    params = problem.vectors()
    costate = params["alpha"].copy()
    reversed_segments: list[Any] = []
    for index in range(len(controls) - 1, -1, -1):
        left = float(breakpoints[index])
        right = float(breakpoints[index + 1])
        control = float(controls[index])
        state_solution = forward_segments[index]

        def right_hand_side(time: float, value: np.ndarray) -> np.ndarray:
            state = np.asarray(state_solution(time), dtype=np.float64)
            return -dH_dN_numpy(state, value, control, problem, params)

        solution = solve_ivp(
            right_hand_side,
            (right, left),
            costate,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            dense_output=True,
            max_step=max((right - left) / 4.0, 1.0e-8),
        )
        if not solution.success:
            raise RuntimeError(
                f"costate integration failed on [{left}, {right}]: "
                f"{solution.message}"
            )
        reversed_segments.append(solution.sol)
        costate = np.asarray(solution.y[:, -1], dtype=np.float64)
    return list(reversed(reversed_segments))


def sample_segments(
    shared_time: np.ndarray,
    breakpoints: np.ndarray,
    controls: np.ndarray,
    state_segments: list[Any],
    costate_segments: list[Any],
    problem: TumorProblem,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = np.empty((shared_time.size, problem.m), dtype=np.float64)
    costate = np.empty_like(state)
    control = np.empty(shared_time.size, dtype=np.float64)
    for row, time in enumerate(shared_time):
        interval = min(
            np.searchsorted(breakpoints, time, side="right") - 1,
            len(controls) - 1,
        )
        interval = max(int(interval), 0)
        state[row] = state_segments[interval](float(time))
        costate[row] = costate_segments[interval](float(time))
        control[row] = controls[interval]
    return state, costate, control


def identity_checks(
    state: np.ndarray,
    costate: np.ndarray,
    control: np.ndarray,
    quantities: dict[str, np.ndarray],
    problem: TumorProblem,
) -> dict[str, float]:
    """Check H_u, dot(psi), and ddot(psi) through independent identities."""

    params = problem.vectors()
    growth_without_control = (
        params["r"][None, :]
        - params["M"][None, :] * np.log1p(state.mean(axis=1))[:, None]
    )
    state_dot = (
        growth_without_control
        - params["phi"][None, :] * control[:, None]
    ) * state
    coupling = (
        params["M"][None, :] * costate * state
    ).sum(axis=1)
    h_state = (
        params["beta"][None, :]
        + costate
        * (
            growth_without_control
            - params["phi"][None, :] * control[:, None]
        )
        - coupling[:, None] / (problem.m + state.sum(axis=1))[:, None]
    )
    costate_dot = -h_state

    # Independent central difference with N and lambda held fixed verifies
    # that psi is the partial derivative H_u, rather than hat(Psi).
    epsilon = 1.0e-5

    def hamiltonian(candidate_control: np.ndarray) -> np.ndarray:
        drift = (
            growth_without_control
            - params["phi"][None, :] * candidate_control[:, None]
        ) * state
        return (
            (params["beta"][None, :] * state).sum(axis=1)
            + problem.gamma * candidate_control
            + (costate * drift).sum(axis=1)
        )

    finite_difference_hu = (
        hamiltonian(control + epsilon) - hamiltonian(control - epsilon)
    ) / (2.0 * epsilon)
    dot_psi_chain = -(
        params["phi"][None, :]
        * (costate_dot * state + costate * state_dot)
    ).sum(axis=1)

    denominator = problem.m + state.sum(axis=1)
    numerator = coupling
    rho = numerator / denominator
    numerator_dot = (
        params["M"][None, :]
        * (costate_dot * state + costate * state_dot)
    ).sum(axis=1)
    denominator_dot = state_dot.sum(axis=1)
    rho_dot = (
        numerator_dot * denominator - numerator * denominator_dot
    ) / np.square(denominator)
    q = (params["phi"][None, :] * state).sum(axis=1)
    q_dot = (params["phi"][None, :] * state_dot).sum(axis=1)
    ddot_psi_chain = (
        params["phi"][None, :] * params["beta"][None, :] * state_dot
    ).sum(axis=1) - rho_dot * q - rho * q_dot

    errors = {
        "H_u_finite_difference_max_abs": float(
            np.max(np.abs(quantities["psi"] - finite_difference_hu))
        ),
        "dot_psi_chain_rule_max_abs": float(
            np.max(np.abs(quantities["dot_psi"] - dot_psi_chain))
        ),
        "ddot_psi_chain_rule_max_abs": float(
            np.max(np.abs(quantities["ddot_psi"] - ddot_psi_chain))
        ),
        "ddot_psi_A_plus_Bu_max_abs": float(
            np.max(
                np.abs(
                    quantities["ddot_psi"]
                    - (quantities["A"] + quantities["B"] * control)
                )
            )
        ),
    }
    coefficient_scale = max(
        1.0,
        abs(problem.alpha),
        abs(problem.beta) / 0.1,
        abs(problem.gamma) / 20.0,
    )
    tolerances = {
        "H_u_finite_difference_max_abs": 2.0e-5 * coefficient_scale,
        "dot_psi_chain_rule_max_abs": 2.0e-9 * coefficient_scale,
        "ddot_psi_chain_rule_max_abs": 2.0e-8 * coefficient_scale,
        "ddot_psi_A_plus_Bu_max_abs": 2.0e-12 * coefficient_scale,
    }
    failures = {
        key: (value, tolerances[key])
        for key, value in errors.items()
        if not math.isfinite(value) or value > tolerances[key]
    }
    if failures:
        raise RuntimeError(f"continuous PMP identity check failed: {failures}")
    return errors


def feedback_closure_error(
    specification: RealizedCase, node_states: np.ndarray
) -> float | None:
    model = specification.feedback_model
    if model is None:
        return None
    cfg = specification.cfg
    grid = torch.linspace(0.0, 1.0, cfg.n + 1, dtype=torch.float64)
    with torch.inference_mode():
        base_logits = model.time_logits(grid)[: cfg.n]
        reconstructed = []
        for index in range(cfg.n):
            normalized_time = torch.tensor(
                [index / cfg.n], dtype=torch.float64
            )
            state = torch.from_numpy(node_states[index].copy()).to(torch.float64)
            action = model.interval_action(
                base_logits[index],
                normalized_time,
                state.unsqueeze(0),
                state_mode="feedback",
            )
            reconstructed.append(float(action.item()))
    error = float(
        np.max(
            np.abs(
                np.asarray(reconstructed, dtype=np.float64)
                - specification.controls
            )
        )
    )
    if error > 2.0e-9:
        raise RuntimeError(
            f"{specification.case_id}: feedback closure error is {error:.3e}"
        )
    return error


def evaluate_case(
    specification: RealizedCase,
    shared_time: np.ndarray,
    initial_state: np.ndarray,
    problem: TumorProblem,
    *,
    rtol: float,
    atol: float,
) -> EvaluatedCase:
    state_segments, node_states = forward_zoh_segments(
        specification.breakpoints,
        specification.controls,
        initial_state,
        problem,
        rtol=rtol,
        atol=atol,
    )
    costate_segments = backward_costate_segments(
        specification.breakpoints,
        specification.controls,
        state_segments,
        problem,
        rtol=rtol,
        atol=atol,
    )
    state, costate, control = sample_segments(
        shared_time,
        specification.breakpoints,
        specification.controls,
        state_segments,
        costate_segments,
        problem,
    )
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(costate)):
        raise RuntimeError(f"{specification.case_id}: nonfinite PMP trajectory")
    if not np.allclose(state[0], initial_state, rtol=0.0, atol=2.0e-10):
        raise RuntimeError(f"{specification.case_id}: initial-state mismatch")
    if not np.allclose(
        costate[-1], problem.vectors()["alpha"], rtol=0.0, atol=2.0e-9
    ):
        raise RuntimeError(f"{specification.case_id}: terminal-costate mismatch")

    diagnostic_result = {
        "diagnostic_N": state,
        "diagnostic_lambda": costate,
        "diagnostic_u": control,
    }
    quantities = strict_singular_quantities(diagnostic_result, problem)
    checks = identity_checks(state, costate, control, quantities, problem)
    closure_error = feedback_closure_error(specification, node_states)
    return EvaluatedCase(
        specification=specification,
        time=shared_time,
        state=state,
        costate=costate,
        control=control,
        node_states=node_states,
        quantities=quantities,
        identity_errors=checks,
        feedback_closure_error=closure_error,
    )


def quantity_array(case: EvaluatedCase, key: str) -> np.ndarray:
    mapping = {
        "H_u": "psi",
        "dH_u_dt": "dot_psi",
        "d2H_u_dt2": "ddot_psi",
    }
    return np.asarray(case.quantities[mapping[key]], dtype=np.float64)


def build_summary(cases: list[EvaluatedCase]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        for key, _plot_label, _axis_label in QUANTITIES:
            values = quantity_array(case, key)
            rows.append(
                {
                    "case_id": case.specification.case_id,
                    "case_label": case.specification.label.replace("$", ""),
                    "quantity": key,
                    "rms": float(np.sqrt(np.mean(np.square(values)))),
                    "mean_abs": float(np.mean(np.abs(values))),
                    "max_abs": float(np.max(np.abs(values))),
                    "checkpoint": str(case.specification.checkpoint),
                }
            )
    return rows


def write_timeseries(path: Path, cases: list[EvaluatedCase]) -> None:
    fields = (
        "case_id",
        "case_label",
        "t",
        "quantity",
        "value",
        "u",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for case in cases:
            plain_label = case.specification.label.replace("$", "")
            for key, _plot_label, _axis_label in QUANTITIES:
                values = quantity_array(case, key)
                for time, value, control in zip(
                    case.time, values, case.control, strict=True
                ):
                    writer.writerow(
                        {
                            "case_id": case.specification.case_id,
                            "case_label": plain_label,
                            "t": f"{time:.12g}",
                            "quantity": key,
                            "value": f"{value:.16g}",
                            "u": f"{control:.16g}",
                        }
                    )


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "case_id",
        "case_label",
        "quantity",
        "rms",
        "mean_abs",
        "max_abs",
        "checkpoint",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for key in ("rms", "mean_abs", "max_abs"):
                formatted[key] = f"{float(row[key]):.12g}"
            writer.writerow(formatted)


def tex_number(value: float) -> str:
    if value == 0.0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    if -2 <= exponent <= 2:
        return f"{value:.4g}"
    mantissa = value / (10.0**exponent)
    return rf"{mantissa:.3f}\times 10^{{{exponent}}}"


def write_latex_table(path: Path, rows: list[dict[str, Any]]) -> None:
    lookup = {
        (row["case_id"], row["quantity"]): row
        for row in rows
    }
    case_order = ("time_only", "feedback_cf", "feedback_der")
    labels = {
        "H_u": r"$H_u$",
        "dH_u_dt": r"$dH_u/dt$",
        "d2H_u_dt2": r"$d^2H_u/dt^2$",
    }
    lines = [
        "% Generated by generate_three_case_hamiltonian_results.py.",
        "% Each numeric cell is RMS / max absolute value on the shared time grid.",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        (
            r"Quantity (RMS / max $|\cdot|$) & Time-only $u(t)$ "
            r"& $u(N,t)$ Case 1 (CF) & $u(N,t)$ Case 2 (DER) \\"
        ),
        r"\midrule",
    ]
    for quantity, label in labels.items():
        cells = []
        for case_id in case_order:
            row = lookup[(case_id, quantity)]
            cells.append(
                rf"${tex_number(float(row['rms']))}\,/\,{tex_number(float(row['max_abs']))}$"
            )
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    path.write_text("\n".join(lines))


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 300,
        }
    )


def plot_results(
    pdf_path: Path,
    png_path: Path,
    cases: list[EvaluatedCase],
    summary_rows: list[dict[str, Any]],
    problem: TumorProblem,
    *,
    dpi: int,
) -> None:
    configure_plotting()
    summary_lookup = {
        (row["case_id"], row["quantity"]): row for row in summary_rows
    }
    colors = ("#355C7D", "#B3544A", "#2D7A68")
    figure, axes = plt.subplots(
        3,
        3,
        figsize=(15.2, 6.4),
        sharex=True,
        sharey="row",
        constrained_layout=False,
    )

    row_limits: dict[str, tuple[float, float]] = {}
    for key, _plot_label, _axis_label in QUANTITIES:
        maximum = max(
            float(np.max(np.abs(quantity_array(case, key)))) for case in cases
        )
        half_range = max(1.08 * maximum, 1.0e-12)
        row_limits[key] = (-half_range, half_range)

    for column, case in enumerate(cases):
        axes[0, column].set_title(case.specification.label, pad=8)
        for row, (key, _plot_label, axis_label) in enumerate(QUANTITIES):
            axis = axes[row, column]
            values = quantity_array(case, key)
            axis.plot(case.time, values, color=colors[row], linewidth=1.25)
            axis.axhline(0.0, color="#111827", linewidth=0.7, linestyle="--")
            axis.set_xlim(0.0, problem.T)
            axis.set_ylim(*row_limits[key])
            axis.grid(True, color="#E5E7EB", linewidth=0.55)
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
            summary = summary_lookup[(case.specification.case_id, key)]
            axis.text(
                0.03,
                0.96,
                "RMS " + f"{float(summary['rms']):.3g}\n"
                + r"max $|\cdot|$ " + f"{float(summary['max_abs']):.3g}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=7.2,
                color="#374151",
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "white",
                    "edgecolor": "#D1D5DB",
                    "alpha": 0.88,
                    "linewidth": 0.55,
                },
            )
            if column == 0:
                axis.set_ylabel(axis_label)
            if row == 2:
                axis.set_xlabel(r"time $t$")

    figure.suptitle(
        "Same initial state and continuous PMP evaluator",
        fontsize=12,
        y=0.992,
    )
    figure.text(
        0.5,
        0.958,
        r"$N_i(0)=10[1+0.10(2x_i-1)]$; each row uses a common vertical scale",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#4B5563",
    )
    figure.subplots_adjust(
        left=0.08, right=0.985, bottom=0.08, top=0.865, wspace=0.16, hspace=0.20
    )
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def write_metadata(
    path: Path,
    cases: list[EvaluatedCase],
    initial_state: np.ndarray,
    args: argparse.Namespace,
) -> None:
    payload = {
        "definition": {
            "H_u": "psi = gamma - sum_i phi_i lambda_i N_i",
            "dH_u_dt": "dot(psi)",
            "d2H_u_dt2": "ddot(psi) = A + B u",
            "evaluator": (
                "DOP853 state forward and standard PMP costate backward; "
                "realized controls held zero-order constant on each interval"
            ),
            "excluded_quantity": (
                "the discrete RK4 interval derivative hat(Psi) is not used"
            ),
        },
        "diagnostic_points": int(args.diagnostic_points),
        "rtol": float(args.rtol),
        "atol": float(args.atol),
        "initial_state": initial_state.tolist(),
        "cases": [
            {
                "case_id": case.specification.case_id,
                "case_label": case.specification.label,
                "checkpoint": str(case.specification.checkpoint),
                "control_intervals": int(case.specification.controls.size),
                "identity_errors": case.identity_errors,
                "feedback_closure_error": case.feedback_closure_error,
            }
            for case in cases
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--time-checkpoint", type=Path, default=DEFAULT_TIME_CHECKPOINT
    )
    parser.add_argument("--cf-checkpoint", type=Path, default=DEFAULT_CF_CHECKPOINT)
    parser.add_argument(
        "--der-checkpoint", type=Path, default=DEFAULT_DER_CHECKPOINT
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--diagnostic-points", type=int, default=4001)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.diagnostic_points < 101:
        parser.error("--diagnostic-points must be at least 101")
    if args.rtol <= 0.0 or args.atol <= 0.0:
        parser.error("integration tolerances must be positive")
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    args.time_checkpoint = resolve_path(args.time_checkpoint)
    args.cf_checkpoint = resolve_path(args.cf_checkpoint)
    args.der_checkpoint = resolve_path(args.der_checkpoint)
    args.out_dir = resolve_path(args.out_dir)
    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    time_case = load_time_only_case(args.time_checkpoint)
    time_problem = problem_from_config(time_case.cfg)
    initial_state = resistant_heavy_initial_state(time_problem)
    cf_case = realize_feedback_case(
        args.cf_checkpoint,
        "cf",
        "feedback_cf",
        r"$u(N,t)$ Case 1 (CF)",
        initial_state,
        rtol=args.rtol,
        atol=args.atol,
    )
    der_case = realize_feedback_case(
        args.der_checkpoint,
        "der",
        "feedback_der",
        r"$u(N,t)$ Case 2 (DER)",
        initial_state,
        rtol=args.rtol,
        atol=args.atol,
    )
    specifications = [time_case, cf_case, der_case]
    problem = assert_common_problem(specifications)
    shared_time = np.linspace(
        0.0, problem.T, args.diagnostic_points, dtype=np.float64
    )
    evaluated = [
        evaluate_case(
            case,
            shared_time,
            initial_state,
            problem,
            rtol=args.rtol,
            atol=args.atol,
        )
        for case in specifications
    ]

    summary_rows = build_summary(evaluated)
    timeseries_path = args.out_dir / "three_case_hamiltonian_timeseries.csv"
    summary_path = args.out_dir / "three_case_hamiltonian_summary.csv"
    table_path = args.out_dir / "three_case_summary_table.tex"
    pdf_path = args.out_dir / "three_case_hamiltonian_grid.pdf"
    png_path = args.out_dir / "three_case_hamiltonian_grid.png"
    metadata_path = args.out_dir / "three_case_hamiltonian_metadata.json"

    write_timeseries(timeseries_path, evaluated)
    write_summary(summary_path, summary_rows)
    write_latex_table(table_path, summary_rows)
    plot_results(
        pdf_path,
        png_path,
        evaluated,
        summary_rows,
        problem,
        dpi=args.dpi,
    )
    write_metadata(metadata_path, evaluated, initial_state, args)

    print("Three-case continuous PMP diagnostics completed.")
    for case in evaluated:
        largest_identity_error = max(case.identity_errors.values())
        closure = (
            "n/a"
            if case.feedback_closure_error is None
            else f"{case.feedback_closure_error:.3e}"
        )
        print(
            f"  {case.specification.case_id}: max identity error "
            f"{largest_identity_error:.3e}; closure error {closure}"
        )
    for path in (
        timeseries_path,
        summary_path,
        table_path,
        pdf_path,
        png_path,
        metadata_path,
    ):
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
