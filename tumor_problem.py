#!/usr/bin/env python3
"""Canonical tumor-control problem and reference evaluation utilities.

The reference evaluator treats a sampled control as zero-order hold (ZOH),
integrates every control interval separately, and augments the state with the
running cost. This keeps control discontinuities out of numerical quadrature
and gives every method the same realized-objective calculation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class TumorProblem:
    T: float = 10.0
    m: int = 21
    umax: float = 3.0
    beta: float = 0.1
    alpha: float = 1.0
    gamma: float = 20.0
    n0: float = 10.0
    m_suppression: float = 0.5

    def vectors(self) -> dict[str, np.ndarray]:
        grid = np.linspace(0.0, 1.0, self.m, dtype=np.float64)
        return {
            "grid": grid,
            "r": 2.0 / (1.0 + 3.0 * grid**4),
            "phi": 1.0 / (1.0 + grid**2),
            "M": np.full(self.m, self.m_suppression, dtype=np.float64),
            "beta": np.full(self.m, self.beta, dtype=np.float64),
            "alpha": np.full(self.m, self.alpha, dtype=np.float64),
            "N0": np.full(self.m, self.n0, dtype=np.float64),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


NOMINAL_TUMOR_PROBLEM = TumorProblem()


def assert_nominal_problem(problem: TumorProblem, *, context: str = "") -> None:
    expected = NOMINAL_TUMOR_PROBLEM.to_dict()
    actual = problem.to_dict()
    mismatches = {key: (actual[key], value) for key, value in expected.items() if actual[key] != value}
    if mismatches:
        prefix = f"{context}: " if context else ""
        raise ValueError(f"{prefix}problem parameters differ from the canonical tumor problem: {mismatches}")


def dynamics_numpy(N: np.ndarray, u: float, problem: TumorProblem, params: dict[str, np.ndarray] | None = None) -> np.ndarray:
    p = problem.vectors() if params is None else params
    N_safe = np.maximum(np.asarray(N, dtype=np.float64), 1e-12)
    G = np.log1p(N_safe.mean())
    return (p["r"] - p["phi"] * float(u) - p["M"] * G) * N_safe


def dH_dN_numpy(
    N: np.ndarray,
    lam: np.ndarray,
    u: float,
    problem: TumorProblem,
    params: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    p = problem.vectors() if params is None else params
    N_safe = np.maximum(np.asarray(N, dtype=np.float64), 1e-12)
    G = np.log1p(N_safe.mean())
    a = p["r"] - p["phi"] * float(u) - p["M"] * G
    coupling = np.sum(p["M"] * lam * N_safe)
    return p["beta"] + lam * a - coupling / (problem.m + N_safe.sum())


def singular_control_numpy(
    N: np.ndarray,
    problem: TumorProblem,
    params: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    p = problem.vectors() if params is None else params
    N_arr = np.maximum(np.asarray(N, dtype=np.float64), 1e-12)
    one_dimensional = N_arr.ndim == 1
    if one_dimensional:
        N_arr = N_arr[None, :]
    G = np.log1p(N_arr.mean(axis=1))
    numerator = (p["beta"] * (p["r"][None, :] - p["M"][None, :] * G[:, None]) * N_arr).sum(axis=1)
    denominator = (p["beta"] * p["phi"] * N_arr).sum(axis=1)
    result = numerator / np.maximum(denominator, 1e-14)
    return result[0] if one_dimensional else result


def prepare_zoh_control(t: np.ndarray, u: np.ndarray, problem: TumorProblem) -> tuple[np.ndarray, np.ndarray]:
    t_arr = np.asarray(t, dtype=np.float64).reshape(-1)
    u_arr = np.asarray(u, dtype=np.float64).reshape(-1)
    if t_arr.size < 2:
        raise ValueError("control time grid must contain at least two points")
    if u_arr.size == t_arr.size:
        u_interval = u_arr[:-1]
    elif u_arr.size == t_arr.size - 1:
        u_interval = u_arr
    else:
        raise ValueError(f"expected len(u) to be len(t) or len(t)-1, got {u_arr.size} and {t_arr.size}")
    if not np.all(np.diff(t_arr) > 0.0):
        raise ValueError("control time grid must be strictly increasing")
    if not np.isclose(t_arr[0], 0.0, atol=1e-10) or not np.isclose(t_arr[-1], problem.T, atol=1e-10):
        raise ValueError(f"control grid must span [0, {problem.T}]")
    return t_arr, np.clip(u_interval, 0.0, problem.umax)


def _forward_segments(
    t: np.ndarray,
    u: np.ndarray,
    problem: TumorProblem,
    *,
    rtol: float,
    atol: float,
) -> tuple[list[Any], np.ndarray]:
    params = problem.vectors()
    y = np.concatenate([params["N0"], np.zeros(1, dtype=np.float64)])
    segments: list[Any] = []
    break_states = [y.copy()]

    for index, control in enumerate(u):
        left, right = float(t[index]), float(t[index + 1])

        def augmented_rhs(_time: float, value: np.ndarray) -> np.ndarray:
            N = np.maximum(value[: problem.m], 1e-12)
            running = float(params["beta"] @ N + problem.gamma * control)
            return np.concatenate([dynamics_numpy(N, control, problem, params), np.array([running])])

        solution = solve_ivp(
            augmented_rhs,
            (left, right),
            y,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            dense_output=True,
            max_step=max((right - left) / 4.0, 1e-8),
        )
        if not solution.success:
            raise RuntimeError(f"state integration failed on [{left}, {right}]: {solution.message}")
        segments.append(solution.sol)
        y = solution.y[:, -1]
        y[: problem.m] = np.maximum(y[: problem.m], 1e-12)
        break_states.append(y.copy())
    return segments, np.asarray(break_states)


def _backward_segments(
    t: np.ndarray,
    u: np.ndarray,
    forward_segments: list[Any],
    problem: TumorProblem,
    *,
    rtol: float,
    atol: float,
) -> list[Any]:
    params = problem.vectors()
    lam = params["alpha"].copy()
    reverse_segments: list[Any] = []

    for index in range(len(u) - 1, -1, -1):
        left, right = float(t[index]), float(t[index + 1])
        control = float(u[index])
        state_solution = forward_segments[index]

        def costate_rhs(time: float, value: np.ndarray) -> np.ndarray:
            N = np.maximum(state_solution(time)[: problem.m], 1e-12)
            return -dH_dN_numpy(N, value, control, problem, params)

        solution = solve_ivp(
            costate_rhs,
            (right, left),
            lam,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            dense_output=True,
            max_step=max((right - left) / 4.0, 1e-8),
        )
        if not solution.success:
            raise RuntimeError(f"costate integration failed on [{left}, {right}]: {solution.message}")
        reverse_segments.append(solution.sol)
        lam = solution.y[:, -1]
    return list(reversed(reverse_segments))


def _sample_segments(
    sample_t: np.ndarray,
    breakpoints: np.ndarray,
    controls: np.ndarray,
    forward_segments: list[Any],
    backward_segments: list[Any] | None,
    problem: TumorProblem,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = np.empty((len(sample_t), problem.m), dtype=np.float64)
    costates = np.empty_like(states) if backward_segments is not None else np.empty((0, problem.m))
    sampled_u = np.empty(len(sample_t), dtype=np.float64)
    for row, time in enumerate(sample_t):
        index = min(np.searchsorted(breakpoints, time, side="right") - 1, len(controls) - 1)
        index = max(index, 0)
        states[row] = forward_segments[index](float(time))[: problem.m]
        sampled_u[row] = controls[index]
        if backward_segments is not None:
            costates[row] = backward_segments[index](float(time))
    return states, costates, sampled_u


def evaluate_zoh_control(
    t: np.ndarray,
    u: np.ndarray,
    problem: TumorProblem = NOMINAL_TUMOR_PROBLEM,
    *,
    diagnostic_points: int = 4001,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    include_diagnostics: bool = True,
    singular_tol_relative: float = 0.005,
    singular_gate_tau_relative: float = 0.0015,
) -> dict[str, Any]:
    """Evaluate a ZOH control with a common objective and PMP diagnostics."""

    breakpoints, controls = prepare_zoh_control(t, u, problem)
    forward_segments, break_states = _forward_segments(breakpoints, controls, problem, rtol=rtol, atol=atol)
    params = problem.vectors()
    final_N = break_states[-1, : problem.m]
    running_cost = float(break_states[-1, problem.m])
    terminal_cost = float(params["alpha"] @ final_N)
    result: dict[str, Any] = {
        "J": terminal_cost + running_cost,
        "terminal_cost": terminal_cost,
        "running_cost": running_cost,
        "final_total_N": float(final_N.sum()),
        "final_mean_N": float(final_N.mean()),
        "u_min": float(controls.min()),
        "u_max": float(controls.max()),
        "u_mean_time": float(np.sum(controls * np.diff(breakpoints)) / problem.T),
        "problem": problem.to_dict(),
    }
    if not include_diagnostics:
        return result

    backward_segments = _backward_segments(
        breakpoints,
        controls,
        forward_segments,
        problem,
        rtol=rtol,
        atol=atol,
    )
    sample_t = np.linspace(0.0, problem.T, diagnostic_points, dtype=np.float64)
    N, lam, sampled_u = _sample_segments(
        sample_t,
        breakpoints,
        controls,
        forward_segments,
        backward_segments,
        problem,
    )
    psi = problem.gamma - (params["phi"][None, :] * lam * N).sum(axis=1)
    u_singular = singular_control_numpy(N, problem, params)
    psi_relative = psi / max(abs(problem.gamma), 1e-12)
    normalized_u = sampled_u / problem.umax
    projected = np.abs(normalized_u - np.clip(normalized_u - psi_relative, 0.0, 1.0))
    singular_admissible = (u_singular >= 0.0) & (u_singular <= problem.umax)
    singular_error = np.abs(sampled_u - u_singular) / problem.umax
    gate_argument = (singular_tol_relative - np.abs(psi_relative)) / singular_gate_tau_relative
    singular_weight = 1.0 / (1.0 + np.exp(-np.clip(gate_argument, -60.0, 60.0)))
    singular_weight *= singular_admissible.astype(np.float64)
    merit_squared = singular_weight * singular_error**2 + (1.0 - singular_weight) * projected**2
    near_singular = (np.abs(psi_relative) <= singular_tol_relative) & singular_admissible
    singular_rms = float(np.sqrt(np.mean(singular_error[near_singular] ** 2))) if np.any(near_singular) else float("nan")
    singular_mae = float(np.mean(singular_error[near_singular])) if np.any(near_singular) else float("nan")

    result.update(
        {
            "diagnostic_points": diagnostic_points,
            "projected_kkt_rms": float(np.sqrt(np.mean(projected**2))),
            "projected_kkt_mean": float(np.mean(projected)),
            "singular_control_rms": singular_rms,
            "singular_control_mae": singular_mae,
            "singular_fraction": float(np.mean(near_singular)),
            "pmp_merit_rms": float(np.sqrt(np.mean(merit_squared))),
            "pmp_merit_mean": float(np.mean(singular_weight * singular_error + (1.0 - singular_weight) * projected)),
            "psi_relative_abs_mean": float(np.mean(np.abs(psi_relative))),
            "diagnostic_t": sample_t,
            "diagnostic_N": N,
            "diagnostic_lambda": lam,
            "diagnostic_u": sampled_u,
            "diagnostic_psi": psi,
            "diagnostic_u_singular": u_singular,
            "diagnostic_singular_weight": singular_weight,
        }
    )
    return result


def serializable_metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if np.isscalar(value) or isinstance(value, dict)}
