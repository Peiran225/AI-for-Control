#!/usr/bin/env python3
"""Common held-out evaluator for the AAAI Table-1 comparison.

The evaluator uses exactly one set of 128 initial-state directions and one
post-training protocol for every retained policy:

* physical objective weights ``(alpha,beta,gamma)=(1,40,8000)``;
* 800 left-endpoint, zero-order-hold control intervals;
* float64 RK4 with four substeps per interval for the reported objective;
* continuous-PMP scalar identities sampled on a 32-times-refined ZOH grid;
* ``R_sing`` evaluated on the half-open interval ``[1.5,8.0)``.

The script intentionally writes one method/run at a time.  TensorFlow-1,
TensorFlow-2, and PyTorch policies can therefore be evaluated in separate
processes, and a failed run cannot overwrite a completed artifact.

For a time-only control, ``--control`` is replayed unchanged on every held-out
initial state.  In particular, ``--method direct_nominal_replay`` is *not*
state-wise direct re-optimization.  That distinction is recorded in every
output row and must remain visible if the result is used in a paper table.
State-wise direct transcription requires 128 fresh optimization runs and a
separate artifact bundle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for search_path in (ROOT, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))


METHOD_CHOICES = (
    "direct_nominal_replay",
    "neural_pmp_learned",
    "neural_pmp_exact",
    "pmp_kkt_time",
    "pmp_kkt_cf",
    "pmp_kkt_der",
    "adaptive_hjb_nn",
    "pi_deeponet",
    "deepbsde",
)


@dataclass(frozen=True)
class Problem:
    T: float = 10.0
    intervals: int = 800
    m: int = 21
    umax: float = 3.0
    alpha: float = 1.0
    beta: float = 40.0
    gamma: float = 8000.0
    n0: float = 10.0
    suppression: float = 0.5

    def vectors(self) -> dict[str, np.ndarray]:
        phenotype = np.linspace(0.0, 1.0, self.m, dtype=np.float64)
        return {
            "r": 2.0 / (1.0 + 3.0 * phenotype**4),
            "phi": 1.0 / (1.0 + phenotype**2),
            "M": np.full(self.m, self.suppression, dtype=np.float64),
            "beta": np.full(self.m, self.beta, dtype=np.float64),
            "alpha": np.full(self.m, self.alpha, dtype=np.float64),
        }


def resolve(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def load_directions(path: Path, expected_samples: int, expected_m: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as source:
        directions = np.asarray(source["directions"], dtype=np.float64)
        families = np.asarray(source["direction_family"]).astype(str)
    if directions.ndim != 2 or directions.shape[1] != expected_m:
        raise ValueError(
            f"{path}: expected directions with {expected_m} columns, "
            f"found {directions.shape}"
        )
    if directions.shape[0] < expected_samples:
        raise ValueError(
            f"{path}: requested {expected_samples} directions, found "
            f"{directions.shape[0]}"
        )
    directions = directions[:expected_samples]
    families = families[:expected_samples]
    if np.any(families != "random"):
        raise ValueError(f"{path}: Table 1 requires the random direction family")
    if not np.all(np.isfinite(directions)):
        raise ValueError(f"{path}: directions contain non-finite values")
    if float(np.max(np.abs(directions))) > 1.0 + 1.0e-12:
        raise ValueError(f"{path}: direction lies outside [-1,1]")
    return directions


def load_control(path: Path, problem: Problem) -> np.ndarray:
    with np.load(path, allow_pickle=False) as source:
        if "u" not in source:
            raise ValueError(f"{path}: missing u array")
        control = np.asarray(source["u"], dtype=np.float64).reshape(-1)
        time_grid = (
            np.asarray(source["t"], dtype=np.float64).reshape(-1)
            if "t" in source
            else None
        )
    if control.size == problem.intervals + 1:
        control = control[:-1]
    if control.size != problem.intervals:
        raise ValueError(
            f"{path}: common evaluator requires {problem.intervals} controls, "
            f"found {control.size}"
        )
    if time_grid is not None:
        expected = np.linspace(0.0, problem.T, problem.intervals + 1)
        if time_grid.size != expected.size or not np.allclose(
            time_grid, expected, rtol=0.0, atol=2.0e-12
        ):
            raise ValueError(f"{path}: control is not stored on the common n=800 grid")
    if not np.all(np.isfinite(control)):
        raise ValueError(f"{path}: non-finite control")
    if float(np.min(control)) < -1.0e-9 or float(np.max(control)) > problem.umax + 1.0e-9:
        raise ValueError(f"{path}: control violates [0,{problem.umax}]")
    return np.clip(control, 0.0, problem.umax)


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


def running_cost(
    state: np.ndarray,
    control: np.ndarray,
    problem: Problem,
    vectors: dict[str, np.ndarray],
) -> np.ndarray:
    return state @ vectors["beta"] + problem.gamma * control


def rk4_microstep(
    state: np.ndarray,
    control: np.ndarray,
    step: float,
    problem: Problem,
    vectors: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    k1 = dynamics(state, control, problem, vectors)
    state2 = state + 0.5 * step * k1
    k2 = dynamics(state2, control, problem, vectors)
    state3 = state + 0.5 * step * k2
    k3 = dynamics(state3, control, problem, vectors)
    state4 = state + step * k3
    k4 = dynamics(state4, control, problem, vectors)
    next_state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    midpoint_state = 0.5 * (state2 + state3)
    return next_state, midpoint_state


def rollout_controls(
    initial: np.ndarray,
    problem: Problem,
    vectors: dict[str, np.ndarray],
    *,
    fixed_control: np.ndarray | None,
    query: Callable[[int, float, np.ndarray], float] | None,
    objective_substeps: int,
) -> tuple[np.ndarray, np.ndarray]:
    if (fixed_control is None) == (query is None):
        raise ValueError("supply exactly one of fixed_control or query")
    batch = initial.shape[0]
    state = np.asarray(initial, dtype=np.float64).copy()
    objective_running = np.zeros(batch, dtype=np.float64)
    controls = np.empty((batch, problem.intervals), dtype=np.float64)
    interval_step = problem.T / problem.intervals
    microstep = interval_step / objective_substeps

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
            raise RuntimeError(f"policy returned non-finite control at interval {index}")
        if float(np.min(control)) < -1.0e-8 or float(np.max(control)) > problem.umax + 1.0e-8:
            raise RuntimeError(
                f"policy returned control outside [0,{problem.umax}] at interval {index}"
            )
        control = np.clip(control, 0.0, problem.umax)
        controls[:, index] = control
        for _ in range(objective_substeps):
            k1 = dynamics(state, control, problem, vectors)
            state2 = state + 0.5 * microstep * k1
            k2 = dynamics(state2, control, problem, vectors)
            state3 = state + 0.5 * microstep * k2
            k3 = dynamics(state3, control, problem, vectors)
            state4 = state + microstep * k3
            k4 = dynamics(state4, control, problem, vectors)
            objective_running += (microstep / 6.0) * (
                running_cost(state, control, problem, vectors)
                + 2.0 * running_cost(state2, control, problem, vectors)
                + 2.0 * running_cost(state3, control, problem, vectors)
                + running_cost(state4, control, problem, vectors)
            )
            state = state + (microstep / 6.0) * (
                k1 + 2.0 * k2 + 2.0 * k3 + k4
            )
            if not np.all(np.isfinite(state)) or float(np.min(state)) <= 0.0:
                raise RuntimeError("RK4 rollout produced nonpositive/non-finite state")
    objective = objective_running + state @ vectors["alpha"]
    return controls, objective


def dH_dN(
    state: np.ndarray,
    costate: np.ndarray,
    control: np.ndarray,
    vectors: dict[str, np.ndarray],
) -> np.ndarray:
    safe_state = np.maximum(state, 1.0e-12)
    drift = (
        vectors["r"][None, :]
        - vectors["phi"][None, :] * control[:, None]
        - vectors["M"][None, :] * np.log1p(safe_state.mean(axis=1))[:, None]
    )
    coupling = (costate * vectors["M"][None, :] * safe_state).sum(axis=1)
    denominator = safe_state.shape[1] + safe_state.sum(axis=1)
    return (
        vectors["beta"][None, :]
        + costate * drift
        - coupling[:, None] / denominator[:, None]
    )


def scalar_quantities(
    state: np.ndarray,
    costate: np.ndarray,
    control: np.ndarray,
    problem: Problem,
    vectors: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    safe_state = np.maximum(state, 1.0e-12)
    G = np.log1p(safe_state.mean(axis=1))
    denominator = problem.m + safe_state.sum(axis=1)
    coupled = (
        vectors["M"][None, :] * costate * safe_state
    ).sum(axis=1)
    rho = coupled / denominator
    phi_N = (vectors["phi"][None, :] * safe_state).sum(axis=1)

    psi = problem.gamma - (
        vectors["phi"][None, :] * costate * safe_state
    ).sum(axis=1)
    dot_psi = (
        vectors["phi"][None, :] * vectors["beta"][None, :] * safe_state
    ).sum(axis=1) - rho * phi_N

    drift0 = (
        vectors["r"][None, :]
        - vectors["M"][None, :] * G[:, None]
    )
    weighted_beta_dot = (
        vectors["phi"][None, :]
        * vectors["beta"][None, :]
        * drift0
        * safe_state
    ).sum(axis=1)
    phi_dot = (vectors["phi"][None, :] * drift0 * safe_state).sum(axis=1)
    coupled_dot = (
        vectors["M"][None, :]
        * safe_state
        * (-vectors["beta"][None, :] + rho[:, None])
    ).sum(axis=1)
    denominator_dot = (drift0 * safe_state).sum(axis=1)
    rho_dot = coupled_dot / denominator - rho * denominator_dot / denominator
    A = weighted_beta_dot - rho_dot * phi_N - rho * phi_dot
    B = (
        -(vectors["phi"][None, :] ** 2 * vectors["beta"][None, :] * safe_state).sum(axis=1)
        + rho * (vectors["phi"][None, :] ** 2 * safe_state).sum(axis=1)
        - rho * phi_N**2 / denominator
    )
    return psi, dot_psi, A + B * control


def singular_residual(
    initial: np.ndarray,
    controls: np.ndarray,
    problem: Problem,
    vectors: dict[str, np.ndarray],
    *,
    refinement: int,
    interior_start: float,
    interior_end: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-sample scalar RMS values and their Euclidean combination."""

    batch = initial.shape[0]
    microsteps = problem.intervals * refinement
    step = problem.T / microsteps
    # The arrays are batch-local; --batch-size limits peak memory.
    states = np.empty((microsteps + 1, batch, problem.m), dtype=np.float64)
    midpoints = np.empty((microsteps, batch, problem.m), dtype=np.float64)
    states[0] = initial
    state = initial.copy()
    for micro in range(microsteps):
        interval = micro // refinement
        control = controls[:, interval]
        state, midpoint = rk4_microstep(state, control, step, problem, vectors)
        if not np.all(np.isfinite(state)) or float(np.min(state)) <= 0.0:
            raise RuntimeError("q-refined state rollout lost positivity")
        states[micro + 1] = state
        midpoints[micro] = midpoint

    costate = np.broadcast_to(vectors["alpha"], (batch, problem.m)).copy()
    sum_psi2 = np.zeros(batch, dtype=np.float64)
    sum_dot2 = np.zeros(batch, dtype=np.float64)
    sum_ddot2 = np.zeros(batch, dtype=np.float64)
    count = 0
    for micro in range(microsteps - 1, -1, -1):
        interval = micro // refinement
        control = controls[:, interval]
        state_left = states[micro]
        state_mid = midpoints[micro]
        state_right = states[micro + 1]

        def rhs(current_state: np.ndarray, current_costate: np.ndarray) -> np.ndarray:
            return -dH_dN(current_state, current_costate, control, vectors)

        k1 = rhs(state_right, costate)
        k2 = rhs(state_mid, costate - 0.5 * step * k1)
        k3 = rhs(state_mid, costate - 0.5 * step * k2)
        k4 = rhs(state_left, costate - step * k3)
        costate = costate - (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        time_value = micro * step
        if interior_start <= time_value < interior_end:
            psi, dot_psi, ddot_psi = scalar_quantities(
                state_left, costate, control, problem, vectors
            )
            sum_psi2 += np.square(psi)
            sum_dot2 += np.square(dot_psi)
            sum_ddot2 += np.square(ddot_psi)
            count += 1
    expected_count = int(round((interior_end - interior_start) / step))
    if count != expected_count:
        raise RuntimeError(
            f"interior point count mismatch: expected {expected_count}, found {count}"
        )
    rms_psi = np.sqrt(sum_psi2 / count)
    rms_dot = np.sqrt(sum_dot2 / count)
    rms_ddot = np.sqrt(sum_ddot2 / count)
    combined = np.sqrt(rms_psi**2 + rms_dot**2 + rms_ddot**2)
    return rms_psi, rms_dot, rms_ddot, combined


def load_policy(
    method: str,
    source: Path,
    *,
    hjb_tau: float,
) -> tuple[Callable[[int, float, np.ndarray], float], Callable[[], None], dict[str, Any]]:
    from compare_feedback_related_work import (
        load_deepbsde,
        load_hjb,
        load_ours,
        load_pi,
    )

    if method in {"pmp_kkt_cf", "pmp_kkt_der"}:
        policy, native_intervals, load_seconds = load_ours(source)
        extra: dict[str, Any] = {}
    elif method == "pi_deeponet":
        policy, native_intervals, load_seconds = load_pi(source)
        extra = {}
    elif method == "adaptive_hjb_nn":
        policy, native_intervals, load_seconds = load_hjb(source, hjb_tau)
        extra = {"hjb_tau_normalized": hjb_tau}
    elif method == "deepbsde":
        checkpoint_dir = source if source.is_dir() else source.parent
        policy, native_intervals, load_seconds, sigma, seed = load_deepbsde(
            checkpoint_dir
        )
        extra = {"deepbsde_sigma": sigma, "checkpoint_seed": seed}
    else:
        raise ValueError(f"{method} is not a feedback-policy method")

    training = np.asarray(policy.objective_weights, dtype=np.float64)
    physical = np.asarray([1.0, 40.0, 8000.0], dtype=np.float64)
    ratios = physical / training
    if not np.allclose(ratios, ratios[0], rtol=1.0e-10, atol=1.0e-12):
        policy.close()
        raise ValueError(
            f"{source}: checkpoint objective {training.tolist()} is not "
            "proportional to the Table-1 physical objective"
        )
    return (
        policy.query,
        policy.close,
        {
            "native_intervals": native_intervals,
            "load_seconds": load_seconds,
            "training_objective_weights": training.tolist(),
            "objective_scale_to_physical": float(ratios[0]),
            **extra,
        },
    )


def sample_statistics(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "sample_std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q95": float(np.quantile(array, 0.95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHOD_CHOICES, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="n=800 control NPZ, feedback checkpoint, or DeepBSDE checkpoint directory",
    )
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-label", default="selected")
    parser.add_argument("--radius", type=float, default=0.20)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--objective-substeps", type=int, default=4)
    parser.add_argument("--diagnostic-refinement", type=int, default=32)
    parser.add_argument("--interior-start", type=float, default=1.5)
    parser.add_argument("--interior-end", type=float, default=8.0)
    parser.add_argument("--hjb-tau", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0 or args.batch_size <= 0:
        raise ValueError("samples and batch-size must be positive")
    if args.objective_substeps <= 0 or args.diagnostic_refinement <= 0:
        raise ValueError("substep/refinement counts must be positive")
    if not 0.0 <= args.interior_start < args.interior_end <= 10.0:
        raise ValueError("invalid diagnostic interval")
    if args.method == "direct_nominal_replay" and "resistant" in args.source.name.lower():
        raise ValueError(
            "direct_nominal_replay requires the nominal direct control artifact"
        )

    problem = Problem()
    vectors = problem.vectors()
    source = resolve(args.source)
    directions_path = resolve(args.directions)
    output_dir = resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} already exists; use a new directory for an auditable run"
        )
    if not source.exists():
        raise FileNotFoundError(source)
    if not directions_path.is_file():
        raise FileNotFoundError(directions_path)
    output_dir.mkdir(parents=True)

    directions = load_directions(directions_path, args.samples, problem.m)
    initial_states = problem.n0 * (1.0 + args.radius * directions)
    if float(np.min(initial_states)) <= 0.0:
        raise ValueError("held-out initial state is not positive")

    time_only = args.method in {
        "direct_nominal_replay",
        "neural_pmp_learned",
        "neural_pmp_exact",
        "pmp_kkt_time",
    }
    fixed_control = load_control(source, problem) if time_only else None
    query: Callable[[int, float, np.ndarray], float] | None = None
    close = lambda: None
    policy_metadata: dict[str, Any] = {}
    if not time_only:
        query, close, policy_metadata = load_policy(
            args.method, source, hjb_tau=args.hjb_tau
        )

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for batch_start in range(0, args.samples, args.batch_size):
            batch_end = min(args.samples, batch_start + args.batch_size)
            initial = initial_states[batch_start:batch_end]
            controls, objectives = rollout_controls(
                initial,
                problem,
                vectors,
                fixed_control=fixed_control,
                query=query,
                objective_substeps=args.objective_substeps,
            )
            rms_psi, rms_dot, rms_ddot, residual = singular_residual(
                initial,
                controls,
                problem,
                vectors,
                refinement=args.diagnostic_refinement,
                interior_start=args.interior_start,
                interior_end=args.interior_end,
            )
            for local_index, sample in enumerate(range(batch_start, batch_end)):
                rows.append(
                    {
                        "method": args.method,
                        "seed_label": args.seed_label,
                        "sample": sample,
                        "radius": args.radius,
                        "J": f"{objectives[local_index]:.12g}",
                        "RMS_H_u": f"{rms_psi[local_index]:.12g}",
                        "RMS_dH_u_dt": f"{rms_dot[local_index]:.12g}",
                        "RMS_d2H_u_dt2": f"{rms_ddot[local_index]:.12g}",
                        "R_sing": f"{residual[local_index]:.12g}",
                        "initial_state": json.dumps(
                            initial[local_index].tolist(), separators=(",", ":")
                        ),
                        "control_semantics": (
                            "fixed nominal direct schedule replayed unchanged"
                            if args.method == "direct_nominal_replay"
                            else (
                                "fixed learned time-only schedule replayed unchanged"
                                if time_only
                                else "closed-loop policy queried at each n=800 left endpoint"
                            )
                        ),
                    }
                )
            print(
                f"[{args.method}/{args.seed_label}] "
                f"completed samples {batch_start}:{batch_end}",
                flush=True,
            )
    finally:
        close()

    elapsed = time.perf_counter() - started
    write_csv(output_dir / "per_sample.csv", rows)
    objective = np.asarray([float(row["J"]) for row in rows])
    residual = np.asarray([float(row["R_sing"]) for row in rows])
    summary = {
        "schema": "table1-heldout-common-v1",
        "method": args.method,
        "seed_label": args.seed_label,
        "source": str(source),
        "source_sha256": sha256(source) if source.is_file() else None,
        "directions": str(directions_path),
        "directions_sha256": sha256(directions_path),
        "problem": problem.__dict__,
        "protocol": {
            "radius": args.radius,
            "samples": args.samples,
            "initial_state_formula": "N_i(0)=10*(1+radius*direction_i)",
            "control_grid": "n=800 left-endpoint zero-order hold",
            "objective_integrator": (
                f"float64 classical RK4, {args.objective_substeps} substeps "
                "per control interval, matching RK4 running-cost quadrature"
            ),
            "scalar_diagnostic": (
                "analytic continuous-PMP H_u, dH_u/dt, and d2H_u/dt2 "
                f"on q={args.diagnostic_refinement} refined ZOH nodes"
            ),
            "diagnostic_interval": [args.interior_start, args.interior_end],
            "diagnostic_interval_semantics": "half-open",
            "R_sing": "sqrt(RMS(H_u)^2+RMS(dH_u/dt)^2+RMS(d2H_u/dt2)^2)",
        },
        "control_semantics": rows[0]["control_semantics"],
        "policy_metadata": policy_metadata,
        "objective_J": sample_statistics(objective),
        "R_sing": sample_statistics(residual),
        "elapsed_seconds": elapsed,
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "outputs": {
            "per_sample_csv": str((output_dir / "per_sample.csv").resolve()),
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
