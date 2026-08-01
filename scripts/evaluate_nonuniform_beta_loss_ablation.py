#!/usr/bin/env python3
"""Independent two-semantics audit for the nonuniform-beta pilot.

This evaluator never trains or modifies a checkpoint.  It evaluates one
retained candidate under two explicitly separated control extensions:

1. ``continuous_stage_query``: a neural policy uses the retained fixed-support
   Transformer query rule and is queried at every RK4 stage.  The discrete
   direct reference is given an explicitly labelled PCHIP extension through
   its n=800 support-node values.
2. ``support_zoh``: the same candidate is sampled at its n=800 support nodes;
   the first 800 values are replayed as left-endpoint interval constants.

Both evaluations use the physical objective weights alpha_i=1,
beta_i=30,...,50, gamma=8000, 25,600 fixed RK4 steps, and the strict
diagnostic set [1.5,8) intersect (T_32 minus T_8), containing 12,480 points.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.interpolate import PchipInterpolator


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT, ROOT / "scripts"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts.evaluate_table1_baseline_continuous_strict import (  # noqa: E402
    batch_dh_dn,
    batch_dynamics,
    common_parameters,
    strict_quantities_batch,
)
from scripts.evaluate_table1_feedback_continuous_strict_fixed_rk4 import (  # noqa: E402
    forward_derivative,
    load_proposed_dense_policy,
    query_policy_grid,
)


FIXED_STEPS = 25_600
INTERIOR_START = 1.5
INTERIOR_END = 8.0


@dataclass(frozen=True)
class NonuniformProblem:
    T: float = 10.0
    intervals: int = 800
    m: int = 21
    umax: float = 3.0
    alpha: float = 1.0
    gamma: float = 8000.0
    n0: float = 10.0
    suppression: float = 0.5

    def vectors(self) -> dict[str, np.ndarray]:
        phenotype = np.linspace(0.0, 1.0, self.m, dtype=np.float64)
        return {
            "r": 2.0 / (1.0 + 3.0 * phenotype**4),
            "phi": 1.0 / (1.0 + phenotype**2),
            "M": np.full(self.m, self.suppression, dtype=np.float64),
            "beta": 40.0 * (0.75 + 0.5 * phenotype),
            "alpha": np.full(self.m, self.alpha, dtype=np.float64),
        }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
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
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


class NodePchipPolicy:
    """Explicit continuous extension through n=800 support-node actions."""

    def __init__(self, time_grid: np.ndarray, control: np.ndarray, umax: float):
        self.time_grid = np.asarray(time_grid, dtype=np.float64)
        self.control = np.asarray(control, dtype=np.float64)
        self.umax = float(umax)
        self.interpolator = PchipInterpolator(
            self.time_grid, self.control, extrapolate=False
        )

    def __call__(self, physical_time: float, states: np.ndarray) -> np.ndarray:
        value = float(
            self.interpolator(
                float(np.clip(physical_time, self.time_grid[0], self.time_grid[-1]))
            )
        )
        return np.full(
            np.asarray(states).shape[0],
            np.clip(value, 0.0, self.umax),
            dtype=np.float64,
        )

    def query_grid(
        self, physical_times: np.ndarray, states: np.ndarray
    ) -> np.ndarray:
        times = np.asarray(physical_times, dtype=np.float64)
        values = np.clip(self.interpolator(times), 0.0, self.umax)
        batch = np.asarray(states).shape[1]
        return np.broadcast_to(values[:, None], (times.size, batch)).copy()


class ZOHPolicy:
    """Left-endpoint hold of the first 800 support actions."""

    def __init__(self, support_control: np.ndarray, problem: NonuniformProblem):
        values = np.asarray(support_control, dtype=np.float64).reshape(-1)
        if values.size != problem.intervals + 1:
            raise ValueError(f"expected 801 support values, found {values.size}")
        self.interval = np.clip(values[:-1], 0.0, problem.umax)
        self.problem = problem

    def values_at(self, physical_times: np.ndarray) -> np.ndarray:
        scaled = np.asarray(physical_times, dtype=np.float64) / self.problem.T
        indices = np.minimum(
            np.floor(scaled * self.problem.intervals).astype(np.int64),
            self.problem.intervals - 1,
        )
        return self.interval[indices]

    def __call__(self, physical_time: float, states: np.ndarray) -> np.ndarray:
        value = float(self.values_at(np.asarray([physical_time]))[0])
        return np.full(np.asarray(states).shape[0], value, dtype=np.float64)

    def query_grid(
        self, physical_times: np.ndarray, states: np.ndarray
    ) -> np.ndarray:
        values = self.values_at(physical_times)
        batch = np.asarray(states).shape[1]
        return np.broadcast_to(values[:, None], (values.size, batch)).copy()


def evaluate_policy(
    policy: Callable[[float, np.ndarray], np.ndarray],
    problem: NonuniformProblem,
    dense_time: np.ndarray,
    strict_mask: np.ndarray,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Fixed-step RK4 state, open-loop adjoint, and scalar diagnostics."""

    started = time.perf_counter()
    parameters = common_parameters(problem)
    components = problem.m
    batch = 1
    step = problem.T / FIXED_STEPS
    fine_time = dense_time
    if fine_time.size != FIXED_STEPS + 1:
        raise ValueError("dense time must be the T_32 integration grid")

    states = np.empty((FIXED_STEPS + 1, batch, components), dtype=np.float64)
    running = np.empty((FIXED_STEPS + 1, batch), dtype=np.float64)
    states[0, 0] = problem.n0
    running[0] = 0.0
    for index in range(FIXED_STEPS):
        current_time = fine_time[index]
        current_state = states[index]

        control1 = policy(current_time, current_state)
        state1, cost1 = forward_derivative(current_state, control1, parameters)
        stage2_state = current_state + 0.5 * step * state1
        control2 = policy(current_time + 0.5 * step, stage2_state)
        state2, cost2 = forward_derivative(stage2_state, control2, parameters)
        stage3_state = current_state + 0.5 * step * state2
        control3 = policy(current_time + 0.5 * step, stage3_state)
        state3, cost3 = forward_derivative(stage3_state, control3, parameters)
        stage4_state = current_state + step * state3
        control4 = policy(current_time + step, stage4_state)
        state4, cost4 = forward_derivative(stage4_state, control4, parameters)

        states[index + 1] = current_state + (step / 6.0) * (
            state1 + 2.0 * state2 + 2.0 * state3 + state4
        )
        running[index + 1] = running[index] + (step / 6.0) * (
            cost1 + 2.0 * cost2 + 2.0 * cost3 + cost4
        )
    if not np.all(np.isfinite(states)) or float(states.min()) <= 0.0:
        raise RuntimeError("forward integration failed")

    node_controls = query_policy_grid(policy, fine_time, states)
    node_drift = batch_dynamics(
        states.reshape(-1, components),
        node_controls.reshape(-1),
        parameters,
    ).reshape(states.shape)
    midpoint_states = (
        0.5 * (states[:-1] + states[1:])
        + (step / 8.0) * (node_drift[:-1] - node_drift[1:])
    )
    midpoint_time = 0.5 * (fine_time[:-1] + fine_time[1:])
    midpoint_controls = query_policy_grid(policy, midpoint_time, midpoint_states)

    costates = np.empty_like(states)
    costates[-1, 0] = parameters["alpha"]

    def rhs(
        state: np.ndarray, costate: np.ndarray, control: np.ndarray
    ) -> np.ndarray:
        return -batch_dh_dn(state, costate, control, parameters)

    for index in range(FIXED_STEPS - 1, -1, -1):
        current = costates[index + 1]
        slope1 = rhs(states[index + 1], current, node_controls[index + 1])
        slope2 = rhs(
            midpoint_states[index],
            current - 0.5 * step * slope1,
            midpoint_controls[index],
        )
        slope3 = rhs(
            midpoint_states[index],
            current - 0.5 * step * slope2,
            midpoint_controls[index],
        )
        slope4 = rhs(
            states[index], current - step * slope3, node_controls[index]
        )
        costates[index] = current - (step / 6.0) * (
            slope1 + 2.0 * slope2 + 2.0 * slope3 + slope4
        )
    if not np.all(np.isfinite(costates)):
        raise RuntimeError("backward integration failed")

    selected_time = dense_time[strict_mask]
    selected_states = states[strict_mask]
    selected_costates = costates[strict_mask]
    selected_controls = query_policy_grid(
        policy, selected_time, selected_states
    )
    psi, dot_psi, ddot_psi = strict_quantities_batch(
        selected_states, selected_costates, selected_controls, parameters
    )
    quantities = {
        "psi": psi[:, 0],
        "dot_psi": dot_psi[:, 0],
        "ddot_psi": ddot_psi[:, 0],
    }
    objective = float(
        running[-1, 0]
        + states[-1, 0] @ np.asarray(parameters["alpha"], dtype=np.float64)
    )
    result: dict[str, float] = {"J": objective}
    for label, values in quantities.items():
        result[f"{label}_rms"] = float(np.sqrt(np.mean(values**2)))
        result[f"{label}_max_abs"] = float(np.max(np.abs(values)))
    result["epsilon"] = float(
        math.sqrt(
            result["psi_rms"] ** 2
            + result["dot_psi_rms"] ** 2
            + result["ddot_psi_rms"] ** 2
        )
    )
    metadata = {
        "elapsed_seconds": time.perf_counter() - started,
        "fixed_rk4_steps": FIXED_STEPS,
        "strict_point_count": int(strict_mask.sum()),
    }
    return result, metadata


def control_structure(
    support_time: np.ndarray,
    support_control: np.ndarray,
    problem: NonuniformProblem,
) -> dict[str, Any]:
    values = np.asarray(support_control, dtype=np.float64)
    upper_threshold = problem.umax - 0.05
    lower_threshold = 0.05
    leave_upper_candidates = np.flatnonzero(values < upper_threshold)
    if leave_upper_candidates.size == 0:
        leave_upper = None
        leave_index = 0
    else:
        leave_index = int(leave_upper_candidates[0])
        leave_upper = float(support_time[leave_index])
    after_midpoint = np.flatnonzero(
        (support_time >= 0.5 * problem.T) & (values <= lower_threshold)
    )
    if after_midpoint.size == 0:
        reach_lower = None
        lower_index = values.size - 1
    else:
        lower_index = int(after_midpoint[0])
        reach_lower = float(support_time[lower_index])
    interior = values[leave_index:lower_index]
    early = values[: max(leave_index, 1)]
    late = values[lower_index:]
    early_median = float(np.median(early))
    interior_median = float(np.median(interior)) if interior.size else None
    late_median = float(np.median(late))
    pass_pattern = bool(
        leave_upper is not None
        and reach_lower is not None
        and leave_index < lower_index
        and early_median >= upper_threshold
        and interior_median is not None
        and lower_threshold < interior_median < upper_threshold
        and late_median <= lower_threshold
    )
    return {
        "leave_upper_time_threshold_2.95": leave_upper,
        "reach_lower_time_threshold_0.05": reach_lower,
        "early_median": early_median,
        "interior_median": interior_median,
        "late_median": late_median,
        "high_interior_low": pass_pattern,
        "upper_support_count": int(np.count_nonzero(values >= upper_threshold)),
        "lower_support_count": int(np.count_nonzero(values <= lower_threshold)),
        "support_total_variation": float(np.abs(np.diff(values)).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Selected Transformer checkpoint. Omit only for the Direct row.",
    )
    parser.add_argument(
        "--direct-npz",
        type=Path,
        required=True,
        help="Nonuniform-beta n=800 Direct artifact containing u, beta, alpha, gamma.",
    )
    parser.add_argument(
        "--source-file",
        action="append",
        default=[],
        type=Path,
        help="Optional additional provenance file to hash; may be repeated.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)

    problem = NonuniformProblem()
    beta = problem.vectors()["beta"]
    dense_time = np.linspace(0.0, problem.T, FIXED_STEPS + 1)
    # T_8 nodes occur every four T_32 nodes.  The strict set excludes all of
    # them, not merely the n=800 base nodes.
    strict_mask = (
        (dense_time >= INTERIOR_START)
        & (dense_time < INTERIOR_END)
        & (np.arange(dense_time.size) % 4 != 0)
    )
    if int(strict_mask.sum()) != 12_480:
        raise RuntimeError("strict-set size drifted")
    support_time = np.linspace(0.0, problem.T, problem.intervals + 1)
    direct_path = args.direct_npz.expanduser().resolve()
    with np.load(direct_path, allow_pickle=False) as source:
        direct_support = np.asarray(source["u"], dtype=np.float64)
        stored_beta = np.asarray(source["beta"], dtype=np.float64)
        stored_alpha = np.asarray(source["alpha"], dtype=np.float64)
        stored_gamma = float(source["gamma"])
    if not (
        np.array_equal(stored_beta, beta)
        and np.array_equal(stored_alpha, np.ones(problem.m))
        and stored_gamma == problem.gamma
    ):
        raise RuntimeError("direct artifact does not use beta=30,...,50")

    cache_record = None
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else None
    if checkpoint is None:
        support_control = direct_support.copy()
        continuous_policy: Any = NodePchipPolicy(
            support_time, support_control, problem.umax
        )
        continuous_semantics = (
            "evaluator-defined PCHIP through direct n=800 support-node actions"
        )
        policy_metadata: dict[str, Any] = {
            "native_object": "n=800 direct ZOH vector",
            "continuous_extension_is_not_an_optimized_direct_object": True,
        }
        source_files = [direct_path]
    else:
        cache_path = output.with_suffix(".policy_cache.npz")
        retained_cache = cache_path if cache_path.is_file() else None
        continuous_policy, policy_metadata, cache_record = (
            load_proposed_dense_policy(
                checkpoint,
                "neural_pmp_time",
                dense_time,
                problem,
                query_batch_size=args.query_batch_size,
                torch_threads=args.torch_threads,
                policy_cache=retained_cache,
            )
        )
        if retained_cache is None:
            if cache_record is None:
                raise RuntimeError("fresh policy load did not return a cache")
            np.savez_compressed(cache_path, **cache_record)
        nominal_states = np.full(
            (support_time.size, 1, problem.m), problem.n0, dtype=np.float64
        )
        support_control = query_policy_grid(
            continuous_policy, support_time, nominal_states
        )[:, 0]
        continuous_semantics = (
            "fixed-support Transformer queries on T_32, raw-logit PCHIP "
            "extension, action queried at every RK4 stage"
        )
        source_files = [direct_path, checkpoint]

    source_files.extend(path.expanduser().resolve() for path in args.source_file)
    missing_sources = [path for path in source_files if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(f"missing provenance files: {missing_sources}")

    support_policy = ZOHPolicy(support_control, problem)
    continuous_result, continuous_timing = evaluate_policy(
        continuous_policy, problem, dense_time, strict_mask
    )
    zoh_result, zoh_timing = evaluate_policy(
        support_policy, problem, dense_time, strict_mask
    )
    direct_continuous = NodePchipPolicy(
        support_time, direct_support, problem.umax
    )
    nominal_states_dense = np.full(
        (dense_time.size, 1, problem.m), problem.n0, dtype=np.float64
    )
    candidate_dense = query_policy_grid(
        continuous_policy, dense_time, nominal_states_dense
    )[:, 0]
    direct_dense = query_policy_grid(
        direct_continuous, dense_time, nominal_states_dense
    )[:, 0]

    payload = {
        "schema": "nonuniform-beta-two-semantics-independent-audit-v1",
        "candidate": args.candidate_name,
        "problem": {
            "alpha": [1.0] * problem.m,
            "beta": beta,
            "beta_formula": "40*(0.75+0.5*x_i), x_i=i/20",
            "gamma": problem.gamma,
            "initial_state": [problem.n0] * problem.m,
        },
        "protocol": {
            "fixed_rk4_steps": FIXED_STEPS,
            "diagnostic_set": "[1.5,8.0) intersect (T_32 minus T_8)",
            "diagnostic_point_count": int(strict_mask.sum()),
            "continuous_policy_semantics": continuous_semantics,
            "support_semantics": "first 800 n=800 support actions, left-endpoint ZOH",
        },
        "source_files": [
            {"path": str(path), "sha256": sha256(path)} for path in source_files
        ],
        "policy_metadata": policy_metadata,
        "control_comparison_to_direct": {
            "support_rmse_801": float(
                np.sqrt(np.mean((support_control - direct_support) ** 2))
            ),
            "support_linf_801": float(
                np.max(np.abs(support_control - direct_support))
            ),
            "continuous_T32_rmse": float(
                np.sqrt(np.mean((candidate_dense - direct_dense) ** 2))
            ),
            "continuous_T32_linf": float(
                np.max(np.abs(candidate_dense - direct_dense))
            ),
        },
        "structure_from_support": control_structure(
            support_time, support_control, problem
        ),
        "structure_from_continuous_T32": control_structure(
            dense_time, candidate_dense, problem
        ),
        "continuous_stage_query": continuous_result,
        "support_zoh": zoh_result,
        "timing": {
            "continuous": continuous_timing,
            "support_zoh": zoh_timing,
        },
    }
    output.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_safe(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
