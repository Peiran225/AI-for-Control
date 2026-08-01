#!/usr/bin/env python3
"""Unified fixed-step continuous-policy evaluator for Table-1 methods.

DeepBSDE and PI-DeepONet extract exact endpoint actions.  Their continuously
queried closed-loop vector fields can therefore be discontinuous at learned
switching surfaces.  Adaptive solvers may spend an unbounded amount of work
trying to resolve those surfaces.  This post-training evaluator applies one
deterministic integration protocol to every Table-1 method without smoothing,
holding, retraining, or changing a policy:

* every classical RK4 stage queries the original continuous policy;
* stored n=800 schedules use the declared midpoint-PCHIP extension;
* feedback checkpoints are queried at every RK stage;
* the main fixed step is T/25,600 and the convergence audit uses T/12,800;
* the open-loop PMP adjoint is integrated backward with RK4, using a cubic
  Hermite reconstruction of the realized state at each stage midpoint; and
* the same 16,120 strict off-n=800 coordinates and epsilon definition are used.

T_32 diagnostic states and costates are exact integration nodes in the main
run.  For the coarser T/12,800 audit, the interlaced half-step diagnostics use
cubic Hermite reconstruction from the converged RK4 node values and endpoint
derivatives.
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
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for search_path in (ROOT, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from evaluate_table1_baseline_continuous_strict import (  # noqa: E402
    LoadedFeedbackPolicy,
    SchedulePolicy,
    batch_dh_dn,
    batch_dynamics,
    common_parameters,
    json_safe,
    load_deepbsde_batch_policy,
    load_interval_control,
    load_pi_batch_policy,
    sha256,
    strict_quantities_batch,
    strict_time_grid,
)
from evaluate_table1_heldout_common import Problem  # noqa: E402


METHODS = (
    "direct_nominal_replay",
    "neural_pmp_learned",
    "neural_pmp_exact",
    "adaptive_hjb_nn",
    "pi_deeponet",
    "deepbsde",
    "neural_pmp_time",
    "neural_pmp_cf",
    "neural_pmp_der",
)


def resolve(path: Path) -> Path:
    expanded = path.expanduser()
    return (
        expanded.resolve()
        if expanded.is_absolute()
        else (ROOT / expanded).resolve()
    )


def forward_derivative(
    states: np.ndarray,
    controls: np.ndarray,
    parameters: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    drift = batch_dynamics(states, controls, parameters)
    running = (
        states @ np.asarray(parameters["beta"], dtype=np.float64)
        + float(parameters["gamma"]) * controls
    )
    return drift, running


def query_policy_grid(
    policy: Callable[[float, np.ndarray], np.ndarray],
    physical_times: np.ndarray,
    states: np.ndarray,
) -> np.ndarray:
    """Query a policy on a time/state grid without introducing a hold."""

    if hasattr(policy, "query_grid"):
        return np.asarray(
            policy.query_grid(physical_times, states), dtype=np.float64
        )
    return np.stack(
        [
            policy(float(current_time), current_states)
            for current_time, current_states in zip(
                np.asarray(physical_times, dtype=np.float64),
                np.asarray(states, dtype=np.float64),
            )
        ],
        axis=0,
    )


def cubic_hermite_grid(
    node_time: np.ndarray,
    node_values: np.ndarray,
    node_derivatives: np.ndarray,
    query_time: np.ndarray,
) -> np.ndarray:
    """Reconstruct values between uniform integration nodes."""

    times = np.asarray(node_time, dtype=np.float64)
    queries = np.asarray(query_time, dtype=np.float64)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("integration-node times are invalid")
    step = float(times[1] - times[0])
    scaled = np.clip(queries / step, 0.0, times.size - 1)
    left = np.minimum(
        np.floor(scaled).astype(np.int64), times.size - 2
    )
    fraction = scaled - left
    terminal = queries >= times[-1]
    left[terminal] = times.size - 2
    fraction[terminal] = 1.0
    s = fraction.reshape((-1,) + (1,) * (node_values.ndim - 1))
    h00 = 2.0 * s**3 - 3.0 * s**2 + 1.0
    h10 = s**3 - 2.0 * s**2 + s
    h01 = -2.0 * s**3 + 3.0 * s**2
    h11 = s**3 - s**2
    return (
        h00 * node_values[left]
        + h10 * step * node_derivatives[left]
        + h01 * node_values[left + 1]
        + h11 * step * node_derivatives[left + 1]
    )


class DensePolicyBatchAdapter:
    """Expose the retained proposed-method DensePolicy in batch form."""

    def __init__(self, policy: Any) -> None:
        self.policy = policy

    def __call__(
        self, physical_time: float, states: np.ndarray
    ) -> np.ndarray:
        return np.asarray(
            [
                self.policy.action(float(physical_time), state)
                for state in np.asarray(states, dtype=np.float64)
            ],
            dtype=np.float64,
        )

    def query_grid(
        self, physical_times: np.ndarray, states: np.ndarray
    ) -> np.ndarray:
        return np.stack(
            [
                self(float(current_time), current_states)
                for current_time, current_states in zip(
                    np.asarray(physical_times, dtype=np.float64),
                    np.asarray(states, dtype=np.float64),
                )
            ],
            axis=0,
        )


def load_proposed_dense_policy(
    checkpoint: Path,
    method: str,
    dense_time: np.ndarray,
    problem: Problem,
    *,
    query_batch_size: int,
    torch_threads: int,
    policy_cache: Path | None,
) -> tuple[
    DensePolicyBatchAdapter,
    dict[str, Any],
    dict[str, np.ndarray] | None,
]:
    """Load Time/CF/DER with the retained fixed-support time-query rule."""

    import torch

    from evaluate_feedback_section5 import load_feedback_checkpoint
    from generate_offgrid_policy_switching_diagnostics import (
        DensePolicy,
        build_grid_flags,
        fixed_support_query_logits,
        load_time_model,
        raw_time_logits,
        validate_numpy_feedback_action,
    )

    torch.set_num_threads(torch_threads)
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    if method == "neural_pmp_time":
        model, config, wrapper = load_time_model(checkpoint)
        feedback_model = None
        feedback_args = None
    else:
        model, config, feedback_args = load_feedback_checkpoint(checkpoint)
        wrapper = None
        feedback_model = model
    for key, expected in {
        "T": problem.T,
        "m": problem.m,
        "n": problem.intervals,
        "n0": problem.n0,
        "umax": problem.umax,
    }.items():
        actual = getattr(config, key)
        if not np.isclose(actual, expected):
            raise ValueError(
                f"{checkpoint}: problem {key}={actual} != {expected}"
            )
    checkpoint_digest = sha256(checkpoint)
    support = torch.linspace(
        0.0, 1.0, config.n + 1, dtype=torch.float64
    )
    cache_record: dict[str, np.ndarray] | None = None
    if policy_cache is not None:
        with np.load(policy_cache, allow_pickle=False) as payload:
            cached_digest = str(payload["checkpoint_sha256"].item())
            cached_time = np.asarray(
                payload["dense_time"], dtype=np.float64
            )
            dense_raw = np.asarray(
                payload["dense_raw_logits"], dtype=np.float64
            )
            support_raw = np.asarray(
                payload["support_raw_logits"], dtype=np.float64
            )
        if cached_digest != checkpoint_digest:
            raise ValueError("policy cache checkpoint hash mismatch")
        if not np.array_equal(cached_time, dense_time):
            raise ValueError("policy cache does not use the T_32 grid")
        cache_status = "loaded and checkpoint-hash verified"
    else:
        started = time.perf_counter()
        model = model.to(dtype=torch.float64, device="cpu")
        time_branch = (
            model
            if method == "neural_pmp_time"
            else model.time_branch
        )
        normalized = torch.from_numpy(dense_time / config.T).to(
            dtype=torch.float64
        )
        on_support, nearest_index, _ = build_grid_flags(
            dense_time, config
        )
        with torch.inference_mode():
            support_tensor = raw_time_logits(time_branch, support)
        dense_tensor = torch.empty_like(normalized)
        support_mask = torch.from_numpy(on_support)
        support_indices = torch.from_numpy(nearest_index[on_support])
        dense_tensor[support_mask] = support_tensor[support_indices]
        dense_tensor[~support_mask] = fixed_support_query_logits(
            time_branch,
            support,
            normalized[~support_mask],
            batch_size=query_batch_size,
        )
        dense_raw = dense_tensor.detach().cpu().numpy()
        support_raw = support_tensor.detach().cpu().numpy()
        cache_record = {
            "checkpoint_sha256": np.asarray(checkpoint_digest),
            "dense_time": dense_time,
            "dense_raw_logits": dense_raw,
            "support_raw_logits": support_raw,
        }
        cache_status = (
            f"fresh fixed-support queries in "
            f"{time.perf_counter() - started:.6g} seconds"
        )
    model = model.cpu()
    dense_policy = DensePolicy(
        (
            "time_only"
            if method == "neural_pmp_time"
            else "feedback_der"
        ),
        config,
        dense_time,
        dense_raw,
        support_raw,
        checkpoint,
        feedback_model=feedback_model,
        feedback_args=feedback_args,
        time_wrapper=wrapper,
    )
    validation_error = (
        None
        if method == "neural_pmp_time"
        else validate_numpy_feedback_action(dense_policy)
    )
    metadata = {
        "checkpoint_sha256": checkpoint_digest,
        "native_intervals": int(config.n),
        "time_context_tokens": int(config.n + 1),
        "time_query_grid": "T_32 fixed-support Transformer queries",
        "raw_time_extension": "PCHIP through fresh T_32 raw logits",
        "state_correction_execution": (
            "none (time-only)"
            if method == "neural_pmp_time"
            else "live NumPy checkpoint query at every RK4 stage"
        ),
        "policy_cache": cache_status,
        "numpy_vs_torch_action_max_abs": validation_error,
    }
    return DensePolicyBatchAdapter(dense_policy), metadata, cache_record


def fixed_rk4_evaluate(
    policy: Callable[[float, np.ndarray], np.ndarray],
    initial_states: np.ndarray,
    dense_time: np.ndarray,
    strict_mask: np.ndarray,
    problem: Problem,
    *,
    fixed_steps: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Evaluate one fixed policy with stagewise continuous-policy RK4."""

    started = time.perf_counter()
    initial_states = np.asarray(initial_states, dtype=np.float64)
    batch, components = initial_states.shape
    dense_intervals = dense_time.size - 1
    fine_intervals = int(fixed_steps)
    if fine_intervals < 1:
        raise ValueError("fixed_steps must be positive")
    if (
        dense_intervals % fine_intervals != 0
        and fine_intervals % dense_intervals != 0
    ):
        raise ValueError(
            "fixed integration grid and T_32 diagnostic grid must be nested"
        )
    step = problem.T / fine_intervals
    fine_time = np.linspace(
        0.0, problem.T, fine_intervals + 1, dtype=np.float64
    )
    parameters = common_parameters(problem)

    # Forward state and running objective.  Policy queries occur at all four
    # RK stages, so this is continuous feedback execution rather than ZOH.
    states = np.empty(
        (fine_intervals + 1, batch, components), dtype=np.float64
    )
    running = np.empty((fine_intervals + 1, batch), dtype=np.float64)
    states[0] = initial_states
    running[0] = 0.0
    forward_started = time.perf_counter()
    for index in range(fine_intervals):
        current_time = fine_time[index]
        current_state = states[index]
        current_running = running[index]

        control1 = policy(current_time, current_state)
        state1, cost1 = forward_derivative(
            current_state, control1, parameters
        )

        stage2_state = current_state + 0.5 * step * state1
        control2 = policy(
            current_time + 0.5 * step, stage2_state
        )
        state2, cost2 = forward_derivative(
            stage2_state, control2, parameters
        )

        stage3_state = current_state + 0.5 * step * state2
        control3 = policy(
            current_time + 0.5 * step, stage3_state
        )
        state3, cost3 = forward_derivative(
            stage3_state, control3, parameters
        )

        stage4_state = current_state + step * state3
        control4 = policy(current_time + step, stage4_state)
        state4, cost4 = forward_derivative(
            stage4_state, control4, parameters
        )

        states[index + 1] = current_state + (step / 6.0) * (
            state1 + 2.0 * state2 + 2.0 * state3 + state4
        )
        running[index + 1] = current_running + (step / 6.0) * (
            cost1 + 2.0 * cost2 + 2.0 * cost3 + cost4
        )
    forward_seconds = time.perf_counter() - forward_started
    if (
        not np.all(np.isfinite(states))
        or float(states.min()) <= 0.0
        or not np.all(np.isfinite(running))
    ):
        raise RuntimeError("fixed-step forward integration is invalid")

    # Query whole grids in subnet-grouped batches.  Cubic Hermite state
    # midpoints are fourth-order accurate when the local branch is smooth.
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
    midpoint_controls = query_policy_grid(
        policy, midpoint_time, midpoint_states
    )

    # Open-loop PMP adjoint along the realized feedback trajectory.  The
    # realized action is held fixed inside partial H / partial N.
    costates = np.empty_like(states)
    costates[-1] = np.broadcast_to(
        np.asarray(parameters["alpha"], dtype=np.float64),
        (batch, components),
    )

    def adjoint_rhs(
        state: np.ndarray,
        costate: np.ndarray,
        control: np.ndarray,
    ) -> np.ndarray:
        return -batch_dh_dn(
            state, costate, control, parameters
        )

    backward_started = time.perf_counter()
    for index in range(fine_intervals - 1, -1, -1):
        current = costates[index + 1]
        slope1 = adjoint_rhs(
            states[index + 1], current, node_controls[index + 1]
        )
        slope2 = adjoint_rhs(
            midpoint_states[index],
            current - 0.5 * step * slope1,
            midpoint_controls[index],
        )
        slope3 = adjoint_rhs(
            midpoint_states[index],
            current - 0.5 * step * slope2,
            midpoint_controls[index],
        )
        slope4 = adjoint_rhs(
            states[index],
            current - step * slope3,
            node_controls[index],
        )
        costates[index] = current - (step / 6.0) * (
            slope1 + 2.0 * slope2 + 2.0 * slope3 + slope4
        )
    backward_seconds = time.perf_counter() - backward_started
    if not np.all(np.isfinite(costates)):
        raise RuntimeError("fixed-step backward integration is invalid")

    # The main T/25,600 run samples T_32 directly at integration nodes.  The
    # coarser T/12,800 convergence audit reconstructs interlaced half-step
    # diagnostics with cubic Hermite interpolation.
    costate_node_derivatives = -batch_dh_dn(
        states.reshape(-1, components),
        costates.reshape(-1, components),
        node_controls.reshape(-1),
        parameters,
    ).reshape(costates.shape)
    if fine_intervals % dense_intervals == 0:
        stride = fine_intervals // dense_intervals
        dense_states = states[::stride]
        dense_costates = costates[::stride]
        diagnostic_nodes = True
    else:
        dense_states = cubic_hermite_grid(
            fine_time, states, node_drift, dense_time
        )
        dense_costates = cubic_hermite_grid(
            fine_time,
            costates,
            costate_node_derivatives,
            dense_time,
        )
        diagnostic_nodes = False
    selected_time = dense_time[strict_mask]
    selected_states = dense_states[strict_mask]
    selected_costates = dense_costates[strict_mask]
    selected_controls = query_policy_grid(
        policy, selected_time, selected_states
    )
    psi, dot_psi, ddot_psi = strict_quantities_batch(
        selected_states,
        selected_costates,
        selected_controls,
        parameters,
    )
    rms_psi = np.sqrt(np.mean(psi**2, axis=0))
    rms_dot = np.sqrt(np.mean(dot_psi**2, axis=0))
    rms_ddot = np.sqrt(np.mean(ddot_psi**2, axis=0))
    epsilon = np.sqrt(rms_psi**2 + rms_dot**2 + rms_ddot**2)
    objective = (
        running[-1]
        + states[-1]
        @ np.asarray(parameters["alpha"], dtype=np.float64)
    )
    results = {
        "J": objective,
        "RMS_psi": rms_psi,
        "RMS_dot_psi": rms_dot,
        "RMS_ddot_psi": rms_ddot,
        "epsilon": epsilon,
    }
    timing = {
        "elapsed_seconds": time.perf_counter() - started,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "fixed_steps": fine_intervals,
        "forward_policy_queries": 4 * fine_intervals,
        "forward_rhs_evaluations": 4 * fine_intervals,
        "backward_rhs_evaluations": 4 * fine_intervals,
        "diagnostic_points_are_integration_nodes": diagnostic_nodes,
    }
    return results, timing


def source_record(path: Path) -> dict[str, Any]:
    if path.is_file():
        return {
            "path": str(path),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }
    files = sorted(item for item in path.rglob("*") if item.is_file())
    return {
        "path": str(path),
        "directory_files": [
            {
                "path": str(item.relative_to(path)),
                "sha256": sha256(item),
                "bytes": item.stat().st_size,
            }
            for item in files
        ],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default="deepbsde")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--seed-label", default="selected")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resistant-radius", type=float, default=0.20)
    parser.add_argument("--dense-points", type=int, default=25_601)
    parser.add_argument("--interior-start", type=float, default=1.5)
    parser.add_argument("--interior-end", type=float, default=8.0)
    parser.add_argument("--hjb-tau", type=float, default=10.0)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument(
        "--policy-cache",
        type=Path,
        help=(
            "checkpoint-hash-verified T_32 raw-logit cache created by a "
            "prior main run of this evaluator"
        ),
    )
    step_group = parser.add_mutually_exclusive_group()
    step_group.add_argument(
        "--fixed-steps",
        type=int,
        help=(
            "number of uniform RK4 intervals on [0,T]; use 12800 for "
            "the coarse audit or 25600 for the selected main result"
        ),
    )
    step_group.add_argument(
        "--fixed-step-subdivisions",
        type=int,
        help=(
            "backward-compatible integer RK4 subdivisions per T_32 "
            "interval; omit to select exactly 25,600 fixed steps"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.fixed_step_subdivisions is not None
        and args.fixed_step_subdivisions < 1
    ):
        raise ValueError("--fixed-step-subdivisions must be positive")
    if args.fixed_steps is not None and args.fixed_steps < 1:
        raise ValueError("--fixed-steps must be positive")
    if not np.isclose(args.resistant_radius, 0.20):
        raise ValueError("approved Table-1 resistant radius is exactly 0.20")
    problem = Problem()
    source = resolve(args.source)
    policy_cache = (
        resolve(args.policy_cache)
        if args.policy_cache is not None
        else None
    )
    output_dir = resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} exists; choose a new immutable output directory"
        )
    if not source.exists():
        raise FileNotFoundError(source)
    if policy_cache is not None and not policy_cache.is_file():
        raise FileNotFoundError(policy_cache)

    dense_time, support, strict_mask = strict_time_grid(
        problem,
        dense_points=args.dense_points,
        interior_start=args.interior_start,
        interior_end=args.interior_end,
    )
    fixed_steps = (
        int(args.fixed_steps)
        if args.fixed_steps is not None
        else (args.dense_points - 1)
        * (
            int(args.fixed_step_subdivisions)
            if args.fixed_step_subdivisions is not None
            else 1
        )
    )
    nominal = np.full(problem.m, problem.n0, dtype=np.float64)
    direction = np.linspace(-1.0, 1.0, problem.m, dtype=np.float64)
    resistant = problem.n0 * (
        1.0 + args.resistant_radius * direction
    )
    initial_states = np.stack((nominal, resistant), axis=0)

    close: Callable[[], None] = lambda: None
    policy_metadata: dict[str, Any]
    policy_cache_record: dict[str, np.ndarray] | None = None
    if args.method == "deepbsde":
        deep_query, close, policy_metadata = (
            load_deepbsde_batch_policy(source, problem)
        )
        policy = LoadedFeedbackPolicy(
            lambda _index, _time, _state: float("nan"),
            method=args.method,
            native_intervals=int(policy_metadata["native_intervals"]),
            problem=problem,
            deep_batch_query=deep_query,
        )
        continuous_extension = (
            "linear interpolation of adjacent DeepBSDE native-slice "
            "actions queried at the same current state"
        )
    elif args.method == "pi_deeponet":
        batch_query, close, policy_metadata = load_pi_batch_policy(
            source, problem
        )
        policy = LoadedFeedbackPolicy(
            lambda _index, _time, _state: float("nan"),
            method=args.method,
            native_intervals=int(policy_metadata["native_intervals"]),
            problem=problem,
            batch_query=batch_query,
        )
        continuous_extension = (
            "native continuous physical-time feedback query"
        )
    elif args.method == "adaptive_hjb_nn":
        from evaluate_table1_heldout_common import load_policy

        query, close, policy_metadata = load_policy(
            args.method, source, hjb_tau=args.hjb_tau
        )
        policy = LoadedFeedbackPolicy(
            query,
            method=args.method,
            native_intervals=int(policy_metadata["native_intervals"]),
            problem=problem,
        )
        continuous_extension = (
            "native continuous physical-time/state feedback query"
        )
    elif args.method in {
        "neural_pmp_time",
        "neural_pmp_cf",
        "neural_pmp_der",
    }:
        policy, policy_metadata, policy_cache_record = (
            load_proposed_dense_policy(
                source,
                args.method,
                dense_time,
                problem,
                query_batch_size=args.query_batch_size,
                torch_threads=args.torch_threads,
                policy_cache=policy_cache,
            )
        )
        continuous_extension = (
            "fixed-support Transformer T_32 raw-logit queries with "
            "PCHIP time extension; "
            + (
                "time-only action queried at every RK4 stage"
                if args.method == "neural_pmp_time"
                else (
                    "checkpoint state correction queried live at every "
                    "RK4 stage"
                )
            )
        )
    else:
        interval_control = load_interval_control(source, problem)
        policy = SchedulePolicy(
            np.stack((interval_control, interval_control), axis=0),
            problem,
        )
        policy_metadata = {
            "native_intervals": int(problem.intervals),
            "schedule_shared_between_states": True,
            "stored_interval_values": int(interval_control.size),
        }
        continuous_extension = (
            "midpoint PCHIP(t_(k+1/2),u_k), with u(0)=u_0, "
            "u(T)=u_799, then clip to [0,3]"
        )
    try:
        result, timing = fixed_rk4_evaluate(
            policy,
            initial_states,
            dense_time,
            strict_mask,
            problem,
            fixed_steps=fixed_steps,
        )
    finally:
        close()

    state_ids = ("nominal", "resistant_heavy_r0p20")
    rows: list[dict[str, Any]] = []
    for index, state_id in enumerate(state_ids):
        rows.append(
            {
                "method": args.method,
                "seed_label": args.seed_label,
                "state_id": state_id,
                "J": f"{result['J'][index]:.15g}",
                "RMS_psi": f"{result['RMS_psi'][index]:.15g}",
                "RMS_dot_psi": f"{result['RMS_dot_psi'][index]:.15g}",
                "RMS_ddot_psi": f"{result['RMS_ddot_psi'][index]:.15g}",
                "epsilon": f"{result['epsilon'][index]:.15g}",
                "initial_state": json.dumps(
                    initial_states[index].tolist(), separators=(",", ":")
                ),
                "source": str(source),
                "continuous_extension": continuous_extension,
            }
        )

    output_dir.mkdir(parents=True)
    if policy_cache_record is not None:
        np.savez_compressed(
            output_dir / "policy_cache.npz", **policy_cache_record
        )
    write_csv(output_dir / "rows.csv", rows)
    fixed_step = problem.T / fixed_steps
    summary = {
        "schema": "table1-unified-continuous-strict-fixed-rk4-v2",
        "method": args.method,
        "seed_label": args.seed_label,
        "rows": rows,
        "problem": problem.__dict__,
        "sources": [source_record(source)],
        "policy_metadata": policy_metadata,
        "protocol": {
            "policy_execution": (
                "continuous stagewise query at every RK4 stage; "
                "no zero-order hold"
            ),
            "continuous_extension": continuous_extension,
            "state_solver": "classical fixed-step RK4",
            "costate": "open-loop PMP adjoint along realized trajectory",
            "costate_solver": (
                "classical fixed-step RK4 with cubic-Hermite state "
                "midpoint reconstruction"
            ),
            "fixed_step": fixed_step,
            "fixed_steps": fixed_steps,
            "fixed_step_subdivisions_per_T32_interval": (
                fixed_steps / (args.dense_points - 1)
            ),
            "dense_grid": "T_32",
            "dense_points": args.dense_points,
            "diagnostic_points_are_integration_nodes": timing[
                "diagnostic_points_are_integration_nodes"
            ],
            "coarse_diagnostic_reconstruction": (
                "not used; diagnostic points are integration nodes"
                if timing["diagnostic_points_are_integration_nodes"]
                else (
                    "cubic Hermite from RK4 node values and endpoint "
                    "derivatives"
                )
            ),
            "common_support_grid": "n=800",
            "singular_interval": [
                args.interior_start,
                args.interior_end,
            ],
            "singular_interval_semantics": "half-open",
            "strict_off_grid_only": True,
            "excluded_support_nodes": int(
                (
                    (dense_time >= args.interior_start)
                    & (dense_time < args.interior_end)
                    & support
                ).sum()
            ),
            "evaluated_coordinates_per_state": int(strict_mask.sum()),
            "epsilon": (
                "sqrt(RMS(psi)^2+RMS(dotpsi)^2+RMS(ddotpsi)^2)"
            ),
            "resistant_state": {
                "radius": args.resistant_radius,
                "direction": direction.tolist(),
                "values": resistant.tolist(),
                "formula": "N_i(0)=10*(1+0.20*z_i), z_i=-1+0.1*(i-1)",
            },
        },
        "mixed_evaluator_checks": {
            "all_rows_share_protocol_object": True,
            "zero_order_hold_used": False,
            "strict_point_count_is_16120": int(strict_mask.sum()) == 16_120,
            "physical_objective_weights": [1.0, 40.0, 8000.0],
        },
        "timing": timing,
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "outputs": {
            "rows_csv": str((output_dir / "rows.csv").resolve()),
            "policy_cache": (
                str((output_dir / "policy_cache.npz").resolve())
                if policy_cache_record is not None
                else (
                    str(policy_cache)
                    if policy_cache is not None
                    else None
                )
            ),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
