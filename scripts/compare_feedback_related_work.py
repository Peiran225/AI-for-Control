#!/usr/bin/env python3
"""Re-evaluate learned feedback policies on a common physical tumor problem.

This script deliberately keeps the feedback comparison separate from the
time-only comparison. Every policy is queried at the left endpoint of every
execution interval, and the resulting constant action is executed with a
fresh, high-accuracy DOP853 solve on that interval. By default each method
uses its native grid; ``--evaluation-intervals`` instead places every policy
on one common execution grid. The reported objective is always one supplied
*unregularized* physical objective. The current paper defaults are

    J = sum_i N_i(T) + integral_0^T [40 sum_i N_i(t) + 8000 u(t)] dt.

Checkpoints may be trained with a positive common rescaling of these three
weights. The loader verifies that proportionality and records the scale
needed to recover the physical objective; it rejects checkpoints trained for
a genuinely different objective.

The learned policies retain their native time representations: 800 nodes for
the two Section-5 Transformer cases, continuous-time inputs for PI-DeepONet
and Adaptive HJB-NN trained on 200 intervals, and 80 piecewise-time subnetworks
for DeepBSDE. On a finer common grid, DeepBSDE reuses the checkpoint's native
subnetwork throughout its original time interval. TensorFlow-1 HJB-NN and
TensorFlow-2 DeepBSDE run in separate worker processes.

The default ``all`` command writes an auditable bundle containing per-run
CSV records, one NPZ trajectory per checkpoint and initial state, a protocol
file, and an aggregate summary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

# Fix the timing device before importing either framework in a worker.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from scipy.integrate import solve_ivp


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for search_path in (ROOT, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from tumor_problem import TumorProblem, dynamics_numpy, evaluate_zoh_control  # noqa: E402


DEFAULT_OUTPUT = (
    ROOT
    / "outputs/related_work_a1_b40_g8000_20260723/feedback_raw"
)
DEFAULT_CASE1 = (
    ROOT
    / "outputs/new_objective_alpha1_grid_20260721_v2/a1_b40_g8000/"
    "feedback/case1_cf_deep_v1_physical.pt"
)
DEFAULT_CASE2 = (
    ROOT
    / "outputs/new_objective_alpha1_grid_20260721_v2/a1_b40_g8000/"
    "feedback/case2_der_deep_v1_physical.pt"
)
DEFAULT_PI_ROOT = (
    ROOT
    / "paper_runs/faithful_related_work/"
    "pi_deeponet_tumor_a1_b40_g8000_3seed_20260723"
)
DEFAULT_DEEPBSDE_ROOT = (
    ROOT
    / "paper_runs/faithful_related_work/"
    "deepbsde_tumor_a1_b40_g8000_3seed_20260723"
)
DEFAULT_HJB_ROOT = (
    ROOT
    / "paper_runs/faithful_related_work/"
    "hjb_nn_tumor_a1_b40_g8000_tau10_3seed_20260723"
)

METHOD_CSV_FIELDS = [
    "family",
    "method",
    "variant",
    "run_id",
    "seed",
    "surrogate_parameter_name",
    "surrogate_parameter",
    "surrogate_parameter_units",
    "surrogate_parameter_physical_equivalent",
    "primary_comparison",
    "state_id",
    "initial_state",
    "checkpoint",
    "checkpoint_sha256",
    "native_intervals",
    "evaluation_intervals",
    "policy_queries",
    "time_extension_semantics",
    "control_semantics",
    "training_objective",
    "objective_scale_to_evaluation",
    "objective",
    "J_unregularized",
    "terminal_cost",
    "running_cost",
    "final_total_N",
    "u_min",
    "u_max",
    "u_mean_time",
    "load_seconds",
    "query_median_ms",
    "query_mean_ms",
    "query_p10_ms",
    "query_p90_ms",
    "query_repeats",
    "closed_loop_median_ms",
    "closed_loop_mean_ms",
    "closed_loop_min_ms",
    "closed_loop_max_ms",
    "closed_loop_repeats",
    "closed_loop_warmups",
    "nominal_common_evaluator_abs_error",
    "trajectory_npz",
]


class LoadedPolicy:
    """Small common interface for a loaded, state-feedback policy."""

    def __init__(
        self,
        *,
        query: Callable[[int, float, np.ndarray], float],
        objective_weights: tuple[float, float, float],
        close: Callable[[], None] | None = None,
    ) -> None:
        self.query = query
        self.objective_weights = tuple(float(value) for value in objective_weights)
        self.close = close or (lambda: None)


def resolve(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
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
        if not math.isfinite(number):
            return None
        return number
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def initial_states(problem: TumorProblem) -> dict[str, np.ndarray]:
    nominal = np.full(problem.m, problem.n0, dtype=np.float64)
    resistant = np.linspace(9.0, 11.0, problem.m, dtype=np.float64)
    if not np.isclose(nominal.sum(), resistant.sum(), rtol=0.0, atol=1.0e-12):
        raise RuntimeError("the two initial states must have equal total burden")
    return {"nominal": nominal, "resistant_heavy": resistant}


def evaluation_problem(args: argparse.Namespace) -> TumorProblem:
    return TumorProblem(
        T=10.0,
        m=21,
        umax=3.0,
        alpha=float(args.physical_alpha),
        beta=float(args.physical_beta),
        gamma=float(args.physical_gamma),
        n0=10.0,
        m_suppression=0.5,
    )


def objective_scale_to_evaluation(
    policy: LoadedPolicy,
    problem: TumorProblem,
) -> float:
    """Return the common positive scale from checkpoint to evaluation weights."""

    training = np.asarray(policy.objective_weights, dtype=np.float64)
    evaluation = np.asarray(
        [problem.alpha, problem.beta, problem.gamma], dtype=np.float64
    )
    if (
        training.shape != (3,)
        or not np.all(np.isfinite(training))
        or np.any(training <= 0.0)
    ):
        raise ValueError(
            "checkpoint objective weights (alpha,beta,gamma) must be finite and positive"
        )
    ratios = evaluation / training
    if not np.allclose(ratios, ratios[0], rtol=1.0e-10, atol=1.0e-12):
        raise ValueError(
            "checkpoint was trained for a non-proportional objective: "
            f"training (alpha,beta,gamma)={training.tolist()}, "
            f"evaluation={evaluation.tolist()}"
        )
    return float(ratios[0])


def quantile(values: list[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def timing_summary(seconds: list[float], prefix: str) -> dict[str, float]:
    milliseconds = [1000.0 * value for value in seconds]
    return {
        f"{prefix}_median_ms": float(statistics.median(milliseconds)),
        f"{prefix}_mean_ms": float(statistics.fmean(milliseconds)),
        f"{prefix}_min_ms": float(min(milliseconds)),
        f"{prefix}_max_ms": float(max(milliseconds)),
    }


def rollout_feedback(
    policy: LoadedPolicy,
    initial_state: np.ndarray,
    intervals: int,
    problem: TumorProblem,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Execute a learned feedback with left-query/ZOH/DOP853 semantics."""

    breakpoints = np.linspace(0.0, problem.T, intervals + 1, dtype=np.float64)
    controls = np.empty(intervals, dtype=np.float64)
    states = np.empty((intervals + 1, problem.m), dtype=np.float64)
    cumulative_running = np.empty(intervals + 1, dtype=np.float64)
    states[0] = np.asarray(initial_state, dtype=np.float64)
    cumulative_running[0] = 0.0
    augmented = np.concatenate([states[0], np.zeros(1, dtype=np.float64)])
    params = problem.vectors()

    for index in range(intervals):
        left = float(breakpoints[index])
        right = float(breakpoints[index + 1])
        control = float(policy.query(index, left, augmented[: problem.m].copy()))
        if not math.isfinite(control) or control < -1.0e-9 or control > problem.umax + 1.0e-9:
            raise RuntimeError(
                f"policy returned infeasible u={control} at interval {index}"
            )
        control = float(np.clip(control, 0.0, problem.umax))
        controls[index] = control

        def rhs(_time: float, value: np.ndarray) -> np.ndarray:
            state = np.maximum(value[: problem.m], 1.0e-12)
            running = float(params["beta"] @ state + problem.gamma * control)
            return np.concatenate(
                [dynamics_numpy(state, control, problem, params), np.array([running])]
            )

        solution = solve_ivp(
            rhs,
            (left, right),
            augmented,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            max_step=max((right - left) / 4.0, 1.0e-8),
        )
        if not solution.success:
            raise RuntimeError(
                f"DOP853 failed on [{left},{right}]: {solution.message}"
            )
        augmented = np.asarray(solution.y[:, -1], dtype=np.float64)
        if (
            not np.all(np.isfinite(augmented))
            or float(np.min(augmented[: problem.m])) <= 0.0
        ):
            raise RuntimeError("feedback rollout produced a nonpositive/nonfinite state")
        states[index + 1] = augmented[: problem.m]
        cumulative_running[index + 1] = augmented[-1]

    terminal = float(params["alpha"] @ states[-1])
    running = float(cumulative_running[-1])
    return {
        "t": breakpoints,
        "u": controls,
        "N": states,
        "cumulative_running_cost": cumulative_running,
        "J": terminal + running,
        "terminal_cost": terminal,
        "running_cost": running,
        "final_total_N": float(states[-1].sum()),
        "u_min": float(controls.min()),
        "u_max": float(controls.max()),
        "u_mean_time": float(np.sum(controls * np.diff(breakpoints)) / problem.T),
    }


def benchmark_query(
    policy: LoadedPolicy,
    trajectory: dict[str, Any],
    *,
    repeats: int,
    warmups: int,
) -> dict[str, float]:
    index = len(trajectory["u"]) // 2
    query_time = float(trajectory["t"][index])
    query_state = np.asarray(trajectory["N"][index], dtype=np.float64)
    for _ in range(warmups):
        policy.query(index, query_time, query_state.copy())
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        policy.query(index, query_time, query_state.copy())
        samples.append(time.perf_counter() - started)
    milliseconds = np.asarray(samples, dtype=np.float64) * 1000.0
    return {
        "query_median_ms": float(np.median(milliseconds)),
        "query_mean_ms": float(np.mean(milliseconds)),
        "query_p10_ms": float(np.quantile(milliseconds, 0.10)),
        "query_p90_ms": float(np.quantile(milliseconds, 0.90)),
    }


def benchmark_rollout(
    policy: LoadedPolicy,
    state: np.ndarray,
    intervals: int,
    problem: TumorProblem,
    *,
    repeats: int,
    warmups: int,
    rtol: float,
    atol: float,
) -> dict[str, float]:
    for _ in range(warmups):
        rollout_feedback(policy, state, intervals, problem, rtol=rtol, atol=atol)
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        rollout_feedback(policy, state, intervals, problem, rtol=rtol, atol=atol)
        samples.append(time.perf_counter() - started)
    return timing_summary(samples, "closed_loop")


def save_trajectory(
    output_dir: Path,
    run_id: str,
    state_id: str,
    trajectory: dict[str, Any],
    initial_state: np.ndarray,
    metadata: dict[str, Any],
) -> Path:
    path = output_dir / "trajectories" / f"{run_id}__{state_id}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        t=np.asarray(trajectory["t"], dtype=np.float64),
        u=np.asarray(trajectory["u"], dtype=np.float64),
        N=np.asarray(trajectory["N"], dtype=np.float64),
        cumulative_running_cost=np.asarray(
            trajectory["cumulative_running_cost"], dtype=np.float64
        ),
        initial_state=np.asarray(initial_state, dtype=np.float64),
        J_unregularized=np.array(trajectory["J"], dtype=np.float64),
        terminal_cost=np.array(trajectory["terminal_cost"], dtype=np.float64),
        running_cost=np.array(trajectory["running_cost"], dtype=np.float64),
        metadata_json=np.array(json.dumps(json_safe(metadata), sort_keys=True)),
    )
    return path


def evaluate_loaded_policy(
    *,
    policy: LoadedPolicy,
    load_seconds: float,
    checkpoint: Path,
    intervals: int,
    metadata: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    problem = evaluation_problem(args)
    objective_scale = objective_scale_to_evaluation(policy, problem)
    training_alpha, training_beta, training_gamma = policy.objective_weights
    evaluation_intervals = (
        int(args.evaluation_intervals)
        if args.evaluation_intervals is not None
        else int(intervals)
    )
    if metadata.get("family") == "ours" and evaluation_intervals != int(intervals):
        raise ValueError(
            "the Transformer checkpoint has one learned time token per native "
            f"interval; evaluate it on its n={intervals} grid"
        )
    surrogate_physical_equivalent: float | str = ""
    if metadata.get("surrogate_parameter_name") == "entropy_tau_normalized":
        surrogate_physical_equivalent = (
            float(metadata["surrogate_parameter"]) * objective_scale
        )
    rows: list[dict[str, Any]] = []
    for state_id, state in initial_states(problem).items():
        print(
            f"[{metadata['run_id']}/{state_id}] primary closed-loop rollout",
            flush=True,
        )
        trajectory = rollout_feedback(
            policy,
            state,
            evaluation_intervals,
            problem,
            rtol=args.rtol,
            atol=args.atol,
        )
        nominal_error = float("nan")
        if state_id == "nominal":
            independent = evaluate_zoh_control(
                trajectory["t"],
                trajectory["u"],
                problem,
                include_diagnostics=False,
                rtol=args.rtol,
                atol=args.atol,
            )
            nominal_error = abs(float(independent["J"]) - float(trajectory["J"]))
            if nominal_error > 2.0e-7:
                raise RuntimeError(
                    f"nominal objective cross-check failed: abs error={nominal_error}"
                )

        query_timing = benchmark_query(
            policy,
            trajectory,
            repeats=args.query_repeats,
            warmups=args.query_warmups,
        )
        rollout_timing = benchmark_rollout(
            policy,
            state,
            evaluation_intervals,
            problem,
            repeats=args.rollout_repeats,
            warmups=args.rollout_warmups,
            rtol=args.rtol,
            atol=args.atol,
        )
        trajectory_path = save_trajectory(
            output_dir,
            metadata["run_id"],
            state_id,
            trajectory,
            state,
            metadata,
        )
        row = {
            **metadata,
            "surrogate_parameter_physical_equivalent": (
                surrogate_physical_equivalent
            ),
            "state_id": state_id,
            "initial_state": json.dumps(state.tolist(), separators=(",", ":")),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "native_intervals": intervals,
            "evaluation_intervals": evaluation_intervals,
            "policy_queries": evaluation_intervals,
            "control_semantics": (
                "learned feedback queried at each common-grid left endpoint; "
                "action held constant on the execution interval"
            ),
            "training_objective": (
                f"{training_alpha:g} sum_i N_i(T) + integral["
                f"{training_beta:g} sum_i N_i(t) + {training_gamma:g} u(t)] dt"
            ),
            "objective_scale_to_evaluation": objective_scale,
            "objective": (
                f"{problem.alpha:g} sum_i N_i(T) + integral["
                f"{problem.beta:g} sum_i N_i(t) + {problem.gamma:g} u(t)] dt"
            ),
            "J_unregularized": trajectory["J"],
            "terminal_cost": trajectory["terminal_cost"],
            "running_cost": trajectory["running_cost"],
            "final_total_N": trajectory["final_total_N"],
            "u_min": trajectory["u_min"],
            "u_max": trajectory["u_max"],
            "u_mean_time": trajectory["u_mean_time"],
            "load_seconds": load_seconds,
            **query_timing,
            "query_repeats": args.query_repeats,
            **rollout_timing,
            "closed_loop_repeats": args.rollout_repeats,
            "closed_loop_warmups": args.rollout_warmups,
            "nominal_common_evaluator_abs_error": nominal_error,
            "trajectory_npz": str(trajectory_path),
        }
        rows.append(row)
        print(
            f"[{metadata['run_id']}/{state_id}] "
            f"J={trajectory['J']:.9f}, query={query_timing['query_median_ms']:.4f} ms, "
            f"rollout={rollout_timing['closed_loop_median_ms']:.2f} ms",
            flush=True,
        )
    return rows


def load_ours(checkpoint: Path) -> tuple[LoadedPolicy, int, float]:
    import torch

    torch.set_num_threads(1)
    from evaluate_feedback_section5 import (
        configured_state_mode,
        load_feedback_checkpoint,
    )

    started = time.perf_counter()
    model, cfg, checkpoint_args = load_feedback_checkpoint(checkpoint)
    if configured_state_mode(checkpoint_args) != "feedback":
        raise ValueError(f"{checkpoint}: expected a state-feedback checkpoint")
    model.eval()
    grid = torch.linspace(0.0, 1.0, cfg.n + 1, dtype=torch.float64)
    with torch.inference_mode():
        base_logits = model.time_logits(grid)[: cfg.n]
    load_seconds = time.perf_counter() - started

    def query(index: int, _time: float, state: np.ndarray) -> float:
        normalized_time = torch.tensor([index / cfg.n], dtype=torch.float64)
        state_tensor = torch.as_tensor(state[None, :], dtype=torch.float64)
        with torch.inference_mode():
            action = model.interval_action(
                base_logits[index],
                normalized_time,
                state_tensor,
                state_mode="feedback",
            )
        return float(action.item())

    return (
        LoadedPolicy(
            query=query,
            objective_weights=(cfg.alpha, cfg.beta, cfg.gamma),
        ),
        int(cfg.n),
        load_seconds,
    )


def load_pi(checkpoint: Path) -> tuple[LoadedPolicy, int, float]:
    import torch

    torch.set_num_threads(1)
    from faithful_related_work.pi_deeponet.core import (
        ImprovedPolicy,
        TrainConfig,
        build_model,
        terminal_branch_values,
    )
    from faithful_related_work.pi_deeponet.problems import TumorAdaptation

    started = time.perf_counter()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = TrainConfig(**payload["train_config"])
    problem_payload = dict(payload["problem"])
    required_problem_keys = {
        "T",
        "state_dim",
        "control_dim",
        "umax",
        "beta",
        "alpha",
        "gamma",
        "initial_N",
        "suppression",
        "state_scale",
    }
    missing_problem_keys = sorted(required_problem_keys - set(problem_payload))
    if missing_problem_keys:
        raise ValueError(
            f"PI-DeepONet checkpoint omits problem fields: {missing_problem_keys}"
        )
    problem = TumorAdaptation(
        **{
            key: problem_payload[key]
            for key in (
                "T",
                "state_dim",
                "control_dim",
                "umax",
                "beta",
                "alpha",
                "gamma",
                "initial_N",
                "suppression",
                "state_scale",
                "normalized_state_lower",
                "normalized_state_upper",
                "name",
            )
            if key in problem_payload
        }
    )
    if payload["problem"]["state_dim"] != problem.state_dim:
        raise ValueError("PI-DeepONet checkpoint uses an unexpected state dimension")
    model = build_model(problem, config).to(dtype=torch.float64, device="cpu")
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    sensors = torch.as_tensor(payload["sensor_states"], dtype=torch.float64)
    terminal_parameter = torch.tensor([1.0], dtype=torch.float64)
    branch = terminal_branch_values(problem, terminal_parameter, sensors)
    improved = ImprovedPolicy(model, problem, config.h, config.tie_tolerance)
    load_seconds = time.perf_counter() - started

    def query(_index: int, physical_time: float, state: np.ndarray) -> float:
        t = torch.tensor([physical_time], dtype=torch.float64)
        x = torch.as_tensor((state / problem.state_scale)[None, :], dtype=torch.float64)
        result = improved(branch, t, x)
        return float(result.control.detach().cpu().numpy().reshape(-1)[0])

    return (
        LoadedPolicy(
            query=query,
            objective_weights=(problem.alpha, problem.beta, problem.gamma),
        ),
        200,
        load_seconds,
    )


def load_deepbsde(checkpoint_dir: Path) -> tuple[LoadedPolicy, int, float, float, int]:
    # This function is called only in the dedicated TF2 worker.
    import tensorflow as tf

    from faithful_related_work.deepbsde.equations import (
        LogStateTumorHJB,
        hamiltonian_argmin_numpy,
    )
    from faithful_related_work.deepbsde.runner import (
        AttrObject,
        _build_official_solver,
        _model_z_at,
    )

    config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    sigma = float(config["eqn_config"]["sigma"])
    seed = int(config["eqn_config"]["seed"])
    started = time.perf_counter()
    equation = LogStateTumorHJB(AttrObject(config["eqn_config"]))
    solver = _build_official_solver(config, equation, seed)
    dummy_dw = np.zeros(
        (1, equation.dim, equation.num_time_interval), dtype=np.float64
    )
    dummy_x = np.broadcast_to(
        equation.x_init.reshape((1, equation.dim, 1)),
        (1, equation.dim, equation.num_time_interval + 1),
    ).copy()
    solver.model(
        (
            tf.convert_to_tensor(dummy_dw, dtype=tf.float64),
            tf.convert_to_tensor(dummy_x, dtype=tf.float64),
        ),
        training=False,
    )
    weights_path = checkpoint_dir / "model.weights.h5"
    solver.model.load_weights(weights_path)
    load_seconds = time.perf_counter() - started

    def query(_index: int, physical_time: float, state: np.ndarray) -> float:
        native_index = min(
            int(
                math.floor(
                    physical_time / equation.total_time * equation.num_time_interval
                    + 1.0e-12
                )
            ),
            equation.num_time_interval - 1,
        )
        z = _model_z_at(solver.model, native_index, np.log(state), equation.dim)
        gradient = z / equation.sigma
        action = hamiltonian_argmin_numpy(
            gradient,
            equation.params["phi"],
            equation.gamma,
            equation.umax,
        )
        return float(np.asarray(action).reshape(-1)[0])

    def close() -> None:
        tf.keras.backend.clear_session()

    return (
        LoadedPolicy(
            query=query,
            objective_weights=(
                float(config["eqn_config"]["alpha"]),
                float(config["eqn_config"]["beta"]),
                float(config["eqn_config"]["gamma"]),
            ),
            close=close,
        ),
        int(equation.num_time_interval),
        load_seconds,
        sigma,
        seed,
    )


def load_hjb(checkpoint: Path, tau: float) -> tuple[LoadedPolicy, int, float]:
    # This function is called only in the dedicated TF1 worker.
    import scipy.io
    import tensorflow.compat.v1 as tf

    from faithful_related_work.hjb_nn.problem import TumorEntropyProblem, TumorParameters
    from faithful_related_work.hjb_nn.runner import _load_official_hjbnet

    tf.reset_default_graph()
    payload = scipy.io.loadmat(checkpoint)
    weights = payload["weights"].reshape(-1)
    biases = payload["biases"].reshape(-1)
    config = type("LoadedHJBConfig", (), {})()
    config.layers = [weights[0].shape[1], *[weight.shape[0] for weight in weights]]
    scaling = {
        key: payload[key]
        for key in ("lb", "ub", "A_lb", "A_ub", "U_lb", "U_ub", "V_min", "V_max")
    }
    parameters = {"weights": weights, "biases": biases}
    manifest_path = checkpoint.with_name("manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"HJB-NN worker manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not math.isclose(
        float(manifest.get("tau", float("nan"))),
        float(tau),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            f"HJB-NN checkpoint tau does not match requested tau={tau}: {manifest_path}"
        )
    declared_problem = manifest.get("problem_parameters")
    if not isinstance(declared_problem, dict):
        raise ValueError(f"HJB-NN problem parameters are missing: {manifest_path}")
    problem_parameters = TumorParameters(**declared_problem)
    started = time.perf_counter()
    model = _load_official_hjbnet()(
        TumorEntropyProblem(tau, parameters=problem_parameters),
        scaling,
        config,
        parameters,
    )
    load_seconds = time.perf_counter() - started

    def query(_index: int, physical_time: float, state: np.ndarray) -> float:
        action = model.eval_U(
            np.array([[physical_time]], dtype=np.float64),
            np.asarray(state, dtype=np.float64).reshape((-1, 1)),
        )
        return float(np.asarray(action).reshape(-1)[0])

    def close() -> None:
        model.sess.close()
        tf.reset_default_graph()

    return (
        LoadedPolicy(
            query=query,
            objective_weights=(
                problem_parameters.alpha,
                problem_parameters.beta,
                problem_parameters.gamma,
            ),
            close=close,
        ),
        200,
        load_seconds,
    )


def run_ours(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases = [
        ("case1", resolve(args.case1), "Case 1 u(N,t)"),
        ("case2", resolve(args.case2), "Case 2 u(N,t)"),
    ]
    for variant, checkpoint, label in cases:
        policy, intervals, load_seconds = load_ours(checkpoint)
        try:
            rows.extend(
                evaluate_loaded_policy(
                    policy=policy,
                    load_seconds=load_seconds,
                    checkpoint=checkpoint,
                    intervals=intervals,
                    metadata={
                        "family": "ours",
                        "method": "Section-5 Transformer feedback",
                        "variant": variant,
                        "run_id": f"ours_{variant}",
                        "seed": "",
                        "surrogate_parameter_name": "",
                        "surrogate_parameter": "",
                        "primary_comparison": True,
                        "time_extension_semantics": "native n=800 Transformer time index",
                    },
                    output_dir=resolve(args.output_dir),
                    args=args,
                )
            )
        finally:
            policy.close()
    return rows


def run_pi(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = resolve(args.pi_root)
    for seed in (0, 1, 2):
        checkpoint = root / f"seed_{seed}/checkpoint_outer_002.pt"
        policy, intervals, load_seconds = load_pi(checkpoint)
        try:
            rows.extend(
                evaluate_loaded_policy(
                    policy=policy,
                    load_seconds=load_seconds,
                    checkpoint=checkpoint,
                    intervals=intervals,
                    metadata={
                        "family": "pi_deeponet",
                        "method": "PI-DeepONet",
                        "variant": "final_predeclared_outer_2",
                        "run_id": f"pi_deeponet_seed_{seed}",
                        "seed": seed,
                        "surrogate_parameter_name": "artificial_viscosity_N",
                        "surrogate_parameter": 3.85,
                        "primary_comparison": True,
                        "time_extension_semantics": "continuous physical-time policy query",
                    },
                    output_dir=resolve(args.output_dir),
                    args=args,
                )
            )
        finally:
            policy.close()
    return rows


def run_deepbsde(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = resolve(args.deepbsde_root)
    sigmas = (0.025, 0.050, 0.100) if args.deepbsde_all_sigmas else (0.025,)
    for sigma in sigmas:
        for seed in (0, 1, 2):
            checkpoint_dir = root / f"sigma_{sigma:.3f}_seed_{seed}"
            policy, intervals, load_seconds, actual_sigma, actual_seed = load_deepbsde(
                checkpoint_dir
            )
            if not math.isclose(actual_sigma, sigma, rel_tol=0.0, abs_tol=1.0e-12):
                raise RuntimeError("DeepBSDE checkpoint sigma does not match directory")
            if actual_seed != seed:
                raise RuntimeError("DeepBSDE checkpoint seed does not match directory")
            checkpoint = checkpoint_dir / "model.weights.h5"
            primary = math.isclose(sigma, 0.025, rel_tol=0.0, abs_tol=1.0e-12)
            try:
                rows.extend(
                    evaluate_loaded_policy(
                        policy=policy,
                        load_seconds=load_seconds,
                        checkpoint=checkpoint,
                        intervals=intervals,
                        metadata={
                            "family": "deepbsde",
                            "method": "DeepBSDE log-state HJB",
                            "variant": f"sigma_{sigma:.3f}",
                            "run_id": f"deepbsde_sigma_{sigma:.3f}_seed_{seed}",
                            "seed": seed,
                            "surrogate_parameter_name": "sigma",
                            "surrogate_parameter": sigma,
                            "primary_comparison": primary,
                            "time_extension_semantics": (
                                "native 80-slice subnet held by time interval; "
                                "current state re-queried at every common-grid node"
                            ),
                        },
                        output_dir=resolve(args.output_dir),
                        args=args,
                    )
                )
            finally:
                policy.close()
    return rows


def run_hjb(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = resolve(args.hjb_root)
    taus = tuple(
        float(value.strip())
        for value in str(args.hjb_taus).split(",")
        if value.strip()
    )
    if not taus:
        raise ValueError("--hjb-taus must contain at least one value")
    for tau in taus:
        tau_tag = f"{tau:g}".replace(".", "p")
        for seed in (0, 1, 2):
            checkpoint = root / f"tau_{tau_tag}/seed_{seed}/V_model.mat"
            policy, intervals, load_seconds = load_hjb(checkpoint, tau)
            try:
                rows.extend(
                    evaluate_loaded_policy(
                        policy=policy,
                        load_seconds=load_seconds,
                        checkpoint=checkpoint,
                        intervals=intervals,
                        metadata={
                            "family": "hjb_nn",
                            "method": "Adaptive HJB-NN",
                            "variant": f"tau_norm_{tau_tag}",
                            "run_id": f"hjb_nn_tau_norm_{tau_tag}_seed_{seed}",
                            "seed": seed,
                            "surrogate_parameter_name": "entropy_tau_normalized",
                            "surrogate_parameter": tau,
                            "surrogate_parameter_units": (
                                "checkpoint objective units"
                            ),
                            "primary_comparison": True,
                            "time_extension_semantics": "continuous physical-time policy query",
                        },
                        output_dir=resolve(args.output_dir),
                        args=args,
                    )
                )
            finally:
                policy.close()
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=METHOD_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def aggregate(args: argparse.Namespace) -> None:
    output_dir = resolve(args.output_dir)
    family_paths = {
        family: output_dir / f"per_run_{family}.csv"
        for family in ("ours", "pi", "deepbsde", "hjb")
    }
    missing = [str(path) for path in family_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing worker outputs: {missing}")
    rows: list[dict[str, str]] = []
    for path in family_paths.values():
        rows.extend(read_rows(path))
    write_rows(output_dir / "per_run.csv", rows)

    grouped: dict[str, dict[str, list[dict[str, str]]]] = {}
    for row in rows:
        key = f"{row['method']} | {row['variant']}"
        grouped.setdefault(key, {}).setdefault(row["state_id"], []).append(row)

    summaries: dict[str, Any] = {}
    for method_key, by_state in grouped.items():
        summaries[method_key] = {}
        for state_id, group in by_state.items():
            objectives = np.array(
                [float(item["J_unregularized"]) for item in group], dtype=np.float64
            )
            queries = np.array(
                [float(item["query_median_ms"]) for item in group], dtype=np.float64
            )
            rollouts = np.array(
                [float(item["closed_loop_median_ms"]) for item in group], dtype=np.float64
            )
            summaries[method_key][state_id] = {
                "runs": len(group),
                "J_mean": float(objectives.mean()),
                "J_sample_sd": float(objectives.std(ddof=1)) if len(group) > 1 else 0.0,
                "J_min": float(objectives.min()),
                "J_max": float(objectives.max()),
                "query_median_ms_across_runs": float(np.median(queries)),
                "closed_loop_median_ms_across_runs": float(np.median(rollouts)),
                "native_intervals": sorted({int(item["native_intervals"]) for item in group}),
                "evaluation_intervals": sorted(
                    {int(item["evaluation_intervals"]) for item in group}
                ),
            }
    two_state: dict[str, Any] = {}
    for method_key, by_state in grouped.items():
        seed_pairs: dict[str, dict[str, float]] = {}
        for state_id, group in by_state.items():
            for item in group:
                identity = str(item["run_id"])
                seed_pairs.setdefault(identity, {})[state_id] = float(item["J_unregularized"])
        paired = [
            0.5 * (values["nominal"] + values["resistant_heavy"])
            for values in seed_pairs.values()
            if set(values) == {"nominal", "resistant_heavy"}
        ]
        if paired:
            array = np.asarray(paired, dtype=np.float64)
            two_state[method_key] = {
                "runs": len(paired),
                "mean_of_two_state_J_mean": float(array.mean()),
                "mean_of_two_state_J_sample_sd": (
                    float(array.std(ddof=1)) if len(array) > 1 else 0.0
                ),
                "mean_of_two_state_J_min": float(array.min()),
                "mean_of_two_state_J_max": float(array.max()),
            }

    write_json(
        output_dir / "summary.json",
        {
            "comparison_scope": "u(N,t) learned-feedback methods only",
            "per_state": summaries,
            "two_state_average": two_state,
            "row_count": len(rows),
            "trajectory_count": len(rows),
        },
    )
    write_protocol(args, output_dir, rows)


def write_protocol(
    args: argparse.Namespace,
    output_dir: Path,
    rows: list[dict[str, Any]] | list[dict[str, str]],
) -> None:
    problem = evaluation_problem(args)
    states = initial_states(problem)
    requested_hjb_taus = [
        float(value.strip())
        for value in str(args.hjb_taus).split(",")
        if value.strip()
    ]
    hjb_scales = sorted(
        {
            float(row["objective_scale_to_evaluation"])
            for row in rows
            if str(row.get("family", "")) == "hjb_nn"
        }
    )
    hjb_physical_equivalents = sorted(
        {tau * scale for tau in requested_hjb_taus for scale in hjb_scales}
    )
    try:
        import scipy

        scipy_version = scipy.__version__
    except Exception:
        scipy_version = "unavailable"
    versions: dict[str, str] = {
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy_version,
    }
    for package_name in ("torch", "tensorflow"):
        try:
            import importlib.metadata

            versions[package_name] = importlib.metadata.version(package_name)
        except Exception:
            versions[package_name] = "unavailable"
    write_json(
        output_dir / "protocol.json",
        {
            "schema_version": 2,
            "scope": "feedback-policy u(N,t) comparison; time-only u(t) methods are intentionally excluded",
            "physical_problem": problem.to_dict(),
            "initial_states": {key: value.tolist() for key, value in states.items()},
            "initial_state_note": (
                "Nominal and resistant-heavy both sum to 210; resistant-heavy is "
                "(9.0,9.1,...,11.0)."
            ),
            "common_unregularized_objective": (
                f"{problem.alpha:g} sum_i N_i(T) + integral_0^T "
                f"[{problem.beta:g} sum_i N_i(t) + {problem.gamma:g} u(t)] dt"
            ),
            "execution": {
                "policy_query": "each learned policy is re-queried from the current state at every execution-grid left endpoint",
                "control_hold": "zero-order hold until the next execution breakpoint",
                "physical_integrator": "scipy.integrate.solve_ivp DOP853 restarted on every control interval",
                "rtol": args.rtol,
                "atol": args.atol,
                "max_step": "one quarter of the execution control interval",
                "evaluation_intervals": (
                    args.evaluation_intervals
                    if args.evaluation_intervals is not None
                    else "method-native"
                ),
                "native_intervals": {
                    "ours_case1": 800,
                    "ours_case2": 800,
                    "PI-DeepONet": 200,
                    "Adaptive_HJB-NN": 200,
                    "DeepBSDE": 80,
                },
                "deepbsde_common_grid_semantics": (
                    "DeepBSDE has a distinct learned subnet for each of its 80 training intervals; "
                    "common-grid queries reuse the checkpoint's native subnetwork within that "
                    "original interval rather than inventing intermediate subnetworks."
                ),
            },
            "method_selection": {
                "ours": "the two selected Section-5 feedback checkpoints (Case 1 and Case 2)",
                "PI-DeepONet": "final predeclared outer iteration 2, all three declared seeds",
                "Adaptive_HJB-NN": (
                    f"normalized entropy tau values {requested_hjb_taus}, all three "
                    "declared seeds; the corresponding physical-objective tau values "
                    f"are {hjb_physical_equivalents}"
                ),
                "DeepBSDE_primary": "sigma=0.025 and all three declared seeds",
                "DeepBSDE_secondary": (
                    "sigma=0.050 and 0.100 are not in the primary comparison because they solve "
                    "different, more viscous surrogate HJB equations; pass --deepbsde-all-sigmas "
                    "to save them as secondary sensitivity runs"
                ),
            },
            "timing": {
                "device": "CPU only (CUDA_VISIBLE_DEVICES=-1)",
                "threads": {
                    key: os.environ.get(key)
                    for key in (
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                    )
                },
                "loaded_query": (
                    "single query at the midpoint node/state of each realized trajectory; "
                    "model loading is outside the timed region"
                ),
                "query_warmups": args.query_warmups,
                "query_repeats": args.query_repeats,
                "closed_loop": (
                    "complete learned-policy queries plus all segmented high-accuracy DOP853 "
                    "physical integrations on the declared execution grid"
                ),
                "closed_loop_warmups": args.rollout_warmups,
                "closed_loop_repeats": args.rollout_repeats,
                "training_time_comparison": "not reported; saved checkpoints were produced under non-identical training protocols",
            },
            "nominal_cross_check": (
                "each nominal saved control is independently passed to the repository common "
                "evaluate_zoh_control implementation; tolerance is 2e-7 in J"
            ),
            "software": versions,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "command": sys.argv,
            "output_dir": output_dir,
            "rows": len(rows),
        },
    )


def run_worker(args: argparse.Namespace) -> None:
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    functions = {
        "ours": run_ours,
        "pi": run_pi,
        "deepbsde": run_deepbsde,
        "hjb": run_hjb,
    }
    rows = functions[args.family](args)
    write_rows(output_dir / f"per_run_{args.family}.csv", rows)
    write_json(
        output_dir / f"worker_{args.family}.json",
        {
            "family": args.family,
            "rows": len(rows),
            "command": sys.argv,
            "completed_at_unix": time.time(),
        },
    )


def orchestrate(args: argparse.Namespace) -> None:
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    common = [
        "--output-dir",
        str(output_dir),
        "--case1",
        str(resolve(args.case1)),
        "--case2",
        str(resolve(args.case2)),
        "--pi-root",
        str(resolve(args.pi_root)),
        "--deepbsde-root",
        str(resolve(args.deepbsde_root)),
        "--hjb-root",
        str(resolve(args.hjb_root)),
        "--rtol",
        str(args.rtol),
        "--atol",
        str(args.atol),
        "--query-warmups",
        str(args.query_warmups),
        "--query-repeats",
        str(args.query_repeats),
        "--rollout-warmups",
        str(args.rollout_warmups),
        "--rollout-repeats",
        str(args.rollout_repeats),
        "--physical-alpha",
        str(args.physical_alpha),
        "--physical-beta",
        str(args.physical_beta),
        "--physical-gamma",
        str(args.physical_gamma),
        "--hjb-taus",
        str(args.hjb_taus),
    ]
    if args.evaluation_intervals is not None:
        common.extend(["--evaluation-intervals", str(args.evaluation_intervals)])
    if args.deepbsde_all_sigmas:
        common.append("--deepbsde-all-sigmas")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    for family in ("ours", "pi", "deepbsde", "hjb"):
        command = [sys.executable, str(Path(__file__).resolve()), family, *common]
        print(f"[orchestrator] {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
    aggregate(args)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "family",
        choices=("all", "aggregate", "ours", "pi", "deepbsde", "hjb"),
        nargs="?",
        default="all",
    )
    result.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--case1", type=Path, default=DEFAULT_CASE1)
    result.add_argument("--case2", type=Path, default=DEFAULT_CASE2)
    result.add_argument("--pi-root", type=Path, default=DEFAULT_PI_ROOT)
    result.add_argument("--deepbsde-root", type=Path, default=DEFAULT_DEEPBSDE_ROOT)
    result.add_argument("--hjb-root", type=Path, default=DEFAULT_HJB_ROOT)
    result.add_argument("--hjb-taus", default="10")
    result.add_argument("--physical-alpha", type=float, default=1.0)
    result.add_argument("--physical-beta", type=float, default=40.0)
    result.add_argument("--physical-gamma", type=float, default=8000.0)
    result.add_argument("--rtol", type=float, default=1.0e-10)
    result.add_argument("--atol", type=float, default=1.0e-12)
    result.add_argument("--query-warmups", type=int, default=10)
    result.add_argument("--query-repeats", type=int, default=100)
    result.add_argument("--rollout-warmups", type=int, default=1)
    result.add_argument("--rollout-repeats", type=int, default=3)
    result.add_argument(
        "--evaluation-intervals",
        type=int,
        help="execute every feedback policy on this common number of ZOH intervals",
    )
    result.add_argument(
        "--deepbsde-all-sigmas",
        action="store_true",
        help="also save sigma=0.050 and 0.100 as secondary sensitivity runs",
    )
    return result


def main() -> None:
    args = parser().parse_args()
    if args.query_warmups < 0 or args.query_repeats < 1:
        raise ValueError("query timing requires nonnegative warmups and positive repeats")
    if args.rollout_warmups < 0 or args.rollout_repeats < 1:
        raise ValueError("rollout timing requires nonnegative warmups and positive repeats")
    if args.evaluation_intervals is not None and args.evaluation_intervals < 1:
        raise ValueError("--evaluation-intervals must be positive")
    if args.rtol <= 0.0 or args.atol <= 0.0:
        raise ValueError("integration tolerances must be positive")
    if (
        args.physical_alpha <= 0.0
        or args.physical_beta <= 0.0
        or args.physical_gamma <= 0.0
    ):
        raise ValueError("physical objective weights must be positive")
    if args.family == "all":
        orchestrate(args)
    elif args.family == "aggregate":
        aggregate(args)
    else:
        run_worker(args)


if __name__ == "__main__":
    main()
