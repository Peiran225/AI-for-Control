#!/usr/bin/env python3
"""Evaluate Table-1 baselines with the continuous strict PMP protocol.

This is deliberately a post-training evaluator.  It never updates a model or
re-optimizes a control.  Its numerical contract matches the retained proposed
methods:

* physical objective weights ``(alpha,beta,gamma)=(1,40,8000)``;
* continuous policy execution with DOP853 for state and open-loop adjoint;
* ``rtol=1e-10``, ``atol=1e-12`` and the ``T_32`` maximum step;
* the 25,601-point ``T_32`` diagnostic grid;
* the half-open singular interval ``[1.5,8.0)``;
* exclusion of every common ``n=800`` support node (16,120 retained points);
* ``epsilon=sqrt(RMS(psi)^2+RMS(dotpsi)^2+RMS(ddotpsi)^2)``.

For a stored interval-control schedule, each interval value is interpreted at
its interval midpoint (the collocation point represented by an interval
constant).  The continuous extension is the PCHIP through
``((k+1/2)T/n,u_k)``, with endpoint anchors ``u(0)=u_0`` and
``u(T)=u_(n-1)``.  The interpolated value is clipped to ``[0,3]``.  This is a
continuous-control evaluation, not zero-order hold.

The script writes one method/seed per immutable output directory.  Run the
separate aggregator after all requested seeds finish.
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
from typing import Any, Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for search_path in (ROOT, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from evaluate_table1_heldout_common import Problem  # noqa: E402


METHODS = (
    "direct_statewise",
    "direct_nominal_replay",
    "neural_pmp_learned",
    "neural_pmp_exact",
    "adaptive_hjb_nn",
    "pi_deeponet",
    "deepbsde",
)


def batch_dynamics(
    states: np.ndarray,
    controls: np.ndarray,
    parameters: dict[str, Any],
) -> np.ndarray:
    safe = np.maximum(np.asarray(states, dtype=np.float64), 1.0e-12)
    growth = (
        parameters["r"][None, :]
        - parameters["phi"][None, :] * controls[:, None]
        - parameters["M"][None, :]
        * np.log1p(safe.mean(axis=1))[:, None]
    )
    return growth * safe


def batch_dh_dn(
    states: np.ndarray,
    costates: np.ndarray,
    controls: np.ndarray,
    parameters: dict[str, Any],
) -> np.ndarray:
    safe = np.maximum(np.asarray(states, dtype=np.float64), 1.0e-12)
    drift = (
        parameters["r"][None, :]
        - parameters["phi"][None, :] * controls[:, None]
        - parameters["M"][None, :]
        * np.log1p(safe.mean(axis=1))[:, None]
    )
    coupling = (
        costates * parameters["M"][None, :] * safe
    ).sum(axis=1)
    denominator = safe.shape[1] + safe.sum(axis=1)
    return (
        parameters["beta"][None, :]
        + costates * drift
        - coupling[:, None] / denominator[:, None]
    )


def strict_quantities_batch(
    states: np.ndarray,
    costates: np.ndarray,
    controls: np.ndarray,
    parameters: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    safe = np.maximum(np.asarray(states, dtype=np.float64), 1.0e-12)
    denominator = safe.shape[2] + safe.sum(axis=2)
    coupled = (
        parameters["M"][None, None, :] * costates * safe
    ).sum(axis=2)
    rho = coupled / denominator
    phi_n = (parameters["phi"][None, None, :] * safe).sum(axis=2)
    psi = float(parameters["gamma"]) - (
        parameters["phi"][None, None, :] * costates * safe
    ).sum(axis=2)
    dot_psi = (
        parameters["phi"][None, None, :]
        * parameters["beta"][None, None, :]
        * safe
    ).sum(axis=2) - rho * phi_n
    drift0 = (
        parameters["r"][None, None, :]
        - parameters["M"][None, None, :]
        * np.log1p(safe.mean(axis=2))[:, :, None]
    )
    weighted_beta_dot = (
        parameters["phi"][None, None, :]
        * parameters["beta"][None, None, :]
        * drift0
        * safe
    ).sum(axis=2)
    phi_dot = (
        parameters["phi"][None, None, :] * drift0 * safe
    ).sum(axis=2)
    coupled_dot = (
        parameters["M"][None, None, :]
        * safe
        * (-parameters["beta"][None, None, :] + rho[:, :, None])
    ).sum(axis=2)
    denominator_dot = (drift0 * safe).sum(axis=2)
    rho_dot = (
        coupled_dot / denominator - rho * denominator_dot / denominator
    )
    a_term = weighted_beta_dot - rho_dot * phi_n - rho * phi_dot
    b_term = (
        -(
            parameters["phi"][None, None, :] ** 2
            * parameters["beta"][None, None, :]
            * safe
        ).sum(axis=2)
        + rho
        * (parameters["phi"][None, None, :] ** 2 * safe).sum(axis=2)
        - rho * phi_n**2 / denominator
    )
    return psi, dot_psi, a_term + b_term * controls


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


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
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def load_interval_control(path: Path, problem: Problem) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        control = np.asarray(payload["u"], dtype=np.float64).reshape(-1)
        stored_time = (
            np.asarray(payload["t"], dtype=np.float64).reshape(-1)
            if "t" in payload
            else None
        )
    if control.size == problem.intervals + 1:
        control = control[:-1]
    if control.size != problem.intervals:
        raise ValueError(
            f"{path}: expected {problem.intervals} interval values, "
            f"found {control.size}"
        )
    expected_time = np.linspace(
        0.0, problem.T, problem.intervals + 1, dtype=np.float64
    )
    if stored_time is not None and (
        stored_time.shape != expected_time.shape
        or not np.allclose(stored_time, expected_time, rtol=0.0, atol=2.0e-12)
    ):
        raise ValueError(f"{path}: stored t is not the common n=800 grid")
    if (
        not np.all(np.isfinite(control))
        or float(control.min()) < -1.0e-9
        or float(control.max()) > problem.umax + 1.0e-9
    ):
        raise ValueError(f"{path}: control is non-finite or infeasible")
    return np.clip(control, 0.0, problem.umax)


class SchedulePolicy:
    """PCHIP continuous extension of one or more fixed schedules."""

    def __init__(
        self,
        schedules: np.ndarray,
        problem: Problem,
    ) -> None:
        values = np.asarray(schedules, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.shape[1] != problem.intervals:
            raise ValueError("schedule width does not match n=800")
        interval_step = problem.T / problem.intervals
        knots = np.concatenate(
            (
                np.array([0.0]),
                (np.arange(problem.intervals, dtype=np.float64) + 0.5)
                * interval_step,
                np.array([problem.T]),
            )
        )
        anchored = np.column_stack(
            (values[:, 0], values, values[:, -1])
        )
        self._interpolator = PchipInterpolator(
            knots, anchored, axis=1, extrapolate=False
        )
        self._umax = float(problem.umax)
        self.schedules = values

    def __call__(self, physical_time: float, states: np.ndarray) -> np.ndarray:
        del states
        values = np.asarray(
            self._interpolator(float(np.clip(physical_time, 0.0, 10.0))),
            dtype=np.float64,
        ).reshape(-1)
        return np.clip(values, 0.0, self._umax)


class LoadedFeedbackPolicy:
    """Continuous execution wrapper around one retained feedback checkpoint."""

    def __init__(
        self,
        query: Callable[[int, float, np.ndarray], float],
        *,
        method: str,
        native_intervals: int,
        problem: Problem,
        deep_batch_query: Optional[
            Callable[[int, np.ndarray], np.ndarray]
        ] = None,
        batch_query: Optional[
            Callable[[np.ndarray, np.ndarray], np.ndarray]
        ] = None,
    ) -> None:
        self._query = query
        self._method = method
        self._native_intervals = int(native_intervals)
        self._problem = problem
        self._deep_batch_query = deep_batch_query
        self._batch_query = batch_query

    def _query_batch(
        self,
        index: int,
        physical_time: float,
        states: np.ndarray,
    ) -> np.ndarray:
        if self._deep_batch_query is not None:
            return np.asarray(
                self._deep_batch_query(
                    index, np.asarray(states, dtype=np.float64)
                ),
                dtype=np.float64,
            ).reshape(-1)
        if self._batch_query is not None:
            return np.asarray(
                self._batch_query(
                    np.full(
                        np.asarray(states).shape[0],
                        physical_time,
                        dtype=np.float64,
                    ),
                    np.asarray(states, dtype=np.float64),
                ),
                dtype=np.float64,
            ).reshape(-1)
        return np.asarray(
            [
                self._query(index, physical_time, state.copy())
                for state in np.asarray(states, dtype=np.float64)
            ],
            dtype=np.float64,
        )

    def __call__(self, physical_time: float, states: np.ndarray) -> np.ndarray:
        current_time = float(
            np.clip(physical_time, 0.0, self._problem.T)
        )
        if self._method != "deepbsde":
            index = min(
                int(
                    math.floor(
                        current_time
                        / self._problem.T
                        * self._native_intervals
                    )
                ),
                self._native_intervals - 1,
            )
            values = self._query_batch(index, current_time, states)
        else:
            # DeepBSDE has one distinct Z-network per native time slice.
            # Query adjacent native networks at the *same current state* and
            # linearly interpolate their resulting actions.  This removes the
            # architecture's original piecewise-time hold without inventing a
            # state interpolation or retraining the checkpoint.
            scaled = (
                current_time
                / self._problem.T
                * self._native_intervals
            )
            left = min(
                int(math.floor(scaled)),
                self._native_intervals - 1,
            )
            right = min(left + 1, self._native_intervals - 1)
            fraction = float(np.clip(scaled - left, 0.0, 1.0))
            native_step = self._problem.T / self._native_intervals
            left_values = self._query_batch(
                left, left * native_step, states
            )
            if right == left:
                values = left_values
            else:
                right_values = self._query_batch(
                    right, right * native_step, states
                )
                values = (
                    (1.0 - fraction) * left_values
                    + fraction * right_values
                )
        if (
            not np.all(np.isfinite(values))
            or float(values.min()) < -1.0e-8
            or float(values.max()) > self._problem.umax + 1.0e-8
        ):
            raise RuntimeError("feedback policy returned an infeasible action")
        return np.clip(values, 0.0, self._problem.umax)

    def query_grid(
        self,
        physical_times: np.ndarray,
        states: np.ndarray,
    ) -> np.ndarray:
        """Vectorize DeepBSDE native-slice queries over diagnostic times."""

        times = np.asarray(physical_times, dtype=np.float64).reshape(-1)
        state_grid = np.asarray(states, dtype=np.float64)
        if (
            state_grid.ndim != 3
            or state_grid.shape[0] != times.size
        ):
            raise ValueError("diagnostic state grid has an invalid shape")
        if self._batch_query is not None:
            batch = state_grid.shape[1]
            values = np.empty((times.size, batch), dtype=np.float64)
            chunk_size = 256
            for start in range(0, times.size, chunk_size):
                stop = min(start + chunk_size, times.size)
                flat_times = np.repeat(times[start:stop], batch)
                flat_states = state_grid[start:stop].reshape(
                    -1, state_grid.shape[2]
                )
                values[start:stop] = np.asarray(
                    self._batch_query(flat_times, flat_states),
                    dtype=np.float64,
                ).reshape(stop - start, batch)
            return np.clip(values, 0.0, self._problem.umax)
        if self._method != "deepbsde" or self._deep_batch_query is None:
            return np.stack(
                [
                    self(float(current_time), current_states)
                    for current_time, current_states in zip(
                        times, state_grid
                    )
                ],
                axis=0,
            )

        scaled = (
            np.clip(times, 0.0, self._problem.T)
            / self._problem.T
            * self._native_intervals
        )
        left = np.minimum(
            np.floor(scaled).astype(np.int64),
            self._native_intervals - 1,
        )
        right = np.minimum(left + 1, self._native_intervals - 1)
        fraction = np.clip(scaled - left, 0.0, 1.0)
        batch = state_grid.shape[1]
        left_values = np.empty((times.size, batch), dtype=np.float64)
        right_values = np.empty_like(left_values)
        for index in np.unique(np.concatenate((left, right))):
            left_mask = left == index
            if np.any(left_mask):
                queried = self._deep_batch_query(
                    int(index),
                    state_grid[left_mask].reshape(-1, state_grid.shape[2]),
                ).reshape(int(left_mask.sum()), batch)
                left_values[left_mask] = queried
            right_mask = right == index
            if np.any(right_mask):
                queried = self._deep_batch_query(
                    int(index),
                    state_grid[right_mask].reshape(
                        -1, state_grid.shape[2]
                    ),
                ).reshape(int(right_mask.sum()), batch)
                right_values[right_mask] = queried
        values = (
            (1.0 - fraction[:, None]) * left_values
            + fraction[:, None] * right_values
        )
        return np.clip(values, 0.0, self._problem.umax)


def load_pi_batch_policy(
    checkpoint: Path,
    problem: Problem,
) -> tuple[
    Callable[[np.ndarray, np.ndarray], np.ndarray],
    Callable[[], None],
    dict[str, Any],
]:
    """Load PI-DeepONet and expose its native batched feedback operation."""

    import torch

    from faithful_related_work.pi_deeponet.core import (
        ImprovedPolicy,
        TrainConfig,
        build_model,
        terminal_branch_values,
    )
    from faithful_related_work.pi_deeponet.problems import TumorAdaptation

    torch.set_num_threads(1)
    started = time.perf_counter()
    payload = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    config = TrainConfig(**payload["train_config"])
    problem_payload = dict(payload["problem"])
    adaptation = TumorAdaptation(
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
    model = build_model(adaptation, config).to(
        dtype=torch.float64, device="cpu"
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    sensors = torch.as_tensor(
        payload["sensor_states"], dtype=torch.float64
    )
    terminal_parameter = torch.tensor([1.0], dtype=torch.float64)
    branch = terminal_branch_values(
        adaptation, terminal_parameter, sensors
    )
    improved = ImprovedPolicy(
        model, adaptation, config.h, config.tie_tolerance
    )
    load_seconds = time.perf_counter() - started
    training = np.asarray(
        [adaptation.alpha, adaptation.beta, adaptation.gamma],
        dtype=np.float64,
    )
    physical = np.asarray(
        [problem.alpha, problem.beta, problem.gamma], dtype=np.float64
    )
    ratios = physical / training
    if not np.allclose(ratios, ratios[0], rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("PI-DeepONet objective is not proportional")

    def query_batch(
        physical_times: np.ndarray, states: np.ndarray
    ) -> np.ndarray:
        times = np.asarray(physical_times, dtype=np.float64).reshape(-1)
        state_batch = np.asarray(states, dtype=np.float64)
        if state_batch.shape != (times.size, adaptation.state_dim):
            raise ValueError("PI-DeepONet batch has an invalid shape")
        time_tensor = torch.as_tensor(times, dtype=torch.float64)
        state_tensor = torch.as_tensor(
            state_batch / adaptation.state_scale, dtype=torch.float64
        )
        branch_batch = branch.expand(times.size, -1)
        result = improved(branch_batch, time_tensor, state_tensor)
        return (
            result.control.detach().cpu().numpy().reshape(-1)
        )

    validation_states = np.stack(
        (
            np.full(adaptation.state_dim, problem.n0),
            np.linspace(8.0, 12.0, adaptation.state_dim),
        )
    )
    validation_times = np.asarray([3.25, 6.75], dtype=np.float64)
    batched_validation = query_batch(
        validation_times, validation_states
    )
    scalar_validation = np.asarray(
        [
            query_batch(
                validation_times[index : index + 1],
                validation_states[index : index + 1],
            )[0]
            for index in range(2)
        ]
    )
    validation_max_abs = float(
        np.max(np.abs(batched_validation - scalar_validation))
    )
    if validation_max_abs > 1.0e-12:
        raise RuntimeError("PI batched inference failed scalar validation")

    metadata = {
        "native_intervals": 200,
        "load_seconds": load_seconds,
        "training_objective_weights": training.tolist(),
        "objective_scale_to_physical": float(ratios[0]),
        "batched_policy_query": True,
        "batched_vs_scalar_max_abs": validation_max_abs,
    }
    return query_batch, (lambda: None), metadata


def load_deepbsde_batch_policy(
    checkpoint_dir: Path,
    problem: Problem,
) -> tuple[
    Callable[[int, np.ndarray], np.ndarray],
    Callable[[], None],
    dict[str, Any],
]:
    """Load DeepBSDE once and expose a batched native-slice action query."""

    import tensorflow as tf

    from faithful_related_work.deepbsde.equations import LogStateTumorHJB
    from faithful_related_work.deepbsde.runner import (
        AttrObject,
        _build_official_solver,
    )

    config = json.loads(
        (checkpoint_dir / "config.json").read_text(encoding="utf-8")
    )
    equation_config = config["eqn_config"]
    sigma = float(equation_config["sigma"])
    seed = int(equation_config["seed"])
    training = np.asarray(
        [
            equation_config["alpha"],
            equation_config["beta"],
            equation_config["gamma"],
        ],
        dtype=np.float64,
    )
    physical = np.asarray(
        [problem.alpha, problem.beta, problem.gamma], dtype=np.float64
    )
    ratios = physical / training
    if not np.allclose(ratios, ratios[0], rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("DeepBSDE checkpoint objective is not proportional")

    started = time.perf_counter()
    equation = LogStateTumorHJB(AttrObject(equation_config))
    solver = _build_official_solver(config, equation, seed)
    dummy_dw = np.zeros(
        (1, equation.dim, equation.num_time_interval),
        dtype=np.float64,
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
    phi = np.asarray(equation.params["phi"], dtype=np.float64)
    z_initial = np.asarray(
        solver.model.z_init.numpy()[0], dtype=np.float64
    )
    subnet_parameters: list[dict[str, Any]] = []
    for subnet in solver.model.subnet:
        batch_norms = []
        for layer in subnet.bn_layers:
            gamma, beta, mean, variance = [
                np.asarray(value, dtype=np.float64)
                for value in layer.get_weights()
            ]
            batch_norms.append(
                {
                    "gamma": gamma,
                    "beta": beta,
                    "mean": mean,
                    "variance": variance,
                    "epsilon": float(layer.epsilon),
                }
            )
        dense_layers = [
            [
                np.asarray(value, dtype=np.float64)
                for value in layer.get_weights()
            ]
            for layer in subnet.dense_layers
        ]
        subnet_parameters.append(
            {"batch_norms": batch_norms, "dense_layers": dense_layers}
        )

    def batch_norm(
        values: np.ndarray, parameters: dict[str, Any]
    ) -> np.ndarray:
        return (
            parameters["gamma"]
            * (values - parameters["mean"])
            / np.sqrt(parameters["variance"] + parameters["epsilon"])
            + parameters["beta"]
        )

    def numpy_subnet(index: int, log_states: np.ndarray) -> np.ndarray:
        parameters = subnet_parameters[index - 1]
        value = batch_norm(
            log_states, parameters["batch_norms"][0]
        )
        dense_layers = parameters["dense_layers"]
        for layer_index in range(len(dense_layers) - 1):
            kernel = dense_layers[layer_index][0]
            value = value @ kernel
            value = batch_norm(
                value, parameters["batch_norms"][layer_index + 1]
            )
            value = np.maximum(value, 0.0)
        final_weights = dense_layers[-1]
        value = value @ final_weights[0]
        if len(final_weights) == 2:
            value = value + final_weights[1]
        return batch_norm(
            value, parameters["batch_norms"][-1]
        )

    validation_states = np.stack(
        (
            np.full(equation.dim, problem.n0, dtype=np.float64),
            problem.n0
            * (1.0 + 0.20 * np.linspace(-1.0, 1.0, equation.dim)),
        )
    )
    validation_errors = []
    for index in (1, equation.num_time_interval // 2, equation.num_time_interval - 1):
        tensorflow_value = np.asarray(
            solver.model.subnet[index - 1](
                tf.convert_to_tensor(
                    np.log(validation_states), dtype=tf.float64
                ),
                training=False,
            ).numpy(),
            dtype=np.float64,
        )
        numpy_value = numpy_subnet(index, np.log(validation_states))
        validation_errors.append(
            float(np.max(np.abs(tensorflow_value - numpy_value)))
        )
    numpy_validation_max_abs = max(validation_errors)
    if numpy_validation_max_abs > 2.0e-10:
        raise RuntimeError(
            "DeepBSDE NumPy inference does not match TensorFlow: "
            f"{numpy_validation_max_abs:.6g}"
        )

    def query_batch(index: int, states: np.ndarray) -> np.ndarray:
        state_batch = np.asarray(states, dtype=np.float64)
        if state_batch.ndim != 2 or state_batch.shape[1] != equation.dim:
            raise ValueError("DeepBSDE batch has an invalid state shape")
        if index == 0:
            z_value = np.broadcast_to(
                z_initial,
                state_batch.shape,
            )
        else:
            z_value = numpy_subnet(index, np.log(state_batch)) / equation.dim
        gradient = z_value / equation.sigma
        switching = equation.gamma - np.sum(
            gradient * phi[None, :], axis=1
        )
        return np.where(switching < 0.0, equation.umax, 0.0)

    def close() -> None:
        tf.keras.backend.clear_session()

    metadata = {
        "native_intervals": int(equation.num_time_interval),
        "load_seconds": load_seconds,
        "training_objective_weights": training.tolist(),
        "objective_scale_to_physical": float(ratios[0]),
        "deepbsde_sigma": sigma,
        "checkpoint_seed": seed,
        "batched_native_slice_query": True,
        "numpy_inference_matches_tensorflow_max_abs": (
            numpy_validation_max_abs
        ),
    }
    return query_batch, close, metadata


def common_parameters(problem: Problem) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in problem.vectors().items()
    }
    parameters["gamma"] = float(problem.gamma)
    return parameters


def strict_time_grid(
    problem: Problem,
    *,
    dense_points: int,
    interior_start: float,
    interior_end: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (dense_points - 1) % problem.intervals != 0:
        raise ValueError("dense grid must contain every n=800 support node")
    dense_time = np.linspace(
        0.0, problem.T, dense_points, dtype=np.float64
    )
    multiplier = (dense_points - 1) // problem.intervals
    support = np.zeros(dense_points, dtype=bool)
    support[::multiplier] = True
    mask = (
        (dense_time >= interior_start)
        & (dense_time < interior_end)
        & ~support
    )
    if (
        dense_points == 25_601
        and interior_start == 1.5
        and interior_end == 8.0
        and int(mask.sum()) != 16_120
    ):
        raise RuntimeError(
            f"T_32 strict point count drifted: {int(mask.sum())} != 16120"
        )
    return dense_time, support, mask


def evaluate(
    policy: Callable[[float, np.ndarray], np.ndarray],
    initial_states: np.ndarray,
    selected_time: np.ndarray,
    problem: Problem,
    *,
    rtol: float,
    atol: float,
    max_step: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    started = time.perf_counter()
    initial_states = np.asarray(initial_states, dtype=np.float64)
    batch, components = initial_states.shape
    parameters = common_parameters(problem)
    initial_augmented = np.zeros((batch, components + 1), dtype=np.float64)
    initial_augmented[:, :components] = initial_states

    def forward_rhs(physical_time: float, flat: np.ndarray) -> np.ndarray:
        augmented = flat.reshape(batch, components + 1)
        states = augmented[:, :components]
        controls = policy(physical_time, states)
        drift = batch_dynamics(states, controls, parameters)
        running = (
            states @ np.asarray(parameters["beta"])
            + float(parameters["gamma"]) * controls
        )
        return np.column_stack((drift, running)).reshape(-1)

    forward = solve_ivp(
        forward_rhs,
        (0.0, problem.T),
        initial_augmented.reshape(-1),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=True,
        max_step=max_step,
    )
    if not forward.success:
        raise RuntimeError(f"state integration failed: {forward.message}")
    terminal = forward.sol(problem.T).reshape(batch, components + 1)
    if (
        not np.all(np.isfinite(terminal))
        or float(terminal[:, :components].min()) <= 0.0
    ):
        raise RuntimeError("state integration produced an invalid state")
    objective = (
        terminal[:, components]
        + terminal[:, :components] @ np.asarray(parameters["alpha"])
    )

    def costate_rhs(physical_time: float, flat: np.ndarray) -> np.ndarray:
        states = forward.sol(physical_time).reshape(
            batch, components + 1
        )[:, :components]
        costates = flat.reshape(batch, components)
        controls = policy(physical_time, states)
        return -batch_dh_dn(
            states, costates, controls, parameters
        ).reshape(-1)

    terminal_costate = np.broadcast_to(
        np.asarray(parameters["alpha"]), (batch, components)
    ).copy()
    backward = solve_ivp(
        costate_rhs,
        (problem.T, 0.0),
        terminal_costate.reshape(-1),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=True,
        max_step=max_step,
    )
    if not backward.success:
        raise RuntimeError(f"costate integration failed: {backward.message}")

    states = (
        forward.sol(selected_time)
        .T.reshape(selected_time.size, batch, components + 1)[
            :, :, :components
        ]
    )
    costates = (
        backward.sol(selected_time)
        .T.reshape(selected_time.size, batch, components)
    )
    if hasattr(policy, "query_grid"):
        controls = policy.query_grid(selected_time, states)
    else:
        controls = np.stack(
            [
                policy(float(current_time), current_states)
                for current_time, current_states in zip(
                    selected_time, states
                )
            ],
            axis=0,
        )
    psi, dot_psi, ddot_psi = strict_quantities_batch(
        states, costates, controls, parameters
    )
    rms_psi = np.sqrt(np.mean(psi**2, axis=0))
    rms_dot = np.sqrt(np.mean(dot_psi**2, axis=0))
    rms_ddot = np.sqrt(np.mean(ddot_psi**2, axis=0))
    epsilon = np.sqrt(rms_psi**2 + rms_dot**2 + rms_ddot**2)
    result = {
        "J": objective,
        "RMS_psi": rms_psi,
        "RMS_dot_psi": rms_dot,
        "RMS_ddot_psi": rms_ddot,
        "epsilon": epsilon,
    }
    timing = {
        "elapsed_seconds": time.perf_counter() - started,
        "forward_nfev": int(forward.nfev),
        "backward_nfev": int(backward.nfev),
    }
    return result, timing


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="nominal/shared n=800 interval-control NPZ",
    )
    parser.add_argument(
        "--resistant-source",
        type=Path,
        help=(
            "optional state-specific r=0.20 Direct NPZ; forbidden for "
            "non-Direct methods"
        ),
    )
    parser.add_argument("--seed-label", default="selected")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resistant-radius", type=float, default=0.20)
    parser.add_argument("--dense-points", type=int, default=25_601)
    parser.add_argument("--interior-start", type=float, default=1.5)
    parser.add_argument("--interior-end", type=float, default=8.0)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--hjb-tau", type=float, default=10.0)
    parser.add_argument(
        "--integration-max-step",
        type=float,
        help=(
            "optional DOP853 maximum step; default is the T_32 spacing. "
            "The diagnostic coordinates remain the full T_32 strict mask."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    problem = Problem()
    source = resolve(args.source)
    resistant_source = (
        resolve(args.resistant_source)
        if args.resistant_source is not None
        else None
    )
    output_dir = resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} exists; choose a new immutable output directory"
        )
    if not source.exists():
        raise FileNotFoundError(source)
    if resistant_source is not None and not resistant_source.is_file():
        raise FileNotFoundError(resistant_source)
    if resistant_source is not None and args.method != "direct_statewise":
        raise ValueError("--resistant-source is only valid for direct_statewise")
    if args.method == "direct_statewise" and resistant_source is None:
        raise ValueError("direct_statewise requires --resistant-source")
    if args.method == "direct_nominal_replay" and resistant_source is not None:
        raise ValueError("direct_nominal_replay uses one shared schedule")
    if not np.isclose(args.resistant_radius, 0.20):
        raise ValueError("approved Table-1 resistant radius is exactly 0.20")

    dense_time, support, strict_mask = strict_time_grid(
        problem,
        dense_points=args.dense_points,
        interior_start=args.interior_start,
        interior_end=args.interior_end,
    )
    selected_time = dense_time[strict_mask]
    integration_max_step = (
        float(args.integration_max_step)
        if args.integration_max_step is not None
        else problem.T / (args.dense_points - 1)
    )
    if not 0.0 < integration_max_step <= problem.T:
        raise ValueError("--integration-max-step must lie in (0,T]")
    nominal = np.full(problem.m, problem.n0, dtype=np.float64)
    direction = np.linspace(-1.0, 1.0, problem.m, dtype=np.float64)
    resistant = problem.n0 * (
        1.0 + args.resistant_radius * direction
    )
    initial_states = np.stack((nominal, resistant), axis=0)

    feedback_method = args.method in {
        "adaptive_hjb_nn",
        "pi_deeponet",
        "deepbsde",
    }
    close = lambda: None
    policy_metadata: dict[str, Any] = {}
    if feedback_method:
        if resistant_source is not None:
            raise ValueError("feedback methods use one retained checkpoint")
        if args.method == "deepbsde":
            deep_batch_query, close, policy_metadata = (
                load_deepbsde_batch_policy(source, problem)
            )
            policy = LoadedFeedbackPolicy(
                lambda _index, _time, _state: float("nan"),
                method=args.method,
                native_intervals=int(
                    policy_metadata["native_intervals"]
                ),
                problem=problem,
                deep_batch_query=deep_batch_query,
            )
        elif args.method == "pi_deeponet":
            batch_query, close, policy_metadata = load_pi_batch_policy(
                source, problem
            )
            policy = LoadedFeedbackPolicy(
                lambda _index, _time, _state: float("nan"),
                method=args.method,
                native_intervals=int(
                    policy_metadata["native_intervals"]
                ),
                problem=problem,
                batch_query=batch_query,
            )
        else:
            from evaluate_table1_heldout_common import load_policy

            query, close, policy_metadata = load_policy(
                args.method, source, hjb_tau=args.hjb_tau
            )
            policy = LoadedFeedbackPolicy(
                query,
                method=args.method,
                native_intervals=int(
                    policy_metadata["native_intervals"]
                ),
                problem=problem,
            )
    else:
        nominal_control = load_interval_control(source, problem)
        resistant_control = (
            load_interval_control(resistant_source, problem)
            if resistant_source is not None
            else nominal_control
        )
        schedules = np.stack((nominal_control, resistant_control), axis=0)
        policy = SchedulePolicy(schedules, problem)
    try:
        result, timing = evaluate(
            policy,
            initial_states,
            selected_time,
            problem,
            rtol=args.rtol,
            atol=args.atol,
            max_step=integration_max_step,
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
                "source": str(
                    source if index == 0 or resistant_source is None
                    else resistant_source
                ),
                "continuous_extension": (
                    "native continuous physical-time feedback query"
                    if args.method in {"adaptive_hjb_nn", "pi_deeponet"}
                    else (
                        "linear interpolation of adjacent DeepBSDE native-"
                        "slice actions queried at the same current state"
                        if args.method == "deepbsde"
                        else (
                            "midpoint PCHIP(t_(k+1/2),u_k), with "
                            "u(0)=u_0, u(T)=u_799, then clip to [0,3]"
                        )
                    )
                ),
            }
        )

    output_dir.mkdir(parents=True)
    write_csv(output_dir / "rows.csv", rows)
    sources = [source] + (
        [resistant_source] if resistant_source is not None else []
    )
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
    summary = {
        "schema": "table1-baseline-continuous-strict-v1",
        "method": args.method,
        "seed_label": args.seed_label,
        "rows": rows,
        "problem": problem.__dict__,
        "sources": [source_record(path) for path in sources],
        "policy_metadata": policy_metadata,
        "protocol": {
            "policy_execution": "continuous; no zero-order hold",
            "discrete_control_extension": {
                "interpolant": "PCHIP",
                "knots": "t_(k+1/2)=(k+1/2)*T/800, k=0,...,799",
                "endpoint_anchors": "u(0)=u_0 and u(T)=u_799",
                "feasibility": "clip interpolated value to [0,3]",
            },
            "feedback_continuous_extension": (
                "native continuous physical-time query"
                if args.method in {"adaptive_hjb_nn", "pi_deeponet"}
                else (
                    "linear interpolation between adjacent native-slice "
                    "actions, both queried at the same current state"
                    if args.method == "deepbsde"
                    else None
                )
            ),
            "state_solver": "DOP853",
            "costate": "open-loop PMP adjoint along realized trajectory",
            "costate_solver": "DOP853",
            "rtol": args.rtol,
            "atol": args.atol,
            "maximum_step": integration_max_step,
            "default_T32_maximum_step_used": (
                args.integration_max_step is None
            ),
            "dense_grid": "T_32",
            "dense_points": args.dense_points,
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
        "outputs": {"rows_csv": str((output_dir / "rows.csv").resolve())},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
