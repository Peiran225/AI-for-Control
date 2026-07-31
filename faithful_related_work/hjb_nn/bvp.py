"""Characteristic BVP data generation for the faithful HJB-NN path."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp

from .problem import TumorEntropyProblem
from .sampling import select_largest_gradient_candidates


Array = np.ndarray


@dataclass(frozen=True)
class BVPSettings:
    tolerance: float = 1.0e-3
    max_nodes: int = 100_000
    adaptive_max_nodes: int = 20_000
    time_march_steps: int = 16
    initial_mesh_nodes: int = 41
    tau_start: float = 10.0
    ode_rtol: float = 1.0e-6
    ode_atol: float = 1.0e-8

    def validate(self) -> None:
        if (
            self.tolerance <= 0.0
            or self.max_nodes < 10
            or self.adaptive_max_nodes < 10
        ):
            raise ValueError("invalid BVP tolerance or max_nodes")
        if self.time_march_steps <= 0 or self.initial_mesh_nodes < 5:
            raise ValueError("invalid BVP mesh settings")
        if self.tau_start <= 0.0:
            raise ValueError("tau_start must be positive")


@dataclass
class CharacteristicDataset:
    t: Array
    X: Array
    A: Array
    V: Array
    U: Array
    trajectory_id: Array
    initial_states: Array

    def as_author_dict(self) -> dict[str, Array]:
        return {
            "t": self.t,
            "X": self.X,
            "A": self.A,
            "V": self.V,
            "U": self.U,
            "trajectory_id": self.trajectory_id,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            t=self.t,
            X=self.X,
            A=self.A,
            V=self.V,
            U=self.U,
            trajectory_id=self.trajectory_id,
            initial_states=self.initial_states,
        )


@dataclass(frozen=True)
class BVPSolutionMetadata:
    success: bool
    message: str
    nodes: int
    max_rms_residual: float
    boundary_residual_max_abs: float
    target_tau: float
    continuation_schedule: tuple[float, ...]
    warm_start: str


def continuation_schedule(target_tau: float, start_tau: float) -> tuple[float, ...]:
    """Build a deterministic geometric continuation ending exactly at target."""
    if target_tau <= 0.0 or start_tau <= 0.0:
        raise ValueError("tau values must be positive")
    current = max(float(start_tau), float(target_tau))
    values = [current]
    # Small temperature steps are materially more reliable than a direct
    # 10 -> 5 -> 2 jump for the 43-dimensional characteristic system.
    while current * 0.8 > target_tau * (1.0 + 1.0e-12):
        current *= 0.8
        values.append(current)
    if not np.isclose(values[-1], target_tau, rtol=0.0, atol=1.0e-14):
        values.append(float(target_tau))
    return tuple(values)


def _fixed_point_initial_guess(
    problem: TumorEntropyProblem,
    initial_state: Array,
    horizon: float,
    mesh_nodes: int,
    settings: BVPSettings,
) -> tuple[Array, Array]:
    """Construct a positive forward/backward initial BVP guess."""
    time = np.linspace(0.0, horizon, mesh_nodes)
    initial_state = np.asarray(initial_state, dtype=np.float64).reshape(problem.N_states)
    control = np.full(mesh_nodes, 0.5 * problem.umax, dtype=np.float64)
    state_solution = None
    costate = np.repeat(problem.alpha[:, None], mesh_nodes, axis=1)

    for _ in range(12):
        def state_rhs(t, state):
            u = float(np.interp(t, time, control))
            crowding = np.log1p(np.mean(state))
            return (problem.r - problem.phi * u - problem.M * crowding) * state

        state_solution = solve_ivp(
            state_rhs,
            (0.0, horizon),
            initial_state,
            t_eval=time,
            dense_output=True,
            method="DOP853",
            rtol=settings.ode_rtol,
            atol=settings.ode_atol,
        )
        if not state_solution.success or np.any(state_solution.y <= 0.0):
            raise RuntimeError(f"initial forward solve failed: {state_solution.message}")

        def costate_rhs(t, adjoint):
            state = state_solution.sol(t)
            u = float(np.interp(t, time, control))
            crowding = np.log1p(np.mean(state))
            coefficient = problem.r - problem.phi * u - problem.M * crowding
            coupling = np.sum(problem.M * adjoint * state)
            return -problem.beta - adjoint * coefficient + coupling / (
                problem.N_states + np.sum(state)
            )

        costate_solution = solve_ivp(
            costate_rhs,
            (horizon, 0.0),
            problem.alpha,
            t_eval=time[::-1],
            method="DOP853",
            rtol=settings.ode_rtol,
            atol=settings.ode_atol,
        )
        if not costate_solution.success:
            raise RuntimeError(f"initial backward solve failed: {costate_solution.message}")
        costate = costate_solution.y[:, ::-1]
        candidate = problem.U_star(
            np.vstack((state_solution.y, costate, np.zeros((1, mesh_nodes))))
        ).reshape(-1)
        control = 0.75 * control + 0.25 * candidate

    assert state_solution is not None
    running = problem.running_cost(
        state_solution.y, control.reshape((1, -1)), regularized=True
    ).reshape(-1)
    accumulated = np.zeros(mesh_nodes, dtype=np.float64)
    for index in range(mesh_nodes - 2, -1, -1):
        accumulated[index] = accumulated[index + 1] + 0.5 * (
            running[index] + running[index + 1]
        ) * (time[index + 1] - time[index])
    return time, np.vstack((state_solution.y, costate, accumulated.reshape((1, -1))))


def _extend_solution(solution, new_horizon: float, extra_nodes: int = 8) -> tuple[Array, Array]:
    old_horizon = float(solution.x[-1])
    extension = np.linspace(old_horizon, new_horizon, extra_nodes + 1)[1:]
    time = np.concatenate((solution.x, extension))
    guess = np.hstack((solution.y, np.repeat(solution.y[:, -1:], extra_nodes, axis=1)))
    return time, guess


def _model_warm_start(
    problem: TumorEntropyProblem,
    initial_state: Array,
    model,
    settings: BVPSettings,
) -> tuple[Array, Array]:
    solution = solve_ivp(
        problem.dynamics,
        (0.0, problem.t1),
        np.asarray(initial_state, dtype=np.float64).reshape(problem.N_states),
        args=(model.eval_U,),
        dense_output=False,
        method="DOP853",
        rtol=settings.ode_rtol,
        atol=settings.ode_atol,
        max_step=max(problem.t1 / 100.0, 1.0e-3),
    )
    if not solution.success or np.any(solution.y <= 0.0):
        raise RuntimeError(f"NN closed-loop warm start failed: {solution.message}")
    time = solution.t.reshape((1, -1))
    predicted_value, predicted_costate = model.bvp_guess(time, solution.y)
    # The BVP's last state is running cost-to-go with terminal value zero,
    # while HJBnet predicts the full value including terminal cost.
    terminal = float(problem.terminal_cost(solution.y[:, -1]))
    running_value = np.asarray(predicted_value, dtype=np.float64) - terminal
    running_value -= running_value[:, -1:]
    return solution.t, np.vstack((solution.y, predicted_costate, running_value))


def solve_characteristic_bvp(
    problem: TumorEntropyProblem,
    initial_state: Array,
    settings: BVPSettings,
    *,
    model=None,
) -> tuple[Any, BVPSolutionMetadata]:
    """Solve one characteristic using time marching or an NN warm start."""
    settings.validate()
    target_tau = float(problem.tau)
    schedule = continuation_schedule(target_tau, settings.tau_start)
    boundary = problem.make_bc(initial_state)
    warm_start_name = "nn" if model is not None else "time_marching"

    try:
        problem.tau = schedule[0]
        if model is not None:
            time_guess, augmented_guess = _model_warm_start(
                problem, initial_state, model, settings
            )
            solution = solve_bvp(
                problem.aug_dynamics,
                boundary,
                time_guess,
                augmented_guess,
                tol=settings.tolerance,
                max_nodes=settings.adaptive_max_nodes,
                verbose=0,
            )
            if not solution.success:
                raise RuntimeError(solution.message)
        else:
            horizons = np.linspace(0.0, problem.t1, settings.time_march_steps + 1)[1:]
            time_guess, augmented_guess = _fixed_point_initial_guess(
                problem,
                initial_state,
                float(horizons[0]),
                settings.initial_mesh_nodes,
                settings,
            )
            solution = None
            for horizon in horizons:
                if solution is not None:
                    time_guess, augmented_guess = _extend_solution(solution, float(horizon))
                solution = solve_bvp(
                    problem.aug_dynamics,
                    boundary,
                    time_guess,
                    augmented_guess,
                    tol=settings.tolerance,
                    max_nodes=settings.max_nodes,
                    verbose=0,
                )
                if not solution.success:
                    raise RuntimeError(f"time-march horizon {horizon:g}: {solution.message}")
            assert solution is not None

        for tau in schedule[1:]:
            problem.tau = tau
            solution = solve_bvp(
                problem.aug_dynamics,
                boundary,
                solution.x,
                solution.y,
                tol=settings.tolerance,
                max_nodes=(
                    settings.adaptive_max_nodes if model is not None else settings.max_nodes
                ),
                verbose=0,
            )
            if not solution.success:
                raise RuntimeError(f"tau continuation to {tau:g}: {solution.message}")

        if np.any(solution.y[: problem.N_states] <= 0.0):
            raise RuntimeError("converged BVP contains a nonpositive tumor state")
        metadata = BVPSolutionMetadata(
            success=True,
            message=str(solution.message),
            nodes=int(solution.x.size),
            max_rms_residual=float(np.max(solution.rms_residuals)),
            boundary_residual_max_abs=float(
                np.max(np.abs(boundary(solution.y[:, 0], solution.y[:, -1])))
            ),
            target_tau=target_tau,
            continuation_schedule=schedule,
            warm_start=warm_start_name,
        )
        return solution, metadata
    finally:
        problem.tau = target_tau


def dataset_from_solutions(
    problem: TumorEntropyProblem,
    solutions: list[Any],
    initial_states: Array,
    *,
    first_trajectory_id: int = 0,
) -> CharacteristicDataset:
    if not solutions:
        raise ValueError("at least one solution is required")
    initial_states = np.asarray(initial_states, dtype=np.float64)
    if initial_states.shape != (problem.N_states, len(solutions)):
        raise ValueError("initial-state columns do not match solutions")

    times: list[Array] = []
    states: list[Array] = []
    costates: list[Array] = []
    values: list[Array] = []
    controls: list[Array] = []
    trajectory_ids: list[Array] = []
    for offset, solution in enumerate(solutions):
        state = solution.y[: problem.N_states]
        costate = solution.y[problem.N_states : 2 * problem.N_states]
        control = problem.U_star(np.vstack((state, costate)))
        terminal = float(problem.terminal_cost(state[:, -1]))
        value = solution.y[-1:] + terminal
        columns = solution.x.size
        times.append(solution.x.reshape((1, -1)))
        states.append(state)
        costates.append(costate)
        controls.append(control)
        values.append(value)
        trajectory_ids.append(
            np.full((1, columns), first_trajectory_id + offset, dtype=np.int64)
        )
    return CharacteristicDataset(
        t=np.hstack(times),
        X=np.hstack(states),
        A=np.hstack(costates),
        V=np.hstack(values),
        U=np.hstack(controls),
        trajectory_id=np.hstack(trajectory_ids),
        initial_states=initial_states,
    )


def generate_initial_dataset(
    problem: TumorEntropyProblem,
    initial_states: Array,
    settings: BVPSettings,
    *,
    first_trajectory_id: int = 0,
) -> tuple[CharacteristicDataset, list[BVPSolutionMetadata]]:
    initial_states = np.asarray(initial_states, dtype=np.float64)
    solutions = []
    metadata = []
    for index in range(initial_states.shape[1]):
        try:
            solution, info = solve_characteristic_bvp(
                problem, initial_states[:, index], settings, model=None
            )
        except Exception as error:
            trajectory_id = first_trajectory_id + index
            raise RuntimeError(
                f"initial characteristic index {index} (trajectory_id={trajectory_id}) failed: {error}"
            ) from error
        solutions.append(solution)
        metadata.append(info)
    return (
        dataset_from_solutions(
            problem,
            solutions,
            initial_states,
            first_trajectory_id=first_trajectory_id,
        ),
        metadata,
    )


class AdaptiveDataController:
    """Generate Algorithm 4.1 data using max-gradient NN warm starts."""

    def __init__(
        self,
        problem: TumorEntropyProblem,
        settings: BVPSettings,
        rng: np.random.Generator,
        output_dir: Path,
        *,
        first_trajectory_id: int,
        max_failures: int = 32,
    ):
        self.problem = problem
        self.settings = settings
        self.rng = rng
        self.output_dir = output_dir
        self.next_trajectory_id = first_trajectory_id
        self.max_failures = max_failures
        self.events: list[dict[str, Any]] = []
        self.generated: list[CharacteristicDataset] = []

    def generate(self, model, desired_points: int, candidates_per_selection: int) -> dict[str, Array]:
        if desired_points <= 0 or candidates_per_selection <= 0:
            raise ValueError("adaptive generation budgets must be positive")
        accumulated: list[CharacteristicDataset] = []
        points = 0
        failures = 0
        while points < desired_points:
            candidates = self.problem.sample_initial_states(
                candidates_per_selection, self.rng
            )
            gradients = model.predict_A(
                np.zeros((1, candidates_per_selection), dtype=np.float64), candidates
            )
            selected, indices, norms = select_largest_gradient_candidates(
                candidates, gradients, count=1
            )
            event_index = len(self.events)
            candidate_path = self.output_dir / f"adaptive_candidates_{event_index:03d}.npz"
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                candidate_path,
                candidates=candidates,
                predicted_gradients=gradients,
                gradient_norms=norms,
                selected_indices=indices,
                selected_states=selected,
            )
            try:
                solution, metadata = solve_characteristic_bvp(
                    self.problem, selected[:, 0], self.settings, model=model
                )
            except (RuntimeError, FloatingPointError, MemoryError) as error:
                if isinstance(error, MemoryError):
                    gc.collect()
                failures += 1
                self.events.append(
                    {
                        "event": event_index,
                        "candidate_artifact": candidate_path.name,
                        "candidate_count": candidates_per_selection,
                        "selected_index": int(indices[0]),
                        "selected_gradient_norm": float(norms[indices[0]]),
                        "success": False,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                if failures >= self.max_failures:
                    raise RuntimeError(
                        "adaptive NN-warm-start BVP generation exceeded "
                        f"{self.max_failures} failures before adding {desired_points} points"
                    ) from error
                continue
            dataset = dataset_from_solutions(
                self.problem,
                [solution],
                selected,
                first_trajectory_id=self.next_trajectory_id,
            )
            self.next_trajectory_id += 1
            accumulated.append(dataset)
            self.generated.append(dataset)
            points += dataset.X.shape[1]
            self.events.append(
                {
                    "event": event_index,
                    "candidate_artifact": candidate_path.name,
                    "candidate_count": candidates_per_selection,
                    "selected_index": int(indices[0]),
                    "selected_gradient_norm": float(norms[indices[0]]),
                    "success": True,
                    "trajectory_id": self.next_trajectory_id - 1,
                    "points_added": int(dataset.X.shape[1]),
                    "bvp": metadata.__dict__,
                }
            )

        merged = CharacteristicDataset(
            t=np.hstack([data.t for data in accumulated]),
            X=np.hstack([data.X for data in accumulated]),
            A=np.hstack([data.A for data in accumulated]),
            V=np.hstack([data.V for data in accumulated]),
            U=np.hstack([data.U for data in accumulated]),
            trajectory_id=np.hstack([data.trajectory_id for data in accumulated]),
            initial_states=np.hstack([data.initial_states for data in accumulated]),
        )
        data = merged.as_author_dict()
        data.update(
            {
                "A_scaled": 2.0 * (data["A"] - model.A_lb) / (model.A_ub - model.A_lb) - 1.0,
                "U_scaled": 2.0 * (data["U"] - model.U_lb) / (model.U_ub - model.U_lb) - 1.0,
                "V_scaled": 2.0 * (data["V"] - model.V_min) / (model.V_max - model.V_min) - 1.0,
            }
        )
        return data
