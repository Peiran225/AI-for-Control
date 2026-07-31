"""Original-paper satellite checkpoint validation and retraining paths."""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io
from scipy.integrate import simpson, solve_bvp, solve_ivp

from .official_adapter import checkpoint_payload, make_instrumented_hjbnet
from .sampling import extract_initial_states, validate_trajectory_disjoint


def _author_root(repo_root: Path) -> Path:
    return repo_root / "external" / "HJB_NN"


def _load_author_api(repo_root: Path):
    author_root = _author_root(repo_root)
    if str(author_root) not in sys.path:
        sys.path.insert(0, str(author_root))
    from examples.satellite.problem_def import config_NN, setup_problem
    from utilities.neural_networks import HJBnet_t0
    from utilities.other import load_NN

    return config_NN, setup_problem, HJBnet_t0, load_NN


def _t0_data(raw: dict[str, Any], problem) -> dict[str, np.ndarray]:
    time = np.asarray(raw["t"], dtype=np.float64)
    indices = np.flatnonzero(np.isclose(time.reshape(-1), 0.0, rtol=0.0, atol=1.0e-12))
    state = np.asarray(raw["X"], dtype=np.float64)[:, indices]
    costate = np.asarray(raw["A"], dtype=np.float64)[:, indices]
    value = np.asarray(raw["V"], dtype=np.float64)[:, indices]
    control = problem.U_star(np.vstack((state, costate)))
    return {
        "t": np.zeros((1, indices.size), dtype=np.float64),
        "X": state,
        "A": costate,
        "V": value,
        "U": control,
        "trajectory_id": np.arange(indices.size, dtype=np.int64).reshape((1, -1)),
    }


def _relative_metrics(
    value_prediction: np.ndarray,
    costate_prediction: np.ndarray,
    control_prediction: np.ndarray,
    data: dict[str, np.ndarray],
) -> dict[str, float]:
    value_error = float(
        np.mean(np.abs(value_prediction - data["V"])) / np.mean(np.abs(data["V"]))
    )
    costate_error = float(
        np.mean(np.linalg.norm(costate_prediction - data["A"], axis=0))
        / np.mean(np.linalg.norm(data["A"], axis=0))
    )
    control_error = float(
        np.mean(np.linalg.norm(control_prediction - data["U"], axis=0))
        / np.mean(np.linalg.norm(data["U"], axis=0))
    )
    return {
        "value_RMAE": value_error,
        "costate_RML2": costate_error,
        "control_RML2": control_error,
    }


def load_satellite_data(repo_root: Path):
    author_root = _author_root(repo_root)
    train_raw = scipy.io.loadmat(author_root / "examples" / "satellite" / "data_train.mat")
    validation_raw = scipy.io.loadmat(author_root / "examples" / "satellite" / "data_test.mat")
    train_initial = extract_initial_states(train_raw["t"], train_raw["X"])
    validation_initial = extract_initial_states(validation_raw["t"], validation_raw["X"])
    validate_trajectory_disjoint(train_initial, validation_initial)
    return train_raw, validation_raw, train_initial, validation_initial


def _network_metadata(config) -> dict[str, Any]:
    layers = [int(value) for value in config.layers]
    parameters = sum(
        layers[index + 1] * layers[index] + layers[index + 1]
        for index in range(len(layers) - 1)
    )
    return {
        "layers": layers,
        "trainable_parameters": int(parameters),
        "hidden_activation": "tanh",
        "output_activation": "linear",
        "time_input": False,
    }


def validate_released_checkpoint(repo_root: Path) -> dict[str, Any]:
    """Validate the released t=0 satellite model on held-out BVP trajectories."""
    config_NN, setup_problem, HJBnet_t0, load_NN = _load_author_api(repo_root)
    problem = setup_problem()
    config = config_NN(problem.N_states, problem.t1, False)
    checkpoint = _author_root(repo_root) / "examples" / "satellite" / "t0" / "V_model.mat"
    parameters, scaling, stored = load_NN(str(checkpoint), return_stats=True)
    train_raw, validation_raw, train_initial, validation_initial = load_satellite_data(repo_root)
    train = _t0_data(train_raw, problem)
    validation = _t0_data(validation_raw, problem)

    released_train_scaling = {
        "lb": np.min(train["X"], axis=1, keepdims=True),
        "ub": np.max(train["X"], axis=1, keepdims=True),
        "A_lb": np.min(train["A"], axis=1, keepdims=True),
        "A_ub": np.max(train["A"], axis=1, keepdims=True),
        "U_lb": np.min(train["U"], axis=1, keepdims=True),
        "U_ub": np.max(train["U"], axis=1, keepdims=True),
        "V_min": np.asarray([np.min(train["V"])], dtype=np.float64),
        "V_max": np.asarray([np.max(train["V"])], dtype=np.float64),
    }
    scaling_differences = {
        key: float(
            np.max(
                np.abs(
                    np.asarray(scaling[key], dtype=np.float64).reshape(-1)
                    - np.asarray(released_train_scaling[key], dtype=np.float64).reshape(-1)
                )
            )
        )
        for key in released_train_scaling
    }
    scaling_matches = all(value <= 1.0e-12 for value in scaling_differences.values())

    model = HJBnet_t0(problem, scaling, config, parameters)
    value = model.predict_V(validation["t"], validation["X"])
    costate = model.predict_A(validation["t"], validation["X"])
    control = model.eval_U(validation["t"], validation["X"])
    metrics = _relative_metrics(value, costate, control, validation)
    model.sess.close()

    stored_train_time, stored_costate_error, stored_control_error = stored
    return {
        "benchmark": "author satellite architecture/data/released checkpoint through audited TensorFlow 2 compatibility port",
        "upstream_tracked_dirty": True,
        "tensorflow2_compatibility_port": True,
        "checkpoint": str(checkpoint.relative_to(repo_root)),
        "author_classes": {
            "problem": "external/HJB_NN/examples/satellite/problem_def.py::setup_problem",
            "network": "external/HJB_NN/utilities/neural_networks.py::HJBnet_t0 via tracked TensorFlow-2 compatibility diff",
            "training": "author full-batch L-BFGS logic via tracked compatibility replacement for removed tensorflow.contrib interface",
        },
        "network": _network_metadata(config),
        "validation_source": "independently generated data_test.mat characteristic BVP trajectories",
        "train_trajectories": int(train_initial.shape[1]),
        "validation_trajectories": int(validation_initial.shape[1]),
        "trajectory_disjoint": True,
        "checkpoint_scaling_matches_released_training_t0": scaling_matches,
        "checkpoint_vs_released_training_t0_scaling_max_abs_difference": scaling_differences,
        "checkpoint_scaling_provenance_warning": (
            "released checkpoint scaling is not reproducible from released data_train.mat t=0 points; "
            "the exact initial training set or saving provenance is therefore missing"
            if not scaling_matches
            else None
        ),
        "validation_points_t0": int(validation["X"].shape[1]),
        "recomputed_metrics": metrics,
        "checkpoint_recorded_metrics": {
            "train_time_seconds": float(np.asarray(stored_train_time).reshape(-1)[0]),
            "costate_RML2": float(np.asarray(stored_costate_error).reshape(-1)[0]),
            "control_RML2": float(np.asarray(stored_control_error).reshape(-1)[0]),
        },
    }


def _eval_controller(controller, time_value: float, state: np.ndarray) -> np.ndarray:
    """Evaluate an author-style controller at one scalar time/state pair."""
    control = controller(
        np.asarray([[time_value]], dtype=np.float64),
        np.asarray(state, dtype=np.float64).reshape((-1, 1)),
    )
    return np.asarray(control, dtype=np.float64).reshape(-1)


def _author_noise_rk4_step(problem, time_value, state, control, step_size):
    """Reproduce simulate_noise.py's released four-substep implementation exactly."""
    substep = step_size / 4.0
    state = np.asarray(state, dtype=np.float64)
    result = np.array(state, copy=True)

    def held_control(_time, _state):
        return np.asarray(control, dtype=np.float64).reshape((-1, 1))

    # Deliberately retain the author script's behavior: every substep starts from
    # `state`, rather than advancing from `result`.  The independent realization
    # below uses a high-accuracy interval solver and is reported separately.
    for _ in range(4):
        k1 = substep * problem.dynamics(time_value, state, held_control)
        k2 = substep * problem.dynamics(
            time_value + substep / 2.0, state + k1 / 2.0, held_control
        )
        k3 = substep * problem.dynamics(
            time_value + substep / 2.0, state + k2 / 2.0, held_control
        )
        k4 = substep * problem.dynamics(time_value + substep, state + k3, held_control)
        result += (k1 + 2.0 * (k2 + k3) + k4) / 6.0
    return result


def _accurate_zoh_rollout(problem, controller, time_grid, noise, initial_state):
    """Independently realize a noisy ZOH policy with tight interval integration."""
    states = np.empty((problem.N_states, time_grid.size), dtype=np.float64)
    controls = np.empty((problem.N_controls, time_grid.size - 1), dtype=np.float64)
    states[:, 0] = initial_state
    accumulated_running_cost = 0.0

    for index in range(time_grid.size - 1):
        control = _eval_controller(
            controller, time_grid[index], states[:, index] + noise[:, index]
        )
        controls[:, index] = control

        def held_control(_time, _state):
            return control.reshape((-1, 1))

        def augmented_dynamics(time_value, augmented_state):
            state = augmented_state[:-1]
            state_rate = problem.dynamics(time_value, state, held_control)
            running = problem.running_cost(
                state.reshape((-1, 1)), control.reshape((-1, 1))
            )
            return np.concatenate((state_rate, [float(np.asarray(running).reshape(-1)[0])]))

        initial_augmented = np.concatenate(
            (states[:, index], [accumulated_running_cost])
        )
        solution = solve_ivp(
            augmented_dynamics,
            (float(time_grid[index]), float(time_grid[index + 1])),
            initial_augmented,
            method="DOP853",
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        if not solution.success:
            raise RuntimeError(f"accurate ZOH rollout failed: {solution.message}")
        states[:, index + 1] = solution.y[:-1, -1]
        accumulated_running_cost = float(solution.y[-1, -1])

    terminal_cost = float(problem.terminal_cost(states[:, -1]))
    return {
        "states": states,
        "controls": controls,
        "running_cost": accumulated_running_cost,
        "terminal_cost": terminal_cost,
        "J": accumulated_running_cost + terminal_cost,
    }


def evaluate_satellite_closed_loop(
    repo_root: Path, output_dir: Path, *, seed: int
) -> dict[str, Any]:
    """Re-run the released noisy-ZOH protocol and independently realize its cost."""
    config_NN, setup_problem, HJBnet_t0, load_NN = _load_author_api(repo_root)
    problem = setup_problem()
    config = config_NN(problem.N_states, problem.t1, False)
    checkpoint = _author_root(repo_root) / "examples" / "satellite" / "t0" / "V_model.mat"
    parameters, scaling = load_NN(str(checkpoint))
    model = HJBnet_t0(problem, scaling, config, parameters)

    np.random.seed(seed)
    candidates = problem.sample_X0(100)
    selected = model.get_largest_A(
        np.zeros((1, candidates.shape[1]), dtype=np.float64), candidates, 1
    )
    selected_index = int(np.asarray(selected).reshape(-1)[-1])
    initial_state = np.asarray(candidates[:, selected_index], dtype=np.float64)
    time_grid = np.arange(0.0, problem.t1 + config.dt / 2.0, config.dt)
    noise = config.sigma * np.random.randn(problem.N_states, time_grid.size)

    author_nn_states = np.empty((problem.N_states, time_grid.size), dtype=np.float64)
    author_lqr_states = np.empty_like(author_nn_states)
    author_nn_states[:, 0] = initial_state
    author_lqr_states[:, 0] = initial_state
    for index in range(time_grid.size - 1):
        nn_control = _eval_controller(
            model.eval_U,
            time_grid[index],
            author_nn_states[:, index] + noise[:, index],
        )
        lqr_control = _eval_controller(
            problem.U_LQR,
            time_grid[index],
            author_lqr_states[:, index] + noise[:, index],
        )
        author_nn_states[:, index + 1] = _author_noise_rk4_step(
            problem, time_grid[index], author_nn_states[:, index], nn_control, config.dt
        )
        author_lqr_states[:, index + 1] = _author_noise_rk4_step(
            problem, time_grid[index], author_lqr_states[:, index], lqr_control, config.dt
        )

    value_nn, costate_nn, author_nn_controls = model.bvp_guess(
        time_grid.reshape((1, -1)), author_nn_states + noise, eval_U=True
    )
    # Reproduce the saved LQR control trace exactly: the released script omits
    # measurement noise only at index zero, although its applied first-interval
    # control does include W[:,0].
    author_lqr_controls = np.empty((problem.N_controls, time_grid.size), dtype=np.float64)
    author_lqr_controls[:, 0] = _eval_controller(problem.U_LQR, 0.0, initial_state)
    for index in range(1, time_grid.size):
        author_lqr_controls[:, index] = _eval_controller(
            problem.U_LQR,
            time_grid[index],
            author_lqr_states[:, index] + noise[:, index],
        )

    bvp = solve_bvp(
        problem.aug_dynamics,
        problem.make_bc(initial_state),
        time_grid,
        np.vstack((author_nn_states, costate_nn, value_nn)),
        tol=config.data_tol,
        max_nodes=config.max_nodes,
    )
    if not bvp.success:
        model.sess.close()
        raise RuntimeError(f"satellite characteristic BVP failed: {bvp.message}")
    author_bvp = bvp.sol(time_grid)
    author_bvp_controls = problem.U_star(author_bvp)

    official_nn_cost = float(
        problem.compute_cost(time_grid, author_nn_states, author_nn_controls)[0, 0]
    )
    official_lqr_cost = float(
        problem.compute_cost(time_grid, author_lqr_states, author_lqr_controls)[0, 0]
    )
    official_raw_bvp_value = float(author_bvp[-1, 0])
    author_bvp_terminal = float(
        problem.terminal_cost(author_bvp[: problem.N_states, -1])
    )

    accurate_nn = _accurate_zoh_rollout(
        problem, model.eval_U, time_grid, noise, initial_state
    )
    accurate_lqr = _accurate_zoh_rollout(
        problem, problem.U_LQR, time_grid, noise, initial_state
    )
    dense_time = np.linspace(0.0, problem.t1, 20001)
    dense_bvp = bvp.sol(dense_time)
    dense_bvp_controls = problem.U_star(dense_bvp)
    dense_running = np.asarray(
        problem.running_cost(dense_bvp[: problem.N_states], dense_bvp_controls),
        dtype=np.float64,
    ).reshape(-1)
    optimal_running = float(simpson(dense_running, x=dense_time))
    optimal_terminal = float(problem.terminal_cost(dense_bvp[: problem.N_states, -1]))
    optimal_realized = optimal_running + optimal_terminal
    model.sess.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "closed_loop.npz",
        t=time_grid,
        noise=noise,
        candidate_initial_states=candidates,
        selected_index=np.asarray([selected_index], dtype=np.int64),
        initial_state=initial_state,
        author_nn_states=author_nn_states,
        author_nn_controls=np.asarray(author_nn_controls, dtype=np.float64),
        author_lqr_states=author_lqr_states,
        author_lqr_controls=author_lqr_controls,
        author_bvp_states=author_bvp[: problem.N_states],
        author_bvp_costates=author_bvp[problem.N_states : 2 * problem.N_states],
        author_bvp_value_state=author_bvp[-1:],
        author_bvp_controls=author_bvp_controls,
        accurate_nn_states=accurate_nn["states"],
        accurate_nn_interval_controls=accurate_nn["controls"],
        accurate_lqr_states=accurate_lqr["states"],
        accurate_lqr_interval_controls=accurate_lqr["controls"],
        dense_bvp_t=dense_time,
        dense_bvp=dense_bvp,
        dense_bvp_controls=dense_bvp_controls,
    )

    return {
        "benchmark": "released satellite noisy-ZOH closed-loop protocol plus independent high-accuracy realization",
        "seed": seed,
        "checkpoint": str(checkpoint.relative_to(repo_root)),
        "protocol": {
            "candidate_initial_states": 100,
            "selection": "largest released-NN predicted costate norm",
            "sample_period": float(config.dt),
            "measurement_noise_sigma": float(config.sigma),
            "zero_order_hold": True,
        },
        "network": _network_metadata(config),
        "exact_figure_3_replay_possible": False,
        "exact_figure_3_limitation": "paper and release do not identify Figure 3's RNG seed or retain its exact final simulation artifact",
        "bvp": {
            "success": bool(bvp.success),
            "status": int(bvp.status),
            "nodes": int(bvp.x.size),
            "tolerance": float(config.data_tol),
        },
        "author_script_reproduction": {
            "NN_cost": official_nn_cost,
            "LQR_cost": official_lqr_cost,
            "raw_BVP_value_state_at_t0": official_raw_bvp_value,
            "raw_BVP_plus_terminal_cost": official_raw_bvp_value + author_bvp_terminal,
            "NN_gap_percent_vs_raw_BVP": 100.0 * (official_nn_cost / official_raw_bvp_value - 1.0),
            "LQR_gap_percent_vs_raw_BVP": 100.0 * (official_lqr_cost / official_raw_bvp_value - 1.0),
        },
        "independent_realization": {
            "NN_running_cost": accurate_nn["running_cost"],
            "NN_terminal_cost": accurate_nn["terminal_cost"],
            "NN_J": accurate_nn["J"],
            "LQR_running_cost": accurate_lqr["running_cost"],
            "LQR_terminal_cost": accurate_lqr["terminal_cost"],
            "LQR_J": accurate_lqr["J"],
            "optimal_running_cost": optimal_running,
            "optimal_terminal_cost": optimal_terminal,
            "optimal_J": optimal_realized,
            "dense_optimal_minus_raw_BVP_plus_terminal": optimal_realized
            - (official_raw_bvp_value + author_bvp_terminal),
            "NN_gap_percent_vs_optimal": 100.0 * (accurate_nn["J"] / optimal_realized - 1.0),
            "LQR_gap_percent_vs_optimal": 100.0 * (accurate_lqr["J"] / optimal_realized - 1.0),
            "integrator": "per-ZOH-interval DOP853, rtol=1e-10, atol=1e-12; BVP objective Simpson on 20001 points",
        },
        "audited_upstream_numerical_quirks": [
            "simulate_noise.py repeats four RK4 increments from the same interval-start state instead of advancing substeps",
            "saved LQR control at t=0 omits W[:,0] although the applied first-interval control includes it",
            "simulate_noise.py reports the BVP value-state before adding terminal cost",
            "problem.compute_cost adds the final running-cost sample in addition to trapezoidal integration",
        ],
    }


def retrain_satellite(
    repo_root: Path,
    output_dir: Path,
    *,
    seed: int,
    max_rounds: int,
    min_rounds: int,
    maxiter: int,
    warm_start_released: bool,
) -> dict[str, Any]:
    """Run the original author satellite t=0 BVP/value+costate/L-BFGS path."""
    import tensorflow.compat.v1 as tf

    config_NN, setup_problem, HJBnet_t0, load_NN = _load_author_api(repo_root)
    problem = setup_problem()
    config = config_NN(problem.N_states, problem.t1, False)
    config.random_seeds = {"train": seed}
    config.batch_size = None
    config.max_rounds = max_rounds
    config.min_rounds = min_rounds
    config.weight_A = np.full(max_rounds, 10.0, dtype=np.float64)
    config.weight_U = np.zeros(max_rounds, dtype=np.float64)
    config.BFGS_opts = {
        "maxiter": [maxiter] * max_rounds,
        # Match the released author default.  maxfun and maxiter are both
        # 100000 in utilities/neural_networks.py for the paper profile; the
        # former 3*maxiter override silently enlarged the official budget.
        "maxfun": [maxiter] * max_rounds,
        "iprint": [0] * max_rounds,
    }
    np.random.seed(seed)
    tf.disable_v2_behavior()
    tf.set_random_seed(seed)

    train_raw, validation_raw, train_initial, validation_initial = load_satellite_data(repo_root)
    train = _t0_data(train_raw, problem)
    validation = _t0_data(validation_raw, problem)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "dataset_train_initial.npz", **train, initial_states=train_initial)
    np.savez_compressed(
        output_dir / "dataset_validation.npz", **validation, initial_states=validation_initial
    )

    checkpoint = _author_root(repo_root) / "examples" / "satellite" / "t0" / "V_model.mat"
    if warm_start_released:
        parameters, scaling = load_NN(str(checkpoint))
    else:
        parameters = None
        scaling = {
            "lb": np.min(train["X"], axis=1, keepdims=True),
            "ub": np.max(train["X"], axis=1, keepdims=True),
            "A_lb": np.min(train["A"], axis=1, keepdims=True),
            "A_ub": np.max(train["A"], axis=1, keepdims=True),
            "U_lb": np.min(train["U"], axis=1, keepdims=True),
            "U_ub": np.max(train["U"], axis=1, keepdims=True),
            "V_min": np.min(train["V"]),
            "V_max": np.max(train["V"]),
        }

    Instrumented = make_instrumented_hjbnet(HJBnet_t0)
    model = Instrumented(
        problem,
        scaling,
        config,
        parameters,
        checkpoint_dir=output_dir / "checkpoints",
    )
    train_for_model = {key: np.array(value, copy=True) for key, value in train.items()}
    validation_for_model = {
        key: np.array(value, copy=True)
        for key, value in validation.items()
        if key in {"t", "X", "A", "V", "U"}
    }
    train_start = time.perf_counter()
    round_iters, errors = model.train(train_for_model, validation_for_model)
    training_wall_time_seconds = time.perf_counter() - train_start
    final_payload = checkpoint_payload(model)
    scipy.io.savemat(output_dir / "V_model.mat", final_payload)
    np.savez_compressed(output_dir / "dataset_train_final.npz", **train_for_model)
    history = {
        "round_iters": [int(item) for item in round_iters],
        "round_iters_semantics": "cumulative loss-callback evaluations, not SciPy iteration counts",
        "round_callback_increment_counts": [
            int(current - previous)
            for previous, current in zip([0, *round_iters[:-1]], round_iters)
        ],
        "train_value_RMAE": [float(item) for item in errors[0]],
        "train_costate_RML2": [float(item) for item in errors[1]],
        "train_control_RML2": [float(item) for item in errors[2]],
        "validation_value_RMAE": [float(item) for item in errors[3]],
        "validation_costate_RML2": [float(item) for item in errors[4]],
        "validation_control_RML2": [float(item) for item in errors[5]],
        "convergence_tests": model.convergence_history,
        "optimizer_rounds": model.optimizer_history,
    }
    (output_dir / "history.json").write_text(json.dumps(history, indent=2, sort_keys=True))
    model.sess.close()
    return {
        "benchmark": "author satellite t=0 retraining through audited TensorFlow 2 compatibility port",
        "upstream_tracked_dirty": True,
        "tensorflow2_compatibility_port": True,
        "seed": seed,
        "full_batch_L_BFGS": True,
        "joint_value_costate_supervision": True,
        "network": _network_metadata(config),
        "trajectory_disjoint_validation": True,
        "adaptive_rounds_enabled": max_rounds > 1,
        "max_rounds": max_rounds,
        "min_rounds": min_rounds,
        "maxiter_per_round": maxiter,
        "maxfun_per_round": maxiter,
        "official_optimizer_budget": maxiter == 100_000,
        "optimizer": {
            "method": "L-BFGS-B",
            "maxiter_per_round": maxiter,
            "maxfun_per_round": maxiter,
            "maxcor": 15,
            "ftol": 1.0e-11,
            "gtol": 1.0e-6,
        },
        "paper_configuration_reconstruction": True,
        "exact_numeric_replay_possible": False,
        "exact_numeric_replay_limitation": "paper and release omit the RNG seed, final adaptive training set, exact initial-set/scaling provenance, and per-round hyperparameter schedule",
        "warm_start_released": warm_start_released,
        "training_wall_time_seconds": training_wall_time_seconds,
        "initial_training_samples": int(train["X"].shape[1]),
        "final_training_samples": int(train_for_model["X"].shape[1]),
        "completed_rounds": len(round_iters),
        "selection_rule": "predeclared author convergence/stopping rule; final completed L-BFGS round exported",
        "all_candidates_retained": True,
        "checkpoints": [str(path.relative_to(output_dir)) for path in model.checkpoint_paths],
        "final_validation": {
            "value_RMAE": history["validation_value_RMAE"][-1],
            "costate_RML2": history["validation_costate_RML2"][-1],
            "control_RML2": history["validation_control_RML2"][-1],
        },
    }
