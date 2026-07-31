#!/usr/bin/env python3
"""CLI for original [7] reproduction and entropy-consistent tumor adaptation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io
from scipy.integrate import solve_ivp

from .bvp import AdaptiveDataController, BVPSettings, generate_initial_dataset
from .official_adapter import checkpoint_payload, make_instrumented_hjbnet
from .problem import TumorEntropyProblem, TumorParameters, evaluate_zoh_cost_pair
from .sampling import (
    git_checkout_provenance,
    refuse_nonempty_output,
    split_initial_state_box,
    validate_trajectory_disjoint,
)
from .satellite import (
    evaluate_satellite_closed_loop,
    retrain_satellite,
    validate_released_checkpoint,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
AUTHOR_ROOT = REPO_ROOT / "external" / "HJB_NN"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def _provenance_files() -> list[Path]:
    return [
        Path(__file__),
        Path(__file__).with_name("problem.py"),
        Path(__file__).with_name("bvp.py"),
        Path(__file__).with_name("sampling.py"),
        Path(__file__).with_name("official_adapter.py"),
        Path(__file__).with_name("satellite.py"),
        REPO_ROOT / "tumor_problem.py",
        AUTHOR_ROOT / "utilities" / "neural_networks.py",
        AUTHOR_ROOT / "utilities" / "optimize.py",
        AUTHOR_ROOT / "utilities" / "other.py",
        AUTHOR_ROOT / "examples" / "problem_def_template.py",
        AUTHOR_ROOT / "examples" / "satellite" / "problem_def.py",
        AUTHOR_ROOT / "examples" / "satellite" / "data_train.mat",
        AUTHOR_ROOT / "examples" / "satellite" / "data_test.mat",
        AUTHOR_ROOT / "examples" / "satellite" / "t0" / "V_model.mat",
    ]


def _snapshot_executed_sources(output_dir: Path) -> dict[str, str]:
    """Retain exact executable sources so hashes remain reconstructable."""
    mapping: dict[str, str] = {}
    for source in _provenance_files():
        if source.suffix == ".mat":
            continue
        relative = source.relative_to(REPO_ROOT)
        destination = output_dir / "executed_source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        mapping[str(relative)] = str(destination.relative_to(output_dir))
    return mapping


def source_provenance() -> dict[str, Any]:
    files = _provenance_files()
    upstream = git_checkout_provenance(AUTHOR_ROOT)
    return {
        "repo_revision": git_revision(REPO_ROOT),
        "upstream_HJB_NN_base_revision": upstream["base_revision"],
        "upstream_tracked_dirty": upstream["tracked_dirty"],
        "upstream_worktree_clean": upstream["worktree_clean"],
        "upstream_git_status_porcelain": upstream["status_porcelain"],
        "upstream_git_diff_name_status": upstream["tracked_diff_name_status"],
        "upstream_git_diff_numstat": upstream["tracked_diff_numstat"],
        "upstream_git_diff_sha256": upstream["tracked_diff_sha256"],
        "upstream_git_diff_bytes": upstream["tracked_diff_bytes"],
        "upstream_untracked_paths": upstream["untracked_paths"],
        "tensorflow2_compatibility_port": True,
        "compatibility_port_scope": {
            "preserved_logic": "author HJBnet architecture, value/costate losses, L-BFGS-B objective and Algorithm 4.1 mechanics",
            "non_verbatim_reason": "TensorFlow 2 has no tensorflow.contrib; checkout uses tensorflow.compat.v1 and an audited local ScipyOptimizerInterface replacement, plus a backward-compatible state-aware make_U_NN call for tumor controls",
        },
        "file_sha256": {str(path.relative_to(REPO_ROOT)): sha256(path) for path in files},
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "dependency_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "tensorflow")
        },
        "runtime_environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "TF_ENABLE_ONEDNN_OPTS",
                "TF_FORCE_GPU_ALLOW_GROWTH",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        "command": sys.argv,
    }


def _load_official_hjbnet():
    if str(AUTHOR_ROOT) not in sys.path:
        sys.path.insert(0, str(AUTHOR_ROOT))
    from utilities.neural_networks import HJBnet

    return HJBnet


@dataclass
class TumorTrainingConfig:
    width: int
    hidden_layers: int
    max_rounds: int
    min_rounds: int
    maxiter: int
    convergence_tolerance: float
    candidates_per_selection: int
    max_batch_points: int

    @property
    def maxfun(self) -> int:
        """Function-evaluation ceiling used by the tumor L-BFGS-B adaptation."""
        return max(1000, 3 * self.maxiter)

    def author_config(self, problem: TumorEntropyProblem):
        config = type("FaithfulTumorHJBConfig", (), {})()
        config.layers = [problem.N_states + 1] + [self.width] * self.hidden_layers + [1]
        config.random_seeds = {"train": None}
        config.ODE_solver = "DOP853"
        config.data_tol = 1.0e-3
        config.max_nodes = 20_000
        config.dt = 0.02
        config.sigma = 0.0
        config.batch_size = None
        config.Ns_scale = 2
        config.Ns_cand = self.candidates_per_selection
        config.Ns_max = self.max_batch_points
        config.conv_tol = self.convergence_tolerance
        config.max_rounds = self.max_rounds
        config.min_rounds = self.min_rounds
        config.weight_A = np.full(self.max_rounds, 10.0, dtype=np.float64)
        config.weight_U = np.zeros(self.max_rounds, dtype=np.float64)
        config.BFGS_opts = {
            "maxiter": [self.maxiter] * self.max_rounds,
            "maxfun": [self.maxfun] * self.max_rounds,
            "iprint": [0] * self.max_rounds,
        }
        return config


def _tumor_network_metadata(
    training: TumorTrainingConfig, state_dim: int = 21
) -> dict[str, Any]:
    layers = [state_dim + 1] + [training.width] * training.hidden_layers + [1]
    parameters = sum(
        layers[index + 1] * layers[index] + layers[index + 1]
        for index in range(len(layers) - 1)
    )
    return {
        "layers": layers,
        "trainable_parameters": int(parameters),
        "input": "time plus 21 tumor-state coordinates",
        "hidden_activation": "tanh",
        "output_activation": "linear",
        "predicted_quantity": "entropy-regularized value V(t,N)",
    }


def _scaling(train: dict[str, np.ndarray]) -> dict[str, Any]:
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
    for lower_key, upper_key in (("lb", "ub"), ("A_lb", "A_ub"), ("U_lb", "U_ub")):
        if np.any(np.asarray(scaling[upper_key]) <= np.asarray(scaling[lower_key])):
            raise RuntimeError(f"degenerate author scaling range: {lower_key}/{upper_key}")
    if float(scaling["V_max"]) <= float(scaling["V_min"]):
        raise RuntimeError("degenerate value scaling range")
    return scaling


def _copy_data(data: dict[str, np.ndarray], *, include_metadata: bool = True):
    keys = {"t", "X", "A", "V", "U"}
    if include_metadata:
        keys.add("trajectory_id")
    return {key: np.array(value, copy=True) for key, value in data.items() if key in keys}


def _write_history_csv(path: Path, history: dict[str, list[float]]) -> None:
    metric_keys = [
        key
        for key, value in history.items()
        if isinstance(value, list) and value and all(np.isscalar(item) for item in value)
    ]
    rows = max((len(history[key]) for key in metric_keys), default=0)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["index", *metric_keys])
        writer.writeheader()
        for index in range(rows):
            row = {"index": index}
            for key in metric_keys:
                if index < len(history[key]) and np.isscalar(history[key][index]):
                    row[key] = history[key][index]
            writer.writerow(row)


def _trajectory_equal_validation_metrics(model, dataset) -> dict[str, Any]:
    """Report held-out errors with each characteristic receiving equal weight."""
    value_prediction = model.predict_V(dataset.t, dataset.X)
    costate_prediction = model.predict_A(dataset.t, dataset.X)
    control_prediction = model.eval_U(dataset.t, dataset.X)
    per_trajectory = []
    for trajectory_id in np.unique(dataset.trajectory_id):
        mask = dataset.trajectory_id.reshape(-1) == trajectory_id
        value_error = float(
            np.mean(np.abs(value_prediction[:, mask] - dataset.V[:, mask]))
            / np.mean(np.abs(dataset.V[:, mask]))
        )
        costate_error = float(
            np.mean(
                np.linalg.norm(
                    costate_prediction[:, mask] - dataset.A[:, mask], axis=0
                )
            )
            / np.mean(np.linalg.norm(dataset.A[:, mask], axis=0))
        )
        control_error = float(
            np.mean(
                np.linalg.norm(
                    control_prediction[:, mask] - dataset.U[:, mask], axis=0
                )
            )
            / np.mean(np.linalg.norm(dataset.U[:, mask], axis=0))
        )
        per_trajectory.append((value_error, costate_error, control_error))
    values = np.asarray(per_trajectory, dtype=np.float64)
    return {
        "value_RMAE": float(np.mean(values[:, 0])),
        "costate_RML2": float(np.mean(values[:, 1])),
        "control_RML2": float(np.mean(values[:, 2])),
        "trajectories": int(values.shape[0]),
        "weighting": "mean of per-trajectory relative metrics; each trajectory has equal weight",
    }


def _canonical_control(
    problem: TumorEntropyProblem,
    model,
    intervals: int,
    output_dir: Path,
) -> dict[str, float]:
    from tumor_problem import TumorProblem, evaluate_zoh_control, serializable_metrics

    time = np.linspace(0.0, problem.t1, intervals + 1)
    states = np.empty((problem.N_states, intervals + 1), dtype=np.float64)
    control = np.empty(intervals, dtype=np.float64)
    states[:, 0] = problem.nominal_initial_state
    # Match the paper's sampled-feedback execution and the common evaluator's
    # left-endpoint ZOH contract. Query TensorFlow once per breakpoint, then
    # integrate that interval with a constant feasible action.
    for index in range(intervals):
        control[index] = float(
            model.eval_U(
                np.array([[time[index]]], dtype=np.float64),
                states[:, index : index + 1],
            )[0, 0]
        )
        if control[index] < 0.0 or control[index] > problem.umax:
            raise RuntimeError("learned feedback returned an infeasible action")

        def constant_control_rhs(_time, state):
            crowding = np.log1p(np.mean(state))
            return (
                problem.r
                - problem.phi * control[index]
                - problem.M * crowding
            ) * state

        interval_solution = solve_ivp(
            constant_control_rhs,
            (float(time[index]), float(time[index + 1])),
            states[:, index],
            method="DOP853",
            rtol=1.0e-9,
            atol=1.0e-11,
        )
        if not interval_solution.success or np.any(interval_solution.y[:, -1] <= 0.0):
            raise RuntimeError(f"ZOH feedback rollout failed: {interval_solution.message}")
        states[:, index + 1] = interval_solution.y[:, -1]
    realized = evaluate_zoh_control(
        time,
        control,
        TumorProblem(
            T=problem.t1,
            m=problem.N_states,
            umax=problem.umax,
            beta=problem.params.beta,
            alpha=problem.params.alpha,
            gamma=problem.params.gamma,
            n0=problem.params.initial_state,
            m_suppression=problem.params.suppression,
        ),
        include_diagnostics=False,
    )
    cost_pair = evaluate_zoh_cost_pair(problem, time, control)
    if not np.isclose(
        cost_pair["unregularized_J"], realized["J"], rtol=0.0, atol=2.0e-8
    ):
        raise RuntimeError(
            "independent unregularized ZOH cost does not match common evaluator: "
            f"{cost_pair['unregularized_J']} vs {realized['J']}"
        )
    native_value = float(
        model.predict_V(
            np.zeros((1, 1), dtype=np.float64),
            problem.nominal_initial_state.reshape((-1, 1)),
        )[0, 0]
    )
    np.savez_compressed(
        output_dir / "canonical_control.npz",
        t=time,
        u=control,
        feedback_rollout_N=states,
        tau=np.array(problem.tau),
        J_regularized_native_value_prediction=np.array(native_value),
        J_regularized_realized=np.array(cost_pair["regularized_J"]),
        regularized_terminal_cost=np.array(cost_pair["terminal_cost"]),
        regularized_running_cost=np.array(cost_pair["regularized_running_cost"]),
        regularized_entropy_integral=np.array(cost_pair["entropy_integral"]),
        J_unregularized_realized=np.array(realized["J"]),
        control_semantics=np.array("left-endpoint ZOH export of learned feedback"),
    )
    return {
        "regularized_native_value_prediction": native_value,
        "regularized_realized_J": cost_pair["regularized_J"],
        "regularized_terminal_cost": cost_pair["terminal_cost"],
        "regularized_running_cost": cost_pair["regularized_running_cost"],
        "regularized_entropy_integral": cost_pair["entropy_integral"],
        "regularized_native_prediction_error": native_value - cost_pair["regularized_J"],
        "regularized_native_prediction_abs_error": abs(
            native_value - cost_pair["regularized_J"]
        ),
        "regularized_native_prediction_relative_error": abs(
            native_value - cost_pair["regularized_J"]
        )
        / max(abs(cost_pair["regularized_J"]), 1.0e-12),
        "unregularized_realized_J": float(realized["J"]),
        **{
            f"unregularized_{key}": float(value)
            for key, value in serializable_metrics(realized).items()
            if np.isscalar(value) and key != "J"
        },
    }


def run_one_tumor(
    output_dir: Path,
    *,
    tau: float,
    seed: int,
    train_trajectories: int,
    validation_trajectories: int,
    bvp_settings: BVPSettings,
    training_config: TumorTrainingConfig,
    evaluation_intervals: int,
    adaptive_max_failures: int,
    problem_parameters: TumorParameters | None = None,
) -> dict[str, Any]:
    import tensorflow.compat.v1 as tf

    run_start = time.perf_counter()
    refuse_nonempty_output(output_dir)
    executed_source_snapshot = _snapshot_executed_sources(output_dir)
    if adaptive_max_failures <= 0:
        raise ValueError("adaptive_max_failures must be positive")
    np.random.seed(seed)
    tf.disable_v2_behavior()
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    problem = TumorEntropyProblem(tau, parameters=problem_parameters)
    train_initial, validation_initial = split_initial_state_box(
        problem.X0_lb,
        problem.X0_ub,
        train_trajectories,
        validation_trajectories,
        seed=seed,
    )
    validate_trajectory_disjoint(train_initial, validation_initial)
    train_bvp_start = time.perf_counter()
    train_dataset, train_bvp = generate_initial_dataset(
        problem, train_initial, bvp_settings, first_trajectory_id=0
    )
    train_bvp_seconds = time.perf_counter() - train_bvp_start
    validation_bvp_start = time.perf_counter()
    validation_dataset, validation_bvp = generate_initial_dataset(
        problem,
        validation_initial,
        bvp_settings,
        first_trajectory_id=1_000_000,
    )
    validation_bvp_seconds = time.perf_counter() - validation_bvp_start
    train_dataset.save(output_dir / "dataset_train_initial.npz")
    validation_dataset.save(output_dir / "dataset_validation.npz")
    json_dump(
        output_dir / "bvp_initial_history.json",
        {
            "train": [asdict(item) for item in train_bvp],
            "validation": [asdict(item) for item in validation_bvp],
        },
    )

    train = train_dataset.as_author_dict()
    validation = validation_dataset.as_author_dict()
    scaling = _scaling(train)
    author_config = training_config.author_config(problem)
    adaptive_controller = AdaptiveDataController(
        problem,
        bvp_settings,
        np.random.default_rng(np.random.SeedSequence([seed, 2])),
        output_dir / "adaptive_data",
        first_trajectory_id=train_trajectories,
        max_failures=adaptive_max_failures,
    )
    OfficialHJBnet = _load_official_hjbnet()
    InstrumentedHJBnet = make_instrumented_hjbnet(OfficialHJBnet)
    model = InstrumentedHJBnet(
        problem,
        scaling,
        author_config,
        parameters=None,
        checkpoint_dir=output_dir / "checkpoints",
        adaptive_controller=adaptive_controller,
    )
    train_for_model = _copy_data(train, include_metadata=True)
    validation_for_model = _copy_data(validation, include_metadata=False)
    training_start = time.perf_counter()
    round_iters, errors = model.train(train_for_model, validation_for_model)
    training_seconds = time.perf_counter() - training_start
    trajectory_equal_validation = _trajectory_equal_validation_metrics(
        model, validation_dataset
    )
    scipy.io.savemat(output_dir / "V_model.mat", checkpoint_payload(model))
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
        "adaptive_events": adaptive_controller.events,
    }
    json_dump(output_dir / "history.json", history)
    _write_history_csv(output_dir / "history.csv", history)
    evaluation_start = time.perf_counter()
    metrics = _canonical_control(problem, model, evaluation_intervals, output_dir)
    evaluation_seconds = time.perf_counter() - evaluation_start
    model.sess.close()

    provenance = source_provenance()
    for original, snapshot in executed_source_snapshot.items():
        if sha256(output_dir / snapshot) != provenance["file_sha256"][original]:
            raise RuntimeError(f"executed source changed during run: {original}")
    manifest = {
        "experiment_label": "entropy-regularized tumor adaptation of [7], not original benchmark",
        "seed": seed,
        "tau": tau,
        "problem_parameters": asdict(problem.params),
        "regularized_training_objective": True,
        "unregularized_common_objective_used_for_training_or_selection": False,
        "control_minimizer": "u=U*sigmoid(-psi/tau), exact for declared binary-entropy regularizer",
        "singular_heuristic_blend": False,
        "initial_state_sampling": "independent uniform coordinates over [5,20]^21",
        "bvp_newton_domain_extension": "N_i=max(N_i,1e-12) only for residual evaluation; accepted trajectories require N_i>0",
        "trajectory_disjoint_validation": True,
        "train_trajectories_initial": train_trajectories,
        "validation_trajectories": validation_trajectories,
        "initial_training_points": int(train_dataset.X.shape[1]),
        "validation_points": int(validation_dataset.X.shape[1]),
        "final_training_points": int(train_for_model["X"].shape[1]),
        "final_training_trajectories": int(
            np.unique(train_for_model["trajectory_id"]).size
        ),
        "completed_rounds": len(round_iters),
        "training": asdict(training_config),
        "network": _tumor_network_metadata(training_config, problem.N_states),
        "optimizer_budget": {
            "method": "L-BFGS-B",
            "maxiter_per_round": training_config.maxiter,
            "maxfun_per_round": training_config.maxfun,
            "round_ceiling": training_config.max_rounds,
            "maxcor": 15,
            "ftol": 1.0e-11,
            "gtol": 1.0e-6,
            "note": "tumor-adaptation budget; not the original satellite paper budget",
        },
        "bvp": asdict(bvp_settings),
        "author_compatibility_port_components": {
            "network_and_joint_loss": "author HJBnet logic through tracked TensorFlow-2 compatibility diff in external/HJB_NN/utilities/neural_networks.py",
            "optimizer": "author L-BFGS-B objective through tracked replacement for removed tensorflow.contrib interface in external/HJB_NN/utilities/optimize.py",
            "adaptive_convergence_test": "external/HJB_NN/utilities/neural_networks.py::HJBnet.convergence_test",
        },
        "adaptive_events": len(adaptive_controller.events),
        "adaptive_successes": sum(
            bool(event["success"]) for event in adaptive_controller.events
        ),
        "adaptive_failures": sum(
            not bool(event["success"]) for event in adaptive_controller.events
        ),
        "adaptive_max_failures": adaptive_max_failures,
        "evaluation_intervals": evaluation_intervals,
        "wall_time_seconds": time.perf_counter() - run_start,
        "phase_wall_time_seconds": {
            "initial_train_bvp": train_bvp_seconds,
            "initial_validation_bvp": validation_bvp_seconds,
            "network_training_and_adaptation": training_seconds,
            "canonical_evaluation": evaluation_seconds,
        },
        "selection_rule": "predeclared author convergence/stopping rule; final completed round exported; unregularized realized J unavailable to training and selection",
        "all_candidates_retained": True,
        "upstream_tracked_dirty": provenance["upstream_tracked_dirty"],
        "tensorflow2_compatibility_port": provenance["tensorflow2_compatibility_port"],
        "checkpoints": [str(path.relative_to(output_dir)) for path in model.checkpoint_paths],
        "final_validation": {
            "value_RMAE": history["validation_value_RMAE"][-1],
            "costate_RML2": history["validation_costate_RML2"][-1],
            "control_RML2": history["validation_control_RML2"][-1],
        },
        "final_validation_trajectory_equal": trajectory_equal_validation,
        "metrics": metrics,
        "artifact_sha256": {
            str(path.relative_to(output_dir)): sha256(path)
            for path in output_dir.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        },
        "provenance": provenance,
        "executed_source_snapshot": executed_source_snapshot,
    }
    json_dump(output_dir / "manifest.json", manifest)
    return manifest


def run_tumor_sweep(args) -> dict[str, Any]:
    output_dir = Path(args.out_dir).resolve()
    refuse_nonempty_output(output_dir)
    taus = [float(item) for item in args.taus.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    if not taus or not seeds:
        raise ValueError("at least one tau and one seed are required")
    problem_parameters = TumorParameters(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
    )
    if args.smoke:
        train_trajectories = min(args.train_trajectories, 1)
        validation_trajectories = min(args.validation_trajectories, 1)
        training = TumorTrainingConfig(8, 1, 1, 1, 1, 0.5, 2, 256)
        bvp = BVPSettings(
            tolerance=5.0e-2,
            max_nodes=2_000,
            adaptive_max_nodes=2_000,
            time_march_steps=2,
            initial_mesh_nodes=15,
            tau_start=max(taus),
            ode_rtol=1.0e-4,
            ode_atol=1.0e-6,
        )
        evaluation_intervals = min(args.evaluation_intervals, 20)
    else:
        train_trajectories = args.train_trajectories
        validation_trajectories = args.validation_trajectories
        training = TumorTrainingConfig(
            args.width,
            args.hidden_layers,
            args.max_rounds,
            args.min_rounds,
            args.maxiter,
            args.convergence_tolerance,
            args.candidates,
            args.max_batch_points,
        )
        bvp = BVPSettings(
            tolerance=args.bvp_tolerance,
            max_nodes=args.bvp_max_nodes,
            adaptive_max_nodes=args.adaptive_bvp_max_nodes,
            time_march_steps=args.time_march_steps,
            initial_mesh_nodes=args.initial_mesh_nodes,
            tau_start=max(taus),
        )
        evaluation_intervals = args.evaluation_intervals

    runs = []
    for tau in taus:
        for seed in seeds:
            name = f"tau_{tau:g}".replace(".", "p")
            run_dir = output_dir / name / f"seed_{seed}"
            manifest = run_one_tumor(
                run_dir,
                tau=tau,
                seed=seed,
                train_trajectories=train_trajectories,
                validation_trajectories=validation_trajectories,
                bvp_settings=bvp,
                training_config=training,
                evaluation_intervals=evaluation_intervals,
                adaptive_max_failures=args.adaptive_max_failures,
                problem_parameters=problem_parameters,
            )
            runs.append(
                {
                    "tau": tau,
                    "seed": seed,
                    "run_dir": str(run_dir.relative_to(output_dir)),
                    **manifest["metrics"],
                }
            )
    provenance = source_provenance()
    summary = {
        "experiment_label": "tau continuation/sweep for entropy-regularized tumor adaptation",
        "taus": taus,
        "seeds": seeds,
        "selection_rule": "all predeclared tau/seed runs retained; no realized-J selection",
        "upstream_tracked_dirty": provenance["upstream_tracked_dirty"],
        "tensorflow2_compatibility_port": provenance["tensorflow2_compatibility_port"],
        "runs": runs,
        "provenance": provenance,
    }
    json_dump(output_dir / "manifest.json", summary)
    with (output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(runs[0].keys()))
        writer.writeheader()
        writer.writerows(runs)
    return summary


def run_single_tumor(args) -> dict[str, Any]:
    """Run one isolated tau/seed worker with a retained failure manifest."""
    output_dir = Path(args.out_dir).resolve()
    refuse_nonempty_output(output_dir)
    training = TumorTrainingConfig(
        args.width,
        args.hidden_layers,
        args.max_rounds,
        args.min_rounds,
        args.maxiter,
        args.convergence_tolerance,
        args.candidates,
        args.max_batch_points,
    )
    bvp = BVPSettings(
        tolerance=args.bvp_tolerance,
        max_nodes=args.bvp_max_nodes,
        adaptive_max_nodes=args.adaptive_bvp_max_nodes,
        time_march_steps=args.time_march_steps,
        initial_mesh_nodes=args.initial_mesh_nodes,
        tau_start=args.tau_start,
    )
    run_start = time.perf_counter()
    problem_parameters = TumorParameters(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
    )
    try:
        return run_one_tumor(
            output_dir,
            tau=args.tau,
            seed=args.seed,
            train_trajectories=args.train_trajectories,
            validation_trajectories=args.validation_trajectories,
            bvp_settings=bvp,
            training_config=training,
            evaluation_intervals=args.evaluation_intervals,
            adaptive_max_failures=args.adaptive_max_failures,
            problem_parameters=problem_parameters,
        )
    except Exception as error:
        output_dir.mkdir(parents=True, exist_ok=True)
        partial_artifacts = {
            str(path.relative_to(output_dir)): sha256(path)
            for path in output_dir.rglob("*")
            if path.is_file() and path.name != "failure_manifest.json"
        }
        failure = {
            "experiment_label": "failed isolated entropy-regularized tumor adaptation of [7]",
            "status": "failed",
            "tau": args.tau,
            "seed": args.seed,
            "problem_parameters": asdict(problem_parameters),
            "training": asdict(training),
            "network": _tumor_network_metadata(training),
            "optimizer_budget": {
                "method": "L-BFGS-B",
                "maxiter_per_round": training.maxiter,
                "maxfun_per_round": training.maxfun,
                "round_ceiling": training.max_rounds,
                "maxcor": 15,
                "ftol": 1.0e-11,
                "gtol": 1.0e-6,
                "note": "tumor-adaptation budget; not the original satellite paper budget",
            },
            "bvp": asdict(bvp),
            "adaptive_max_failures": args.adaptive_max_failures,
            "wall_time_seconds_before_failure": time.perf_counter() - run_start,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "partial_artifact_sha256": partial_artifacts,
            "provenance": source_provenance(),
        }
        json_dump(output_dir / "failure_manifest.json", failure)
        raise


def aggregate_tumor_workers(args) -> dict[str, Any]:
    """Aggregate a complete predeclared grid of isolated successful workers."""
    output_dir = Path(args.out_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    root_manifest = output_dir / "manifest.json"
    if root_manifest.exists():
        raise FileExistsError(f"refusing to overwrite existing aggregate manifest: {root_manifest}")
    taus = [float(item) for item in args.taus.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    if not taus or not seeds:
        raise ValueError("at least one tau and one seed are required")
    runs: list[dict[str, Any]] = []
    shared_training = None
    shared_bvp = None
    shared_optimizer_budget = None
    shared_evaluation_intervals = None
    shared_adaptive_max_failures = None
    shared_source_signature = None
    for tau in taus:
        tau_name = f"tau_{tau:g}".replace(".", "p")
        for seed in seeds:
            run_dir = output_dir / tau_name / f"seed_{seed}"
            failure_path = run_dir / "failure_manifest.json"
            manifest_path = run_dir / "manifest.json"
            if failure_path.exists():
                raise RuntimeError(f"declared worker failed: {failure_path}")
            if not manifest_path.is_file():
                raise FileNotFoundError(f"declared worker manifest missing: {manifest_path}")
            manifest = json.loads(manifest_path.read_text())
            if float(manifest["tau"]) != tau or int(manifest["seed"]) != seed:
                raise RuntimeError(f"worker identity mismatch: {manifest_path}")
            for label, value in (
                ("training", manifest.get("training")),
                ("bvp", manifest.get("bvp")),
                ("optimizer_budget", manifest.get("optimizer_budget")),
            ):
                if not isinstance(value, dict):
                    raise RuntimeError(f"worker {label} declaration missing: {manifest_path}")
                current = {
                    "training": shared_training,
                    "bvp": shared_bvp,
                    "optimizer_budget": shared_optimizer_budget,
                }[label]
                if current is not None and value != current:
                    raise RuntimeError(
                        f"worker {label} budget mismatch: {manifest_path}"
                    )
                if label == "training":
                    shared_training = value
                elif label == "bvp":
                    shared_bvp = value
                else:
                    shared_optimizer_budget = value
            for label, value in (
                ("evaluation_intervals", manifest.get("evaluation_intervals")),
                ("adaptive_max_failures", manifest.get("adaptive_max_failures")),
            ):
                if not isinstance(value, int) or value <= 0:
                    raise RuntimeError(f"worker {label} declaration invalid: {manifest_path}")
                current = (
                    shared_evaluation_intervals
                    if label == "evaluation_intervals"
                    else shared_adaptive_max_failures
                )
                if current is not None and value != current:
                    raise RuntimeError(f"worker {label} mismatch: {manifest_path}")
                if label == "evaluation_intervals":
                    shared_evaluation_intervals = value
                else:
                    shared_adaptive_max_failures = value
            worker_provenance = manifest.get("provenance")
            if not isinstance(worker_provenance, dict):
                raise RuntimeError(f"worker provenance missing: {manifest_path}")
            source_signature = {
                key: worker_provenance.get(key)
                for key in (
                    "repo_revision",
                    "upstream_HJB_NN_base_revision",
                    "upstream_git_diff_sha256",
                    "file_sha256",
                    "python",
                    "platform",
                    "dependency_versions",
                )
            }
            if any(value is None for value in source_signature.values()):
                raise RuntimeError(f"worker source signature incomplete: {manifest_path}")
            if shared_source_signature is not None and source_signature != shared_source_signature:
                raise RuntimeError(f"worker source/dependency mismatch: {manifest_path}")
            shared_source_signature = source_signature
            for relative, expected in manifest["artifact_sha256"].items():
                artifact = run_dir / relative
                if not artifact.is_file() or sha256(artifact) != expected:
                    raise RuntimeError(f"worker artifact hash mismatch: {artifact}")
            actual_artifacts = {
                str(path.relative_to(run_dir))
                for path in run_dir.rglob("*")
                if path.is_file() and path.name != "manifest.json"
            }
            declared_artifacts = set(manifest["artifact_sha256"])
            if actual_artifacts != declared_artifacts:
                raise RuntimeError(
                    f"worker artifact set differs from manifest: {manifest_path}"
                )
            runs.append(
                {
                    "tau": tau,
                    "seed": seed,
                    "run_dir": str(run_dir.relative_to(output_dir)),
                    "worker_manifest_sha256": sha256(manifest_path),
                    **manifest["metrics"],
                }
            )
    provenance = source_provenance()
    aggregate_source_signature = {
        key: provenance.get(key)
        for key in (
            "repo_revision",
            "upstream_HJB_NN_base_revision",
            "upstream_git_diff_sha256",
            "file_sha256",
            "python",
            "platform",
            "dependency_versions",
        )
    }
    if aggregate_source_signature != shared_source_signature:
        raise RuntimeError(
            "aggregate source/dependency signature differs from completed workers"
        )
    source_signature_sha256 = hashlib.sha256(
        json.dumps(shared_source_signature, sort_keys=True).encode("utf-8")
    ).hexdigest()
    summary = {
        "experiment_label": "tau continuation/sweep for entropy-regularized tumor adaptation",
        "status": "completed",
        "execution_layout": "isolated tau/seed workers aggregated only after every declared worker passed artifact verification",
        "taus": taus,
        "seeds": seeds,
        "training": shared_training,
        "bvp": shared_bvp,
        "optimizer_budget": shared_optimizer_budget,
        "evaluation_intervals": shared_evaluation_intervals,
        "adaptive_max_failures": shared_adaptive_max_failures,
        "worker_source_signature_sha256": source_signature_sha256,
        "worker_source_and_dependencies_uniform": True,
        "selection_rule": "all predeclared tau/seed runs retained; no realized-J selection",
        "upstream_tracked_dirty": provenance["upstream_tracked_dirty"],
        "tensorflow2_compatibility_port": provenance["tensorflow2_compatibility_port"],
        "runs": runs,
        "provenance": provenance,
    }
    json_dump(root_manifest, summary)
    with (output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(runs[0].keys()))
        writer.writeheader()
        writer.writerows(runs)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-satellite")
    validate.add_argument("--out-dir", default="paper_runs/faithful_hjb_nn_satellite_validation")

    closed_loop = subparsers.add_parser("evaluate-satellite-closed-loop")
    closed_loop.add_argument(
        "--out-dir", default="paper_runs/faithful_hjb_nn_satellite_closed_loop"
    )
    closed_loop.add_argument("--seed", type=int, default=0)

    satellite = subparsers.add_parser("retrain-satellite")
    satellite.add_argument("--out-dir", default="paper_runs/faithful_hjb_nn_satellite_retrain")
    satellite.add_argument("--seed", type=int, default=0)
    satellite.add_argument("--max-rounds", type=int, default=10)
    satellite.add_argument("--min-rounds", type=int, default=1)
    satellite.add_argument("--maxiter", type=int, default=100000)
    satellite.add_argument("--warm-start-released", action="store_true")

    tumor = subparsers.add_parser("run-tumor")
    tumor.add_argument("--out-dir", default="paper_runs/faithful_hjb_nn_tumor")
    tumor.add_argument("--taus", default="10,5,2")
    tumor.add_argument("--seeds", default="0,1,2")
    tumor.add_argument("--train-trajectories", type=int, default=16)
    tumor.add_argument("--validation-trajectories", type=int, default=16)
    tumor.add_argument("--width", type=int, default=96)
    tumor.add_argument("--hidden-layers", type=int, default=3)
    tumor.add_argument("--max-rounds", type=int, default=3)
    tumor.add_argument("--min-rounds", type=int, default=2)
    tumor.add_argument("--maxiter", type=int, default=5000)
    tumor.add_argument("--convergence-tolerance", type=float, default=0.5)
    tumor.add_argument("--candidates", type=int, default=32)
    tumor.add_argument("--max-batch-points", type=int, default=8192)
    tumor.add_argument("--bvp-tolerance", type=float, default=1.0e-3)
    tumor.add_argument("--bvp-max-nodes", type=int, default=100000)
    tumor.add_argument("--adaptive-bvp-max-nodes", type=int, default=20000)
    tumor.add_argument("--time-march-steps", type=int, default=16)
    tumor.add_argument("--initial-mesh-nodes", type=int, default=41)
    tumor.add_argument("--evaluation-intervals", type=int, default=200)
    tumor.add_argument("--adaptive-max-failures", type=int, default=256)
    tumor.add_argument("--alpha", type=float, default=1.0)
    tumor.add_argument("--beta", type=float, default=0.1)
    tumor.add_argument("--gamma", type=float, default=20.0)
    tumor.add_argument("--smoke", action="store_true")

    single = subparsers.add_parser("run-tumor-one")
    single.add_argument("--out-dir", required=True)
    single.add_argument("--tau", type=float, required=True)
    single.add_argument("--seed", type=int, required=True)
    single.add_argument("--tau-start", type=float, default=10.0)
    single.add_argument("--train-trajectories", type=int, default=16)
    single.add_argument("--validation-trajectories", type=int, default=16)
    single.add_argument("--width", type=int, default=96)
    single.add_argument("--hidden-layers", type=int, default=3)
    single.add_argument("--max-rounds", type=int, default=3)
    single.add_argument("--min-rounds", type=int, default=2)
    single.add_argument("--maxiter", type=int, default=5000)
    single.add_argument("--convergence-tolerance", type=float, default=0.5)
    single.add_argument("--candidates", type=int, default=32)
    single.add_argument("--max-batch-points", type=int, default=8192)
    single.add_argument("--bvp-tolerance", type=float, default=1.0e-3)
    single.add_argument("--bvp-max-nodes", type=int, default=100000)
    single.add_argument("--adaptive-bvp-max-nodes", type=int, default=20000)
    single.add_argument("--time-march-steps", type=int, default=16)
    single.add_argument("--initial-mesh-nodes", type=int, default=41)
    single.add_argument("--evaluation-intervals", type=int, default=200)
    single.add_argument("--adaptive-max-failures", type=int, default=256)
    single.add_argument("--alpha", type=float, default=1.0)
    single.add_argument("--beta", type=float, default=0.1)
    single.add_argument("--gamma", type=float, default=20.0)

    aggregate = subparsers.add_parser("aggregate-tumor")
    aggregate.add_argument("--out-dir", required=True)
    aggregate.add_argument("--taus", default="10,5,2")
    aggregate.add_argument("--seeds", default="0,1,2")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-satellite":
        output_dir = Path(args.out_dir).resolve()
        refuse_nonempty_output(output_dir)
        result = validate_released_checkpoint(REPO_ROOT)
        result["provenance"] = source_provenance()
        result["upstream_tracked_dirty"] = result["provenance"]["upstream_tracked_dirty"]
        result["tensorflow2_compatibility_port"] = result["provenance"]["tensorflow2_compatibility_port"]
        checkpoint = REPO_ROOT / result["checkpoint"]
        result["checkpoint_sha256"] = sha256(checkpoint)
        json_dump(output_dir / "manifest.json", result)
    elif args.command == "evaluate-satellite-closed-loop":
        output_dir = Path(args.out_dir).resolve()
        refuse_nonempty_output(output_dir)
        result = evaluate_satellite_closed_loop(REPO_ROOT, output_dir, seed=args.seed)
        result["provenance"] = source_provenance()
        result["upstream_tracked_dirty"] = result["provenance"]["upstream_tracked_dirty"]
        result["tensorflow2_compatibility_port"] = result["provenance"][
            "tensorflow2_compatibility_port"
        ]
        result["artifact_sha256"] = {
            str(path.relative_to(output_dir)): sha256(path)
            for path in output_dir.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        }
        json_dump(output_dir / "manifest.json", result)
    elif args.command == "retrain-satellite":
        output_dir = Path(args.out_dir).resolve()
        refuse_nonempty_output(output_dir)
        result = retrain_satellite(
            REPO_ROOT,
            output_dir,
            seed=args.seed,
            max_rounds=args.max_rounds,
            min_rounds=args.min_rounds,
            maxiter=args.maxiter,
            warm_start_released=args.warm_start_released,
        )
        result["provenance"] = source_provenance()
        result["upstream_tracked_dirty"] = result["provenance"]["upstream_tracked_dirty"]
        result["tensorflow2_compatibility_port"] = result["provenance"]["tensorflow2_compatibility_port"]
        result["artifact_sha256"] = {
            str(path.relative_to(output_dir)): sha256(path)
            for path in output_dir.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        }
        json_dump(output_dir / "manifest.json", result)
    elif args.command == "run-tumor":
        result = run_tumor_sweep(args)
    elif args.command == "run-tumor-one":
        result = run_single_tumor(args)
    else:
        result = aggregate_tumor_workers(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
