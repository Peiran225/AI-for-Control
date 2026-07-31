#!/usr/bin/env python3
"""Independently audit the final multi-seed related-work tumor artifacts.

This auditor deliberately does not import ``tumor_problem`` or any method
runner.  It validates the retained artifact hashes and selection declarations,
then executes every reported left-ZOH control with a separate segmented DOP853
implementation of the declared tumor dynamics and objective.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
from scipy.integrate import solve_ivp
from scipy.stats import spearmanr
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = ROOT / "paper_runs" / "faithful_related_work"
DEFAULT_REFERENCE = ROOT / "paper_runs" / "canonical_results" / "manifest.json"
DEFAULT_OUTPUT = DEFAULT_RESULTS_ROOT / "related_work_tumor_independent_audit"

EXPECTED_SEEDS = (0, 1, 2)
EXPECTED_DEEPBSDE_SIGMAS = (0.1, 0.05, 0.025)
EXPECTED_NEURAL_PMP_STARTS = ("zero", "mid", "front", "back", "random")


@dataclass(frozen=True)
class Evaluation:
    J: float
    terminal_cost: float
    running_cost: float
    breakpoint_states: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def strict_int(value: object, label: str) -> int:
    require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{label} must be a JSON integer",
    )
    return int(value)


def verify_hash(path: Path, expected: object, label: str) -> str:
    require(isinstance(expected, str) and len(expected) == 64, f"invalid {label} hash")
    actual = sha256_file(path)
    require(actual == expected, f"{label} SHA-256 mismatch: {path}")
    return actual


def verify_npz_grid(
    path: Path, expected_intervals: int
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    require("t" in arrays and "u" in arrays, f"control NPZ lacks t/u: {path}")
    t = np.asarray(arrays["t"], dtype=np.float64).reshape(-1)
    u = np.asarray(arrays["u"], dtype=np.float64).reshape(-1)
    require(t.size == expected_intervals + 1, f"unexpected t grid: {path}")
    require(u.size == expected_intervals, f"unexpected u grid: {path}")
    require(np.all(np.isfinite(t)) and np.all(np.isfinite(u)), f"non-finite control: {path}")
    require(abs(float(t[0])) <= 1.0e-14 and abs(float(t[-1]) - 10.0) <= 1.0e-12, f"wrong horizon: {path}")
    require(np.all(np.diff(t) > 0.0), f"control breakpoints are not increasing: {path}")
    expected_t = np.linspace(0.0, 10.0, expected_intervals + 1, dtype=np.float64)
    require(
        np.allclose(t, expected_t, rtol=0.0, atol=2.0e-14),
        f"control grid is not the declared uniform mesh: {path}",
    )
    require(float(np.min(u)) >= -1.0e-12 and float(np.max(u)) <= 3.0 + 1.0e-12, f"control violates [0,3]: {path}")
    return t, u, arrays


def independent_evaluate(t: np.ndarray, u: np.ndarray) -> Evaluation:
    """Execute one left-ZOH control, restarting DOP853 at every breakpoint."""

    m = 21
    grid = np.linspace(0.0, 1.0, m, dtype=np.float64)
    r = 2.0 / (1.0 + 3.0 * grid**4)
    phi = 1.0 / (1.0 + grid**2)
    suppression = np.full(m, 0.5, dtype=np.float64)
    state = np.concatenate([np.full(m, 10.0, dtype=np.float64), np.array([0.0])])
    states = [state[:m].copy()]

    for left, right, control in zip(t[:-1], t[1:], u):
        ui = float(control)

        def rhs(_time: float, augmented: np.ndarray) -> np.ndarray:
            population = augmented[:m]
            growth_inhibition = math.log1p(float(np.mean(population)))
            dynamics = (r - phi * ui - suppression * growth_inhibition) * population
            running = 0.1 * float(np.sum(population)) + 20.0 * ui
            return np.concatenate([dynamics, np.array([running])])

        solution = solve_ivp(
            rhs,
            (float(left), float(right)),
            state,
            method="DOP853",
            rtol=1.0e-10,
            atol=1.0e-12,
            t_eval=np.array([float(right)]),
            max_step=max((float(right) - float(left)) / 4.0, 1.0e-8),
        )
        require(solution.success, f"independent DOP853 failed: {solution.message}")
        state = np.asarray(solution.y[:, -1], dtype=np.float64)
        require(np.all(np.isfinite(state)) and np.all(state[:m] > 0.0), "invalid realized state")
        states.append(state[:m].copy())

    terminal = float(np.sum(state[:m]))
    running = float(state[m])
    return Evaluation(
        J=terminal + running,
        terminal_cost=terminal,
        running_cost=running,
        breakpoint_states=np.stack(states),
    )


def sample_stats(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    require(array.size >= 2, "sample statistics require at least two values")
    return float(np.mean(array)), float(np.std(array, ddof=1))


def state_discrepancy(arrays: Mapping[str, np.ndarray], evaluated: Evaluation) -> float | None:
    for key in ("N", "states", "breakpoint_states"):
        if key in arrays:
            saved = np.asarray(arrays[key], dtype=np.float64)
            if saved.shape == evaluated.breakpoint_states.shape:
                return float(np.max(np.abs(saved - evaluated.breakpoint_states)))
    return None


def scalar_array(arrays: Mapping[str, np.ndarray], key: str, label: str) -> object:
    require(key in arrays, f"{label} is missing")
    value = np.asarray(arrays[key])
    require(value.shape == (), f"{label} must be scalar")
    return value.item()


def array_sha256(*arrays: np.ndarray) -> str:
    """Reproduce the retained dataset fingerprint without importing method code."""

    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(str(contiguous.shape).encode("utf-8"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def independent_euler_dataset(samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    states = rng.uniform(0.1, 20.0, size=(samples, 21))
    actions = rng.uniform(0.0, 3.0, size=(samples, 1))
    grid = np.linspace(0.0, 1.0, 21, dtype=np.float64)
    r = 2.0 / (1.0 + 3.0 * grid**4)
    phi = 1.0 / (1.0 + grid**2)
    targets: list[np.ndarray] = []
    for state, action in zip(states, actions):
        inhibition = math.log1p(float(np.mean(state)))
        rhs = (r - phi * float(action[0]) - 0.5 * inhibition) * state
        targets.append(state + 0.05 * rhs)
    inputs = np.concatenate((states, actions), axis=1).astype(np.float32)
    return inputs, np.stack(targets).astype(np.float32)


def independent_euler_objective(control: np.ndarray) -> float:
    state = np.full(21, 10.0, dtype=np.float64)
    value = 0.0
    grid = np.linspace(0.0, 1.0, 21, dtype=np.float64)
    r = 2.0 / (1.0 + 3.0 * grid**4)
    phi = 1.0 / (1.0 + grid**2)
    for action in np.asarray(control, dtype=np.float64).reshape(-1):
        value += 0.05 * (0.1 * float(np.sum(state)) + 20.0 * float(action))
        inhibition = math.log1p(float(np.mean(state)))
        state = state + 0.05 * (r - phi * float(action) - 0.5 * inhibition) * state
    return float(value + np.sum(state))


class IndependentDynamicsMLP(nn.Module):
    """Minimal reconstruction of the declared 22-128-128-21 MLP."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(22, 128, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, 128, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(128, 21, dtype=torch.float32),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def learned_rollout(model: nn.Module, initial: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    states = [initial]
    state = initial
    for action in control:
        state = model(torch.cat((state, action.reshape(1))))
        states.append(state)
    return torch.stack(states)


def learned_objective(states: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
    beta = torch.full((21,), 0.1, dtype=torch.float32)
    alpha = torch.ones(21, dtype=torch.float32)
    value = torch.dot(alpha, states[-1])
    for index in range(control.shape[0]):
        value = value + 0.05 * (
            torch.dot(beta, states[index]) + 20.0 * control[index].reshape(-1)[0]
        )
    return value


def independent_controller_validation(
    model: nn.Module, validation_states: np.ndarray, control: np.ndarray
) -> tuple[float, np.ndarray]:
    tensor_control = torch.as_tensor(control[:, None], dtype=torch.float32)
    values: list[torch.Tensor] = []
    nominal_states: torch.Tensor | None = None
    with torch.no_grad():
        for index, initial in enumerate(validation_states):
            states = learned_rollout(
                model, torch.as_tensor(initial, dtype=torch.float32), tensor_control
            )
            values.append(learned_objective(states, tensor_control))
            if index == 0:
                nominal_states = states
    require(nominal_states is not None, "controller validation set is empty")
    return (
        float(torch.stack(values).mean()),
        nominal_states.detach().cpu().numpy().astype(np.float64),
    )


def audit_deepbsde(root: Path, reference_J: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    expected_dirs = {
        root / f"sigma_{sigma:.3f}_seed_{seed}"
        for sigma in EXPECTED_DEEPBSDE_SIGMAS
        for seed in EXPECTED_SEEDS
    }
    actual_dirs = {path for path in root.iterdir() if path.is_dir()}
    require(actual_dirs == expected_dirs, "DeepBSDE tumor grid is incomplete or contains extras")

    runs: list[dict[str, Any]] = []
    verified = 0
    for sigma in EXPECTED_DEEPBSDE_SIGMAS:
        for seed in EXPECTED_SEEDS:
            run_dir = root / f"sigma_{sigma:.3f}_seed_{seed}"
            manifest_path = run_dir / "manifest.json"
            manifest = load_json(manifest_path, "DeepBSDE worker manifest")
            require(float(manifest.get("sigma")) == sigma, f"DeepBSDE sigma mismatch: {run_dir}")
            require(strict_int(manifest.get("seed"), "DeepBSDE seed") == seed, f"DeepBSDE seed mismatch: {run_dir}")
            require(manifest.get("official_solver_unchanged") is True, f"DeepBSDE solver not certified: {run_dir}")
            require(manifest.get("state_coordinates") == "x=log(N)", f"DeepBSDE coordinates drift: {run_dir}")
            require(manifest.get("state_clipping") is False, f"DeepBSDE state clipping enabled: {run_dir}")
            control_values = manifest.get("control_values")
            require(
                isinstance(control_values, list)
                and control_values
                and all(float(value) in (0.0, 3.0) for value in control_values),
                f"DeepBSDE control values are not endpoint minimizers: {run_dir}",
            )
            require(
                manifest.get("control_extraction")
                == "exact argmin of u*(gamma-phi^T grad_x V) over [0,umax]",
                f"DeepBSDE control-extraction rule drift: {run_dir}",
            )
            config = manifest.get("config")
            require(isinstance(config, Mapping), f"DeepBSDE config missing: {run_dir}")
            eqn = config.get("eqn_config")
            net = config.get("net_config")
            require(isinstance(eqn, Mapping) and isinstance(net, Mapping), f"DeepBSDE config malformed: {run_dir}")
            require(strict_int(eqn.get("dim"), "DeepBSDE dim") == 21, "DeepBSDE dimension drift")
            require(strict_int(eqn.get("num_time_interval"), "DeepBSDE intervals") == 80, "DeepBSDE interval drift")
            require(strict_int(net.get("num_iterations"), "DeepBSDE iterations") == 2000, "DeepBSDE budget drift")

            artifacts = manifest.get("artifacts")
            require(isinstance(artifacts, Mapping) and artifacts, f"DeepBSDE artifact map missing: {run_dir}")
            expected_files = {"manifest.json", *map(str, artifacts.keys())}
            actual_files = {path.name for path in run_dir.iterdir() if path.is_file()}
            require(actual_files == expected_files, f"DeepBSDE artifact set drift: {run_dir}")
            for relative, declaration in artifacts.items():
                require(isinstance(declaration, Mapping), f"DeepBSDE artifact declaration malformed: {relative}")
                verify_hash(run_dir / str(relative), declaration.get("sha256"), "DeepBSDE artifact")
                verified += 1
            for source, declaration in manifest.get("official_sources", {}).items():
                require(isinstance(declaration, Mapping), f"DeepBSDE source declaration malformed: {source}")
                expected_source_hash = declaration.get(
                    "sha256", declaration.get("worktree_sha256")
                )
                verify_hash(ROOT / source, expected_source_hash, "DeepBSDE source")
                verified += 1

            control_path = run_dir / "canonical_control.npz"
            t, u, arrays = verify_npz_grid(control_path, 80)
            require(
                scalar_array(arrays, "control_semantics", "DeepBSDE control semantics")
                == "ZOH interval controls from exact HJB Hamiltonian argmin",
                f"DeepBSDE control semantics drift: {run_dir}",
            )
            require(
                np.all(np.logical_or(np.isclose(u, 0.0), np.isclose(u, 3.0))),
                f"DeepBSDE control is not endpoint-valued: {run_dir}",
            )
            require(
                "grad_log_value" in arrays and "psi" in arrays,
                f"DeepBSDE HJB diagnostics missing: {run_dir}",
            )
            gradient = np.asarray(arrays["grad_log_value"], dtype=np.float64)
            psi = np.asarray(arrays["psi"], dtype=np.float64).reshape(-1)
            phi = 1.0 / (1.0 + np.linspace(0.0, 1.0, 21) ** 2)
            reconstructed_psi = 20.0 - gradient @ phi
            require(
                gradient.shape == (80, 21)
                and psi.shape == (80,)
                and np.allclose(psi, reconstructed_psi, rtol=0.0, atol=2.0e-12),
                f"DeepBSDE switching-function artifact drift: {run_dir}",
            )
            reconstructed_u = np.where(
                psi > 0.0, 0.0, np.where(psi < 0.0, 3.0, 1.5)
            )
            require(
                np.array_equal(u, reconstructed_u),
                f"DeepBSDE saved control disagrees with endpoint argmin: {run_dir}",
            )
            evaluated = independent_evaluate(t, u)
            recorded = manifest.get("realized_metrics")
            require(isinstance(recorded, Mapping), f"DeepBSDE metrics missing: {run_dir}")
            discrepancy = abs(evaluated.J - float(recorded["J"]))
            require(discrepancy <= 2.0e-8, f"DeepBSDE independent J mismatch: {run_dir}")
            history = pd.read_csv(run_dir / "training_history.csv")
            require(list(history.columns) == ["step", "terminal_matching_loss", "y_init", "elapsed_seconds"], "DeepBSDE history schema drift")
            require(int(history.iloc[-1]["step"]) == 2000, "DeepBSDE history did not reach fixed budget")
            runs.append(
                {
                    "method_id": "deepbsde",
                    "variant": f"sigma={sigma:.3f}",
                    "sigma": sigma,
                    "seed": seed,
                    "selected_for_report": True,
                    "intervals": 80,
                    "native_name": "Y0",
                    "native_value": float(history.iloc[-1]["y_init"]),
                    "recorded_J": float(recorded["J"]),
                    "audited_J": evaluated.J,
                    "J_abs_discrepancy": discrepancy,
                    "breakpoint_state_abs_discrepancy": state_discrepancy(arrays, evaluated),
                    "control_path": str(control_path.relative_to(ROOT)),
                    "control_sha256": sha256_file(control_path),
                    "worker_manifest_sha256": sha256_file(manifest_path),
                    "local_adaptation_source_bound_in_worker_manifest": all(
                        relative in manifest.get("official_sources", {})
                        for relative in (
                            "faithful_related_work/deepbsde/runner.py",
                            "faithful_related_work/deepbsde/equations.py",
                        )
                    ),
                }
            )

    rows: list[dict[str, Any]] = []
    for sigma in EXPECTED_DEEPBSDE_SIGMAS:
        subset = [run for run in runs if run["sigma"] == sigma]
        native_mean, native_sd = sample_stats([run["native_value"] for run in subset])
        common_mean, common_sd = sample_stats([run["audited_J"] for run in subset])
        rows.append(
            {
                "method_id": "deepbsde",
                "method": "DeepBSDE tumor adaptation",
                "variant": f"sigma={sigma:.3f}",
                "seed_count": 3,
                "native_name": "Y0",
                "native_mean": native_mean,
                "native_sample_sd": native_sd,
                "common_J_mean": common_mean,
                "common_J_sample_sd": common_sd,
                "gap_percent_mean": 100.0 * (common_mean - reference_J) / reference_J,
                "gap_percent_sample_sd": 100.0 * common_sd / reference_J,
                "selection_rule": "all declared sigma/seed cells retained; fixed final optimizer step",
                "claim_scope": (
                    "official solver with a log-state tumor equation; finite sigma is a viscous "
                    "surrogate; retained manifests do not bind the local tumor adaptation layer"
                ),
            }
        )
    return rows, runs, verified


def _audit_neural_pmp_legacy(root: Path, reference_J: float) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    manifest_path = root / "manifest.json"
    manifest = load_json(manifest_path, "Neural-PMP tumor manifest")
    artifacts = manifest.get("artifacts")
    require(isinstance(artifacts, Mapping) and artifacts, "Neural-PMP artifact map missing")
    actual_files = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    require(actual_files == {"manifest.json", *map(str, artifacts.keys())}, "Neural-PMP artifact set drift")
    for relative, expected in artifacts.items():
        verify_hash(root / str(relative), expected, "Neural-PMP artifact")
    verified = len(artifacts)

    config = load_json(root / "run_config_and_summary.json", "Neural-PMP run summary")
    require(config.get("paper") == "Pontryagin Optimal Control via Neural Networks, arXiv:2212.14566", "Neural-PMP paper identity drift")
    require(config.get("algorithm_invariant") == "raw Hamiltonian gradient -> gradient step -> action projection", "Neural-PMP update ordering drift")
    require(config.get("gradient_clipping") is False, "Neural-PMP gradient clipping enabled")
    arguments = config.get("runner_arguments")
    require(isinstance(arguments, Mapping), "Neural-PMP runner arguments missing")
    require(arguments.get("seeds") == "0,1,2", "Neural-PMP seed grid drift")
    require(arguments.get("starts") == ",".join(EXPECTED_NEURAL_PMP_STARTS), "Neural-PMP start grid drift")
    require(strict_int(arguments.get("n"), "Neural-PMP intervals") == 200, "Neural-PMP interval drift")
    require(strict_int(arguments.get("dynamics_train_samples"), "Neural-PMP dynamics samples") == 2000, "Neural-PMP data budget drift")
    require(strict_int(arguments.get("dynamics_epochs"), "Neural-PMP dynamics epochs") == 1000, "Neural-PMP training budget drift")
    require(strict_int(arguments.get("control_iters"), "Neural-PMP control iterations") == 300, "Neural-PMP control budget drift")
    selection = config.get("selection")
    require(isinstance(selection, Mapping), "Neural-PMP selection declaration missing")
    require(selection.get("true_or_realized_objective_available_to_rule") is False, "Neural-PMP selection leaked true J")
    require(selection.get("cross_seed_performance_selection") is False, "Neural-PMP selected across seeds")
    require(strict_int(selection.get("canonical_seed"), "Neural-PMP canonical seed") == 0, "Neural-PMP canonical seed drift")

    frame = pd.read_csv(root / "all_runs.csv")
    require(len(frame) == 15, "Neural-PMP run table is not 3 seeds x 5 starts")
    require(set(frame["seed"].astype(int)) == set(EXPECTED_SEEDS), "Neural-PMP seed table drift")
    require(set(frame["start"].astype(str)) == set(EXPECTED_NEURAL_PMP_STARTS), "Neural-PMP start table drift")
    selected_map = {str(key): int(value) for key, value in selection.get("selected_run_index_by_seed", {}).items()}
    require(selected_map == {"0": 0, "1": 5, "2": 10}, "Neural-PMP selected-run map drift")

    runs: list[dict[str, Any]] = []
    for seed in EXPECTED_SEEDS:
        subset = frame.loc[frame["seed"].astype(int) == seed].sort_values("run_index")
        require(len(subset) == 5, f"Neural-PMP seed {seed} does not have five starts")
        selected_index = int(subset.loc[subset["selection_value"].idxmin(), "run_index"])
        require(selected_index == selected_map[str(seed)], f"Neural-PMP selection rule mismatch for seed {seed}")
        row = subset.loc[subset["run_index"].astype(int) == selected_index].iloc[0]
        control_path = root / str(row["control_checkpoint"])
        require(str(row["control_checkpoint"]) in artifacts, "Neural-PMP selected control lacks root hash")
        t, u, arrays = verify_npz_grid(control_path, 200)
        evaluated = independent_evaluate(t, u)
        if seed == 0:
            expected = float(config["selected_realized_metrics"]["J"])
            require(abs(evaluated.J - expected) <= 2.0e-8, "Neural-PMP canonical independent J mismatch")
        runs.append(
            {
                "method_id": "neural_pmp",
                "variant": "learned dynamics",
                "seed": seed,
                "start": str(row["start"]),
                "intervals": 200,
                "native_name": "learned_dynamics_validation_objective",
                "native_value": float(row["selection_value"]),
                "audited_J": evaluated.J,
                "breakpoint_state_abs_discrepancy": state_discrepancy(arrays, evaluated),
                "control_path": str(control_path.relative_to(ROOT)),
                "control_sha256": sha256_file(control_path),
            }
        )

    native_mean, native_sd = sample_stats([run["native_value"] for run in runs])
    common_mean, common_sd = sample_stats([run["audited_J"] for run in runs])
    result = {
        "method_id": "neural_pmp",
        "method": "Neural-PMP learned-dynamics tumor adaptation",
        "variant": "3 seeded dynamics models; one native-validation-selected start per seed",
        "seed_count": 3,
        "native_name": "learned_dynamics_validation_objective",
        "native_mean": native_mean,
        "native_sample_sd": native_sd,
        "common_J_mean": common_mean,
        "common_J_sample_sd": common_sd,
        "gap_percent_mean": 100.0 * (common_mean - reference_J) / reference_J,
        "gap_percent_sample_sd": 100.0 * common_sd / reference_J,
        "selection_rule": selection["rule"],
        "claim_scope": "paper Algorithm-1 update ordering with a custom tumor environment and declared reduced tumor budget",
        "execution_source_hash_bound_in_original_manifest": False,
    }
    return result, runs, verified


def audit_neural_pmp(
    root: Path, reference_J: float
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """Audit all 15 controls and every native-validation selection layer."""

    manifest = load_json(root / "manifest.json", "Neural-PMP tumor manifest")
    artifacts = manifest.get("artifacts")
    require(isinstance(artifacts, Mapping) and artifacts, "Neural-PMP artifact map missing")
    actual_files = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    require(
        actual_files == {"manifest.json", *map(str, artifacts)},
        "Neural-PMP artifact set drift",
    )
    for relative, expected in artifacts.items():
        verify_hash(root / str(relative), expected, "Neural-PMP artifact")
    verified = len(artifacts)

    config = load_json(root / "run_config_and_summary.json", "Neural-PMP run summary")
    require(
        config.get("paper")
        == "Pontryagin Optimal Control via Neural Networks, arXiv:2212.14566",
        "Neural-PMP paper identity drift",
    )
    require(
        config.get("algorithm_invariant")
        == "raw Hamiltonian gradient -> gradient step -> action projection",
        "Neural-PMP update ordering drift",
    )
    require(config.get("gradient_clipping") is False, "Neural-PMP gradient clipping enabled")
    arguments = config.get("runner_arguments")
    require(isinstance(arguments, Mapping), "Neural-PMP runner arguments missing")
    require(arguments.get("seeds") == "0,1,2", "Neural-PMP seed grid drift")
    require(
        arguments.get("starts") == ",".join(EXPECTED_NEURAL_PMP_STARTS),
        "Neural-PMP start grid drift",
    )
    expected_integer_arguments = {
        "n": 200,
        "m": 21,
        "dynamics_train_samples": 2000,
        "dynamics_validation_samples": 500,
        "dynamics_epochs": 1000,
        "dynamics_validation_interval": 50,
        "dynamics_hidden": 128,
        "control_iters": 300,
        "control_eval_interval": 10,
        "selection_seed": 20260712,
        "selection_states": 8,
    }
    for field, expected in expected_integer_arguments.items():
        require(
            strict_int(arguments.get(field), f"Neural-PMP {field}") == expected,
            f"Neural-PMP {field} drift",
        )
    selection = config.get("selection")
    require(isinstance(selection, Mapping), "Neural-PMP selection declaration missing")
    require(
        selection.get("true_or_realized_objective_available_to_rule") is False
        and selection.get("cross_seed_performance_selection") is False,
        "Neural-PMP selection declaration permits realized-J or cross-seed selection",
    )
    selected_map = {
        str(key): int(value)
        for key, value in selection.get("selected_run_index_by_seed", {}).items()
    }
    require(
        selected_map == {"0": 0, "1": 5, "2": 10},
        "Neural-PMP selected-run map drift",
    )

    validation_path = root / "datasets" / "control_validation_initial_states.npz"
    with np.load(validation_path, allow_pickle=False) as archive:
        require(archive.files == ["states"], "Neural-PMP controller-validation schema drift")
        validation_states = np.asarray(archive["states"])
    require(
        validation_states.shape == (8, 21) and validation_states.dtype == np.float32,
        "Neural-PMP controller-validation shape/dtype drift",
    )
    rng = np.random.default_rng(20260712)
    expected_validation_states = [np.full(21, 10.0, dtype=np.float64)]
    for _ in range(7):
        perturbation = rng.normal(0.0, 0.02, size=21)
        expected_validation_states.append(
            np.maximum(10.0 * (1.0 + perturbation), 1.0e-8)
        )
    require(
        np.array_equal(
            validation_states, np.stack(expected_validation_states).astype(np.float32)
        ),
        "Neural-PMP controller-validation states do not reconstruct from declared seed",
    )

    models: dict[int, IndependentDynamicsMLP] = {}
    dynamics_best_epochs: dict[str, int] = {}
    dynamics_mse_discrepancies: list[float] = []
    for seed in EXPECTED_SEEDS:
        dataset_path = root / "datasets" / f"dynamics_seed_{seed}.npz"
        with np.load(dataset_path, allow_pickle=False) as archive:
            require(
                set(archive.files)
                == {"train_inputs", "train_targets", "validation_inputs", "validation_targets"},
                f"Neural-PMP dynamics dataset schema drift for seed {seed}",
            )
            dataset = {name: np.asarray(archive[name]) for name in archive.files}
        expected_train = independent_euler_dataset(2000, seed * 2 + 101)
        expected_validation = independent_euler_dataset(500, seed * 2 + 102)
        require(
            np.array_equal(dataset["train_inputs"], expected_train[0])
            and np.array_equal(dataset["train_targets"], expected_train[1])
            and np.array_equal(dataset["validation_inputs"], expected_validation[0])
            and np.array_equal(dataset["validation_targets"], expected_validation[1]),
            f"Neural-PMP seed-{seed} dataset does not reconstruct from declared RNG rules",
        )

        checkpoint = torch.load(
            root / "dynamics" / f"dynamics_seed_{seed}.pt",
            map_location="cpu",
            weights_only=False,
        )
        require(isinstance(checkpoint, Mapping), f"Neural-PMP checkpoint malformed for seed {seed}")
        require(
            checkpoint.get("paper") == "arXiv:2212.14566"
            and checkpoint.get("mode") == "learned"
            and strict_int(checkpoint.get("seed"), "Neural-PMP checkpoint seed") == seed,
            f"Neural-PMP checkpoint identity drift for seed {seed}",
        )
        require(
            checkpoint.get("dataset_sha256")
            == array_sha256(
                dataset["train_inputs"],
                dataset["train_targets"],
                dataset["validation_inputs"],
                dataset["validation_targets"],
            ),
            f"Neural-PMP checkpoint/dataset fingerprint mismatch for seed {seed}",
        )
        model_config = checkpoint.get("model_config")
        require(
            isinstance(model_config, Mapping)
            and model_config.get("dtype") == "torch.float32"
            and int(model_config.get("state_dim", -1)) == 21
            and int(model_config.get("action_dim", -1)) == 1
            and int(model_config.get("hidden_dim", -1)) == 128,
            f"Neural-PMP model declaration drift for seed {seed}",
        )

        dynamics_history = pd.read_csv(
            root / "dynamics" / f"dynamics_seed_{seed}_history.csv"
        )
        require(
            list(dynamics_history.columns)
            == ["seed", "epoch", "training_mse", "validation_mse"]
            and len(dynamics_history) == 21
            and list(dynamics_history["epoch"].astype(int)) == list(range(0, 1001, 50))
            and set(dynamics_history["seed"].astype(int)) == {seed},
            f"Neural-PMP dynamics history budget drift for seed {seed}",
        )
        best_row = dynamics_history.loc[dynamics_history["validation_mse"].idxmin()]
        best_epoch = strict_int(checkpoint.get("best_epoch"), "Neural-PMP best epoch")
        require(
            best_epoch == int(best_row["epoch"])
            and abs(float(checkpoint["best_validation_mse"]) - float(best_row["validation_mse"]))
            <= 1.0e-14,
            f"Neural-PMP dynamics validation selection mismatch for seed {seed}",
        )
        dynamics_best_epochs[str(seed)] = best_epoch

        model = IndependentDynamicsMLP()
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            predicted = model(torch.as_tensor(dataset["validation_inputs"], dtype=torch.float32))
            validation_mse = float(
                torch.mean(
                    (predicted - torch.as_tensor(dataset["validation_targets"], dtype=torch.float32))
                    ** 2
                )
            )
        mse_discrepancy = abs(validation_mse - float(checkpoint["best_validation_mse"]))
        require(
            mse_discrepancy <= 1.0e-12,
            f"Neural-PMP checkpoint validation MSE mismatch for seed {seed}",
        )
        dynamics_mse_discrepancies.append(mse_discrepancy)
        models[seed] = model

    frame = pd.read_csv(root / "all_runs.csv")
    expected_pairs = [
        (seed, start) for seed in EXPECTED_SEEDS for start in EXPECTED_NEURAL_PMP_STARTS
    ]
    require(
        len(frame) == 15
        and list(frame["run_index"].astype(int)) == list(range(15))
        and list(zip(frame["seed"].astype(int), frame["start"].astype(str)))
        == expected_pairs,
        "Neural-PMP run table is not the exact declared 3-seed x 5-start grid",
    )
    history = pd.read_csv(root / "history.csv")
    require(
        {"run_index", "seed", "start", "iteration", "validation_objective"}.issubset(
            history.columns
        ),
        "Neural-PMP controller history schema drift",
    )

    runs: list[dict[str, Any]] = []
    validation_discrepancies: list[float] = []
    discrete_discrepancies: list[float] = []
    for _, row in frame.iterrows():
        run_index = int(row["run_index"])
        seed = int(row["seed"])
        start = str(row["start"])
        selected = run_index == selected_map[str(seed)]
        control_relative = str(row["control_checkpoint"])
        require(control_relative in artifacts, f"Neural-PMP control lacks root hash: {control_relative}")
        require(
            str(row["dynamics_checkpoint_sha256"])
            == sha256_file(root / str(row["dynamics_checkpoint"])),
            f"Neural-PMP dynamics checkpoint linkage mismatch for run {run_index}",
        )
        control_path = root / control_relative
        t, u, arrays = verify_npz_grid(control_path, 200)
        require(
            scalar_array(arrays, "dynamics_checkpoint_sha256", "Neural-PMP dynamics hash")
            == str(row["dynamics_checkpoint_sha256"]),
            f"Neural-PMP control/dynamics hash mismatch for run {run_index}",
        )
        run_history = history.loc[history["run_index"].astype(int) == run_index]
        require(
            len(run_history) == 31
            and list(run_history["iteration"].astype(int)) == list(range(0, 301, 10))
            and set(run_history["seed"].astype(int)) == {seed}
            and set(run_history["start"].astype(str)) == {start},
            f"Neural-PMP controller history budget drift for run {run_index}",
        )
        best_history = run_history.loc[run_history["validation_objective"].idxmin()]
        require(
            int(best_history["iteration"]) == int(row["best_iteration"])
            and abs(float(best_history["validation_objective"]) - float(row["selection_value"]))
            <= 1.0e-12
            and int(scalar_array(arrays, "best_iteration", "Neural-PMP best iteration"))
            == int(row["best_iteration"])
            and abs(
                float(scalar_array(arrays, "validation_objective", "Neural-PMP validation objective"))
                - float(row["selection_value"])
            )
            <= 1.0e-12,
            f"Neural-PMP controller-iterate selection mismatch for run {run_index}",
        )
        validation_value, recomputed_learned = independent_controller_validation(
            models[seed], validation_states, u
        )
        validation_discrepancy = abs(validation_value - float(row["selection_value"]))
        require(
            validation_discrepancy <= 1.0e-10,
            f"Neural-PMP recomputed controller validation mismatch for run {run_index}",
        )
        validation_discrepancies.append(validation_discrepancy)
        learned_states = np.asarray(arrays.get("learned_states"), dtype=np.float64)
        require(
            learned_states.shape == (201, 21)
            and np.allclose(learned_states, recomputed_learned, rtol=0.0, atol=2.0e-6),
            f"Neural-PMP saved learned rollout mismatch for run {run_index}",
        )

        evaluated = independent_evaluate(t, u)
        learned_difference = learned_states - evaluated.breakpoint_states
        discrete_J = independent_euler_objective(u)
        discrete_discrepancy = abs(
            discrete_J - float(row["true_discrete_objective_post_selection"])
        )
        require(
            discrete_discrepancy <= 2.0e-8,
            f"Neural-PMP post-selection Euler objective mismatch for run {run_index}",
        )
        discrete_discrepancies.append(discrete_discrepancy)
        record: dict[str, Any] = {
            "method_id": "neural_pmp",
            "variant": "learned dynamics",
            "run_index": run_index,
            "seed": seed,
            "start": start,
            "selected_for_report": selected,
            "intervals": 200,
            "native_name": "learned_dynamics_validation_objective",
            "native_value": float(row["selection_value"]),
            "best_control_iteration": int(row["best_iteration"]),
            "audited_J": evaluated.J,
            "recorded_discrete_euler_J": float(row["true_discrete_objective_post_selection"]),
            "audited_discrete_euler_J": discrete_J,
            "discrete_euler_J_abs_discrepancy": discrete_discrepancy,
            "learned_state_min": float(np.min(learned_states)),
            "learned_negative_entry_count": int(np.count_nonzero(learned_states < 0.0)),
            "learned_negative_timepoint_count": int(
                np.count_nonzero(np.any(learned_states < 0.0, axis=1))
            ),
            "learned_vs_realized_state_max_abs": float(np.max(np.abs(learned_difference))),
            "learned_vs_realized_state_rmse": float(
                np.sqrt(np.mean(learned_difference**2))
            ),
            "learned_terminal_total": float(np.sum(learned_states[-1])),
            "realized_terminal_total": evaluated.terminal_cost,
            "control_path": str(control_path.relative_to(ROOT)),
            "control_sha256": sha256_file(control_path),
        }
        if seed == 0 and selected:
            recorded_J = float(config["selected_realized_metrics"]["J"])
            discrepancy = abs(evaluated.J - recorded_J)
            require(discrepancy <= 2.0e-8, "Neural-PMP canonical independent J mismatch")
            record.update(recorded_J=recorded_J, J_abs_discrepancy=discrepancy)
        runs.append(record)

    for seed in EXPECTED_SEEDS:
        subset = frame.loc[frame["seed"].astype(int) == seed]
        selected_index = int(subset.loc[subset["selection_value"].idxmin(), "run_index"])
        require(
            selected_index == selected_map[str(seed)],
            f"Neural-PMP start-selection rule mismatch for seed {seed}",
        )
    selected_runs = [run for run in runs if run["selected_for_report"]]
    require(len(selected_runs) == 3, "Neural-PMP selected run count drift")
    ranks: dict[str, int] = {}
    selected_best: dict[str, dict[str, Any]] = {}
    for seed in EXPECTED_SEEDS:
        subset = [run for run in runs if run["seed"] == seed]
        ordered = sorted(subset, key=lambda run: float(run["audited_J"]))
        selected_run = next(run for run in subset if run["selected_for_report"])
        ranks[str(seed)] = 1 + next(
            index
            for index, run in enumerate(ordered)
            if run["run_index"] == selected_run["run_index"]
        )
        selected_best[str(seed)] = {
            "selected_start": selected_run["start"],
            "selected_common_J": selected_run["audited_J"],
            "best_common_J_start_diagnostic_only": ordered[0]["start"],
            "best_common_J_diagnostic_only": ordered[0]["audited_J"],
        }
    correlation = spearmanr(
        [float(run["native_value"]) for run in runs],
        [float(run["audited_J"]) for run in runs],
    )
    require(math.isfinite(float(correlation.statistic)), "Neural-PMP correlation is invalid")

    native_mean, native_sd = sample_stats(
        [float(run["native_value"]) for run in selected_runs]
    )
    common_mean, common_sd = sample_stats(
        [float(run["audited_J"]) for run in selected_runs]
    )
    result = {
        "method_id": "neural_pmp",
        "method": "Neural-PMP learned-dynamics tumor adaptation",
        "variant": "3 seeded dynamics models; one native-validation-selected start per seed",
        "seed_count": 3,
        "native_name": "learned_dynamics_validation_objective",
        "native_mean": native_mean,
        "native_sample_sd": native_sd,
        "common_J_mean": common_mean,
        "common_J_sample_sd": common_sd,
        "gap_percent_mean": 100.0 * (common_mean - reference_J) / reference_J,
        "gap_percent_sample_sd": 100.0 * common_sd / reference_J,
        "selection_rule": selection["rule"],
        "selection_diagnostics": {
            "layers": [
                "best dynamics checkpoint by dynamics-validation MSE",
                "best control iterate by the fixed 8-state controller-validation objective",
                "best of 5 starts by the same fixed 8-state controller-validation objective",
            ],
            "dynamics_best_epochs_by_seed": dynamics_best_epochs,
            "controller_validation_state_count": 8,
            "controller_validation_set_reused_for_iterate_and_start_selection": True,
            "held_out_native_test_set": False,
            "selected_common_J_rank_within_five_by_seed": ranks,
            "selected_and_best_common_J_by_seed": selected_best,
            "all_15_native_vs_common_J_spearman": float(correlation.statistic),
            "common_J_was_available_to_selection": False,
        },
        "learned_rollout_diagnostics": {
            "selected_by_seed": [
                {
                    key: run[key]
                    for key in (
                        "seed",
                        "learned_negative_timepoint_count",
                        "learned_negative_entry_count",
                        "learned_state_min",
                        "learned_vs_realized_state_max_abs",
                        "learned_vs_realized_state_rmse",
                        "learned_terminal_total",
                        "realized_terminal_total",
                    )
                }
                for run in selected_runs
            ],
            "selected_max_abs_state_difference": max(
                float(run["learned_vs_realized_state_max_abs"]) for run in selected_runs
            ),
            "selected_negative_timepoint_total": sum(
                int(run["learned_negative_timepoint_count"]) for run in selected_runs
            ),
        },
        "audit_recomputed_selection_metrics": {
            "max_dynamics_validation_mse_abs_discrepancy": max(
                dynamics_mse_discrepancies, default=0.0
            ),
            "max_controller_validation_objective_abs_discrepancy": max(
                validation_discrepancies, default=0.0
            ),
            "max_post_selection_discrete_J_abs_discrepancy": max(
                discrete_discrepancies, default=0.0
            ),
        },
        "claim_scope": (
            "paper/released-code-informed isolated reimplementation of Algorithm-1 update "
            "ordering with a custom tumor environment and reduced tumor budget"
        ),
        "execution_source_hash_bound_in_original_manifest": False,
    }
    return result, runs, verified


def _audit_pi_deeponet_legacy(root: Path, reference_J: float) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    manifest_path = root / "manifest.json"
    manifest = load_json(manifest_path, "PI-DeepONet tumor manifest")
    require(manifest.get("experiment_label") == "tumor adaptation", "PI-DeepONet experiment identity drift")
    require(manifest.get("upstream") == {"author_code_used": False, "kind": "paper specification", "repository_revision": None}, "PI-DeepONet source claim drift")
    experiment = manifest.get("experiment")
    training = manifest.get("train_config")
    require(isinstance(experiment, Mapping) and isinstance(training, Mapping), "PI-DeepONet protocol missing")
    require(experiment.get("seeds") == [0, 1, 2], "PI-DeepONet seed grid drift")
    require(experiment.get("selection_rule") == "final_predeclared_outer_iteration", "PI-DeepONet selection drift")
    require(experiment.get("selection_uses_nominal_realized_objective") is False, "PI-DeepONet selection leaked J")
    require(strict_int(experiment.get("evaluation_intervals"), "PI-DeepONet intervals") == 200, "PI-DeepONet interval drift")
    require(strict_int(training.get("outer_iterations"), "PI-DeepONet outer iterations") == 3, "PI-DeepONet outer budget drift")
    require(strict_int(training.get("steps_per_outer"), "PI-DeepONet steps per outer") == 300, "PI-DeepONet optimizer budget drift")
    require(training.get("gradient_clip") is None, "PI-DeepONet gradient clipping enabled")
    summary_decl = manifest.get("summary_csv")
    require(isinstance(summary_decl, Mapping), "PI-DeepONet summary hash missing")
    verify_hash(root / str(summary_decl["path"]), summary_decl.get("sha256"), "PI-DeepONet summary")
    verified = 1

    seed_entries = manifest.get("seeds")
    require(isinstance(seed_entries, list) and len(seed_entries) == 3, "PI-DeepONet seed manifests incomplete")
    runs: list[dict[str, Any]] = []
    for entry in seed_entries:
        require(isinstance(entry, Mapping), "PI-DeepONet seed entry malformed")
        seed = strict_int(entry.get("seed"), "PI-DeepONet seed")
        require(seed in EXPECTED_SEEDS, f"unexpected PI-DeepONet seed {seed}")
        require(entry.get("selection_rule") == "final_predeclared_outer_iteration", "PI-DeepONet seed selection drift")
        seed_dir = root / f"seed_{seed}"
        candidates = entry.get("candidates")
        require(isinstance(candidates, list) and [candidate.get("outer") for candidate in candidates] == [0, 1, 2], "PI-DeepONet candidate grid drift")
        for candidate in candidates:
            require(candidate.get("domain_exit") is False, "PI-DeepONet candidate exits declared domain")
            verify_hash(seed_dir / str(candidate["checkpoint"]), candidate.get("checkpoint_sha256"), "PI-DeepONet checkpoint")
            verify_hash(seed_dir / str(candidate["solution"]), candidate.get("solution_sha256"), "PI-DeepONet solution")
            verified += 2
        verify_hash(seed_dir / str(entry["history"]), entry.get("history_sha256"), "PI-DeepONet history")
        verify_hash(seed_dir / str(entry["selected_solution"]), entry.get("selected_solution_sha256"), "PI-DeepONet selected solution")
        verified += 2
        selected = candidates[-1]
        require(entry.get("selected_solution_sha256") == selected.get("solution_sha256"), "PI-DeepONet did not export final outer iteration")
        control_path = seed_dir / str(entry["selected_solution"])
        t, u, arrays = verify_npz_grid(control_path, 200)
        evaluated = independent_evaluate(t, u)
        recorded = float(selected["J_realized"])
        discrepancy = abs(evaluated.J - recorded)
        require(discrepancy <= 2.0e-8, f"PI-DeepONet independent J mismatch for seed {seed}")
        runs.append(
            {
                "method_id": "pi_deeponet",
                "variant": "paper-derived tumor adaptation",
                "seed": seed,
                "outer": 2,
                "intervals": 200,
                "native_name": None,
                "native_value": None,
                "recorded_J": recorded,
                "audited_J": evaluated.J,
                "J_abs_discrepancy": discrepancy,
                "breakpoint_state_abs_discrepancy": state_discrepancy(arrays, evaluated),
                "control_path": str(control_path.relative_to(ROOT)),
                "control_sha256": sha256_file(control_path),
            }
        )

    common_mean, common_sd = sample_stats([run["audited_J"] for run in runs])
    result = {
        "method_id": "pi_deeponet",
        "method": "PI-DeepONet tumor adaptation",
        "variant": "paper-derived implementation; final predeclared outer iteration",
        "seed_count": 3,
        "native_name": None,
        "native_mean": None,
        "native_sample_sd": None,
        "common_J_mean": common_mean,
        "common_J_sample_sd": common_sd,
        "gap_percent_mean": 100.0 * (common_mean - reference_J) / reference_J,
        "gap_percent_sample_sd": 100.0 * common_sd / reference_J,
        "selection_rule": "final predeclared outer iteration for every seed; no J-based selection",
        "claim_scope": "paper-derived algorithmic adaptation; author code and key numerical hyperparameters were unavailable",
        "execution_source_hash_bound_in_original_manifest": False,
    }
    return result, runs, verified


def audit_pi_deeponet(
    root: Path, reference_J: float
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """Audit all nine outer-iteration controls and the finite-difference domain."""

    manifest = load_json(root / "manifest.json", "PI-DeepONet tumor manifest")
    require(
        manifest.get("experiment_label") == "tumor adaptation",
        "PI-DeepONet experiment identity drift",
    )
    require(
        manifest.get("upstream")
        == {
            "author_code_used": False,
            "kind": "paper specification",
            "repository_revision": None,
        },
        "PI-DeepONet source claim drift",
    )
    experiment = manifest.get("experiment")
    training = manifest.get("train_config")
    problem = manifest.get("problem")
    require(
        isinstance(experiment, Mapping)
        and isinstance(training, Mapping)
        and isinstance(problem, Mapping),
        "PI-DeepONet protocol missing",
    )
    require(experiment.get("seeds") == [0, 1, 2], "PI-DeepONet seed grid drift")
    require(
        experiment.get("selection_rule") == "final_predeclared_outer_iteration"
        and experiment.get("selection_uses_nominal_realized_objective") is False,
        "PI-DeepONet selection protocol drift",
    )
    require(
        strict_int(experiment.get("evaluation_intervals"), "PI-DeepONet intervals") == 200
        and strict_int(training.get("outer_iterations"), "PI-DeepONet outer iterations") == 3
        and strict_int(training.get("steps_per_outer"), "PI-DeepONet steps per outer") == 300,
        "PI-DeepONet fixed budget drift",
    )
    require(training.get("gradient_clip") is None, "PI-DeepONet gradient clipping enabled")
    h = float(training.get("h", np.nan))
    require(h == 0.02, "PI-DeepONet finite-difference spacing drift")
    require(
        float(problem.get("normalized_state_lower", np.nan)) == 0.0
        and float(problem.get("normalized_state_upper", np.nan)) == 1.0
        and problem.get("state_coordinate") == "x=N/state_scale",
        "PI-DeepONet declared normalized domain drift",
    )

    summary_decl = manifest.get("summary_csv")
    require(isinstance(summary_decl, Mapping), "PI-DeepONet summary hash missing")
    verify_hash(root / str(summary_decl["path"]), summary_decl.get("sha256"), "PI-DeepONet summary")
    verified = 1
    seed_entries = manifest.get("seeds")
    require(
        isinstance(seed_entries, list)
        and [entry.get("seed") for entry in seed_entries if isinstance(entry, Mapping)]
        == [0, 1, 2],
        "PI-DeepONet seed manifests are not the exact declared grid",
    )

    runs: list[dict[str, Any]] = []
    for entry in seed_entries:
        require(isinstance(entry, Mapping), "PI-DeepONet seed entry malformed")
        seed = strict_int(entry.get("seed"), "PI-DeepONet seed")
        require(
            entry.get("selection_rule") == "final_predeclared_outer_iteration",
            "PI-DeepONet seed selection drift",
        )
        seed_dir = root / f"seed_{seed}"
        candidates = entry.get("candidates")
        require(
            isinstance(candidates, list)
            and [candidate.get("outer") for candidate in candidates] == [0, 1, 2],
            "PI-DeepONet candidate grid drift",
        )
        expected_files = {"history.csv", "solution.npz"}
        for outer in range(3):
            expected_files.update(
                {f"checkpoint_outer_{outer:03d}.pt", f"solution_outer_{outer:03d}.npz"}
            )
        require(
            {path.name for path in seed_dir.iterdir() if path.is_file()} == expected_files,
            f"PI-DeepONet seed-{seed} artifact set drift",
        )

        history = pd.read_csv(seed_dir / str(entry["history"]))
        require(
            {"outer", "step"}.issubset(history.columns),
            f"PI-DeepONet history schema drift for seed {seed}",
        )
        for outer in range(3):
            subset = history.loc[history["outer"].astype(int) == outer]
            require(
                len(subset) == 21
                and list(subset["step"].astype(int)) == [1, *range(15, 301, 15)],
                f"PI-DeepONet optimizer budget drift for seed {seed}, outer {outer}",
            )
        verify_hash(
            seed_dir / str(entry["history"]),
            entry.get("history_sha256"),
            "PI-DeepONet history",
        )
        verify_hash(
            seed_dir / str(entry["selected_solution"]),
            entry.get("selected_solution_sha256"),
            "PI-DeepONet selected solution",
        )
        verified += 2
        require(
            entry.get("selected_solution_sha256") == candidates[-1].get("solution_sha256"),
            "PI-DeepONet did not export final outer iteration",
        )

        for candidate in candidates:
            outer = strict_int(candidate.get("outer"), "PI-DeepONet outer")
            require(
                candidate.get("domain_exit") is False,
                "PI-DeepONet candidate center trajectory exits declared domain",
            )
            verify_hash(
                seed_dir / str(candidate["checkpoint"]),
                candidate.get("checkpoint_sha256"),
                "PI-DeepONet checkpoint",
            )
            control_path = seed_dir / str(candidate["solution"])
            verify_hash(
                control_path,
                candidate.get("solution_sha256"),
                "PI-DeepONet solution",
            )
            verified += 2
            t, u, arrays = verify_npz_grid(control_path, 200)
            require(
                scalar_array(arrays, "control_execution", "PI-DeepONet control execution")
                == "zero_order_hold"
                and scalar_array(arrays, "selection_rule", "PI-DeepONet selection rule")
                == "final_predeclared_outer_iteration",
                f"PI-DeepONet control semantics drift for seed {seed}, outer {outer}",
            )
            x = np.asarray(arrays.get("x"), dtype=np.float64)
            require(
                x.shape == (201, 21)
                and float(np.min(x)) >= -1.0e-12
                and float(np.max(x)) <= 1.0 + 1.0e-12,
                f"PI-DeepONet center state leaves declared domain for seed {seed}, outer {outer}",
            )
            stencil_outside = np.logical_or(x[:-1] - h < 0.0, x[:-1] + h > 1.0)
            stencil_outside_timepoints = int(
                np.count_nonzero(np.any(stencil_outside, axis=1))
            )
            evaluated = independent_evaluate(t, u)
            recorded = float(candidate["J_realized"])
            discrepancy = abs(evaluated.J - recorded)
            require(
                discrepancy <= 2.0e-8,
                f"PI-DeepONet independent J mismatch for seed {seed}, outer {outer}",
            )
            physical_discrepancy = state_discrepancy(arrays, evaluated)
            require(
                physical_discrepancy is not None,
                f"PI-DeepONet physical rollout missing for seed {seed}, outer {outer}",
            )
            runs.append(
                {
                    "method_id": "pi_deeponet",
                    "variant": "paper-derived tumor adaptation",
                    "seed": seed,
                    "outer": outer,
                    "selected_for_report": outer == 2,
                    "intervals": 200,
                    "native_name": None,
                    "native_value": None,
                    "recorded_J": recorded,
                    "audited_J": evaluated.J,
                    "J_abs_discrepancy": discrepancy,
                    "breakpoint_state_abs_discrepancy": physical_discrepancy,
                    "center_domain_exit": False,
                    "finite_difference_h": h,
                    "stencil_outside_domain_timepoint_count": stencil_outside_timepoints,
                    "control_path": str(control_path.relative_to(ROOT)),
                    "control_sha256": sha256_file(control_path),
                }
            )

    selected_runs = [run for run in runs if run["selected_for_report"]]
    require(len(selected_runs) == 3, "PI-DeepONet selected run count drift")
    outer_stats: dict[str, dict[str, float]] = {}
    for outer in range(3):
        values = [float(run["audited_J"]) for run in runs if run["outer"] == outer]
        mean, sd = sample_stats(values)
        outer_stats[str(outer)] = {"common_J_mean": mean, "common_J_sample_sd": sd}
    every_final_worse = all(
        next(run for run in runs if run["seed"] == seed and run["outer"] == 2)["audited_J"]
        > next(run for run in runs if run["seed"] == seed and run["outer"] == 0)["audited_J"]
        for seed in EXPECTED_SEEDS
    )
    require(every_final_worse, "PI-DeepONet retained worsening diagnostic changed")
    common_mean, common_sd = sample_stats(
        [float(run["audited_J"]) for run in selected_runs]
    )
    result = {
        "method_id": "pi_deeponet",
        "method": "PI-DeepONet tumor adaptation",
        "variant": "paper-derived implementation; final declared outer iteration",
        "seed_count": 3,
        "native_name": None,
        "native_mean": None,
        "native_sample_sd": None,
        "common_J_mean": common_mean,
        "common_J_sample_sd": common_sd,
        "gap_percent_mean": 100.0 * (common_mean - reference_J) / reference_J,
        "gap_percent_sample_sd": 100.0 * common_sd / reference_J,
        "selection_rule": "final declared outer iteration for every seed; no J-based selection",
        "outer_iteration_diagnostics": {
            "common_J_by_outer": outer_stats,
            "every_seed_final_outer_worse_than_outer_zero": every_final_worse,
            "policy_improvement_or_convergence_observed": False,
        },
        "finite_difference_domain_diagnostics": {
            "h": h,
            "declared_normalized_state_domain": [0.0, 1.0],
            "selected_stencil_outside_timepoint_count_by_seed": {
                str(run["seed"]): int(run["stencil_outside_domain_timepoint_count"])
                for run in selected_runs
            },
            "center_only_domain_exit_flag_is_insufficient": True,
        },
        "paper_theorem_scope": {
            "A2_unique_argmin_satisfied": False,
            "A2_lipschitz_control_map_satisfied": False,
            "reason": (
                "linear control cost plus control-affine dynamics gives a bang-bang argmin; "
                "it is nonunique at switching coefficient zero and discontinuous across zero"
            ),
            "tumor_adaptation_covered_by_paper_convergence_theorem": False,
        },
        "claim_scope": (
            "paper-derived mechanism outside the paper's A2 convergence assumptions; author "
            "code and key numerical hyperparameters unavailable; no policy-convergence claim"
        ),
        "execution_source_hash_bound_in_original_manifest": False,
    }
    return result, runs, verified


def write_outputs(output: Path, summary: dict[str, Any], runs: list[dict[str, Any]], inputs: dict[str, str]) -> None:
    output.mkdir(parents=True, exist_ok=False)
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_path = output / "summary.csv"
    fields = [
        "method_id", "method", "variant", "seed_count", "native_name",
        "native_mean", "native_sample_sd", "common_J_mean", "common_J_sample_sd",
        "gap_percent_mean", "gap_percent_sample_sd", "selection_rule", "claim_scope",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary["rows"])

    runs_path = output / "audited_runs.json"
    runs_path.write_text(json.dumps(runs, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    source_paths = [
        Path(__file__).resolve(),
        ROOT / "faithful_related_work" / "deepbsde" / "equations.py",
        ROOT / "faithful_related_work" / "deepbsde" / "runner.py",
        ROOT / "external" / "DeepBSDE" / "equation.py",
        ROOT / "external" / "DeepBSDE" / "solver.py",
        ROOT / "faithful_related_work" / "neural_pmp" / "core.py",
        ROOT / "faithful_related_work" / "neural_pmp" / "dynamics.py",
        ROOT / "faithful_related_work" / "neural_pmp" / "run_tumor.py",
        ROOT / "faithful_related_work" / "neural_pmp" / "tumor.py",
        ROOT / "faithful_related_work" / "pi_deeponet" / "core.py",
        ROOT / "faithful_related_work" / "pi_deeponet" / "experiment.py",
        ROOT / "faithful_related_work" / "pi_deeponet" / "problems.py",
        ROOT / "faithful_related_work" / "pi_deeponet" / "run_tumor.py",
        ROOT / "external" / "pi_deeponet_paper" / "source" / "main_arxiv.tex",
    ]
    source_hashes = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in source_paths
        if path.is_file()
    }
    manifest = {
        "schema_version": 2,
        "status": "passed",
        "scope": "final multi-seed DeepBSDE, Neural-PMP, and PI-DeepONet tumor adaptations",
        "evaluator_independence": {
            "imports_tumor_problem": False,
            "imports_method_runner": False,
            "method": "DOP853 restarted at every left-ZOH breakpoint",
            "rtol": 1.0e-10,
            "atol": 1.0e-12,
        },
        "excluded": [
            {
                "method_id": "pinn_pi",
                "reason": "faithful LyZNet artifact is original-pendulum smoke with tumor_comparable=false; no faithful tumor control exists",
            }
        ],
        "input_manifest_sha256": inputs,
        "audit_source_sha256": source_hashes,
        "artifacts": {
            "summary.json": sha256_file(summary_path),
            "summary.csv": sha256_file(csv_path),
            "audited_runs.json": sha256_file(runs_path),
        },
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
    }
    (output / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--reference-manifest", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    reference_manifest = load_json(args.reference_manifest, "canonical reference manifest")
    reference_J = float(reference_manifest["selected"]["direct time-mesh"]["J"])
    require(math.isfinite(reference_J) and reference_J > 0.0, "canonical reference J is invalid")

    deep_root = args.results_root / "deepbsde_tumor_full"
    neural_root = args.results_root / "neural_pmp_tumor_budgeted_3seed"
    pi_root = args.results_root / "pi_deeponet_tumor_final_fidelity"
    deep_rows, deep_runs, deep_verified = audit_deepbsde(deep_root, reference_J)
    neural_row, neural_runs, neural_verified = audit_neural_pmp(neural_root, reference_J)
    pi_row, pi_runs, pi_verified = audit_pi_deeponet(pi_root, reference_J)
    rows = [*deep_rows, neural_row, pi_row]
    runs = [*deep_runs, *neural_runs, *pi_runs]
    state_discrepancies = [
        float(run["breakpoint_state_abs_discrepancy"])
        for run in runs
        if run.get("breakpoint_state_abs_discrepancy") is not None
    ]
    objective_discrepancies = [
        float(run["J_abs_discrepancy"])
        for run in runs
        if run.get("J_abs_discrepancy") is not None
    ]
    selected_runs = [run for run in runs if run.get("selected_for_report") is True]
    summary = {
        "schema_version": 2,
        "status": "passed",
        "reference_J": reference_J,
        "reference_manifest": str(args.reference_manifest.resolve().relative_to(ROOT)),
        "reference_manifest_sha256": sha256_file(args.reference_manifest),
        "selection_uses_common_realized_J": False,
        "rows": rows,
        "candidate_control_count": len(runs),
        "reported_run_count": len(selected_runs),
        "all_candidate_controls_independently_evaluated": True,
        "verified_declared_artifact_or_source_hashes": deep_verified + neural_verified + pi_verified,
        "max_discrepancies": {
            "recorded_objective_J_abs": max(objective_discrepancies, default=0.0),
            "recorded_objective_comparison_count": len(objective_discrepancies),
            "saved_physical_breakpoint_state_abs": max(state_discrepancies, default=0.0),
            "saved_physical_state_comparison_count": len(state_discrepancies),
        },
        "provenance_limit": (
            "Retained DeepBSDE worker manifests bind the external solver/equation but not "
            "the local tumor adaptation layer; Neural-PMP and PI-DeepONet root manifests "
            "hash retained result artifacts but do not bind their executed local runner "
            "sources. This audit records current source hashes and does not claim they prove "
            "execution-time source identity."
        ),
    }
    input_paths = [
        *[
            deep_root / f"sigma_{sigma:.3f}_seed_{seed}" / "manifest.json"
            for sigma in EXPECTED_DEEPBSDE_SIGMAS
            for seed in EXPECTED_SEEDS
        ],
        neural_root / "manifest.json",
        neural_root / "run_config_and_summary.json",
        pi_root / "manifest.json",
        args.reference_manifest.resolve(),
    ]
    inputs = {
        str(path.resolve().relative_to(ROOT)): sha256_file(path)
        for path in input_paths
    }
    write_outputs(args.out_dir, summary, runs, inputs)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
