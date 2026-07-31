#!/usr/bin/env python3
"""Fail-closed, independent numerical audit of the final HJB-NN tumor sweep.

This script intentionally does *not* import or call
``faithful_related_work.hjb_nn.problem``.  It reconstructs the nominal tumor
ODE and both declared running costs below, then integrates each interval of
the exported left-endpoint ZOH control separately with SciPy's DOP853 solver.

The input is the aggregate ``manifest.json`` produced by
``faithful_related_work.hjb_nn.runner aggregate-tumor``.  A successful audit
writes ``summary.csv`` and ``audit_manifest.json`` to a previously absent or
empty output directory.  Any missing declaration, incomplete grid, unsafe
path, hash mismatch, infeasible control, malformed NPZ, numerical mismatch,
or nonpositive trajectory aborts before either result file is published.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.integrate import solve_ivp


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TAUS = (10.0, 5.0, 2.0)
EXPECTED_SEEDS = (0, 1, 2)
EXPECTED_EXECUTED_SOURCE_KEYS = (
    "external/HJB_NN/examples/problem_def_template.py",
    "external/HJB_NN/examples/satellite/problem_def.py",
    "external/HJB_NN/utilities/neural_networks.py",
    "external/HJB_NN/utilities/optimize.py",
    "external/HJB_NN/utilities/other.py",
    "faithful_related_work/hjb_nn/bvp.py",
    "faithful_related_work/hjb_nn/official_adapter.py",
    "faithful_related_work/hjb_nn/problem.py",
    "faithful_related_work/hjb_nn/runner.py",
    "faithful_related_work/hjb_nn/sampling.py",
    "faithful_related_work/hjb_nn/satellite.py",
    "tumor_problem.py",
)
SOURCE_SIGNATURE_KEYS = (
    "repo_revision",
    "upstream_HJB_NN_base_revision",
    "upstream_git_diff_sha256",
    "file_sha256",
    "python",
    "platform",
    "dependency_versions",
)


@dataclass(frozen=True)
class AuditProtocol:
    """The declarations and retained data that define an auditable sweep."""

    taus: tuple[float, ...]
    seeds: tuple[int, ...]
    training: Mapping[str, Any]
    network: Mapping[str, Any]
    optimizer_budget: Mapping[str, Any]
    bvp: Mapping[str, Any]
    evaluation_intervals: int
    adaptive_max_failures: int
    train_trajectories: int
    validation_trajectories: int
    completed_rounds_min: int
    completed_rounds_max: int
    state_box_lower: float
    state_box_upper: float
    sampling_declaration: str
    executed_source_keys: tuple[str, ...]


FINAL_V3_PROTOCOL = AuditProtocol(
    taus=EXPECTED_TAUS,
    seeds=EXPECTED_SEEDS,
    training={
        "width": 96,
        "hidden_layers": 3,
        "max_rounds": 3,
        "min_rounds": 2,
        "maxiter": 5000,
        "convergence_tolerance": 0.5,
        "candidates_per_selection": 32,
        "max_batch_points": 8192,
    },
    network={
        "layers": [22, 96, 96, 96, 1],
        "trainable_parameters": 20_929,
        "input": "time plus 21 tumor-state coordinates",
        "hidden_activation": "tanh",
        "output_activation": "linear",
        "predicted_quantity": "entropy-regularized value V(t,N)",
    },
    optimizer_budget={
        "method": "L-BFGS-B",
        "maxiter_per_round": 5000,
        "maxfun_per_round": 15_000,
        "round_ceiling": 3,
        "maxcor": 15,
        "ftol": 1.0e-11,
        "gtol": 1.0e-6,
        "note": "tumor-adaptation budget; not the original satellite paper budget",
    },
    bvp={
        "tolerance": 1.0e-3,
        "max_nodes": 100_000,
        "adaptive_max_nodes": 20_000,
        "time_march_steps": 16,
        "initial_mesh_nodes": 41,
        "tau_start": 10.0,
        "ode_rtol": 1.0e-6,
        "ode_atol": 1.0e-8,
    },
    evaluation_intervals=200,
    adaptive_max_failures=256,
    train_trajectories=16,
    validation_trajectories=16,
    completed_rounds_min=2,
    completed_rounds_max=3,
    state_box_lower=5.0,
    state_box_upper=20.0,
    sampling_declaration="independent uniform coordinates over [5,20]^21",
    executed_source_keys=EXPECTED_EXECUTED_SOURCE_KEYS,
)


@dataclass(frozen=True)
class NominalTumorProblem:
    """Independent transcription of the canonical parameters in tumor_problem.py."""

    final_time: float = 10.0
    state_dim: int = 21
    umax: float = 3.0
    beta: float = 0.1
    alpha: float = 1.0
    gamma: float = 20.0
    initial_state: float = 10.0
    suppression: float = 0.5

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        phenotype = np.linspace(0.0, 1.0, self.state_dim, dtype=np.float64)
        growth = 2.0 / (1.0 + 3.0 * phenotype**4)
        drug_sensitivity = 1.0 / (1.0 + phenotype**2)
        suppression = np.full(self.state_dim, self.suppression, dtype=np.float64)
        return growth, drug_sensitivity, suppression


@dataclass(frozen=True)
class AuditTolerances:
    integration_rtol: float = 2.0e-12
    integration_atol: float = 2.0e-13
    integration_max_step_fraction: float = 0.125
    saved_cost_rtol: float = 5.0e-10
    saved_cost_atol: float = 5.0e-8
    declaration_rtol: float = 2.0e-13
    declaration_atol: float = 2.0e-11
    breakpoint_state_rtol: float = 5.0e-8
    breakpoint_state_atol: float = 5.0e-9
    entropy_identity_atol: float = 2.0e-9
    grid_atol: float = 1.0e-12
    control_bound_atol: float = 2.0e-12


@dataclass(frozen=True)
class IndependentIntegration:
    breakpoint_states: np.ndarray
    terminal_cost: float
    unregularized_running_cost: float
    regularized_running_cost: float
    entropy_integral: float
    unregularized_J: float
    regularized_J: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is missing or is a symlink: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {label} {path}: {error}") from error
    _require(isinstance(value, dict), f"{label} must contain one JSON object: {path}")
    return value


def _safe_child(root: Path, relative: str, label: str) -> Path:
    _require(isinstance(relative, str) and relative, f"{label} path is empty or non-string")
    raw = Path(relative)
    _require(not raw.is_absolute() and ".." not in raw.parts, f"unsafe {label} path: {relative}")
    resolved_root = root.resolve()
    resolved = (resolved_root / raw).resolve()
    _require(
        resolved == resolved_root or resolved_root in resolved.parents,
        f"{label} escapes aggregate root: {relative}",
    )
    return resolved


def _finite_scalar(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} must be a numeric scalar") from error
    _require(math.isfinite(result), f"{label} must be finite, got {result}")
    return result


def _strict_integer(value: Any, label: str) -> int:
    """Accept an integer JSON value, never a bool, float, or numeric string."""

    _require(
        isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)),
        f"{label} must be a strict integer, got {value!r}",
    )
    return int(value)


def _csv_integer(value: Any, label: str) -> int:
    _require(
        isinstance(value, str) and re.fullmatch(r"0|[1-9][0-9]*", value) is not None,
        f"{label} must be a canonical nonnegative integer, got {value!r}",
    )
    return int(value)


def _require_exact_json(actual: Any, expected: Any, label: str) -> None:
    """Compare JSON-shaped declarations without Python's int/float coercion."""

    if isinstance(expected, dict):
        _require(isinstance(actual, dict), f"{label} must be an object")
        _require(
            set(actual) == set(expected),
            f"{label} keys differ from final-v3 protocol: "
            f"extra={sorted(set(actual) - set(expected))}, "
            f"missing={sorted(set(expected) - set(actual))}",
        )
        for key in expected:
            _require_exact_json(actual[key], expected[key], f"{label}.{key}")
        return
    if isinstance(expected, list):
        _require(isinstance(actual, list), f"{label} must be a list")
        _require(len(actual) == len(expected), f"{label} length mismatch")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _require_exact_json(actual_item, expected_item, f"{label}[{index}]")
        return
    _require(type(actual) is type(expected), f"{label} type mismatch: {type(actual).__name__} != {type(expected).__name__}")
    if isinstance(expected, float):
        _require(math.isfinite(actual), f"{label} must be finite")
    _require(actual == expected, f"{label} mismatch: actual={actual!r}, expected={expected!r}")


def _load_npz(path: Path, label: str) -> dict[str, np.ndarray]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is missing or symlinked: {path}")
    try:
        with np.load(path, allow_pickle=False) as loaded:
            return {key: np.array(loaded[key], copy=True) for key in loaded.files}
    except Exception as error:
        raise RuntimeError(f"cannot safely load {label} {path}: {error}") from error


def _assert_close(
    actual: float,
    expected: float,
    label: str,
    *,
    rtol: float,
    atol: float,
) -> float:
    actual = _finite_scalar(actual, f"actual {label}")
    expected = _finite_scalar(expected, f"expected {label}")
    difference = abs(actual - expected)
    _require(
        difference <= atol + rtol * abs(expected),
        f"{label} mismatch: actual={actual:.17g}, expected={expected:.17g}, "
        f"abs_diff={difference:.6g}, rtol={rtol:g}, atol={atol:g}",
    )
    return difference


def _npz_scalar(data: Mapping[str, np.ndarray], key: str) -> float:
    _require(key in data, f"canonical_control.npz is missing {key}")
    array = np.asarray(data[key])
    _require(array.size == 1, f"canonical_control.npz field {key} must be scalar, got {array.shape}")
    return _finite_scalar(array.reshape(-1)[0], f"canonical_control.npz::{key}")


def _npz_string(data: Mapping[str, np.ndarray], key: str) -> str:
    _require(key in data, f"canonical_control.npz is missing {key}")
    array = np.asarray(data[key])
    _require(array.size == 1 and array.dtype.kind in "US", f"{key} must be one Unicode/byte string")
    value = array.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def entropy_density(action: float, tau: float, problem: NominalTumorProblem) -> float:
    """Return tau*U*(p log p + (1-p) log(1-p)), with 0 log 0 = 0."""

    _require(math.isfinite(action), "control is nonfinite")
    _require(0.0 <= action <= problem.umax, f"control {action} is outside [0,{problem.umax}]")
    p = action / problem.umax
    p_term = 0.0 if p == 0.0 else p * math.log(p)
    q = 1.0 - p
    q_term = 0.0 if q == 0.0 else q * math.log(q)
    return tau * problem.umax * (p_term + q_term)


def integrate_zoh_independent(
    breakpoints: Sequence[float] | np.ndarray,
    controls: Sequence[float] | np.ndarray,
    tau: float,
    *,
    problem: NominalTumorProblem = NominalTumorProblem(),
    tolerances: AuditTolerances = AuditTolerances(),
) -> IndependentIntegration:
    """Integrate state plus two running costs, one DOP853 solve per ZOH interval."""

    time = np.asarray(breakpoints, dtype=np.float64).reshape(-1)
    control = np.asarray(controls, dtype=np.float64).reshape(-1)
    _require(time.size >= 2, "breakpoint grid needs at least two points")
    _require(control.size == time.size - 1, "left-endpoint ZOH requires len(u)=len(t)-1")
    _require(np.all(np.isfinite(time)), "breakpoint grid contains nonfinite values")
    _require(np.all(np.diff(time) > 0.0), "breakpoint grid must be strictly increasing")
    _require(abs(float(time[0])) <= tolerances.grid_atol, "breakpoint grid does not start at zero")
    _require(
        abs(float(time[-1]) - problem.final_time) <= tolerances.grid_atol,
        f"breakpoint grid does not end at {problem.final_time}",
    )
    _require(math.isfinite(tau) and tau > 0.0, f"tau must be finite and positive, got {tau}")
    _require(np.all(np.isfinite(control)), "control contains nonfinite values")
    _require(
        np.min(control) >= -tolerances.control_bound_atol
        and np.max(control) <= problem.umax + tolerances.control_bound_atol,
        f"control violates [0,{problem.umax}] bounds: min={control.min()}, max={control.max()}",
    )
    # Values inside only a roundoff-width halo are snapped to their declared
    # bound; larger violations are rejected above, never silently clipped.
    control = np.where(np.abs(control) <= tolerances.control_bound_atol, 0.0, control)
    control = np.where(
        np.abs(control - problem.umax) <= tolerances.control_bound_atol,
        problem.umax,
        control,
    )

    growth, drug_sensitivity, suppression = problem.arrays()
    # y = (N_1,...,N_21, accumulated unregularized running cost,
    #      accumulated entropy-regularized running cost)
    augmented = np.concatenate(
        (np.full(problem.state_dim, problem.initial_state, dtype=np.float64), np.zeros(2))
    )
    breakpoint_states = [augmented[: problem.state_dim].copy()]
    exact_entropy_integral = 0.0

    for index, action_value in enumerate(control):
        action = float(action_value)
        left = float(time[index])
        right = float(time[index + 1])
        entropy = entropy_density(action, tau, problem)
        exact_entropy_integral += entropy * (right - left)

        def rhs(_time: float, value: np.ndarray) -> np.ndarray:
            state = value[: problem.state_dim]
            if not np.all(np.isfinite(state)) or np.any(state <= 0.0):
                raise FloatingPointError("independent ZOH trajectory left the positive orthant")
            crowding = math.log1p(float(np.mean(state)))
            state_rhs = (growth - drug_sensitivity * action - suppression * crowding) * state
            base_running = problem.beta * float(np.sum(state)) + problem.gamma * action
            return np.concatenate(
                (state_rhs, np.array([base_running, base_running + entropy], dtype=np.float64))
            )

        solution = solve_ivp(
            rhs,
            (left, right),
            augmented,
            method="DOP853",
            rtol=tolerances.integration_rtol,
            atol=tolerances.integration_atol,
            max_step=(right - left) * tolerances.integration_max_step_fraction,
        )
        _require(solution.success, f"DOP853 failed on interval {index}: {solution.message}")
        augmented = np.asarray(solution.y[:, -1], dtype=np.float64)
        _require(np.all(np.isfinite(augmented)), f"nonfinite DOP853 result on interval {index}")
        _require(
            np.all(augmented[: problem.state_dim] > 0.0),
            f"nonpositive breakpoint state after interval {index}",
        )
        breakpoint_states.append(augmented[: problem.state_dim].copy())

    terminal = problem.alpha * float(np.sum(augmented[: problem.state_dim]))
    unregularized_running = float(augmented[-2])
    regularized_running = float(augmented[-1])
    numerical_entropy = regularized_running - unregularized_running
    _assert_close(
        numerical_entropy,
        exact_entropy_integral,
        "integrated entropy identity",
        rtol=0.0,
        atol=tolerances.entropy_identity_atol,
    )
    return IndependentIntegration(
        breakpoint_states=np.asarray(breakpoint_states, dtype=np.float64).T,
        terminal_cost=terminal,
        unregularized_running_cost=unregularized_running,
        regularized_running_cost=regularized_running,
        entropy_integral=exact_entropy_integral,
        unregularized_J=terminal + unregularized_running,
        regularized_J=terminal + regularized_running,
    )


def _verify_worker_artifacts(run_dir: Path, worker: Mapping[str, Any]) -> tuple[int, dict[str, str]]:
    declared = worker.get("artifact_sha256")
    _require(isinstance(declared, dict) and declared, f"worker artifact_sha256 missing: {run_dir}")
    normalized: dict[str, str] = {}
    for relative, expected in declared.items():
        _require(
            isinstance(expected, str) and len(expected) == 64,
            f"invalid declared SHA-256 for {relative} in {run_dir}",
        )
        artifact = _safe_child(run_dir, relative, "worker artifact")
        _require(artifact.is_file() and not artifact.is_symlink(), f"worker artifact missing/symlink: {artifact}")
        actual = sha256_file(artifact)
        _require(actual == expected, f"worker artifact SHA-256 mismatch: {artifact}")
        normalized[str(Path(relative))] = actual

    actual_paths: set[str] = set()
    for path in run_dir.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"symlink is forbidden inside worker artifact tree: {path}")
        if path.is_file() and path.name != "manifest.json":
            actual_paths.add(str(path.relative_to(run_dir)))
    _require(
        actual_paths == set(normalized),
        f"worker artifact set differs from manifest at {run_dir}: "
        f"undeclared={sorted(actual_paths - set(normalized))}, missing={sorted(set(normalized) - actual_paths)}",
    )
    return len(normalized), normalized


def _source_signature(provenance: Any, label: str) -> dict[str, Any]:
    _require(isinstance(provenance, dict), f"{label} provenance is missing")
    signature = {key: provenance.get(key) for key in SOURCE_SIGNATURE_KEYS}
    _require(all(value is not None for value in signature.values()), f"{label} source signature is incomplete")
    return signature


def _verify_source_snapshots(
    run_dir: Path,
    worker: Mapping[str, Any],
    artifact_hashes: Mapping[str, str],
    expected_source_keys: Sequence[str],
) -> int:
    mapping = worker.get("executed_source_snapshot")
    provenance = worker.get("provenance")
    _require(isinstance(mapping, dict) and mapping, f"executed source snapshot mapping missing: {run_dir}")
    _require(isinstance(provenance, dict), f"worker provenance missing: {run_dir}")
    source_hashes = provenance.get("file_sha256")
    _require(isinstance(source_hashes, dict) and source_hashes, f"worker source hashes missing: {run_dir}")
    expected_keys = set(expected_source_keys)
    _require(
        set(mapping) == expected_keys,
        f"executed source snapshot keys differ from the expected protocol at {run_dir}: "
        f"extra={sorted(set(mapping) - expected_keys)}, "
        f"missing={sorted(expected_keys - set(mapping))}",
    )
    for original, snapshot in mapping.items():
        _require(original in source_hashes, f"snapshot source has no provenance hash: {original}")
        _require(
            snapshot == f"executed_source_snapshot/{original}",
            f"unexpected executed source snapshot destination for {original}: {snapshot}",
        )
        snapshot_path = _safe_child(run_dir, snapshot, "executed source snapshot")
        relative = str(snapshot_path.relative_to(run_dir))
        _require(relative in artifact_hashes, f"source snapshot is not a declared worker artifact: {relative}")
        _require(
            artifact_hashes[relative] == source_hashes[original],
            f"source snapshot differs from declared executed source: {original}",
        )
    return len(mapping)


def _state_keys(states: np.ndarray) -> set[bytes]:
    states = np.asarray(states, dtype=np.float64)
    _require(states.ndim == 2, f"initial states must be a matrix, got {states.shape}")
    return {np.ascontiguousarray(states[:, index]).tobytes() for index in range(states.shape[1])}


def _expected_initial_state_split(
    seed: int,
    protocol: AuditProtocol,
    state_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full((state_dim, 1), protocol.state_box_lower, dtype=np.float64)
    upper = np.full((state_dim, 1), protocol.state_box_upper, dtype=np.float64)
    train = np.random.default_rng(seed).uniform(
        lower,
        upper,
        size=(state_dim, protocol.train_trajectories),
    )
    validation = np.random.default_rng(np.random.SeedSequence([seed, 1])).uniform(
        lower,
        upper,
        size=(state_dim, protocol.validation_trajectories),
    )
    return train, validation


def _dataset_initial_states(
    path: Path,
    *,
    label: str,
    expected_count: int,
    expected_states: np.ndarray,
    problem: NominalTumorProblem,
    protocol: AuditProtocol,
) -> tuple[np.ndarray, int]:
    data = _load_npz(path, label)
    for key in ("t", "X", "trajectory_id", "initial_states"):
        _require(key in data, f"{label} is missing {key}: {path}")
    time = np.asarray(data["t"], dtype=np.float64).reshape(-1)
    states = np.asarray(data["X"], dtype=np.float64)
    initial = np.asarray(data["initial_states"], dtype=np.float64)
    trajectory_id = np.asarray(data["trajectory_id"]).reshape(-1)
    _require(states.shape == (problem.state_dim, time.size), f"{label} X/t shape mismatch: {path}")
    _require(trajectory_id.size == time.size, f"{label} trajectory_id/t shape mismatch: {path}")
    _require(initial.shape == (problem.state_dim, expected_count), f"{label} initial_states shape mismatch: {path}")
    _require(
        np.all(np.isfinite(time)) and np.all(np.isfinite(states)) and np.all(np.isfinite(initial)),
        f"{label} contains nonfinite time/state values: {path}",
    )
    _require(
        np.min(initial) >= protocol.state_box_lower
        and np.max(initial) <= protocol.state_box_upper,
        f"{label} initial_states leave the declared sampling box: {path}",
    )
    start_indices = np.flatnonzero(np.isclose(time, 0.0, rtol=0.0, atol=1.0e-12))
    _require(start_indices.size == expected_count, f"{label} has {start_indices.size} t=0 starts, expected {expected_count}: {path}")
    _require(
        np.array_equal(states[:, start_indices], initial),
        f"{label} initial_states do not equal its t=0 X columns: {path}",
    )
    _require(
        np.unique(trajectory_id).size == expected_count,
        f"{label} trajectory_id count differs from declared trajectories: {path}",
    )
    _require(
        np.array_equal(initial, expected_states),
        f"{label} initial_states are not the seed-reconstructable independent coordinate-uniform samples: {path}",
    )
    return initial, int(time.size)


def _verified_bvp_metadata(
    record: Any,
    *,
    label: str,
    tau: float,
    tolerance: float,
) -> tuple[float, float]:
    _require(isinstance(record, dict), f"{label} must be an object")
    _require(record.get("success") is True, f"{label} is not successful")
    _assert_close(record.get("target_tau"), tau, f"{label} target_tau", rtol=0.0, atol=0.0)
    rms = _finite_scalar(record.get("max_rms_residual"), f"{label} max_rms_residual")
    _require(rms >= 0.0 and rms <= tolerance, f"{label} max_rms_residual={rms} exceeds {tolerance}")
    boundary = _finite_scalar(
        record.get("boundary_residual_max_abs"),
        f"{label} boundary_residual_max_abs",
    )
    _require(boundary >= 0.0, f"{label} boundary residual must be nonnegative")
    return rms, boundary


def _verify_bvp_histories(
    run_dir: Path,
    worker: Mapping[str, Any],
    *,
    tau: float,
    protocol: AuditProtocol,
) -> tuple[float, float]:
    initial_path = run_dir / "bvp_initial_history.json"
    history_path = run_dir / "history.json"
    initial = _read_json_object(initial_path, "initial BVP history")
    history = _read_json_object(history_path, "training history")
    tolerance = _finite_scalar(protocol.bvp["tolerance"], "protocol BVP tolerance")
    max_rms = 0.0
    max_boundary = 0.0
    for split, expected_count in (
        ("train", protocol.train_trajectories),
        ("validation", protocol.validation_trajectories),
    ):
        records = initial.get(split)
        _require(isinstance(records, list) and len(records) == expected_count, f"initial BVP {split} must contain {expected_count} records")
        for index, record in enumerate(records):
            rms, boundary = _verified_bvp_metadata(
                record,
                label=f"initial BVP {split}[{index}]",
                tau=tau,
                tolerance=tolerance,
            )
            max_rms = max(max_rms, rms)
            max_boundary = max(max_boundary, boundary)

    events = history.get("adaptive_events")
    _require(isinstance(events, list), f"adaptive_events missing: {history_path}")
    declared_events = _strict_integer(worker.get("adaptive_events"), "worker adaptive_events")
    declared_successes = _strict_integer(worker.get("adaptive_successes"), "worker adaptive_successes")
    declared_failures = _strict_integer(worker.get("adaptive_failures"), "worker adaptive_failures")
    successes = sum(event.get("success") is True for event in events if isinstance(event, dict))
    failures = sum(event.get("success") is False for event in events if isinstance(event, dict))
    _require(all(isinstance(event, dict) for event in events), f"adaptive event is not an object: {history_path}")
    _require(successes + failures == len(events), f"adaptive event success flags are not strict booleans: {history_path}")
    _require(len(events) == declared_events, f"adaptive event count disagrees with worker manifest: {history_path}")
    _require(successes == declared_successes, f"adaptive success count disagrees with worker manifest: {history_path}")
    _require(failures == declared_failures, f"adaptive failure count disagrees with worker manifest: {history_path}")
    _require(
        declared_failures <= protocol.adaptive_max_failures,
        f"adaptive failures exceed the declared failure budget: {history_path}",
    )
    for index, event in enumerate(events):
        if event["success"] is True:
            rms, boundary = _verified_bvp_metadata(
                event.get("bvp"),
                label=f"adaptive event[{index}].bvp",
                tau=tau,
                tolerance=tolerance,
            )
            max_rms = max(max_rms, rms)
            max_boundary = max(max_boundary, boundary)

    completed_rounds = _strict_integer(worker.get("completed_rounds"), "worker completed_rounds")
    for field in ("round_iters", "convergence_tests", "optimizer_rounds"):
        values = history.get(field)
        _require(
            isinstance(values, list) and len(values) == completed_rounds,
            f"history {field} count disagrees with completed_rounds: {history_path}",
        )
    return max_rms, max_boundary


def _check_summary_csv(summary_path: Path, aggregate_runs: Mapping[tuple[float, int], Mapping[str, Any]], tolerances: AuditTolerances) -> str:
    _require(summary_path.is_file() and not summary_path.is_symlink(), f"aggregate summary.csv missing: {summary_path}")
    with summary_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    _require(len(rows) == 9, f"aggregate summary.csv must contain 9 rows, found {len(rows)}")
    csv_by_key: dict[tuple[float, int], dict[str, str]] = {}
    for row in rows:
        key = (
            _finite_scalar(row.get("tau"), "summary.csv tau"),
            _csv_integer(row.get("seed"), "summary.csv seed"),
        )
        _require(key not in csv_by_key, f"duplicate summary.csv row {key}")
        csv_by_key[key] = row
    _require(set(csv_by_key) == set(aggregate_runs), "summary.csv grid differs from aggregate manifest")
    for key, aggregate_row in aggregate_runs.items():
        csv_row = csv_by_key[key]
        _require(csv_row.get("run_dir") == aggregate_row.get("run_dir"), f"summary.csv run_dir mismatch for {key}")
        for metric in (
            "regularized_native_value_prediction",
            "regularized_realized_J",
            "unregularized_realized_J",
        ):
            _require(metric in aggregate_row and metric in csv_row, f"summary.csv/aggregate missing {metric} for {key}")
            _assert_close(
                _finite_scalar(csv_row[metric], f"summary.csv {metric}"),
                _finite_scalar(aggregate_row[metric], f"aggregate {metric}"),
                f"summary.csv {metric} for {key}",
                rtol=tolerances.declaration_rtol,
                atol=tolerances.declaration_atol,
            )
    return sha256_file(summary_path)


def _refuse_nonempty_output(output_dir: Path) -> None:
    if output_dir.exists():
        _require(output_dir.is_dir() and not output_dir.is_symlink(), f"output path is not a directory: {output_dir}")
        _require(not any(output_dir.iterdir()), f"refusing nonempty output directory: {output_dir}")


def audit_aggregate(
    aggregate_manifest: Path,
    output_dir: Path,
    *,
    problem: NominalTumorProblem = NominalTumorProblem(),
    tolerances: AuditTolerances = AuditTolerances(),
    expected_protocol: AuditProtocol = FINAL_V3_PROTOCOL,
) -> dict[str, Any]:
    """Audit one exact sweep and publish only after every check passes.

    The CLI always uses :data:`FINAL_V3_PROTOCOL`.  ``expected_protocol`` is
    exposed solely so lightweight, explicit synthetic fixtures can exercise
    the auditor without pretending to be a final-v3 run.
    """

    aggregate_manifest = aggregate_manifest.resolve()
    output_dir = output_dir.resolve()
    _refuse_nonempty_output(output_dir)
    aggregate = _read_json_object(aggregate_manifest, "aggregate manifest")
    aggregate_root = aggregate_manifest.parent
    _require(aggregate.get("status") == "completed", "aggregate status is not completed")
    _require(aggregate.get("worker_source_and_dependencies_uniform") is True, "aggregate does not declare uniform worker sources/dependencies")

    taus = aggregate.get("taus")
    seeds = aggregate.get("seeds")
    _require(isinstance(taus, list) and len(taus) == len(expected_protocol.taus), "aggregate tau count differs from expected protocol")
    _require(isinstance(seeds, list) and len(seeds) == len(expected_protocol.seeds), "aggregate seed count differs from expected protocol")
    tau_values = tuple(_finite_scalar(value, "aggregate tau") for value in taus)
    seed_values = tuple(_strict_integer(value, "aggregate seed") for value in seeds)
    _require(len(set(tau_values)) == len(expected_protocol.taus) and set(tau_values) == set(expected_protocol.taus), f"expected tau grid {expected_protocol.taus}, got {tau_values}")
    _require(len(set(seed_values)) == len(expected_protocol.seeds) and set(seed_values) == set(expected_protocol.seeds), f"expected seed grid {expected_protocol.seeds}, got {seed_values}")
    expected_grid = {
        (tau, seed)
        for tau in expected_protocol.taus
        for seed in expected_protocol.seeds
    }

    _require_exact_json(aggregate.get("training"), dict(expected_protocol.training), "aggregate.training")
    _require_exact_json(aggregate.get("bvp"), dict(expected_protocol.bvp), "aggregate.bvp")
    _require_exact_json(
        aggregate.get("optimizer_budget"),
        dict(expected_protocol.optimizer_budget),
        "aggregate.optimizer_budget",
    )
    _require(
        _strict_integer(aggregate.get("evaluation_intervals"), "aggregate evaluation_intervals")
        == expected_protocol.evaluation_intervals,
        "aggregate evaluation_intervals differs from expected protocol",
    )
    _require(
        _strict_integer(aggregate.get("adaptive_max_failures"), "aggregate adaptive_max_failures")
        == expected_protocol.adaptive_max_failures,
        "aggregate adaptive_max_failures differs from expected protocol",
    )

    runs = aggregate.get("runs")
    _require(isinstance(runs, list) and len(runs) == len(expected_grid), f"aggregate must contain exactly {len(expected_grid)} runs")
    aggregate_runs: dict[tuple[float, int], Mapping[str, Any]] = {}
    for run in runs:
        _require(isinstance(run, dict), "aggregate run entry is not an object")
        key = (
            _finite_scalar(run.get("tau"), "aggregate run tau"),
            _strict_integer(run.get("seed"), "aggregate run seed"),
        )
        _require(key not in aggregate_runs, f"duplicate aggregate run {key}")
        aggregate_runs[key] = run
    _require(set(aggregate_runs) == expected_grid, "aggregate run entries do not form the declared 3x3 grid")

    summary_path = aggregate_root / "summary.csv"
    aggregate_summary_sha256 = _check_summary_csv(summary_path, aggregate_runs, tolerances)

    expected_source_signature_hash = aggregate.get("worker_source_signature_sha256")
    _require(
        isinstance(expected_source_signature_hash, str) and len(expected_source_signature_hash) == 64,
        "aggregate worker_source_signature_sha256 is missing/invalid",
    )
    aggregate_signature = _source_signature(aggregate.get("provenance"), "aggregate")
    shared_worker_signature: dict[str, Any] | None = None
    output_rows: list[dict[str, Any]] = []
    max_discrepancies = {
        "canonical_unregularized_J_abs": 0.0,
        "canonical_regularized_J_abs": 0.0,
        "worker_unregularized_J_abs": 0.0,
        "worker_regularized_J_abs": 0.0,
        "aggregate_unregularized_J_abs": 0.0,
        "aggregate_regularized_J_abs": 0.0,
        "breakpoint_state_abs": 0.0,
        "breakpoint_state_relative": 0.0,
        "entropy_identity_abs": 0.0,
    }
    total_artifacts_hashed = 0
    total_source_snapshots_checked = 0
    initial_splits_by_seed: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for tau, seed in sorted(expected_grid, key=lambda item: (-item[0], item[1])):
        aggregate_run = aggregate_runs[(tau, seed)]
        run_dir = _safe_child(aggregate_root, aggregate_run.get("run_dir"), "aggregate run_dir")
        _require(run_dir.is_dir() and not run_dir.is_symlink(), f"worker directory missing/symlink: {run_dir}")
        worker_path = run_dir / "manifest.json"
        worker = _read_json_object(worker_path, "worker manifest")
        declared_worker_hash = aggregate_run.get("worker_manifest_sha256")
        _require(isinstance(declared_worker_hash, str) and len(declared_worker_hash) == 64, f"aggregate worker manifest hash missing for {(tau, seed)}")
        actual_worker_hash = sha256_file(worker_path)
        _require(actual_worker_hash == declared_worker_hash, f"worker manifest SHA-256 mismatch: {worker_path}")
        _assert_close(worker.get("tau"), tau, "worker tau", rtol=0.0, atol=0.0)
        _require(_strict_integer(worker.get("seed"), "worker seed") == seed, f"worker seed mismatch: {worker_path}")
        _require(worker.get("regularized_training_objective") is True, f"worker does not declare regularized training: {worker_path}")
        _require(worker.get("unregularized_common_objective_used_for_training_or_selection") is False, f"worker used common J for selection/training: {worker_path}")
        _require(worker.get("all_candidates_retained") is True, f"worker does not retain all candidates: {worker_path}")
        _require(worker.get("trajectory_disjoint_validation") is True, f"worker does not declare disjoint trajectory validation: {worker_path}")
        _require(
            worker.get("initial_state_sampling") == expected_protocol.sampling_declaration,
            f"worker sampling declaration differs from expected protocol: {worker_path}",
        )
        _require_exact_json(worker.get("training"), dict(expected_protocol.training), "worker.training")
        _require_exact_json(worker.get("network"), dict(expected_protocol.network), "worker.network")
        _require_exact_json(worker.get("bvp"), dict(expected_protocol.bvp), "worker.bvp")
        _require_exact_json(
            worker.get("optimizer_budget"),
            dict(expected_protocol.optimizer_budget),
            "worker.optimizer_budget",
        )
        intervals = _strict_integer(worker.get("evaluation_intervals"), "worker evaluation_intervals")
        _require(intervals == expected_protocol.evaluation_intervals, f"worker evaluation_intervals differs from expected protocol: {worker_path}")
        _require(
            _strict_integer(worker.get("adaptive_max_failures"), "worker adaptive_max_failures")
            == expected_protocol.adaptive_max_failures,
            f"worker adaptive_max_failures differs from expected protocol: {worker_path}",
        )
        _require(
            _strict_integer(worker.get("train_trajectories_initial"), "worker train_trajectories_initial")
            == expected_protocol.train_trajectories,
            f"worker train trajectory count differs from expected protocol: {worker_path}",
        )
        _require(
            _strict_integer(worker.get("validation_trajectories"), "worker validation_trajectories")
            == expected_protocol.validation_trajectories,
            f"worker validation trajectory count differs from expected protocol: {worker_path}",
        )
        completed_rounds = _strict_integer(worker.get("completed_rounds"), "worker completed_rounds")
        _require(
            expected_protocol.completed_rounds_min
            <= completed_rounds
            <= expected_protocol.completed_rounds_max,
            f"worker completed_rounds={completed_rounds} is outside expected protocol range "
            f"[{expected_protocol.completed_rounds_min},{expected_protocol.completed_rounds_max}]: {worker_path}",
        )

        worker_signature = _source_signature(worker.get("provenance"), f"worker {(tau, seed)}")
        if shared_worker_signature is None:
            shared_worker_signature = worker_signature
        _require(worker_signature == shared_worker_signature, f"worker source/dependency signature mismatch: {worker_path}")
        _require(worker_signature == aggregate_signature, f"worker and aggregate source signatures differ: {worker_path}")

        artifact_count, artifact_hashes = _verify_worker_artifacts(run_dir, worker)
        total_artifacts_hashed += artifact_count
        total_source_snapshots_checked += _verify_source_snapshots(
            run_dir,
            worker,
            artifact_hashes,
            expected_protocol.executed_source_keys,
        )
        _require("canonical_control.npz" in artifact_hashes, f"canonical control is not hash-declared: {run_dir}")
        for required_artifact in (
            "dataset_train_initial.npz",
            "dataset_validation.npz",
            "dataset_train_final.npz",
            "bvp_initial_history.json",
            "history.json",
        ):
            _require(required_artifact in artifact_hashes, f"required final-v3 artifact is not hash-declared: {run_dir / required_artifact}")

        expected_train, expected_validation = _expected_initial_state_split(
            seed,
            expected_protocol,
            problem.state_dim,
        )
        train_initial, initial_training_points = _dataset_initial_states(
            run_dir / "dataset_train_initial.npz",
            label="initial training dataset",
            expected_count=expected_protocol.train_trajectories,
            expected_states=expected_train,
            problem=problem,
            protocol=expected_protocol,
        )
        validation_initial, validation_points = _dataset_initial_states(
            run_dir / "dataset_validation.npz",
            label="validation dataset",
            expected_count=expected_protocol.validation_trajectories,
            expected_states=expected_validation,
            problem=problem,
            protocol=expected_protocol,
        )
        _require(
            not (_state_keys(train_initial) & _state_keys(validation_initial)),
            f"training and validation initial states overlap: {run_dir}",
        )
        if seed in initial_splits_by_seed:
            prior_train, prior_validation = initial_splits_by_seed[seed]
            _require(
                np.array_equal(train_initial, prior_train)
                and np.array_equal(validation_initial, prior_validation),
                f"initial-state split changed across tau for seed {seed}: {run_dir}",
            )
        else:
            initial_splits_by_seed[seed] = (train_initial.copy(), validation_initial.copy())

        _require(
            _strict_integer(worker.get("initial_training_points"), "worker initial_training_points")
            == initial_training_points,
            f"initial training point count disagrees with retained dataset: {worker_path}",
        )
        _require(
            _strict_integer(worker.get("validation_points"), "worker validation_points")
            == validation_points,
            f"validation point count disagrees with retained dataset: {worker_path}",
        )
        final_dataset = _load_npz(run_dir / "dataset_train_final.npz", "final training dataset")
        _require("X" in final_dataset, f"final training dataset is missing X: {run_dir}")
        final_X = np.asarray(final_dataset["X"], dtype=np.float64)
        _require(final_X.ndim == 2 and final_X.shape[0] == problem.state_dim and np.all(np.isfinite(final_X)), f"invalid final training X: {run_dir}")
        final_training_points = _strict_integer(worker.get("final_training_points"), "worker final_training_points")
        _require(final_X.shape[1] == final_training_points, f"final training point count disagrees with retained dataset: {worker_path}")
        max_bvp_rms, max_boundary_residual = _verify_bvp_histories(
            run_dir,
            worker,
            tau=tau,
            protocol=expected_protocol,
        )
        canonical_path = run_dir / "canonical_control.npz"

        data = _load_npz(canonical_path, "canonical control")
        time = np.asarray(data.get("t"), dtype=np.float64).reshape(-1)
        control = np.asarray(data.get("u"), dtype=np.float64).reshape(-1)
        npz_tau = _npz_scalar(data, "tau")
        _assert_close(npz_tau, tau, "canonical tau", rtol=0.0, atol=0.0)
        semantics = _npz_string(data, "control_semantics").lower()
        _require("left-endpoint" in semantics and "zoh" in semantics, f"canonical control does not declare left-endpoint ZOH semantics: {canonical_path}")
        _require(time.size == intervals + 1 and control.size == intervals, f"canonical grid size disagrees with evaluation_intervals: {canonical_path}")
        expected_time = np.linspace(0.0, problem.final_time, intervals + 1, dtype=np.float64)
        _require(
            np.array_equal(time, expected_time),
            f"canonical breakpoints are not the exact uniform linspace({0.0},{problem.final_time},{intervals + 1}): {canonical_path}",
        )

        integrated = integrate_zoh_independent(time, control, tau, problem=problem, tolerances=tolerances)
        npz_native = _npz_scalar(data, "J_regularized_native_value_prediction")
        npz_unregularized = _npz_scalar(data, "J_unregularized_realized")
        npz_regularized = _npz_scalar(data, "J_regularized_realized")
        canonical_unreg_diff = _assert_close(
            npz_unregularized,
            integrated.unregularized_J,
            f"canonical unregularized J for {(tau, seed)}",
            rtol=tolerances.saved_cost_rtol,
            atol=tolerances.saved_cost_atol,
        )
        canonical_reg_diff = _assert_close(
            npz_regularized,
            integrated.regularized_J,
            f"canonical regularized J for {(tau, seed)}",
            rtol=tolerances.saved_cost_rtol,
            atol=tolerances.saved_cost_atol,
        )

        component_fields = {
            "regularized_terminal_cost": integrated.terminal_cost,
            "regularized_running_cost": integrated.regularized_running_cost,
            "regularized_entropy_integral": integrated.entropy_integral,
        }
        for field, audited_value in component_fields.items():
            _assert_close(
                _npz_scalar(data, field),
                audited_value,
                f"canonical {field} for {(tau, seed)}",
                rtol=tolerances.saved_cost_rtol,
                atol=tolerances.saved_cost_atol,
            )

        worker_metrics = worker.get("metrics")
        _require(isinstance(worker_metrics, dict), f"worker metrics missing: {worker_path}")
        worker_native = _finite_scalar(
            worker_metrics.get("regularized_native_value_prediction"),
            f"worker native value for {(tau, seed)}",
        )
        aggregate_native = _finite_scalar(
            aggregate_run.get("regularized_native_value_prediction"),
            f"aggregate native value for {(tau, seed)}",
        )
        _assert_close(
            npz_native,
            worker_native,
            f"canonical/worker native value declaration for {(tau, seed)}",
            rtol=tolerances.declaration_rtol,
            atol=tolerances.declaration_atol,
        )
        _assert_close(
            npz_native,
            aggregate_native,
            f"canonical/aggregate native value declaration for {(tau, seed)}",
            rtol=tolerances.declaration_rtol,
            atol=tolerances.declaration_atol,
        )
        worker_unreg_diff = _assert_close(
            worker_metrics.get("unregularized_realized_J"),
            integrated.unregularized_J,
            f"worker unregularized J for {(tau, seed)}",
            rtol=tolerances.saved_cost_rtol,
            atol=tolerances.saved_cost_atol,
        )
        worker_reg_diff = _assert_close(
            worker_metrics.get("regularized_realized_J"),
            integrated.regularized_J,
            f"worker regularized J for {(tau, seed)}",
            rtol=tolerances.saved_cost_rtol,
            atol=tolerances.saved_cost_atol,
        )
        aggregate_unreg_diff = _assert_close(
            aggregate_run.get("unregularized_realized_J"),
            integrated.unregularized_J,
            f"aggregate unregularized J for {(tau, seed)}",
            rtol=tolerances.saved_cost_rtol,
            atol=tolerances.saved_cost_atol,
        )
        aggregate_reg_diff = _assert_close(
            aggregate_run.get("regularized_realized_J"),
            integrated.regularized_J,
            f"aggregate regularized J for {(tau, seed)}",
            rtol=tolerances.saved_cost_rtol,
            atol=tolerances.saved_cost_atol,
        )
        # Declarations should also agree with one another much more tightly
        # than their independently recomputed integration tolerance.
        _assert_close(
            npz_unregularized,
            _finite_scalar(worker_metrics.get("unregularized_realized_J"), "worker J"),
            f"canonical/worker unregularized declaration for {(tau, seed)}",
            rtol=tolerances.declaration_rtol,
            atol=tolerances.declaration_atol,
        )
        _assert_close(
            npz_regularized,
            _finite_scalar(worker_metrics.get("regularized_realized_J"), "worker J_tau"),
            f"canonical/worker regularized declaration for {(tau, seed)}",
            rtol=tolerances.declaration_rtol,
            atol=tolerances.declaration_atol,
        )
        _assert_close(
            npz_unregularized,
            _finite_scalar(aggregate_run.get("unregularized_realized_J"), "aggregate J"),
            f"canonical/aggregate unregularized declaration for {(tau, seed)}",
            rtol=tolerances.declaration_rtol,
            atol=tolerances.declaration_atol,
        )
        _assert_close(
            npz_regularized,
            _finite_scalar(aggregate_run.get("regularized_realized_J"), "aggregate J_tau"),
            f"canonical/aggregate regularized declaration for {(tau, seed)}",
            rtol=tolerances.declaration_rtol,
            atol=tolerances.declaration_atol,
        )

        _require("feedback_rollout_N" in data, f"canonical_control.npz is missing feedback_rollout_N: {canonical_path}")
        saved_states = np.asarray(data["feedback_rollout_N"], dtype=np.float64)
        _require(saved_states.shape == integrated.breakpoint_states.shape, f"feedback_rollout_N shape mismatch: {canonical_path}")
        _require(np.all(np.isfinite(saved_states)) and np.all(saved_states > 0.0), f"feedback_rollout_N contains invalid states: {canonical_path}")
        _require(
            np.allclose(
                saved_states,
                integrated.breakpoint_states,
                rtol=tolerances.breakpoint_state_rtol,
                atol=tolerances.breakpoint_state_atol,
            ),
            f"saved breakpoint states disagree with independent DOP853 rollout: {canonical_path}",
        )
        state_difference = np.abs(saved_states - integrated.breakpoint_states)
        state_abs = float(np.max(state_difference))
        state_relative = float(
            np.max(state_difference / np.maximum(np.abs(integrated.breakpoint_states), 1.0e-300))
        )

        entropy_identity_diff = abs(
            (integrated.regularized_J - integrated.unregularized_J) - integrated.entropy_integral
        )
        _require(entropy_identity_diff <= tolerances.entropy_identity_atol, f"J_tau-J entropy identity failed for {(tau, seed)}")
        discrepancies = {
            "canonical_unregularized_J_abs": canonical_unreg_diff,
            "canonical_regularized_J_abs": canonical_reg_diff,
            "worker_unregularized_J_abs": worker_unreg_diff,
            "worker_regularized_J_abs": worker_reg_diff,
            "aggregate_unregularized_J_abs": aggregate_unreg_diff,
            "aggregate_regularized_J_abs": aggregate_reg_diff,
            "breakpoint_state_abs": state_abs,
            "breakpoint_state_relative": state_relative,
            "entropy_identity_abs": entropy_identity_diff,
        }
        for key, value in discrepancies.items():
            max_discrepancies[key] = max(max_discrepancies[key], value)

        output_rows.append(
            {
                "tau": tau,
                "seed": seed,
                "run_dir": str(run_dir.relative_to(aggregate_root)),
                "intervals": control.size,
                "completed_rounds": completed_rounds,
                "initial_training_points": initial_training_points,
                "final_training_points": final_training_points,
                "adaptive_successes": _strict_integer(worker.get("adaptive_successes"), "worker adaptive_successes"),
                "adaptive_failures": _strict_integer(worker.get("adaptive_failures"), "worker adaptive_failures"),
                "max_accepted_bvp_rms_residual": max_bvp_rms,
                "max_boundary_residual_abs": max_boundary_residual,
                "control_min": float(np.min(control)),
                "control_max": float(np.max(control)),
                "audited_terminal_cost": integrated.terminal_cost,
                "audited_unregularized_running_cost": integrated.unregularized_running_cost,
                "audited_entropy_integral": integrated.entropy_integral,
                "audited_regularized_running_cost": integrated.regularized_running_cost,
                "audited_unregularized_J": integrated.unregularized_J,
                "audited_regularized_J": integrated.regularized_J,
                "saved_regularized_native_value_prediction": npz_native,
                "saved_unregularized_J": npz_unregularized,
                "saved_regularized_J": npz_regularized,
                **{f"max_or_abs_{key}": value for key, value in discrepancies.items()},
                "worker_manifest_sha256": actual_worker_hash,
                "canonical_control_sha256": artifact_hashes["canonical_control.npz"],
                "worker_artifacts_hashed": artifact_count,
                "source_snapshots_checked": len(worker.get("executed_source_snapshot", {})),
            }
        )

    _require(
        set(initial_splits_by_seed) == set(expected_protocol.seeds),
        "not every seed supplied a verified initial-state split",
    )
    state_keys_by_seed = {
        seed: _state_keys(np.concatenate(split, axis=1))
        for seed, split in initial_splits_by_seed.items()
    }
    for index, seed in enumerate(expected_protocol.seeds):
        for other_seed in expected_protocol.seeds[index + 1 :]:
            overlap = state_keys_by_seed[seed] & state_keys_by_seed[other_seed]
            _require(
                not overlap,
                f"initial states repeat across seeds {seed} and {other_seed}: {len(overlap)} duplicates",
            )

    _require(shared_worker_signature is not None, "no worker source signature was read")
    actual_signature_hash = hashlib.sha256(
        json.dumps(shared_worker_signature, sort_keys=True).encode("utf-8")
    ).hexdigest()
    _require(
        actual_signature_hash == expected_source_signature_hash,
        "aggregate worker_source_signature_sha256 does not match worker provenance",
    )

    formula_payload = {
        "problem": asdict(problem),
        "phenotype_grid": "linspace(0,1,21)",
        "growth": "r_i=2/(1+3*x_i^4)",
        "drug_sensitivity": "phi_i=1/(1+x_i^2)",
        "crowding": "G(N)=log(1+mean_i N_i)",
        "dynamics": "dN_i=(r_i-phi_i*u-0.5*G(N))*N_i",
        "unregularized_running_cost": "0.1*sum_i(N_i)+20*u",
        "terminal_cost": "sum_i(N_i(T))",
        "entropy_density": "tau*3*[p*log(p)+(1-p)*log(1-p)], p=u/3, 0log0=0",
        "regularized_running_cost": "unregularized_running_cost+entropy_density",
    }
    source_paths = [Path(__file__).resolve(), REPO_ROOT / "tumor_problem.py"]
    source_hashes = {
        str(path.relative_to(REPO_ROOT)): sha256_file(path)
        for path in source_paths
    }

    # Publish only after every worker passed.  summary.csv is written first so
    # its digest can be sealed into the final audit manifest.
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_output = output_dir / "summary.csv"
    with summary_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    audit_manifest = {
        "status": "passed",
        "audit_scope": "independent hash and high-accuracy ZOH objective audit of exact final HJB-NN tumor 3x3 grid",
        "independence": {
            "calls_faithful_related_work_hjb_nn_problem_evaluator": False,
            "implementation": "formulas transcribed in this script; one DOP853 state+two-cost solve per declared ZOH interval",
            "native_value_prediction_limit": "finite/hash/cross-declaration checked only; cannot be mathematically recomputed without independently reimplementing and executing the saved neural network",
        },
        "aggregate_manifest": str(aggregate_manifest),
        "aggregate_manifest_sha256": sha256_file(aggregate_manifest),
        "aggregate_summary_csv_sha256": aggregate_summary_sha256,
        "audited_summary_csv_sha256": sha256_file(summary_output),
        "aggregate_worker_source_signature_sha256": actual_signature_hash,
        "grid": {
            "taus": list(expected_protocol.taus),
            "seeds": list(expected_protocol.seeds),
            "runs": len(expected_grid),
        },
        "expected_protocol": asdict(expected_protocol),
        "formula": formula_payload,
        "formula_sha256": hashlib.sha256(
            json.dumps(formula_payload, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "tolerances": asdict(tolerances),
        "source_sha256": source_hashes,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "dependency_versions": {
                name: importlib.metadata.version(name) for name in ("numpy", "scipy")
            },
        },
        "integrity_checks": {
            "aggregate_manifest_hashed": True,
            "aggregate_summary_cross_checked_and_hashed": True,
            "worker_manifest_reference_hashes_verified": len(expected_grid),
            "worker_artifacts_hashed": total_artifacts_hashed,
            "canonical_control_hashes_verified": len(expected_grid),
            "executed_source_snapshots_verified": total_source_snapshots_checked,
            "worker_source_and_dependency_signatures_uniform": True,
            "seed_reconstructed_coordinate_uniform_splits_verified": len(expected_grid),
            "cross_tau_same_seed_splits_identical": True,
            "train_validation_and_cross_seed_splits_disjoint": True,
            "initial_and_adaptive_bvp_metadata_verified": True,
        },
        "max_discrepancies": max_discrepancies,
        "runs": output_rows,
    }
    (output_dir / "audit_manifest.json").write_text(
        json.dumps(audit_manifest, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return audit_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = audit_aggregate(args.aggregate_manifest, args.out_dir)
    print(
        json.dumps(
            {
                "status": result["status"],
                "runs": result["grid"]["runs"],
                "max_discrepancies": result["max_discrepancies"],
                "audit_manifest": str(Path(args.out_dir).resolve() / "audit_manifest.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
