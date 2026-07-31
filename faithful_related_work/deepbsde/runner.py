"""Run the faithful DeepBSDE reproduction and emit auditable artifacts.

This module imports the solver directly from ``external/DeepBSDE/solver.py``.
It does not copy or modify the official architecture: the per-time-step
Z-networks, Euler BSDE recursion, terminal matching loss and Adam optimizer are
all supplied by that file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import math
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from scipy.special import gammaln, roots_genlaguerre

from tumor_problem import TumorProblem, evaluate_zoh_control, serializable_metrics

from .equations import LogStateTumorHJB, hamiltonian_argmin_numpy, switching_function_numpy


REPO_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = REPO_ROOT / "external" / "DeepBSDE"
DEFAULT_TUMOR_CONFIG = Path(__file__).resolve().parent / "configs" / "tumor_sweep.json"
DEFAULT_HJB_LQ_CONFIG = Path(__file__).resolve().parent / "configs" / "hjb_lq_smoke.json"
DEFAULT_HJB_LQ_FULL_SPEC = Path(__file__).resolve().parent / "configs" / "hjb_lq_full_benchmark.json"
OFFICIAL_HJB_LQ_CONFIG = OFFICIAL_ROOT / "configs" / "hjb_lq_d100.json"


class AttrObject:
    """Recursively expose a JSON mapping through attributes."""

    def __init__(self, mapping: dict[str, Any]):
        self._mapping = mapping
        for key, value in mapping.items():
            setattr(self, key, AttrObject(value) if isinstance(value, dict) else value)

    def to_dict(self) -> dict[str, Any]:
        return self._mapping


class SolverConfig:
    """Config interface expected by the official ``BSDESolver``."""

    def __init__(self, mapping: dict[str, Any]):
        self._mapping = mapping
        self.eqn_config = AttrObject(mapping["eqn_config"])
        self.net_config = AttrObject(mapping["net_config"])

    def to_dict(self) -> dict[str, Any]:
        return self._mapping


def _load_module(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_official_solver_module() -> Any:
    """Load the unchanged official solver under an isolated module name."""

    return _load_module("faithful_deepbsde_official_solver", OFFICIAL_ROOT / "solver.py")


def load_official_equation_module() -> Any:
    """Load official example equations for the original-paper smoke test."""

    return _load_module("faithful_deepbsde_official_equation", OFFICIAL_ROOT / "equation.py")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _upstream_git_bytes(*args: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(OFFICIAL_ROOT), *args],
        stderr=subprocess.PIPE,
    )


def deepbsde_upstream_provenance() -> dict[str, Any]:
    """Describe and verify the nested upstream checkout without mutating it."""

    head = _upstream_git_bytes("rev-parse", "HEAD").decode().strip()
    status_lines = _upstream_git_bytes(
        "status", "--porcelain=v1", "--untracked-files=all"
    ).decode().splitlines()
    tracked_status = [line for line in status_lines if not line.startswith("??")]
    untracked_paths = [line[3:] for line in status_lines if line.startswith("??")]
    tracked_paths = sorted({line[3:] for line in tracked_status})

    component_paths = {
        "solver": "solver.py",
        "equation": "equation.py",
        "hjb_lq_config": "configs/hjb_lq_d100.json",
    }
    components: dict[str, Any] = {}
    contents: dict[str, tuple[bytes, bytes]] = {}
    for name, relative in component_paths.items():
        head_bytes = _upstream_git_bytes("show", f"HEAD:{relative}")
        worktree_bytes = (OFFICIAL_ROOT / relative).read_bytes()
        contents[name] = (head_bytes, worktree_bytes)
        components[name] = {
            "path": relative,
            "head_blob_sha1": _upstream_git_bytes("rev-parse", f"HEAD:{relative}").decode().strip(),
            "head_sha256": _sha256_bytes(head_bytes),
            "worktree_sha256": _sha256_bytes(worktree_bytes),
            "entire_file_matches_head": worktree_bytes == head_bytes,
        }

    equation_head, equation_worktree = contents["equation"]
    equation_prefix_matches = equation_worktree.startswith(equation_head)
    appended = equation_worktree[len(equation_head):] if equation_prefix_matches else b""
    recognized_tumor_append = appended == b"" or appended.startswith(b"\n\nclass TumorHJB(Equation):")
    equation_append_only = equation_prefix_matches and recognized_tumor_append
    components["equation"].update(
        {
            "original_upstream_prefix_matches_head": equation_prefix_matches,
            "original_upstream_prefix_bytes": len(equation_head),
            "appended_bytes": len(appended),
            "appended_sha256": _sha256_bytes(appended),
            "append_only_after_upstream_eof": equation_append_only,
            "recognized_local_append": "TumorHJB" if appended else None,
            "original_hjblq_class_matches_head": equation_prefix_matches,
        }
    )

    numstat_lines = _upstream_git_bytes("diff", "--numstat", "--").decode().splitlines()
    numstat = []
    for line in numstat_lines:
        added, deleted, path = line.split("\t", 2)
        numstat.append({"path": path, "added_lines": int(added), "deleted_lines": int(deleted)})

    permitted_tracked_diff = (
        set(tracked_paths).issubset({"equation.py"})
        and equation_append_only
        and components["solver"]["entire_file_matches_head"]
        and components["hjb_lq_config"]["entire_file_matches_head"]
    )
    return {
        "upstream_repo": "external/DeepBSDE",
        "upstream_head": head,
        "upstream_tracked_dirty": bool(tracked_paths),
        "tracked_status": tracked_status,
        "tracked_diff_summary": numstat,
        "tracked_diff_policy": "only an append after upstream equation.py EOF is permitted",
        "tracked_diff_is_permitted_append_only": permitted_tracked_diff,
        "untracked_paths": untracked_paths,
        "untracked_tumor_config_present": "configs/tumor_hjb_d21.json" in untracked_paths,
        "components": components,
        "faithful_tumor_equation_source": "faithful_related_work/deepbsde/equations.py",
        "legacy_appended_external_tumor_class_used": False,
    }


def validate_hjb_lq_upstream_provenance(provenance: dict[str, Any]) -> None:
    """Fail closed unless all original HJB-LQ code/config bytes match HEAD."""

    components = provenance["components"]
    failures = []
    if not components["solver"]["entire_file_matches_head"]:
        failures.append("solver.py differs from upstream HEAD")
    if not components["hjb_lq_config"]["entire_file_matches_head"]:
        failures.append("configs/hjb_lq_d100.json differs from upstream HEAD")
    if not components["equation"]["original_upstream_prefix_matches_head"]:
        failures.append("the original equation.py prefix differs from upstream HEAD")
    if not components["equation"]["append_only_after_upstream_eof"]:
        failures.append("equation.py change is not a recognized EOF-only TumorHJB append")
    if not provenance["tracked_diff_is_permitted_append_only"]:
        failures.append("nested checkout contains a non-permitted tracked diff")
    if failures:
        raise RuntimeError("DeepBSDE upstream provenance check failed: " + "; ".join(failures))


def seed_everything(seed: int) -> None:
    """Reset NumPy/TensorFlow state and request deterministic TF kernels."""

    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    tf.keras.backend.clear_session()
    np.random.seed(int(seed))
    tf.keras.utils.set_random_seed(int(seed))
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _prepare_empty_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run directory: {path}")


def _write_history(path: Path, history: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "terminal_matching_loss", "y_init", "elapsed_seconds"])
        for step, loss, y_init, elapsed in np.asarray(history):
            writer.writerow([int(step), float(loss), float(y_init), float(elapsed)])


def hjb_lq_reference_value(dim: int, total_time: float, lambd: float = 1.0, order: int = 128) -> float:
    """Evaluate the paper's exact Cole--Hopf/Feynman--Kac formula.

    For ``x=0`` and ``g(x)=log((1+||x||^2)/2)``, equation (14) of the
    paper reduces to a one-dimensional generalized Gauss--Laguerre integral.
    This is deterministic quadrature, not a fitted or simulation-derived
    reference value.
    """

    if dim <= 0 or total_time <= 0.0 or lambd <= 0.0 or order <= 0:
        raise ValueError("dim, total_time, lambd and quadrature order must be positive")
    shape = 0.5 * float(dim)
    nodes, weights = roots_genlaguerre(int(order), shape - 1.0)
    transformed_terminal = np.power(2.0 / (1.0 + 4.0 * float(total_time) * nodes), float(lambd))
    expectation = float(np.sum(weights * transformed_terminal) / np.exp(gammaln(shape)))
    if not 0.0 < expectation <= 1.0:
        raise FloatingPointError(f"invalid Cole--Hopf expectation: {expectation}")
    return -math.log(expectation) / float(lambd)


def _write_hjb_lq_history(path: Path, history: np.ndarray, reference_value: float) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "step",
                "terminal_matching_loss",
                "y_init",
                "absolute_error",
                "relative_error",
                "elapsed_seconds",
            ]
        )
        for step, loss, y_init, elapsed in np.asarray(history):
            absolute_error = abs(float(y_init) - reference_value)
            writer.writerow(
                [
                    int(step),
                    float(loss),
                    float(y_init),
                    absolute_error,
                    absolute_error / abs(reference_value),
                    float(elapsed),
                ]
            )


def _rk4_log_step(equation: LogStateTumorHJB, x: np.ndarray, control: float, dt: float, substeps: int) -> np.ndarray:
    state = np.asarray(x, dtype=np.float64).copy()
    h = float(dt) / int(substeps)
    for _ in range(int(substeps)):
        k1 = equation.controlled_log_drift_numpy(state, control)
        k2 = equation.controlled_log_drift_numpy(state + 0.5 * h * k1, control)
        k3 = equation.controlled_log_drift_numpy(state + 0.5 * h * k2, control)
        k4 = equation.controlled_log_drift_numpy(state + h * k3, control)
        state = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    if not np.all(np.isfinite(state)):
        raise FloatingPointError("non-finite deterministic log-state rollout; no clipping was applied")
    return state


def _model_z_at(model: tf.keras.Model, interval: int, x: np.ndarray, dim: int) -> np.ndarray:
    if interval == 0:
        return np.asarray(model.z_init.numpy()[0], dtype=np.float64)
    tensor = tf.convert_to_tensor(np.asarray(x, dtype=np.float64)[None, :], dtype=tf.float64)
    # This is exactly the normalization used in official NonsharedModel.call.
    return np.asarray(model.subnet[interval - 1](tensor, training=False).numpy()[0], dtype=np.float64) / dim


def extract_canonical_control(
    model: tf.keras.Model,
    equation: LogStateTumorHJB,
    *,
    rk4_substeps: int = 8,
) -> dict[str, np.ndarray]:
    """Roll out the exact HJB Hamiltonian argmin on deterministic dynamics."""

    interval_count = equation.num_time_interval
    time_grid = np.linspace(0.0, equation.total_time, interval_count + 1, dtype=np.float64)
    log_states = np.empty((interval_count + 1, equation.dim), dtype=np.float64)
    z_values = np.empty((interval_count, equation.dim), dtype=np.float64)
    gradients = np.empty_like(z_values)
    psi = np.empty(interval_count, dtype=np.float64)
    controls = np.empty(interval_count, dtype=np.float64)
    log_states[0] = equation.x_init

    for index in range(interval_count):
        z_values[index] = _model_z_at(model, index, log_states[index], equation.dim)
        gradients[index] = z_values[index] / equation.sigma
        psi[index] = float(
            switching_function_numpy(gradients[index], equation.params["phi"], equation.gamma)
        )
        controls[index] = float(
            hamiltonian_argmin_numpy(
                gradients[index],
                equation.params["phi"],
                equation.gamma,
                equation.umax,
            )
        )
        log_states[index + 1] = _rk4_log_step(
            equation,
            log_states[index],
            controls[index],
            equation.delta_t,
            rk4_substeps,
        )

    populations = np.exp(log_states)
    if not np.all(np.isfinite(populations)):
        raise FloatingPointError("non-finite physical state reconstructed from log-state")
    return {
        "t": time_grid,
        "u": controls,
        "log_N": log_states,
        "N": populations,
        "z": z_values,
        "grad_log_value": gradients,
        "psi": psi,
    }


def _source_records() -> dict[str, dict[str, Any]]:
    provenance = deepbsde_upstream_provenance()
    records: dict[str, dict[str, Any]] = {}
    for name in ["solver", "equation", "hjb_lq_config"]:
        component = provenance["components"][name]
        relative = f"external/DeepBSDE/{component['path']}"
        records[relative] = {
            "worktree_sha256": component["worktree_sha256"],
            "upstream_head_blob_sha1": component["head_blob_sha1"],
            "upstream_head_sha256": component["head_sha256"],
            "entire_file_matches_upstream_head": component["entire_file_matches_head"],
        }
        if name == "equation":
            records[relative].update(
                {
                    "original_upstream_prefix_matches_head": component["original_upstream_prefix_matches_head"],
                    "append_only_after_upstream_eof": component["append_only_after_upstream_eof"],
                    "appended_bytes": component["appended_bytes"],
                }
            )
    for path in (Path(__file__).resolve(), Path(__file__).resolve().parent / "equations.py"):
        relative = str(path.relative_to(REPO_ROOT))
        records[relative] = {
            "worktree_sha256": sha256_file(path),
            "local_adaptation_source": True,
        }
    return records


def _build_official_solver(config_dict: dict[str, Any], equation: Any, seed: int) -> Any:
    seed_everything(seed)
    dtype = str(config_dict["net_config"].get("dtype", "float64"))
    tf.keras.backend.set_floatx(dtype)
    config = SolverConfig(config_dict)
    official_solver = load_official_solver_module()
    return official_solver.BSDESolver(config, equation)


def _train_official(config_dict: dict[str, Any], equation: Any, seed: int) -> tuple[Any, np.ndarray]:
    solver = _build_official_solver(config_dict, equation, seed)
    return solver, solver.train()


def tumor_initial_conditioning(solver: Any, equation: LogStateTumorHJB) -> dict[str, Any]:
    """Record the exact t=0 bang-bang-generator conditioning before training.

    The official solver initializes each component of ``Z_0`` uniformly on
    ``[-0.1, 0.1]``.  If ``psi_0 > 0``, the control contribution
    ``U*min(psi_0, 0)`` is locally constant in Z and therefore contributes no
    direct gradient.  Terminal matching still supplies gradients through the
    stochastic ``Z dW`` term, so this is a control-generator dead zone rather
    than a claim that all learning gradients vanish.
    """

    z_init = np.asarray(solver.model.z_init.numpy()[0], dtype=np.float64)
    phi_dot_z = float(equation.params["phi"] @ z_init)
    psi = float(equation.gamma - phi_dot_z / equation.sigma)
    uniform_half_width = 0.1
    approximate_psi_std = float(
        uniform_half_width
        * np.linalg.norm(equation.params["phi"])
        / (math.sqrt(3.0) * equation.sigma)
    )
    gaussian_approx_positive_probability = 0.5 * (
        1.0 + math.erf(equation.gamma / (math.sqrt(2.0) * approximate_psi_std))
    )
    max_phi_dot_z_under_initializer = float(uniform_half_width * np.sum(equation.params["phi"]))
    threshold = equation.gamma * equation.sigma
    return {
        "z_init": z_init.tolist(),
        "phi_dot_z_init": phi_dot_z,
        "threshold_phi_dot_z_for_active_control_branch": threshold,
        "max_phi_dot_z_under_official_initializer_support": max_phi_dot_z_under_initializer,
        "dead_zone_guaranteed_by_initializer_support": bool(threshold >= max_phi_dot_z_under_initializer),
        "psi_init": psi,
        "control_generator_branch_active": bool(psi < 0.0),
        "control_generator_local_dead_zone": bool(psi > 0.0),
        "tie": bool(psi == 0.0),
        "official_z_init_distribution": "iid Uniform[-0.1, 0.1]",
        "gaussian_approximation_only": {
            "psi_standard_deviation": approximate_psi_std,
            "probability_psi_positive": gaussian_approx_positive_probability,
        },
        "interpretation": (
            "psi>0 makes U*min(psi,0) locally constant in Z; terminal matching "
            "still backpropagates through the Z*dW term"
        ),
    }


def run_tumor_experiment(config_dict: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Train one ``(sigma, seed)`` tumor run and save all audit artifacts."""

    output_dir = Path(output_dir)
    _prepare_empty_directory(output_dir)
    seed = int(config_dict["eqn_config"]["seed"])
    # Construct after seeding so equation and network ownership are explicit.
    seed_everything(seed)
    equation = LogStateTumorHJB(AttrObject(config_dict["eqn_config"]))
    solver = _build_official_solver(config_dict, equation, seed)
    initial_conditioning = tumor_initial_conditioning(solver, equation)
    history = solver.train()

    history_path = output_dir / "training_history.csv"
    checkpoint_path = output_dir / "model.weights.h5"
    control_path = output_dir / "canonical_control.npz"
    metrics_path = output_dir / "realized_metrics.json"
    config_path = output_dir / "config.json"
    _write_history(history_path, history)
    solver.model.save_weights(checkpoint_path)

    rollout = extract_canonical_control(
        solver.model,
        equation,
        rk4_substeps=int(config_dict.get("artifact_config", {}).get("rk4_substeps", 8)),
    )
    np.savez(
        control_path,
        **rollout,
        sigma=np.array(equation.sigma, dtype=np.float64),
        seed=np.array(seed, dtype=np.int64),
        control_semantics=np.array("ZOH interval controls from exact HJB Hamiltonian argmin"),
    )
    realized = evaluate_zoh_control(
        rollout["t"],
        rollout["u"],
        TumorProblem(
            T=equation.total_time,
            m=equation.dim,
            umax=equation.umax,
            beta=float(config_dict["eqn_config"]["beta"]),
            alpha=float(config_dict["eqn_config"]["alpha"]),
            gamma=float(config_dict["eqn_config"]["gamma"]),
            n0=float(config_dict["eqn_config"]["n0"]),
            m_suppression=float(config_dict["eqn_config"]["m_suppression"]),
        ),
        include_diagnostics=False,
    )
    metrics_path.write_text(
        json.dumps(serializable_metrics(realized), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    config_path.write_text(
        json.dumps(config_dict, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    artifacts = {}
    for path in [history_path, checkpoint_path, control_path, metrics_path, config_path]:
        artifacts[path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    upstream_provenance = deepbsde_upstream_provenance()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "method": "Han-Jentzen-E DeepBSDE with log-state tumor HJB adaptation",
        "official_solver_unchanged": upstream_provenance["components"]["solver"]["entire_file_matches_head"],
        "official_sources": _source_records(),
        "upstream_tracked_dirty": upstream_provenance["upstream_tracked_dirty"],
        "upstream_provenance": upstream_provenance,
        "git_head": _git_head(),
        "seed": seed,
        "sigma": equation.sigma,
        "state_coordinates": "x=log(N)",
        "state_clipping": False,
        "control_extraction": "exact argmin of u*(gamma-phi^T grad_x V) over [0,umax]",
        "control_values": sorted(float(value) for value in np.unique(rollout["u"])),
        "initial_control_generator_conditioning": initial_conditioning,
        "final_extracted_control_diagnostics": {
            "psi_negative_fraction": float(np.mean(rollout["psi"] < 0.0)),
            "psi_positive_fraction": float(np.mean(rollout["psi"] > 0.0)),
            "umax_fraction": float(np.mean(rollout["u"] == equation.umax)),
            "zero_fraction": float(np.mean(rollout["u"] == 0.0)),
        },
        "hyperparameter_selection": "predeclared config only; realized J is evaluation-only",
        "conditioning_adaptation": "none; official Z initializer retained",
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "config": config_dict,
        "realized_metrics": serializable_metrics(realized),
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def run_tumor_sweep(config_dict: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Run every declared viscosity/seed pair and write a sweep manifest."""

    output_dir = Path(output_dir)
    _prepare_empty_directory(output_dir)
    sigmas = [float(value) for value in config_dict["sweep"]["sigmas"]]
    seeds = [int(value) for value in config_dict["sweep"]["seeds"]]
    if not {0.1, 0.05, 0.025}.issubset(set(sigmas)):
        raise ValueError("the declared full sweep must include sigma=0.10, 0.05 and 0.025")

    runs = []
    for sigma in sigmas:
        for seed in seeds:
            run_config = json.loads(json.dumps(config_dict))
            run_config.pop("sweep", None)
            run_config["eqn_config"]["sigma"] = sigma
            run_config["eqn_config"]["seed"] = seed
            name = f"sigma_{sigma:.3f}_seed_{seed}"
            manifest = run_tumor_experiment(run_config, output_dir / name)
            run_manifest = output_dir / name / "manifest.json"
            runs.append(
                {
                    "name": name,
                    "sigma": sigma,
                    "seed": seed,
                    "manifest": str(run_manifest.relative_to(output_dir)),
                    "manifest_sha256": sha256_file(run_manifest),
                    "realized_J": float(manifest["realized_metrics"]["J"]),
                }
            )
    sweep_manifest = {
        "schema_version": 1,
        "method": "DeepBSDE log-state viscosity sweep",
        "sigmas": sigmas,
        "seeds": seeds,
        "selection_rule": "none; every declared run is retained",
        "runs": runs,
    }
    (output_dir / "sweep_manifest.json").write_text(
        json.dumps(sweep_manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return sweep_manifest


def _run_hjb_lq_single(
    config_dict: dict[str, Any],
    output_dir: Path,
    *,
    seed: int,
    mode: str,
    config_source: Path,
    quadrature_order: int = 128,
) -> dict[str, Any]:
    """Run one official HJB-LQ solver instance with exact-value errors."""

    if mode not in {"smoke", "full"}:
        raise ValueError(f"unsupported benchmark mode: {mode}")
    output_dir = Path(output_dir)
    _prepare_empty_directory(output_dir)
    upstream_provenance = deepbsde_upstream_provenance()
    validate_hjb_lq_upstream_provenance(upstream_provenance)
    seed_everything(seed)
    official_equations = load_official_equation_module()
    equation = official_equations.HJBLQ(AttrObject(config_dict["eqn_config"]))
    reference_value = hjb_lq_reference_value(
        equation.dim,
        equation.total_time,
        lambd=equation.lambd,
        order=quadrature_order,
    )
    solver, history = _train_official(config_dict, equation, seed)
    history_path = output_dir / "training_history_with_error.csv"
    checkpoint_path = output_dir / "model.weights.h5"
    config_path = output_dir / "solver_config.json"
    _write_hjb_lq_history(history_path, history, reference_value)
    solver.model.save_weights(checkpoint_path)
    config_path.write_text(
        json.dumps(config_dict, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    final_y_init = float(history[-1, 2])
    absolute_error = abs(final_y_init - reference_value)
    source_record = {
        "path": str(config_source.relative_to(REPO_ROOT)),
        "sha256": sha256_file(config_source),
    }
    manifest = {
        "schema_version": 1,
        "method": "Han-Jentzen-E 100D HJB-LQ official benchmark",
        "mode": mode,
        "result_status": (
            "wiring_smoke_not_a_numerical_reproduction"
            if mode == "smoke"
            else "full_numerical_reproduction_run"
        ),
        "not_a_numerical_reproduction": mode == "smoke",
        "official_equation_class": "external/DeepBSDE/equation.py::HJBLQ",
        "official_solver_class": "external/DeepBSDE/solver.py::BSDESolver",
        "official_sources": _source_records(),
        "upstream_tracked_dirty": upstream_provenance["upstream_tracked_dirty"],
        "upstream_provenance": upstream_provenance,
        "solver_config_source": source_record,
        "official_full_config_unchanged": (
            mode == "full"
            and config_source.resolve() == OFFICIAL_HJB_LQ_CONFIG.resolve()
            and upstream_provenance["components"]["hjb_lq_config"]["entire_file_matches_head"]
        ),
        "seed": int(seed),
        "history_rows": int(len(history)),
        "reference": {
            "value": reference_value,
            "method": "deterministic generalized Gauss-Laguerre evaluation of paper equation (14)",
            "quadrature_order": int(quadrature_order),
            "paper_equation": "u(0,0)=-(1/lambda) log E[exp(-lambda*g(sqrt(2)*W_T))]",
            "paper_url": "https://arxiv.org/abs/1707.02568",
        },
        "paper_reported_five_run_relative_error": 0.0017,
        "final_terminal_matching_loss": float(history[-1, 1]),
        "final_y_init": final_y_init,
        "final_absolute_error": absolute_error,
        "final_relative_error": absolute_error / abs(reference_value),
        "artifacts": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in [history_path, checkpoint_path, config_path]
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def run_hjb_lq_smoke(config_dict: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Run a tiny official-path wiring check; never classify it as reproduction."""

    return _run_hjb_lq_single(
        config_dict,
        output_dir,
        seed=int(config_dict.get("seed", 0)),
        mode="smoke",
        config_source=DEFAULT_HJB_LQ_CONFIG,
    )


def run_hjb_lq_full(benchmark_spec: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Run the predeclared five-seed official 100D HJB-LQ benchmark."""

    output_dir = Path(output_dir)
    _prepare_empty_directory(output_dir)
    upstream_provenance = deepbsde_upstream_provenance()
    validate_hjb_lq_upstream_provenance(upstream_provenance)
    config_source = (REPO_ROOT / benchmark_spec["official_solver_config"]).resolve()
    if config_source != OFFICIAL_HJB_LQ_CONFIG.resolve():
        raise ValueError(
            "full HJB-LQ reproduction must use external/DeepBSDE/configs/hjb_lq_d100.json unchanged"
        )
    config_dict = load_json(config_source)
    expected = {"dim": 100, "total_time": 1.0, "num_time_interval": 20}
    actual = {key: config_dict["eqn_config"][key] for key in expected}
    if actual != expected or int(config_dict["net_config"]["num_iterations"]) != 2000:
        raise ValueError(f"official HJB-LQ config does not match the declared full benchmark: {actual}")
    seeds = [int(value) for value in benchmark_spec["seeds"]]
    if seeds != [0, 1, 2, 3, 4]:
        raise ValueError("full paper benchmark requires the predeclared five seeds [0,1,2,3,4]")
    quadrature_order = int(benchmark_spec.get("reference_quadrature_order", 128))

    runs = []
    for seed in seeds:
        name = f"seed_{seed}"
        manifest = _run_hjb_lq_single(
            config_dict,
            output_dir / name,
            seed=seed,
            mode="full",
            config_source=config_source,
            quadrature_order=quadrature_order,
        )
        run_manifest = output_dir / name / "manifest.json"
        runs.append(
            {
                "seed": seed,
                "manifest": str(run_manifest.relative_to(output_dir)),
                "manifest_sha256": sha256_file(run_manifest),
                "final_y_init": manifest["final_y_init"],
                "final_relative_error": manifest["final_relative_error"],
                "terminal_matching_loss": manifest["final_terminal_matching_loss"],
            }
        )

    estimates = np.asarray([run["final_y_init"] for run in runs], dtype=np.float64)
    errors = np.asarray([run["final_relative_error"] for run in runs], dtype=np.float64)
    benchmark_config_path = output_dir / "benchmark_spec.json"
    benchmark_config_path.write_text(
        json.dumps(benchmark_spec, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "method": "Han-Jentzen-E official 100D HJB-LQ five-run benchmark",
        "mode": "full",
        "result_status": "full_numerical_reproduction_run",
        "selection_rule": "none; all five predeclared seeds retained",
        "realized_J_used_for_selection": False,
        "official_solver_config": {
            "path": str(config_source.relative_to(REPO_ROOT)),
            "sha256": sha256_file(config_source),
            "unchanged": True,
        },
        "upstream_tracked_dirty": upstream_provenance["upstream_tracked_dirty"],
        "upstream_provenance": upstream_provenance,
        "seeds": seeds,
        "reference_value": hjb_lq_reference_value(100, 1.0, order=quadrature_order),
        "mean_y_init": float(np.mean(estimates)),
        "sample_std_y_init": float(np.std(estimates, ddof=1)),
        "mean_relative_error": float(np.mean(errors)),
        "max_relative_error": float(np.max(errors)),
        "paper_reported_relative_error": 0.0017,
        "paper_comparison_note": "compare only after all five full runs complete; smoke outputs are excluded",
        "runs": runs,
        "benchmark_spec": {
            "path": benchmark_config_path.name,
            "sha256": sha256_file(benchmark_config_path),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def smoke_tumor_config(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Return an explicitly small-budget version without changing equations."""

    result = json.loads(json.dumps(config_dict))
    result["sweep"]["seeds"] = [0]
    result["eqn_config"]["num_time_interval"] = 4
    result["net_config"].update(
        {
            "num_hiddens": [8, 8],
            "lr_values": [0.001, 0.001],
            "lr_boundaries": [1],
            "num_iterations": 1,
            "batch_size": 4,
            "valid_size": 4,
            "logging_frequency": 1,
            "verbose": False,
        }
    )
    result["artifact_config"]["rk4_substeps"] = 2
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    hjb = subparsers.add_parser("hjb-lq-smoke", help="validate the unchanged official HJB-LQ path")
    hjb.add_argument("--config", type=Path, default=DEFAULT_HJB_LQ_CONFIG)
    hjb.add_argument("--output-dir", type=Path, required=True)

    hjb_full = subparsers.add_parser(
        "hjb-lq-full",
        help="run the predeclared five-seed official 100D HJB-LQ numerical reproduction",
    )
    hjb_full.add_argument("--benchmark-config", type=Path, default=DEFAULT_HJB_LQ_FULL_SPEC)
    hjb_full.add_argument("--output-dir", type=Path, required=True)

    tumor = subparsers.add_parser("tumor-sweep", help="run the declared sigma/seed sweep")
    tumor.add_argument("--config", type=Path, default=DEFAULT_TUMOR_CONFIG)
    tumor.add_argument("--output-dir", type=Path, required=True)
    tumor.add_argument("--smoke", action="store_true", help="retain all sigmas but use one seed and one update")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "hjb-lq-smoke":
        config = load_json(args.config)
        manifest = run_hjb_lq_smoke(config, args.output_dir)
    elif args.command == "hjb-lq-full":
        manifest = run_hjb_lq_full(load_json(args.benchmark_config), args.output_dir)
    else:
        config = load_json(args.config)
        if args.smoke:
            config = smoke_tumor_config(config)
        manifest = run_tumor_sweep(config, args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
