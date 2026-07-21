#!/usr/bin/env python3
"""Strict, matched-resolution CPU benchmark for nominal time-only controls.

The protocol separates three questions:

1. how long it takes to construct a control from scratch;
2. how long a frozen artifact takes to deploy; and
3. whether the timed methods reach comparable numerical quality.

The direct baseline is rebuilt as a genuinely sequential n=200 -> 400 -> 800
continuation in every repeat.  Every final n=800 control produced by a timed
chain is saved and audited.  Network checkpoints must be native n=800
checkpoints for the same problem and the same base Transformer architecture.
"""

from __future__ import annotations

import os


# These variables must be set before NumPy, SciPy, or Torch is imported.  The
# previous values are retained in the metadata so the benchmark remains easy
# to audit, while the effective benchmark setting is always one CPU thread.
THREAD_ENV_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
PREIMPORT_THREAD_ENV = {name: os.environ.get(name) for name in THREAD_ENV_NAMES}
for thread_env_name in THREAD_ENV_NAMES:
    os.environ[thread_env_name] = "1"

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import scipy
from scipy.optimize import minimize
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for search_path in (ROOT, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from diagnose_reduced_objective_hessian import make_reduced_objective, projected_kkt  # noqa: E402
from boundary_control import BoundaryProjectedControl  # noqa: E402
from refine_time_only_singular_plateau import build_model  # noqa: E402
from run_direct_openloop_cost import rk4_objective_interval_controls  # noqa: E402
from train_paper_pmp_kkt import (  # noqa: E402
    ProblemConfig,
    TimeMLP,
    TimeTransformer,
    build_params,
)
from train_teacher_free_resolution_curriculum import (  # noqa: E402
    FixedBoxProjection,
    interval_crossing_width,
)
from tumor_problem import TumorProblem, evaluate_zoh_control  # noqa: E402


DEFAULT_REFINED = (
    ROOT
    / "outputs/teacher_free_n800_strict_20260720/"
    "network_lbfgs_after_learn_tau/selected_checkpoint.pt"
)
DEFAULT_DIRECT_J = (
    ROOT
    / "outputs/fair_efficiency_benchmark_20260720/"
    "direct_j_n800_float64_seed4_efficient/objective_rk4/seed_4/best_objective.pt"
)
DEFAULT_PMP = (
    ROOT
    / "outputs/fair_efficiency_benchmark_20260720/"
    "pmp_kkt_n800_float64_seed4/best_pmp_kkt.pt"
)
DEFAULT_MLP = (
    ROOT
    / "outputs/fair_efficiency_benchmark_20260720/"
    "pmp_kkt_mlp_n800_float64_seed4/best_pmp_kkt.pt"
)
DEFAULT_NEURAL_PMP = (
    ROOT
    / "outputs/fair_efficiency_benchmark_20260720/"
    "neural_pmp_n800_float64_seed0_mid_lr008/best_neural_pmp_solution.npz"
)
DEFAULT_TRAINING_TIMING_MANIFEST = (
    ROOT
    / "outputs/fair_efficiency_benchmark_20260720/"
    "matched_n800_training_timing.json"
)
DEFAULT_OUT = ROOT / "outputs/fair_efficiency_benchmark_20260720/strict_unified_benchmark"

CONTINUATION_SUMMARIES = (
    "outputs/teacher_free_n800_20260720/curriculum_pmp_seed4_temp070_scale108/summary.json",
    "outputs/teacher_free_n800_20260720/curriculum_temp070_fixedpoint_step80_lr1e4_learntau/summary.json",
    "outputs/teacher_free_n800_20260720/curriculum_temp070_learntau_final_low_lr3e5/summary.json",
    "outputs/teacher_free_n800_strict_20260720/fixedpoint_detached_s20_p12_lr3e6/summary.json",
    "outputs/teacher_free_n800_strict_20260720/fixedpoint_detached_s20_p16_lr1e6_round2/summary.json",
    "outputs/teacher_free_n800_strict_20260720/fixedpoint_learn_tau_s20_p20_lr5e7/summary.json",
    "outputs/teacher_free_n800_strict_20260720/network_lbfgs_after_learn_tau/summary.json",
)
RECORDED_CONTINUATION_SECONDS = 1012.7790871670004

NETWORK_LABEL_DIRECT_J = "Direct-J Transformer"
NETWORK_LABEL_PMP = "PMP/KKT Transformer"
NETWORK_LABEL_REFINED = "Transformer with projected-gradient refinement"
NETWORK_LABEL_MLP = "PMP/KKT MLP"
NEURAL_PMP_LABEL = "Neural-PMP exact-dynamics PMP-gradient schedule"
DIRECT_LABEL = "Direct time mesh n=800"


def resolve(path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


MANIFEST_METHOD_TO_BENCHMARK_LABEL = {
    "Direct-J Transformer": NETWORK_LABEL_DIRECT_J,
    "PMP/KKT Transformer": NETWORK_LABEL_PMP,
    "PMP/KKT TimeMLP": NETWORK_LABEL_MLP,
    "Known-dynamics Neural-PMP": NEURAL_PMP_LABEL,
}


def load_training_timing_manifest(
    path: Path, artifacts: Mapping[str, Path]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("runs"), list):
        raise ValueError(f"{path}: invalid training timing manifest")
    matched = payload.get("matched_configuration")
    if not isinstance(matched, dict):
        raise ValueError(f"{path}: matched_configuration is missing")
    expected_config = {
        "grid_intervals": 800,
        "grid_nodes": 801,
        "dtype": "float64",
        "device": "cpu",
    }
    for key, expected in expected_config.items():
        if matched.get(key) != expected:
            raise ValueError(
                f"{path}: matched_configuration[{key!r}]={matched.get(key)!r}, "
                f"expected {expected!r}"
            )

    entries: dict[str, dict[str, Any]] = {}
    for raw in payload["runs"]:
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: every runs entry must be a mapping")
        manifest_method = str(raw.get("method", ""))
        if manifest_method not in MANIFEST_METHOD_TO_BENCHMARK_LABEL:
            continue
        label = MANIFEST_METHOD_TO_BENCHMARK_LABEL[manifest_method]
        if label in entries:
            raise ValueError(f"{path}: duplicate timing entry for {label}")
        checkpoint = raw.get("checkpoint")
        scope = raw.get("timing_scope")
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError(f"{path}: {manifest_method} checkpoint is missing")
        if not isinstance(scope, str) or not scope.strip():
            raise ValueError(f"{path}: {manifest_method} timing_scope is missing")
        for time_key in ("process_wall_seconds", "training_loop_wall_seconds"):
            value = raw.get(time_key)
            if value is not None and (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or float(value) <= 0.0
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{path}: invalid {time_key} for {manifest_method}")
        entries[label] = dict(raw)

    for label, artifact in artifacts.items():
        entry = entries.get(label)
        if entry is None:
            raise ValueError(f"{path}: no timing entry for included artifact {label}")
        manifest_artifact = resolve(Path(str(entry["checkpoint"])))
        if manifest_artifact != artifact:
            raise ValueError(
                f"{path}: {label} checkpoint resolves to {manifest_artifact}, "
                f"but benchmark uses {artifact}"
            )
    return payload, entries


def validate_manifest_seconds(
    manifest_path: Path,
    entry: Mapping[str, Any],
    field: str,
    observed: float | None,
    label: str,
) -> None:
    expected = entry.get(field)
    if observed is None:
        return
    if expected is None or not math.isclose(
        float(observed), float(expected), rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise ValueError(
            f"{manifest_path}: {label} benchmark time {observed!r} does not match "
            f"manifest {field}={expected!r}"
        )


def array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def summarize_ms(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
        "p25_ms": percentile(values, 0.25),
        "p75_ms": percentile(values, 0.75),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def summarize_seconds(values: list[float]) -> dict[str, float]:
    return {
        "median_seconds": statistics.median(values),
        "mean_seconds": statistics.fmean(values),
        "p25_seconds": percentile(values, 0.25),
        "p75_seconds": percentile(values, 0.75),
        "p95_seconds": percentile(values, 0.95),
        "min_seconds": min(values),
        "max_seconds": max(values),
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    # bool is a subclass of int in Python, so it must be handled first.
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def require_mapping(payload: Mapping[str, Any], key: str, artifact: Path) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{artifact}: required mapping {key!r} is missing")
    return dict(value)


def architecture_from_args(args: Mapping[str, Any], artifact: Path) -> dict[str, Any]:
    required = ("d_model", "heads", "layers", "init_u")
    missing = [key for key in required if key not in args]
    if missing:
        raise ValueError(f"{artifact}: missing model configuration fields {missing}")
    model_kind = str(args.get("model", "transformer")).lower()
    if model_kind != "transformer":
        raise ValueError(f"{artifact}: expected a Transformer, found {model_kind!r}")
    config = {
        "model": "transformer",
        "d_model": int(args["d_model"]),
        "heads": int(args["heads"]),
        "layers": int(args["layers"]),
        "init_u": float(args["init_u"]),
    }
    if config["d_model"] <= 0 or config["heads"] <= 0 or config["layers"] <= 0:
        raise ValueError(f"{artifact}: nonpositive Transformer architecture field")
    if config["d_model"] % config["heads"] != 0:
        raise ValueError(f"{artifact}: d_model must be divisible by heads")
    if args.get("float64") is not True:
        raise ValueError(f"{artifact}: checkpoint must record float64=True")
    if str(args.get("device", "")).lower() != "cpu":
        raise ValueError(f"{artifact}: checkpoint must record device='cpu'")
    config["checkpoint_training_dtype"] = "float64"
    config["checkpoint_training_device"] = "cpu"
    return config


def mlp_architecture_from_args(args: Mapping[str, Any], artifact: Path) -> dict[str, Any]:
    if str(args.get("model", "")).lower() != "mlp":
        raise ValueError(f"{artifact}: expected model='mlp'")
    hidden_raw = args.get("hidden")
    if isinstance(hidden_raw, str):
        hidden = tuple(int(value.strip()) for value in hidden_raw.split(",") if value.strip())
    elif isinstance(hidden_raw, (list, tuple)):
        hidden = tuple(int(value) for value in hidden_raw)
    else:
        raise ValueError(f"{artifact}: missing or invalid MLP hidden-layer declaration")
    if hidden != (128, 128):
        raise ValueError(f"{artifact}: expected hidden=(128, 128), found {hidden}")
    if "init_u" not in args:
        raise ValueError(f"{artifact}: missing MLP init_u")
    if args.get("float64") is not True:
        raise ValueError(f"{artifact}: checkpoint must record float64=True")
    if str(args.get("device", "")).lower() != "cpu":
        raise ValueError(f"{artifact}: checkpoint must record device='cpu'")
    return {
        "model": "mlp",
        "hidden": list(hidden),
        "init_u": float(args["init_u"]),
        "checkpoint_training_dtype": "float64",
        "checkpoint_training_device": "cpu",
    }


def problem_from_payload(payload: Mapping[str, Any], artifact: Path) -> ProblemConfig:
    raw = require_mapping(payload, "problem", artifact)
    required = tuple(ProblemConfig.__dataclass_fields__)
    missing = [field for field in required if field not in raw]
    if missing:
        raise ValueError(f"{artifact}: missing problem fields {missing}")
    return ProblemConfig(**{field: raw[field] for field in required})


def validate_same_problem(reference: ProblemConfig, candidate: ProblemConfig, label: str) -> None:
    for field, expected in asdict(reference).items():
        observed = getattr(candidate, field)
        if isinstance(expected, int):
            matches = int(observed) == expected
        else:
            matches = math.isclose(
                float(observed), float(expected), rel_tol=0.0, abs_tol=1.0e-12
            )
        if not matches:
            raise ValueError(
                f"{label}: problem field {field!r} is {observed!r}, expected {expected!r}"
            )


def validate_embedded_problem_args(
    args: Mapping[str, Any], cfg: ProblemConfig, artifact: Path
) -> None:
    """Check every problem field when a plain checkpoint repeats it in args."""
    for field, expected in asdict(cfg).items():
        if field not in args:
            raise ValueError(f"{artifact}: args is missing repeated problem field {field!r}")
        observed = args[field]
        if isinstance(expected, int):
            matches = int(observed) == expected
        else:
            matches = math.isclose(
                float(observed), float(expected), rel_tol=0.0, abs_tol=1.0e-12
            )
        if not matches:
            raise ValueError(
                f"{artifact}: args[{field!r}]={observed!r} disagrees with problem={expected!r}"
            )


def load_plain_transformer(
    path: Path,
) -> tuple[torch.nn.Module, ProblemConfig, dict[str, Any], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: checkpoint payload is not a mapping")
    payload = dict(payload)
    cfg = problem_from_payload(payload, path)
    args = require_mapping(payload, "args", path)
    validate_embedded_problem_args(args, cfg, path)
    architecture = architecture_from_args(args, path)
    model = TimeTransformer(
        architecture["d_model"],
        architecture["heads"],
        architecture["layers"],
        cfg.umax,
        architecture["init_u"],
    ).to(device="cpu", dtype=torch.float64)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, cfg, payload, architecture


def load_mlp(
    path: Path,
) -> tuple[torch.nn.Module, ProblemConfig, dict[str, Any], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: checkpoint payload is not a mapping")
    payload = dict(payload)
    cfg = problem_from_payload(payload, path)
    args = require_mapping(payload, "args", path)
    validate_embedded_problem_args(args, cfg, path)
    architecture = mlp_architecture_from_args(args, path)
    model = TimeMLP(
        tuple(architecture["hidden"]), cfg.umax, architecture["init_u"]
    ).to(device="cpu", dtype=torch.float64)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    return model, cfg, payload, architecture


def load_refined_transformer(
    path: Path,
) -> tuple[torch.nn.Module, ProblemConfig, dict[str, Any], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: checkpoint payload is not a mapping")
    payload = dict(payload)
    cfg = problem_from_payload(payload, path)
    base_args = require_mapping(payload, "base_model_args", path)
    # ``base_model_args.n`` describes the sequence length of the historical
    # source model, not a learned architectural dimension.  The checkpoint's
    # top-level ``problem.n`` is the native resolution used by the refinement.
    # All other repeated problem fields must still agree exactly.
    if "n" not in base_args or int(base_args["n"]) <= 0:
        raise ValueError(f"{path}: base_model_args.n is missing or invalid")
    for field, expected in asdict(cfg).items():
        if field == "n":
            continue
        if field not in base_args:
            raise ValueError(f"{path}: base_model_args is missing problem field {field!r}")
        observed = base_args[field]
        if isinstance(expected, int):
            matches = int(observed) == expected
        else:
            matches = math.isclose(
                float(observed), float(expected), rel_tol=0.0, abs_tol=1.0e-12
            )
        if not matches:
            raise ValueError(
                f"{path}: base_model_args[{field!r}]={observed!r} "
                f"disagrees with problem={expected!r}"
            )
    architecture = architecture_from_args(base_args, path)
    wrapper = require_mapping(payload, "wrapper", path)
    wrapper_class = str(wrapper.get("class", ""))
    base = build_model(base_args, cfg).to(device="cpu", dtype=torch.float64)
    if wrapper_class == "BoundaryProjectedControl":
        scale = float(wrapper["scale"])
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"{path}: invalid wrapper scale")
        model = BoundaryProjectedControl(
            base,
            umax=cfg.umax,
            scale_mode="fixed",
            initial_scale=scale,
        ).to(device="cpu", dtype=torch.float64)
    elif wrapper_class == "FixedBoxProjection":
        scale = float(wrapper["scale"])
        temperature = float(wrapper.get("temperature", 1.0))
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"{path}: invalid wrapper scale")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(f"{path}: invalid wrapper temperature")
        model = FixedBoxProjection(
            base,
            cfg.umax,
            scale,
            temperature=temperature,
            learn_temperature=bool(wrapper.get("learn_temperature", False)),
        ).to(device="cpu", dtype=torch.float64)
    else:
        raise ValueError(f"{path}: unsupported refined wrapper {wrapper_class!r}")
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    architecture = {
        **architecture,
        "wrapper": wrapper_class,
        "historical_source_sequence_length": int(base_args["n"]),
        "native_checkpoint_sequence_length": int(cfg.n),
    }
    return model, cfg, payload, architecture


def validate_checkpoint_set(
    entries: list[tuple[str, Path, ProblemConfig, dict[str, Any]]]
) -> ProblemConfig:
    label0, path0, reference, architecture0 = entries[0]
    if reference.n != 800:
        raise ValueError(f"{path0}: {label0} must be a native n=800 checkpoint")
    base_fields = (
        "model",
        "d_model",
        "heads",
        "layers",
        "init_u",
        "checkpoint_training_dtype",
        "checkpoint_training_device",
    )
    base0 = {key: architecture0[key] for key in base_fields}
    for label, path, cfg, architecture in entries[1:]:
        if cfg.n != 800:
            raise ValueError(f"{path}: {label} must be a native n=800 checkpoint")
        validate_same_problem(reference, cfg, f"{label} ({path})")
        base = {key: architecture[key] for key in base0}
        if base != base0:
            raise ValueError(
                f"{path}: base Transformer configuration {base} does not match {base0}"
            )
    return reference


def parameter_count(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def raw_box_violation(values: np.ndarray, lower: float, upper: float) -> float:
    if values.size == 0 or not np.all(np.isfinite(values)):
        return float("inf")
    return float(max(0.0, lower - float(values.min()), float(values.max()) - upper))


def load_neural_pmp_solution(
    path: Path, cfg: ProblemConfig
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load and validate the native-grid oracle-dynamics PMP-gradient output.

    The historical NPZ predates a problem-metadata field.  We therefore bind it
    to the full common problem by checking its grid, initial/terminal
    conditions, stored forward dynamics, switching function, and Hamiltonian.
    """
    required = ("t", "u", "N", "lambda_", "psi", "H")
    with np.load(path, allow_pickle=False) as payload:
        missing = [key for key in required if key not in payload.files]
        if missing:
            raise ValueError(f"{path}: missing Neural-PMP arrays {missing}")
        arrays = {key: np.asarray(payload[key]).copy() for key in required}
    for key, values in arrays.items():
        if values.dtype != np.float64:
            raise ValueError(f"{path}: {key} must be stored as float64")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path}: {key} contains nonfinite values")

    time_grid = arrays["t"].reshape(-1)
    control = arrays["u"].reshape(-1)
    states = arrays["N"]
    costates = arrays["lambda_"]
    stored_psi = arrays["psi"].reshape(-1)
    stored_hamiltonian = arrays["H"].reshape(-1)
    expected_grid = np.linspace(0.0, cfg.T, cfg.n + 1, dtype=np.float64)
    expected_shapes = {
        "t": (cfg.n + 1,),
        "u": (cfg.n,),
        "N": (cfg.n + 1, cfg.m),
        "lambda_": (cfg.n + 1, cfg.m),
        "psi": (cfg.n + 1,),
        "H": (cfg.n + 1,),
    }
    observed_shapes = {
        "t": time_grid.shape,
        "u": control.shape,
        "N": states.shape,
        "lambda_": costates.shape,
        "psi": stored_psi.shape,
        "H": stored_hamiltonian.shape,
    }
    for key, expected_shape in expected_shapes.items():
        if observed_shapes[key] != expected_shape:
            raise ValueError(
                f"{path}: {key} has shape {observed_shapes[key]}, expected {expected_shape}"
            )
    grid_error = float(np.max(np.abs(time_grid - expected_grid)))
    if grid_error > 1.0e-12:
        raise ValueError(f"{path}: time grid does not match native n={cfg.n}, T={cfg.T}")

    x = np.linspace(0.0, 1.0, cfg.m, dtype=np.float64)
    r = 2.0 / (1.0 + 3.0 * x**4)
    phi = 1.0 / (1.0 + x**2)
    suppression = np.full(cfg.m, cfg.m_suppression, dtype=np.float64)
    beta = np.full(cfg.m, cfg.beta, dtype=np.float64)
    alpha = np.full(cfg.m, cfg.alpha, dtype=np.float64)
    initial_error = float(np.max(np.abs(states[0] - cfg.n0)))
    terminal_costate_error = float(np.max(np.abs(costates[-1] - alpha)))
    if initial_error > 1.0e-10:
        raise ValueError(f"{path}: stored initial state disagrees with n0={cfg.n0}")
    if terminal_costate_error > 1.0e-10:
        raise ValueError(f"{path}: stored terminal costate disagrees with alpha={cfg.alpha}")

    dt = cfg.T / cfg.n
    predicted_states = np.empty_like(states)
    predicted_states[0] = states[0]
    for index, action in enumerate(control):
        crowding = np.log1p(states[index].mean())
        derivative = (
            r - phi * float(action) - suppression * crowding
        ) * states[index]
        predicted_states[index + 1] = np.maximum(
            states[index] + dt * derivative, 1.0e-10
        )
    forward_error = float(np.max(np.abs(predicted_states - states)))
    if forward_error > 1.0e-8:
        raise ValueError(
            f"{path}: stored trajectory is inconsistent with the common known dynamics "
            f"(max error {forward_error:.3g})"
        )

    recomputed_psi = cfg.gamma - (costates * phi[None, :] * states).sum(axis=1)
    psi_error = float(np.max(np.abs(recomputed_psi - stored_psi)))
    if psi_error > 1.0e-8:
        raise ValueError(
            f"{path}: stored switching function disagrees with gamma/problem fields"
        )
    control_nodes = np.r_[control, control[-1]]
    dynamics_nodes = np.empty_like(states)
    for index, action in enumerate(control_nodes):
        crowding = np.log1p(states[index].mean())
        dynamics_nodes[index] = (
            r - phi * float(action) - suppression * crowding
        ) * states[index]
    recomputed_hamiltonian = (
        states @ beta
        + cfg.gamma * control_nodes
        + (dynamics_nodes * costates).sum(axis=1)
    )
    hamiltonian_error = float(
        np.max(np.abs(recomputed_hamiltonian - stored_hamiltonian))
    )
    if hamiltonian_error > 1.0e-8:
        raise ValueError(
            f"{path}: stored Hamiltonian disagrees with beta/gamma/problem fields"
        )
    return control, {
        "native_n": cfg.n,
        "decision_variables": cfg.n,
        "dtype": "float64",
        "grid_max_error": grid_error,
        "initial_state_max_error": initial_error,
        "terminal_costate_max_error": terminal_costate_error,
        "forward_dynamics_max_error": forward_error,
        "switching_function_max_error": psi_error,
        "hamiltonian_max_error": hamiltonian_error,
        "problem_binding": (
            "validated from grid, initial/terminal conditions, forward dynamics, "
            "switching function, and Hamiltonian identities"
        ),
    }


def prepare_network_control(
    model: torch.nn.Module, grid: torch.Tensor, cfg: ProblemConfig
) -> tuple[np.ndarray, float]:
    raw_nodes = (
        model(grid).detach().cpu().numpy().astype(np.float64, copy=True).reshape(-1)
    )
    if raw_nodes.size != cfg.n + 1:
        raise ValueError(
            f"network returned {raw_nodes.size} nodes for native n={cfg.n}; expected {cfg.n + 1}"
        )
    violation = raw_box_violation(raw_nodes, 0.0, cfg.umax)
    interval_control = np.clip(raw_nodes[: cfg.n], 0.0, cfg.umax)
    return interval_control, violation


def time_loaded_control_preparation(
    model: torch.nn.Module,
    grid: torch.Tensor,
    cfg: ProblemConfig,
    warmups: int,
    repeats: int,
) -> tuple[np.ndarray, float, dict[str, float]]:
    durations: list[float] = []
    with torch.inference_mode():
        for _ in range(warmups):
            control, violation = prepare_network_control(model, grid, cfg)
        for _ in range(repeats):
            started = time.perf_counter_ns()
            control, violation = prepare_network_control(model, grid, cfg)
            durations.append((time.perf_counter_ns() - started) / 1.0e6)
    return control, violation, summarize_ms(durations)


Loader = Callable[
    [Path], tuple[torch.nn.Module, ProblemConfig, dict[str, Any], dict[str, Any]]
]


def time_hot_process_reload_and_first_control(
    loader: Loader,
    path: Path,
    grid: torch.Tensor,
    expected_cfg: ProblemConfig,
    repeats: int,
) -> dict[str, float]:
    """Time reloads in one already-running process; this is not a cold start."""
    durations: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        model, cfg, _, _ = loader(path)
        validate_same_problem(expected_cfg, cfg, f"hot-process reload of {path}")
        with torch.inference_mode():
            prepare_network_control(model, grid, expected_cfg)
        durations.append((time.perf_counter_ns() - started) / 1.0e6)
    return summarize_ms(durations)


def numpy_rk4_rollout(control: np.ndarray, cfg: ProblemConfig) -> float:
    x = np.linspace(0.0, 1.0, cfg.m, dtype=np.float64)
    r = 2.0 / (1.0 + 3.0 * x**4)
    phi = 1.0 / (1.0 + x**2)
    M = np.full(cfg.m, cfg.m_suppression, dtype=np.float64)
    beta = np.full(cfg.m, cfg.beta, dtype=np.float64)
    alpha = np.full(cfg.m, cfg.alpha, dtype=np.float64)
    state = np.full(cfg.m, cfg.n0, dtype=np.float64)
    accumulated = 0.0
    dt = cfg.T / cfg.n

    def dynamics(state_value: np.ndarray, action: float) -> np.ndarray:
        crowding = np.log1p(state_value.mean())
        return (r - phi * action - M * crowding) * state_value

    for action in control:
        k1_n = dynamics(state, float(action))
        k1_j = float(beta @ state + cfg.gamma * action)
        state2 = np.maximum(state + 0.5 * dt * k1_n, 1.0e-10)
        k2_n = dynamics(state2, float(action))
        k2_j = float(beta @ state2 + cfg.gamma * action)
        state3 = np.maximum(state + 0.5 * dt * k2_n, 1.0e-10)
        k3_n = dynamics(state3, float(action))
        k3_j = float(beta @ state3 + cfg.gamma * action)
        state4 = np.maximum(state + dt * k3_n, 1.0e-10)
        k4_n = dynamics(state4, float(action))
        k4_j = float(beta @ state4 + cfg.gamma * action)
        state = np.maximum(
            state + dt * (k1_n + 2.0 * k2_n + 2.0 * k3_n + k4_n) / 6.0,
            1.0e-10,
        )
        accumulated += dt * (k1_j + 2.0 * k2_j + 2.0 * k3_j + k4_j) / 6.0
    return float(accumulated + alpha @ state)


def time_network_joint_deployment(
    model: torch.nn.Module,
    grid: torch.Tensor,
    cfg: ProblemConfig,
    warmups: int,
    repeats: int,
) -> tuple[float, float, dict[str, float]]:
    """Time forward + NumPy conversion + bound check + clip + state rollout."""
    durations: list[float] = []
    with torch.inference_mode():
        for _ in range(warmups):
            control, violation = prepare_network_control(model, grid, cfg)
            objective = numpy_rk4_rollout(control, cfg)
        for _ in range(repeats):
            started = time.perf_counter_ns()
            control, violation = prepare_network_control(model, grid, cfg)
            objective = numpy_rk4_rollout(control, cfg)
            durations.append((time.perf_counter_ns() - started) / 1.0e6)
    return objective, violation, summarize_ms(durations)


def time_cached_direct_deployment(
    control: np.ndarray,
    cfg: ProblemConfig,
    warmups: int,
    repeats: int,
) -> tuple[float, dict[str, float]]:
    """Time an in-memory control copy and the same state rollout jointly."""
    durations: list[float] = []
    for _ in range(warmups):
        deployed = control.copy()
        objective = numpy_rk4_rollout(deployed, cfg)
    for _ in range(repeats):
        started = time.perf_counter_ns()
        deployed = control.copy()
        objective = numpy_rk4_rollout(deployed, cfg)
        durations.append((time.perf_counter_ns() - started) / 1.0e6)
    return objective, summarize_ms(durations)


def time_in_memory_copy(control: np.ndarray, repeats: int) -> dict[str, float]:
    durations: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        deployed = control.copy()
        durations.append((time.perf_counter_ns() - started) / 1.0e6)
    del deployed
    return summarize_ms(durations)


def time_hot_process_direct_reload(path: Path, repeats: int) -> dict[str, float]:
    durations: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        with np.load(path) as payload:
            loaded = np.asarray(payload["u"], dtype=np.float64).copy()
        durations.append((time.perf_counter_ns() - started) / 1.0e6)
    del loaded
    return summarize_ms(durations)


def reduced_quality(control: np.ndarray, cfg: ProblemConfig) -> dict[str, Any]:
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    reduced = make_reduced_objective(cfg, params)
    tensor = torch.tensor(control, dtype=torch.float64)
    gradient = torch.func.grad(reduced)(tensor).detach().cpu().numpy()
    projected, _, free, active = projected_kkt(control, gradient, 0.0, cfg.umax, 1.0e-6)
    started = time.perf_counter()
    hessian_raw = torch.func.hessian(reduced)(tensor).detach().cpu().numpy()
    hessian_seconds = time.perf_counter() - started
    hessian = 0.5 * (hessian_raw + hessian_raw.T)
    lower_active = control <= 1.0e-6
    upper_active = control >= cfg.umax - 1.0e-6
    strong = (lower_active & (gradient > 1.0e-4)) | (
        upper_active & (gradient < -1.0e-4)
    )
    weak = active & ~strong
    tested = free | weak
    tested_indices = np.flatnonzero(tested)
    restricted_min = None
    if tested_indices.size:
        restricted_min = float(
            np.linalg.eigvalsh(hessian[np.ix_(tested_indices, tested_indices)]).min()
        )
    return {
        "rk4_J": float(reduced(tensor).detach()),
        "gradient_linf": float(np.max(np.abs(gradient))),
        "projected_gradient_linf": float(np.max(np.abs(projected))),
        "projected_gradient_rms": float(np.sqrt(np.mean(projected**2))),
        "restricted_hessian_lambda_min": restricted_min,
        "free_variables": int(np.sum(free)),
        "active_variables": int(np.sum(active)),
        "hessian_audit_seconds": hessian_seconds,
    }


def dop853_quality(control: np.ndarray, cfg: ProblemConfig) -> dict[str, float]:
    problem = TumorProblem(
        T=cfg.T,
        m=cfg.m,
        umax=cfg.umax,
        beta=cfg.beta,
        alpha=cfg.alpha,
        gamma=cfg.gamma,
        n0=cfg.n0,
        m_suppression=cfg.m_suppression,
    )
    time_grid = np.linspace(0.0, cfg.T, cfg.n + 1, dtype=np.float64)
    metrics = evaluate_zoh_control(
        time_grid,
        control,
        problem,
        diagnostic_points=4001,
        include_diagnostics=False,
    )
    return {
        "DOP853_J": float(metrics["J"]),
        "terminal_total": float(metrics["final_total_N"]),
    }


def transition_widths(control: np.ndarray, cfg: ProblemConfig) -> tuple[float, float]:
    time_grid = np.linspace(0.0, cfg.T, cfg.n + 1, dtype=np.float64)
    nodes = np.r_[control, control[-1]]
    return (
        float(interval_crossing_width(time_grid, nodes, "early")),
        float(interval_crossing_width(time_grid, nodes, "late")),
    )


def audit_control(
    method: str,
    control: np.ndarray,
    raw_violation: float,
    cfg: ProblemConfig,
    artifact: Path,
    source_kind: str,
    direct_repeat: int | None = None,
) -> dict[str, Any]:
    if control.shape != (cfg.n,):
        raise ValueError(f"{method}: expected {cfg.n} interval controls, got {control.shape}")
    if not np.all(np.isfinite(control)):
        raise ValueError(f"{method}: control contains nonfinite values")
    print(f"quality audit: {method}", flush=True)
    reduced = reduced_quality(control, cfg)
    continuous = dop853_quality(control, cfg)
    early, late = transition_widths(control, cfg)
    return {
        "method": method,
        "source_kind": source_kind,
        "direct_repeat": direct_repeat,
        "artifact": str(artifact),
        "artifact_sha256": sha256(artifact),
        "control_array_sha256": array_sha256(control),
        **continuous,
        **reduced,
        "independent_numpy_rk4_J": numpy_rk4_rollout(control, cfg),
        "raw_box_violation_before_clip": raw_violation,
        "early_10_90_width": early,
        "late_10_90_width": late,
    }


def interpolate_interval_control(
    source: np.ndarray, source_n: int, target_n: int, T: float
) -> np.ndarray:
    if source.shape != (source_n,):
        raise ValueError(f"expected source shape {(source_n,)}, got {source.shape}")
    source_t = np.linspace(0.0, T, source_n, endpoint=False, dtype=np.float64)
    target_t = np.linspace(0.0, T, target_n, endpoint=False, dtype=np.float64)
    return np.interp(target_t, source_t, source).astype(np.float64, copy=False)


def solve_direct_stage(
    cfg: ProblemConfig,
    initial: np.ndarray,
    maxiter: int,
    maxfun: int,
) -> tuple[Any, float]:
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    initial_state = params["N0"]

    def fun_and_grad(values: np.ndarray) -> tuple[float, np.ndarray]:
        tensor = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        objective = rk4_objective_interval_controls(tensor, initial_state, cfg, params)
        objective.backward()
        return (
            float(objective.detach()),
            tensor.grad.detach().cpu().numpy().astype(np.float64),
        )

    started = time.perf_counter()
    result = minimize(
        fun_and_grad,
        initial.copy(),
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, cfg.umax)] * cfg.n,
        options={
            "maxiter": maxiter,
            "maxfun": maxfun,
            "ftol": 1.0e-10,
            "gtol": 1.0e-7,
            "maxls": 50,
        },
    )
    elapsed = time.perf_counter() - started
    result.x = np.asarray(result.x, dtype=np.float64).copy()
    if result.x.shape != (cfg.n,) or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"direct n={cfg.n} returned an invalid control")
    return result, elapsed


def benchmark_direct_chains(
    cfg_800: ProblemConfig,
    repeats: int,
    controls_dir: Path,
    maxiter: int,
    maxfun: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run paired fresh n=200 -> 400 -> 800 chains and save each exact result."""
    cfg_200 = ProblemConfig(**{**asdict(cfg_800), "n": 200})
    cfg_400 = ProblemConfig(**{**asdict(cfg_800), "n": 400})
    stage_rows: list[dict[str, Any]] = []
    final_runs: list[dict[str, Any]] = []
    controls_dir.mkdir(parents=True, exist_ok=False)

    for repeat in range(repeats):
        print(f"direct sequential chain {repeat + 1}/{repeats}", flush=True)
        chain_started = time.perf_counter()
        initial_200 = np.full(cfg_200.n, 0.5 * cfg_200.umax, dtype=np.float64)
        result_200, elapsed_200 = solve_direct_stage(
            cfg_200, initial_200, maxiter, maxfun
        )
        initial_400 = interpolate_interval_control(
            result_200.x, cfg_200.n, cfg_400.n, cfg_800.T
        )
        result_400, elapsed_400 = solve_direct_stage(
            cfg_400, initial_400, maxiter, maxfun
        )
        initial_800 = interpolate_interval_control(
            result_400.x, cfg_400.n, cfg_800.n, cfg_800.T
        )
        result_800, elapsed_800 = solve_direct_stage(
            cfg_800, initial_800, maxiter, maxfun
        )
        chain_elapsed = time.perf_counter() - chain_started

        artifact = controls_dir / f"direct_chain_repeat_{repeat + 1:02d}_n800.npz"
        np.savez_compressed(
            artifact,
            t=np.linspace(0.0, cfg_800.T, cfg_800.n + 1, dtype=np.float64),
            u=result_800.x,
            repeat=np.asarray(repeat, dtype=np.int64),
            chain_elapsed_seconds=np.asarray(chain_elapsed, dtype=np.float64),
        )
        with np.load(artifact) as saved_payload:
            saved_control = np.asarray(saved_payload["u"], dtype=np.float64).copy()
        control_hash = array_sha256(result_800.x)
        if saved_control.shape != result_800.x.shape or array_sha256(saved_control) != control_hash:
            raise RuntimeError(f"saved direct artifact does not match timed repeat {repeat + 1}")
        final_runs.append(
            {
                "repeat": repeat,
                "control": saved_control,
                "artifact": artifact,
                "artifact_sha256": sha256(artifact),
                "control_array_sha256": control_hash,
                "chain_elapsed_seconds": chain_elapsed,
                "sum_stage_seconds": elapsed_200 + elapsed_400 + elapsed_800,
            }
        )
        for stage, cfg, result, elapsed, initialization in (
            ("n=200", cfg_200, result_200, elapsed_200, "fixed midpoint u=umax/2"),
            ("n=400", cfg_400, result_400, elapsed_400, "interpolated result from same repeat n=200"),
            ("n=800", cfg_800, result_800, elapsed_800, "interpolated result from same repeat n=400"),
        ):
            stage_rows.append(
                {
                    "repeat": repeat,
                    "stage": stage,
                    "n": cfg.n,
                    "elapsed_seconds": elapsed,
                    "chain_elapsed_seconds": chain_elapsed,
                    "objective": float(result.fun),
                    "nit": int(result.nit),
                    "nfev": int(result.nfev),
                    "success": bool(result.success),
                    "status": int(result.status),
                    "message": str(result.message),
                    "initialization": initialization,
                    "final_n800_artifact": str(artifact),
                    "final_n800_artifact_sha256": sha256(artifact),
                    "final_n800_control_array_sha256": control_hash,
                }
            )
        print(
            f"  chain={chain_elapsed:.3f}s, n=800 J={result_800.fun:.9f}, "
            f"success={bool(result_800.success)}",
            flush=True,
        )
    return stage_rows, final_runs


def continuation_seconds() -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    total = 0.0
    for relative in CONTINUATION_SUMMARIES:
        path = ROOT / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        seconds = float(payload.get("wall_seconds", payload.get("total_training_wall_seconds")))
        rows.append({"source": str(path), "seconds": seconds, "sha256": sha256(path)})
        total += seconds
    if not math.isclose(total, RECORDED_CONTINUATION_SECONDS, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError(
            "recorded continuation total changed: "
            f"observed {total:.12f}s, expected {RECORDED_CONTINUATION_SECONDS:.12f}s"
        )
    return total, rows


def selected_run_training_seconds(
    payload: Mapping[str, Any], cli_value: float | None
) -> tuple[float | None, str]:
    if cli_value is not None:
        if cli_value <= 0.0 or not math.isfinite(cli_value):
            raise ValueError("selected-run training time must be positive and finite")
        return float(cli_value), "CLI-supplied timing for the selected run"
    value = payload.get("training_wall_seconds")
    if value is None:
        return None, "not recorded in the selected checkpoint"
    seconds = float(value)
    if seconds <= 0.0 or not math.isfinite(seconds):
        raise ValueError("checkpoint training_wall_seconds must be positive and finite")
    return seconds, "recorded in the selected checkpoint"


def selected_process_seconds(cli_value: float | None) -> tuple[float | None, str]:
    if cli_value is None:
        return None, "full-process timing was not supplied via CLI"
    if cli_value <= 0.0 or not math.isfinite(cli_value):
        raise ValueError("selected process time must be positive and finite")
    return float(cli_value), "CLI-supplied full-process timing for the selected run"


def add_quality_gate(
    quality_rows: list[dict[str, Any]],
    reference_row: dict[str, Any],
    objective_gap_percent: float,
    projected_gradient_linf: float,
    minimum_hessian: float,
    box_tolerance: float,
) -> None:
    reference_objective = float(reference_row["DOP853_J"])
    reference_passes = None
    for row in quality_rows:
        gap = float(row["DOP853_J"] - reference_objective)
        relative_gap = 100.0 * gap / abs(reference_objective)
        row["objective_gap_vs_selected_timed_direct"] = gap
        row["relative_objective_gap_percent"] = relative_gap
        row["passes_objective_quality_gate"] = bool(
            max(0.0, relative_gap) <= objective_gap_percent
        )
        row["passes_projected_gradient_quality_gate"] = bool(
            float(row["projected_gradient_linf"]) <= projected_gradient_linf
        )
        hessian_min = row["restricted_hessian_lambda_min"]
        row["passes_hessian_quality_gate"] = bool(
            hessian_min is not None and float(hessian_min) > minimum_hessian
        )
        row["passes_box_quality_gate"] = bool(
            float(row["raw_box_violation_before_clip"]) <= box_tolerance
        )
        row["passes_all_quality_gates"] = bool(
            row["passes_objective_quality_gate"]
            and row["passes_projected_gradient_quality_gate"]
            and row["passes_hessian_quality_gate"]
            and row["passes_box_quality_gate"]
        )
        if row is reference_row:
            reference_passes = bool(row["passes_all_quality_gates"])
    if reference_passes is None:
        raise RuntimeError("selected direct reference was not present in quality rows")
    for row in quality_rows:
        row["equal_quality_timing_eligible"] = bool(
            reference_passes and row["passes_all_quality_gates"]
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refined", type=Path, default=DEFAULT_REFINED)
    parser.add_argument("--direct_j", type=Path, default=DEFAULT_DIRECT_J)
    parser.add_argument("--pmp", type=Path, default=DEFAULT_PMP)
    parser.add_argument("--mlp", type=Path, default=DEFAULT_MLP)
    parser.add_argument("--neural_pmp_solution", type=Path, default=DEFAULT_NEURAL_PMP)
    parser.add_argument(
        "--training_timing_manifest",
        type=Path,
        default=DEFAULT_TRAINING_TIMING_MANIFEST,
    )
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=50)
    parser.add_argument("--inference_repeats", type=int, default=1000)
    parser.add_argument("--reload_repeats", type=int, default=20)
    parser.add_argument("--joint_warmups", type=int, default=5)
    parser.add_argument("--joint_repeats", type=int, default=50)
    parser.add_argument("--direct_repeats", type=int, default=5)
    parser.add_argument("--direct_maxiter", type=int, default=600)
    parser.add_argument("--direct_maxfun", type=int, default=2000)
    parser.add_argument("--direct_j_selected_run_seconds", type=float)
    parser.add_argument("--pmp_selected_run_seconds", type=float)
    parser.add_argument("--mlp_selected_run_seconds", type=float)
    parser.add_argument("--neural_pmp_selected_run_seconds", type=float)
    parser.add_argument("--quality_max_relative_objective_gap_percent", type=float, default=1.0e-4)
    parser.add_argument("--quality_max_projected_gradient_linf", type=float, default=1.0e-4)
    parser.add_argument("--quality_min_restricted_hessian", type=float, default=0.0)
    parser.add_argument("--quality_box_tolerance", type=float, default=1.0e-12)
    args = parser.parse_args()

    if args.threads != 1:
        raise ValueError("the strict comparison is fixed to one CPU thread")
    if any(int(os.environ[name]) != 1 for name in THREAD_ENV_NAMES):
        raise RuntimeError("single-thread environment was not established before imports")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as error:
        raise RuntimeError("Torch inter-op threads could not be fixed to one") from error
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("Torch did not accept the one-thread benchmark setting")

    for name, value in (
        ("inference_repeats", args.inference_repeats),
        ("reload_repeats", args.reload_repeats),
        ("joint_repeats", args.joint_repeats),
        ("direct_repeats", args.direct_repeats),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    for name, value in (
        (
            "quality_max_relative_objective_gap_percent",
            args.quality_max_relative_objective_gap_percent,
        ),
        (
            "quality_max_projected_gradient_linf",
            args.quality_max_projected_gradient_linf,
        ),
        ("quality_min_restricted_hessian", args.quality_min_restricted_hessian),
        ("quality_box_tolerance", args.quality_box_tolerance),
    ):
        if value < 0.0 or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and nonnegative")

    refined_path = resolve(args.refined)
    direct_j_path = resolve(args.direct_j)
    pmp_path = resolve(args.pmp)
    mlp_path = resolve(args.mlp)
    neural_pmp_path = resolve(args.neural_pmp_solution)
    training_timing_manifest_path = resolve(args.training_timing_manifest)
    out_dir = resolve(args.out_dir)
    for artifact in (refined_path, direct_j_path, pmp_path):
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
    if not training_timing_manifest_path.is_file():
        raise FileNotFoundError(training_timing_manifest_path)
    optional_artifacts = {
        NETWORK_LABEL_MLP: mlp_path,
        NEURAL_PMP_LABEL: neural_pmp_path,
    }
    skipped_optional_artifacts = {
        label: str(path) for label, path in optional_artifacts.items() if not path.is_file()
    }
    for label, path in skipped_optional_artifacts.items():
        print(f"optional baseline skipped because artifact is absent: {label}: {path}", flush=True)
    timing_bound_artifacts = {
        NETWORK_LABEL_DIRECT_J: direct_j_path,
        NETWORK_LABEL_PMP: pmp_path,
        **({NETWORK_LABEL_MLP: mlp_path} if mlp_path.is_file() else {}),
        **({NEURAL_PMP_LABEL: neural_pmp_path} if neural_pmp_path.is_file() else {}),
    }
    training_timing_manifest, training_timing_entries = load_training_timing_manifest(
        training_timing_manifest_path, timing_bound_artifacts
    )
    training_timing_manifest_sha256 = sha256(training_timing_manifest_path)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty directory {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    direct_j_model, direct_j_cfg, direct_j_payload, direct_j_arch = load_plain_transformer(
        direct_j_path
    )
    pmp_model, pmp_cfg, pmp_payload, pmp_arch = load_plain_transformer(pmp_path)
    refined_model, refined_cfg, refined_payload, refined_arch = load_refined_transformer(
        refined_path
    )
    cfg = validate_checkpoint_set(
        [
            (NETWORK_LABEL_DIRECT_J, direct_j_path, direct_j_cfg, direct_j_arch),
            (NETWORK_LABEL_PMP, pmp_path, pmp_cfg, pmp_arch),
            (NETWORK_LABEL_REFINED, refined_path, refined_cfg, refined_arch),
        ]
    )
    mlp_model: torch.nn.Module | None = None
    mlp_payload: dict[str, Any] | None = None
    mlp_arch: dict[str, Any] | None = None
    if mlp_path.is_file():
        mlp_model, mlp_cfg, mlp_payload, mlp_arch = load_mlp(mlp_path)
        if mlp_cfg.n != 800:
            raise ValueError(f"{mlp_path}: {NETWORK_LABEL_MLP} must be a native n=800 checkpoint")
        validate_same_problem(cfg, mlp_cfg, f"{NETWORK_LABEL_MLP} ({mlp_path})")

    neural_pmp_control: np.ndarray | None = None
    neural_pmp_validation: dict[str, Any] | None = None
    if neural_pmp_path.is_file():
        neural_pmp_control, neural_pmp_validation = load_neural_pmp_solution(
            neural_pmp_path, cfg
        )
    grid = torch.linspace(0.0, 1.0, cfg.n + 1, dtype=torch.float64)

    # The repeated direct construction is executed as a paired sequential chain.
    direct_stage_rows, direct_final_runs = benchmark_direct_chains(
        cfg,
        args.direct_repeats,
        out_dir / "timed_direct_controls",
        args.direct_maxiter,
        args.direct_maxfun,
    )

    networks: list[tuple[str, torch.nn.Module, Path, Loader]] = [
        (NETWORK_LABEL_DIRECT_J, direct_j_model, direct_j_path, load_plain_transformer),
        (NETWORK_LABEL_PMP, pmp_model, pmp_path, load_plain_transformer),
        (NETWORK_LABEL_REFINED, refined_model, refined_path, load_refined_transformer),
    ]
    if mlp_model is not None:
        networks.append((NETWORK_LABEL_MLP, mlp_model, mlp_path, load_mlp))
    network_parameter_counts = {
        label: parameter_count(model) for label, model, _, _ in networks
    }
    network_controls: dict[str, np.ndarray] = {}
    network_violations: dict[str, float] = {}
    preparation_summaries: dict[str, dict[str, float]] = {}
    hot_reload_summaries: dict[str, dict[str, float]] = {}
    for label, model, path, loader in networks:
        print(f"loaded control preparation: {label}", flush=True)
        control, violation, preparation = time_loaded_control_preparation(
            model, grid, cfg, args.warmups, args.inference_repeats
        )
        network_controls[label] = control
        network_violations[label] = violation
        preparation_summaries[label] = preparation
        hot_reload_summaries[label] = time_hot_process_reload_and_first_control(
            loader, path, grid, cfg, args.reload_repeats
        )

    quality_rows: list[dict[str, Any]] = []
    for label, _, path, _ in networks:
        quality_rows.append(
            audit_control(
                label,
                network_controls[label],
                network_violations[label],
                cfg,
                path,
                "native n=800 network checkpoint",
            )
        )
    if neural_pmp_control is not None:
        quality_rows.append(
            audit_control(
                NEURAL_PMP_LABEL,
                neural_pmp_control,
                raw_box_violation(neural_pmp_control, 0.0, cfg.umax),
                cfg,
                neural_pmp_path,
                "native n=800 free-control exact-dynamics PMP-gradient result",
            )
        )

    direct_quality_rows: list[dict[str, Any]] = []
    for run in direct_final_runs:
        repeat = int(run["repeat"])
        row = audit_control(
            f"{DIRECT_LABEL} / timed repeat {repeat + 1}",
            run["control"],
            raw_box_violation(run["control"], 0.0, cfg.umax),
            cfg,
            run["artifact"],
            "exact final control from a timed sequential chain",
            direct_repeat=repeat,
        )
        if row["control_array_sha256"] != run["control_array_sha256"]:
            raise RuntimeError("quality audit control does not match the timed direct control")
        direct_quality_rows.append(row)
        quality_rows.append(row)

    # The best DOP853 objective among the exact timed direct outputs defines the
    # common reference and the cached direct deployment artifact.
    reference_row = min(direct_quality_rows, key=lambda row: float(row["DOP853_J"]))
    selected_repeat = int(reference_row["direct_repeat"])
    selected_direct_run = next(
        run for run in direct_final_runs if int(run["repeat"]) == selected_repeat
    )
    for row in quality_rows:
        row["model_parameter_count"] = network_parameter_counts.get(row["method"])
        row["decision_variables"] = (
            cfg.n
            if row["method"] == NEURAL_PMP_LABEL
            or row["method"].startswith(f"{DIRECT_LABEL} /")
            else None
        )
    add_quality_gate(
        quality_rows,
        reference_row,
        args.quality_max_relative_objective_gap_percent,
        args.quality_max_projected_gradient_linf,
        args.quality_min_restricted_hessian,
        args.quality_box_tolerance,
    )
    quality_by_method = {row["method"]: row for row in quality_rows}

    deployment_rows: list[dict[str, Any]] = []
    for label, model, path, _ in networks:
        print(f"joint control-and-state deployment: {label}", flush=True)
        joint_j, joint_violation, joint = time_network_joint_deployment(
            model, grid, cfg, args.joint_warmups, args.joint_repeats
        )
        preparation = preparation_summaries[label]
        hot_reload = hot_reload_summaries[label]
        quality = quality_by_method[label]
        deployment_rows.append(
            {
                "method": label,
                "artifact": str(path),
                "artifact_sha256": sha256(path),
                "model_parameter_count": network_parameter_counts[label],
                "decision_variables": None,
                "deployment_scope": (
                    "loaded n=800 forward + tensor-to-NumPy + raw bound check + "
                    "clip + n=800 state rollout"
                ),
                "timing_scope": (
                    "Loaded-control preparation and joint deployment are timed separately; "
                    "the joint block excludes checkpoint reload, artifact writes, plotting, "
                    "DOP853 evaluation, and gradient/Hessian audits. Hot-process reload is "
                    "reported in its own columns."
                ),
                "loaded_control_preparation_median_ms": preparation["median_ms"],
                "loaded_control_preparation_p95_ms": preparation["p95_ms"],
                "joint_end_to_end_median_ms": joint["median_ms"],
                "joint_end_to_end_p95_ms": joint["p95_ms"],
                "hot_process_reload_plus_first_control_median_ms": hot_reload["median_ms"],
                "hot_process_reload_plus_first_control_p95_ms": hot_reload["p95_ms"],
                "joint_run_numpy_rk4_J": joint_j,
                "joint_run_raw_box_violation_before_clip": joint_violation,
                "equal_quality_timing_eligible": quality["equal_quality_timing_eligible"],
            }
        )

    selected_direct = selected_direct_run["control"]
    direct_artifact = selected_direct_run["artifact"]
    direct_copy = time_in_memory_copy(selected_direct, args.inference_repeats)
    direct_hot_reload = time_hot_process_direct_reload(direct_artifact, args.reload_repeats)
    direct_joint_j, direct_joint = time_cached_direct_deployment(
        selected_direct, cfg, args.joint_warmups, args.joint_repeats
    )
    deployment_rows.append(
        {
            "method": f"Cached {DIRECT_LABEL} / timed repeat {selected_repeat + 1}",
            "artifact": str(direct_artifact),
            "artifact_sha256": sha256(direct_artifact),
            "model_parameter_count": None,
            "decision_variables": cfg.n,
            "deployment_scope": "in-memory control copy + the same n=800 state rollout",
            "timing_scope": (
                "Joint deployment includes the in-memory schedule copy and state rollout; "
                "it excludes NPZ reload, artifact writes, plotting, DOP853 evaluation, and "
                "gradient/Hessian audits. Hot-process NPZ reload is reported separately."
            ),
            "loaded_control_preparation_median_ms": direct_copy["median_ms"],
            "loaded_control_preparation_p95_ms": direct_copy["p95_ms"],
            "joint_end_to_end_median_ms": direct_joint["median_ms"],
            "joint_end_to_end_p95_ms": direct_joint["p95_ms"],
            "hot_process_reload_plus_first_control_median_ms": direct_hot_reload["median_ms"],
            "hot_process_reload_plus_first_control_p95_ms": direct_hot_reload["p95_ms"],
            "joint_run_numpy_rk4_J": direct_joint_j,
            "joint_run_raw_box_violation_before_clip": raw_box_violation(
                selected_direct, 0.0, cfg.umax
            ),
            "equal_quality_timing_eligible": reference_row["equal_quality_timing_eligible"],
        }
    )
    if neural_pmp_control is not None:
        neural_copy = time_in_memory_copy(neural_pmp_control, args.inference_repeats)
        neural_joint_j, neural_joint = time_cached_direct_deployment(
            neural_pmp_control, cfg, args.joint_warmups, args.joint_repeats
        )
        neural_quality = quality_by_method[NEURAL_PMP_LABEL]
        deployment_rows.append(
            {
                "method": NEURAL_PMP_LABEL,
                "artifact": str(neural_pmp_path),
                "artifact_sha256": sha256(neural_pmp_path),
                "model_parameter_count": None,
                "decision_variables": cfg.n,
                "deployment_scope": (
                    "in-memory native n=800 schedule copy + the same n=800 state rollout"
                ),
                "timing_scope": (
                    "Joint deployment includes the in-memory schedule copy and state rollout; "
                    "it excludes NPZ reload, artifact writes, plotting, DOP853 evaluation, "
                    "and gradient/Hessian audits. No network inference or reload is applicable."
                ),
                "loaded_control_preparation_median_ms": neural_copy["median_ms"],
                "loaded_control_preparation_p95_ms": neural_copy["p95_ms"],
                "joint_end_to_end_median_ms": neural_joint["median_ms"],
                "joint_end_to_end_p95_ms": neural_joint["p95_ms"],
                "hot_process_reload_plus_first_control_median_ms": None,
                "hot_process_reload_plus_first_control_p95_ms": None,
                "joint_run_numpy_rk4_J": neural_joint_j,
                "joint_run_raw_box_violation_before_clip": raw_box_violation(
                    neural_pmp_control, 0.0, cfg.umax
                ),
                "equal_quality_timing_eligible": neural_quality[
                    "equal_quality_timing_eligible"
                ],
            }
        )

    chain_times = [float(run["chain_elapsed_seconds"]) for run in direct_final_runs]
    chain_summary = summarize_seconds(chain_times)
    stage_summaries: dict[str, dict[str, float]] = {}
    for stage in ("n=200", "n=400", "n=800"):
        stage_summaries[stage] = summarize_seconds(
            [
                float(row["elapsed_seconds"])
                for row in direct_stage_rows
                if row["stage"] == stage
            ]
        )

    direct_j_training, direct_j_training_source = selected_run_training_seconds(
        direct_j_payload, args.direct_j_selected_run_seconds
    )
    pmp_training, pmp_training_source = selected_process_seconds(
        args.pmp_selected_run_seconds
    )
    mlp_training: float | None = None
    mlp_training_source: str | None = None
    if mlp_payload is not None:
        mlp_training, mlp_training_source = selected_process_seconds(
            args.mlp_selected_run_seconds
        )
    neural_pmp_training: float | None = None
    neural_pmp_training_source: str | None = None
    if neural_pmp_control is not None:
        neural_pmp_training, neural_pmp_training_source = selected_process_seconds(
            args.neural_pmp_selected_run_seconds
        )
    validate_manifest_seconds(
        training_timing_manifest_path,
        training_timing_entries[NETWORK_LABEL_DIRECT_J],
        "training_loop_wall_seconds",
        direct_j_training,
        NETWORK_LABEL_DIRECT_J,
    )
    validate_manifest_seconds(
        training_timing_manifest_path,
        training_timing_entries[NETWORK_LABEL_PMP],
        "process_wall_seconds",
        pmp_training,
        NETWORK_LABEL_PMP,
    )
    if mlp_payload is not None:
        validate_manifest_seconds(
            training_timing_manifest_path,
            training_timing_entries[NETWORK_LABEL_MLP],
            "process_wall_seconds",
            mlp_training,
            NETWORK_LABEL_MLP,
        )
    if neural_pmp_control is not None:
        validate_manifest_seconds(
            training_timing_manifest_path,
            training_timing_entries[NEURAL_PMP_LABEL],
            "process_wall_seconds",
            neural_pmp_training,
            NEURAL_PMP_LABEL,
        )
    continuation_total, continuation_rows = continuation_seconds()

    # A repeated construction-time summary is marked equal-quality eligible
    # only if every exact control contributing to that timing summary passes.
    direct_quality_eligible = bool(
        all(row["equal_quality_timing_eligible"] for row in direct_quality_rows)
    )
    construction_rows: list[dict[str, Any]] = []
    for stage in ("n=200", "n=400", "n=800"):
        summary = stage_summaries[stage]
        construction_rows.append(
            {
                "method": DIRECT_LABEL,
                "scope": f"paired sequential-chain {stage} stage",
                "seconds": summary["median_seconds"],
                "lower_bound_seconds": None,
                "model_parameter_count": None,
                "decision_variables": int(stage.removeprefix("n=")),
                "timing_status": f"median of {args.direct_repeats} fresh paired repeats",
                "timing_scope": (
                    "L-BFGS-B optimizer wall time for this stage only; excludes interpolation, "
                    "artifact serialization, plotting, state/derivative quality audits, and CSV/JSON output."
                ),
                "source": "direct_solve_repeats.csv",
                "training_timing_manifest": None,
                "training_timing_manifest_sha256": None,
                "equal_quality_timing_eligible": direct_quality_eligible,
            }
        )
    construction_rows.append(
        {
            "method": DIRECT_LABEL,
            "scope": "complete fresh n=200 -> n=400 -> n=800 chain",
            "seconds": chain_summary["median_seconds"],
            "lower_bound_seconds": None,
            "model_parameter_count": None,
            "decision_variables": cfg.n,
            "timing_status": f"median of {args.direct_repeats} true sequential repeats",
            "timing_scope": (
                "Complete in-process n=200 -> n=400 -> n=800 optimizer chain including "
                "interpolation; excludes artifact serialization, plotting, state/derivative "
                "quality audits, and CSV/JSON output."
            ),
            "source": "direct_solve_repeats.csv and timed_direct_controls/",
            "training_timing_manifest": None,
            "training_timing_manifest_sha256": None,
            "equal_quality_timing_eligible": direct_quality_eligible,
        }
    )
    for label, seconds, timing_source, path in (
        (NETWORK_LABEL_DIRECT_J, direct_j_training, direct_j_training_source, direct_j_path),
        (NETWORK_LABEL_PMP, pmp_training, pmp_training_source, pmp_path),
    ):
        construction_rows.append(
            {
                "method": label,
                "scope": "one selected native n=800 training run",
                "seconds": seconds,
                "lower_bound_seconds": None,
                "model_parameter_count": network_parameter_counts[label],
                "decision_variables": None,
                "timing_status": (
                    "single-run descriptive timing; not a repeated statistical estimate; "
                    + timing_source
                ),
                "timing_scope": training_timing_entries[label]["timing_scope"],
                "source": (
                    f"artifact {path}; timing manifest {training_timing_manifest_path} "
                    f"(sha256 {training_timing_manifest_sha256})"
                ),
                "training_timing_manifest": str(training_timing_manifest_path),
                "training_timing_manifest_sha256": training_timing_manifest_sha256,
                "equal_quality_timing_eligible": quality_by_method[label][
                    "equal_quality_timing_eligible"
                ],
            }
        )
    if mlp_model is not None and mlp_training_source is not None:
        construction_rows.append(
            {
                "method": NETWORK_LABEL_MLP,
                "scope": "one selected native n=800 PMP/KKT MLP training run",
                "seconds": mlp_training,
                "lower_bound_seconds": None,
                "model_parameter_count": network_parameter_counts[NETWORK_LABEL_MLP],
                "decision_variables": None,
                "timing_status": (
                    "single-run descriptive timing; not a repeated statistical estimate; "
                    + mlp_training_source
                ),
                "timing_scope": training_timing_entries[NETWORK_LABEL_MLP]["timing_scope"],
                "source": (
                    f"artifact {mlp_path}; timing manifest {training_timing_manifest_path} "
                    f"(sha256 {training_timing_manifest_sha256})"
                ),
                "training_timing_manifest": str(training_timing_manifest_path),
                "training_timing_manifest_sha256": training_timing_manifest_sha256,
                "equal_quality_timing_eligible": quality_by_method[NETWORK_LABEL_MLP][
                    "equal_quality_timing_eligible"
                ],
            }
        )
    construction_rows.append(
        {
            "method": NETWORK_LABEL_REFINED,
            "scope": "recorded continuation stages only",
            "seconds": None,
            "lower_bound_seconds": continuation_total,
            "model_parameter_count": network_parameter_counts[NETWORK_LABEL_REFINED],
            "decision_variables": None,
            "timing_status": (
                "strict lower bound; original base-checkpoint training time is unavailable"
            ),
            "timing_scope": (
                "Sum of seven recorded continuation-stage training wall times only; the "
                "original base-checkpoint construction time is unavailable."
            ),
            "source": "sum of the seven bound continuation summary files",
            "training_timing_manifest": None,
            "training_timing_manifest_sha256": None,
            "equal_quality_timing_eligible": quality_by_method[NETWORK_LABEL_REFINED][
                "equal_quality_timing_eligible"
            ],
        }
    )
    if neural_pmp_control is not None and neural_pmp_training_source is not None:
        construction_rows.append(
            {
                "method": NEURAL_PMP_LABEL,
                "scope": (
                    "one predeclared native n=800 free-control exact-dynamics "
                    "PMP-gradient run"
                ),
                "seconds": neural_pmp_training,
                "lower_bound_seconds": None,
                "model_parameter_count": None,
                "decision_variables": cfg.n,
                "timing_status": (
                    "single-run descriptive timing; not a repeated statistical estimate; "
                    + neural_pmp_training_source
                ),
                "timing_scope": training_timing_entries[NEURAL_PMP_LABEL]["timing_scope"],
                "source": (
                    f"artifact {neural_pmp_path}; timing manifest {training_timing_manifest_path} "
                    f"(sha256 {training_timing_manifest_sha256})"
                ),
                "training_timing_manifest": str(training_timing_manifest_path),
                "training_timing_manifest_sha256": training_timing_manifest_sha256,
                "equal_quality_timing_eligible": quality_by_method[NEURAL_PMP_LABEL][
                    "equal_quality_timing_eligible"
                ],
            }
        )

    metadata = {
        "protocol": {
            "problem": asdict(cfg),
            "cpu_threads": 1,
            "dtype": "float64 throughout deployment, direct construction, and common evaluation",
            "network_resolution_requirement": "native n=800 checkpoints queried at 801 nodes",
            "base_transformer_architecture": {
                key: direct_j_arch[key]
                for key in (
                    "model",
                    "d_model",
                    "heads",
                    "layers",
                    "init_u",
                    "checkpoint_training_dtype",
                    "checkpoint_training_device",
                )
            },
            "network_architectures": {
                NETWORK_LABEL_DIRECT_J: direct_j_arch,
                NETWORK_LABEL_PMP: pmp_arch,
                NETWORK_LABEL_REFINED: refined_arch,
                **({NETWORK_LABEL_MLP: mlp_arch} if mlp_arch is not None else {}),
            },
            "network_parameter_counts": network_parameter_counts,
            "schedule_decision_variables": {
                DIRECT_LABEL: cfg.n,
                **({NEURAL_PMP_LABEL: cfg.n} if neural_pmp_control is not None else {}),
            },
            "optional_baselines_skipped": skipped_optional_artifacts,
            "neural_pmp_artifact_validation": neural_pmp_validation,
            "inference_warmups": args.warmups,
            "inference_repeats": args.inference_repeats,
            "hot_process_reload_repeats": args.reload_repeats,
            "joint_deployment_warmups": args.joint_warmups,
            "joint_deployment_repeats": args.joint_repeats,
            "direct_chain_repeats": args.direct_repeats,
            "direct_chain": (
                "each repeat starts from the fixed n=200 midpoint control, then passes "
                "that repeat's optimized result through interpolation to n=400 and n=800"
            ),
            "timing_scope_policy": (
                "Use each construction/deployment row's timing_scope field. Direct solver, "
                "network training-loop, full-process training, and deployment timings have "
                "different explicitly recorded boundaries."
            ),
            "timed_direct_quality_binding": (
                "every saved final n=800 control is audited and linked by array SHA-256"
            ),
        },
        "training_timing_manifest": {
            "path": str(training_timing_manifest_path),
            "sha256": training_timing_manifest_sha256,
            "benchmark_date": training_timing_manifest.get("benchmark_date"),
            "hardware": training_timing_manifest.get("hardware"),
            "matched_configuration": training_timing_manifest.get(
                "matched_configuration"
            ),
            "validated_methods": sorted(timing_bound_artifacts),
        },
        "quality_gate": {
            "reference": reference_row["method"],
            "reference_artifact": reference_row["artifact"],
            "maximum_relative_objective_excess_percent": args.quality_max_relative_objective_gap_percent,
            "maximum_projected_gradient_linf": args.quality_max_projected_gradient_linf,
            "minimum_restricted_hessian_lambda": args.quality_min_restricted_hessian,
            "maximum_raw_box_violation": args.quality_box_tolerance,
            "meaning": (
                "Only rows with equal_quality_timing_eligible=true should be used for "
                "equal-quality speed comparisons."
            ),
        },
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "thread_environment_before_script": PREIMPORT_THREAD_ENV,
            "effective_thread_environment": {
                name: os.environ.get(name) for name in THREAD_ENV_NAMES
            },
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        },
        "interpretation": {
            "construction": (
                "Direct timing is repeated. Network and Neural-PMP construction timings are "
                "single selected-run measurements when available and do not imply statistical "
                "equivalence."
            ),
            "refined_construction": (
                f"Only the recorded continuation lower bound ({continuation_total:.6f} s) "
                "is reported; the original base training time is unavailable."
            ),
            "deployment": (
                "Joint timings are measured in one block; they are not sums of independent medians."
            ),
            "hot_process_reload": (
                "Reload timings occur in one running Python process with a warm operating-system "
                "file cache and are not cold-start measurements."
            ),
            "break_even": (
                "Not applicable to this fixed nominal problem: every resulting schedule, whether "
                "produced by a network, direct optimization, or Neural-PMP, can be cached; none of "
                "the compared methods accepts a new problem instance as input."
            ),
        },
        "direct_chain_summary": chain_summary,
        "direct_stage_summaries": stage_summaries,
        "recorded_refinement_continuation": continuation_rows,
    }

    write_csv(out_dir / "construction_times.csv", construction_rows)
    write_csv(out_dir / "deployment_times.csv", deployment_rows)
    write_csv(out_dir / "quality_metrics.csv", quality_rows)
    write_csv(out_dir / "direct_solve_repeats.csv", direct_stage_rows)
    (out_dir / "benchmark_results.json").write_text(
        json.dumps(
            json_safe(
                {
                    "metadata": metadata,
                    "construction": construction_rows,
                    "deployment": deployment_rows,
                    "quality": quality_rows,
                    "direct_stage_runs": direct_stage_rows,
                }
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "# Strict time-only efficiency benchmark",
        "",
        (
            f"The direct reference is {reference_row['method']} and every reported direct "
            "control was produced by a timed fresh n=200 -> 400 -> 800 chain."
        ),
        "",
        "| Method | Joint deployment median (ms) | Equal-quality eligible |",
        "|---|---:|:---:|",
    ]
    for row in deployment_rows:
        summary_lines.append(
            f"| {row['method']} | {float(row['joint_end_to_end_median_ms']):.4f} | "
            f"{'yes' if row['equal_quality_timing_eligible'] else 'no'} |"
        )
    summary_lines.extend(
        [
            "",
            (
                f"The direct construction median is {chain_summary['median_seconds']:.4f} s "
                f"over {args.direct_repeats} true sequential repeats."
            ),
            (
                f"The refined network has a recorded continuation-time lower bound of "
                f"{continuation_total:.6f} s; its original base training time is unavailable."
            ),
            "",
            (
                "Only rows marked equal-quality eligible may be compared when making a speed "
                "claim; see `quality_metrics.csv` for the common gate."
            ),
            (
                "All outputs here are fixed nominal schedules and can be cached, so repeated "
                "network inference is not an intrinsic deployment advantage over a cached direct "
                "or Neural-PMP schedule."
            ),
        ]
    )
    (out_dir / "README.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
