#!/usr/bin/env python3
"""Continuous-policy held-out PMP diagnostics for Table 1.

This evaluator applies one retained PMP policy, without retraining, to the
same fixed held-out initial states used by ``evaluate_table1_heldout_common``.
It intentionally matches the continuous fixed-state diagnostics:

* arbitrary time queries keep the original Transformer support context fixed;
* the time logits are interpolated by PCHIP between direct policy queries;
* state and open-loop PMP costate trajectories are integrated by DOP853;
* ``psi=H_u``, ``dot(psi)``, and ``ddot(psi)`` use the analytic identities;
* the mask is ``[interior_start, interior_end)`` with Transformer support
  nodes excluded; and
* each state's reported residual is

      R_sing = sqrt(RMS(psi)^2 + RMS(dot(psi))^2
                    + RMS(ddot(psi))^2).

The ODEs are evaluated in small state batches to avoid repeating Python and
solver overhead 128 times.  ``--verify-first-sample`` compares the batched
implementation with the existing scalar fixed-state evaluator.

The resulting standard deviation across held-out states is diagnostic only.
For a paper table whose ``+/-`` denotes training-seed variability, first take
the held-out mean from each independently trained seed, then compute the
sample standard deviation across those seed means.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.integrate import solve_ivp


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for search_path in (ROOT, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from evaluate_feedback_section5 import load_feedback_checkpoint  # noqa: E402
from evaluate_table1_heldout_common import load_directions  # noqa: E402
from generate_offgrid_policy_switching_diagnostics import (  # noqa: E402
    DensePolicy,
    build_grid_flags,
    fixed_support_query_logits,
    integrate_trajectory,
    load_time_model,
    numpy_time_features,
    raw_time_logits,
    time_action_from_raw,
    validate_numpy_feedback_action,
)
from generate_three_case_hamiltonian_results import (  # noqa: E402
    problem_from_config,
)


METHODS = ("time_only", "feedback_cf", "feedback_der")
QUANTITIES = ("H_u", "dH_u_dt", "d2H_u_dt2")


def resolve(path: Path) -> Path:
    expanded = path.expanduser()
    return (
        expanded.resolve()
        if expanded.is_absolute()
        else (ROOT / expanded).resolve()
    )


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


def statistics(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "sample_std_across_states": (
            float(np.std(array, ddof=1)) if array.size > 1 else 0.0
        ),
        "median": float(np.median(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q95": float(np.quantile(array, 0.95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def build_dense_policy(
    method: str,
    checkpoint: Path,
    dense_time: np.ndarray,
    *,
    device: torch.device,
    query_batch_size: int,
    cache_path: Path | None = None,
) -> tuple[DensePolicy, dict[str, Any], float]:
    started = time.perf_counter()
    if method == "time_only":
        model, cfg, wrapper = load_time_model(checkpoint)
        source_args = None
        feedback_model = None
        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        recorded_multiplier = None
        refinement = payload.get("refinement")
        if isinstance(refinement, dict) and refinement.get("multiplier") is not None:
            recorded_multiplier = int(refinement["multiplier"])
    else:
        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        model, cfg, source_args = load_feedback_checkpoint(checkpoint)
        wrapper = {}
        feedback_model = model
        recorded_multiplier = None
        refinement = payload.get("refinement")
        if isinstance(refinement, dict) and refinement.get("multiplier") is not None:
            recorded_multiplier = int(refinement["multiplier"])
        elif getattr(source_args, "offgrid_refinement_multiplier", None) is not None:
            recorded_multiplier = int(source_args.offgrid_refinement_multiplier)

    model = model.to(device=device, dtype=torch.float64)
    support = torch.linspace(
        0.0, 1.0, cfg.n + 1, device=device, dtype=torch.float64
    )
    normalized = torch.from_numpy(dense_time / cfg.T).to(
        device=device, dtype=torch.float64
    )
    on_support, nearest, _ = build_grid_flags(dense_time, cfg)
    if int(on_support.sum()) != cfg.n + 1:
        raise RuntimeError(
            f"expected {cfg.n + 1} support nodes, found "
            f"{int(on_support.sum())}"
        )
    on_support_tensor = torch.from_numpy(on_support).to(device=device)
    support_indices = torch.from_numpy(nearest[on_support]).to(device=device)
    query_model = model.time_branch if feedback_model is not None else model

    def compute_dense_raw() -> tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            current_support_raw = raw_time_logits(query_model, support)
        current_dense_raw = torch.empty_like(normalized)
        current_dense_raw[on_support_tensor] = current_support_raw[
            support_indices
        ]
        off_support_tensor = ~on_support_tensor
        current_dense_raw[off_support_tensor] = fixed_support_query_logits(
            query_model,
            support,
            normalized[off_support_tensor],
            batch_size=query_batch_size,
        )
        return (
            current_dense_raw.cpu().numpy(),
            current_support_raw.cpu().numpy(),
        )

    cache_metadata: dict[str, Any] = {
        "enabled": cache_path is not None,
        "path": str(cache_path) if cache_path is not None else None,
        "loaded": False,
        "checkpoint_sha256": sha256(checkpoint),
    }
    if cache_path is None:
        dense_raw_array, support_raw_array = compute_dense_raw()
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
        with lock_path.open("a+b") as lock_stream:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
            if cache_path.is_file():
                with np.load(cache_path, allow_pickle=False) as cache:
                    schema = str(cache["schema"].item())
                    cached_method = str(cache["method"].item())
                    cached_checkpoint = str(
                        cache["checkpoint_sha256"].item()
                    )
                    cached_time = np.asarray(
                        cache["dense_time"], dtype=np.float64
                    )
                    if schema != "table1-dense-policy-cache-v1":
                        raise ValueError(
                            f"{cache_path}: unsupported cache schema "
                            f"{schema!r}"
                        )
                    if cached_method != method:
                        raise ValueError(
                            f"{cache_path}: method mismatch "
                            f"{cached_method!r} != {method!r}"
                        )
                    if cached_checkpoint != cache_metadata[
                        "checkpoint_sha256"
                    ]:
                        raise ValueError(
                            f"{cache_path}: checkpoint hash mismatch"
                        )
                    if not np.array_equal(cached_time, dense_time):
                        raise ValueError(
                            f"{cache_path}: dense time grid mismatch"
                        )
                    dense_raw_array = np.asarray(
                        cache["dense_raw"], dtype=np.float64
                    )
                    support_raw_array = np.asarray(
                        cache["support_raw"], dtype=np.float64
                    )
                cache_metadata["loaded"] = True
            else:
                dense_raw_array, support_raw_array = compute_dense_raw()
                temporary = cache_path.with_name(
                    f".{cache_path.name}.{os.getpid()}.tmp.npz"
                )
                np.savez(
                    temporary,
                    schema=np.asarray("table1-dense-policy-cache-v1"),
                    method=np.asarray(method),
                    checkpoint_sha256=np.asarray(
                        cache_metadata["checkpoint_sha256"]
                    ),
                    dense_time=dense_time,
                    dense_raw=dense_raw_array,
                    support_raw=support_raw_array,
                )
                os.replace(temporary, cache_path)
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        cache_metadata["sha256"] = sha256(cache_path)

    if dense_raw_array.shape != dense_time.shape:
        raise ValueError("dense-policy cache has an invalid raw-logit shape")
    if support_raw_array.shape != (cfg.n + 1,):
        raise ValueError("dense-policy cache has an invalid support shape")
    reconstruction_error = float(
        np.max(
            np.abs(
                dense_raw_array[on_support]
                - support_raw_array[nearest[on_support]]
            )
        )
    )

    model = model.cpu()
    policy = DensePolicy(
        method,
        cfg,
        dense_time,
        dense_raw_array,
        support_raw_array,
        checkpoint,
        feedback_model=model if feedback_model is not None else None,
        feedback_args=source_args,
        time_wrapper=wrapper if feedback_model is None else None,
    )
    metadata = {
        "problem": vars(cfg),
        "recorded_refinement_multiplier": recorded_multiplier,
        "time_context_tokens": cfg.n + 1,
        "fixed_support_on_grid_reconstruction_max_abs": reconstruction_error,
        "numpy_vs_torch_feedback_action_max_abs": (
            validate_numpy_feedback_action(policy)
            if feedback_model is not None
            else None
        ),
        "dense_policy_cache": cache_metadata,
    }
    return policy, metadata, time.perf_counter() - started


def state_features_batch(
    policy: DensePolicy,
    normalized_time: float,
    states: np.ndarray,
) -> np.ndarray:
    if policy.feedback_model is None or policy.feedback_args is None:
        raise RuntimeError("state features require a feedback policy")
    states = np.asarray(states, dtype=np.float64)
    reference = policy.reference_at(normalized_time)
    mode = str(
        getattr(policy.feedback_args, "state_feature_mode", "log_absolute")
    )
    if mode == "log_absolute":
        scale = float(policy.feedback_model.state_log_scale)
        component = np.log1p(states) / scale
        summaries = np.column_stack(
            (component.mean(axis=1), component.std(axis=1), component.max(axis=1))
        )
    elif mode == "relative_nominal":
        component = (states - reference[None, :]) / np.maximum(
            reference[None, :], 1.0e-8
        )
        summaries = np.column_stack(
            (component.mean(axis=1), component.std(axis=1), component.max(axis=1))
        )
    elif mode == "total_burden":
        total = np.maximum(states.sum(axis=1), np.finfo(np.float64).eps)
        reference_total = max(
            float(reference.sum()), np.finfo(np.float64).eps
        )
        component = np.zeros_like(states)
        summaries = np.column_stack(
            (np.log(total / reference_total), np.zeros_like(total), np.zeros_like(total))
        )
    elif mode == "burden_composition":
        if policy._feature_r is None or policy._feature_phi is None:
            raise RuntimeError("feedback feature vectors are unavailable")
        eps = np.finfo(np.float64).eps
        total = np.maximum(states.sum(axis=1), eps)
        reference_total = max(float(reference.sum()), eps)
        composition = states / total[:, None]
        reference_composition = reference / reference_total
        log_composition = np.log(np.maximum(composition, eps))
        log_reference = np.log(np.maximum(reference_composition, eps))
        component = (
            log_composition
            - log_composition.mean(axis=1, keepdims=True)
            - log_reference[None, :]
            + log_reference.mean()
        )
        delta = composition - reference_composition[None, :]
        r_centered = policy._feature_r - policy._feature_r.mean()
        phi_centered = policy._feature_phi - policy._feature_phi.mean()
        r_scale = max(float(np.sqrt(np.mean(r_centered**2))), eps)
        phi_scale = max(float(np.sqrt(np.mean(phi_centered**2))), eps)
        summaries = np.column_stack(
            (
                np.log(total / reference_total),
                delta @ r_centered / r_scale,
                delta @ phi_centered / phi_scale,
            )
        )
    else:
        raise ValueError(f"unsupported state feature mode {mode!r}")
    time_part = np.broadcast_to(
        numpy_time_features(normalized_time), (states.shape[0], 6)
    )
    return np.concatenate((time_part, component, summaries), axis=1)


def state_logits_batch(
    policy: DensePolicy,
    normalized_time: float,
    states: np.ndarray,
) -> np.ndarray:
    value = state_features_batch(policy, normalized_time, states)
    for index, (weight, bias) in enumerate(policy._linears):
        value = value @ weight.T + bias
        if index < len(policy._linears) - 1:
            value = np.tanh(value)
    return np.asarray(value[:, 0], dtype=np.float64)


def actions_batch(
    policy: DensePolicy,
    physical_time: float,
    states: np.ndarray,
) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    raw = policy.raw_at(physical_time)
    if policy.feedback_model is None:
        value = time_action_from_raw(
            np.asarray([raw], dtype=np.float64),
            policy.cfg,
            policy.time_wrapper or {},
        )[0]
        return np.full(states.shape[0], float(value), dtype=np.float64)

    if policy.feedback_args is None:
        raise RuntimeError("feedback arguments are unavailable")
    normalized = float(
        np.clip(physical_time / policy.cfg.T, 0.0, 1.0)
    )
    correction = state_logits_batch(policy, normalized, states)
    if bool(
        getattr(policy.feedback_args, "center_state_correction", False)
    ):
        reference = policy.reference_at(normalized)[None, :]
        correction -= state_logits_batch(policy, normalized, reference)[0]
    combined = raw + float(
        getattr(policy.feedback_args, "correction_gain", 1.0)
    ) * correction
    parameterization = str(
        getattr(
            policy.feedback_args,
            "action_parameterization",
            "logit-temperature",
        )
    )
    if parameterization == "linear-raw-box":
        return np.clip(combined, 0.0, policy.cfg.umax)
    scale = float(getattr(policy.feedback_args, "action_scale", 1.0))
    offset = float(getattr(policy.feedback_args, "action_offset", 0.0))
    temperature = float(
        getattr(policy.feedback_args, "action_temperature", 1.0)
    )
    probability = np.empty_like(combined)
    nonnegative = combined >= 0.0
    probability[nonnegative] = 1.0 / (
        1.0 + np.exp(-combined[nonnegative] / temperature)
    )
    exponential = np.exp(combined[~nonnegative] / temperature)
    probability[~nonnegative] = exponential / (1.0 + exponential)
    return np.clip(
        scale * policy.cfg.umax * probability - offset,
        0.0,
        policy.cfg.umax,
    )


def batch_dynamics(
    states: np.ndarray,
    controls: np.ndarray,
    parameters: dict[str, np.ndarray],
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
    parameters: dict[str, np.ndarray],
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
    parameters: dict[str, np.ndarray],
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


def evaluate_batch(
    policy: DensePolicy,
    initial_states: np.ndarray,
    selected_time: np.ndarray,
    *,
    rtol: float,
    atol: float,
    max_step: float,
    report_scale_factor: float,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    started = time.perf_counter()
    batch, components = initial_states.shape
    problem = policy.cfg
    tumor_problem = problem_from_config(problem)
    parameters = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in tumor_problem.vectors().items()
    }
    parameters["gamma"] = float(tumor_problem.gamma)

    initial_augmented = np.zeros((batch, components + 1), dtype=np.float64)
    initial_augmented[:, :components] = initial_states

    def forward_rhs(physical_time: float, flat: np.ndarray) -> np.ndarray:
        augmented = flat.reshape(batch, components + 1)
        states = augmented[:, :components]
        controls = actions_batch(policy, physical_time, states)
        drift = batch_dynamics(states, controls, parameters)
        running = (
            states @ parameters["beta"]
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
    terminal_augmented = forward.sol(problem.T).reshape(
        batch, components + 1
    )
    if (
        not np.all(np.isfinite(terminal_augmented))
        or float(terminal_augmented[:, :components].min()) <= 0.0
    ):
        raise RuntimeError("state integration produced invalid states")
    normalized_objective = (
        terminal_augmented[:, components]
        + terminal_augmented[:, :components] @ parameters["alpha"]
    )

    def costate_rhs(physical_time: float, flat: np.ndarray) -> np.ndarray:
        states = forward.sol(physical_time).reshape(
            batch, components + 1
        )[:, :components]
        costates = flat.reshape(batch, components)
        controls = actions_batch(policy, physical_time, states)
        return -batch_dh_dn(
            states, costates, controls, parameters
        ).reshape(-1)

    terminal_costate = np.broadcast_to(
        parameters["alpha"], (batch, components)
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

    selected_states = (
        forward.sol(selected_time)
        .T.reshape(selected_time.size, batch, components + 1)[
            :, :, :components
        ]
    )
    selected_costates = (
        backward.sol(selected_time)
        .T.reshape(selected_time.size, batch, components)
    )
    selected_controls = np.stack(
        [
            actions_batch(policy, float(current_time), current_states)
            for current_time, current_states in zip(
                selected_time, selected_states
            )
        ],
        axis=0,
    )
    psi, dot_psi, ddot_psi = strict_quantities_batch(
        selected_states,
        selected_costates,
        selected_controls,
        parameters,
    )
    scaled = {
        "H_u": report_scale_factor * psi,
        "dH_u_dt": report_scale_factor * dot_psi,
        "d2H_u_dt2": report_scale_factor * ddot_psi,
    }
    rms = {
        quantity: np.sqrt(np.mean(values**2, axis=0))
        for quantity, values in scaled.items()
    }
    residual = np.sqrt(
        rms["H_u"] ** 2
        + rms["dH_u_dt"] ** 2
        + rms["d2H_u_dt2"] ** 2
    )
    result = {
        "normalized_objective": normalized_objective,
        "physical_objective": report_scale_factor * normalized_objective,
        **{f"{quantity}_rms": value for quantity, value in rms.items()},
        "R_sing": residual,
    }
    timing = {
        "batch_wall_seconds": time.perf_counter() - started,
        "forward_nfev": int(forward.nfev),
        "backward_nfev": int(backward.nfev),
    }
    return result, timing


def scalar_verification(
    policy: DensePolicy,
    initial_state: np.ndarray,
    dense_time: np.ndarray,
    mask: np.ndarray,
    batched: dict[str, np.ndarray],
    *,
    rtol: float,
    atol: float,
    report_scale_factor: float,
) -> dict[str, Any]:
    scalar = integrate_trajectory(
        policy,
        initial_state,
        dense_time,
        rtol=rtol,
        atol=atol,
        max_step=policy.cfg.T / (dense_time.size - 1),
    )
    expected: dict[str, float] = {
        "physical_objective": (
            report_scale_factor * scalar.normalized_objective
        )
    }
    for quantity in QUANTITIES:
        values = (
            report_scale_factor
            * np.asarray(scalar.quantities[quantity], dtype=np.float64)[mask]
        )
        expected[f"{quantity}_rms"] = float(
            np.sqrt(np.mean(values**2))
        )
    expected["R_sing"] = float(
        np.sqrt(
            sum(expected[f"{quantity}_rms"] ** 2 for quantity in QUANTITIES)
        )
    )
    differences = {
        key: abs(float(batched[key][0]) - value)
        for key, value in expected.items()
    }
    # DOP853 may take a slightly different adaptive path after the same
    # independent systems are packed into one vector.  The tolerance is
    # therefore relative to the reported physical value, while remaining
    # far below the precision displayed in Table 1.
    tolerances = {
        key: 2.0e-7 + 2.0e-6 * abs(value)
        for key, value in expected.items()
    }
    failed = {
        key: {
            "absolute_difference": differences[key],
            "tolerance": tolerances[key],
            "batched": float(batched[key][0]),
            "scalar": value,
        }
        for key, value in expected.items()
        if differences[key] > tolerances[key]
    }
    maximum = max(differences.values())
    if failed:
        raise RuntimeError(
            "batched evaluator does not match scalar evaluator: "
            + json.dumps(failed, sort_keys=True)
        )
    return {
        "scalar_expected": expected,
        "absolute_differences": differences,
        "maximum_absolute_difference": maximum,
        "absolute_plus_relative_tolerances": tolerances,
        "passed": True,
    }


def run(args: argparse.Namespace) -> None:
    if args.samples <= 0 or args.batch_size <= 0:
        raise ValueError("samples and batch-size must be positive")
    if args.sample_start < 0:
        raise ValueError("sample-start must be nonnegative")
    if args.dense_points < 3:
        raise ValueError("dense-points must be at least 3")
    if not 0.0 <= args.interior_start < args.interior_end:
        raise ValueError("invalid singular interval")
    torch.set_num_threads(args.torch_threads)
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)

    checkpoint = resolve(args.checkpoint)
    directions_path = resolve(args.directions)
    output_dir = resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} already exists; choose a new auditable output path"
        )
    if not checkpoint.is_file() or not directions_path.is_file():
        raise FileNotFoundError("checkpoint or directions file is missing")
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    load_started = time.perf_counter()
    dense_time = np.linspace(
        0.0, 10.0, args.dense_points, dtype=np.float64
    )
    policy, policy_metadata, policy_seconds = build_dense_policy(
        args.method,
        checkpoint,
        dense_time,
        device=device,
        query_batch_size=args.query_batch_size,
        cache_path=(
            resolve(args.dense_policy_cache)
            if args.dense_policy_cache is not None
            else None
        ),
    )
    if not np.isclose(policy.cfg.T, 10.0):
        raise ValueError(f"expected T=10, found {policy.cfg.T}")
    if policy.cfg.n != 800:
        raise ValueError(f"expected n=800, found {policy.cfg.n}")
    on_support, _, _ = build_grid_flags(dense_time, policy.cfg)
    mask = (
        (dense_time >= args.interior_start)
        & (dense_time < args.interior_end)
        & ~on_support
    )
    if not np.any(mask):
        raise ValueError("strict off-grid singular mask is empty")
    selected_time = dense_time[mask]

    all_directions = load_directions(
        directions_path, args.sample_start + args.samples, policy.cfg.m
    )
    directions = all_directions[
        args.sample_start : args.sample_start + args.samples
    ]
    initial_states = policy.cfg.n0 * (1.0 + args.radius * directions)
    if float(initial_states.min()) <= 0.0:
        raise ValueError("held-out state is nonpositive")

    collected: dict[str, list[np.ndarray]] = {
        "normalized_objective": [],
        "physical_objective": [],
        "H_u_rms": [],
        "dH_u_dt_rms": [],
        "d2H_u_dt2_rms": [],
        "R_sing": [],
    }
    batch_timings: list[dict[str, float]] = []
    verification: dict[str, Any] | None = None
    for start in range(0, args.samples, args.batch_size):
        stop = min(start + args.batch_size, args.samples)
        evaluated, timing = evaluate_batch(
            policy,
            initial_states[start:stop],
            selected_time,
            rtol=args.rtol,
            atol=args.atol,
            max_step=policy.cfg.T / (args.dense_points - 1),
            report_scale_factor=args.report_scale_factor,
        )
        for key in collected:
            collected[key].append(evaluated[key])
        batch_timings.append(
            {
                "start": start,
                "stop": stop,
                **timing,
            }
        )
        if args.verify_first_sample and start == 0:
            verification = scalar_verification(
                policy,
                initial_states[0],
                dense_time,
                mask,
                evaluated,
                rtol=args.rtol,
                atol=args.atol,
                report_scale_factor=args.report_scale_factor,
            )
        print(
            f"completed held-out states {start + 1}-{stop} / "
            f"{args.samples} in {timing['batch_wall_seconds']:.2f}s",
            flush=True,
        )

    arrays = {
        key: np.concatenate(parts, axis=0)
        for key, parts in collected.items()
    }
    rows: list[dict[str, Any]] = []
    for index in range(args.samples):
        rows.append(
            {
                "sample_index": args.sample_start + index,
                "initial_total": float(initial_states[index].sum()),
                "initial_min": float(initial_states[index].min()),
                "initial_max": float(initial_states[index].max()),
                **{key: float(value[index]) for key, value in arrays.items()},
            }
        )

    output_dir.mkdir(parents=True)
    with (output_dir / "per_sample.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        output_dir / "initial_states.npz",
        sample_index=np.arange(
            args.sample_start,
            args.sample_start + args.samples,
            dtype=np.int64,
        ),
        directions=directions,
        initial_states=initial_states,
    )
    summary = {
        "schema": "table1-heldout-continuous-pmp-v1",
        "method": args.method,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "directions": str(directions_path),
        "directions_sha256": sha256(directions_path),
        "samples": args.samples,
        "sample_start": args.sample_start,
        "sample_stop_exclusive": args.sample_start + args.samples,
        "radius": args.radius,
        "initial_state_definition": "N0_i = n0 * (1 + radius * direction_i)",
        "continuous_policy_protocol": {
            "fixed_transformer_support_context": True,
            "off_support_query": "direct masked Transformer query",
            "dense_logit_interpolant": "PCHIP",
            "state_solver": "DOP853",
            "costate": (
                "open-loop PMP adjoint along each realized closed-loop "
                "trajectory"
            ),
            "costate_solver": "DOP853",
            "rtol": args.rtol,
            "atol": args.atol,
            "maximum_step": policy.cfg.T / (args.dense_points - 1),
            "dense_points": args.dense_points,
            "singular_interval": [
                args.interior_start,
                args.interior_end,
            ],
            "strict_off_grid_only": True,
            "excluded_transformer_support_nodes": int(
                (
                    (dense_time >= args.interior_start)
                    & (dense_time < args.interior_end)
                    & on_support
                ).sum()
            ),
            "evaluated_coordinates_per_state": int(mask.sum()),
            "report_scale_factor": args.report_scale_factor,
            "R_sing": (
                "sqrt(RMS(H_u)^2 + RMS(dH_u/dt)^2 + "
                "RMS(d2H_u/dt2)^2)"
            ),
        },
        "aggregation_note": (
            "These statistics aggregate states within one retained training "
            "seed. Paper-table +/- across training seeds must be computed "
            "from independently trained per-seed held-out means."
        ),
        "policy_metadata": policy_metadata,
        "statistics_across_heldout_states": {
            key: statistics(value) for key, value in arrays.items()
        },
        "table_seed_values": {
            key: float(np.mean(value)) for key, value in arrays.items()
        },
        "verification": verification,
        "timing": {
            "policy_construction_seconds": policy_seconds,
            "total_seconds": time.perf_counter() - load_started,
            "batches": batch_timings,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_safe(summary), indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument(
        "--sample-start",
        type=int,
        default=0,
        help=(
            "zero-based first direction; use disjoint shards on a "
            "multi-core server"
        ),
    )
    parser.add_argument("--radius", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dense-points", type=int, default=25601)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument(
        "--dense-policy-cache",
        type=Path,
        help=(
            "optional shared NPZ cache for fixed-context dense time logits; "
            "concurrent shards coordinate through a file lock"
        ),
    )
    parser.add_argument("--interior-start", type=float, default=1.5)
    parser.add_argument("--interior-end", type=float, default=8.0)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--report-scale-factor", type=float, default=400.0)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verify-first-sample", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
