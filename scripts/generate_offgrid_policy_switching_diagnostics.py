#!/usr/bin/env python3
"""Evaluate learned policies directly at times outside the training grid.

This is a diagnostic-only script.  It does not modify the paper or any
checkpoint.  Every time query is evaluated against the fixed 801-token
training support sequence, rather than upsampling the ``n=800`` realized
control by zero-order hold or interpolation.  A masked 802-token Transformer
call keeps all support tokens independent of the added query, and lets the
query attend only to the fixed support.  Consequently, a query at a training
node exactly reconstructs the original policy output.  The evaluation grid
may add one or more strictly interior query points per training interval.

For state--time policies, the dense time-branch logits are used as a smooth
continuous input to the learned state correction while a high-accuracy
closed-loop state trajectory is integrated.  Every reported control is a
direct policy evaluation.  Continuous PMP state and
costate trajectories then give the scalar diagnostics

    H_u(t),  d H_u(t) / dt,  d^2 H_u(t) / dt^2.

Because a self-attention policy is sequence-to-sequence, feeding the complete
dense grid as a replacement sequence would change its attention context.  This script
avoids that ambiguity by fixing the original support context and reports the
on-grid reconstruction error as a numerical validation of the protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for search_path in (ROOT, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from evaluate_feedback_section5 import (  # noqa: E402
    configured_state_mode,
    load_feedback_checkpoint,
)
from generate_three_case_hamiltonian_results import (  # noqa: E402
    identity_checks,
    problem_from_config,
    resistant_heavy_initial_state,
)
from make_canonical_report_figures import strict_singular_quantities  # noqa: E402
from train_paper_pmp_kkt import (  # noqa: E402
    ProblemConfig,
    TimeTransformer,
    time_features,
)
from tumor_problem import dH_dN_numpy, dynamics_numpy  # noqa: E402


DEFAULT_TIME_CHECKPOINT = (
    ROOT
    / "outputs/new_objective_alpha1_grid_20260721_v2/a1_b40_g8000/"
    "time_only/deep_refine_kkt_head_ols_exact_v3/"
    "selected_checkpoint_physical.pt"
)
DEFAULT_CF_CHECKPOINT = (
    ROOT
    / "outputs/new_objective_alpha1_grid_20260721_v2/a1_b40_g8000/"
    "feedback/case1_cf_deep_v1_physical.pt"
)
DEFAULT_DER_CHECKPOINT = (
    ROOT
    / "outputs/scalar_only_der_continuation_20260725/"
    "formal_sharp_burden_dynamic_singularcal_cont100/"
    "best_feedback_section5_physical.pt"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "outputs/offgrid_policy_switching_diagnostics_20260725/"
    "dense1601_midpoint_query"
)

CASE_ORDER = ("time_only", "feedback_cf", "feedback_der")
CASE_LABELS = {
    "time_only": r"PMP/KKT time-only $u_\theta(t)$",
    "feedback_cf": r"PMP/KKT-CF $u_\theta(N,t)$",
    "feedback_der": r"PMP/KKT-DER $u_\theta(N,t)$",
}
STATE_ORDER = ("nominal", "resistant_heavy")
STATE_LABELS = {
    "nominal": "Nominal",
    "resistant_heavy": "Resistant-heavy",
}
STATE_STYLES = {
    "nominal": {"color": "#2563A6", "linestyle": "-", "linewidth": 1.15},
    "resistant_heavy": {
        "color": "#C25536",
        "linestyle": "--",
        "linewidth": 1.15,
    },
}
QUANTITY_ORDER = ("H_u", "dH_u_dt", "d2H_u_dt2")
QUANTITY_LABELS = {
    "H_u": r"$H_u(t)=\psi(t)$",
    "dH_u_dt": r"$\mathrm{d}H_u/\mathrm{d}t=\dot{\psi}(t)$",
    "d2H_u_dt2": r"$\mathrm{d}^2H_u/\mathrm{d}t^2=\ddot{\psi}(t)$",
}


def resolve(path: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate.resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_time_logits(model: TimeTransformer, normalized_time: torch.Tensor) -> torch.Tensor:
    hidden = model.input(time_features(normalized_time))
    hidden = model.encoder(hidden.unsqueeze(0)).squeeze(0)
    return model.output(hidden).squeeze(-1)


def fixed_support_query_logits(
    model: TimeTransformer,
    support_time: torch.Tensor,
    query_time: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    """Query arbitrary times without changing the training support context.

    Each batch item contains the same support tokens followed by one query
    token.  The attention mask prevents support tokens from attending to the
    query and prevents the query from attending to itself.  Thus its only
    attention context is the original support sequence.  If the query equals
    one support coordinate, all layer inputs and attention sets match that
    support token exactly, so the output must reconstruct the original logit.
    """

    if support_time.ndim != 1 or query_time.ndim != 1:
        raise ValueError("support_time and query_time must be one-dimensional")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    support_count = int(support_time.numel())
    sequence_length = support_count + 1
    mask = torch.zeros(
        (sequence_length, sequence_length),
        dtype=torch.bool,
        device=support_time.device,
    )
    mask[:support_count, support_count] = True
    mask[support_count, support_count] = True
    support_features = time_features(support_time)
    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, int(query_time.numel()), batch_size):
            current = query_time[start : start + batch_size]
            batch = int(current.numel())
            features = torch.cat(
                [
                    support_features.unsqueeze(0).expand(batch, -1, -1),
                    time_features(current).unsqueeze(1),
                ],
                dim=1,
            )
            hidden = model.input(features)
            encoded = model.encoder(hidden, mask=mask)
            outputs.append(model.output(encoded[:, -1, :]).squeeze(-1))
    return torch.cat(outputs, dim=0)


def load_time_model(
    path: Path,
) -> tuple[TimeTransformer, ProblemConfig, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ProblemConfig(**checkpoint["problem"])
    settings = dict(checkpoint.get("base_model_args", checkpoint.get("args", {})))
    required = ("d_model", "heads", "layers", "init_u")
    missing = [key for key in required if key not in settings]
    if missing:
        raise KeyError(f"{path}: missing time-model settings {missing}")
    model = TimeTransformer(
        int(settings["d_model"]),
        int(settings["heads"]),
        int(settings["layers"]),
        float(cfg.umax),
        float(settings["init_u"]),
    ).double()
    source_state = checkpoint["model_state"]
    if any(key.startswith("base.") for key in source_state):
        source_state = {
            key.removeprefix("base."): value
            for key, value in source_state.items()
            if key.startswith("base.")
        }
    elif any(key.startswith("time_branch.") for key in source_state):
        source_state = {
            key.removeprefix("time_branch."): value
            for key, value in source_state.items()
            if key.startswith("time_branch.")
        }
    model.load_state_dict(source_state)
    model.eval()
    wrapper = dict(checkpoint.get("wrapper", {}))
    return model, cfg, wrapper


def time_action_from_raw(
    raw: np.ndarray,
    cfg: ProblemConfig,
    wrapper: dict[str, Any],
) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    wrapper_class = str(wrapper.get("class", ""))
    if wrapper_class == "LinearRawBoxProjection":
        return np.clip(raw, 0.0, cfg.umax)
    probability = 1.0 / (1.0 + np.exp(-raw))
    if wrapper_class == "AffineBoundaryProjectedControl":
        scale = float(wrapper["scale"])
        offset = float(wrapper["offset"])
        base_control = cfg.umax * probability
        return np.clip(scale * base_control - offset, 0.0, cfg.umax)
    if wrapper_class == "FixedBoxProjection":
        scale = float(wrapper["scale"])
        temperature = float(wrapper.get("temperature", 1.0))
        logit = np.log(
            np.clip(probability, 1.0e-12, 1.0 - 1.0e-12)
            / np.clip(1.0 - probability, 1.0e-12, 1.0)
        )
        return np.clip(
            scale * cfg.umax / (1.0 + np.exp(-logit / temperature)),
            0.0,
            cfg.umax,
        )
    if wrapper_class == "BoundaryProjectedControl":
        scale = float(wrapper.get("scale", wrapper.get("initial_scale", 1.0)))
        return np.clip(scale * cfg.umax * probability, 0.0, cfg.umax)
    return cfg.umax * probability


def numpy_time_features(normalized_time: float) -> np.ndarray:
    value = float(normalized_time)
    return np.asarray(
        [
            value,
            value**2,
            math.sin(2.0 * math.pi * value),
            math.cos(2.0 * math.pi * value),
            math.sin(4.0 * math.pi * value),
            math.cos(4.0 * math.pi * value),
        ],
        dtype=np.float64,
    )


@dataclass
class DensePolicy:
    case_id: str
    cfg: ProblemConfig
    dense_time: np.ndarray
    dense_raw_logits: np.ndarray
    coarse_raw_logits: np.ndarray
    checkpoint: Path
    feedback_model: torch.nn.Module | None = None
    feedback_args: argparse.Namespace | None = None
    time_wrapper: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self._raw_interpolator = PchipInterpolator(
            self.dense_time,
            self.dense_raw_logits,
            extrapolate=False,
        )
        self._linears: list[tuple[np.ndarray, np.ndarray]] = []
        self._nominal_reference: np.ndarray | None = None
        self._feature_r: np.ndarray | None = None
        self._feature_phi: np.ndarray | None = None
        if self.feedback_model is not None:
            linears = [
                module
                for module in self.feedback_model.state_branch.modules()
                if isinstance(module, torch.nn.Linear)
            ]
            self._linears = [
                (
                    layer.weight.detach().cpu().numpy().astype(np.float64),
                    layer.bias.detach().cpu().numpy().astype(np.float64),
                )
                for layer in linears
            ]
            self._nominal_reference = (
                self.feedback_model.nominal_reference.detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            self._feature_r = (
                self.feedback_model.feature_r.detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            self._feature_phi = (
                self.feedback_model.feature_phi.detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )

    def raw_at(self, time: float) -> float:
        clipped = float(np.clip(time, 0.0, self.cfg.T))
        return float(self._raw_interpolator(clipped))

    def reference_at(self, normalized_time: float) -> np.ndarray:
        if self._nominal_reference is None:
            raise RuntimeError("time-only policy has no nominal reference")
        last = self._nominal_reference.shape[0] - 1
        position = float(np.clip(normalized_time * last, 0.0, last))
        lower = int(math.floor(position))
        upper = min(lower + 1, last)
        fraction = position - lower
        return (
            (1.0 - fraction) * self._nominal_reference[lower]
            + fraction * self._nominal_reference[upper]
        )

    def state_features(
        self,
        normalized_time: float,
        state: np.ndarray,
    ) -> np.ndarray:
        if self.feedback_model is None or self.feedback_args is None:
            raise RuntimeError("state features requested for time-only policy")
        mode = str(getattr(self.feedback_args, "state_feature_mode", "log_absolute"))
        state = np.asarray(state, dtype=np.float64)
        reference = self.reference_at(normalized_time)
        if mode == "log_absolute":
            scale = float(self.feedback_model.state_log_scale)
            component = np.log1p(state) / scale
            summaries = np.asarray(
                [component.mean(), component.std(), component.max()],
                dtype=np.float64,
            )
        elif mode == "relative_nominal":
            component = (state - reference) / np.maximum(reference, 1.0e-8)
            summaries = np.asarray(
                [component.mean(), component.std(), component.max()],
                dtype=np.float64,
            )
        elif mode == "total_burden":
            total = max(float(state.sum()), np.finfo(np.float64).eps)
            reference_total = max(
                float(reference.sum()), np.finfo(np.float64).eps
            )
            component = np.zeros_like(state)
            summaries = np.asarray(
                [math.log(total / reference_total), 0.0, 0.0],
                dtype=np.float64,
            )
        elif mode == "burden_composition":
            if self._feature_r is None or self._feature_phi is None:
                raise RuntimeError("feature vectors are unavailable")
            eps = np.finfo(np.float64).eps
            total = max(float(state.sum()), eps)
            reference_total = max(float(reference.sum()), eps)
            composition = state / total
            reference_composition = reference / reference_total
            log_composition = np.log(np.maximum(composition, eps))
            log_reference = np.log(np.maximum(reference_composition, eps))
            component = (
                log_composition - log_composition.mean()
                - log_reference
                + log_reference.mean()
            )
            delta = composition - reference_composition
            r_centered = self._feature_r - self._feature_r.mean()
            phi_centered = self._feature_phi - self._feature_phi.mean()
            r_scale = max(float(np.sqrt(np.mean(r_centered**2))), eps)
            phi_scale = max(float(np.sqrt(np.mean(phi_centered**2))), eps)
            summaries = np.asarray(
                [
                    math.log(total / reference_total),
                    float(np.dot(delta, r_centered) / r_scale),
                    float(np.dot(delta, phi_centered) / phi_scale),
                ],
                dtype=np.float64,
            )
        else:
            raise ValueError(f"unsupported state feature mode {mode!r}")
        return np.concatenate(
            [numpy_time_features(normalized_time), component, summaries]
        )

    def state_logits(
        self,
        normalized_time: float,
        states: list[np.ndarray],
    ) -> np.ndarray:
        """Evaluate one or more state-branch inputs in a single matrix pass."""

        value = np.stack(
            [
                self.state_features(normalized_time, state)
                for state in states
            ],
            axis=0,
        )
        for index, (weight, bias) in enumerate(self._linears):
            value = value @ weight.T + bias
            if index < len(self._linears) - 1:
                value = np.tanh(value)
        return np.asarray(value, dtype=np.float64).reshape(len(states), -1)[:, 0]

    def state_logit(self, normalized_time: float, state: np.ndarray) -> float:
        return float(self.state_logits(normalized_time, [state])[0])

    def action(self, time: float, state: np.ndarray) -> float:
        raw = self.raw_at(time)
        if self.feedback_model is None:
            transformed = time_action_from_raw(
                np.asarray([raw]),
                self.cfg,
                self.time_wrapper or {},
            )
            return float(transformed[0])
        if self.feedback_args is None:
            raise RuntimeError("feedback arguments are unavailable")
        normalized_time = float(np.clip(time / self.cfg.T, 0.0, 1.0))
        if bool(getattr(self.feedback_args, "center_state_correction", False)):
            reference = self.reference_at(normalized_time)
            logits = self.state_logits(normalized_time, [state, reference])
            correction = float(logits[0] - logits[1])
        else:
            correction = self.state_logit(normalized_time, state)
        combined = (
            raw
            + float(getattr(self.feedback_args, "correction_gain", 1.0))
            * correction
        )
        parameterization = str(
            getattr(
                self.feedback_args,
                "action_parameterization",
                "logit-temperature",
            )
        )
        if parameterization == "linear-raw-box":
            return float(np.clip(combined, 0.0, self.cfg.umax))
        scale = float(getattr(self.feedback_args, "action_scale", 1.0))
        offset = float(getattr(self.feedback_args, "action_offset", 0.0))
        temperature = float(
            getattr(self.feedback_args, "action_temperature", 1.0)
        )
        return float(
            np.clip(
                scale
                * self.cfg.umax
                / (1.0 + math.exp(-combined / temperature))
                - offset,
                0.0,
                self.cfg.umax,
            )
        )


@dataclass
class TrajectoryResult:
    time: np.ndarray
    state: np.ndarray
    costate: np.ndarray
    control: np.ndarray
    hamiltonian: np.ndarray
    quantities: dict[str, np.ndarray]
    identity_errors: dict[str, float]
    normalized_running_cost: float
    normalized_objective: float


def validate_numpy_feedback_action(policy: DensePolicy) -> float | None:
    if policy.feedback_model is None or policy.feedback_args is None:
        return None
    rng = np.random.default_rng(20260725)
    errors: list[float] = []
    for _ in range(16):
        time = float(rng.uniform(0.0, policy.cfg.T))
        state = policy.cfg.n0 * rng.uniform(0.8, 1.2, size=policy.cfg.m)
        normalized = time / policy.cfg.T
        raw = policy.raw_at(time)
        with torch.inference_mode():
            torch_action = policy.feedback_model.interval_action(
                torch.tensor(raw, dtype=torch.float64),
                torch.tensor([normalized], dtype=torch.float64),
                torch.from_numpy(state).to(torch.float64).unsqueeze(0),
                state_mode=configured_state_mode(policy.feedback_args),
            )
        numpy_action = policy.action(time, state)
        errors.append(abs(float(torch_action.item()) - numpy_action))
    maximum = max(errors)
    if maximum > 2.0e-11:
        raise RuntimeError(
            f"{policy.case_id}: NumPy policy evaluator differs from PyTorch by "
            f"{maximum:.3e}"
        )
    return maximum


def integrate_trajectory(
    policy: DensePolicy,
    initial_state: np.ndarray,
    dense_time: np.ndarray,
    *,
    rtol: float,
    atol: float,
    max_step: float,
) -> TrajectoryResult:
    problem = problem_from_config(policy.cfg)
    params = problem.vectors()

    def state_rhs(time: float, augmented: np.ndarray) -> np.ndarray:
        state = augmented[: problem.m]
        action = policy.action(time, state)
        running = float(params["beta"] @ state + problem.gamma * action)
        return np.concatenate(
            (
                dynamics_numpy(state, action, problem),
                np.asarray([running], dtype=np.float64),
            )
        )

    state_solution = solve_ivp(
        state_rhs,
        (0.0, problem.T),
        np.concatenate(
            (
                np.asarray(initial_state, dtype=np.float64),
                np.zeros(1, dtype=np.float64),
            )
        ),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=True,
        max_step=max_step,
    )
    if not state_solution.success:
        raise RuntimeError(f"state integration failed: {state_solution.message}")
    augmented_state = np.asarray(
        state_solution.sol(dense_time), dtype=np.float64
    ).T
    state = augmented_state[:, : problem.m]
    if not np.all(np.isfinite(state)) or float(state.min()) <= 0.0:
        raise RuntimeError("closed-loop state is nonfinite or nonpositive")

    def costate_rhs(time: float, costate: np.ndarray) -> np.ndarray:
        current_state = np.asarray(
            state_solution.sol(time), dtype=np.float64
        )[: problem.m]
        control = policy.action(time, current_state)
        return -dH_dN_numpy(current_state, costate, control, problem, params)

    costate_solution = solve_ivp(
        costate_rhs,
        (problem.T, 0.0),
        params["alpha"].copy(),
        method="DOP853",
        rtol=rtol,
        atol=atol,
        dense_output=True,
        max_step=max_step,
    )
    if not costate_solution.success:
        raise RuntimeError(
            f"costate integration failed: {costate_solution.message}"
        )
    costate = np.asarray(costate_solution.sol(dense_time), dtype=np.float64).T
    control = np.asarray(
        [
            policy.action(float(time), current_state)
            for time, current_state in zip(dense_time, state)
        ],
        dtype=np.float64,
    )
    diagnostic = {
        "diagnostic_N": state,
        "diagnostic_lambda": costate,
        "diagnostic_u": control,
    }
    quantities = strict_singular_quantities(diagnostic, problem)
    errors = identity_checks(state, costate, control, quantities, problem)
    drift = np.asarray(
        [
            dynamics_numpy(current_state, float(action), problem)
            for current_state, action in zip(state, control)
        ]
    )
    hamiltonian = (
        (params["beta"][None, :] * state).sum(axis=1)
        + problem.gamma * control
        + (costate * drift).sum(axis=1)
    )
    normalized_running_cost = float(augmented_state[-1, problem.m])
    normalized_objective = float(
        params["alpha"] @ state[-1] + normalized_running_cost
    )
    return TrajectoryResult(
        time=dense_time,
        state=state,
        costate=costate,
        control=control,
        hamiltonian=hamiltonian,
        quantities={
            "H_u": np.asarray(quantities["psi"], dtype=np.float64),
            "dH_u_dt": np.asarray(quantities["dot_psi"], dtype=np.float64),
            "d2H_u_dt2": np.asarray(quantities["ddot_psi"], dtype=np.float64),
        },
        identity_errors=errors,
        normalized_running_cost=normalized_running_cost,
        normalized_objective=normalized_objective,
    )


def build_grid_flags(
    dense_time: np.ndarray,
    cfg: ProblemConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaled = dense_time / cfg.T * cfg.n
    nearest_index = np.rint(scaled).astype(np.int64)
    nearest_index = np.clip(nearest_index, 0, cfg.n)
    nearest_time = nearest_index / cfg.n * cfg.T
    distance = np.abs(dense_time - nearest_time)
    tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, cfg.T)
    on_grid = distance <= tolerance
    return on_grid, nearest_index, distance


def build_refinement_flags(
    dense_time: np.ndarray,
    cfg: ProblemConfig,
    refinement_multiplier: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify scalar-refinement queries and strictly held-out dense times."""

    if refinement_multiplier < 1:
        raise ValueError("refinement_multiplier must be positive")
    dense_intervals = dense_time.size - 1
    refinement_intervals = cfg.n * refinement_multiplier
    if dense_intervals % refinement_intervals:
        raise ValueError(
            "dense-points - 1 must be divisible by "
            "n * refinement-multiplier"
        )
    stride = dense_intervals // refinement_intervals
    on_refinement_grid = np.arange(dense_time.size) % stride == 0
    on_transformer_support, _, _ = build_grid_flags(dense_time, cfg)
    refinement_query = on_refinement_grid & ~on_transformer_support
    held_out = ~on_refinement_grid
    return on_refinement_grid, refinement_query, held_out


def write_timeseries(
    path: Path,
    results: dict[str, dict[str, TrajectoryResult]],
    on_grid: np.ndarray,
    refinement_query: np.ndarray,
    held_out_from_refinement: np.ndarray,
    nearest_index: np.ndarray,
    distance: np.ndarray,
) -> None:
    fields = (
        "case_id",
        "case_label",
        "state_id",
        "state_label",
        "t",
        "is_training_grid",
        "is_off_grid",
        "is_refinement_query",
        "is_held_out_from_refinement",
        "nearest_training_grid_index",
        "distance_to_training_grid",
        "u",
        "total_population",
        "H",
        "H_u",
        "dH_u_dt",
        "d2H_u_dt2",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for case_id in CASE_ORDER:
            for state_id in STATE_ORDER:
                result = results[case_id][state_id]
                for index, time in enumerate(result.time):
                    writer.writerow(
                        {
                            "case_id": case_id,
                            "case_label": CASE_LABELS[case_id].replace("$", ""),
                            "state_id": state_id,
                            "state_label": STATE_LABELS[state_id],
                            "t": f"{float(time):.16g}",
                            "is_training_grid": int(on_grid[index]),
                            "is_off_grid": int(not on_grid[index]),
                            "is_refinement_query": int(
                                refinement_query[index]
                            ),
                            "is_held_out_from_refinement": int(
                                held_out_from_refinement[index]
                            ),
                            "nearest_training_grid_index": int(
                                nearest_index[index]
                            ),
                            "distance_to_training_grid": f"{distance[index]:.16g}",
                            "u": f"{result.control[index]:.16g}",
                            "total_population": (
                                f"{result.state[index].sum():.16g}"
                            ),
                            "H": f"{result.hamiltonian[index]:.16g}",
                            "H_u": f"{result.quantities['H_u'][index]:.16g}",
                            "dH_u_dt": (
                                f"{result.quantities['dH_u_dt'][index]:.16g}"
                            ),
                            "d2H_u_dt2": (
                                f"{result.quantities['d2H_u_dt2'][index]:.16g}"
                            ),
                        }
                    )


def build_summary_rows(
    results: dict[str, dict[str, TrajectoryResult]],
    on_grid: np.ndarray,
    on_refinement_grid: np.ndarray,
    refinement_query: np.ndarray,
    held_out_from_refinement: np.ndarray,
    *,
    interior_start: float,
    interior_end: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_id in CASE_ORDER:
        for state_id in STATE_ORDER:
            result = results[case_id][state_id]
            regions = {
                "full_horizon": np.ones(result.time.shape, dtype=bool),
                "interior": (result.time >= interior_start)
                & (result.time < interior_end),
            }
            subsets = {
                "all_points": np.ones(result.time.shape, dtype=bool),
                "training_grid_coordinates": on_grid,
                "off_grid_coordinates": ~on_grid,
                "transformer_support": on_grid,
                "refinement_queries": refinement_query,
                "refinement_grid": on_refinement_grid,
                "held_out_from_refinement": held_out_from_refinement,
            }
            for region, region_mask in regions.items():
                for subset, subset_mask in subsets.items():
                    mask = region_mask & subset_mask
                    for quantity in QUANTITY_ORDER:
                        values = result.quantities[quantity][mask]
                        if values.size == 0:
                            rms = None
                            mean_abs = None
                            max_abs = None
                        else:
                            absolute = np.abs(values)
                            rms = float(np.sqrt(np.mean(values**2)))
                            mean_abs = float(np.mean(absolute))
                            max_abs = float(np.max(absolute))
                        rows.append(
                            {
                                "case_id": case_id,
                                "state_id": state_id,
                                "region": region,
                                "subset": subset,
                                "quantity": quantity,
                                "count": int(values.size),
                                "rms": rms,
                                "mean_abs": mean_abs,
                                "max_abs": max_abs,
                            }
                        )
    return rows


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "stixsans",
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_results(
    path: Path,
    results: dict[str, dict[str, TrajectoryResult]],
    on_grid: np.ndarray,
    refinement_query: np.ndarray,
    held_out_from_refinement: np.ndarray,
    *,
    xlim: tuple[float, float],
    title: str,
) -> None:
    configure_plotting()
    figure, axes = plt.subplots(
        4,
        3,
        figsize=(10.5, 7.2),
        sharex=True,
        constrained_layout=False,
    )
    for column, case_id in enumerate(CASE_ORDER):
        axes[0, column].set_title(CASE_LABELS[case_id], pad=5.0)
        for state_id in STATE_ORDER:
            result = results[case_id][state_id]
            visible = (result.time >= xlim[0]) & (result.time <= xlim[1])
            style = STATE_STYLES[state_id]
            axes[0, column].plot(
                result.time[visible],
                result.control[visible],
                label=STATE_LABELS[state_id],
                **style,
            )
            for row, quantity in enumerate(QUANTITY_ORDER, start=1):
                axes[row, column].plot(
                    result.time[visible],
                    result.quantities[quantity][visible],
                    **style,
                )
        nominal = results[case_id]["nominal"]
        visible = (nominal.time >= xlim[0]) & (nominal.time <= xlim[1])
        on_indices = np.flatnonzero(on_grid & visible)[::40]
        refinement_indices = np.flatnonzero(
            refinement_query & visible
        )[::280]
        held_out_indices = np.flatnonzero(
            held_out_from_refinement & visible
        )[::280]
        marker_values = (
            nominal.control,
            nominal.quantities["H_u"],
            nominal.quantities["dH_u_dt"],
            nominal.quantities["d2H_u_dt2"],
        )
        for row, values in enumerate(marker_values):
            axes[row, column].scatter(
                nominal.time[on_indices],
                values[on_indices],
                marker="o",
                s=12,
                linewidths=0.65,
                facecolors="white",
                edgecolors="#222222",
                zorder=5,
                label="training-grid coordinate" if row == 0 else None,
            )
            axes[row, column].scatter(
                nominal.time[refinement_indices],
                values[refinement_indices],
                marker="|",
                s=18,
                linewidths=0.8,
                color="#555555",
                zorder=5,
                label="scalar-refinement query" if row == 0 else None,
            )
            axes[row, column].scatter(
                nominal.time[held_out_indices],
                values[held_out_indices],
                marker="x",
                s=12,
                linewidths=0.7,
                color="#222222",
                zorder=5,
                label="held-out dense query" if row == 0 else None,
            )
        for row in range(4):
            axes[row, column].set_xlim(*xlim)
            axes[row, column].grid(alpha=0.18, linewidth=0.45)
            axes[row, column].axhline(
                0.0, color="#666666", linewidth=0.55, alpha=0.65
            )
        axes[0, column].set_ylim(-0.08, 3.08)
    axes[0, 0].set_ylabel(r"control $u_\theta$")
    for row, quantity in enumerate(QUANTITY_ORDER, start=1):
        axes[row, 0].set_ylabel(QUANTITY_LABELS[quantity])
    for axis in axes[-1]:
        axis.set_xlabel(r"time $t$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.018),
    )
    figure.suptitle(title, y=0.982, fontsize=10.5, fontweight="semibold")
    figure.text(
        0.5,
        0.052,
        (
            f"Lines use {on_grid.size:,} direct policy queries; "
            f"{int(held_out_from_refinement.sum()):,} query times were not "
            "used by scalar refinement. Markers are thinned for visibility."
        ),
        ha="center",
        va="center",
        fontsize=7.2,
        color="#333333",
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.105,
        top=0.925,
        hspace=0.17,
        wspace=0.22,
    )
    figure.savefig(path, dpi=240)
    figure.savefig(path.with_suffix(".png"), dpi=240)
    plt.close(figure)


def evaluate(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.torch_threads)
    # Match the differentiable Transformer path used during refinement.
    # PyTorch's fused inference-only MHA fast path is numerically different
    # enough to perturb this highly sensitive control problem.
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    time_checkpoint = resolve(args.time_checkpoint)
    cf_checkpoint = resolve(args.cf_checkpoint)
    der_checkpoint = resolve(args.der_checkpoint)

    time_model, time_cfg, time_wrapper = load_time_model(time_checkpoint)
    cf_model, cf_cfg, cf_args = load_feedback_checkpoint(cf_checkpoint)
    der_model, der_cfg, der_args = load_feedback_checkpoint(der_checkpoint)
    if time_cfg != cf_cfg or time_cfg != der_cfg:
        raise ValueError("the three checkpoints use different physical problems")
    if args.dense_points < 3:
        raise ValueError("dense-points must be at least 3")
    cfg = time_cfg
    dense_time = np.linspace(0.0, cfg.T, args.dense_points, dtype=np.float64)
    dense_normalized = torch.from_numpy(dense_time / cfg.T).to(torch.float64)
    coarse_normalized = torch.linspace(0.0, 1.0, cfg.n + 1, dtype=torch.float64)
    on_grid, nearest_index, distance = build_grid_flags(dense_time, cfg)
    if int(on_grid.sum()) != cfg.n + 1:
        raise RuntimeError(
            f"expected {cfg.n + 1} common grid coordinates, found "
            f"{int(on_grid.sum())}"
        )
    off_grid_count = int((~on_grid).sum())
    if off_grid_count == 0:
        raise RuntimeError("the diagnostic grid contains no off-grid times")
    (
        on_refinement_grid,
        refinement_query,
        held_out_from_refinement,
    ) = build_refinement_flags(
        dense_time,
        cfg,
        args.refinement_multiplier,
    )
    off_grid_tensor = torch.from_numpy(~on_grid)
    on_grid_indices = torch.from_numpy(nearest_index[on_grid])

    policy_specs: dict[str, DensePolicy] = {}
    with torch.inference_mode():
        time_coarse_raw = raw_time_logits(time_model, coarse_normalized)
        cf_coarse_raw = cf_model.time_logits(coarse_normalized)
        der_coarse_raw = der_model.time_logits(coarse_normalized)
    query_cache = out_dir / "fixed_support_query_cache.npz"
    cache_hashes = {
        "time_checkpoint_sha256": sha256(time_checkpoint),
        "cf_checkpoint_sha256": sha256(cf_checkpoint),
        "der_checkpoint_sha256": sha256(der_checkpoint),
    }
    cache_loaded = False
    if query_cache.is_file():
        cached = np.load(query_cache, allow_pickle=False)
        cache_loaded = (
            np.array_equal(cached["dense_time"], dense_time)
            and all(
                str(cached[key].item()) == value
                for key, value in cache_hashes.items()
            )
        )
        if cache_loaded:
            print(f"Reusing {query_cache}", flush=True)
            time_dense_raw = torch.from_numpy(cached["time_dense_raw"])
            cf_dense_raw = torch.from_numpy(cached["cf_dense_raw"])
            der_dense_raw = torch.from_numpy(cached["der_dense_raw"])
    if not cache_loaded:
        time_dense_raw = torch.empty_like(dense_normalized)
        cf_dense_raw = torch.empty_like(dense_normalized)
        der_dense_raw = torch.empty_like(dense_normalized)
        time_dense_raw[torch.from_numpy(on_grid)] = time_coarse_raw[on_grid_indices]
        cf_dense_raw[torch.from_numpy(on_grid)] = cf_coarse_raw[on_grid_indices]
        der_dense_raw[torch.from_numpy(on_grid)] = der_coarse_raw[on_grid_indices]
        print("[time_only] strictly off-grid fixed-support queries", flush=True)
        time_dense_raw[off_grid_tensor] = fixed_support_query_logits(
            time_model,
            coarse_normalized,
            dense_normalized[off_grid_tensor],
            batch_size=args.query_batch_size,
        )
        print("[feedback_cf] strictly off-grid fixed-support queries", flush=True)
        cf_dense_raw[off_grid_tensor] = fixed_support_query_logits(
            cf_model.time_branch,
            coarse_normalized,
            dense_normalized[off_grid_tensor],
            batch_size=args.query_batch_size,
        )
        if all(
            torch.equal(
                cf_model.time_branch.state_dict()[key],
                der_model.time_branch.state_dict()[key],
            )
            for key in cf_model.time_branch.state_dict()
        ):
            print(
                "[feedback_der] reusing identical dense time branch",
                flush=True,
            )
            der_dense_raw = cf_dense_raw.clone()
        else:
            print("[feedback_der] strictly off-grid fixed-support queries", flush=True)
            der_dense_raw[off_grid_tensor] = fixed_support_query_logits(
                der_model.time_branch,
                coarse_normalized,
                dense_normalized[off_grid_tensor],
                batch_size=args.query_batch_size,
            )
        np.savez_compressed(
            query_cache,
            dense_time=dense_time,
            time_dense_raw=time_dense_raw.cpu().numpy(),
            cf_dense_raw=cf_dense_raw.cpu().numpy(),
            der_dense_raw=der_dense_raw.cpu().numpy(),
            **{key: np.asarray(value) for key, value in cache_hashes.items()},
        )
        print(f"Cached direct query logits at {query_cache}", flush=True)
    policy_specs["time_only"] = DensePolicy(
        "time_only",
        cfg,
        dense_time,
        time_dense_raw.cpu().numpy(),
        time_coarse_raw.cpu().numpy(),
        time_checkpoint,
        time_wrapper=time_wrapper,
    )
    policy_specs["feedback_cf"] = DensePolicy(
        "feedback_cf",
        cfg,
        dense_time,
        cf_dense_raw.cpu().numpy(),
        cf_coarse_raw.cpu().numpy(),
        cf_checkpoint,
        feedback_model=cf_model,
        feedback_args=cf_args,
    )
    policy_specs["feedback_der"] = DensePolicy(
        "feedback_der",
        cfg,
        dense_time,
        der_dense_raw.cpu().numpy(),
        der_coarse_raw.cpu().numpy(),
        der_checkpoint,
        feedback_model=der_model,
        feedback_args=der_args,
    )

    problem = problem_from_config(cfg)
    states = {
        "nominal": np.full(problem.m, problem.n0, dtype=np.float64),
        "resistant_heavy": resistant_heavy_initial_state(problem),
    }
    numpy_action_errors = {
        case_id: validate_numpy_feedback_action(policy)
        for case_id, policy in policy_specs.items()
    }
    results: dict[str, dict[str, TrajectoryResult]] = {}
    max_step = (
        args.max_step
        if args.max_step is not None
        else cfg.T / (args.dense_points - 1)
    )
    for case_id in CASE_ORDER:
        results[case_id] = {}
        for state_id in STATE_ORDER:
            print(
                f"[{case_id}/{state_id}] dense direct-query closed-loop PMP",
                flush=True,
            )
            results[case_id][state_id] = integrate_trajectory(
                policy_specs[case_id],
                states[state_id],
                dense_time,
                rtol=args.rtol,
                atol=args.atol,
                max_step=max_step,
            )

    rows = build_summary_rows(
        results,
        on_grid,
        on_refinement_grid,
        refinement_query,
        held_out_from_refinement,
        interior_start=args.interior_start,
        interior_end=args.interior_end,
    )
    write_summary(out_dir / "offgrid_switching_summary.csv", rows)
    write_timeseries(
        out_dir / "offgrid_switching_timeseries.csv",
        results,
        on_grid,
        refinement_query,
        held_out_from_refinement,
        nearest_index,
        distance,
    )
    arrays: dict[str, np.ndarray] = {
        "time": dense_time,
        "is_training_grid": on_grid.astype(np.int8),
        "is_transformer_support": on_grid.astype(np.int8),
        "is_refinement_query": refinement_query.astype(np.int8),
        "is_held_out_from_refinement": held_out_from_refinement.astype(
            np.int8
        ),
        "nearest_training_grid_index": nearest_index,
        "distance_to_training_grid": distance,
    }
    for case_id in CASE_ORDER:
        for state_id in STATE_ORDER:
            result = results[case_id][state_id]
            prefix = f"{case_id}__{state_id}"
            arrays[f"{prefix}__u"] = result.control
            arrays[f"{prefix}__state"] = result.state
            arrays[f"{prefix}__costate"] = result.costate
            arrays[f"{prefix}__H"] = result.hamiltonian
            arrays[f"{prefix}__normalized_objective"] = np.asarray(
                result.normalized_objective
            )
            arrays[f"{prefix}__physical_objective"] = np.asarray(
                args.report_scale_factor * result.normalized_objective
            )
            for quantity in QUANTITY_ORDER:
                arrays[f"{prefix}__{quantity}"] = result.quantities[quantity]
    np.savez_compressed(out_dir / "offgrid_trajectories.npz", **arrays)

    plot_results(
        out_dir / "offgrid_policy_switching_full_horizon.pdf",
        results,
        on_grid,
        refinement_query,
        held_out_from_refinement,
        xlim=(0.0, cfg.T),
        title="Direct off-grid policy queries and scalar switching-function diagnostics",
    )
    plot_results(
        out_dir / "offgrid_policy_switching_interior.pdf",
        results,
        on_grid,
        refinement_query,
        held_out_from_refinement,
        xlim=(args.interior_start, args.interior_end),
        title="Direct off-grid policy queries on the interior interval",
    )

    common_dense_indices = np.flatnonzero(on_grid)
    reconstruction: dict[str, dict[str, float]] = {}
    for case_id, policy in policy_specs.items():
        dense_common = policy.dense_raw_logits[common_dense_indices]
        raw_difference = dense_common - policy.coarse_raw_logits
        record = {
            "raw_logit_max_abs": float(np.max(np.abs(raw_difference))),
            "raw_logit_mean_abs": float(np.mean(np.abs(raw_difference))),
            "raw_logit_rms": float(np.sqrt(np.mean(raw_difference**2))),
        }
        if case_id == "time_only":
            dense_action = time_action_from_raw(
                dense_common, cfg, policy.time_wrapper or {}
            )
            coarse_action = time_action_from_raw(
                policy.coarse_raw_logits, cfg, policy.time_wrapper or {}
            )
            action_difference = dense_action - coarse_action
            record.update(
                {
                    "control_max_abs": float(
                        np.max(np.abs(action_difference))
                    ),
                    "control_mean_abs": float(
                        np.mean(np.abs(action_difference))
                    ),
                    "control_rms": float(
                        np.sqrt(np.mean(action_difference**2))
                    ),
                }
            )
        reconstruction[case_id] = record

    offgrid_examples = dense_time[~on_grid]
    near_one_point_five = offgrid_examples[
        np.argsort(np.abs(offgrid_examples - 1.5))[:4]
    ]
    nearest_offgrid_text = ", ".join(
        f"{float(value):.7f}" for value in sorted(near_one_point_five)
    )
    manifest = {
        "schema": "direct-offgrid-policy-switching-diagnostics-v1",
        "query_semantics": {
            "time_branch": (
                "Each query is appended to the fixed 801-token training support "
                "sequence. A masked 802-token Transformer call keeps support "
                "tokens independent of the query and lets the query attend "
                f"only to support. All {dense_time.size:,} plotted controls are "
                "direct policy queries, not an interpolation or zero-order "
                "hold of the n=800 realized controls."
            ),
            "feedback": (
                "At every reported time, the checkpoint state branch is queried "
                "with the continuous closed-loop state. PCHIP is used only to "
                "supply the already-directly-queried time-branch logits at "
                "internal ODE-solver stages."
            ),
            "attention_context_protocol": (
                "The Transformer is sequence-to-sequence. The fixed-support "
                "mask prevents the attention-context drift that would occur if "
                f"all {dense_time.size:,} times were supplied as a replacement "
                "sequence. "
                "On-grid reconstruction error is reported explicitly."
            ),
        },
        "problem": {
            "T": cfg.T,
            "n_training_intervals": cfg.n,
            "m": cfg.m,
            "umax": cfg.umax,
            "alpha": cfg.alpha,
            "beta": cfg.beta,
            "gamma": cfg.gamma,
        },
        "evaluation_grid": {
            "points": int(dense_time.size),
            "step": float(dense_time[1] - dense_time[0]),
            "training_grid_coordinates": int(on_grid.sum()),
            "strictly_off_grid_coordinates": off_grid_count,
            "off_grid_fraction": float((~on_grid).mean()),
            "scalar_refinement_multiplier": args.refinement_multiplier,
            "scalar_refinement_query_coordinates": int(
                refinement_query.sum()
            ),
            "held_out_from_scalar_refinement_coordinates": int(
                held_out_from_refinement.sum()
            ),
            "off_grid_examples_near_t_1_5": sorted(
                float(value) for value in near_one_point_five
            ),
            "interior_summary_interval": (
                f"{args.interior_start:g} <= t < {args.interior_end:g}"
            ),
            "note": (
                "For n=800 and T=10, t=1.5 is itself a training-grid "
                f"coordinate; nearby genuinely off-grid examples in this "
                f"evaluation are {nearest_offgrid_text}."
            ),
        },
        "initial_states": {
            key: [float(value) for value in state]
            for key, state in states.items()
        },
        "solver": {
            "method": "DOP853",
            "rtol": args.rtol,
            "atol": args.atol,
            "max_step": max_step,
        },
        "checkpoints": {
            case_id: {
                "path": str(policy.checkpoint),
                "sha256": sha256(policy.checkpoint),
            }
            for case_id, policy in policy_specs.items()
        },
        "numpy_feedback_action_max_abs_error": numpy_action_errors,
        "fixed_support_on_grid_reconstruction": reconstruction,
        "identity_checks": {
            case_id: {
                state_id: result.identity_errors
                for state_id, result in state_results.items()
            }
            for case_id, state_results in results.items()
        },
        "objectives": {
            case_id: {
                state_id: {
                    "normalized": result.normalized_objective,
                    "physical": (
                        args.report_scale_factor
                        * result.normalized_objective
                    ),
                }
                for state_id, result in state_results.items()
            }
            for case_id, state_results in results.items()
        },
        "outputs": {
            "timeseries_csv": "offgrid_switching_timeseries.csv",
            "summary_csv": "offgrid_switching_summary.csv",
            "trajectory_npz": "offgrid_trajectories.npz",
            "full_horizon_pdf": "offgrid_policy_switching_full_horizon.pdf",
            "interior_pdf": "offgrid_policy_switching_interior.pdf",
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["evaluation_grid"], indent=2), flush=True)
    print(f"Wrote {out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--time-checkpoint", type=Path, default=DEFAULT_TIME_CHECKPOINT
    )
    parser.add_argument("--cf-checkpoint", type=Path, default=DEFAULT_CF_CHECKPOINT)
    parser.add_argument(
        "--der-checkpoint", type=Path, default=DEFAULT_DER_CHECKPOINT
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dense-points", type=int, default=1601)
    parser.add_argument(
        "--refinement-multiplier",
        type=int,
        default=1,
        help=(
            "number of scalar-refinement subintervals per original Transformer "
            "interval; denser coordinates are reported as held out"
        ),
    )
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--interior-start", type=float, default=1.0)
    parser.add_argument("--interior-end", type=float, default=8.2)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--max-step", type=float)
    parser.add_argument(
        "--report-scale-factor",
        type=float,
        default=400.0,
        help="convert normalized training weights to the reported physical scale",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
