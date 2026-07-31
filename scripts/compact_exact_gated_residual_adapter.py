#!/usr/bin/env python3
"""Compact exact-gated residual adapter for a frozen DER feedback policy.

The adapter reads the existing policy's state features and contains one small
hidden layer.  Its scalar logit correction is multiplied by an exact flat-tube
gate, which is exactly zero on the protected source-policy trajectories and
uses a quintic smoothstep across its transition band.  The original DER policy
is frozen and no full probe branch is copied.  Because the protected reference
is piecewise linear and uses a hard nearest-trajectory minimum, the gate is not
claimed to be globally C2.

This module defines the native artifact/evaluator contract.  Teacher controls
may initialize the compact adapter, but a final artifact must record that the
teacher was used only during initialization and that subsequent checkpoint
selection used scalar PMP/DER residuals.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.refine_feedback_gated_residual_adapter import (  # noqa: E402
    ProtectedTrajectoryGate,
    calibrated_zero_tube_squared,
)
from scripts.continue_feedback_full_state_gated_fallback import (  # noqa: E402
    continuous_policy_stage_trace,
    load_gated_feedback_checkpoint,
)
from scripts.evaluate_feedback_section5 import (  # noqa: E402
    load_feedback_checkpoint,
)
from scripts.feedback_continuous_policy_rk4 import (  # noqa: E402
    continuous_feedback_pmp_pack,
    pchip_midpoint_logits,
)
from scripts.initialize_feedback_capability_distillation import (  # noqa: E402
    capability_query_batch,
    pointwise_action_mse,
    validate_positive_teacher_audit,
)
from scripts.refine_feedback_last_layer_near_null import (  # noqa: E402
    iid_and_composition_states,
)
from scripts.refine_feedback_svd_null_projected_kkt import (  # noqa: E402
    PolicyRollout,
    antithetic_initial_states,
    posthoc_test_metrics,
    structured_initial_states,
)
from scripts.train_feedback_section5 import (  # noqa: E402
    NestedFeedbackTransformer,
)
from train_paper_pmp_kkt import (  # noqa: E402
    ProblemConfig,
    build_params,
    parse_hidden,
)


COMPACT_ADAPTER_FORMAT = "exact_gated_compact_residual_adapter_v1"
GATE_TRANSITION = "quintic_smoothstep_c2"
REQUIRED_PROTECTED_RADII = (0.0, 0.10, 0.20)
SUPPORTED_WIDTHS = (8, 16, 32)
DEVELOPMENT_AUDIT_SEED = 641_903_202
RESERVED_FINAL_BLIND_SEED = 202_617_01


def source_feature_dimension(source: NestedFeedbackTransformer) -> int:
    """Infer the exact input dimension of the existing state features."""

    parameter = next(source.parameters())
    time = torch.zeros(
        1, device=parameter.device, dtype=parameter.dtype
    )
    state = torch.full(
        (1, source.m),
        10.0,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    return int(source.state_features(time, state).shape[-1])


def compact_parameter_count(input_dimension: int, width: int) -> int:
    """Return parameters in Linear(d,w)-Tanh-Linear(w,1)."""

    if input_dimension < 1 or width < 1:
        raise ValueError("input dimension and width must be positive")
    return (input_dimension + 1) * width + (width + 1)


class CompactStateFeatureAdapter(nn.Module):
    """One-hidden-layer scalar adapter over existing DER state features."""

    def __init__(
        self,
        input_dimension: int,
        width: int,
        *,
        activation: str = "tanh",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if width not in SUPPORTED_WIDTHS:
            raise ValueError(
                f"adapter width must be one of {SUPPORTED_WIDTHS}"
            )
        if activation != "tanh":
            raise ValueError("the compact adapter currently requires tanh")
        self.input_dimension = int(input_dimension)
        self.width = int(width)
        self.activation_name = activation
        self.input = nn.Linear(
            input_dimension,
            width,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.output = nn.Linear(
            width,
            1,
            bias=True,
            device=device,
            dtype=dtype,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.zeros_(self.input.bias)
        # Exact zero output makes installation identity-preserving before
        # teacher initialization.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output(torch.tanh(self.input(features))).squeeze(-1)


class ExactGatedCompactFeedback(nn.Module):
    """Frozen ordinary DER plus a small exact-gated scalar correction."""

    def __init__(
        self,
        source: NestedFeedbackTransformer,
        gate: ProtectedTrajectoryGate,
        *,
        width: int,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        self.source = source
        for parameter in self.source.parameters():
            parameter.requires_grad_(False)
        source_parameter = next(source.parameters())
        dimension = source_feature_dimension(source)
        self.adapter = CompactStateFeatureAdapter(
            dimension,
            width,
            activation=activation,
            device=source_parameter.device,
            dtype=source_parameter.dtype,
        )
        self.gate = gate

    @property
    def adapter_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.adapter.parameters())

    @property
    def m(self) -> int:
        return int(self.source.m)

    @property
    def umax(self) -> float:
        return float(self.source.umax)

    @property
    def time_branch(self) -> nn.Module:
        """Expose the frozen temporal branch to common dense-grid utilities."""

        return self.source.time_branch

    def train(self, mode: bool = True) -> "ExactGatedCompactFeedback":
        super().train(mode)
        self.source.eval()
        return self

    def time_logits(self, normalized_time_grid: torch.Tensor) -> torch.Tensor:
        return self.source.time_logits(normalized_time_grid)

    def residual_logit(
        self,
        normalized_time: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.source.state_features(normalized_time, state)
        gate = self.gate(normalized_time, state)
        return gate * self.adapter(features), gate

    def interval_action(
        self,
        base_logit: torch.Tensor,
        normalized_time: torch.Tensor,
        state: torch.Tensor,
        *,
        state_blind: bool = False,
        state_mode: str | None = None,
    ) -> torch.Tensor:
        if state_mode is None:
            state_mode = "fixed_query" if state_blind else "feedback"
        if state_mode == "w_zero":
            source_correction = torch.zeros(
                state.shape[0],
                device=state.device,
                dtype=state.dtype,
            )
            adapter_correction = torch.zeros_like(source_correction)
        elif state_mode == "fixed_query":
            query = torch.full_like(state, 10.0)
            source_correction = self.source.state_logits(
                normalized_time, query
            )
            adapter_correction, _ = self.residual_logit(
                normalized_time, query
            )
        elif state_mode == "feedback":
            source_correction = self.source.state_logits(
                normalized_time, state
            )
            adapter_correction, _ = self.residual_logit(
                normalized_time, state
            )
        else:
            raise ValueError(f"unknown state mode: {state_mode}")
        combined = (
            base_logit
            + self.source.correction_gain * source_correction
            + adapter_correction
        )
        if self.source.action_parameterization == "linear-raw-box":
            return torch.clamp(combined, 0.0, self.source.umax)
        return torch.clamp(
            self.source.action_scale
            * self.source.umax
            * torch.sigmoid(
                combined / self.source.action_temperature
            )
            - self.source.action_offset,
            0.0,
            self.source.umax,
        )


def _constructor_from_payload(
    payload: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[
    NestedFeedbackTransformer,
    ExactGatedCompactFeedback,
    ProblemConfig,
]:
    source_args = argparse.Namespace(**payload["source_args"])
    cfg = ProblemConfig(**payload["problem"])
    source = NestedFeedbackTransformer(
        cfg.m,
        cfg.umax,
        source_args.state_scale,
        parse_hidden(source_args.state_hidden),
        source_args.d_model,
        source_args.heads,
        source_args.layers,
        source_args.init_u,
        source_args.correction_gain,
        source_args.state_feature_mode,
        bool(source_args.center_state_correction),
        float(source_args.action_temperature),
        float(source_args.action_scale),
        str(source_args.action_parameterization),
        float(source_args.action_offset),
    ).to(device=device, dtype=torch.float64)
    source.load_state_dict(payload["source_model_state"])
    source.set_nominal_reference(
        payload["nominal_reference"].to(
            device=device, dtype=torch.float64
        )
    )
    params = build_params(cfg, device, torch.float64)
    source.set_feature_vectors(params["r"], params["phi"])
    source.eval()
    gate = ProtectedTrajectoryGate(
        payload["protected_states"].to(
            device=device, dtype=torch.float64
        ),
        distance_scale=float(payload["gate_distance_scale"]),
        zero_tube_squared=float(payload["gate_zero_tube_squared"]),
    ).to(device=device, dtype=torch.float64)
    spec = payload["adapter_spec"]
    model = ExactGatedCompactFeedback(
        source,
        gate,
        width=int(spec["width"]),
        activation=str(spec["activation"]),
    ).to(device=device, dtype=torch.float64)
    model.adapter.load_state_dict(payload["adapter_state"])
    model.eval()
    return source, model, cfg


def validate_compact_artifact(payload: dict[str, Any]) -> None:
    if payload.get("format") != COMPACT_ADAPTER_FORMAT:
        raise ValueError("unsupported compact exact-gated artifact")
    if payload.get("gate_transition") != GATE_TRANSITION:
        raise ValueError(
            "compact artifact does not use the exact flat-tube gate with "
            "quintic transition"
        )
    radii = tuple(float(value) for value in payload["protected_radii"])
    if radii != REQUIRED_PROTECTED_RADII:
        raise ValueError("protected radii must be exactly 0,.10,.20")
    spec = payload.get("adapter_spec", {})
    width = int(spec.get("width", -1))
    if width not in SUPPORTED_WIDTHS:
        raise ValueError("unsupported compact adapter width")
    if spec.get("activation") != "tanh":
        raise ValueError("unsupported compact adapter activation")
    protocol = payload.get("training_protocol", {})
    if protocol.get("teacher_role") != "initialization_only":
        raise ValueError("teacher must be restricted to initialization")
    if protocol.get("cleanup_loss") != "scalar_PMP_DER":
        raise ValueError("final cleanup must use scalar PMP/DER")
    if protocol.get("checkpoint_selection") != "validation_PMP_residual":
        raise ValueError("checkpoint selection must use validation PMP residual")
    if (
        protocol.get("identity_evaluator")
        != "native_continuous_policy_RK4_stage_requery"
    ):
        raise ValueError(
            "protected identity was not audited on the native continuous "
            "policy"
        )
    if (
        protocol.get("transition_query_acceptance")
        != "cleanup_train_validation_flat_regions_only"
    ):
        raise ValueError("gate transition acceptance contract is missing")
    if (
        protocol.get("physical_objective_used_in_cleanup_loss")
        is not False
    ):
        raise ValueError("physical objective entered scalar PMP cleanup")
    if (
        protocol.get(
            "physical_objective_used_in_checkpoint_selection"
        )
        is not False
    ):
        raise ValueError("physical objective entered checkpoint selection")
    if protocol.get("physical_objective_used_for_teacher_admission") is not True:
        raise ValueError("teacher admission provenance omits its J-based audit")
    cleanup = protocol.get("cleanup", {})
    for key in (
        "selected_train_PMP_gate_distribution",
        "selected_validation_PMP_gate_distribution",
    ):
        distribution = cleanup.get(key, {})
        if distribution.get("transition_fraction") != 0.0:
            raise ValueError(
                f"artifact does not certify flat-region PMP queries: {key}"
            )


def load_compact_adapter_artifact(
    path: Path,
    *,
    device: torch.device,
) -> tuple[
    dict[str, Any],
    NestedFeedbackTransformer,
    ExactGatedCompactFeedback,
    ProblemConfig,
]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_compact_artifact(payload)
    source, model, cfg = _constructor_from_payload(
        payload, device=device
    )
    spec = payload["adapter_spec"]
    expected_dimension = source_feature_dimension(source)
    if int(spec["input_dimension"]) != expected_dimension:
        raise ValueError("adapter input dimension does not match state features")
    expected_count = compact_parameter_count(
        expected_dimension, int(spec["width"])
    )
    if int(spec["parameter_count"]) != expected_count:
        raise ValueError("recorded compact parameter count is inconsistent")
    if model.adapter_parameter_count != expected_count:
        raise ValueError("loaded compact parameter count is inconsistent")
    node_time = torch.linspace(
        0.0,
        1.0,
        model.gate.protected_states.shape[1],
        device=device,
        dtype=torch.float64,
    )
    protected = model.gate.protected_states
    node_gate = model.gate(
        node_time.repeat(protected.shape[0]),
        protected.reshape(-1, cfg.m),
    )
    if not torch.equal(node_gate, torch.zeros_like(node_gate)):
        raise ValueError("compact gate is not exactly zero on protected nodes")
    params = build_params(cfg, device, torch.float64)
    protected_initial = structured_initial_states(
        REQUIRED_PROTECTED_RADII,
        cfg,
        device,
        torch.float64,
    )
    with torch.no_grad():
        source_trace = continuous_policy_trace(
            source, protected_initial, cfg, params
        )
        reconstructed_states = _node_states_from_trace(source_trace)
    if not torch.equal(
        reconstructed_states, model.gate.protected_states
    ):
        raise ValueError(
            "stored protected states are not the source native continuous "
            "policy trajectories"
        )
    identity = continuous_policy_identity_metrics(
        source, model, protected_initial, cfg, params
    )
    _assert_exact_identity(identity)
    return payload, source, model, cfg


def compact_artifact_payload(
    *,
    source_payload: dict[str, Any],
    source: NestedFeedbackTransformer,
    model: ExactGatedCompactFeedback,
    cfg: ProblemConfig,
    protected_radii: tuple[float, ...],
    gate_zero_tube_squared: float,
    teacher_provenance: dict[str, Any],
    cleanup_provenance: dict[str, Any],
) -> dict[str, Any]:
    """Create a self-contained, evaluator-compatible compact artifact."""

    if tuple(protected_radii) != REQUIRED_PROTECTED_RADII:
        raise ValueError("protected radii must be exactly 0,.10,.20")
    if teacher_provenance.get("role") != "initialization_only":
        raise ValueError("teacher provenance must be initialization-only")
    if cleanup_provenance.get("loss") != "scalar_PMP_DER":
        raise ValueError("cleanup provenance must use scalar PMP/DER")
    if cleanup_provenance.get("selection") != "validation_PMP_residual":
        raise ValueError("cleanup provenance must select by PMP residual")
    dimension = source_feature_dimension(source)
    width = int(model.adapter.width)
    count = compact_parameter_count(dimension, width)
    if count != model.adapter_parameter_count:
        raise RuntimeError("compact adapter parameter inventory is inconsistent")
    source_args = copy.deepcopy(source_payload["args"])
    return {
        "format": COMPACT_ADAPTER_FORMAT,
        "source_args": source_args,
        "problem": asdict(cfg),
        "source_model_state": {
            key: value.detach().cpu().clone()
            for key, value in source.state_dict().items()
        },
        "nominal_reference": source.nominal_reference.detach().cpu().clone(),
        "protected_states": model.gate.protected_states.detach().cpu().clone(),
        "protected_radii": list(protected_radii),
        "gate_distance_scale": float(model.gate.distance_scale),
        "gate_zero_tube_squared": float(gate_zero_tube_squared),
        "gate_transition": GATE_TRANSITION,
        "gate_description": (
            "exact flat tube; quintic transition; no global C2 claim"
        ),
        "adapter_spec": {
            "family": "state_feature_single_hidden",
            "input_dimension": dimension,
            "width": width,
            "activation": model.adapter.activation_name,
            "parameter_count": count,
        },
        "adapter_state": {
            key: value.detach().cpu().clone()
            for key, value in model.adapter.state_dict().items()
        },
        "training_protocol": {
            "teacher_role": "initialization_only",
            "teacher": copy.deepcopy(teacher_provenance),
            "cleanup_loss": "scalar_PMP_DER",
            "cleanup": copy.deepcopy(cleanup_provenance),
            "checkpoint_selection": "validation_PMP_residual",
            "identity_evaluator": (
                "native_continuous_policy_RK4_stage_requery"
            ),
            "transition_query_acceptance": (
                "cleanup_train_validation_flat_regions_only"
            ),
            "physical_objective_used_in_cleanup_loss": False,
            "physical_objective_used_in_checkpoint_selection": False,
            "physical_objective_used_for_teacher_admission": True,
            "physical_objective_used_in_development_audit": True,
        },
    }


def parameter_inventory_for_paper_model(
    *,
    feature_dimension: int = 30,
) -> dict[int, dict[str, float | int]]:
    """Return exact counts and fractions of the 20,609-parameter DER branch."""

    full_branch = 20_609
    return {
        width: {
            "parameters": compact_parameter_count(feature_dimension, width),
            "fraction_of_20609": (
                compact_parameter_count(feature_dimension, width) / full_branch
            ),
            "percent_of_20609": (
                100.0
                * compact_parameter_count(feature_dimension, width)
                / full_branch
            ),
        }
        for width in SUPPORTED_WIDTHS
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_indices(
    count: int,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    if count < 1 or batch_size < 1:
        raise ValueError("dataset and batch sizes must be positive")
    return torch.randint(
        count,
        (min(count, batch_size),),
        generator=generator,
        device="cpu",
    ).to(device=device)


def _teacher_focus_distribution(
    queries: dict[str, torch.Tensor],
    *,
    target_mode: str,
    quantile: float,
    uniform_fraction: float,
    focus_power: float,
) -> tuple[torch.Tensor, dict[str, float | str]]:
    """Return a uniform/focused mixture over capability-bearing queries."""

    if target_mode == "absolute_teacher":
        increment = queries["teacher_action"] - queries["source_action"]
    elif target_mode == "capability_increment":
        increment = queries["teacher_increment"]
    else:
        raise ValueError(f"unknown teacher target mode: {target_mode}")
    if not 0.0 <= quantile < 1.0:
        raise ValueError("teacher focus quantile must lie in [0,1)")
    if not 0.0 < uniform_fraction <= 1.0:
        raise ValueError("teacher uniform fraction must lie in (0,1]")
    if focus_power <= 0.0:
        raise ValueError("teacher focus power must be positive")

    magnitude = increment.detach().abs()
    threshold = torch.quantile(magnitude, quantile)
    focused_score = (magnitude - threshold).clamp_min(0.0).pow(focus_power)
    count = magnitude.numel()
    uniform = torch.full_like(magnitude, 1.0 / count)
    if float(focused_score.sum()) == 0.0:
        focused = uniform
    else:
        focused = focused_score / focused_score.sum()
    probability = (
        uniform_fraction * uniform
        + (1.0 - uniform_fraction) * focused
    )
    probability = probability / probability.sum()
    return probability, {
        "target_mode": target_mode,
        "focus_statistic": (
            "|teacher_action-source_action|"
            if target_mode == "absolute_teacher"
            else "|teacher_action-locked_teacher_action|"
        ),
        "focus_quantile": quantile,
        "focus_threshold": float(threshold),
        "focus_power": focus_power,
        "uniform_probability_mass": uniform_fraction,
        "focused_probability_mass": 1.0 - uniform_fraction,
        "nonzero_increment_fraction": float((magnitude > 0.0).double().mean()),
        "above_threshold_fraction": float(
            (magnitude > threshold).double().mean()
        ),
        "increment_mean_abs": float(magnitude.mean()),
        "increment_rms": float(magnitude.square().mean().sqrt()),
        "increment_max_abs": float(magnitude.max()),
    }


def _teacher_target_batch(
    queries: dict[str, torch.Tensor],
    target_mode: str,
) -> dict[str, torch.Tensor]:
    """Choose the teacher target without changing the recorded query data."""

    batch = dict(queries)
    if target_mode == "absolute_teacher":
        batch["target_action"] = queries["teacher_action"]
    elif target_mode != "capability_increment":
        raise ValueError(f"unknown teacher target mode: {target_mode}")
    return batch


def _sample_focused_indices(
    probability: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if probability.ndim != 1 or probability.numel() < 1:
        raise ValueError("sampling probability must be a nonempty vector")
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    return torch.multinomial(
        probability.cpu(),
        min(probability.numel(), batch_size),
        replacement=True,
        generator=generator,
    ).to(device=probability.device)


def _probability_weighted_action_mse(
    model: ExactGatedCompactFeedback,
    batch: dict[str, torch.Tensor],
    probability: torch.Tensor,
) -> torch.Tensor:
    prediction = model.interval_action(
        batch["student_base_logit"],
        batch["normalized_time"],
        batch["state"],
        state_mode="feedback",
    )
    return (
        probability
        * (prediction - batch["target_action"]).square()
    ).sum()


def _scalar_pmp_pack(
    model: ExactGatedCompactFeedback,
    initial: torch.Tensor,
    state_weights: torch.Tensor | None,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
    *,
    interval_start: float,
    interval_end: float,
    w0: float,
    w1: float,
    w2: float,
) -> dict[str, torch.Tensor]:
    """Evaluate only scalar DER residuals on the native continuous policy."""

    time_grid = torch.linspace(
        0.0,
        1.0,
        cfg.n + 1,
        device=initial.device,
        dtype=initial.dtype,
    )
    node_logits = model.time_logits(time_grid)
    midpoint_logits = pchip_midpoint_logits(time_grid, node_logits)
    continuous = continuous_feedback_pmp_pack(
        model,
        initial,
        cfg,
        time_grid,
        node_logits,
        midpoint_logits,
        params,
        state_mode="feedback",
    )
    sample_time = time_grid[:-1] * cfg.T
    weighted_mask = (
        (sample_time >= interval_start)
        & (sample_time < interval_end)
    ).to(initial.dtype).unsqueeze(0)
    if state_weights is None:
        loss_mask = weighted_mask
        loss_denominator = (
            weighted_mask.sum() * initial.shape[0]
        ).clamp_min(torch.finfo(initial.dtype).eps)
    else:
        weights = state_weights.to(
            device=initial.device, dtype=initial.dtype
        )
        loss_mask = weighted_mask * weights[:, None]
        loss_denominator = (
            weighted_mask.sum() * weights.sum()
        ).clamp_min(torch.finfo(initial.dtype).eps)

    def mean_square(value: torch.Tensor) -> torch.Tensor:
        return (loss_mask * value.square()).sum() / loss_denominator

    component_losses = {
        "H_u": mean_square(continuous.quantities["psi"]),
        "dH_u_dt": mean_square(continuous.quantities["dot_psi"]),
        "d2H_u_dt2": mean_square(continuous.quantities["ddot_psi"]),
    }
    loss = (
        w0 * component_losses["H_u"]
        + w1 * component_losses["dH_u_dt"]
        + w2 * component_losses["d2H_u_dt2"]
    )
    return {
        "loss": loss,
        "component_losses": component_losses,
        "quantities": continuous.quantities,
        "weighted_mask": weighted_mask,
        "denominator": (
            weighted_mask.sum() * initial.shape[0]
        ).clamp_min(torch.finfo(initial.dtype).eps),
        "node_states": continuous.states,
        "midpoint_states": continuous.midpoint_states,
        "node_controls": continuous.node_controls,
        "midpoint_controls": continuous.midpoint_controls,
    }


def _pmp_residual_metrics(
    pack: dict[str, torch.Tensor], *, scale: float
) -> dict[str, float]:
    mask = pack["weighted_mask"]
    denominator = pack["denominator"]
    result: dict[str, float] = {}
    for label, key in (
        ("H_u", "psi"),
        ("dH_u_dt", "dot_psi"),
        ("d2H_u_dt2", "ddot_psi"),
    ):
        value = pack["quantities"][key]
        active = mask.expand_as(value) > 0.0
        rms = ((mask * value.square()).sum() / denominator).sqrt()
        result[f"{label}_physical_rms"] = float(
            scale * rms.detach().cpu()
        )
        result[f"{label}_physical_max_abs"] = float(
            scale * value[active].abs().max().detach().cpu()
        )
    return result


def continuous_policy_trace(
    model: torch.nn.Module,
    initial: torch.Tensor,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
    *,
    state_mode: str = "feedback",
) -> dict[str, torch.Tensor]:
    """Run the native policy with a fresh query at every RK4 stage."""

    normalized_time = torch.linspace(
        0.0,
        1.0,
        cfg.n + 1,
        device=initial.device,
        dtype=initial.dtype,
    )
    node_logits = model.time_logits(normalized_time)
    midpoint_logits = pchip_midpoint_logits(
        normalized_time, node_logits
    )
    return continuous_policy_stage_trace(
        model,
        initial,
        cfg,
        normalized_time,
        node_logits,
        midpoint_logits,
        params,
        state_mode=state_mode,
    )


def _node_states_from_trace(
    trace: dict[str, torch.Tensor],
) -> torch.Tensor:
    stage_states = trace["states"]
    return torch.cat(
        (
            stage_states[:, :, 0, :],
            trace["terminal_state"].unsqueeze(1),
        ),
        dim=1,
    )


def _continuous_trace_objective(
    trace: dict[str, torch.Tensor],
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
) -> torch.Tensor:
    stage_running = (
        (trace["states"] * params["beta"]).sum(dim=-1)
        + params["gamma"] * trace["controls"]
    )
    weights = torch.tensor(
        (1.0, 2.0, 2.0, 1.0),
        device=stage_running.device,
        dtype=stage_running.dtype,
    )
    running = (
        (cfg.T / cfg.n)
        * (stage_running * weights).sum(dim=(1, 2))
        / 6.0
    )
    terminal = (
        trace["terminal_state"] * params["alpha"]
    ).sum(dim=-1)
    return terminal + running


@torch.no_grad()
def continuous_policy_rollout(
    model: torch.nn.Module,
    initial: torch.Tensor,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
    *,
    state_mode: str = "feedback",
) -> PolicyRollout:
    trace = continuous_policy_trace(
        model, initial, cfg, params, state_mode=state_mode
    )
    return PolicyRollout(
        states=_node_states_from_trace(trace),
        controls=trace["controls"],
        stage_states=trace["states"],
        objectives=_continuous_trace_objective(trace, cfg, params),
    )


@torch.no_grad()
def continuous_policy_identity_metrics(
    source: NestedFeedbackTransformer,
    adapted: ExactGatedCompactFeedback,
    initial: torch.Tensor,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Bit-exact identity audit on the native continuous RK4 policy."""

    source_trace = continuous_policy_trace(source, initial, cfg, params)
    adapted_trace = continuous_policy_trace(adapted, initial, cfg, params)
    batch = initial.shape[0]
    stage_time = source_trace["normalized_time"].unsqueeze(0).expand(
        batch, -1, -1
    )
    stage_gate = adapted.gate(
        stage_time.reshape(-1),
        source_trace["states"].reshape(-1, cfg.m),
    )
    source_nodes = _node_states_from_trace(source_trace)
    node_time = torch.linspace(
        0.0,
        1.0,
        cfg.n + 1,
        device=initial.device,
        dtype=initial.dtype,
    )
    node_gate = adapted.gate(
        node_time.repeat(batch),
        source_nodes.reshape(-1, cfg.m),
    )
    source_objective = _continuous_trace_objective(
        source_trace, cfg, params
    )
    adapted_objective = _continuous_trace_objective(
        adapted_trace, cfg, params
    )
    return {
        "gate_max_abs": float(
            torch.cat((stage_gate, node_gate)).abs().max().cpu()
        ),
        "control_max_abs": float(
            (
                source_trace["controls"] - adapted_trace["controls"]
            ).abs().max().cpu()
        ),
        "state_max_abs": float(
            max(
                (
                    source_trace["states"] - adapted_trace["states"]
                ).abs().max().cpu(),
                (
                    source_trace["terminal_state"]
                    - adapted_trace["terminal_state"]
                ).abs().max().cpu(),
            )
        ),
        "objective_max_abs": float(
            (source_objective - adapted_objective).abs().max().cpu()
        ),
    }


@torch.no_grad()
def continuous_gate_distribution(
    model: ExactGatedCompactFeedback,
    initial: torch.Tensor,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Classify native stage queries as flat-zero, transition, or flat-one."""

    trace = continuous_policy_trace(model, initial, cfg, params)
    batch = initial.shape[0]
    stage_time = trace["normalized_time"].unsqueeze(0).expand(
        batch, -1, -1
    )
    value = model.gate(
        stage_time.reshape(-1),
        trace["states"].reshape(-1, cfg.m),
    )
    zero = value == 0.0
    one = value == 1.0
    transition = ~(zero | one)
    return {
        "minimum": float(value.min().cpu()),
        "mean": float(value.mean().cpu()),
        "maximum": float(value.max().cpu()),
        "flat_zero_fraction": float(zero.double().mean().cpu()),
        "transition_fraction": float(
            transition.double().mean().cpu()
        ),
        "flat_one_fraction": float(one.double().mean().cpu()),
    }


@torch.no_grad()
def pmp_pack_gate_distribution(
    model: ExactGatedCompactFeedback,
    pack: dict[str, torch.Tensor],
    cfg: ProblemConfig,
) -> dict[str, float]:
    """Classify exactly the node/midpoint queries used by the PMP pack."""

    node_states = pack["node_states"]
    midpoint_states = pack["midpoint_states"]
    batch = node_states.shape[0]
    node_time = torch.linspace(
        0.0,
        1.0,
        cfg.n + 1,
        device=node_states.device,
        dtype=node_states.dtype,
    )
    midpoint_time = 0.5 * (node_time[:-1] + node_time[1:])
    node_gate = model.gate(
        node_time.repeat(batch),
        node_states.reshape(-1, cfg.m),
    )
    midpoint_gate = model.gate(
        midpoint_time.repeat(batch),
        midpoint_states.reshape(-1, cfg.m),
    )
    value = torch.cat((node_gate, midpoint_gate))
    zero = value == 0.0
    one = value == 1.0
    transition = ~(zero | one)
    return {
        "minimum": float(value.min().cpu()),
        "mean": float(value.mean().cpu()),
        "maximum": float(value.max().cpu()),
        "flat_zero_fraction": float(zero.double().mean().cpu()),
        "transition_fraction": float(
            transition.double().mean().cpu()
        ),
        "flat_one_fraction": float(one.double().mean().cpu()),
    }


def _assert_no_transition_queries(
    label: str, distribution: dict[str, float]
) -> None:
    if distribution["transition_fraction"] != 0.0:
        raise RuntimeError(
            f"{label} contains gate-transition queries; scalar time "
            "derivatives would depend on the piecewise reference: "
            f"{distribution}"
        )


def _assert_exact_identity(identity: dict[str, float]) -> None:
    for name in (
        "gate_max_abs",
        "control_max_abs",
        "state_max_abs",
        "objective_max_abs",
    ):
        if identity[name] != 0.0:
            raise RuntimeError(
                f"protected-trajectory identity is not exact: "
                f"{name}={identity[name]}"
            )


def paired_bootstrap_mean(
    values: torch.Tensor,
    *,
    seed: int,
    repeats: int,
) -> dict[str, float | int]:
    """Development-only paired bootstrap interval for a mean difference."""

    if repeats < 1:
        raise ValueError("bootstrap repeats must be positive")
    sample = values.detach().to(device="cpu", dtype=torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    means: list[torch.Tensor] = []
    block = 1000
    for start in range(0, repeats, block):
        count = min(block, repeats - start)
        indices = torch.randint(
            sample.numel(),
            (count, sample.numel()),
            generator=generator,
            device="cpu",
        )
        means.append(sample[indices].mean(dim=1))
    bootstrap = torch.cat(means)
    return {
        "repeats": repeats,
        "seed": seed,
        "lower_95": float(torch.quantile(bootstrap, 0.025)),
        "upper_95": float(torch.quantile(bootstrap, 0.975)),
    }


def _compatibility_check(
    student_cfg: ProblemConfig,
    teacher_cfg: ProblemConfig,
    source_args: argparse.Namespace,
) -> None:
    for name in ("T", "n", "m", "umax", "n0"):
        if float(getattr(student_cfg, name)) != float(
            getattr(teacher_cfg, name)
        ):
            raise ValueError(f"student/teacher mismatch in {name}")
    if str(getattr(source_args, "option", "")).lower() != "der":
        raise ValueError("the compact source must be a DER checkpoint")


def _teacher_initialize(
    model: ExactGatedCompactFeedback,
    train_queries: dict[str, torch.Tensor],
    validation_queries: dict[str, torch.Tensor],
    *,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    evaluate_every: int,
    optimizer_seed: int,
    target_mode: str,
    focus_quantile: float,
    uniform_fraction: float,
    focus_power: float,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    train_queries = _teacher_target_batch(train_queries, target_mode)
    validation_queries = _teacher_target_batch(
        validation_queries, target_mode
    )
    train_probability, train_focus = _teacher_focus_distribution(
        train_queries,
        target_mode=target_mode,
        quantile=focus_quantile,
        uniform_fraction=uniform_fraction,
        focus_power=focus_power,
    )
    validation_probability, validation_focus = (
        _teacher_focus_distribution(
            validation_queries,
            target_mode=target_mode,
            quantile=focus_quantile,
            uniform_fraction=uniform_fraction,
            focus_power=focus_power,
        )
    )
    optimizer = torch.optim.AdamW(
        model.adapter.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(optimizer_seed)
    minimum_validation = math.inf
    minimum_validation_step = 0
    final_train_mse = math.nan
    final_validation_mse = math.nan

    def evaluate(step: int) -> None:
        nonlocal minimum_validation, minimum_validation_step
        nonlocal final_train_mse, final_validation_mse
        model.eval()
        with torch.no_grad():
            train_mse = pointwise_action_mse(model, train_queries)
            validation_mse = pointwise_action_mse(
                model, validation_queries
            )
            train_focused_mse = _probability_weighted_action_mse(
                model, train_queries, train_probability
            )
            validation_focused_mse = _probability_weighted_action_mse(
                model, validation_queries, validation_probability
            )
        row = {
            "phase": "teacher_initialization",
            "step": step,
            "train_loss": float(train_mse),
            "validation_loss": float(validation_mse),
            "train_focused_loss": float(train_focused_mse),
            "validation_focused_loss": float(validation_focused_mse),
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        final_train_mse = float(train_mse)
        final_validation_mse = float(validation_mse)
        if final_validation_mse < minimum_validation:
            minimum_validation = final_validation_mse
            minimum_validation_step = step

    evaluate(0)
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        indices = _sample_focused_indices(
            train_probability,
            batch_size,
            generator,
        )
        loss = pointwise_action_mse(model, train_queries, indices)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.adapter.parameters(), max_grad_norm
        )
        optimizer.step()
        if step % evaluate_every == 0 or step == steps:
            evaluate(step)
    # This phase is initialization rather than final model selection.  Keep
    # the state produced by the prescribed teacher-fitting schedule.  The
    # only selectable checkpoint is chosen subsequently, after the teacher is
    # removed, by validation scalar PMP/DER residual.
    model.eval()
    return {
        "selected_initialization_step": steps,
        "final_train_teacher_mse": final_train_mse,
        "final_validation_teacher_mse": final_validation_mse,
        "minimum_validation_teacher_mse_diagnostic": minimum_validation,
        "minimum_validation_teacher_mse_step_diagnostic": (
            minimum_validation_step
        ),
        "used_for_final_checkpoint_selection": False,
        "capability_focused_sampling": {
            "train": train_focus,
            "validation": validation_focus,
        },
    }


def _cleanup(
    model: ExactGatedCompactFeedback,
    source: NestedFeedbackTransformer,
    protected_initial: torch.Tensor,
    train_initial: torch.Tensor,
    train_weights: torch.Tensor,
    validation_initial: torch.Tensor,
    validation_weights: torch.Tensor,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
    *,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    max_grad_norm: float,
    evaluate_every: int,
    optimizer_seed: int,
    interval_start: float,
    interval_end: float,
    w0: float,
    w1: float,
    w2: float,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Remove all teacher supervision and select only by validation PMP loss."""

    optimizer = torch.optim.AdamW(
        model.adapter.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(optimizer_seed)
    best_value = math.inf
    best_step = 0
    best_state = copy.deepcopy(model.adapter.state_dict())
    best_metrics: dict[str, float] | None = None

    def evaluate(step: int) -> None:
        nonlocal best_value, best_step, best_state, best_metrics
        model.eval()
        with torch.no_grad():
            train_pack = _scalar_pmp_pack(
                model,
                train_initial,
                train_weights,
                cfg,
                params,
                interval_start=interval_start,
                interval_end=interval_end,
                w0=w0,
                w1=w1,
                w2=w2,
            )
            validation_pack = _scalar_pmp_pack(
                model,
                validation_initial,
                validation_weights,
                cfg,
                params,
                interval_start=interval_start,
                interval_end=interval_end,
                w0=w0,
                w1=w1,
                w2=w2,
            )
            identity = continuous_policy_identity_metrics(
                source, model, protected_initial, cfg, params
            )
        _assert_exact_identity(identity)
        _assert_no_transition_queries(
            "PMP cleanup training evaluator",
            pmp_pack_gate_distribution(model, train_pack, cfg),
        )
        _assert_no_transition_queries(
            "PMP cleanup validation evaluator",
            pmp_pack_gate_distribution(model, validation_pack, cfg),
        )
        train_loss = float(train_pack["loss"])
        validation_loss = float(validation_pack["loss"])
        row = {
            "phase": "scalar_PMP_DER_cleanup",
            "step": step,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if validation_loss < best_value:
            best_value = validation_loss
            best_step = step
            best_state = copy.deepcopy(model.adapter.state_dict())
            best_metrics = _pmp_residual_metrics(
                validation_pack, scale=1.0 / cfg.alpha
            )

    evaluate(0)
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        indices = _sample_indices(
            train_initial.shape[0],
            batch_size,
            generator,
            train_initial.device,
        )
        batch_weights = train_weights.index_select(0, indices)
        # Preserve the equal-mass weighting convention after minibatching.
        batch_weights = batch_weights / batch_weights.sum()
        pack = _scalar_pmp_pack(
            model,
            train_initial.index_select(0, indices),
            batch_weights,
            cfg,
            params,
            interval_start=interval_start,
            interval_end=interval_end,
            w0=w0,
            w1=w1,
            w2=w2,
        )
        _assert_no_transition_queries(
            "PMP cleanup minibatch",
            pmp_pack_gate_distribution(model, pack, cfg),
        )
        pack["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            model.adapter.parameters(), max_grad_norm
        )
        optimizer.step()
        if step % evaluate_every == 0 or step == steps:
            evaluate(step)
    model.adapter.load_state_dict(best_state)
    model.eval()
    if best_metrics is None:
        raise RuntimeError("PMP cleanup produced no selectable checkpoint")
    return {
        "best_step": best_step,
        "best_validation_PMP_residual": best_value,
        "best_validation_physical_metrics": best_metrics,
    }


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    if args.width not in SUPPORTED_WIDTHS:
        raise ValueError(f"width must be one of {SUPPORTED_WIDTHS}")
    if args.dev_seed != DEVELOPMENT_AUDIT_SEED:
        raise ValueError(
            f"development seed must be {DEVELOPMENT_AUDIT_SEED}"
        )
    all_seeds = {
        args.teacher_train_seed,
        args.teacher_validation_seed,
        args.teacher_optimizer_seed,
        args.cleanup_train_seed,
        args.cleanup_validation_seed,
        args.cleanup_optimizer_seed,
        args.dev_seed,
    }
    if RESERVED_FINAL_BLIND_SEED in all_seeds:
        raise ValueError("the reserved final blind seed was requested")
    if len(all_seeds) != 7:
        raise ValueError("all training, validation, optimizer, and dev seeds differ")
    if args.teacher_steps < 0 or args.cleanup_steps < 0:
        raise ValueError("training step counts must be nonnegative")
    torch.set_num_threads(args.threads)
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(args.adapter_seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not args.allow_gpu:
        raise ValueError("GPU use requires --allow-gpu")
    dtype = torch.float64

    student_path = args.student_checkpoint.expanduser().resolve()
    teacher_path = args.teacher_checkpoint.expanduser().resolve()
    audit_path = args.teacher_audit.expanduser().resolve()
    positive_audit = validate_positive_teacher_audit(
        audit_path, teacher_path
    )
    source_payload = torch.load(
        student_path, map_location="cpu", weights_only=False
    )
    source, cfg, source_args = load_feedback_checkpoint(student_path)
    teacher, teacher_cfg, _, _ = load_gated_feedback_checkpoint(teacher_path)
    _compatibility_check(cfg, teacher_cfg, source_args)
    source.to(device=device, dtype=dtype).eval()
    teacher.to(device=device, dtype=dtype).eval()
    params = build_params(cfg, device, dtype)
    source.set_feature_vectors(params["r"], params["phi"])
    teacher.set_feature_vectors(params["r"], params["phi"])

    protected_initial = structured_initial_states(
        REQUIRED_PROTECTED_RADII, cfg, device, dtype
    )
    with torch.no_grad():
        protected_trace = continuous_policy_trace(
            source, protected_initial, cfg, params
        )
        protected_node_states = _node_states_from_trace(protected_trace)
    zero_tube_squared, tube_calibration = calibrated_zero_tube_squared(
        protected_node_states,
        protected_trace["states"],
        distance_scale=args.gate_distance_scale,
        safety_factor=args.gate_tube_safety_factor,
    )
    gate = ProtectedTrajectoryGate(
        protected_node_states,
        distance_scale=args.gate_distance_scale,
        zero_tube_squared=zero_tube_squared,
    ).to(device=device, dtype=dtype)
    model = ExactGatedCompactFeedback(
        source, gate, width=args.width
    ).to(device=device, dtype=dtype)
    expected_count = compact_parameter_count(
        source_feature_dimension(source), args.width
    )
    if model.adapter_parameter_count != expected_count:
        raise RuntimeError("compact adapter parameter count mismatch")
    if sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    ) != expected_count:
        raise RuntimeError("parameters outside the compact adapter are trainable")

    teacher_train_initial, _, _ = iid_and_composition_states(
        args.teacher_train_states,
        args.teacher_train_seed,
        args.radius,
        cfg,
        device,
        dtype,
    )
    teacher_validation_initial, _, _ = iid_and_composition_states(
        args.teacher_validation_states,
        args.teacher_validation_seed,
        args.radius,
        cfg,
        device,
        dtype,
    )
    cleanup_train_initial, cleanup_train_weights, _ = (
        iid_and_composition_states(
            args.cleanup_train_states,
            args.cleanup_train_seed,
            args.radius,
            cfg,
            device,
            dtype,
        )
    )
    cleanup_validation_initial, cleanup_validation_weights, _ = (
        iid_and_composition_states(
            args.cleanup_validation_states,
            args.cleanup_validation_seed,
            args.radius,
            cfg,
            device,
            dtype,
        )
    )

    teacher_train_queries = capability_query_batch(
        source, teacher, teacher_train_initial, cfg, params
    )
    teacher_validation_queries = capability_query_batch(
        source, teacher, teacher_validation_initial, cfg, params
    )
    history: list[dict[str, Any]] = []
    teacher_started = time.perf_counter()
    teacher_result = _teacher_initialize(
        model,
        teacher_train_queries,
        teacher_validation_queries,
        steps=args.teacher_steps,
        batch_size=args.teacher_batch_size,
        lr=args.teacher_lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        evaluate_every=args.teacher_evaluate_every,
        optimizer_seed=args.teacher_optimizer_seed,
        target_mode=args.teacher_target_mode,
        focus_quantile=args.teacher_focus_quantile,
        uniform_fraction=args.teacher_uniform_fraction,
        focus_power=args.teacher_focus_power,
        history=history,
    )
    teacher_seconds = time.perf_counter() - teacher_started
    identity_after_teacher = continuous_policy_identity_metrics(
        source, model, protected_initial, cfg, params
    )
    _assert_exact_identity(identity_after_teacher)

    cleanup_started = time.perf_counter()
    cleanup_result = _cleanup(
        model,
        source,
        protected_initial,
        cleanup_train_initial,
        cleanup_train_weights,
        cleanup_validation_initial,
        cleanup_validation_weights,
        cfg,
        params,
        steps=args.cleanup_steps,
        batch_size=args.cleanup_batch_size,
        lr=args.cleanup_lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        evaluate_every=args.cleanup_evaluate_every,
        optimizer_seed=args.cleanup_optimizer_seed,
        interval_start=args.interval_start,
        interval_end=args.interval_end,
        w0=args.w0,
        w1=args.w1,
        w2=args.w2,
        history=history,
    )
    cleanup_seconds = time.perf_counter() - cleanup_started
    identity = continuous_policy_identity_metrics(
        source, model, protected_initial, cfg, params
    )
    _assert_exact_identity(identity)
    _assert_no_transition_queries(
        "selected cleanup training trajectories",
        continuous_gate_distribution(
            model, cleanup_train_initial, cfg, params
        ),
    )
    _assert_no_transition_queries(
        "selected cleanup validation trajectories",
        continuous_gate_distribution(
            model, cleanup_validation_initial, cfg, params
        ),
    )
    with torch.no_grad():
        selected_train_pack = _scalar_pmp_pack(
            model,
            cleanup_train_initial,
            cleanup_train_weights,
            cfg,
            params,
            interval_start=args.interval_start,
            interval_end=args.interval_end,
            w0=args.w0,
            w1=args.w1,
            w2=args.w2,
        )
        selected_validation_pack = _scalar_pmp_pack(
            model,
            cleanup_validation_initial,
            cleanup_validation_weights,
            cfg,
            params,
            interval_start=args.interval_start,
            interval_end=args.interval_end,
            w0=args.w0,
            w1=args.w1,
            w2=args.w2,
        )
    selected_train_pmp_gate = pmp_pack_gate_distribution(
        model, selected_train_pack, cfg
    )
    selected_validation_pmp_gate = pmp_pack_gate_distribution(
        model, selected_validation_pack, cfg
    )
    _assert_no_transition_queries(
        "selected PMP training queries", selected_train_pmp_gate
    )
    _assert_no_transition_queries(
        "selected PMP validation queries", selected_validation_pmp_gate
    )

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    teacher_provenance = {
        "role": "initialization_only",
        "path": str(teacher_path),
        "sha256": sha256(teacher_path),
        "positive_audit_path": str(audit_path),
        "positive_audit_sha256": sha256(audit_path),
        "positive_audit_mean_advantage": positive_audit[
            "candidate_advantage_over_frozen_time"
        ]["mean"],
        "positive_audit_paired_lower_95": positive_audit[
            "paired_bootstrap"
        ]["candidate_advantage_over_frozen_time"]["lower_95"],
        "admission_uses_physical_objective": True,
        "used_after_initialization": False,
    }
    cleanup_provenance = {
        "loss": "scalar_PMP_DER",
        "selection": "validation_PMP_residual",
        "teacher_supervision": False,
        "physical_objective_used_in_cleanup_loss": False,
        "physical_objective_used_in_checkpoint_selection": False,
        "selected_train_PMP_gate_distribution": selected_train_pmp_gate,
        "selected_validation_PMP_gate_distribution": (
            selected_validation_pmp_gate
        ),
        "interval": [args.interval_start, args.interval_end],
        "weights": [args.w0, args.w1, args.w2],
        "train_seed": args.cleanup_train_seed,
        "validation_seed": args.cleanup_validation_seed,
        **cleanup_result,
    }
    artifact = compact_artifact_payload(
        source_payload=source_payload,
        source=source,
        model=model,
        cfg=cfg,
        protected_radii=REQUIRED_PROTECTED_RADII,
        gate_zero_tube_squared=zero_tube_squared,
        teacher_provenance=teacher_provenance,
        cleanup_provenance=cleanup_provenance,
    )
    artifact.update(
        {
            "source_checkpoint": str(student_path),
            "source_checkpoint_sha256": sha256(student_path),
            "gate_tube_calibration": tube_calibration,
            "training_history": copy.deepcopy(history),
            "teacher_initialization": teacher_result,
            "protected_identity": identity,
            "seeds": {
                "teacher_train": args.teacher_train_seed,
                "teacher_validation": args.teacher_validation_seed,
                "teacher_optimizer": args.teacher_optimizer_seed,
                "cleanup_train": args.cleanup_train_seed,
                "cleanup_validation": args.cleanup_validation_seed,
                "cleanup_optimizer": args.cleanup_optimizer_seed,
                "development_audit": args.dev_seed,
                "reserved_final_blind": RESERVED_FINAL_BLIND_SEED,
                "reserved_final_blind_used": False,
            },
        }
    )
    artifact_path = out_dir / "best_compact_exact_gated_adapter.pt"
    torch.save(artifact, artifact_path)
    (
        reloaded_payload,
        reloaded_source,
        reloaded_model,
        reloaded_cfg,
    ) = load_compact_adapter_artifact(artifact_path, device=device)
    if reloaded_cfg != cfg:
        raise RuntimeError("serialized problem configuration changed")
    reloaded_params = build_params(reloaded_cfg, device, dtype)
    reloaded_identity = continuous_policy_identity_metrics(
        reloaded_source,
        reloaded_model,
        protected_initial,
        reloaded_cfg,
        reloaded_params,
    )
    _assert_exact_identity(reloaded_identity)

    # Development audit happens strictly after residual-only selection.
    dev_initial = antithetic_initial_states(
        args.dev_count,
        args.dev_seed,
        args.radius,
        cfg,
        device,
        dtype,
        antithetic=False,
    )
    with torch.no_grad():
        dev_candidate = continuous_policy_rollout(
            reloaded_model,
            dev_initial,
            cfg,
            reloaded_params,
            state_mode="feedback",
        )
        dev_source = continuous_policy_rollout(
            reloaded_source,
            dev_initial,
            cfg,
            reloaded_params,
            state_mode="feedback",
        )
        dev_time = continuous_policy_rollout(
            reloaded_source,
            dev_initial,
            cfg,
            reloaded_params,
            state_mode="w_zero",
        )
    dev_metrics = posthoc_test_metrics(
        dev_candidate,
        dev_source,
        dev_time,
        physical_scale_factor=1.0 / cfg.alpha,
    )
    dev_time_advantage = (
        dev_time.objectives - dev_candidate.objectives
    ) / cfg.alpha
    dev_source_advantage = (
        dev_source.objectives - dev_candidate.objectives
    ) / cfg.alpha
    dev_metrics.update(
        {
            "protocol": "development_random_state_audit_v1",
            "seed": args.dev_seed,
            "count": args.dev_count,
            "radius": args.radius,
            "used_for_training": False,
            "used_for_checkpoint_selection": False,
            "used_for_hyperparameter_tuning": True,
            "final_blind": False,
            "paired_bootstrap": {
                "candidate_advantage_over_frozen_time": (
                    paired_bootstrap_mean(
                        dev_time_advantage,
                        seed=args.dev_bootstrap_seed,
                        repeats=args.dev_bootstrap_repeats,
                    )
                ),
                "candidate_advantage_over_source_feedback": (
                    paired_bootstrap_mean(
                        dev_source_advantage,
                        seed=args.dev_bootstrap_seed + 1,
                        repeats=args.dev_bootstrap_repeats,
                    )
                ),
            },
        }
    )
    (out_dir / "development_audit.json").write_text(
        json.dumps(dev_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    teacher_gate_distribution = continuous_gate_distribution(
        reloaded_model,
        teacher_train_initial,
        cfg,
        reloaded_params,
    )
    cleanup_train_gate_distribution = continuous_gate_distribution(
        reloaded_model,
        cleanup_train_initial,
        cfg,
        reloaded_params,
    )
    cleanup_validation_gate_distribution = continuous_gate_distribution(
        reloaded_model,
        cleanup_validation_initial,
        cfg,
        reloaded_params,
    )
    dev_gate_distribution = continuous_gate_distribution(
        reloaded_model,
        dev_initial,
        cfg,
        reloaded_params,
    )
    _assert_no_transition_queries(
        "cleanup training trajectories",
        cleanup_train_gate_distribution,
    )
    _assert_no_transition_queries(
        "cleanup validation trajectories",
        cleanup_validation_gate_distribution,
    )
    with (out_dir / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "artifact": str(artifact_path),
        "artifact_sha256": sha256(artifact_path),
        "width": args.width,
        "adapter_parameters": expected_count,
        "teacher_initialization": teacher_result,
        "cleanup": cleanup_result,
        "protected_identity": reloaded_identity,
        "gate_on_teacher_training_trajectories": teacher_gate_distribution,
        "gate_on_cleanup_training_trajectories": (
            cleanup_train_gate_distribution
        ),
        "gate_on_cleanup_validation_trajectories": (
            cleanup_validation_gate_distribution
        ),
        "gate_on_development_trajectories": dev_gate_distribution,
        "development_audit": dev_metrics,
        "elapsed_seconds": {
            "teacher_initialization": teacher_seconds,
            "scalar_PMP_DER_cleanup": cleanup_seconds,
        },
        "physical_objective_used_in_cleanup_loss": False,
        "physical_objective_used_in_checkpoint_selection": False,
        "physical_objective_used_for_teacher_admission": True,
        "physical_objective_used_in_development_audit": True,
        "reserved_final_blind_seed_used": False,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-audit", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, choices=SUPPORTED_WIDTHS, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--allow-gpu",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--adapter-seed", type=int, default=20261629)
    parser.add_argument("--radius", type=float, default=0.20)
    parser.add_argument("--gate-distance-scale", type=float, default=0.02)
    parser.add_argument("--gate-tube-safety-factor", type=float, default=2.0)
    parser.add_argument("--teacher-train-states", type=int, default=24)
    parser.add_argument("--teacher-validation-states", type=int, default=16)
    parser.add_argument("--teacher-train-seed", type=int, default=20261630)
    parser.add_argument("--teacher-validation-seed", type=int, default=20261631)
    parser.add_argument("--teacher-optimizer-seed", type=int, default=20261632)
    parser.add_argument("--teacher-steps", type=int, default=300)
    parser.add_argument("--teacher-batch-size", type=int, default=4096)
    parser.add_argument("--teacher-lr", type=float, default=1.0e-3)
    parser.add_argument("--teacher-evaluate-every", type=int, default=25)
    parser.add_argument(
        "--teacher-target-mode",
        choices=("absolute_teacher", "capability_increment"),
        default="absolute_teacher",
    )
    parser.add_argument("--teacher-focus-quantile", type=float, default=0.90)
    parser.add_argument("--teacher-uniform-fraction", type=float, default=0.20)
    parser.add_argument("--teacher-focus-power", type=float, default=1.0)
    parser.add_argument("--cleanup-train-states", type=int, default=16)
    parser.add_argument("--cleanup-validation-states", type=int, default=16)
    parser.add_argument("--cleanup-train-seed", type=int, default=20261640)
    parser.add_argument("--cleanup-validation-seed", type=int, default=20261641)
    parser.add_argument("--cleanup-optimizer-seed", type=int, default=20261642)
    parser.add_argument("--cleanup-steps", type=int, default=80)
    parser.add_argument("--cleanup-batch-size", type=int, default=2)
    parser.add_argument("--cleanup-lr", type=float, default=3.0e-6)
    parser.add_argument("--cleanup-evaluate-every", type=int, default=10)
    parser.add_argument("--interval-start", type=float, default=1.5)
    parser.add_argument("--interval-end", type=float, default=8.0)
    parser.add_argument("--w0", type=float, default=1.0)
    parser.add_argument("--w1", type=float, default=1.0)
    parser.add_argument("--w2", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--dev-count", type=int, default=128)
    parser.add_argument("--dev-seed", type=int, default=DEVELOPMENT_AUDIT_SEED)
    parser.add_argument("--dev-bootstrap-repeats", type=int, default=20000)
    parser.add_argument("--dev-bootstrap-seed", type=int, default=641903203)
    return parser


if __name__ == "__main__":
    print(
        json.dumps(
            run_training(build_parser().parse_args()),
            indent=2,
            sort_keys=True,
        )
    )
