#!/usr/bin/env python3
"""Full-state-branch gated fallback with teacher-free PMP/KKT cleanup.

The policy shares the locked checkpoint's time branch and combines centered
state logits as

    h = h_locked + g(t, N) * (h_probe - h_locked).

The fixed gate is exactly zero in flat tubes around locked protected
trajectories.  Only the probe state branch is trainable.  Direct controls are
not loaded by this script: the capability-probe checkpoint is used solely to
initialize that branch, after which checkpoint selection uses validation
PMP/KKT residuals and never the physical objective.
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

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for search_path in (ROOT, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from evaluate_feedback_section5 import (  # noqa: E402
    load_feedback_checkpoint as load_standard_feedback_checkpoint,
)
from feedback_continuous_policy_rk4 import (  # noqa: E402
    continuous_feedback_pmp_pack,
    continuous_feedback_state_rk4,
    pchip_midpoint_logits,
)
from feedback_section5_rk4_reference import dynamics  # noqa: E402
from natural_cubic_anchor import (  # noqa: E402
    natural_cubic_second_derivatives,
    natural_cubic_uniform_value,
)
from refine_feedback_offgrid_scalar import (  # noqa: E402
    fine_problem,
    fixed_support_dense_logits,
)
from train_feedback_section5 import (  # noqa: E402
    NestedFeedbackTransformer,
    make_fixed_directions,
    rk4_objective_per_sample,
    simulate_feedback_rk4_stagewise,
)
from train_paper_pmp_kkt import (  # noqa: E402
    ProblemConfig,
    build_params,
    parse_hidden,
)


CHECKPOINT_FORMAT = "flat_tube_full_state_probe_v5"
REQUIRED_PROTECTED_RADII = (0.0, 0.10, 0.20, 0.40, 0.60)
GATE_PROTOCOL = "C2_quintic_flat_tube_product_natural_cubic_anchor"
STRUCTURAL_GUARD_PROTOCOL = (
    "continuous_nodes_midpoints_stages_and_standard_nodes_stages_exact"
)
ANCHOR_FAMILY_PROTOCOL = {
    "continuous_fixed_support": {
        "count": len(REQUIRED_PROTECTED_RADII),
        "provenance": "fixed_support_PCHIP_continuous_RK4",
    },
    "standard_base_zoh_rk4_nodes_resampled": {
        "count": len(REQUIRED_PROTECTED_RADII),
        "provenance": (
            "base_grid_left_endpoint_ZOH_RK4_nodes_natural_cubic_resampled"
        ),
    },
    "total": 2 * len(REQUIRED_PROTECTED_RADII),
}
STANDARD_BASE_GRID_PROTOCOL = {
    "n": 800,
    "nodes": 801,
    "control": "left_endpoint_ZOH",
    "integrator": "RK4",
}


def standard_base_grid_protocol(n: int) -> dict[str, Any]:
    return {
        "n": int(n),
        "nodes": int(n) + 1,
        "control": "left_endpoint_ZOH",
        "integrator": "RK4",
    }
PMP_TRAIN_SEED = 20260810
PMP_VALIDATION_SEED = 20260811
OPTIMIZER_SEED = 20260812
RESERVED_BLIND_SEED = 20260901


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def tensor_group_sha256(
    values: dict[str, torch.Tensor],
    *,
    prefixes: tuple[str, ...],
) -> str:
    digest = hashlib.sha256()
    selected = [
        (key, value)
        for key, value in values.items()
        if key.startswith(prefixes)
    ]
    if not selected:
        raise ValueError(f"no tensors matched prefixes {prefixes}")
    for key, value in sorted(selected):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def parse_radii(text: str) -> tuple[float, ...]:
    radii = tuple(
        dict.fromkeys(
            float(value.strip())
            for value in text.split(",")
            if value.strip()
        )
    )
    if not radii or any(not math.isfinite(value) or value < 0.0 for value in radii):
        raise argparse.ArgumentTypeError(
            "protected radii must be finite nonnegative values"
        )
    return radii


def require_protocol_radii(radii: tuple[float, ...]) -> None:
    missing = [
        required
        for required in REQUIRED_PROTECTED_RADII
        if not any(
            math.isclose(required, value, rel_tol=0.0, abs_tol=1.0e-12)
            for value in radii
        )
    ]
    if missing or len(radii) != len(REQUIRED_PROTECTED_RADII):
        raise ValueError(
            "protected radii must be exactly the set 0,.1,.2,.4,.6; "
            f"received {radii}"
        )


def structured_initial_states(
    radii: tuple[float, ...],
    cfg: ProblemConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    direction = torch.linspace(
        -1.0, 1.0, cfg.m, device=device, dtype=dtype
    )
    radius = torch.as_tensor(radii, device=device, dtype=dtype)
    return cfg.n0 * (1.0 + radius[:, None] * direction[None, :])


def random_initial_states(
    count: int,
    seed: int,
    radius: float,
    cfg: ProblemConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if count <= 0:
        raise ValueError("random state count must be positive")
    directions = make_fixed_directions(count, cfg.m, seed, dtype).to(device)
    return cfg.n0 * (1.0 + float(radius) * directions)


class FlatTubeProbeFeedbackTransformer(NestedFeedbackTransformer):
    """One shared time branch with frozen source and trainable probe state MLPs."""

    def __init__(
        self,
        m: int,
        umax: float,
        state_scale: float,
        state_hidden: tuple[int, ...],
        d_model: int,
        heads: int,
        layers: int,
        init_u: float,
        correction_gain: float,
        state_feature_mode: str,
        center_state_correction: bool,
        action_temperature: float,
        action_scale: float,
        action_parameterization: str,
        action_offset: float,
        *,
        anchor_time: torch.Tensor,
        anchor_states: torch.Tensor,
        gate_normalization: float,
        gate_tube: float,
        gate_transition: float,
    ) -> None:
        super().__init__(
            m,
            umax,
            state_scale,
            state_hidden,
            d_model,
            heads,
            layers,
            init_u,
            correction_gain,
            state_feature_mode,
            center_state_correction,
            action_temperature,
            action_scale,
            action_parameterization,
            action_offset,
        )
        if anchor_time.ndim != 1 or anchor_time.numel() < 2:
            raise ValueError("anchor_time must contain at least two nodes")
        if anchor_states.ndim != 3:
            raise ValueError("anchor_states must have shape (protected,time,m)")
        if anchor_states.shape[1:] != (anchor_time.numel(), m):
            raise ValueError(
                "anchor states do not match the anchor time grid and phenotype count"
            )
        expected_time = torch.linspace(
            0.0,
            1.0,
            anchor_time.numel(),
            device=anchor_time.device,
            dtype=anchor_time.dtype,
        )
        if not torch.allclose(
            anchor_time, expected_time, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("anchor_time must be a uniform normalized grid")
        if gate_normalization <= 0.0:
            raise ValueError("gate normalization must be positive")
        if gate_tube < 0.0 or gate_transition <= 0.0:
            raise ValueError(
                "gate tube must be nonnegative and transition must be positive"
            )

        self.probe_state_branch = copy.deepcopy(self.state_branch)
        self.register_buffer(
            "gate_anchor_time", anchor_time.detach().clone(), persistent=True
        )
        self.register_buffer(
            "gate_anchor_states",
            anchor_states.detach().clone(),
            persistent=True,
        )
        self.register_buffer(
            "gate_anchor_second_derivatives",
            natural_cubic_second_derivatives(anchor_states),
            persistent=True,
        )
        self.gate_normalization = float(gate_normalization)
        self.gate_tube = float(gate_tube)
        self.gate_transition = float(gate_transition)
        self.freeze_locked_parameters()

    def freeze_locked_parameters(self) -> None:
        for parameter in self.time_branch.parameters():
            parameter.requires_grad_(False)
        for parameter in self.state_branch.parameters():
            parameter.requires_grad_(False)
        for parameter in self.probe_state_branch.parameters():
            parameter.requires_grad_(True)

    def adaptation_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.probe_state_branch.parameters())

    def train(self, mode: bool = True) -> "FlatTubeProbeFeedbackTransformer":
        super().train(mode)
        self.time_branch.eval()
        self.state_branch.eval()
        self.probe_state_branch.train(mode)
        return self

    def _anchor_states_at(self, normalized_time: torch.Tensor) -> torch.Tensor:
        return natural_cubic_uniform_value(
            self.gate_anchor_states,
            self.gate_anchor_second_derivatives,
            normalized_time,
        )

    def gate_values(
        self,
        normalized_time: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        if normalized_time.ndim == 0:
            normalized_time = normalized_time.expand(state.shape[0])
        anchors = self._anchor_states_at(normalized_time)
        relative = (
            state.unsqueeze(1) - anchors
        ) / self.gate_normalization
        distance_squared = relative.square().mean(dim=-1)
        inner_squared = self.gate_tube**2
        outer_squared = (self.gate_tube + self.gate_transition) ** 2
        coordinate = (
            (distance_squared - inner_squared)
            / (outer_squared - inner_squared)
        ).clamp(0.0, 1.0)
        # Quintic smootherstep is C2 with zero first/second derivatives at both
        # flat-region boundaries.  The product is zero in the union of tubes.
        per_anchor = coordinate.pow(3) * (
            10.0 - 15.0 * coordinate + 6.0 * coordinate.square()
        )
        return per_anchor.prod(dim=1)

    def _centered_branch_logits(
        self,
        branch: torch.nn.Module,
        normalized_time: torch.Tensor,
        state: torch.Tensor,
        *,
        features: torch.Tensor | None = None,
        reference_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features is None:
            features = self.state_features(normalized_time, state)
        value = branch(features).squeeze(-1)
        if self.center_state_correction:
            if reference_features is None:
                reference = self.nominal_state_at(normalized_time)
                reference_features = self.state_features(
                    normalized_time, reference
                )
            value = value - branch(reference_features).squeeze(-1)
        return value

    def locked_state_logits(
        self, normalized_time: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        return self._centered_branch_logits(
            self.state_branch, normalized_time, state
        )

    def state_logits(
        self, normalized_time: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        features = self.state_features(normalized_time, state)
        reference_features: torch.Tensor | None = None
        if self.center_state_correction:
            reference = self.nominal_state_at(normalized_time)
            reference_features = self.state_features(
                normalized_time, reference
            )
        locked = self._centered_branch_logits(
            self.state_branch,
            normalized_time,
            state,
            features=features,
            reference_features=reference_features,
        )
        probe = self._centered_branch_logits(
            self.probe_state_branch,
            normalized_time,
            state,
            features=features,
            reference_features=reference_features,
        )
        gate = self.gate_values(normalized_time, state)
        return locked + gate * (probe - locked)

    def _action_from_correction(
        self, base_logit: torch.Tensor, correction: torch.Tensor
    ) -> torch.Tensor:
        combined_logit = base_logit + self.correction_gain * correction
        if self.action_parameterization == "linear-raw-box":
            return torch.clamp(combined_logit, 0.0, self.umax)
        return torch.clamp(
            self.action_scale
            * self.umax
            * torch.sigmoid(combined_logit / self.action_temperature)
            - self.action_offset,
            0.0,
            self.umax,
        )

    def interval_action(
        self,
        base_logit: torch.Tensor,
        normalized_time: torch.Tensor,
        state: torch.Tensor,
        *,
        state_blind: bool = False,
        state_mode: str | None = None,
    ) -> torch.Tensor:
        if state_mode != "locked_feedback":
            return super().interval_action(
                base_logit,
                normalized_time,
                state,
                state_blind=state_blind,
                state_mode=state_mode,
            )
        correction = self.locked_state_logits(normalized_time, state)
        return self._action_from_correction(base_logit, correction)


def _model_constructor_values(
    source_args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "state_scale": float(source_args.state_scale),
        "state_hidden": parse_hidden(source_args.state_hidden),
        "d_model": int(source_args.d_model),
        "heads": int(source_args.heads),
        "layers": int(source_args.layers),
        "init_u": float(source_args.init_u),
        "correction_gain": float(source_args.correction_gain),
        "state_feature_mode": str(
            getattr(source_args, "state_feature_mode", "log_absolute")
        ),
        "center_state_correction": bool(
            getattr(source_args, "center_state_correction", False)
        ),
        "action_temperature": float(
            getattr(source_args, "action_temperature", 1.0)
        ),
        "action_scale": float(getattr(source_args, "action_scale", 1.0)),
        "action_parameterization": str(
            getattr(
                source_args,
                "action_parameterization",
                "logit-temperature",
            )
        ),
        "action_offset": float(getattr(source_args, "action_offset", 0.0)),
    }


def _assert_equal_tensor_dicts(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    *,
    label: str,
) -> None:
    if set(left) != set(right):
        raise ValueError(f"{label} tensor keys differ")
    unequal = [
        key
        for key in sorted(left)
        if (
            left[key].dtype != right[key].dtype
            or left[key].device != right[key].device
            or not torch.equal(left[key], right[key])
        )
    ]
    if unequal:
        raise ValueError(f"{label} tensors differ: {unequal[:5]}")


def validate_checkpoint_pair(
    locked: NestedFeedbackTransformer,
    probe: NestedFeedbackTransformer,
    locked_cfg: ProblemConfig,
    probe_cfg: ProblemConfig,
    locked_args: argparse.Namespace,
    probe_args: argparse.Namespace,
) -> None:
    if locked_cfg != probe_cfg:
        raise ValueError("locked and capability-probe problems differ")
    if _model_constructor_values(locked_args) != _model_constructor_values(
        probe_args
    ):
        raise ValueError(
            "locked and capability-probe architecture/action arguments differ"
        )
    if not bool(getattr(locked_args, "center_state_correction", False)):
        raise ValueError(
            "full-state fallback requires centered state-branch logits"
        )
    if str(getattr(locked_args, "state_feature_mode", "")) not in {
        "log_absolute",
        "relative_nominal",
        "burden_composition",
    }:
        raise ValueError(
            "full-state fallback requires phenotype-resolved state features"
        )
    if str(getattr(probe_args, "initialization", "")) != (
        "direct-trajectory supervised state-branch fit"
    ):
        raise ValueError(
            "capability probe must declare direct-trajectory state-branch "
            "initialization"
        )
    _assert_equal_tensor_dicts(
        locked.time_branch.state_dict(),
        probe.time_branch.state_dict(),
        label="time branch",
    )
    locked_shapes = {
        key: tuple(value.shape)
        for key, value in locked.state_branch.state_dict().items()
    }
    probe_shapes = {
        key: tuple(value.shape)
        for key, value in probe.state_branch.state_dict().items()
    }
    if locked_shapes != probe_shapes:
        raise ValueError("locked and capability-probe state branches differ")
    if all(
        torch.equal(
            locked.state_branch.state_dict()[key],
            probe.state_branch.state_dict()[key],
        )
        for key in locked.state_branch.state_dict()
    ):
        raise ValueError(
            "capability-probe state branch is identical to the locked branch"
        )
    if not torch.equal(locked.nominal_reference, probe.nominal_reference):
        raise ValueError(
            "locked and capability-probe nominal references differ"
        )


def construct_gated_model(
    locked: NestedFeedbackTransformer,
    probe: NestedFeedbackTransformer,
    cfg: ProblemConfig,
    source_args: argparse.Namespace,
    *,
    anchor_time: torch.Tensor,
    anchor_states: torch.Tensor,
    gate_tube: float,
    gate_transition: float,
) -> FlatTubeProbeFeedbackTransformer:
    values = _model_constructor_values(source_args)
    locked_parameter = next(locked.parameters())
    probe_parameter = next(probe.parameters())
    if (
        probe_parameter.dtype != locked_parameter.dtype
        or probe_parameter.device != locked_parameter.device
    ):
        raise ValueError(
            "locked and capability-probe dtype/device must match"
        )
    model = FlatTubeProbeFeedbackTransformer(
        cfg.m,
        cfg.umax,
        **values,
        anchor_time=anchor_time,
        anchor_states=anchor_states,
        gate_normalization=cfg.n0,
        gate_tube=gate_tube,
        gate_transition=gate_transition,
    ).to(
        device=locked_parameter.device,
        dtype=locked_parameter.dtype,
    )
    model.time_branch.load_state_dict(locked.time_branch.state_dict())
    model.state_branch.load_state_dict(locked.state_branch.state_dict())
    model.probe_state_branch.load_state_dict(probe.state_branch.state_dict())
    _assert_equal_tensor_dicts(
        model.time_branch.state_dict(),
        locked.time_branch.state_dict(),
        label="constructed locked time branch",
    )
    _assert_equal_tensor_dicts(
        model.state_branch.state_dict(),
        locked.state_branch.state_dict(),
        label="constructed locked state branch",
    )
    _assert_equal_tensor_dicts(
        model.probe_state_branch.state_dict(),
        probe.state_branch.state_dict(),
        label="constructed probe state branch",
    )
    model.set_nominal_reference(locked.nominal_reference)
    model.freeze_locked_parameters()
    return model


def load_gated_feedback_checkpoint(
    path: Path,
    *,
    _allow_incomplete_posthoc: bool = False,
) -> tuple[
    FlatTubeProbeFeedbackTransformer,
    ProblemConfig,
    argparse.Namespace,
    dict[str, Any],
]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{path} is not a {CHECKPOINT_FORMAT} checkpoint"
        )
    args = argparse.Namespace(**payload["args"])
    cfg = ProblemConfig(**payload["problem"])
    if not bool(getattr(args, "center_state_correction", False)):
        raise ValueError("gated checkpoint does not use centered state logits")
    if str(getattr(args, "state_feature_mode", "")) not in {
        "log_absolute",
        "relative_nominal",
        "burden_composition",
    }:
        raise ValueError(
            "gated checkpoint does not use phenotype-resolved state features"
        )
    state = payload["model_state"]
    required_buffers = {
        "gate_anchor_time",
        "gate_anchor_states",
        "gate_anchor_second_derivatives",
    }
    if not required_buffers.issubset(state):
        raise ValueError("gated checkpoint is missing anchor buffers")
    if state["gate_anchor_states"].shape[0] != ANCHOR_FAMILY_PROTOCOL["total"]:
        raise ValueError("gated checkpoint does not contain both anchor families")
    gate = payload["gated_fallback"]
    required_protocol = {
        "centered_state_logits": True,
        "shared_time_branch": True,
        "gate": GATE_PROTOCOL,
        "structural_guard": STRUCTURAL_GUARD_PROTOCOL,
        "anchor_families": ANCHOR_FAMILY_PROTOCOL,
        "protected_tolerance": 0.0,
        "source_weights_loaded_without_dtype_roundtrip": True,
        "train_scope": "probe_state_branch_only",
        "direct_supervision_used_in_cleanup": False,
        "physical_objective_used_in_loss": False,
        "physical_objective_used_in_selection": False,
        "posthoc_identity_audit": True,
        "train_seed": PMP_TRAIN_SEED,
        "validation_seed": PMP_VALIDATION_SEED,
        "optimizer_seed": OPTIMIZER_SEED,
        "reserved_blind_seed": RESERVED_BLIND_SEED,
        "reserved_blind_used_for_training_or_selection": False,
    }
    wrong_protocol = {
        key: (gate.get(key), expected)
        for key, expected in required_protocol.items()
        if gate.get(key) != expected
    }
    if wrong_protocol:
        raise ValueError(
            f"gated checkpoint protocol metadata mismatch: {wrong_protocol}"
        )
    if type(gate.get("formal_protocol")) is not bool:
        raise ValueError("gated checkpoint formal_protocol flag is missing")
    if int(gate.get("base_n", -1)) != cfg.n:
        raise ValueError("gated checkpoint base_n differs from the problem")
    if gate.get("standard_base_grid") != standard_base_grid_protocol(cfg.n):
        raise ValueError("gated checkpoint standard base-grid metadata differs")
    if gate["formal_protocol"] and cfg.n != 800:
        raise ValueError("formal gated checkpoint must use base n=800")
    if not _allow_incomplete_posthoc:
        if gate.get("posthoc_identity_audit_completed") is not True:
            raise ValueError(
                "gated checkpoint has not completed its posthoc identity audit"
            )
        posthoc = gate.get("posthoc_protected_identity", {})
        if not posthoc or any(float(value) != 0.0 for value in posthoc.values()):
            raise ValueError(
                "gated checkpoint posthoc protected identity is not exact"
            )
        posthoc_reference = gate.get("posthoc_identity_reference", {})
        if (
            posthoc_reference.get("sha256")
            != gate.get("locked_checkpoint", {}).get("sha256")
            or posthoc_reference.get("evaluator")
            != standard_base_grid_protocol(cfg.n)
        ):
            raise ValueError(
                "gated checkpoint posthoc identity reference is inconsistent"
            )
    require_protocol_radii(
        tuple(float(value) for value in gate["protected_radii"])
    )
    if not math.isclose(
        float(gate["gate_normalization"]),
        cfg.n0,
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("gated checkpoint normalization differs from n0")
    expected_second = natural_cubic_second_derivatives(
        state["gate_anchor_states"]
    )
    if not torch.allclose(
        state["gate_anchor_second_derivatives"],
        expected_second,
        rtol=0.0,
        atol=64.0 * torch.finfo(expected_second.dtype).eps,
    ):
        raise ValueError(
            "gated checkpoint spline derivatives are inconsistent with anchors"
        )
    floating_state_tensors = [
        value
        for value in state.values()
        if value.is_floating_point()
    ]
    state_dtypes = {value.dtype for value in floating_state_tensors}
    if state_dtypes != {torch.float64}:
        raise ValueError(
            "gated checkpoint tensors must be uniformly float64"
        )
    checkpoint_dtype = next(iter(state_dtypes))
    if payload["nominal_reference"].dtype != checkpoint_dtype:
        raise ValueError(
            "gated checkpoint nominal reference dtype is inconsistent"
        )
    model = FlatTubeProbeFeedbackTransformer(
        cfg.m,
        cfg.umax,
        **_model_constructor_values(args),
        anchor_time=state["gate_anchor_time"],
        anchor_states=state["gate_anchor_states"],
        gate_normalization=float(gate["gate_normalization"]),
        gate_tube=float(gate["gate_tube"]),
        gate_transition=float(gate["gate_transition"]),
    ).to(device=torch.device("cpu"), dtype=checkpoint_dtype)
    model.load_state_dict(state, strict=True)
    _assert_equal_tensor_dicts(
        model.state_dict(), state, label="native gated checkpoint reload"
    )
    model.set_nominal_reference(
        payload["nominal_reference"].to(dtype=checkpoint_dtype)
    )
    params = build_params(cfg, torch.device("cpu"), checkpoint_dtype)
    model.set_feature_vectors(params["r"], params["phi"])
    model.freeze_locked_parameters()
    model.eval()
    return model, cfg, args, payload


def build_locked_anchors(
    locked: NestedFeedbackTransformer,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
    protected_initial: torch.Tensor,
    *,
    multiplier: int,
    query_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build C2 tubes around both evaluators' exact locked trajectories.

    The first family follows the continuous fixed-support/PCHIP evaluator used
    by the PMP cleanup.  The second family starts from the *base* ``n=800``
    left-endpoint/ZOH RK4 nodes used by the task evaluator.  Those base nodes
    are natural-cubic resampled onto the same dense uniform grid; the original
    base knots are then copied back exactly.  This is intentionally different
    from re-integrating the standard policy on the fine grid, which would
    define a different discrete trajectory.
    """

    anchor_cfg = fine_problem(cfg, multiplier)
    anchor_time, anchor_raw = fixed_support_dense_logits(
        locked,
        cfg,
        multiplier,
        query_batch_size=query_batch_size,
    )
    anchor_midpoint_raw = pchip_midpoint_logits(anchor_time, anchor_raw)
    with torch.no_grad():
        continuous_anchor_states, _ = continuous_feedback_state_rk4(
            locked,
            protected_initial,
            anchor_cfg,
            anchor_time,
            anchor_raw,
            anchor_midpoint_raw,
            params,
            state_mode="feedback",
        )
        standard_base_states, _, _, _ = simulate_feedback_rk4_stagewise(
            locked,
            protected_initial,
            cfg,
            params,
            state_mode="feedback",
        )
        standard_base_second = natural_cubic_second_derivatives(
            standard_base_states
        )
        standard_anchor_states = natural_cubic_uniform_value(
            standard_base_states,
            standard_base_second,
            anchor_time,
        ).permute(1, 0, 2).contiguous()

        expected_dense_nodes = cfg.n * multiplier + 1
        if anchor_time.numel() != expected_dense_nodes:
            raise RuntimeError("dense anchor grid is inconsistent with multiplier")
        base_indices = torch.arange(
            0,
            expected_dense_nodes,
            multiplier,
            device=anchor_time.device,
        )
        # Preserve the task evaluator's base knots bit for bit.  The final
        # natural spline is fitted to this dense array and remains C2.
        standard_anchor_states[:, base_indices, :] = standard_base_states
        if not torch.equal(
            standard_anchor_states[:, base_indices, :],
            standard_base_states,
        ):
            raise RuntimeError("standard task anchor lost exact base nodes")

    return anchor_time, torch.cat(
        (continuous_anchor_states, standard_anchor_states), dim=0
    )


def scalar_selection_value(
    pack: dict[str, torch.Tensor],
    *,
    w0: float,
    w1: float,
    w2: float,
) -> torch.Tensor:
    components = pack["component_losses"]
    return (
        float(w0) * components["H_u"]
        + float(w1) * components["dH_u_dt"]
        + float(w2) * components["d2H_u_dt2"]
    )


def pmp_residual_pack(
    model: FlatTubeProbeFeedbackTransformer,
    initial_state: torch.Tensor,
    cfg: ProblemConfig,
    normalized_time: torch.Tensor,
    raw_logits: torch.Tensor,
    midpoint_raw_logits: torch.Tensor,
    params: dict[str, torch.Tensor],
    *,
    continuous_collocation: str,
    option: str,
    interval_start: float,
    interval_end: float,
    w0: float,
    w1: float,
    w2: float,
    cf_weight: float,
    cf_scalar_weight: float,
) -> dict[str, Any]:
    """Build only PMP/KKT residuals; deliberately do not compute physical J."""

    if continuous_collocation not in {"nodes", "nodes-midpoints"}:
        raise ValueError(
            "continuous_collocation must be 'nodes' or 'nodes-midpoints'"
        )
    continuous = continuous_feedback_pmp_pack(
        model,
        initial_state,
        cfg,
        normalized_time,
        raw_logits,
        midpoint_raw_logits,
        params,
        state_mode="feedback",
    )
    controls = continuous.node_controls[:, :-1]
    if continuous_collocation == "nodes":
        sample_controls = controls
        quantities = continuous.quantities
        sample_time = normalized_time[:-1] * cfg.T
        collocation_weights = None
    else:
        sample_controls = torch.stack(
            [controls, continuous.midpoint_controls], dim=-1
        )
        quantities = {
            key: torch.stack(
                [
                    continuous.quantities[key],
                    continuous.midpoint_quantities[key],
                ],
                dim=-1,
            )
            for key in continuous.quantities
        }
        midpoint_time = 0.5 * (
            normalized_time[:-1] + normalized_time[1:]
        )
        sample_time = torch.stack(
            [normalized_time[:-1], midpoint_time], dim=-1
        ) * cfg.T
        collocation_weights = torch.full(
            (1, 1, 2),
            0.5,
            device=controls.device,
            dtype=controls.dtype,
        )
    mask = (
        (sample_time >= interval_start)
        & (sample_time < interval_end)
    ).to(controls.dtype)
    weighted_mask = mask.unsqueeze(0)
    if collocation_weights is not None:
        weighted_mask = weighted_mask * collocation_weights
    mask_mass = weighted_mask.sum()
    if not bool(mask_mass > 0.0):
        raise ValueError(
            "PMP interval contains no collocation samples on this grid"
        )
    denominator = mask_mass * controls.shape[0]

    def mean_square(value: torch.Tensor) -> torch.Tensor:
        return (weighted_mask * value.square()).sum() / denominator

    pmp_components = {
        "H_u": mean_square(quantities["psi"]),
        "dH_u_dt": mean_square(quantities["dot_psi"]),
        "d2H_u_dt2": mean_square(quantities["ddot_psi"]),
    }
    pmp_loss = (
        float(w0) * pmp_components["H_u"]
        + float(w1) * pmp_components["dH_u_dt"]
        + float(w2) * pmp_components["d2H_u_dt2"]
    )
    if option == "der":
        component_losses = pmp_components
        loss = pmp_loss
    elif option == "cf":
        component_losses = {
            "closed_form_control": mean_square(
                sample_controls - quantities["u_state"]
            ),
            **pmp_components,
        }
        loss = (
            float(cf_weight) * component_losses["closed_form_control"]
            + float(cf_scalar_weight) * pmp_loss
        )
    else:
        raise ValueError("option must be 'cf' or 'der'")
    return {
        "loss": loss,
        "component_losses": component_losses,
        "pmp_selection": pmp_loss,
    }


def _flat_policy_queries(
    model: FlatTubeProbeFeedbackTransformer,
    states: torch.Tensor,
    normalized_time: torch.Tensor,
) -> torch.Tensor:
    batch, points, m = states.shape
    flat_states = states.reshape(batch * points, m)
    flat_time = (
        normalized_time.unsqueeze(0)
        .expand(batch, -1)
        .reshape(batch * points)
    )
    return model.gate_values(flat_time, flat_states).reshape(batch, points)


def continuous_policy_stage_trace(
    model: FlatTubeProbeFeedbackTransformer,
    initial: torch.Tensor,
    cfg: ProblemConfig,
    normalized_time: torch.Tensor,
    raw_logits: torch.Tensor,
    midpoint_raw_logits: torch.Tensor,
    params: dict[str, torch.Tensor],
    *,
    state_mode: str,
) -> dict[str, torch.Tensor]:
    """Return every actual RK4 policy-query state/control/time."""

    state = initial
    step = cfg.T / cfg.n
    batch = initial.shape[0]
    stage_states: list[torch.Tensor] = []
    stage_controls: list[torch.Tensor] = []
    stage_times: list[torch.Tensor] = []

    def action(
        raw: torch.Tensor,
        time: torch.Tensor,
        query_state: torch.Tensor,
    ) -> torch.Tensor:
        return model.interval_action(
            raw,
            time.expand(batch),
            query_state,
            state_mode=state_mode,
        )

    for index in range(cfg.n):
        left_time = normalized_time[index]
        middle_time = 0.5 * (
            normalized_time[index] + normalized_time[index + 1]
        )
        right_time = normalized_time[index + 1]

        control1 = action(raw_logits[index], left_time, state)
        slope1 = dynamics(state, control1, params)
        state2 = state + 0.5 * step * slope1

        control2 = action(
            midpoint_raw_logits[index], middle_time, state2
        )
        slope2 = dynamics(state2, control2, params)
        state3 = state + 0.5 * step * slope2

        control3 = action(
            midpoint_raw_logits[index], middle_time, state3
        )
        slope3 = dynamics(state3, control3, params)
        state4 = state + step * slope3

        control4 = action(
            raw_logits[index + 1], right_time, state4
        )
        slope4 = dynamics(state4, control4, params)
        stage_states.append(
            torch.stack((state, state2, state3, state4), dim=1)
        )
        stage_controls.append(
            torch.stack(
                (control1, control2, control3, control4), dim=1
            )
        )
        stage_times.append(
            torch.stack(
                (left_time, middle_time, middle_time, right_time)
            )
        )
        state = state + (step / 6.0) * (
            slope1 + 2.0 * slope2 + 2.0 * slope3 + slope4
        )

    return {
        "states": torch.stack(stage_states, dim=1),
        "controls": torch.stack(stage_controls, dim=1),
        "normalized_time": torch.stack(stage_times),
        "terminal_state": state,
    }


def structural_guard_metrics(
    model: FlatTubeProbeFeedbackTransformer,
    initial: torch.Tensor,
    cfg: ProblemConfig,
    normalized_time: torch.Tensor,
    raw_logits: torch.Tensor,
    midpoint_raw_logits: torch.Tensor,
    params: dict[str, torch.Tensor],
) -> dict[str, float]:
    with torch.no_grad():
        locked = continuous_feedback_pmp_pack(
            model,
            initial,
            cfg,
            normalized_time,
            raw_logits,
            midpoint_raw_logits,
            params,
            state_mode="locked_feedback",
        )
        candidate = continuous_feedback_pmp_pack(
            model,
            initial,
            cfg,
            normalized_time,
            raw_logits,
            midpoint_raw_logits,
            params,
            state_mode="feedback",
        )
        locked_stages = continuous_policy_stage_trace(
            model,
            initial,
            cfg,
            normalized_time,
            raw_logits,
            midpoint_raw_logits,
            params,
            state_mode="locked_feedback",
        )
        candidate_stages = continuous_policy_stage_trace(
            model,
            initial,
            cfg,
            normalized_time,
            raw_logits,
            midpoint_raw_logits,
            params,
            state_mode="feedback",
        )
        midpoint_time = 0.5 * (
            normalized_time[:-1] + normalized_time[1:]
        )
        node_gate = _flat_policy_queries(
            model, candidate.states, normalized_time
        )
        midpoint_gate = _flat_policy_queries(
            model, candidate.midpoint_states, midpoint_time
        )
        batch, intervals, stages, m = candidate_stages["states"].shape
        stage_gate = model.gate_values(
            candidate_stages["normalized_time"]
            .unsqueeze(0)
            .expand(batch, -1, -1)
            .reshape(batch * intervals * stages),
            candidate_stages["states"].reshape(
                batch * intervals * stages, m
            ),
        )
    return {
        "protected_state_max_abs": float(
            (candidate.states - locked.states).abs().max().cpu()
        ),
        "protected_node_control_max_abs": float(
            (
                candidate.node_controls - locked.node_controls
            ).abs().max().cpu()
        ),
        "protected_midpoint_control_max_abs": float(
            (
                candidate.midpoint_controls - locked.midpoint_controls
            ).abs().max().cpu()
        ),
        "protected_stage_state_max_abs": float(
            (
                candidate_stages["states"] - locked_stages["states"]
            ).abs().max().cpu()
        ),
        "protected_stage_control_max_abs": float(
            (
                candidate_stages["controls"]
                - locked_stages["controls"]
            ).abs().max().cpu()
        ),
        "protected_stage_gate_max": float(stage_gate.max().cpu()),
        "protected_gate_max": float(
            torch.maximum(node_gate.max(), midpoint_gate.max()).cpu()
        ),
    }


def standard_task_guard_metrics(
    model: FlatTubeProbeFeedbackTransformer,
    initial: torch.Tensor,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
    *,
    include_objective: bool = False,
) -> dict[str, float]:
    """Check exact identity under the base left-endpoint/ZOH RK4 evaluator.

    Objective values are deliberately optional.  Training/validation
    checkpoint ranking calls this with ``include_objective=False``; the
    objective identity is evaluated only after checkpoint selection.
    """

    with torch.no_grad():
        locked_states, locked_controls, locked_stages, _ = (
            simulate_feedback_rk4_stagewise(
                model,
                initial,
                cfg,
                params,
                state_mode="locked_feedback",
            )
        )
        candidate_states, candidate_controls, candidate_stages, _ = (
            simulate_feedback_rk4_stagewise(
                model,
                initial,
                cfg,
                params,
                state_mode="feedback",
            )
        )
        node_time = torch.linspace(
            0.0,
            1.0,
            cfg.n + 1,
            device=initial.device,
            dtype=initial.dtype,
        )[:-1]
        batch = initial.shape[0]
        node_gate = model.gate_values(
            node_time.unsqueeze(0).expand(batch, -1).reshape(-1),
            candidate_states[:, :-1].reshape(-1, cfg.m),
        )

        metrics = {
            "protected_standard_gate_max": float(node_gate.max().cpu()),
            "protected_standard_control_max_abs": float(
                (candidate_controls - locked_controls).abs().max().cpu()
            ),
            "protected_standard_stage_control_max_abs": float(
                (
                    candidate_controls.unsqueeze(-1).expand(-1, -1, 4)
                    - locked_controls.unsqueeze(-1).expand(-1, -1, 4)
                ).abs().max().cpu()
            ),
            "protected_standard_node_state_max_abs": float(
                (candidate_states - locked_states).abs().max().cpu()
            ),
            "protected_standard_stage_state_max_abs": float(
                (candidate_stages - locked_stages).abs().max().cpu()
            ),
        }
        if include_objective:
            locked_objective = rk4_objective_per_sample(
                locked_states,
                locked_controls,
                locked_stages,
                cfg,
                params,
            )
            candidate_objective = rk4_objective_per_sample(
                candidate_states,
                candidate_controls,
                candidate_stages,
                cfg,
                params,
            )
            metrics["protected_standard_objective_max_abs"] = float(
                (candidate_objective - locked_objective).abs().max().cpu()
            )
    return metrics


def external_standard_task_identity_metrics(
    model: FlatTubeProbeFeedbackTransformer,
    source: NestedFeedbackTransformer,
    initial: torch.Tensor,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
    *,
    include_objective: bool,
) -> dict[str, float]:
    """Compare the candidate with an independently reloaded locked source."""

    with torch.no_grad():
        source_states, source_controls, source_stages, _ = (
            simulate_feedback_rk4_stagewise(
                source,
                initial,
                cfg,
                params,
                state_mode="feedback",
            )
        )
        candidate_states, candidate_controls, candidate_stages, _ = (
            simulate_feedback_rk4_stagewise(
                model,
                initial,
                cfg,
                params,
                state_mode="feedback",
            )
        )
        metrics = {
            "protected_external_standard_control_max_abs": float(
                (candidate_controls - source_controls).abs().max().cpu()
            ),
            "protected_external_standard_stage_control_max_abs": float(
                (
                    candidate_controls.unsqueeze(-1).expand(-1, -1, 4)
                    - source_controls.unsqueeze(-1).expand(-1, -1, 4)
                ).abs().max().cpu()
            ),
            "protected_external_standard_node_state_max_abs": float(
                (candidate_states - source_states).abs().max().cpu()
            ),
            "protected_external_standard_stage_state_max_abs": float(
                (candidate_stages - source_stages).abs().max().cpu()
            ),
        }
        if include_objective:
            source_objective = rk4_objective_per_sample(
                source_states,
                source_controls,
                source_stages,
                cfg,
                params,
            )
            candidate_objective = rk4_objective_per_sample(
                candidate_states,
                candidate_controls,
                candidate_stages,
                cfg,
                params,
            )
            metrics["protected_external_standard_objective_max_abs"] = float(
                (candidate_objective - source_objective).abs().max().cpu()
            )
    return metrics


def guard_passed(metrics: dict[str, float], tolerance: float) -> bool:
    return all(value <= tolerance for value in metrics.values())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def parameter_inventory(
    model: FlatTubeProbeFeedbackTransformer,
) -> dict[str, Any]:
    trainable_names = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    return {
        "total_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "time_branch_parameters": sum(
            parameter.numel() for parameter in model.time_branch.parameters()
        ),
        "locked_state_branch_parameters": sum(
            parameter.numel() for parameter in model.state_branch.parameters()
        ),
        "probe_state_branch_parameters": sum(
            parameter.numel()
            for parameter in model.probe_state_branch.parameters()
        ),
        "trainable_names": trainable_names,
    }


def assert_only_probe_trainable(
    model: FlatTubeProbeFeedbackTransformer,
) -> None:
    bad = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and not name.startswith("probe_state_branch.")
    ]
    frozen_probe = [
        name
        for name, parameter in model.probe_state_branch.named_parameters()
        if not parameter.requires_grad
    ]
    if bad or frozen_probe:
        raise RuntimeError(
            f"invalid trainable parameter set: bad={bad}, "
            f"frozen_probe={frozen_probe}"
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting this only before parallel work begins.
        pass
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    if args.smoke and args.device != "cpu":
        raise ValueError("smoke mode is CPU-only")
    if (
        not math.isfinite(args.protected_tolerance)
        or args.protected_tolerance != 0.0
    ):
        raise ValueError("protected tolerance is hard-locked to exact zero")
    expected_seeds = {
        "train_seed": PMP_TRAIN_SEED,
        "validation_seed": PMP_VALIDATION_SEED,
        "optimizer_seed": OPTIMIZER_SEED,
        "blind_seed": RESERVED_BLIND_SEED,
    }
    wrong_seeds = {
        name: (getattr(args, name), expected)
        for name, expected in expected_seeds.items()
        if getattr(args, name) != expected
    }
    if wrong_seeds:
        raise ValueError(
            f"seed protocol mismatch: {wrong_seeds}"
        )
    if len(set(expected_seeds.values())) != len(expected_seeds):
        raise RuntimeError("protocol seeds must be pairwise distinct")
    torch.manual_seed(args.optimizer_seed)
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    dtype = torch.float64

    protected_radii = parse_radii(args.protected_radii)
    require_protocol_radii(protected_radii)
    if args.epochs < 0 or args.eval_every <= 0:
        raise ValueError("epochs must be nonnegative and eval-every positive")
    if args.smoke_epochs < 0:
        raise ValueError("smoke epochs must be nonnegative")
    if (
        args.anchor_multiplier <= 0
        or args.smoke_anchor_multiplier <= 0
        or args.pmp_multiplier <= 0
        or args.smoke_pmp_multiplier <= 0
    ):
        raise ValueError("anchor and PMP multipliers must be positive")
    if (
        args.train_random_states <= 0
        or args.validation_random_states <= 0
        or args.smoke_random_states <= 0
    ):
        raise ValueError("PMP state counts must be positive")
    if not (
        math.isfinite(args.interval_start)
        and math.isfinite(args.interval_end)
        and args.interval_start < args.interval_end
    ):
        raise ValueError("PMP interval must be finite and nonempty")
    pmp_weights = (args.w0, args.w1, args.w2)
    if (
        any(not math.isfinite(weight) or weight < 0.0 for weight in pmp_weights)
        or not any(weight > 0.0 for weight in pmp_weights)
    ):
        raise ValueError(
            "PMP weights must be finite, nonnegative, and not all zero"
        )
    if any(
        not math.isfinite(weight) or weight < 0.0
        for weight in (args.cf_weight, args.cf_scalar_weight)
    ):
        raise ValueError("closed-form loss weights must be finite and nonnegative")
    if (
        not math.isfinite(args.lr)
        or args.lr <= 0.0
        or not math.isfinite(args.weight_decay)
        or args.weight_decay < 0.0
    ):
        raise ValueError("learning rate/weight decay are invalid")

    out_dir = args.out_dir.expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to use nonempty {out_dir}")

    locked_path = args.locked_checkpoint.expanduser().resolve()
    probe_path = args.capability_probe_checkpoint.expanduser().resolve()
    locked_hash = sha256(locked_path)
    probe_hash = sha256(probe_path)
    if args.expected_locked_sha256 and (
        args.expected_locked_sha256 != locked_hash
    ):
        raise RuntimeError("locked checkpoint SHA256 mismatch")
    if args.expected_probe_sha256 and (
        args.expected_probe_sha256 != probe_hash
    ):
        raise RuntimeError("capability-probe checkpoint SHA256 mismatch")

    locked, cfg, locked_args = load_standard_feedback_checkpoint(locked_path)
    probe, probe_cfg, probe_args = load_standard_feedback_checkpoint(probe_path)
    validate_checkpoint_pair(
        locked, probe, cfg, probe_cfg, locked_args, probe_args
    )
    if not args.smoke and cfg.n != 800:
        raise ValueError("the formal protocol requires base cfg.n == 800")
    if not (
        0.0 <= args.interval_start
        < args.interval_end
        <= cfg.T
    ):
        raise ValueError(
            f"PMP interval must lie within [0, {cfg.T}]"
        )
    locked.to(device=device, dtype=dtype).eval()
    probe.to(device=device, dtype=dtype).eval()
    params = build_params(cfg, device, dtype)
    locked.set_feature_vectors(params["r"], params["phi"])
    probe.set_feature_vectors(params["r"], params["phi"])

    protected_initial = structured_initial_states(
        protected_radii, cfg, device, dtype
    )
    anchor_multiplier = (
        args.smoke_anchor_multiplier if args.smoke
        else args.anchor_multiplier
    )
    anchor_time, anchor_states = build_locked_anchors(
        locked,
        cfg,
        params,
        protected_initial,
        multiplier=anchor_multiplier,
        query_batch_size=args.query_batch_size,
    )
    model = construct_gated_model(
        locked,
        probe,
        cfg,
        locked_args,
        anchor_time=anchor_time,
        anchor_states=anchor_states,
        gate_tube=args.gate_tube,
        gate_transition=args.gate_transition,
    ).to(device=device, dtype=dtype)
    model.set_feature_vectors(params["r"], params["phi"])
    model.eval()
    del locked, probe
    assert_only_probe_trainable(model)

    frozen_initial = {
        key: value
        for key, value in cpu_state_dict(model).items()
        if key.startswith(("time_branch.", "state_branch."))
    }
    pmp_multiplier = (
        args.smoke_pmp_multiplier if args.smoke else args.pmp_multiplier
    )
    pmp_cfg = fine_problem(cfg, pmp_multiplier)
    pmp_time, pmp_raw = fixed_support_dense_logits(
        model,
        cfg,
        pmp_multiplier,
        query_batch_size=args.query_batch_size,
    )
    pmp_midpoint_raw = pchip_midpoint_logits(pmp_time, pmp_raw)
    train_count = (
        args.smoke_random_states if args.smoke else args.train_random_states
    )
    validation_count = (
        args.smoke_random_states
        if args.smoke
        else args.validation_random_states
    )
    epochs = args.smoke_epochs if args.smoke else args.epochs
    train_states = random_initial_states(
        train_count,
        args.train_seed,
        args.random_radius,
        pmp_cfg,
        device,
        dtype,
    )
    validation_states = random_initial_states(
        validation_count,
        args.validation_seed,
        args.random_radius,
        pmp_cfg,
        device,
        dtype,
    )

    option = (
        str(getattr(locked_args, "option", "cf"))
        if args.option == "auto"
        else args.option
    )
    if option not in {"cf", "der"}:
        raise ValueError(f"unsupported option {option}")
    cf_weight = args.cf_weight if option == "cf" else 0.0
    cf_scalar_weight = args.cf_scalar_weight if option == "cf" else 0.0
    trainable = model.adaptation_parameters()
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    history: list[dict[str, Any]] = []
    best_validation = math.inf
    best_step = 0
    best_state = cpu_state_dict(model)
    started = time.perf_counter()

    def pack_for(states: torch.Tensor) -> dict[str, torch.Tensor]:
        return pmp_residual_pack(
            model,
            states,
            pmp_cfg,
            pmp_time,
            pmp_raw,
            pmp_midpoint_raw,
            params,
            continuous_collocation=args.continuous_collocation,
            option=option,
            interval_start=args.interval_start,
            interval_end=args.interval_end,
            w0=args.w0,
            w1=args.w1,
            w2=args.w2,
            cf_weight=cf_weight,
            cf_scalar_weight=cf_scalar_weight,
        )

    def evaluate(step: int) -> dict[str, Any]:
        nonlocal best_validation, best_step, best_state
        model.eval()
        with torch.no_grad():
            train_pack = pack_for(train_states)
            validation_pack = pack_for(validation_states)
            train_selection = scalar_selection_value(
                train_pack, w0=args.w0, w1=args.w1, w2=args.w2
            )
            validation_selection = scalar_selection_value(
                validation_pack,
                w0=args.w0,
                w1=args.w1,
                w2=args.w2,
            )
        guard = structural_guard_metrics(
            model,
            protected_initial,
            pmp_cfg,
            pmp_time,
            pmp_raw,
            pmp_midpoint_raw,
            params,
        )
        guard.update(
            standard_task_guard_metrics(
                model,
                protected_initial,
                cfg,
                params,
                include_objective=False,
            )
        )
        eligible = guard_passed(guard, args.protected_tolerance)
        row = {
            "step": step,
            "train_loss": float(train_pack["loss"].cpu()),
            "validation_loss": float(validation_pack["loss"].cpu()),
            "train_selection_pmp": float(train_selection.cpu()),
            "validation_selection_pmp": float(
                validation_selection.cpu()
            ),
            "guard_passed": eligible,
            **guard,
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(row)
        if eligible and row["validation_selection_pmp"] < best_validation:
            best_validation = row["validation_selection_pmp"]
            best_step = step
            best_state = cpu_state_dict(model)
        print(
            f"[{step:04d}] train={row['train_loss']:.8g} "
            f"val={row['validation_loss']:.8g} "
            f"select={row['validation_selection_pmp']:.8g} "
            f"guard={'pass' if eligible else 'fail'}",
            flush=True,
        )
        return row

    evaluate(0)
    for step in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pack = pack_for(train_states)
        pack["loss"].backward()
        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optimizer.step()
        if (
            step == 1
            or step % args.eval_every == 0
            or step == epochs
        ):
            evaluate(step)

    if not math.isfinite(best_validation):
        raise RuntimeError("no gated checkpoint passed the structural guard")
    model.load_state_dict(best_state, strict=True)
    model.freeze_locked_parameters()
    model.eval()
    assert_only_probe_trainable(model)
    frozen_final = {
        key: value
        for key, value in cpu_state_dict(model).items()
        if key.startswith(("time_branch.", "state_branch."))
    }
    _assert_equal_tensor_dicts(
        frozen_initial, frozen_final, label="frozen source/time"
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    output_args = vars(locked_args).copy()
    output_args.update(
        {
            "policy_arch": CHECKPOINT_FORMAT,
            "state_mode": "feedback",
            "option": option,
        }
    )
    inventory = parameter_inventory(model)
    model_state = cpu_state_dict(model)
    payload = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "model_state": model_state,
        "args": output_args,
        "problem": asdict(cfg),
        "nominal_reference": model.nominal_reference.detach().cpu().clone(),
        "best_validation_loss": best_validation,
        "best_selection_loss": best_validation,
        "best_epoch": best_step,
        "selection_metric": "fresh_validation_scalar_PMP_KKT_residual",
        "gated_fallback": {
            "formula": (
                "h_locked + fixed_gate * (h_probe - h_locked)"
            ),
            "centered_state_logits": True,
            "shared_time_branch": True,
            "gate": GATE_PROTOCOL,
            "structural_guard": STRUCTURAL_GUARD_PROTOCOL,
            "anchor_families": ANCHOR_FAMILY_PROTOCOL,
            "base_n": cfg.n,
            "standard_base_grid": standard_base_grid_protocol(cfg.n),
            "formal_protocol": not args.smoke,
            "protected_tolerance": args.protected_tolerance,
            "source_weights_loaded_without_dtype_roundtrip": True,
            "gate_normalization": cfg.n0,
            "gate_tube": args.gate_tube,
            "gate_transition": args.gate_transition,
            "protected_radii": list(protected_radii),
            "anchor_multiplier": anchor_multiplier,
            "train_scope": "probe_state_branch_only",
            "direct_supervision_used_in_cleanup": False,
            "physical_objective_used_in_loss": False,
            "physical_objective_used_in_selection": False,
            "posthoc_identity_audit": True,
            "train_seed": args.train_seed,
            "validation_seed": args.validation_seed,
            "optimizer_seed": args.optimizer_seed,
            "reserved_blind_seed": args.blind_seed,
            "reserved_blind_used_for_training_or_selection": False,
            "locked_checkpoint": {
                "path": str(locked_path),
                "sha256": locked_hash,
            },
            "capability_probe_checkpoint": {
                "path": str(probe_path),
                "sha256": probe_hash,
                "usage": "probe_state_branch_initialization_only",
            },
            "parameter_inventory": inventory,
            "frozen_time_sha256": tensor_group_sha256(
                model_state, prefixes=("time_branch.",)
            ),
            "frozen_locked_state_sha256": tensor_group_sha256(
                model_state, prefixes=("state_branch.",)
            ),
        },
    }
    provisional_checkpoint = (
        out_dir / "provisional_gated_feedback_section5.pt"
    )
    torch.save(payload, provisional_checkpoint)
    write_csv(out_dir / "history.csv", history)

    reloaded, reload_cfg, _, reload_payload = load_gated_feedback_checkpoint(
        provisional_checkpoint,
        _allow_incomplete_posthoc=True,
    )
    if reload_cfg != cfg:
        raise RuntimeError("native gated reload changed the problem")
    reloaded.to(device=device, dtype=dtype).eval()
    reload_params = build_params(reload_cfg, device, dtype)
    reloaded.set_feature_vectors(reload_params["r"], reload_params["phi"])
    reload_guard = structural_guard_metrics(
        reloaded,
        protected_initial,
        pmp_cfg,
        pmp_time,
        pmp_raw,
        pmp_midpoint_raw,
        reload_params,
    )
    reload_guard.update(
        standard_task_guard_metrics(
            reloaded,
            protected_initial,
            cfg,
            reload_params,
            include_objective=False,
        )
    )
    if not guard_passed(reload_guard, args.protected_tolerance):
        raise RuntimeError("native reloaded checkpoint failed structural guard")

    external_source, external_cfg, _ = load_standard_feedback_checkpoint(
        locked_path
    )
    if external_cfg != cfg:
        raise RuntimeError("independently reloaded source problem changed")
    external_source.to(device=device, dtype=dtype).eval()
    external_source.set_feature_vectors(
        reload_params["r"], reload_params["phi"]
    )
    _assert_equal_tensor_dicts(
        reloaded.time_branch.state_dict(),
        external_source.time_branch.state_dict(),
        label="post-save external time branch",
    )
    _assert_equal_tensor_dicts(
        reloaded.state_branch.state_dict(),
        external_source.state_branch.state_dict(),
        label="post-save external locked state branch",
    )

    # This is deliberately post-save and post-selection.  It cannot affect
    # checkpoint ranking, training, or hyperparameters.
    posthoc_identity = external_standard_task_identity_metrics(
        reloaded,
        external_source,
        protected_initial,
        cfg,
        reload_params,
        include_objective=True,
    )
    if not guard_passed(posthoc_identity, args.protected_tolerance):
        raise RuntimeError(
            "post-selection protected task/objective identity failed"
        )
    reload_payload["gated_fallback"].update(
        {
            "posthoc_identity_audit_completed": True,
            "posthoc_protected_identity": posthoc_identity,
            "posthoc_identity_reference": {
                "path": str(locked_path),
                "sha256": locked_hash,
                "evaluator": standard_base_grid_protocol(cfg.n),
            },
        }
    )
    checkpoint = out_dir / "best_gated_feedback_section5.pt"
    torch.save(reload_payload, checkpoint)
    provisional_checkpoint.unlink()

    # Validate the final, metadata-complete artifact once more.
    reloaded, reload_cfg, _, reload_payload = load_gated_feedback_checkpoint(
        checkpoint
    )
    if reload_cfg != cfg:
        raise RuntimeError("final native gated reload changed the problem")
    reloaded.to(device=device, dtype=dtype).eval()
    reload_params = build_params(reload_cfg, device, dtype)
    reloaded.set_feature_vectors(reload_params["r"], reload_params["phi"])
    _assert_equal_tensor_dicts(
        reloaded.time_branch.state_dict(),
        external_source.time_branch.state_dict(),
        label="final external time branch",
    )
    _assert_equal_tensor_dicts(
        reloaded.state_branch.state_dict(),
        external_source.state_branch.state_dict(),
        label="final external locked state branch",
    )
    final_reload_guard = structural_guard_metrics(
        reloaded,
        protected_initial,
        pmp_cfg,
        pmp_time,
        pmp_raw,
        pmp_midpoint_raw,
        reload_params,
    )
    final_reload_guard.update(
        standard_task_guard_metrics(
            reloaded,
            protected_initial,
            cfg,
            reload_params,
            include_objective=False,
        )
    )
    if not guard_passed(final_reload_guard, args.protected_tolerance):
        raise RuntimeError("final metadata-complete artifact failed identity")
    final_posthoc_identity = reload_payload["gated_fallback"][
        "posthoc_protected_identity"
    ]
    try:
        load_standard_feedback_checkpoint(checkpoint)
    except (KeyError, RuntimeError, TypeError, ValueError):
        legacy_loader_failed_closed = True
    else:
        raise RuntimeError(
            "legacy loader silently accepted the gated checkpoint"
        )
    if reload_payload["checkpoint_format"] != CHECKPOINT_FORMAT:
        raise RuntimeError("native gated reload lost its format marker")

    summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_format": CHECKPOINT_FORMAT,
        "mode": "smoke" if args.smoke else "full",
        "option": option,
        "step0": history[0],
        "selected_step": best_step,
        "selected_validation_pmp": best_validation,
        "history": history,
        "parameter_inventory": inventory,
        "final_native_reload_guard": final_reload_guard,
        "posthoc_protected_identity": final_posthoc_identity,
        "native_gated_reload_passed": True,
        "legacy_loader_failed_closed": legacy_loader_failed_closed,
        "physical_objective_used_in_loss": False,
        "physical_objective_used_in_selection": False,
        "physical_objective_computed_posthoc": True,
        "reserved_blind_seed": args.blind_seed,
        "reserved_blind_used": False,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(safe(summary), indent=2, sort_keys=True), flush=True)
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--locked-checkpoint", type=Path, required=True)
    result.add_argument(
        "--capability-probe-checkpoint", type=Path, required=True
    )
    result.add_argument("--out-dir", type=Path, required=True)
    result.add_argument("--expected-locked-sha256", default="")
    result.add_argument("--expected-probe-sha256", default="")
    result.add_argument("--device", default="cpu")
    result.add_argument("--threads", type=int, default=4)
    result.add_argument(
        "--optimizer-seed", type=int, default=OPTIMIZER_SEED
    )

    result.add_argument(
        "--protected-radii", default="0,0.10,0.20,0.40,0.60"
    )
    result.add_argument("--gate-tube", type=float, default=0.005)
    result.add_argument("--gate-transition", type=float, default=0.010)
    result.add_argument("--protected-tolerance", type=float, default=0.0)
    result.add_argument("--anchor-multiplier", type=int, default=16)
    result.add_argument("--query-batch-size", type=int, default=16)

    result.add_argument("--option", choices=("auto", "cf", "der"), default="auto")
    result.add_argument("--pmp-multiplier", type=int, default=2)
    result.add_argument(
        "--continuous-collocation",
        choices=("nodes", "nodes-midpoints"),
        default="nodes",
    )
    result.add_argument("--interval-start", type=float, default=1.5)
    result.add_argument("--interval-end", type=float, default=8.0)
    result.add_argument("--w0", type=float, default=1.0)
    result.add_argument("--w1", type=float, default=1.0)
    result.add_argument("--w2", type=float, default=4.0)
    result.add_argument("--cf-weight", type=float, default=0.01)
    result.add_argument("--cf-scalar-weight", type=float, default=1.0)
    result.add_argument("--random-radius", type=float, default=0.20)
    result.add_argument("--train-random-states", type=int, default=32)
    result.add_argument("--validation-random-states", type=int, default=32)
    result.add_argument("--train-seed", type=int, default=PMP_TRAIN_SEED)
    result.add_argument(
        "--validation-seed", type=int, default=PMP_VALIDATION_SEED
    )
    result.add_argument(
        "--blind-seed", type=int, default=RESERVED_BLIND_SEED
    )

    result.add_argument("--epochs", type=int, default=50)
    result.add_argument("--eval-every", type=int, default=1)
    result.add_argument("--lr", type=float, default=1.0e-5)
    result.add_argument("--weight-decay", type=float, default=0.0)
    result.add_argument("--grad-clip", type=float, default=1.0)

    result.add_argument(
        "--smoke", action=argparse.BooleanOptionalAction, default=False
    )
    result.add_argument("--smoke-epochs", type=int, default=1)
    result.add_argument("--smoke-random-states", type=int, default=2)
    result.add_argument("--smoke-pmp-multiplier", type=int, default=1)
    result.add_argument("--smoke-anchor-multiplier", type=int, default=2)
    return result


if __name__ == "__main__":
    run(parser().parse_args())
