#!/usr/bin/env python3
"""Fit a tiny protected feedback residual, then clean it with PMP/KKT loss.

The source feedback checkpoint is frozen.  A single ``Linear(128, 1)`` head
reuses the final hidden representation of its state branch.  Its output is
multiplied by a fixed distance gate that is exactly zero on source-policy
closed-loop trajectories at the protected structured radii.  Consequently the
adapted policy is mathematically identical to the source policy on those
trajectories (up to the deterministic arithmetic used to reproduce them).

Training has two deliberately separated phases:

1. fit 24 state-specific direct-control teachers, selecting by the MSE on
   eight held-out teacher trajectories;
2. optimize only the PMP/KKT residual on the same 24 initial states, selecting
   by PMP/KKT residual on the eight validation states.

The physical objective J is not part of either loss or checkpoint ranking.
This is an experimental prototype and does not edit manuscript artifacts.
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
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.refine_feedback_svd_null_projected_kkt import (  # noqa: E402
    antithetic_initial_states,
    configure_loss,
    load_locked_feedback_checkpoint,
    policy_rollout,
    posthoc_test_metrics,
    state_hidden,
    structured_initial_states,
)
from scripts.train_feedback_section5 import (  # noqa: E402
    NestedFeedbackTransformer,
    scalar_metrics,
    section5_loss,
)
from train_paper_pmp_kkt import (  # noqa: E402
    ProblemConfig,
    build_params,
    parse_hidden,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_radii(text: str | Iterable[float]) -> tuple[float, ...]:
    if isinstance(text, str):
        values = [float(piece) for piece in text.split(",") if piece.strip()]
    else:
        values = [float(value) for value in text]
    if not values:
        raise ValueError("at least one protected radius is required")
    result: list[float] = []
    for value in values:
        if not math.isfinite(value) or not 0.0 <= value < 1.0:
            raise ValueError("protected radii must lie in [0, 1)")
        if not any(math.isclose(value, old, abs_tol=1.0e-12) for old in result):
            result.append(value)
    return tuple(result)


class ProtectedTrajectoryGate(nn.Module):
    """Fixed gate that vanishes on any protected source trajectory.

    At time ``t``, let ``d²`` be the smallest mean squared relative distance
    from the queried state to the protected source states interpolated at
    ``t``.  The gate is zero inside the calibrated tube, one outside the
    following transition band, and uses the quintic smoothstep
    ``6 z^5 - 15 z^4 + 10 z^3`` in between.  Its first two derivatives vanish
    at both joins, so the fixed gate is C2 where the active protected
    trajectory is unique.
    """

    def __init__(
        self,
        protected_states: torch.Tensor,
        *,
        distance_scale: float,
        zero_tube_squared: float = 0.0,
    ) -> None:
        super().__init__()
        if protected_states.ndim != 3:
            raise ValueError(
                "protected states must have shape (trajectories, nodes, m)"
            )
        if protected_states.shape[1] < 2:
            raise ValueError("protected trajectories need at least two nodes")
        if distance_scale <= 0.0:
            raise ValueError("distance_scale must be positive")
        if zero_tube_squared < 0.0:
            raise ValueError("zero_tube_squared must be nonnegative")
        self.register_buffer(
            "protected_states", protected_states.detach().clone()
        )
        self.distance_scale = float(distance_scale)
        self.zero_tube_squared = float(zero_tube_squared)

    @property
    def intervals(self) -> int:
        return int(self.protected_states.shape[1] - 1)

    def protected_state_at(
        self, normalized_time: torch.Tensor
    ) -> torch.Tensor:
        if normalized_time.ndim == 0:
            normalized_time = normalized_time.reshape(1)
        position = (
            normalized_time.clamp(0.0, 1.0) * self.intervals
        )
        lower = torch.floor(position).to(torch.long)
        upper = (lower + 1).clamp_max(self.intervals)
        fraction = (position - lower.to(position.dtype)).reshape(-1, 1, 1)
        lower_state = self.protected_states[:, lower, :].permute(1, 0, 2)
        upper_state = self.protected_states[:, upper, :].permute(1, 0, 2)
        return (1.0 - fraction) * lower_state + fraction * upper_state

    def distance_squared(
        self, normalized_time: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        if state.ndim != 2:
            raise ValueError("state must have shape (batch, m)")
        if normalized_time.ndim == 0:
            normalized_time = normalized_time.expand(state.shape[0])
        if normalized_time.shape != (state.shape[0],):
            raise ValueError("time must be scalar or have shape (batch,)")
        references = self.protected_state_at(normalized_time)
        scale = references.abs().clamp_min(1.0)
        distance_squared = (
            ((state[:, None, :] - references) / scale)
            .square()
            .mean(dim=-1)
        )
        return distance_squared.min(dim=1).values

    def forward(
        self, normalized_time: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        nearest = self.distance_squared(normalized_time, state)
        # Clamp is joined to constants through a quintic whose first and
        # second derivatives vanish at z=0 and z=1.  This preserves the exact
        # flat zero tube without injecting a C0/C1 kink into higher-order PMP
        # diagnostics.
        z = (
            (nearest - self.zero_tube_squared)
            / self.distance_scale**2
        ).clamp(0.0, 1.0)
        return z.pow(3) * (
            10.0 - 15.0 * z + 6.0 * z.square()
        )


def calibrated_zero_tube_squared(
    protected_states: torch.Tensor,
    protected_stage_states: torch.Tensor,
    *,
    distance_scale: float,
    safety_factor: float,
) -> tuple[float, dict[str, float]]:
    """Calibrate a flat tube from all source RK4 node and stage queries."""

    if safety_factor < 1.0:
        raise ValueError("gate tube safety factor must be at least one")
    gate = ProtectedTrajectoryGate(
        protected_states,
        distance_scale=distance_scale,
        zero_tube_squared=0.0,
    ).to(
        device=protected_states.device,
        dtype=protected_states.dtype,
    )
    batch, intervals, stages, m = protected_stage_states.shape
    if stages != 4 or m != protected_states.shape[-1]:
        raise ValueError("expected four RK4 stage states per interval")
    stage_offsets = torch.tensor(
        [0.0, 0.5, 0.5, 1.0],
        device=protected_states.device,
        dtype=protected_states.dtype,
    )
    interval = torch.arange(
        intervals,
        device=protected_states.device,
        dtype=protected_states.dtype,
    )
    stage_time = (
        interval[:, None] + stage_offsets[None, :]
    ) / intervals
    flat_stage_time = stage_time.reshape(-1).repeat(batch)
    flat_stage_state = protected_stage_states.reshape(-1, m)
    stage_distance = gate.distance_squared(
        flat_stage_time, flat_stage_state
    )
    node_time = torch.linspace(
        0.0,
        1.0,
        protected_states.shape[1],
        device=protected_states.device,
        dtype=protected_states.dtype,
    )
    node_distance = gate.distance_squared(
        node_time.repeat(batch),
        protected_states.reshape(-1, m),
    )
    observed_max = max(
        float(stage_distance.max().detach().cpu()),
        float(node_distance.max().detach().cpu()),
    )
    # The tiny additive margin also covers roundoff when the same evaluator is
    # reconstructed after serialization.
    calibrated = max(
        observed_max * safety_factor,
        observed_max + 64.0 * torch.finfo(protected_states.dtype).eps,
    )
    return calibrated, {
        "observed_node_max_distance_squared": float(
            node_distance.max().detach().cpu()
        ),
        "observed_stage_max_distance_squared": float(
            stage_distance.max().detach().cpu()
        ),
        "safety_factor": float(safety_factor),
        "zero_tube_squared": float(calibrated),
    }


class GatedResidualFeedback(nn.Module):
    """Frozen source policy plus one gated 128-to-1 residual head."""

    def __init__(
        self,
        source: NestedFeedbackTransformer,
        gate: ProtectedTrajectoryGate,
    ) -> None:
        super().__init__()
        self.source = source
        for parameter in self.source.parameters():
            parameter.requires_grad_(False)
        source_parameter = next(source.parameters())
        hidden = state_hidden(
            source,
            torch.zeros(
                1,
                device=source_parameter.device,
                dtype=source_parameter.dtype,
            ),
            torch.full(
                (1, source.m),
                10.0,
                device=source_parameter.device,
                dtype=source_parameter.dtype,
            ),
        )
        self.residual_head = nn.Linear(
            hidden.shape[-1],
            1,
            bias=True,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.gate = gate

    @property
    def adapter_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.residual_head.parameters())

    def train(self, mode: bool = True) -> "GatedResidualFeedback":
        super().train(mode)
        # The locked Transformer/state feature extractor stays deterministic.
        self.source.eval()
        return self

    def time_logits(self, normalized_time_grid: torch.Tensor) -> torch.Tensor:
        return self.source.time_logits(normalized_time_grid)

    def residual_logit(
        self, normalized_time: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = state_hidden(self.source, normalized_time, state)
        residual = self.residual_head(hidden).squeeze(-1)
        gate = self.gate(normalized_time, state)
        return gate * residual, gate

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
                state.shape[0], device=state.device, dtype=state.dtype
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
            * torch.sigmoid(combined / self.source.action_temperature)
            - self.source.action_offset,
            0.0,
            self.source.umax,
        )


def collect_oracles(
    roots: list[Path],
    *,
    expected_n: int,
    expected_m: int,
) -> list[dict[str, torch.Tensor | int | str]]:
    records: dict[int, dict[str, torch.Tensor | int | str]] = {}
    for root in roots:
        root = root.expanduser().resolve()
        for path in sorted(root.glob("random_*.npz")):
            with np.load(path) as data:
                index = int(data["sample_index"])
                state = np.asarray(data["N"], dtype=np.float64)
                control = np.asarray(data["u"], dtype=np.float64)
                initial = np.asarray(data["initial_state"], dtype=np.float64)
            if state.shape != (expected_n + 1, expected_m):
                raise ValueError(f"unexpected state shape in {path}: {state.shape}")
            if control.shape != (expected_n,):
                raise ValueError(
                    f"unexpected control shape in {path}: {control.shape}"
                )
            if index in records:
                raise ValueError(f"duplicate oracle index {index}")
            records[index] = {
                "index": index,
                "path": str(path),
                "initial": torch.from_numpy(initial),
                "states": torch.from_numpy(state),
                "controls": torch.from_numpy(control),
            }
    ordered = [records[index] for index in sorted(records)]
    if len(ordered) < 32:
        raise ValueError(f"need at least 32 direct teachers, found {len(ordered)}")
    return ordered[:32]


def oracle_tensor(
    records: list[dict[str, torch.Tensor | int | str]],
    key: str,
    device: torch.device,
) -> torch.Tensor:
    return torch.stack(
        [record[key] for record in records], dim=0  # type: ignore[list-item]
    ).to(device=device, dtype=torch.float64)


def tensor_row_hashes(value: torch.Tensor) -> list[str]:
    rows = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    return [
        hashlib.sha256(row.numpy().tobytes()).hexdigest()
        for row in rows
    ]


def assert_disjoint_state_sets(
    named_states: dict[str, torch.Tensor],
) -> dict[str, list[str]]:
    hashes = {
        name: tensor_row_hashes(states)
        for name, states in named_states.items()
    }
    names = list(hashes)
    for index, left_name in enumerate(names):
        left = set(hashes[left_name])
        if len(left) != len(hashes[left_name]):
            raise ValueError(f"duplicate states inside {left_name}")
        for right_name in names[index + 1 :]:
            overlap = left.intersection(hashes[right_name])
            if overlap:
                raise ValueError(
                    f"state sets {left_name} and {right_name} overlap"
                )
    return hashes


def teacher_predictions(
    model: GatedResidualFeedback,
    oracle_states: torch.Tensor,
    cfg: ProblemConfig,
    *,
    chunk: int,
) -> torch.Tensor:
    batch = oracle_states.shape[0]
    times = torch.arange(
        cfg.n,
        device=oracle_states.device,
        dtype=oracle_states.dtype,
    ) / cfg.n
    base = model.time_logits(
        torch.linspace(
            0.0,
            1.0,
            cfg.n + 1,
            device=oracle_states.device,
            dtype=oracle_states.dtype,
        )
    )[: cfg.n]
    predictions: list[torch.Tensor] = []
    flat_time = times.repeat(batch)
    flat_state = oracle_states[:, :-1].reshape(-1, cfg.m)
    flat_base = base.repeat(batch)
    for start in range(0, flat_state.shape[0], chunk):
        stop = min(start + chunk, flat_state.shape[0])
        predictions.append(
            model.interval_action(
                flat_base[start:stop],
                flat_time[start:stop],
                flat_state[start:stop],
                state_mode="feedback",
            )
        )
    return torch.cat(predictions).reshape(batch, cfg.n)


def teacher_mse(
    model: GatedResidualFeedback,
    states: torch.Tensor,
    controls: torch.Tensor,
    cfg: ProblemConfig,
    *,
    chunk: int,
) -> torch.Tensor:
    prediction = teacher_predictions(model, states, cfg, chunk=chunk)
    return (prediction - controls).square().mean()


@torch.no_grad()
def trajectory_gate_distribution(
    gate: ProtectedTrajectoryGate,
    states: torch.Tensor,
    cfg: ProblemConfig,
    *,
    chunk: int = 16384,
) -> dict[str, float]:
    batch = states.shape[0]
    flat_time = (
        torch.arange(
            cfg.n,
            device=states.device,
            dtype=states.dtype,
        )
        / cfg.n
    ).repeat(batch)
    flat_state = states[:, :-1].reshape(-1, cfg.m)
    values: list[torch.Tensor] = []
    for start in range(0, flat_state.shape[0], chunk):
        stop = min(start + chunk, flat_state.shape[0])
        values.append(gate(flat_time[start:stop], flat_state[start:stop]))
    value = torch.cat(values)
    return {
        "minimum": float(value.min().cpu()),
        "mean": float(value.mean().cpu()),
        "maximum": float(value.max().cpu()),
        "positive_fraction": float((value > 0.0).double().mean().cpu()),
        "above_0p5_fraction": float((value > 0.5).double().mean().cpu()),
    }


@torch.no_grad()
def protected_identity_metrics(
    source: NestedFeedbackTransformer,
    adapted: GatedResidualFeedback,
    initial: torch.Tensor,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
) -> dict[str, float]:
    source_rollout = policy_rollout(
        source, initial, cfg, params, state_mode="feedback"
    )
    adapted_rollout = policy_rollout(
        adapted, initial, cfg, params, state_mode="feedback"
    )
    batch = initial.shape[0]
    time = torch.arange(
        cfg.n,
        device=initial.device,
        dtype=initial.dtype,
    ) / cfg.n
    gate = adapted.gate(
        time.repeat(batch),
        source_rollout.states[:, :-1].reshape(-1, cfg.m),
    )
    return {
        "gate_max_abs": float(gate.abs().max().cpu()),
        "control_max_abs": float(
            (source_rollout.controls - adapted_rollout.controls)
            .abs()
            .max()
            .cpu()
        ),
        "state_max_abs": float(
            (source_rollout.states - adapted_rollout.states)
            .abs()
            .max()
            .cpu()
        ),
        "objective_max_abs": float(
            (source_rollout.objectives - adapted_rollout.objectives)
            .abs()
            .max()
            .cpu()
        ),
    }


def load_adapter_artifact(
    path: Path,
    *,
    device: torch.device,
) -> tuple[
    dict[str, Any],
    NestedFeedbackTransformer,
    GatedResidualFeedback,
    ProblemConfig,
]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") == "exact_gated_compact_residual_adapter_v1":
        from scripts.compact_exact_gated_residual_adapter import (
            load_compact_adapter_artifact,
        )

        return load_compact_adapter_artifact(path, device=device)
    if payload.get("format") not in {
        "gated_residual_adapter_v1",
        "gated_residual_adapter_v2",
    }:
        raise ValueError("unsupported gated residual artifact")
    if payload.get("format") == "gated_residual_adapter_v2":
        if payload.get("gate_transition") != "quintic_smoothstep_c2":
            raise ValueError("unexpected v2 gate transition")
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
    ).double()
    source.load_state_dict(payload["source_model_state"])
    source.set_nominal_reference(
        payload["nominal_reference"].to(dtype=torch.float64)
    )
    source.to(device=device, dtype=torch.float64).eval()
    feature_params = build_params(cfg, device, torch.float64)
    source.set_feature_vectors(
        feature_params["r"], feature_params["phi"]
    )
    gate = ProtectedTrajectoryGate(
        payload["protected_states"].to(
            device=device, dtype=torch.float64
        ),
        distance_scale=float(payload["gate_distance_scale"]),
        zero_tube_squared=float(payload["gate_zero_tube_squared"]),
    ).to(device=device, dtype=torch.float64)
    adapter = GatedResidualFeedback(source, gate).to(
        device=device, dtype=torch.float64
    )
    adapter.residual_head.load_state_dict(payload["residual_head_state"])
    adapter.eval()
    return payload, source, adapter, cfg


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def load_initial_head(
    path: Path,
    *,
    model: GatedResidualFeedback,
    source_payload: dict[str, Any],
    train_records: list[dict[str, torch.Tensor | int | str]],
    validation_records: list[dict[str, torch.Tensor | int | str]],
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "residual_head_state" not in payload:
        raise ValueError("initial-head artifact has no residual_head_state")
    state = payload["residual_head_state"]
    expected = model.residual_head.state_dict()
    if state.keys() != expected.keys():
        raise ValueError("initial-head state-dict keys do not match")
    if sum(value.numel() for value in state.values()) != 129:
        raise ValueError("initial-head artifact is not a 129-parameter head")
    for key, value in state.items():
        if value.shape != expected[key].shape:
            raise ValueError(f"initial-head shape mismatch at {key}")
    if payload.get("problem") != source_payload["problem"]:
        raise ValueError("initial-head ProblemConfig does not match source")
    archived_source = payload.get("source_model_state")
    if archived_source is None or archived_source.keys() != source_payload[
        "model_state"
    ].keys():
        raise ValueError("initial-head source state is missing or mismatched")
    if not all(
        torch.equal(
            archived_source[key].detach().cpu(),
            source_payload["model_state"][key].detach().cpu(),
        )
        for key in archived_source
    ):
        raise ValueError("initial-head was fit from another source model")
    split = payload.get("direct_teacher_indices", {})
    expected_train = [int(record["index"]) for record in train_records]
    expected_validation = [
        int(record["index"]) for record in validation_records
    ]
    if split.get("train") != expected_train:
        raise ValueError("initial-head teacher training split differs")
    if split.get("validation") != expected_validation:
        raise ValueError("initial-head teacher validation split differs")
    if bool(payload.get("physical_objective_used", False)):
        raise ValueError("initial-head artifact used the physical objective")
    if bool(payload.get("physical_objective_used_for_ranking", False)):
        raise ValueError("initial-head artifact ranked with physical objective")
    model.residual_head.load_state_dict(state)
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "format": payload.get("format"),
        "solver": payload.get("solver"),
    }


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.optimizer_seed)
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    source_payload, source, cfg, source_args = load_locked_feedback_checkpoint(
        args.checkpoint.expanduser().resolve()
    )
    source.to(device=device, dtype=torch.float64).eval()
    params = build_params(cfg, device, torch.float64)
    source.set_feature_vectors(params["r"], params["phi"])
    protected_radii = parse_radii(args.protected_radii)
    protected_initial = structured_initial_states(
        protected_radii, cfg, device, torch.float64
    )
    with torch.no_grad():
        protected_rollout = policy_rollout(
            source,
            protected_initial,
            cfg,
            params,
            state_mode="feedback",
        )
    zero_tube_squared, tube_calibration = calibrated_zero_tube_squared(
        protected_rollout.states,
        protected_rollout.stage_states,
        distance_scale=args.gate_distance_scale,
        safety_factor=args.gate_tube_safety_factor,
    )
    gate = ProtectedTrajectoryGate(
        protected_rollout.states,
        distance_scale=args.gate_distance_scale,
        zero_tube_squared=zero_tube_squared,
    ).to(device=device, dtype=torch.float64)
    model = GatedResidualFeedback(source, gate).to(
        device=device, dtype=torch.float64
    )
    if model.adapter_parameter_count != args.expected_adapter_parameters:
        raise ValueError(
            f"adapter has {model.adapter_parameter_count} parameters, "
            f"expected {args.expected_adapter_parameters}"
        )
    trainable_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if trainable_count != model.adapter_parameter_count:
        raise RuntimeError(
            f"{trainable_count} parameters are trainable; expected only "
            f"the {model.adapter_parameter_count}-parameter residual head"
        )
    if any(
        bool(parameter.detach().count_nonzero())
        for parameter in model.residual_head.parameters()
    ):
        raise RuntimeError("residual head did not start from exact zero")

    records = collect_oracles(
        [Path(value) for value in args.oracle_root],
        expected_n=cfg.n,
        expected_m=cfg.m,
    )
    train_records = records[: args.teacher_train_count]
    validation_records = records[
        args.teacher_train_count :
        args.teacher_train_count + args.teacher_validation_count
    ]
    if len(train_records) != args.teacher_train_count:
        raise ValueError("not enough training teachers")
    if len(validation_records) != args.teacher_validation_count:
        raise ValueError("not enough validation teachers")
    train_oracle_states = oracle_tensor(train_records, "states", device)
    train_oracle_controls = oracle_tensor(train_records, "controls", device)
    validation_oracle_states = oracle_tensor(
        validation_records, "states", device
    )
    validation_oracle_controls = oracle_tensor(
        validation_records, "controls", device
    )
    teacher_train_initial = oracle_tensor(
        train_records, "initial", device
    )
    teacher_validation_initial = oracle_tensor(
        validation_records, "initial", device
    )
    cleanup_train_initial = antithetic_initial_states(
        args.cleanup_train_count,
        args.cleanup_train_seed,
        args.cleanup_radius,
        cfg,
        device,
        torch.float64,
        antithetic=True,
    )
    cleanup_validation_initial = antithetic_initial_states(
        args.cleanup_validation_count,
        args.cleanup_validation_seed,
        args.cleanup_radius,
        cfg,
        device,
        torch.float64,
        antithetic=True,
    )
    blind_test_initial = antithetic_initial_states(
        args.blind_test_count,
        args.blind_test_seed,
        args.cleanup_radius,
        cfg,
        device,
        torch.float64,
        antithetic=False,
    )
    state_hashes = assert_disjoint_state_sets(
        {
            "teacher_train": teacher_train_initial,
            "teacher_validation": teacher_validation_initial,
            "pmp_train": cleanup_train_initial,
            "pmp_validation": cleanup_validation_initial,
            "blind_test": blind_test_initial,
        }
    )

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []

    initial_head_metadata: dict[str, Any] | None = None
    if args.initial_head is not None:
        initial_head_metadata = load_initial_head(
            args.initial_head.expanduser().resolve(),
            model=model,
            source_payload=source_payload,
            train_records=train_records,
            validation_records=validation_records,
        )

    teacher_optimizer = torch.optim.AdamW(
        model.residual_head.parameters(),
        lr=args.teacher_lr,
        weight_decay=args.weight_decay,
    )
    best_teacher_value = math.inf
    best_teacher_epoch = 0
    best_teacher_state = copy.deepcopy(model.residual_head.state_dict())

    def evaluate_teacher(epoch: int) -> None:
        nonlocal best_teacher_value, best_teacher_epoch, best_teacher_state
        model.eval()
        with torch.no_grad():
            train_value = float(
                teacher_mse(
                    model,
                    train_oracle_states,
                    train_oracle_controls,
                    cfg,
                    chunk=args.teacher_chunk,
                ).cpu()
            )
            validation_value = float(
                teacher_mse(
                    model,
                    validation_oracle_states,
                    validation_oracle_controls,
                    cfg,
                    chunk=args.teacher_chunk,
                ).cpu()
            )
        history.append(
            {
                "phase": "direct_teacher_initialization",
                "step": epoch,
                "train_loss": train_value,
                "validation_loss": validation_value,
            }
        )
        if validation_value < best_teacher_value:
            best_teacher_value = validation_value
            best_teacher_epoch = epoch
            best_teacher_state = copy.deepcopy(
                model.residual_head.state_dict()
            )
        print(
            f"[teacher {epoch:04d}] train={train_value:.7g} "
            f"val={validation_value:.7g}",
            flush=True,
        )

    teacher_started = time.perf_counter()
    evaluate_teacher(0)
    for epoch in range(1, args.teacher_epochs + 1):
        model.train()
        teacher_optimizer.zero_grad(set_to_none=True)
        loss = teacher_mse(
            model,
            train_oracle_states,
            train_oracle_controls,
            cfg,
            chunk=args.teacher_chunk,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.residual_head.parameters(), args.grad_clip
        )
        teacher_optimizer.step()
        if (
            epoch == 1
            or epoch % args.teacher_eval_every == 0
            or epoch == args.teacher_epochs
        ):
            evaluate_teacher(epoch)
    model.residual_head.load_state_dict(best_teacher_state)
    teacher_elapsed_seconds = time.perf_counter() - teacher_started

    option = (
        str(getattr(source_args, "option", "der"))
        if args.option == "auto"
        else args.option
    )
    loss_args = configure_loss(source_args, option, args)
    cleanup_optimizer = torch.optim.AdamW(
        model.residual_head.parameters(),
        lr=args.cleanup_lr,
        weight_decay=args.weight_decay,
    )
    best_cleanup_value = math.inf
    best_cleanup_epoch = 0
    best_cleanup_state = copy.deepcopy(model.residual_head.state_dict())
    best_cleanup_metrics: dict[str, float] | None = None

    def evaluate_cleanup(epoch: int) -> None:
        nonlocal best_cleanup_value, best_cleanup_epoch
        nonlocal best_cleanup_state, best_cleanup_metrics
        model.eval()
        with torch.no_grad():
            train_pack = section5_loss(
                model, cleanup_train_initial, cfg, params, loss_args
            )
            validation_pack = section5_loss(
                model, cleanup_validation_initial, cfg, params, loss_args
            )
        train_metrics = scalar_metrics(train_pack, cfg, loss_args)
        validation_metrics = scalar_metrics(
            validation_pack, cfg, loss_args
        )
        value = validation_metrics["loss"]
        history.append(
            {
                "phase": "pmp_kkt_cleanup",
                "step": epoch,
                "train_loss": train_metrics["loss"],
                "validation_loss": value,
            }
        )
        if value < best_cleanup_value:
            best_cleanup_value = value
            best_cleanup_epoch = epoch
            best_cleanup_state = copy.deepcopy(
                model.residual_head.state_dict()
            )
            best_cleanup_metrics = copy.deepcopy(validation_metrics)
        print(
            f"[cleanup {epoch:04d}] train={train_metrics['loss']:.7g} "
            f"val={value:.7g}",
            flush=True,
        )

    cleanup_started = time.perf_counter()
    evaluate_cleanup(0)
    for epoch in range(1, args.cleanup_epochs + 1):
        model.train()
        cleanup_optimizer.zero_grad(set_to_none=True)
        batch_start = (
            (epoch - 1) * args.cleanup_batch_size
        ) % cleanup_train_initial.shape[0]
        indices = (
            torch.arange(
                args.cleanup_batch_size,
                device=device,
                dtype=torch.long,
            )
            + batch_start
        ) % cleanup_train_initial.shape[0]
        pack = section5_loss(
            model,
            cleanup_train_initial.index_select(0, indices),
            cfg,
            params,
            loss_args,
        )
        pack["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            model.residual_head.parameters(), args.grad_clip
        )
        cleanup_optimizer.step()
        if (
            epoch == 1
            or epoch % args.cleanup_eval_every == 0
            or epoch == args.cleanup_epochs
        ):
            evaluate_cleanup(epoch)
    model.residual_head.load_state_dict(best_cleanup_state)
    cleanup_elapsed_seconds = time.perf_counter() - cleanup_started
    model.eval()

    teacher_train_gate = trajectory_gate_distribution(
        model.gate, train_oracle_states, cfg
    )
    teacher_validation_gate = trajectory_gate_distribution(
        model.gate, validation_oracle_states, cfg
    )
    identity = protected_identity_metrics(
        source, model, protected_initial, cfg, params
    )
    with torch.no_grad():
        random_gate = model.gate(
            torch.zeros(
                blind_test_initial.shape[0],
                device=device,
                dtype=torch.float64,
            ),
            blind_test_initial,
        )
    if identity["gate_max_abs"] != 0.0:
        raise RuntimeError("protected gate is not exactly zero")
    if identity["control_max_abs"] != 0.0:
        raise RuntimeError("protected controls changed")
    if float(random_gate.max().cpu()) <= 0.0:
        raise RuntimeError("gate is zero on all random validation states")
    if best_cleanup_metrics is None:
        raise RuntimeError("cleanup produced no checkpoint")

    artifact = {
        "format": "gated_residual_adapter_v2",
        "source_checkpoint": str(args.checkpoint.expanduser().resolve()),
        "source_checkpoint_sha256": sha256(
            args.checkpoint.expanduser().resolve()
        ),
        "source_model_state": source_payload["model_state"],
        "source_args": vars(source_args),
        "problem": source_payload["problem"],
        "nominal_reference": source_payload["nominal_reference"],
        "protected_radii": protected_radii,
        "protected_states": model.gate.protected_states.detach().cpu(),
        "gate_distance_scale": args.gate_distance_scale,
        "gate_zero_tube_squared": zero_tube_squared,
        "gate_transition": "quintic_smoothstep_c2",
        "gate_tube_calibration": tube_calibration,
        "residual_head_state": {
            key: value.detach().cpu()
            for key, value in model.residual_head.state_dict().items()
        },
        "adapter_parameter_count": model.adapter_parameter_count,
        "training_contract": {
            "teacher_train_count": args.teacher_train_count,
            "teacher_validation_count": args.teacher_validation_count,
            "teacher_selection_metric": "validation direct-control MSE",
            "cleanup_loss": "PMP/KKT residual only",
            "cleanup_selection_metric": "validation PMP/KKT loss",
            "physical_objective_in_loss": False,
            "physical_objective_in_ranking": False,
            "pmp_train_seed": args.cleanup_train_seed,
            "pmp_validation_seed": args.cleanup_validation_seed,
            "blind_test_seed": args.blind_test_seed,
            "blind_test_used_for_selection": False,
            "initial_head": initial_head_metadata,
        },
        "state_hashes": state_hashes,
    }
    artifact_path = out_dir / "best_gated_residual_adapter.pt"
    torch.save(artifact, artifact_path)
    # Reload the serialized native source + adapter and enforce the hard
    # compatibility invariants after the exact save/load path.
    (
        reloaded_payload,
        reloaded_source,
        reloaded_model,
        reloaded_cfg,
    ) = load_adapter_artifact(artifact_path, device=device)
    if reloaded_cfg != cfg:
        raise RuntimeError("ProblemConfig changed after adapter reload")
    source_state = source.state_dict()
    reloaded_source_state = reloaded_source.state_dict()
    if source_state.keys() != reloaded_source_state.keys():
        raise RuntimeError("source model keys changed after reload")
    if not all(
        torch.equal(source_state[key], reloaded_source_state[key])
        for key in source_state
    ):
        raise RuntimeError("a frozen source-model tensor changed")
    for key, value in source_state.items():
        if key.startswith("time_branch.") and not torch.equal(
            value, reloaded_source_state[key]
        ):
            raise RuntimeError(f"time branch changed at {key}")
    if not torch.equal(
        source.nominal_reference,
        reloaded_source.nominal_reference,
    ):
        raise RuntimeError("nominal reference changed after reload")
    action_fields = (
        "umax",
        "correction_gain",
        "action_temperature",
        "action_scale",
        "action_parameterization",
        "action_offset",
    )
    for name in action_fields:
        if getattr(source, name) != getattr(reloaded_source, name):
            raise RuntimeError(f"action configuration changed: {name}")
    reloaded_identity = protected_identity_metrics(
        reloaded_source,
        reloaded_model,
        protected_initial,
        cfg,
        params,
    )
    if reloaded_identity["gate_max_abs"] > 1.0e-12:
        raise RuntimeError("reloaded protected gate exceeds 1e-12")
    if reloaded_identity["control_max_abs"] > 1.0e-10:
        raise RuntimeError("reloaded protected control drift exceeds 1e-10")
    with torch.no_grad():
        source_blind = policy_rollout(
            reloaded_source,
            blind_test_initial,
            cfg,
            params,
            state_mode="feedback",
        )
        candidate_blind = policy_rollout(
            reloaded_model,
            blind_test_initial,
            cfg,
            params,
            state_mode="feedback",
        )
        frozen_time_blind = policy_rollout(
            reloaded_source,
            blind_test_initial,
            cfg,
            params,
            state_mode="w_zero",
        )
    blind_posthoc = posthoc_test_metrics(
        candidate_blind,
        source_blind,
        frozen_time_blind,
        physical_scale_factor=1.0 / cfg.alpha,
    )
    write_history(out_dir / "history.csv", history)
    summary = {
        "artifact": str(artifact_path),
        "source_checkpoint": str(args.checkpoint.expanduser().resolve()),
        "option": option,
        "adapter_parameter_count": model.adapter_parameter_count,
        "protected_radii": protected_radii,
        "gate_transition": "quintic_smoothstep_c2",
        "protected_identity": identity,
        "reloaded_protected_identity": reloaded_identity,
        "gate_tube_calibration": tube_calibration,
        "random_validation_gate": {
            "minimum": float(random_gate.min().cpu()),
            "mean": float(random_gate.mean().cpu()),
            "maximum": float(random_gate.max().cpu()),
        },
        "direct_teacher_gate": {
            "train_24": teacher_train_gate,
            "validation_8": teacher_validation_gate,
        },
        "teacher_phase": {
            "best_epoch": best_teacher_epoch,
            "validation_mse": best_teacher_value,
            "elapsed_seconds": teacher_elapsed_seconds,
            "initial_head": initial_head_metadata,
        },
        "cleanup_phase": {
            "best_epoch": best_cleanup_epoch,
            "validation_pmp_kkt_loss": best_cleanup_value,
            "validation_metrics": best_cleanup_metrics,
            "elapsed_seconds": cleanup_elapsed_seconds,
        },
        "blind_test": {
            "seed": args.blind_test_seed,
            "count": args.blind_test_count,
            "used_for_training": False,
            "used_for_checkpoint_selection": False,
            **blind_posthoc,
        },
        "state_set_hashes": state_hashes,
        "source_invariants": {
            "all_state_dict_tensors_equal": True,
            "time_branch_tensors_equal": True,
            "problem_config_equal": True,
            "nominal_reference_equal": True,
            "action_config_equal": True,
        },
        "physical_objective_in_loss": False,
        "physical_objective_in_ranking": False,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {artifact_path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--initial-head",
        type=Path,
        default=None,
        help=(
            "optional read-only 129-parameter teacher-fit head; it is "
            "strictly validated against the source checkpoint and 24/8 split"
        ),
    )
    parser.add_argument(
        "--oracle-root", action="append", required=True
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--option", choices=["auto", "cf", "der"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--optimizer-seed", type=int, default=20260730)
    parser.add_argument(
        "--protected-radii", default="0,0.10,0.20,0.40,0.60"
    )
    parser.add_argument("--gate-distance-scale", type=float, default=0.02)
    parser.add_argument(
        "--gate-tube-safety-factor", type=float, default=2.0
    )
    parser.add_argument("--expected-adapter-parameters", type=int, default=129)
    parser.add_argument("--teacher-train-count", type=int, default=24)
    parser.add_argument("--teacher-validation-count", type=int, default=8)
    parser.add_argument("--teacher-epochs", type=int, default=300)
    parser.add_argument("--teacher-eval-every", type=int, default=10)
    parser.add_argument("--teacher-lr", type=float, default=3.0e-3)
    parser.add_argument("--teacher-chunk", type=int, default=8192)
    parser.add_argument("--cleanup-epochs", type=int, default=100)
    parser.add_argument("--cleanup-eval-every", type=int, default=10)
    parser.add_argument("--cleanup-batch-size", type=int, default=2)
    parser.add_argument("--cleanup-lr", type=float, default=1.0e-4)
    parser.add_argument("--cleanup-radius", type=float, default=0.20)
    parser.add_argument("--cleanup-train-count", type=int, default=24)
    parser.add_argument("--cleanup-validation-count", type=int, default=8)
    parser.add_argument("--cleanup-train-seed", type=int, default=20260810)
    parser.add_argument(
        "--cleanup-validation-seed", type=int, default=20260811
    )
    parser.add_argument("--blind-test-count", type=int, default=128)
    parser.add_argument("--blind-test-seed", type=int, default=20260820)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--cf-candidate-weight", type=float, default=0.01)
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--w0", type=float, default=1.0)
    parser.add_argument("--w1", type=float, default=1.0)
    parser.add_argument("--w2", type=float, default=4.0)
    parser.add_argument("--w-lc", type=float, default=1.0)
    parser.add_argument("--psi-scale", type=float, default=1.0)
    parser.add_argument("--dot-scale", type=float, default=1.0)
    parser.add_argument("--ddot-scale", type=float, default=1.0)
    parser.add_argument("--B-scale", type=float, default=1.0)
    parser.add_argument("--full-gradient-weight", type=float, default=0.0)
    parser.add_argument("--full-gradient-max-weight", type=float, default=0.0)
    parser.add_argument("--full-gradient-projection-step", type=float, default=1.0)
    parser.add_argument("--full-gradient-scale", type=float, default=0.0)
    parser.add_argument("--full-gradient-max-tau", type=float, default=0.1)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
