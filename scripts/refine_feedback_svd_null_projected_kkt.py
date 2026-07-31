#!/usr/bin/env python3
"""Refine a locked feedback policy inside an approximate anchor nullspace.

The incoming CF or DER policy is left structurally unchanged.  All parameters
are frozen except a low-dimensional vector ``z`` that parameterizes an update
to the existing final state-branch weight,

    delta_w = V z,

where the columns of ``V`` span an approximate right nullspace of the hidden
feature rows collected on protected structured-shift trajectories.  Training
uses only the trajectory-wise PMP optimality-gap and the complete projected
reduced-gradient/KKT residual.  The physical objective is never a loss or a
checkpoint-ranking metric.

After selection, ``delta_w`` is folded into the native final ``nn.Linear`` and
the result is saved as a standard ``best_feedback_section5.pt`` checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.train_feedback_section5 import (  # noqa: E402
    NestedFeedbackTransformer,
    make_fixed_directions,
    rk4_objective_per_sample,
    scalar_metrics,
    section5_loss,
    simulate_feedback_rk4_stagewise,
    test_states_from_directions,
)
from train_paper_pmp_kkt import (  # noqa: E402
    ProblemConfig,
    build_params,
    parse_hidden,
)


@dataclass(frozen=True)
class NullspaceReport:
    rows: int
    columns: int
    strict_tolerance: float
    strict_rank: int
    strict_nullity: int
    approximate_relative_tolerance: float
    approximate_tolerance: float
    approximate_rank: int
    approximate_nullity: int
    largest_singular_value: float
    smallest_singular_value: float
    basis_constraint_max_abs: float
    basis_constraint_rms: float


@dataclass
class GuardRollout:
    radii: tuple[float, ...]
    states: torch.Tensor
    controls: torch.Tensor
    stage_states: torch.Tensor
    objectives: torch.Tensor
    terminal_resistant_burden: torch.Tensor
    common_r_sing: torch.Tensor | None = None


@dataclass
class PolicyRollout:
    states: torch.Tensor
    controls: torch.Tensor
    stage_states: torch.Tensor
    objectives: torch.Tensor


class SVDNullLinear(nn.Module):
    """A temporary final linear layer with ``weight = base + V z``."""

    def __init__(
        self,
        source: nn.Linear,
        basis: torch.Tensor,
    ) -> None:
        super().__init__()
        if source.out_features != 1:
            raise ValueError("the state-branch output layer must be scalar")
        if basis.ndim != 2 or basis.shape[0] != source.in_features:
            raise ValueError(
                "nullspace basis must have shape "
                f"({source.in_features}, k), received {tuple(basis.shape)}"
            )
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.register_buffer(
            "base_weight", source.weight.detach().clone()
        )
        if source.bias is None:
            self.register_buffer("base_bias", torch.empty(0))
        else:
            self.register_buffer(
                "base_bias", source.bias.detach().clone()
            )
        self.register_buffer("basis", basis.detach().clone())
        self.z = nn.Parameter(
            torch.zeros(
                basis.shape[1],
                device=basis.device,
                dtype=basis.dtype,
            )
        )

    def effective_weight(self) -> torch.Tensor:
        update = self.basis @ self.z
        return self.base_weight + update.reshape_as(self.base_weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        bias = self.base_bias if self.base_bias.numel() else None
        return F.linear(value, self.effective_weight(), bias)

    def folded_linear(self) -> nn.Linear:
        layer = nn.Linear(
            self.in_features,
            self.out_features,
            bias=bool(self.base_bias.numel()),
            device=self.base_weight.device,
            dtype=self.base_weight.dtype,
        )
        with torch.no_grad():
            layer.weight.copy_(self.effective_weight())
            if layer.bias is not None:
                layer.bias.copy_(self.base_bias)
        return layer


def load_locked_feedback_checkpoint(
    path: Path,
) -> tuple[
    dict[str, Any],
    NestedFeedbackTransformer,
    ProblemConfig,
    argparse.Namespace,
]:
    """Load a self-contained feedback checkpoint without source-path access.

    Some archived checkpoints retain an absolute ``time_checkpoint`` path from
    the server even though their native ``model_state`` already contains the
    complete time branch.  Reconstructing directly from that state keeps this
    refinement portable and avoids silently depending on the old path.
    """

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    argument_values = dict(checkpoint.get("args", {}))
    defaults = {
        "state_scale": 15.0,
        "state_hidden": "128,128",
        "d_model": 64,
        "heads": 4,
        "layers": 2,
        "init_u": 1.5,
        "correction_gain": 1.0,
        "state_feature_mode": "burden_composition",
        "center_state_correction": True,
        "action_temperature": 1.0,
        "action_scale": 1.0,
        "action_parameterization": "logit-temperature",
        "action_offset": 0.0,
        "option": "der",
        "loss_variant": "lc_live",
        "state_mode": "feedback",
        "training_integrator": "rk4",
    }
    for key, value in defaults.items():
        argument_values.setdefault(key, value)
    source_args = argparse.Namespace(**argument_values)
    cfg = ProblemConfig(**checkpoint["problem"])
    model = NestedFeedbackTransformer(
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
    state = checkpoint["model_state"]
    if not any(key.startswith("time_branch.") for key in state):
        raise ValueError(
            "checkpoint is not self-contained: model_state has no time branch"
        )
    model.load_state_dict(state)
    if "nominal_reference" not in checkpoint:
        raise ValueError("checkpoint is missing its nominal reference")
    model.set_nominal_reference(
        checkpoint["nominal_reference"].to(dtype=torch.float64)
    )
    return checkpoint, model, cfg, source_args


def parse_radii(text: str | Iterable[float]) -> tuple[float, ...]:
    if isinstance(text, str):
        pieces = [piece.strip() for piece in text.split(",")]
        values = [float(piece) for piece in pieces if piece]
    else:
        values = [float(value) for value in text]
    if not values:
        raise ValueError("at least one structured radius is required")
    result: list[float] = []
    for value in values:
        if not math.isfinite(value) or not 0.0 <= value < 1.0:
            raise ValueError(
                "structured radii must be finite and lie in [0, 1)"
            )
        if not any(math.isclose(value, prior) for prior in result):
            result.append(value)
    return tuple(result)


def structured_initial_states(
    radii: Iterable[float],
    cfg: ProblemConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    direction = torch.linspace(
        -1.0, 1.0, cfg.m, device=device, dtype=dtype
    )
    return torch.stack(
        [
            cfg.n0 * (1.0 + float(radius) * direction)
            for radius in radii
        ],
        dim=0,
    )


def antithetic_initial_states(
    count: int,
    seed: int,
    radius: float,
    cfg: ProblemConfig,
    device: torch.device,
    dtype: torch.dtype,
    *,
    antithetic: bool,
) -> torch.Tensor:
    if count <= 0:
        raise ValueError("random-state count must be positive")
    if not 0.0 <= radius < 1.0:
        raise ValueError("random-state radius must lie in [0, 1)")
    if antithetic:
        half = (count + 1) // 2
        directions = make_fixed_directions(
            half, cfg.m, seed, torch.float64
        )
        directions = torch.cat((directions, -directions), dim=0)[:count]
    else:
        directions = make_fixed_directions(
            count, cfg.m, seed, torch.float64
        )
    return test_states_from_directions(
        directions, radius, cfg, device, dtype
    )


def final_state_linear(
    model: NestedFeedbackTransformer,
) -> tuple[int, nn.Linear]:
    candidates = [
        (int(name), module)
        for name, module in model.state_branch.named_children()
        if isinstance(module, nn.Linear)
    ]
    if not candidates:
        raise ValueError("state branch contains no linear layer")
    index, layer = candidates[-1]
    if layer.out_features != 1:
        raise ValueError("final state-branch layer must have one output")
    return index, layer


def state_hidden(
    model: NestedFeedbackTransformer,
    normalized_time: torch.Tensor,
    state: torch.Tensor,
) -> torch.Tensor:
    value = model.state_features(normalized_time, state)
    modules = list(model.state_branch.children())
    if not modules or not isinstance(modules[-1], (nn.Linear, SVDNullLinear)):
        raise ValueError("state branch does not end in a supported linear head")
    for module in modules[:-1]:
        value = module(value)
    return value


def anchor_feature_matrix(
    model: NestedFeedbackTransformer,
    states: torch.Tensor,
    cfg: ProblemConfig,
) -> torch.Tensor:
    """Rows whose product with ``delta_w`` changes anchor corrections."""

    if states.ndim != 3 or states.shape[1:] != (cfg.n + 1, cfg.m):
        raise ValueError(
            "anchor trajectories must have shape "
            f"(batch, {cfg.n + 1}, {cfg.m})"
        )
    batch = states.shape[0]
    time = torch.linspace(
        0.0,
        1.0,
        cfg.n + 1,
        device=states.device,
        dtype=states.dtype,
    )
    flat_time = time.repeat(batch)
    flat_states = states.reshape(-1, cfg.m)
    hidden = state_hidden(model, flat_time, flat_states)
    if model.center_state_correction:
        reference = model.nominal_state_at(flat_time)
        hidden = hidden - state_hidden(model, flat_time, reference)
    return hidden


def approximate_right_nullspace(
    matrix: torch.Tensor,
    relative_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, NullspaceReport]:
    if matrix.ndim != 2 or not matrix.numel():
        raise ValueError("constraint matrix must be a nonempty matrix")
    if not 0.0 < relative_tolerance < 1.0:
        raise ValueError("SVD relative tolerance must lie in (0, 1)")
    cpu_matrix = matrix.detach().to(device="cpu", dtype=torch.float64)
    _, singular_values, vh = torch.linalg.svd(
        cpu_matrix, full_matrices=True
    )
    largest = float(singular_values[0])
    smallest = float(singular_values[-1])
    strict_tolerance = (
        torch.finfo(cpu_matrix.dtype).eps
        * max(cpu_matrix.shape)
        * largest
    )
    strict_rank = int((singular_values > strict_tolerance).sum())
    approximate_tolerance = relative_tolerance * largest
    approximate_rank = int(
        (singular_values > approximate_tolerance).sum()
    )
    basis = vh[approximate_rank:].transpose(0, 1).contiguous()
    constrained = cpu_matrix @ basis
    if constrained.numel():
        constraint_max = float(constrained.abs().max())
        constraint_rms = float(constrained.square().mean().sqrt())
    else:
        constraint_max = 0.0
        constraint_rms = 0.0
    report = NullspaceReport(
        rows=cpu_matrix.shape[0],
        columns=cpu_matrix.shape[1],
        strict_tolerance=strict_tolerance,
        strict_rank=strict_rank,
        strict_nullity=cpu_matrix.shape[1] - strict_rank,
        approximate_relative_tolerance=relative_tolerance,
        approximate_tolerance=approximate_tolerance,
        approximate_rank=approximate_rank,
        approximate_nullity=cpu_matrix.shape[1] - approximate_rank,
        largest_singular_value=largest,
        smallest_singular_value=smallest,
        basis_constraint_max_abs=constraint_max,
        basis_constraint_rms=constraint_rms,
    )
    return basis, singular_values, report


def configure_loss(
    source: argparse.Namespace,
    option: str,
    args: argparse.Namespace,
) -> argparse.Namespace:
    values = vars(source).copy()
    values.update(
        {
            "option": option,
            "loss_variant": "literal" if option == "cf" else "lc_live",
            "state_mode": "feedback",
            "training_integrator": "rk4",
            "gate_gradient_mode": "live",
            "singular_eps": float(values.get("singular_eps", 0.1)),
            "singular_tau": float(values.get("singular_tau", 0.03)),
            "dot_eps": float(values.get("dot_eps", 0.1)),
            "dot_tau": float(values.get("dot_tau", 0.03)),
            "b_min": float(values.get("b_min", 1.0e-8)),
            "w0": args.w0,
            "w1": args.w1,
            "w2": args.w2,
            "w_lc": args.w_lc,
            "psi_scale": args.psi_scale,
            "dot_scale": args.dot_scale,
            "ddot_scale": args.ddot_scale,
            "B_scale": args.B_scale,
            "singular_loss_weight": (
                args.cf_candidate_weight if option == "cf" else 1.0
            ),
            "nonsingular_loss_weight": args.boundary_weight,
            "smooth_weight": 0.0,
            "smooth_second_weight": 0.0,
            "smooth_max_weight": 0.0,
            "full_gradient_weight": args.full_gradient_weight,
            "full_gradient_scale": args.full_gradient_scale,
            "full_gradient_residual": "projected",
            "full_gradient_projection_step": (
                args.full_gradient_projection_step
            ),
            "full_gradient_max_weight": args.full_gradient_max_weight,
            "full_gradient_max_tau": args.full_gradient_max_tau,
        }
    )
    return argparse.Namespace(**values)


@torch.no_grad()
def policy_rollout(
    model: NestedFeedbackTransformer,
    initial: torch.Tensor,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
    *,
    state_mode: str,
) -> PolicyRollout:
    states, controls, stage_states, _ = simulate_feedback_rk4_stagewise(
        model, initial, cfg, params, state_mode=state_mode
    )
    objectives = rk4_objective_per_sample(
        states, controls, stage_states, cfg, params
    )
    return PolicyRollout(
        states=states,
        controls=controls,
        stage_states=stage_states,
        objectives=objectives,
    )


@torch.no_grad()
def guard_rollout(
    model: NestedFeedbackTransformer,
    radii: tuple[float, ...],
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
    *,
    singular_refinement: int = 0,
    singular_interior_start: float = 1.5,
    singular_interior_end: float = 8.0,
) -> GuardRollout:
    initial = structured_initial_states(
        radii, cfg, next(model.parameters()).device, next(model.parameters()).dtype
    )
    rollout = policy_rollout(
        model, initial, cfg, params, state_mode="feedback"
    )
    resistant_mask = params["x"] >= 0.7
    resistant_burden = rollout.states[:, -1, resistant_mask].sum(dim=-1)
    common_r_sing: torch.Tensor | None = None
    if singular_refinement > 0:
        # This is the exact common evaluator behind Figure 3, not a
        # training-loss surrogate. It is diagnostic-only and outside autograd.
        from evaluate_table1_heldout_common import (  # noqa: PLC0415
            Problem,
            singular_residual,
        )

        problem = Problem()
        if (
            problem.intervals != cfg.n
            or problem.m != cfg.m
            or not math.isclose(problem.T, cfg.T)
        ):
            raise ValueError(
                "common singular evaluator does not match the checkpoint grid"
            )
        *_components, combined = singular_residual(
            initial.detach().cpu().numpy(),
            rollout.controls.detach().cpu().numpy(),
            problem,
            problem.vectors(),
            refinement=singular_refinement,
            interior_start=singular_interior_start,
            interior_end=singular_interior_end,
        )
        common_r_sing = torch.as_tensor(
            combined,
            device=rollout.controls.device,
            dtype=rollout.controls.dtype,
        )
    return GuardRollout(
        radii=radii,
        states=rollout.states,
        controls=rollout.controls,
        stage_states=rollout.stage_states,
        objectives=rollout.objectives,
        terminal_resistant_burden=resistant_burden,
        common_r_sing=common_r_sing,
    )


def guard_metrics(
    candidate: GuardRollout,
    reference: GuardRollout,
    *,
    physical_scale_factor: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if candidate.radii != reference.radii:
        raise ValueError("guard rollouts use different radius orderings")
    control_drift = (candidate.controls - reference.controls).abs()
    state_drift = (candidate.states - reference.states).abs()
    objective_drift = (
        (candidate.objectives - reference.objectives).abs()
        * physical_scale_factor
    )
    resistant_drift = (
        candidate.terminal_resistant_burden
        - reference.terminal_resistant_burden
    ).abs()
    r_sing_relative_drift: torch.Tensor | None = None
    if (candidate.common_r_sing is None) != (reference.common_r_sing is None):
        raise ValueError("guard rollouts disagree on common R_sing availability")
    if candidate.common_r_sing is not None:
        assert reference.common_r_sing is not None
        r_sing_relative_drift = (
            candidate.common_r_sing - reference.common_r_sing
        ).abs() / reference.common_r_sing.abs().clamp_min(1.0e-12)
    strict_radii = tuple(
        float(value) for value in getattr(args, "_strict_guard_radii", ())
    )
    display_thresholds = {
        "control_max_abs": (
            args.max_guard_control_drift
            if args.max_display_guard_control_drift is None
            else args.max_display_guard_control_drift
        ),
        "state_max_abs": (
            args.max_guard_state_drift
            if args.max_display_guard_state_drift is None
            else args.max_display_guard_state_drift
        ),
        "physical_objective_abs": (
            args.max_guard_physical_objective_drift
            if args.max_display_guard_physical_objective_drift is None
            else args.max_display_guard_physical_objective_drift
        ),
        "terminal_resistant_burden_abs": (
            args.max_guard_terminal_resistant_drift
            if args.max_display_guard_terminal_resistant_drift is None
            else args.max_display_guard_terminal_resistant_drift
        ),
    }
    max_display_r_sing = getattr(
        args, "max_display_guard_rsing_relative_drift", None
    )
    if max_display_r_sing is not None:
        display_thresholds["r_sing_relative_abs"] = (
            max_display_r_sing
        )
    strict_thresholds = {
        "control_max_abs": args.max_guard_control_drift,
        "state_max_abs": args.max_guard_state_drift,
        "physical_objective_abs": args.max_guard_physical_objective_drift,
        "terminal_resistant_burden_abs": (
            args.max_guard_terminal_resistant_drift
        ),
    }
    max_strict_r_sing = getattr(
        args, "max_guard_rsing_relative_drift", None
    )
    if max_strict_r_sing is not None:
        strict_thresholds["r_sing_relative_abs"] = (
            max_strict_r_sing
        )
    per_radius: dict[str, dict[str, Any]] = {}
    all_passed = True
    for index, radius in enumerate(candidate.radii):
        strict = any(
            math.isclose(radius, protected, rel_tol=0.0, abs_tol=1.0e-12)
            for protected in strict_radii
        )
        thresholds = strict_thresholds if strict else display_thresholds
        metrics = {
            "control_max_abs": float(control_drift[index].max().cpu()),
            "control_rms": float(
                control_drift[index].square().mean().sqrt().cpu()
            ),
            "state_max_abs": float(state_drift[index].max().cpu()),
            "physical_objective_abs": float(objective_drift[index].cpu()),
            "terminal_resistant_burden_abs": float(
                resistant_drift[index].cpu()
            ),
        }
        if r_sing_relative_drift is not None:
            assert candidate.common_r_sing is not None
            assert reference.common_r_sing is not None
            metrics.update(
                {
                    "common_r_sing": float(
                        candidate.common_r_sing[index].cpu()
                    ),
                    "reference_common_r_sing": float(
                        reference.common_r_sing[index].cpu()
                    ),
                    "r_sing_relative_abs": float(
                        r_sing_relative_drift[index].cpu()
                    ),
                }
            )
        failures = [
            name
            for name, threshold in thresholds.items()
            if metrics[name] > threshold
        ]
        radius_passed = not failures
        all_passed = all_passed and radius_passed
        per_radius[f"{radius:g}"] = {
            **metrics,
            "guard_tier": "strict" if strict else "display",
            "passed": radius_passed,
            "failed_metrics": failures,
            "thresholds": dict(thresholds),
        }
    maxima = {
        "control_max_abs": float(control_drift.max().cpu()),
        "control_rms": float(
            control_drift.square().mean().sqrt().cpu()
        ),
        "state_max_abs": float(state_drift.max().cpu()),
        "physical_objective_abs": float(objective_drift.max().cpu()),
        "terminal_resistant_burden_abs": float(
            resistant_drift.max().cpu()
        ),
    }
    if r_sing_relative_drift is not None:
        maxima["r_sing_relative_abs"] = float(
            r_sing_relative_drift.max().cpu()
        )
    return {
        "passed": all_passed,
        "maxima": maxima,
        "per_radius": per_radius,
        "strict_radii": strict_radii,
        "strict_thresholds": strict_thresholds,
        "display_thresholds": display_thresholds,
    }


def flatten_guard(metrics: dict[str, Any]) -> dict[str, Any]:
    maxima = metrics["maxima"]
    return {
        "guard_passed": bool(metrics["passed"]),
        "guard_control_max_abs": maxima["control_max_abs"],
        "guard_control_rms": maxima["control_rms"],
        "guard_state_max_abs": maxima["state_max_abs"],
        "guard_physical_objective_abs": maxima[
            "physical_objective_abs"
        ],
        "guard_terminal_resistant_burden_abs": maxima[
            "terminal_resistant_burden_abs"
        ],
        "guard_r_sing_relative_abs": maxima.get(
            "r_sing_relative_abs", math.nan
        ),
    }


def posthoc_test_metrics(
    candidate: PolicyRollout,
    source_feedback: PolicyRollout,
    frozen_time: PolicyRollout,
    *,
    physical_scale_factor: float,
) -> dict[str, Any]:
    candidate_j = candidate.objectives * physical_scale_factor
    source_j = source_feedback.objectives * physical_scale_factor
    time_j = frozen_time.objectives * physical_scale_factor

    def summary(value: torch.Tensor) -> dict[str, float]:
        detached = value.detach().cpu()
        return {
            "mean": float(detached.mean()),
            "sample_sd": float(
                detached.std(unbiased=True)
                if detached.numel() > 1
                else torch.zeros((), dtype=detached.dtype)
            ),
            "minimum": float(detached.min()),
            "maximum": float(detached.max()),
        }

    time_advantage = time_j - candidate_j
    source_advantage = source_j - candidate_j
    return {
        "candidate_physical_objective": summary(candidate_j),
        "source_feedback_physical_objective": summary(source_j),
        "frozen_time_physical_objective": summary(time_j),
        "candidate_advantage_over_frozen_time": {
            **summary(time_advantage),
            "positive_fraction": float(
                (time_advantage > 0.0)
                .to(time_advantage.dtype)
                .mean()
                .cpu()
            ),
        },
        "candidate_advantage_over_source_feedback": {
            **summary(source_advantage),
            "positive_fraction": float(
                (source_advantage > 0.0)
                .to(source_advantage.dtype)
                .mean()
                .cpu()
            ),
        },
        "used_for_training": False,
        "used_for_checkpoint_selection": False,
    }


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    if args.epochs < 0:
        raise ValueError("epochs must be nonnegative")
    if args.eval_every <= 0:
        raise ValueError("eval-every must be positive")
    if args.z_max_norm <= 0.0:
        raise ValueError("z-max-norm must be positive")
    if args.full_gradient_ramp_epochs < 0:
        raise ValueError("full-gradient-ramp-epochs must be nonnegative")
    if min(
        args.full_gradient_weight,
        args.full_gradient_max_weight,
        args.cf_candidate_weight,
        args.boundary_weight,
        args.w0,
        args.w1,
        args.w2,
        args.w_lc,
    ) < 0.0:
        raise ValueError("loss weights must be nonnegative")

    basis_radii = parse_radii(args.basis_protected_radii)
    guard_radii = parse_radii(args.guard_radii)
    strict_guard_radii = parse_radii(args.strict_guard_radii)
    missing_strict = [
        radius
        for radius in strict_guard_radii
        if not any(
            math.isclose(radius, guarded, rel_tol=0.0, abs_tol=1.0e-12)
            for guarded in guard_radii
        )
    ]
    if missing_strict:
        raise ValueError(
            "strict guard radii must be included in --guard-radii: "
            f"{missing_strict}"
        )
    args._strict_guard_radii = strict_guard_radii
    singular_guard_enabled = (
        args.max_guard_rsing_relative_drift is not None
        or args.max_display_guard_rsing_relative_drift is not None
    )
    if singular_guard_enabled and args.guard_singular_refinement <= 0:
        raise ValueError(
            "guard-singular-refinement must be positive when an R_sing guard "
            "is enabled"
        )
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = torch.float64
    torch.manual_seed(args.optimizer_seed)

    checkpoint_path = args.checkpoint.expanduser().resolve()
    source_checkpoint, model, cfg, source_args = (
        load_locked_feedback_checkpoint(checkpoint_path)
    )
    model.to(device=device, dtype=dtype)
    params = build_params(cfg, device, dtype)
    model.set_feature_vectors(params["r"], params["phi"])
    model.eval()
    option = (
        str(getattr(source_args, "option", "der"))
        if args.option == "auto"
        else args.option
    )
    if option not in {"cf", "der"}:
        raise ValueError("could not determine whether the checkpoint is CF or DER")
    loss_args = configure_loss(source_args, option, args)

    all_radii = tuple(
        dict.fromkeys((*basis_radii, *guard_radii))
    )
    singular_guard_kwargs = {
        "singular_refinement": (
            args.guard_singular_refinement if singular_guard_enabled else 0
        ),
        "singular_interior_start": args.guard_singular_interior_start,
        "singular_interior_end": args.guard_singular_interior_end,
    }
    source_all = guard_rollout(
        model, all_radii, cfg, params, **singular_guard_kwargs
    )
    radius_to_index = {
        radius: index for index, radius in enumerate(all_radii)
    }

    def select_rollout(radii: tuple[float, ...]) -> GuardRollout:
        indices = torch.tensor(
            [radius_to_index[radius] for radius in radii],
            device=device,
            dtype=torch.long,
        )
        return GuardRollout(
            radii=radii,
            states=source_all.states.index_select(0, indices),
            controls=source_all.controls.index_select(0, indices),
            stage_states=source_all.stage_states.index_select(0, indices),
            objectives=source_all.objectives.index_select(0, indices),
            terminal_resistant_burden=(
                source_all.terminal_resistant_burden.index_select(
                    0, indices
                )
            ),
            common_r_sing=(
                None
                if source_all.common_r_sing is None
                else source_all.common_r_sing.index_select(0, indices)
            ),
        )

    source_basis = select_rollout(basis_radii)
    source_guard = select_rollout(guard_radii)
    with torch.no_grad():
        basis_constraints = anchor_feature_matrix(
            model, source_basis.states, cfg
        )
        guard_constraints = anchor_feature_matrix(
            model, source_guard.states, cfg
        )
    basis_cpu, basis_singular_values, basis_report = (
        approximate_right_nullspace(
            basis_constraints, args.svd_relative_tolerance
        )
    )
    _, guard_singular_values, guard_report = (
        approximate_right_nullspace(
            guard_constraints, args.svd_relative_tolerance
        )
    )
    basis = basis_cpu.to(device=device, dtype=dtype)

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    inspection = {
        "checkpoint": str(checkpoint_path),
        "option": option,
        "basis_protected_radii": basis_radii,
        "guard_radii": guard_radii,
        "strict_guard_radii": strict_guard_radii,
        "basis_constraint_set": asdict(basis_report),
        "full_guard_constraint_set": asdict(guard_report),
    }
    (out_dir / "nullspace_inspection.json").write_text(
        json.dumps(inspection, indent=2) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "basis": basis_cpu,
            "basis_singular_values": basis_singular_values,
            "guard_singular_values": guard_singular_values,
            "basis_radii": basis_radii,
            "guard_radii": guard_radii,
            "relative_tolerance": args.svd_relative_tolerance,
        },
        out_dir / "svd_nullspace.pt",
    )
    print(
        "SVD nullspace: "
        f"basis radii={basis_radii} -> "
        f"{basis_report.approximate_nullity} dimensions; "
        f"full guard radii={guard_radii} -> "
        f"{guard_report.approximate_nullity} dimensions",
        flush=True,
    )
    if args.inspect_only:
        print(json.dumps(inspection, indent=2), flush=True)
        return
    if basis.shape[1] == 0:
        raise RuntimeError(
            "the requested approximate nullspace is empty; inspect the "
            "reported singular values or increase --svd-relative-tolerance"
        )

    head_index, source_head = final_state_linear(model)
    head_key = f"state_branch.{head_index}.weight"
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapter = SVDNullLinear(source_head, basis)
    model.state_branch[head_index] = adapter
    optimizer = torch.optim.AdamW(
        [adapter.z],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_states = antithetic_initial_states(
        args.train_random_states,
        args.train_seed,
        args.random_radius,
        cfg,
        device,
        dtype,
        antithetic=args.antithetic_states,
    )
    validation_states = antithetic_initial_states(
        args.validation_random_states,
        args.validation_seed,
        args.random_radius,
        cfg,
        device,
        dtype,
        antithetic=args.antithetic_states,
    )
    test_states = antithetic_initial_states(
        args.test_random_states,
        args.test_seed,
        args.random_radius,
        cfg,
        device,
        dtype,
        antithetic=False,
    )
    physical_scale_factor = (
        args.physical_scale_factor
        if args.physical_scale_factor > 0.0
        else 1.0 / cfg.alpha
    )
    with torch.no_grad():
        source_test_rollout = policy_rollout(
            model, test_states, cfg, params, state_mode="feedback"
        )
        frozen_time_test_rollout = policy_rollout(
            model, test_states, cfg, params, state_mode="w_zero"
        )
    history: list[dict[str, Any]] = []
    best_value = math.inf
    best_epoch = 0
    best_z = adapter.z.detach().clone()
    best_guard: dict[str, Any] | None = None
    best_validation_metrics: dict[str, float] | None = None

    def evaluate(epoch: int) -> float:
        nonlocal best_value, best_epoch, best_z
        nonlocal best_guard, best_validation_metrics
        model.eval()
        loss_args._current_full_gradient_weight = (
            args.full_gradient_weight
        )
        loss_args._current_full_gradient_max_weight = (
            args.full_gradient_max_weight
        )
        with torch.no_grad():
            train_pack = section5_loss(
                model, train_states, cfg, params, loss_args
            )
            validation_pack = section5_loss(
                model, validation_states, cfg, params, loss_args
            )
            candidate_guard = guard_rollout(
                model,
                guard_radii,
                cfg,
                params,
                **singular_guard_kwargs,
            )
        train_metrics = scalar_metrics(train_pack, cfg, loss_args)
        validation_metrics = scalar_metrics(
            validation_pack, cfg, loss_args
        )
        compatibility = guard_metrics(
            candidate_guard,
            source_guard,
            physical_scale_factor=physical_scale_factor,
            args=args,
        )
        eligible = bool(compatibility["passed"]) or not args.require_guard_pass
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_opt_gap": train_metrics["opt_gap"],
            "train_projected_kkt_rms": train_metrics[
                "full_gradient_residual_rms"
            ],
            "train_projected_kkt_linf": train_metrics[
                "full_gradient_residual_linf"
            ],
            "validation_loss": validation_metrics["loss"],
            "validation_opt_gap": validation_metrics["opt_gap"],
            "validation_projected_kkt_rms": validation_metrics[
                "full_gradient_residual_rms"
            ],
            "validation_projected_kkt_linf": validation_metrics[
                "full_gradient_residual_linf"
            ],
            "z_norm": float(adapter.z.norm().detach().cpu()),
            "eligible": eligible,
            **flatten_guard(compatibility),
        }
        history.append(row)
        # Compatibility is only an eligibility check.  Among eligible
        # candidates, ranking uses validation PMP/KKT loss and nothing else.
        if eligible and row["validation_loss"] < best_value:
            best_value = row["validation_loss"]
            best_epoch = epoch
            best_z = adapter.z.detach().clone()
            best_guard = copy.deepcopy(compatibility)
            best_validation_metrics = copy.deepcopy(validation_metrics)
        print(
            f"[{epoch:04d}] train={row['train_loss']:.7g} "
            f"val={row['validation_loss']:.7g} "
            f"Ginf={row['validation_projected_kkt_linf']:.3g} "
            f"|z|={row['z_norm']:.3g} "
            f"anchor_du={row['guard_control_max_abs']:.3g} "
            f"guard={'pass' if compatibility['passed'] else 'fail'}",
            flush=True,
        )
        return row["validation_loss"]

    evaluate(0)
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        ramp = (
            1.0
            if args.full_gradient_ramp_epochs == 0
            else min(1.0, epoch / args.full_gradient_ramp_epochs)
        )
        loss_args._current_full_gradient_weight = (
            args.full_gradient_weight * ramp
        )
        loss_args._current_full_gradient_max_weight = (
            args.full_gradient_max_weight * ramp
        )
        pack = section5_loss(
            model, train_states, cfg, params, loss_args
        )
        if float(pack["raw_minima"].min().detach().cpu()) <= 0.0:
            raise RuntimeError(
                "positivity clamp became active; the discrete adjoint is no "
                "longer valid for this candidate"
            )
        pack["loss"].backward()
        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_([adapter.z], args.grad_clip)
        optimizer.step()
        with torch.no_grad():
            norm = adapter.z.norm()
            if norm > args.z_max_norm:
                adapter.z.mul_(args.z_max_norm / norm)
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            evaluate(epoch)

    if best_guard is None or best_validation_metrics is None:
        raise RuntimeError("no candidate passed the requested compatibility guard")
    with torch.no_grad():
        adapter.z.copy_(best_z)
    model.eval()
    with torch.no_grad():
        adapter_guard = guard_rollout(
            model,
            guard_radii,
            cfg,
            params,
            **singular_guard_kwargs,
        )
        adapter_validation = section5_loss(
            model, validation_states, cfg, params, loss_args
        )
    folded_weight = adapter.effective_weight().detach().clone()
    delta_weight = folded_weight - adapter.base_weight
    model.state_branch[head_index] = adapter.folded_linear()
    model.eval()
    with torch.no_grad():
        folded_guard = guard_rollout(
            model,
            guard_radii,
            cfg,
            params,
            **singular_guard_kwargs,
        )
        folded_validation = section5_loss(
            model, validation_states, cfg, params, loss_args
        )
        folded_test_rollout = policy_rollout(
            model, test_states, cfg, params, state_mode="feedback"
        )
    folding_max_abs = max(
        float(
            (folded_guard.controls - adapter_guard.controls)
            .abs()
            .max()
            .cpu()
        ),
        float(
            (folded_validation["controls"] - adapter_validation["controls"])
            .abs()
            .max()
            .cpu()
        ),
    )
    if folding_max_abs > args.max_folding_error:
        raise RuntimeError(
            "native final-layer folding changed policy outputs by "
            f"{folding_max_abs:.3g}"
        )

    final_guard = guard_metrics(
        folded_guard,
        source_guard,
        physical_scale_factor=physical_scale_factor,
        args=args,
    )
    if args.require_guard_pass and not final_guard["passed"]:
        raise RuntimeError("the folded checkpoint failed the final hard guard")
    native_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    if head_key not in native_state:
        raise RuntimeError(f"native state dict is missing {head_key}")
    output_args = vars(source_args).copy()
    output_args.update(
        {
            "option": option,
            "state_mode": "feedback",
            "training_integrator": "rk4",
            "action_temperature": model.action_temperature,
            "action_scale": model.action_scale,
            "action_parameterization": model.action_parameterization,
            "action_offset": model.action_offset,
            "svd_null_refinement": True,
            "svd_null_dimension": int(basis.shape[1]),
        }
    )
    payload = dict(source_checkpoint)
    payload.update(
        {
            "model_state": native_state,
            "args": output_args,
            "problem": cfg.__dict__,
            "best_validation_loss": best_value,
            "best_epoch": best_epoch,
            "selection_metric": (
                "validation_trajectory_PMP_optimality_gap_plus_projected_KKT"
            ),
            # The complete time branch is already present in model_state.
            # Clearing this stale absolute dependency keeps the exported
            # native checkpoint portable across the server and local host.
            "time_checkpoint": None,
            "nominal_reference": model.nominal_reference.detach().cpu(),
            "initialization_checkpoint": str(checkpoint_path),
            "refinement": {
                "optimizer": "AdamW",
                "train_scope": "existing_final_state_linear_SVD_nullspace",
                "parameterization": "delta_w=V_z_folded_into_final_linear",
                "nullspace_dimension": int(basis.shape[1]),
                "basis_protected_radii": basis_radii,
                "hard_guard_radii": guard_radii,
                "svd_relative_tolerance": args.svd_relative_tolerance,
                "random_state_radius": args.random_radius,
                "train_random_states": args.train_random_states,
                "validation_random_states": args.validation_random_states,
                "train_seed": args.train_seed,
                "validation_seed": args.validation_seed,
                "untouched_test_random_states": args.test_random_states,
                "untouched_test_seed": args.test_seed,
                "antithetic_states": args.antithetic_states,
                "PMP_optimality_gap_used": True,
                "projected_KKT_used": True,
                "physical_objective_used_as_loss": False,
                "physical_objective_used_for_ranking": False,
                "physical_objective_used_only_as_compatibility_guard": True,
                "native_architecture_export": True,
                "source_time_checkpoint": source_checkpoint.get(
                    "time_checkpoint"
                ),
            },
        }
    )
    checkpoint_output = out_dir / "best_feedback_section5.pt"
    torch.save(payload, checkpoint_output)
    write_history(out_dir / "history.csv", history)
    summary = {
        "checkpoint": str(checkpoint_output),
        "initialization_checkpoint": str(checkpoint_path),
        "option": option,
        "best_epoch": best_epoch,
        "best_validation_loss": best_value,
        "best_validation_metrics": best_validation_metrics,
        "basis_protected_radii": basis_radii,
        "guard_radii": guard_radii,
        "basis_constraint_set": asdict(basis_report),
        "full_guard_constraint_set": asdict(guard_report),
        "selected_z": best_z.detach().cpu().tolist(),
        "selected_z_norm": float(best_z.norm().cpu()),
        "delta_weight_norm": float(delta_weight.norm().cpu()),
        "final_guard": final_guard,
        "untouched_test": {
            "radius": args.random_radius,
            "count": args.test_random_states,
            "seed": args.test_seed,
            **posthoc_test_metrics(
                folded_test_rollout,
                source_test_rollout,
                frozen_time_test_rollout,
                physical_scale_factor=physical_scale_factor,
            ),
        },
        "folding_max_abs": folding_max_abs,
        "selection_contract": {
            "ranking_metric": "validation PMP/KKT loss",
            "objective_in_training_loss": False,
            "objective_in_ranking_metric": False,
            "objective_as_hard_compatibility_guard": True,
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "z": best_z.detach().cpu(),
            "basis": basis_cpu,
            "delta_weight": delta_weight.detach().cpu(),
            "folded_weight": folded_weight.detach().cpu(),
            "final_weight_key": head_key,
        },
        out_dir / "selected_null_update.pt",
    )
    print(f"Wrote {checkpoint_output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--option", choices=["auto", "cf", "der"], default="auto"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--basis-protected-radii",
        default="0,0.20",
        help=(
            "structured radii whose trajectory feature rows define the SVD "
            "nullspace; radius 0 is the nominal state"
        ),
    )
    parser.add_argument(
        "--guard-radii",
        default="0,0.10,0.20,0.40,0.60",
        help=(
            "structured radii that every selected checkpoint must preserve"
        ),
    )
    parser.add_argument(
        "--strict-guard-radii",
        default="0,0.20",
        help=(
            "guard radii tied to displayed table values; these use the "
            "--max-guard-* thresholds. Other guard radii use the optional "
            "--max-display-guard-* thresholds"
        ),
    )
    parser.add_argument(
        "--svd-relative-tolerance", type=float, default=1.0e-8
    )
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--random-radius", type=float, default=0.20)
    parser.add_argument("--train-random-states", type=int, default=64)
    parser.add_argument(
        "--validation-random-states", type=int, default=64
    )
    parser.add_argument("--train-seed", type=int, default=20260730)
    parser.add_argument("--validation-seed", type=int, default=20260731)
    parser.add_argument("--test-random-states", type=int, default=128)
    parser.add_argument("--test-seed", type=int, default=20260801)
    parser.add_argument(
        "--antithetic-states",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--optimizer-seed", type=int, default=20260730)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--z-max-norm", type=float, default=0.05)
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
    parser.add_argument(
        "--full-gradient-weight", type=float, default=50.0
    )
    parser.add_argument(
        "--full-gradient-max-weight", type=float, default=5.0
    )
    parser.add_argument(
        "--full-gradient-ramp-epochs", type=int, default=20
    )
    parser.add_argument(
        "--full-gradient-projection-step", type=float, default=1.0
    )
    parser.add_argument(
        "--full-gradient-scale", type=float, default=0.0
    )
    parser.add_argument(
        "--full-gradient-max-tau", type=float, default=0.1
    )
    parser.add_argument(
        "--physical-scale-factor",
        type=float,
        default=0.0,
        help="0 infers the paper scale as 1/alpha",
    )
    parser.add_argument(
        "--max-guard-control-drift", type=float, default=1.0e-8
    )
    parser.add_argument(
        "--max-guard-state-drift", type=float, default=1.0e-6
    )
    parser.add_argument(
        "--max-guard-physical-objective-drift",
        type=float,
        default=5.0e-4,
    )
    parser.add_argument(
        "--max-guard-terminal-resistant-drift",
        type=float,
        default=1.0e-6,
    )
    parser.add_argument(
        "--max-display-guard-control-drift",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-display-guard-state-drift",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-display-guard-physical-objective-drift",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-display-guard-terminal-resistant-drift",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-guard-rsing-relative-drift",
        type=float,
        default=None,
        help="optional strict-radius relative drift limit for common R_sing",
    )
    parser.add_argument(
        "--max-display-guard-rsing-relative-drift",
        type=float,
        default=None,
        help=(
            "optional Figure-3-radius relative drift limit for common R_sing"
        ),
    )
    parser.add_argument(
        "--guard-singular-refinement",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--guard-singular-interior-start",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--guard-singular-interior-end",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--require-guard-pass",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--max-folding-error", type=float, default=1.0e-12
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
