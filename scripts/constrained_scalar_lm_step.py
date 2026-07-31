#!/usr/bin/env python3
"""Small active-set solver for scalar-PMP constrained LM steps.

This module is intentionally independent of the feedback training scripts.  It
solves the local subproblem

    min_d  1/2 ||r + J d||^2 + 1/2 lambda d^T D d
    s.t.   G d <= b,

where ``r`` and ``J`` are the random-state continuous scalar-PMP residual and
Jacobian, while each row of ``G`` is the gradient of a protected scalar
``L_opt`` (for example nominal and structured-r=0.20).  Thus neither the
physical objective, a direct-control target, nor a control-trust term enters
the step.

There are normally only two protected losses, so enumerating their possible
active sets is simpler and more reliable than adding another optimization
dependency.  A caller must still evaluate the nonlinear losses and backtrack
the returned step until the hard protection limits are satisfied.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Callable

import torch


@dataclass(frozen=True)
class ConstrainedLMStep:
    step: torch.Tensor
    active_constraints: tuple[int, ...]
    quadratic_value: float
    linearized_feasible: bool
    step_norm: float


@dataclass(frozen=True)
class NonlinearBacktrackingResult:
    point: torch.Tensor
    objective: float
    accepted: bool
    scale: float
    backtracking_step: int


def nonlinear_feasible_backtracking(
    current_point: torch.Tensor,
    step: torch.Tensor,
    current_objective: float,
    evaluate_trial: Callable[[torch.Tensor], tuple[float, bool]],
    *,
    maximum_steps: int,
    factor: float,
    acceptance_tolerance: float,
) -> NonlinearBacktrackingResult:
    """Backtrack until both the true objective and nonlinear gates pass."""

    if current_point.shape != step.shape:
        raise ValueError("current point and step must have matching shapes")
    if maximum_steps <= 0 or not 0.0 < factor < 1.0:
        raise ValueError("invalid nonlinear backtracking configuration")
    if acceptance_tolerance < 0.0:
        raise ValueError("acceptance_tolerance must be nonnegative")
    for backtracking_step in range(maximum_steps):
        scale = factor**backtracking_step
        trial_point = current_point + scale * step
        trial_objective, feasible = evaluate_trial(trial_point)
        if (
            feasible
            and trial_objective + acceptance_tolerance
            < current_objective
        ):
            return NonlinearBacktrackingResult(
                point=trial_point.detach(),
                objective=float(trial_objective),
                accepted=True,
                scale=float(scale),
                backtracking_step=backtracking_step,
            )
    return NonlinearBacktrackingResult(
        point=current_point.detach().clone(),
        objective=float(current_objective),
        accepted=False,
        scale=0.0,
        backtracking_step=-1,
    )


def forward_jacobian_columns(
    function: Callable[[torch.Tensor], torch.Tensor],
    point: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate a vector function and its Jacobian one JVP column at a time.

    Column-wise JVP avoids materializing a vectorized batch of full RK4
    trajectories.  It is therefore the safer default for the low-dimensional
    (typically 8--35 parameter) near-null last-layer problem.
    """

    if point.ndim != 1:
        raise ValueError("the LM coordinate must be one-dimensional")
    value = function(point)
    if value.ndim != 1:
        raise ValueError("the residual function must return a vector")
    identity = torch.eye(
        point.numel(),
        device=point.device,
        dtype=point.dtype,
    )
    columns = []
    for direction in identity:
        _, tangent = torch.func.jvp(
            function,
            (point,),
            (direction,),
        )
        columns.append(tangent)
    return value, torch.stack(columns, dim=1)


def der_residual_vector_from_pack(
    pack: dict[str, torch.Tensor],
    *,
    weights: tuple[float, float, float] = (1.0, 1.0, 4.0),
) -> torch.Tensor:
    """Flatten the residual whose squared norm is the DER scalar loss.

    This matches ``scalar_pack(..., option="der", residual_p=2)`` including
    its time mask and optional per-state weights.
    """

    time_mask = pack["weighted_mask"]
    values = pack["quantities"]["psi"]
    if time_mask.shape[0] != 1:
        raise ValueError("expected a shared time mask with leading size one")
    state_weights = pack.get("state_weights")
    if state_weights is None:
        state_weights = torch.ones(
            values.shape[0],
            device=values.device,
            dtype=values.dtype,
        )
    if state_weights.shape != (values.shape[0],):
        raise ValueError("state_weights do not match the residual batch")
    expanded_mask = time_mask.expand_as(values)
    state_shape = (values.shape[0], *([1] * (values.ndim - 1)))
    loss_mask = expanded_mask * state_weights.view(state_shape)
    denominator = (
        time_mask.sum() * state_weights.sum()
    ).clamp_min(torch.finfo(values.dtype).eps)
    square_root_weight = (loss_mask / denominator).sqrt()
    blocks = []
    for weight, name in zip(
        weights,
        ("psi", "dot_psi", "ddot_psi"),
    ):
        if weight < 0.0:
            raise ValueError("DER residual weights must be nonnegative")
        if weight:
            blocks.append(
                math.sqrt(float(weight))
                * square_root_weight
                * pack["quantities"][name]
            )
    if not blocks:
        raise ValueError("at least one DER residual weight must be positive")
    return torch.cat([block.reshape(-1) for block in blocks])


def _solve_positive_definite(
    matrix: torch.Tensor,
    right_hand_side: torch.Tensor,
) -> torch.Tensor:
    """Solve a symmetric positive-definite system via Cholesky."""

    factor = torch.linalg.cholesky(matrix)
    if right_hand_side.ndim == 1:
        return torch.cholesky_solve(
            right_hand_side.unsqueeze(-1),
            factor,
        ).squeeze(-1)
    return torch.cholesky_solve(right_hand_side, factor)


def constrained_lm_step(
    residual: torch.Tensor,
    jacobian: torch.Tensor,
    protected_gradients: torch.Tensor,
    protected_budgets: torch.Tensor,
    *,
    damping: float,
    maximum_step_norm: float,
    feasibility_tolerance: float = 1.0e-12,
    multiplier_tolerance: float = 1.0e-12,
) -> ConstrainedLMStep:
    """Return the best linearized feasible damped Gauss--Newton step.

    ``protected_budgets[i]`` is ``limit_i - current_loss_i`` and must be
    nonnegative.  The caller should form each protected gradient from the same
    continuous-policy scalar ``L_opt`` used by the experiment.
    """

    if residual.ndim != 1:
        raise ValueError("residual must be one-dimensional")
    if jacobian.ndim != 2 or jacobian.shape[0] != residual.numel():
        raise ValueError("jacobian must have shape (residuals, parameters)")
    parameter_count = jacobian.shape[1]
    if (
        protected_gradients.ndim != 2
        or protected_gradients.shape[1] != parameter_count
    ):
        raise ValueError(
            "protected_gradients must have shape (constraints, parameters)"
        )
    if protected_budgets.shape != (protected_gradients.shape[0],):
        raise ValueError("one protected budget is required per constraint")
    if not bool(
        torch.isfinite(residual).all()
        and torch.isfinite(jacobian).all()
        and torch.isfinite(protected_gradients).all()
        and torch.isfinite(protected_budgets).all()
    ):
        raise ValueError("all inputs must be finite")
    if bool((protected_budgets < 0.0).any()):
        raise ValueError(
            "the current iterate must satisfy every protected-loss limit"
        )
    if damping <= 0.0 or maximum_step_norm <= 0.0:
        raise ValueError("damping and maximum_step_norm must be positive")

    normal = jacobian.T @ jacobian
    eps = torch.finfo(jacobian.dtype).eps
    diagonal = normal.diagonal().clamp_min(eps)
    system = normal + float(damping) * torch.diag(diagonal)
    return constrained_quadratic_step(
        jacobian.T @ residual,
        system,
        protected_gradients,
        protected_budgets,
        maximum_step_norm=maximum_step_norm,
        feasibility_tolerance=feasibility_tolerance,
        multiplier_tolerance=multiplier_tolerance,
    )


def constrained_quadratic_step(
    gradient: torch.Tensor,
    positive_definite_curvature: torch.Tensor,
    protected_gradients: torch.Tensor,
    protected_budgets: torch.Tensor,
    *,
    maximum_step_norm: float,
    feasibility_tolerance: float = 1.0e-12,
    multiplier_tolerance: float = 1.0e-12,
) -> ConstrainedLMStep:
    """Solve a small positive-definite QP with linear protection bounds."""

    if gradient.ndim != 1:
        raise ValueError("gradient must be one-dimensional")
    parameter_count = gradient.numel()
    if positive_definite_curvature.shape != (
        parameter_count,
        parameter_count,
    ):
        raise ValueError("curvature must be square and match the gradient")
    if (
        protected_gradients.ndim != 2
        or protected_gradients.shape[1] != parameter_count
    ):
        raise ValueError(
            "protected_gradients must have shape (constraints, parameters)"
        )
    if protected_budgets.shape != (protected_gradients.shape[0],):
        raise ValueError("one protected budget is required per constraint")
    if not bool(
        torch.isfinite(gradient).all()
        and torch.isfinite(positive_definite_curvature).all()
        and torch.isfinite(protected_gradients).all()
        and torch.isfinite(protected_budgets).all()
    ):
        raise ValueError("all inputs must be finite")
    if bool((protected_budgets < 0.0).any()):
        raise ValueError(
            "the current iterate must satisfy every protected-loss limit"
        )
    if maximum_step_norm <= 0.0:
        raise ValueError("maximum_step_norm must be positive")

    system = 0.5 * (
        positive_definite_curvature
        + positive_definite_curvature.T
    )
    eps = torch.finfo(system.dtype).eps
    inverse_gradient = _solve_positive_definite(system, gradient)

    constraint_count = protected_gradients.shape[0]
    candidates: list[tuple[float, tuple[int, ...], torch.Tensor]] = []
    for active_count in range(constraint_count + 1):
        for active in combinations(range(constraint_count), active_count):
            if active:
                index = torch.as_tensor(
                    active,
                    device=gradient.device,
                    dtype=torch.long,
                )
                active_matrix = protected_gradients[index]
                inverse_active_t = _solve_positive_definite(
                    system,
                    active_matrix.T,
                )
                schur = active_matrix @ inverse_active_t
                # Near-dependent protection gradients are harmless after a
                # tiny symmetric regularization of the small Schur system.
                schur_scale = schur.diagonal().abs().max().clamp_min(eps)
                schur = schur + (
                    16.0 * eps * schur_scale
                ) * torch.eye(
                    len(active),
                    device=schur.device,
                    dtype=schur.dtype,
                )
                right_hand_side = -(
                    protected_budgets[index]
                    + active_matrix @ inverse_gradient
                )
                try:
                    multipliers = torch.linalg.solve(
                        schur,
                        right_hand_side,
                    )
                except torch.linalg.LinAlgError:
                    continue
                if bool((multipliers < -multiplier_tolerance).any()):
                    continue
                step = -inverse_gradient - inverse_active_t @ multipliers
            else:
                step = -inverse_gradient

            if bool(
                (
                    protected_gradients @ step
                    > protected_budgets + feasibility_tolerance
                ).any()
            ):
                continue
            norm = step.norm()
            if float(norm) > maximum_step_norm:
                step = step * (float(maximum_step_norm) / float(norm))
            value = (
                0.5 * step.dot(system @ step) + gradient.dot(step)
            )
            candidates.append((float(value), active, step))

    if not candidates:
        zero = torch.zeros(
            parameter_count,
            device=gradient.device,
            dtype=gradient.dtype,
        )
        return ConstrainedLMStep(
            step=zero,
            active_constraints=(),
            quadratic_value=0.0,
            linearized_feasible=False,
            step_norm=0.0,
        )
    value, active, step = min(candidates, key=lambda item: item[0])
    return ConstrainedLMStep(
        step=step,
        active_constraints=active,
        quadratic_value=value,
        linearized_feasible=True,
        step_norm=float(step.norm()),
    )
