"""Core discrete-time Neural-PMP algorithm.

This follows Algorithm 1 and Eqs. (32)--(33) of arXiv:2212.14566:

1. roll the learned (or known) one-step dynamics forward;
2. initialize the terminal costate with the terminal-cost gradient;
3. propagate costates backward through the discrete Hamiltonian;
4. compute the *raw* Hamiltonian action gradient;
5. take the gradient step and only then project the action.

In particular, the Hamiltonian gradient is never clipped to the action set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor, nn

StageCost = Callable[[Tensor, Tensor, int], Tensor]
TerminalCost = Callable[[Tensor], Tensor]


@dataclass
class NeuralPMPResult:
    control: Tensor
    states: Tensor
    costates: Tensor
    raw_gradient: Tensor
    history: list[dict[str, float | int]]
    best_iteration: int
    stop_reason: str


def _one_step(model: nn.Module | Callable[[Tensor, Tensor], Tensor], x: Tensor, u: Tensor) -> Tensor:
    """Call either a two-argument dynamics map or a concatenated-input MLP."""
    try:
        value = model(x, u)  # type: ignore[misc]
    except TypeError:
        value = model(torch.cat((x, u), dim=-1))  # type: ignore[misc]
    return value.reshape_as(x)


def rollout(
    dynamics: nn.Module | Callable[[Tensor, Tensor], Tensor],
    initial_state: Tensor,
    control: Tensor,
) -> Tensor:
    """Roll out a discrete one-step map without altering its predictions."""
    states = [initial_state]
    state = initial_state
    for k in range(control.shape[0]):
        state = _one_step(dynamics, state, control[k])
        states.append(state)
    return torch.stack(states)


def objective(
    states: Tensor,
    control: Tensor,
    stage_cost: StageCost,
    terminal_cost: TerminalCost,
) -> Tensor:
    value = terminal_cost(states[-1])
    for k in range(control.shape[0]):
        value = value + stage_cost(states[k], control[k], k)
    return value


def discrete_adjoint_gradient(
    dynamics: nn.Module | Callable[[Tensor, Tensor], Tensor],
    initial_state: Tensor,
    control: Tensor,
    stage_cost: StageCost,
    terminal_cost: TerminalCost,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return states, costates, and exact discrete Hamiltonian gradients.

    The returned ``costates[k]`` is dJ/dx_k and ``raw_gradient[k]`` is
    dH_k/du_k = dJ/du_k for the current trajectory.  Each local derivative is
    evaluated by autograd, while the backward recursion itself is explicit and
    matches Algorithm 1.
    """
    if control.ndim != 2:
        raise ValueError("control must have shape [horizon, action_dim]")

    with torch.no_grad():
        states_detached = rollout(dynamics, initial_state, control).detach()

    terminal_state = states_detached[-1].clone().requires_grad_(True)
    terminal_value = terminal_cost(terminal_state)
    (lambda_next,) = torch.autograd.grad(terminal_value, terminal_state)

    horizon = int(control.shape[0])
    reverse_costates: list[Tensor] = [lambda_next.detach()]
    reverse_gradients: list[Tensor] = []

    for k in range(horizon - 1, -1, -1):
        xk = states_detached[k].clone().requires_grad_(True)
        uk = control[k].detach().clone().requires_grad_(True)
        next_state = _one_step(dynamics, xk, uk)
        hamiltonian = stage_cost(xk, uk, k) + torch.dot(lambda_next.detach(), next_state)
        lambda_k, gradient_k = torch.autograd.grad(hamiltonian, (xk, uk))
        reverse_costates.append(lambda_k.detach())
        reverse_gradients.append(gradient_k.detach())
        lambda_next = lambda_k.detach()

    costates = torch.stack(list(reversed(reverse_costates)))
    raw_gradient = torch.stack(list(reversed(reverse_gradients)))
    return states_detached, costates, raw_gradient


def full_autodiff_gradient(
    dynamics: nn.Module | Callable[[Tensor, Tensor], Tensor],
    initial_state: Tensor,
    control: Tensor,
    stage_cost: StageCost,
    terminal_cost: TerminalCost,
) -> tuple[Tensor, Tensor, Tensor]:
    """Reference gradient obtained by differentiating the full rollout graph."""
    differentiable_control = control.detach().clone().requires_grad_(True)
    states = rollout(dynamics, initial_state, differentiable_control)
    value = objective(states, differentiable_control, stage_cost, terminal_cost)
    (gradient,) = torch.autograd.grad(value, differentiable_control)
    return value.detach(), states.detach(), gradient.detach()


def project_action(candidate: Tensor, lower: Tensor | float, upper: Tensor | float) -> Tensor:
    """Euclidean projection onto a box action set (paper Eq. 33)."""
    lower_tensor = torch.as_tensor(lower, dtype=candidate.dtype, device=candidate.device)
    upper_tensor = torch.as_tensor(upper, dtype=candidate.dtype, device=candidate.device)
    if bool(torch.any(lower_tensor > upper_tensor)):
        raise ValueError("action lower bound exceeds upper bound")
    return torch.maximum(torch.minimum(candidate, upper_tensor), lower_tensor)


def projected_gradient_residual(
    control: Tensor,
    raw_gradient: Tensor,
    lower: Tensor | float,
    upper: Tensor | float,
) -> Tensor:
    return torch.max(torch.abs(control - project_action(control - raw_gradient, lower, upper)))


def _validation_score(
    dynamics: nn.Module | Callable[[Tensor, Tensor], Tensor],
    initial_states: Sequence[Tensor],
    control: Tensor,
    stage_cost: StageCost,
    terminal_cost: TerminalCost,
) -> float:
    with torch.no_grad():
        values = [
            objective(rollout(dynamics, state, control), control, stage_cost, terminal_cost)
            for state in initial_states
        ]
    return float(torch.stack(values).mean())


def solve_neural_pmp(
    *,
    dynamics: nn.Module | Callable[[Tensor, Tensor], Tensor],
    initial_state: Tensor,
    initial_control: Tensor,
    stage_cost: StageCost,
    terminal_cost: TerminalCost,
    action_lower: Tensor | float,
    action_upper: Tensor | float,
    learning_rate: float,
    max_iterations: int,
    validation_initial_states: Sequence[Tensor] | None = None,
    evaluation_interval: int = 10,
    projected_tolerance: float = 1e-8,
    patience_evaluations: int = 50,
    minimum_improvement: float = 1e-10,
    return_policy: str = "best_validation",
) -> NeuralPMPResult:
    """Run Algorithm 1 with a fully predeclared stopping/checkpoint rule.

    ``best_validation`` selects only by learned-model validation objective.
    ``fixed_final`` returns the final fixed-budget iterate and treats validation
    as diagnostic only. A true-system or canonical realized objective is not
    accepted, so it cannot influence either policy.
    """
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if max_iterations < 0 or evaluation_interval <= 0:
        raise ValueError("invalid iteration settings")
    if patience_evaluations <= 0:
        raise ValueError("patience_evaluations must be positive")
    if return_policy not in {"best_validation", "fixed_final"}:
        raise ValueError("return_policy must be best_validation or fixed_final")

    control = project_action(initial_control.detach().clone(), action_lower, action_upper)
    validation_states = list(validation_initial_states or [initial_state])
    history: list[dict[str, float | int]] = []
    best_control = control.clone()
    best_score = float("inf")
    best_iteration = 0
    stale_evaluations = 0
    stop_reason = "max_iterations"
    last_iteration = 0

    for iteration in range(max_iterations + 1):
        last_iteration = iteration
        states, costates, raw_gradient = discrete_adjoint_gradient(
            dynamics, initial_state, control, stage_cost, terminal_cost
        )
        residual = float(projected_gradient_residual(control, raw_gradient, action_lower, action_upper))

        should_evaluate = iteration == 0 or iteration % evaluation_interval == 0 or iteration == max_iterations
        if should_evaluate:
            training_value = float(objective(states, control, stage_cost, terminal_cost))
            validation_score = _validation_score(
                dynamics, validation_states, control, stage_cost, terminal_cost
            )
            improved = validation_score < best_score - minimum_improvement
            if improved:
                best_score = validation_score
                best_control = control.clone()
                best_iteration = iteration
                stale_evaluations = 0
            else:
                stale_evaluations += 1
            history.append(
                {
                    "iteration": iteration,
                    "training_objective": training_value,
                    "validation_objective": validation_score,
                    "projected_gradient_residual": residual,
                    "raw_gradient_l2": float(torch.linalg.vector_norm(raw_gradient)),
                    "control_min": float(control.min()),
                    "control_max": float(control.max()),
                }
            )
            if return_policy == "best_validation" and residual <= projected_tolerance:
                stop_reason = "projected_tolerance"
                break
            if return_policy == "best_validation" and stale_evaluations >= patience_evaluations:
                stop_reason = "validation_patience"
                break

        if iteration == max_iterations:
            break

        # Paper Eqs. (32)--(33): use raw dH/du, then project the action.
        # There is deliberately no gradient clipping here.
        control = project_action(
            control - learning_rate * raw_gradient,
            action_lower,
            action_upper,
        ).detach()

    if return_policy == "fixed_final":
        selected_control = control
        selected_iteration = last_iteration
        stop_reason = "fixed_budget_final"
    else:
        selected_control = best_control
        selected_iteration = best_iteration
    best_states, best_costates, best_gradient = discrete_adjoint_gradient(
        dynamics, initial_state, selected_control, stage_cost, terminal_cost
    )
    return NeuralPMPResult(
        control=selected_control,
        states=best_states,
        costates=best_costates,
        raw_gradient=best_gradient,
        history=history,
        best_iteration=selected_iteration,
        stop_reason=stop_reason,
    )
