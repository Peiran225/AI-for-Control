from __future__ import annotations

import torch

from scripts.constrained_scalar_lm_step import (
    constrained_lm_step,
    der_residual_vector_from_pack,
    forward_jacobian_columns,
    nonlinear_feasible_backtracking,
)


def test_der_residual_vector_matches_weighted_scalar_loss() -> None:
    dtype = torch.float64
    quantities = {
        "psi": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=dtype),
        "dot_psi": torch.tensor([[0.5, -1.0], [1.5, -2.0]], dtype=dtype),
        "ddot_psi": torch.tensor([[2.0, 1.0], [-1.0, -2.0]], dtype=dtype),
    }
    time_mask = torch.tensor([[1.0, 0.5]], dtype=dtype)
    state_weights = torch.tensor([0.25, 0.75], dtype=dtype)
    pack = {
        "quantities": quantities,
        "weighted_mask": time_mask,
        "state_weights": state_weights,
    }
    residual = der_residual_vector_from_pack(pack)
    loss_mask = time_mask.expand(2, -1) * state_weights[:, None]
    denominator = time_mask.sum() * state_weights.sum()
    expected = sum(
        weight * (loss_mask * quantities[name].square()).sum() / denominator
        for weight, name in (
            (1.0, "psi"),
            (1.0, "dot_psi"),
            (4.0, "ddot_psi"),
        )
    )
    torch.testing.assert_close(residual.square().sum(), expected)


def test_forward_jacobian_columns_matches_analytic_jacobian() -> None:
    dtype = torch.float64
    point = torch.tensor([0.4, -0.7], dtype=dtype)

    def function(value: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                value[0].square() + 3.0 * value[1],
                value[0] * value[1],
            )
        )

    result, jacobian = forward_jacobian_columns(function, point)
    torch.testing.assert_close(result, function(point))
    torch.testing.assert_close(
        jacobian,
        torch.tensor(
            [[0.8, 3.0], [-0.7, 0.4]],
            dtype=dtype,
        ),
    )


def test_nonlinear_backtracking_rejects_full_step_and_accepts_half() -> None:
    dtype = torch.float64
    current = torch.tensor([0.0], dtype=dtype)
    step = torch.tensor([2.0], dtype=dtype)

    def evaluate(point: torch.Tensor) -> tuple[float, bool]:
        objective = float((point - 1.0).square().sum())
        # This nonlinear protection gate rejects the full step at x=2 but
        # accepts the half step at x=1.
        feasible = float(point.abs().max()) <= 1.25
        return objective, feasible

    result = nonlinear_feasible_backtracking(
        current,
        step,
        current_objective=1.0,
        evaluate_trial=evaluate,
        maximum_steps=4,
        factor=0.5,
        acceptance_tolerance=0.0,
    )
    assert result.accepted
    assert result.backtracking_step == 1
    assert result.scale == 0.5
    torch.testing.assert_close(
        result.point,
        torch.tensor([1.0], dtype=dtype),
    )
    assert result.objective == 0.0


def test_unconstrained_step_matches_damped_normal_equation() -> None:
    dtype = torch.float64
    residual = torch.tensor([1.0, -2.0, 0.5], dtype=dtype)
    jacobian = torch.tensor(
        [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]],
        dtype=dtype,
    )
    damping = 0.2
    normal = jacobian.T @ jacobian
    expected = torch.linalg.solve(
        normal + damping * torch.diag(normal.diagonal()),
        -(jacobian.T @ residual),
    )
    result = constrained_lm_step(
        residual,
        jacobian,
        torch.empty((0, 2), dtype=dtype),
        torch.empty((0,), dtype=dtype),
        damping=damping,
        maximum_step_norm=100.0,
    )
    torch.testing.assert_close(result.step, expected)
    assert result.linearized_feasible
    assert result.active_constraints == ()


def test_active_constraint_blocks_loss_increase() -> None:
    dtype = torch.float64
    # Unconstrained minimization moves d_0 positive, but the protected loss
    # requires d_0 <= 0 at the current protection boundary.
    residual = torch.tensor([-1.0, 0.0], dtype=dtype)
    jacobian = torch.eye(2, dtype=dtype)
    gradients = torch.tensor([[1.0, 0.0]], dtype=dtype)
    budgets = torch.zeros(1, dtype=dtype)
    result = constrained_lm_step(
        residual,
        jacobian,
        gradients,
        budgets,
        damping=1.0e-3,
        maximum_step_norm=10.0,
    )
    assert result.linearized_feasible
    assert result.active_constraints == (0,)
    assert float(result.step[0]) <= 1.0e-12


def test_two_protection_constraints_and_step_cap() -> None:
    dtype = torch.float64
    residual = torch.tensor([-10.0, -10.0], dtype=dtype)
    jacobian = torch.eye(2, dtype=dtype)
    gradients = torch.eye(2, dtype=dtype)
    budgets = torch.tensor([0.2, 0.3], dtype=dtype)
    result = constrained_lm_step(
        residual,
        jacobian,
        gradients,
        budgets,
        damping=1.0e-6,
        maximum_step_norm=0.25,
    )
    assert result.linearized_feasible
    assert float(result.step.norm()) <= 0.25 + 1.0e-12
    assert bool((gradients @ result.step <= budgets + 1.0e-12).all())


def test_rejects_already_violated_protection_limit() -> None:
    dtype = torch.float64
    try:
        constrained_lm_step(
            torch.ones(1, dtype=dtype),
            torch.ones((1, 1), dtype=dtype),
            torch.ones((1, 1), dtype=dtype),
            torch.tensor([-1.0], dtype=dtype),
            damping=1.0,
            maximum_step_norm=1.0,
        )
    except ValueError as error:
        assert "current iterate" in str(error)
    else:
        raise AssertionError("negative protection budget should be rejected")
