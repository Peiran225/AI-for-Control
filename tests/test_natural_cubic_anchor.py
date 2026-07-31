from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for search_path in (ROOT, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from natural_cubic_anchor import (  # noqa: E402
    natural_cubic_second_derivatives,
    natural_cubic_uniform_value,
    natural_cubic_uniform_value_derivatives,
)


DTYPE = torch.float64


def _quintic_flat_union_gate(
    anchor_values: torch.Tensor,
    anchor_seconds: torch.Tensor,
    normalized_time: torch.Tensor,
    state: torch.Tensor,
    *,
    normalization: float,
    tube: float,
    transition: float,
) -> torch.Tensor:
    anchors = natural_cubic_uniform_value(
        anchor_values, anchor_seconds, normalized_time
    )
    relative = (state[:, None, :] - anchors) / normalization
    distance_squared = relative.square().mean(dim=-1)
    inner_squared = tube * tube
    outer_squared = (tube + transition) ** 2
    coordinate = (
        (distance_squared - inner_squared)
        / (outer_squared - inner_squared)
    ).clamp(0.0, 1.0)
    per_anchor = coordinate.pow(3) * (
        10.0 - 15.0 * coordinate + 6.0 * coordinate.square()
    )
    return per_anchor.prod(dim=1)


def _analytic_paths(time: torch.Tensor) -> torch.Tensor:
    first = torch.stack(
        (
            10.0 + torch.sin(torch.pi * time),
            9.0 + 0.5 * torch.sin(2.0 * torch.pi * time),
        ),
        dim=-1,
    )
    second = torch.stack(
        (
            11.0 + 0.75 * torch.sin(torch.pi * time),
            8.5 + 0.25 * torch.sin(2.0 * torch.pi * time),
        ),
        dim=-1,
    )
    return torch.stack((first, second), dim=0)


def test_natural_boundary_and_tridiagonal_equations() -> None:
    generator = torch.Generator().manual_seed(17)
    values = torch.randn(3, 19, 4, generator=generator, dtype=DTYPE)
    seconds = natural_cubic_second_derivatives(values)
    h = 1.0 / (values.shape[1] - 1)

    assert torch.equal(seconds[:, 0, :], torch.zeros_like(seconds[:, 0, :]))
    assert torch.equal(seconds[:, -1, :], torch.zeros_like(seconds[:, -1, :]))
    lhs = seconds[:, :-2, :] + 4.0 * seconds[:, 1:-1, :]
    lhs = lhs + seconds[:, 2:, :]
    rhs = 6.0 * (
        values[:, 2:, :] - 2.0 * values[:, 1:-1, :] + values[:, :-2, :]
    ) / (h * h)
    assert torch.allclose(lhs, rhs, rtol=2.0e-13, atol=2.0e-11)


def test_anchor_knots_are_reproduced_to_roundoff() -> None:
    grid = torch.linspace(0.0, 1.0, 33, dtype=DTYPE)
    values = _analytic_paths(grid)
    seconds = natural_cubic_second_derivatives(values)
    interpolated = natural_cubic_uniform_value(values, seconds, grid)
    expected = values.permute(1, 0, 2)
    assert torch.allclose(interpolated, expected, rtol=0.0, atol=2.0e-15)


def test_value_first_and_second_derivatives_are_continuous_at_knots() -> None:
    grid = torch.linspace(0.0, 1.0, 21, dtype=DTYPE)
    values = _analytic_paths(grid)
    seconds = natural_cubic_second_derivatives(values)
    epsilon = 1.0e-10

    for index in (1, 5, 10, 16, 19):
        knot = float(index) / float(grid.numel() - 1)
        left = torch.tensor([knot - epsilon], dtype=DTYPE)
        right = torch.tensor([knot + epsilon], dtype=DTYPE)
        left_result = natural_cubic_uniform_value_derivatives(
            values, seconds, left
        )
        right_result = natural_cubic_uniform_value_derivatives(
            values, seconds, right
        )
        assert torch.allclose(
            left_result[0], right_result[0], rtol=0.0, atol=3.0e-8
        )
        assert torch.allclose(
            left_result[1], right_result[1], rtol=0.0, atol=2.0e-7
        )
        assert torch.allclose(
            left_result[2], right_result[2], rtol=0.0, atol=3.0e-8
        )


def test_autograd_matches_closed_form_time_derivatives() -> None:
    grid = torch.linspace(0.0, 1.0, 25, dtype=DTYPE)
    values = _analytic_paths(grid)
    seconds = natural_cubic_second_derivatives(values)
    query = torch.tensor([0.137, 0.413, 0.777], dtype=DTYPE, requires_grad=True)
    weights = torch.tensor(
        [[[0.7, -0.2], [0.3, 0.5]]], dtype=DTYPE
    ).expand(query.numel(), -1, -1)

    value, first, second = natural_cubic_uniform_value_derivatives(
        values, seconds, query
    )
    scalar_by_query = (value * weights).sum(dim=(1, 2))
    automatic_first = torch.autograd.grad(
        scalar_by_query.sum(), query, create_graph=True
    )[0]
    automatic_second = torch.autograd.grad(
        automatic_first.sum(), query
    )[0]
    expected_first = (first * weights).sum(dim=(1, 2))
    expected_second = (second * weights).sum(dim=(1, 2))
    assert torch.allclose(
        automatic_first, expected_first, rtol=2.0e-13, atol=2.0e-13
    )
    assert torch.allclose(
        automatic_second, expected_second, rtol=2.0e-12, atol=2.0e-12
    )


def test_nodes_midpoints_and_rk4_stage_times_remain_in_exact_flat_tube() -> None:
    nodes = 65
    grid = torch.linspace(0.0, 1.0, nodes, dtype=DTYPE)
    anchor_values = _analytic_paths(grid)
    anchor_seconds = natural_cubic_second_derivatives(anchor_values)

    intervals = torch.arange(nodes - 1, dtype=DTYPE) / (nodes - 1)
    step = 1.0 / (nodes - 1)
    query = torch.cat(
        (
            grid,
            intervals + 0.25 * step,
            intervals + 0.50 * step,
            intervals + 0.75 * step,
        )
    )
    for protected_index in range(anchor_values.shape[0]):
        analytic = _analytic_paths(query)[protected_index]
        gate = _quintic_flat_union_gate(
            anchor_values,
            anchor_seconds,
            query,
            analytic,
            normalization=10.0,
            tube=1.0e-4,
            transition=2.0e-3,
        )
        assert torch.equal(gate, torch.zeros_like(gate))


def test_outer_gate_has_continuous_first_and_second_time_derivatives() -> None:
    grid = torch.linspace(0.0, 1.0, 25, dtype=DTYPE)
    anchor_values = 0.02 * _analytic_paths(grid)
    anchor_seconds = natural_cubic_second_derivatives(anchor_values)
    epsilon = 1.0e-8

    def gate_derivatives(time_value: float) -> tuple[float, float, float]:
        time = torch.tensor([time_value], dtype=DTYPE, requires_grad=True)
        state = torch.tensor([[0.17, 0.17]], dtype=DTYPE)
        gate = _quintic_flat_union_gate(
            anchor_values,
            anchor_seconds,
            time,
            state,
            normalization=1.0,
            tube=0.0,
            transition=0.16,
        )
        first = torch.autograd.grad(gate.sum(), time, create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), time)[0]
        return (
            float(gate.detach()),
            float(first.detach()),
            float(second.detach()),
        )

    for index in (3, 8, 13, 20):
        knot = float(index) / float(grid.numel() - 1)
        left = gate_derivatives(knot - epsilon)
        right = gate_derivatives(knot + epsilon)
        assert abs(left[0] - right[0]) < 2.0e-8
        assert abs(left[1] - right[1]) < 2.0e-7
        assert abs(left[2] - right[2]) < 2.0e-5
