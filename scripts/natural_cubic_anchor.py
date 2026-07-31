#!/usr/bin/env python3
"""Natural-cubic interpolation for fixed, uniformly sampled anchor paths.

The anchor values are treated as immutable data.  The returned interpolant is
differentiable with respect to normalized query time and is C2 across every
interior knot.  Second derivatives are precomputed once with a vectorized
Thomas solve on CPU and can then be stored alongside the anchor values in a
self-contained checkpoint.
"""

from __future__ import annotations

import numpy as np
import torch


def _validate_anchor_values(values: torch.Tensor) -> None:
    if values.ndim != 3:
        raise ValueError(
            "anchor values must have shape (protected, nodes, features)"
        )
    if values.shape[1] < 2:
        raise ValueError("at least two anchor nodes are required")
    if not values.is_floating_point():
        raise TypeError("anchor values must use a floating-point dtype")
    if not torch.isfinite(values).all():
        raise ValueError("anchor values must be finite")


def natural_cubic_second_derivatives(
    values: torch.Tensor,
) -> torch.Tensor:
    """Return knotwise natural-spline second derivatives.

    ``values`` has shape ``(protected, nodes, features)`` on the uniform
    normalized grid ``0, 1/(nodes-1), ..., 1``.  The solve is intentionally
    detached because protected anchor paths are fixed protocol data; query-time
    differentiation remains fully supported by the evaluator below.
    """

    _validate_anchor_values(values)
    protected, nodes, features = values.shape
    result = np.zeros((protected, nodes, features), dtype=np.float64)
    if nodes == 2:
        return torch.as_tensor(
            result, device=values.device, dtype=values.dtype
        )

    y = (
        values.detach()
        .to(device="cpu", dtype=torch.float64)
        .contiguous()
        .numpy()
    )
    h = 1.0 / float(nodes - 1)
    rhs = 6.0 * (y[:, 2:, :] - 2.0 * y[:, 1:-1, :] + y[:, :-2, :])
    rhs /= h * h

    interior = nodes - 2
    modified_upper = np.empty(interior, dtype=np.float64)
    modified_rhs = np.empty_like(rhs)

    denominator = 4.0
    modified_upper[0] = 1.0 / denominator if interior > 1 else 0.0
    modified_rhs[:, 0, :] = rhs[:, 0, :] / denominator
    for index in range(1, interior):
        denominator = 4.0 - modified_upper[index - 1]
        modified_upper[index] = (
            1.0 / denominator if index < interior - 1 else 0.0
        )
        modified_rhs[:, index, :] = (
            rhs[:, index, :] - modified_rhs[:, index - 1, :]
        ) / denominator

    solution = np.empty_like(rhs)
    solution[:, -1, :] = modified_rhs[:, -1, :]
    for index in range(interior - 2, -1, -1):
        solution[:, index, :] = (
            modified_rhs[:, index, :]
            - modified_upper[index] * solution[:, index + 1, :]
        )
    result[:, 1:-1, :] = solution
    return torch.as_tensor(result, device=values.device, dtype=values.dtype)


def _interval_data(
    values: torch.Tensor,
    second_derivatives: torch.Tensor,
    normalized_time: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
]:
    _validate_anchor_values(values)
    if second_derivatives.shape != values.shape:
        raise ValueError("second derivatives must match anchor-value shape")
    if normalized_time.ndim == 0:
        normalized_time = normalized_time.unsqueeze(0)
    elif normalized_time.ndim != 1:
        raise ValueError("normalized query time must be scalar or one-dimensional")
    if not normalized_time.is_floating_point():
        raise TypeError("normalized query time must be floating point")

    nodes = values.shape[1]
    last = nodes - 1
    position = normalized_time.clamp(0.0, 1.0) * last
    left = torch.floor(position).long().clamp(min=0, max=last - 1)
    right = left + 1
    fraction = position - left.to(position.dtype)
    a = 1.0 - fraction
    b = fraction

    def gather(source: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return source.index_select(1, indices).permute(1, 0, 2)

    return (
        gather(values, left),
        gather(values, right),
        gather(second_derivatives, left),
        gather(second_derivatives, right),
        a[:, None, None],
        1.0 / float(last),
    )


def natural_cubic_uniform_value_derivatives(
    values: torch.Tensor,
    second_derivatives: torch.Tensor,
    normalized_time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate value and its first two normalized-time derivatives."""

    y0, y1, m0, m1, a, h = _interval_data(
        values, second_derivatives, normalized_time
    )
    b = 1.0 - a
    value = (
        a * y0
        + b * y1
        + (h * h / 6.0)
        * ((a.pow(3) - a) * m0 + (b.pow(3) - b) * m1)
    )
    first = (
        (y1 - y0) / h
        + (h / 6.0)
        * (
            (1.0 - 3.0 * a.square()) * m0
            + (3.0 * b.square() - 1.0) * m1
        )
    )
    second = a * m0 + b * m1
    return value, first, second


def natural_cubic_uniform_value(
    values: torch.Tensor,
    second_derivatives: torch.Tensor,
    normalized_time: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the C2 anchor interpolant at normalized query times."""

    value, _, _ = natural_cubic_uniform_value_derivatives(
        values, second_derivatives, normalized_time
    )
    return value
