"""Reusable bounded-control wrappers for time-only Transformer checkpoints."""

from __future__ import annotations

import math

import torch
from torch import nn


class BoundaryProjectedControl(nn.Module):
    """Scale a bounded base control and project it onto ``[0, umax]``.

    The wrapper is kept in ``scripts/`` because it is shared by training,
    evaluation, and timing entry points.  Its state-dict layout matches the
    earlier experiment-local implementation.
    """

    def __init__(
        self,
        base: nn.Module,
        *,
        umax: float,
        scale_mode: str,
        initial_scale: float,
        scale_min: float = 1.01,
        scale_max: float = 1.03,
    ) -> None:
        super().__init__()
        self.base = base
        self.umax = float(umax)
        self.scale_mode = scale_mode
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        if not math.isfinite(self.umax) or self.umax <= 0.0:
            raise ValueError("umax must be positive and finite")
        if scale_mode == "fixed":
            self.register_buffer(
                "fixed_scale", torch.tensor(float(initial_scale), dtype=torch.float64)
            )
        elif scale_mode == "learnable":
            if not self.scale_min < float(initial_scale) < self.scale_max:
                raise ValueError(
                    "a learnable initial scale must lie strictly between "
                    "scale_min and scale_max"
                )
            fraction = (float(initial_scale) - self.scale_min) / (
                self.scale_max - self.scale_min
            )
            raw = math.log(fraction / (1.0 - fraction))
            self.raw_scale = nn.Parameter(torch.tensor(raw, dtype=torch.float64))
        else:
            raise ValueError(f"unsupported scale_mode {scale_mode!r}")

    def scale_value(self) -> torch.Tensor:
        if self.scale_mode == "fixed":
            return self.fixed_scale
        return self.scale_min + (self.scale_max - self.scale_min) * torch.sigmoid(
            self.raw_scale
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.scale_value() * self.base(t), 0.0, self.umax)
