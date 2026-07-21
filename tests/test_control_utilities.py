from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from scripts.boundary_control import BoundaryProjectedControl
from scripts.control_switch_metrics import switch_metrics


class ConstantControl(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = float(value)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return torch.full_like(t, self.value)


def test_boundary_projected_control_reaches_upper_bound() -> None:
    model = BoundaryProjectedControl(
        ConstantControl(2.95),
        umax=3.0,
        scale_mode="fixed",
        initial_scale=1.02,
    )
    output = model(torch.linspace(0.0, 1.0, 5, dtype=torch.float64))
    torch.testing.assert_close(output, torch.full_like(output, 3.0))
    assert set(model.state_dict()) == {"fixed_scale"}


def test_switch_metrics_reports_both_transitions() -> None:
    t = np.linspace(0.0, 10.0, 201)
    u = np.full_like(t, 1.0)
    u[t < 0.5] = 3.0
    u[t >= 9.0] = 3.0

    early = switch_metrics(t, u, "early")
    late = switch_metrics(t, u, "late")

    assert early["junction"] == pytest.approx(0.5)
    assert late["junction"] == pytest.approx(9.0)
    assert early["largest_abs_jump_at_junction"] == pytest.approx(2.0)
    assert late["largest_abs_jump_at_junction"] == pytest.approx(2.0)
