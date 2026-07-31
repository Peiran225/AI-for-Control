from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from scripts.refine_feedback_svd_null_projected_kkt import (
    GuardRollout,
    SVDNullLinear,
    approximate_right_nullspace,
    configure_loss,
    guard_metrics,
    parse_radii,
)


def test_parse_radii_preserves_order_and_removes_duplicates() -> None:
    assert parse_radii("0,.10,.20,.10,.60") == (
        0.0,
        0.1,
        0.2,
        0.6,
    )


def test_approximate_nullspace_reports_exact_and_soft_dimensions() -> None:
    matrix = torch.diag(
        torch.tensor([4.0, 1.0, 1.0e-9], dtype=torch.float64)
    )
    basis, singular_values, report = approximate_right_nullspace(
        matrix, 1.0e-8
    )
    assert singular_values.shape == (3,)
    assert report.strict_rank == 3
    assert report.strict_nullity == 0
    assert report.approximate_rank == 2
    assert report.approximate_nullity == 1
    assert basis.shape == (3, 1)
    assert float((matrix @ basis).abs().max()) <= 1.1e-9


def test_null_linear_folds_to_the_native_linear_exactly() -> None:
    dtype = torch.float64
    source = nn.Linear(5, 1, dtype=dtype)
    basis, _ = torch.linalg.qr(
        torch.randn(5, 2, dtype=dtype), mode="reduced"
    )
    layer = SVDNullLinear(source, basis)
    with torch.no_grad():
        layer.z.copy_(torch.tensor([0.2, -0.3], dtype=dtype))
    folded = layer.folded_linear()
    query = torch.randn(11, 5, dtype=dtype)
    torch.testing.assert_close(
        layer(query), folded(query), atol=1.0e-14, rtol=1.0e-14
    )
    expected = source.weight + (basis @ layer.z).reshape_as(source.weight)
    torch.testing.assert_close(folded.weight, expected)


def test_loss_configuration_uses_projected_kkt_and_no_smoothing() -> None:
    common = argparse.Namespace(
        w0=1.0,
        w1=1.0,
        w2=4.0,
        w_lc=1.0,
        psi_scale=1.0,
        dot_scale=1.0,
        ddot_scale=1.0,
        B_scale=1.0,
        cf_candidate_weight=0.01,
        boundary_weight=1.0,
        full_gradient_weight=50.0,
        full_gradient_scale=0.0,
        full_gradient_projection_step=1.0,
        full_gradient_max_weight=5.0,
        full_gradient_max_tau=0.1,
    )
    cf = configure_loss(argparse.Namespace(), "cf", common)
    der = configure_loss(argparse.Namespace(), "der", common)
    assert cf.singular_loss_weight == 0.01
    assert der.singular_loss_weight == 1.0
    assert der.w2 == 4.0
    assert cf.nonsingular_loss_weight == der.nonsingular_loss_weight == 1.0
    assert cf.full_gradient_residual == der.full_gradient_residual == "projected"
    assert cf.full_gradient_weight == der.full_gradient_weight == 50.0
    assert cf.full_gradient_max_weight == der.full_gradient_max_weight == 5.0
    assert cf.smooth_weight == der.smooth_weight == 0.0


def test_guard_metrics_supports_strict_and_display_radius_tiers() -> None:
    reference = GuardRollout(
        radii=(0.0, 0.2, 0.6),
        states=torch.zeros(3, 2, 1, dtype=torch.float64),
        controls=torch.zeros(3, 1, dtype=torch.float64),
        stage_states=torch.zeros(3, 1, 4, 1, dtype=torch.float64),
        objectives=torch.zeros(3, dtype=torch.float64),
        terminal_resistant_burden=torch.zeros(3, dtype=torch.float64),
        common_r_sing=torch.tensor(
            [10.0, 20.0, 30.0], dtype=torch.float64
        ),
    )
    candidate = GuardRollout(
        radii=reference.radii,
        states=reference.states.clone(),
        controls=reference.controls.clone(),
        stage_states=reference.stage_states.clone(),
        objectives=reference.objectives.clone(),
        terminal_resistant_burden=(
            reference.terminal_resistant_burden.clone()
        ),
        common_r_sing=reference.common_r_sing.clone(),
    )
    candidate.controls[2, 0] = 5.0e-5
    candidate.states[2, -1, 0] = 5.0e-4
    candidate.objectives[2] = 5.0e-3
    candidate.terminal_resistant_burden[2] = 2.0e-4
    candidate.common_r_sing[2] = 30.015
    args = argparse.Namespace(
        _strict_guard_radii=(0.0, 0.2),
        max_guard_control_drift=1.0e-8,
        max_guard_state_drift=1.0e-6,
        max_guard_physical_objective_drift=5.0e-4,
        max_guard_terminal_resistant_drift=1.0e-6,
        max_display_guard_control_drift=1.0e-4,
        max_display_guard_state_drift=1.0e-3,
        max_display_guard_physical_objective_drift=1.0e-2,
        max_display_guard_terminal_resistant_drift=5.0e-4,
        max_guard_rsing_relative_drift=1.0e-6,
        max_display_guard_rsing_relative_drift=1.0e-3,
    )
    metrics = guard_metrics(
        candidate,
        reference,
        physical_scale_factor=1.0,
        args=args,
    )
    assert metrics["passed"]
    assert metrics["per_radius"]["0.2"]["guard_tier"] == "strict"
    assert metrics["per_radius"]["0.6"]["guard_tier"] == "display"
    assert metrics["per_radius"]["0.6"]["passed"]

    candidate.controls[1, 0] = 5.0e-8
    metrics = guard_metrics(
        candidate,
        reference,
        physical_scale_factor=1.0,
        args=args,
    )
    assert not metrics["passed"]
    assert metrics["per_radius"]["0.2"]["failed_metrics"] == [
        "control_max_abs"
    ]
