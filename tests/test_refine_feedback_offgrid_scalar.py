from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scripts.refine_feedback_offgrid_scalar import (
    fine_feedback_rollout,
    fine_problem,
    physical_metrics_by_sample,
    scalar_pack,
)
from scripts.feedback_continuous_policy_rk4 import (
    continuous_feedback_pmp_pack,
    hermite_midpoint_costates,
    pchip_midpoint_logits,
)
from scripts.initialize_feedback_from_direct_trajectories import (
    build_nominal_reference,
)
from scripts.train_feedback_section5 import NestedFeedbackTransformer
from train_paper_pmp_kkt import ProblemConfig, build_params


def make_cfg() -> ProblemConfig:
    return ProblemConfig(
        T=2.0,
        n=4,
        m=5,
        umax=3.0,
        beta=0.1,
        alpha=0.0025,
        gamma=20.0,
        n0=10.0,
        m_suppression=0.5,
    )


def test_fine_problem_only_changes_resolution() -> None:
    base = make_cfg()
    fine = fine_problem(base, 4)
    assert fine.n == 16
    assert fine.T == base.T
    assert fine.m == base.m
    assert fine.alpha == base.alpha
    assert fine.beta == base.beta
    assert fine.gamma == base.gamma


def test_fine_rollout_and_scalar_pack_have_matching_shapes() -> None:
    cfg = fine_problem(make_cfg(), 2)
    model = NestedFeedbackTransformer(
        cfg.m,
        cfg.umax,
        15.0,
        (8,),
        8,
        2,
        1,
        1.5,
        state_feature_mode="burden_composition",
        center_state_correction=True,
    ).double()
    model.eval()
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    model.set_feature_vectors(params["r"], params["phi"])
    initial = torch.full((2, cfg.m), cfg.n0, dtype=torch.float64)
    time = torch.linspace(0.0, 1.0, cfg.n + 1, dtype=torch.float64)
    raw = model.time_logits(time).detach()
    with torch.no_grad():
        reference, _, _ = fine_feedback_rollout(
            model,
            initial[:1],
            cfg,
            time,
            raw,
            params,
            state_mode="w_zero",
        )
    model.set_nominal_reference(reference[0])
    pack = scalar_pack(
        model,
        initial,
        cfg,
        time,
        raw,
        params,
        option="der",
        interval_start=0.5,
        interval_end=1.5,
    )
    assert pack["states"].shape == (2, cfg.n + 1, cfg.m)
    assert pack["controls"].shape == (2, cfg.n)
    assert pack["quantities"]["psi"].shape == (2, cfg.n, 4)
    assert pack["loss"].ndim == 0
    assert torch.isfinite(pack["loss"])
    p4_pack = scalar_pack(
        model,
        initial,
        cfg,
        time,
        raw,
        params,
        option="der",
        interval_start=0.5,
        interval_end=1.5,
        residual_p=4.0,
    )
    assert torch.isfinite(p4_pack["loss"])
    for name in ("H_u", "dH_u_dt", "d2H_u_dt2"):
        assert (
            p4_pack["component_losses"][name]
            >= pack["component_losses"][name]
        )
    metrics = physical_metrics_by_sample(
        pack,
        scale=2.0,
        names=["nominal", "resistant_heavy"],
    )
    assert set(metrics) == {"nominal", "resistant_heavy"}
    assert all(
        torch.isfinite(torch.tensor(value))
        for row in metrics.values()
        for value in row.values()
    )
    cf_pack = scalar_pack(
        model,
        initial,
        cfg,
        time,
        raw,
        params,
        option="cf",
        interval_start=0.5,
        interval_end=1.5,
    )
    assert torch.allclose(
        cf_pack["loss"],
        cf_pack["component_losses"]["closed_form_control"],
    )
    trusted_pack = scalar_pack(
        model,
        initial,
        cfg,
        time,
        raw,
        params,
        option="cf",
        interval_start=0.5,
        interval_end=1.5,
        cf_scalar_weight=0.5,
        control_trust_weight=2.0,
        anchor_controls=cf_pack["controls"].detach(),
    )
    assert torch.allclose(
        trusted_pack["component_losses"]["control_trust"],
        torch.zeros_like(trusted_pack["loss"]),
    )
    assert trusted_pack["loss"] >= cf_pack["loss"]


def test_continuous_policy_pack_supports_node_and_midpoint_collocation() -> None:
    cfg = fine_problem(make_cfg(), 2)
    model = NestedFeedbackTransformer(
        cfg.m,
        cfg.umax,
        15.0,
        (8,),
        8,
        2,
        1,
        1.5,
        state_feature_mode="burden_composition",
        center_state_correction=True,
    ).double()
    model.eval()
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    model.set_feature_vectors(params["r"], params["phi"])
    initial = torch.full((2, cfg.m), cfg.n0, dtype=torch.float64)
    initial[1] *= torch.linspace(
        0.9, 1.1, cfg.m, dtype=torch.float64
    )
    time = torch.linspace(0.0, 1.0, cfg.n + 1, dtype=torch.float64)
    raw = model.time_logits(time).detach()
    midpoint_raw = pchip_midpoint_logits(time, raw)
    with torch.no_grad():
        reference, _, _ = fine_feedback_rollout(
            model,
            initial[:1],
            cfg,
            time,
            raw,
            params,
            state_mode="w_zero",
        )
    model.set_nominal_reference(reference[0])
    continuous = continuous_feedback_pmp_pack(
        model,
        initial,
        cfg,
        time,
        raw,
        midpoint_raw,
        params,
        state_mode="feedback",
    )
    assert continuous.midpoint_costates.shape == (2, cfg.n, cfg.m)
    assert continuous.midpoint_quantities["psi"].shape == (2, cfg.n)
    assert torch.isfinite(continuous.midpoint_costates).all()
    pack = scalar_pack(
        model,
        initial,
        cfg,
        time,
        raw,
        params,
        trajectory_mode="continuous-policy",
        midpoint_raw_logits=midpoint_raw,
        option="der",
        interval_start=0.5,
        interval_end=1.5,
    )
    assert pack["states"].shape == (2, cfg.n + 1, cfg.m)
    assert pack["controls"].shape == (2, cfg.n)
    assert pack["sample_controls"].shape == (2, cfg.n)
    assert pack["quantities"]["psi"].shape == (2, cfg.n)
    assert torch.allclose(
        pack["quantities"]["psi"],
        continuous.quantities["psi"],
    )
    assert torch.isfinite(pack["loss"])
    pack["loss"].backward()
    final_layer = [
        module
        for module in model.state_branch.modules()
        if isinstance(module, torch.nn.Linear)
    ][-1]
    assert final_layer.weight.grad is not None
    assert torch.isfinite(final_layer.weight.grad).all()
    assert float(final_layer.weight.grad.norm()) > 0.0

    midpoint_pack = scalar_pack(
        model,
        initial,
        cfg,
        time,
        raw,
        params,
        trajectory_mode="continuous-policy",
        midpoint_raw_logits=midpoint_raw,
        continuous_collocation="nodes-midpoints",
        option="der",
        interval_start=0.5,
        interval_end=1.5,
    )
    assert midpoint_pack["sample_controls"].shape == (2, cfg.n, 2)
    assert midpoint_pack["quantities"]["psi"].shape == (2, cfg.n, 2)
    assert torch.allclose(
        midpoint_pack["quantities"]["psi"][..., 0],
        continuous.quantities["psi"],
    )
    assert torch.allclose(
        midpoint_pack["quantities"]["psi"][..., 1],
        continuous.midpoint_quantities["psi"],
    )
    assert torch.isfinite(midpoint_pack["loss"])


def test_hermite_midpoint_costate_has_fourth_order_convergence() -> None:
    """The midpoint formula has the expected orientation and order."""

    def error(intervals: int) -> float:
        cfg = ProblemConfig(
            T=1.0,
            n=intervals,
            m=1,
            umax=3.0,
            beta=0.2,
            alpha=0.4,
            gamma=1.0,
            n0=1.0,
            m_suppression=0.5,
        )
        dtype = torch.float64
        time = torch.linspace(0.0, cfg.T, cfg.n + 1, dtype=dtype)
        midpoint_time = 0.5 * (time[:-1] + time[1:])
        beta = torch.tensor([cfg.beta], dtype=dtype)
        rate = torch.tensor([0.3], dtype=dtype)
        alpha = torch.tensor([cfg.alpha], dtype=dtype)
        exact = (
            (alpha + beta / rate)
            * torch.exp(rate * (cfg.T - time)).unsqueeze(-1)
            - beta / rate
        ).unsqueeze(0)
        exact_midpoint = (
            (alpha + beta / rate)
            * torch.exp(rate * (cfg.T - midpoint_time)).unsqueeze(-1)
            - beta / rate
        ).unsqueeze(0)
        states = torch.ones((1, cfg.n + 1, 1), dtype=dtype)
        controls = torch.zeros((1, cfg.n + 1), dtype=dtype)
        params = {
            "beta": beta,
            "r": rate,
            "phi": torch.zeros(1, dtype=dtype),
            "M": torch.zeros(1, dtype=dtype),
        }
        candidate = hermite_midpoint_costates(
            states,
            controls,
            exact,
            cfg,
            params,
        )
        return float((candidate - exact_midpoint).abs().max())

    coarse = error(4)
    fine = error(8)
    assert fine < coarse / 12.0


def test_pchip_midpoint_logits_preserve_linear_time_logit() -> None:
    time = torch.linspace(0.0, 1.0, 9, dtype=torch.float64)
    raw = 1.25 - 0.75 * time
    midpoint = pchip_midpoint_logits(time, raw)
    expected_time = 0.5 * (time[:-1] + time[1:])
    assert torch.allclose(
        midpoint,
        1.25 - 0.75 * expected_time,
        atol=2.0e-15,
        rtol=0.0,
    )


def test_dop853_nominal_reference_uses_requested_dense_grid() -> None:
    cfg = make_cfg()
    model = NestedFeedbackTransformer(
        cfg.m,
        cfg.umax,
        15.0,
        (8,),
        8,
        2,
        1,
        1.5,
        state_feature_mode="burden_composition",
        center_state_correction=True,
    ).double()
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    model.set_feature_vectors(params["r"], params["phi"])
    args = argparse.Namespace(
        nominal_reference_method="dop853",
        nominal_reference_multiplier=2,
        nominal_reference_query_batch_size=2,
        nominal_reference_rtol=1.0e-9,
        nominal_reference_atol=1.0e-11,
    )
    reference = build_nominal_reference(
        model,
        cfg,
        params,
        Path("/tmp/unused-time-checkpoint.pt"),
        args,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert reference.shape == (2 * cfg.n + 1, cfg.m)
    assert torch.isfinite(reference).all()
    assert torch.all(reference > 0.0)
