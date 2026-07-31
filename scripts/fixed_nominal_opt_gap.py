#!/usr/bin/env python3
"""Common fixed-nominal validation metric for staged training traces.

The paper's optimization-dynamics figure must not join unrelated native
training losses.  This module evaluates every recorded checkpoint with one
frozen DER optimality-gap definition on the nominal initial state
``N_i(0)=n0``.  The metric includes both the regime-aware singular residual
and the nonsingular boundary residual from ``section5_loss``.

The evaluator is deliberately independent of the optimizer used by a stage.
It can consume either a nested state--time policy or a live time-only
Transformer wrapped by one of the box-projection modules used in this
repository.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from scripts.train_feedback_section5 import section5_loss
from train_paper_pmp_kkt import ProblemConfig, time_features


@dataclass(frozen=True)
class FixedDERMetricSpec:
    """Locked numerical definition used for every trace checkpoint."""

    singular_eps: float = 0.1
    singular_tau: float = 0.03
    dot_eps: float = 0.1
    dot_tau: float = 0.03
    b_min: float = 1.0e-8
    w0: float = 1.0
    w1: float = 1.0
    w2: float = 4.0
    w_lc: float = 1.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


DEFAULT_DER_METRIC = FixedDERMetricSpec()


def _loss_args(
    spec: FixedDERMetricSpec,
    *,
    state_mode: str,
) -> argparse.Namespace:
    """Build only the arguments read by ``section5_loss``."""

    return argparse.Namespace(
        option="der",
        # Match the active manuscript equations exactly.  In particular, the
        # DER admissibility indicator includes B <= 0.
        loss_variant="literal",
        state_mode=state_mode,
        training_integrator="rk4",
        singular_eps=spec.singular_eps,
        singular_tau=spec.singular_tau,
        dot_eps=spec.dot_eps,
        dot_tau=spec.dot_tau,
        b_min=spec.b_min,
        gate_gradient_mode="live",
        w0=spec.w0,
        w1=spec.w1,
        w2=spec.w2,
        w_lc=spec.w_lc,
        psi_scale=1.0,
        dot_scale=1.0,
        ddot_scale=1.0,
        B_scale=1.0,
        singular_loss_weight=1.0,
        nonsingular_loss_weight=1.0,
        smooth_weight=0.0,
        smooth_second_weight=0.0,
        smooth_max_weight=0.0,
        smooth_max_tau=0.02,
        full_gradient_weight=0.0,
        full_gradient_max_weight=0.0,
    )


class TimeOnlyPolicyAdapter(nn.Module):
    """Expose a live time-only wrapper through the feedback-policy interface."""

    def __init__(
        self,
        time_model: nn.Module,
        cfg: ProblemConfig,
    ) -> None:
        super().__init__()
        if not hasattr(time_model, "base"):
            raise TypeError(
                "the time-only validation adapter expects a wrapper with "
                "a `.base` TimeTransformer"
            )
        self.time_branch = time_model.base
        self.umax = float(cfg.umax)
        wrapper_name = type(time_model).__name__
        self.action_parameterization = (
            "linear-raw-box"
            if wrapper_name == "LinearRawBoxProjection"
            else "logit-temperature"
        )
        self.action_scale = float(
            getattr(time_model, "scale", 1.0)
        )
        self.action_offset = float(
            getattr(time_model, "offset", 0.0)
        )
        self.action_temperature = float(
            getattr(time_model, "temperature", 1.0)
        )

    def time_logits(self, normalized_time_grid: torch.Tensor) -> torch.Tensor:
        hidden = self.time_branch.input(
            time_features(normalized_time_grid)
        )
        hidden = self.time_branch.encoder(hidden.unsqueeze(0)).squeeze(0)
        return self.time_branch.output(hidden).squeeze(-1)

    def interval_action(
        self,
        base_logit: torch.Tensor,
        normalized_time: torch.Tensor,
        state: torch.Tensor,
        *,
        state_blind: bool = False,
        state_mode: str | None = None,
    ) -> torch.Tensor:
        del normalized_time, state_blind, state_mode
        if self.action_parameterization == "linear-raw-box":
            value = base_logit
        else:
            value = (
                self.action_scale
                * self.umax
                * torch.sigmoid(base_logit / self.action_temperature)
                - self.action_offset
            )
        return torch.clamp(value, 0.0, self.umax).expand(state.shape[0])


def evaluate_fixed_nominal_der(
    model: nn.Module,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
    *,
    state_mode: str,
    spec: FixedDERMetricSpec = DEFAULT_DER_METRIC,
) -> dict[str, float]:
    """Evaluate the locked nominal DER optimality gap.

    Parameters
    ----------
    model:
        A ``NestedFeedbackTransformer``-compatible module.
    cfg:
        The fixed validation problem.  For the paper trace this is the base
        ``n=800`` problem even when a refinement stage trains on a denser grid.
    params:
        Dynamics parameters built from exactly ``cfg``.
    state_mode:
        ``"w_zero"`` for a time-only policy and ``"feedback"`` for a
        state--time policy.
    """

    if state_mode not in {"w_zero", "feedback"}:
        raise ValueError("state_mode must be 'w_zero' or 'feedback'")
    # Keep every stage on the same attention arithmetic path.  The
    # inference-only fused MHA kernel is numerically close, but the small
    # logit difference is amplified by this sensitive controlled rollout.
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    try:
        parameter = next(model.parameters())
    except StopIteration as error:
        raise ValueError("validation model has no parameters") from error
    initial_state = torch.full(
        (1, cfg.m),
        cfg.n0,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    was_training = model.training
    model.eval()
    with torch.no_grad():
        pack = section5_loss(
            model,
            initial_state,
            cfg,
            params,
            _loss_args(spec, state_mode=state_mode),
            state_mode=state_mode,
        )
    model.train(was_training)
    return {
        "fixed_nominal_lopt_der": float(pack["opt_gap"].detach().cpu()),
        "fixed_nominal_singular_component": float(
            pack["singular_component"].detach().cpu()
        ),
        "fixed_nominal_boundary_component": float(
            pack["nonsingular_component"].detach().cpu()
        ),
        "fixed_nominal_invalid_component": float(
            pack["invalid_component"].detach().cpu()
        ),
        "fixed_nominal_q_mean": float(pack["q"].mean().detach().cpu()),
    }


def metric_metadata(
    spec: FixedDERMetricSpec = DEFAULT_DER_METRIC,
) -> dict[str, Any]:
    return {
        "name": "fixed_nominal_der_optimality_gap",
        "initial_state": "N_i(0)=n0 for every phenotype",
        "integrator": "float64 RK4 with matching discrete adjoint",
        "loss_variant": "literal",
        "contains_singular_and_boundary_terms": True,
        "residual_scales": [1.0, 1.0, 1.0],
        **spec.as_dict(),
    }
