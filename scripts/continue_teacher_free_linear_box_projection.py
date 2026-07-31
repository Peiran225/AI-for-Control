#!/usr/bin/env python3
"""Strict teacher-free linear-head / hard-box-projection ablation.

This script changes only the output parameterization of an existing
teacher-free time-only Transformer.  The nested sigmoid/temperature output is
replaced by

    u(t) = projection_[0, umax](w^T h_theta(t) + b),

an affine raw-control head followed by the exact piecewise-linear box
projection.  The source backbone is copied without perturbation.  The new
affine head is initialized once by least squares against the source model's
*pre-projection* control.  This is a parameterization transfer from a strict
teacher-free checkpoint, not a training target: after initialization, neither
the source control nor any direct/manual control is used by the optimizer.

Training minimizes only the complete projected reduced-gradient residual

    G_h(u) = u - projection_[0, umax](u - grad F_h(u)),

where the differentiable RK4 rollout inside F_h retains the full N=N(u)
dependence.  The objective value is deliberately withheld from training and
selection.  It is evaluated only after the residual-selected checkpoint has
been frozen.  The unmodified source checkpoint remains an explicit selection
candidate, so this ablation cannot silently replace it with a worse result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT, ROOT / "scripts"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts.refine_time_only_singular_plateau import (  # noqa: E402
    build_model,
    rk4_reduced_objective,
)
from scripts.train_teacher_free_resolution_curriculum import (  # noqa: E402
    FixedBoxProjection,
    evaluate,
    interval_crossing_width,
    make_plot,
)
from train_paper_pmp_kkt import (  # noqa: E402
    ProblemConfig,
    build_params,
    set_seed,
    time_features,
)


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


class LinearRawBoxProjection(nn.Module):
    """Transformer affine raw-control head plus exact box projection.

    ``torch.clamp`` is the Euclidean projection onto the scalar box and is a
    continuous piecewise-linear map.  There is no sigmoid, temperature, scale,
    time mask, switch parameter, or post-forward correction in this wrapper.
    """

    def __init__(self, base: nn.Module, umax: float) -> None:
        super().__init__()
        self.base = base
        self.umax = float(umax)

    def hidden(self, normalized_t: torch.Tensor) -> torch.Tensor:
        embedded = self.base.input(time_features(normalized_t)).unsqueeze(0)
        return self.base.encoder(embedded).squeeze(0)

    def raw_control(self, normalized_t: torch.Tensor) -> torch.Tensor:
        return self.base.output(self.hidden(normalized_t)).squeeze(-1)

    def forward(self, normalized_t: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.raw_control(normalized_t), 0.0, self.umax)


class _HardClampIdentityBackward(torch.autograd.Function):
    """Literal hard clamp forward with an identity straight-through backward."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, raw: torch.Tensor, lower: float, upper: float
    ) -> torch.Tensor:
        del ctx
        return torch.clamp(raw, float(lower), float(upper))

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, gradient: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        del ctx
        return gradient, None, None


class SemismoothLinearRawBoxProjection(LinearRawBoxProjection):
    """Exact box values with an identity straight-through derivative in training.

    The forward value is bitwise the same hard projection used by
    ``LinearRawBoxProjection``.  Only the parameter-space backward pass uses
    a straight-through identity surrogate.  Outside the box this is not a
    member of the projection's generalized Jacobian.  The surrogate lets
    a residual-only optimizer release a node that is saturated at the wrong
    bound; the saved and selected checkpoint is always re-evaluated with the
    literal hard-projection wrapper.
    """

    def forward(self, normalized_t: torch.Tensor) -> torch.Tensor:
        raw = self.raw_control(normalized_t)
        return _HardClampIdentityBackward.apply(raw, 0.0, self.umax)


def source_model_from_payload(
    source: dict[str, Any], cfg: ProblemConfig
) -> FixedBoxProjection:
    wrapper = dict(source["wrapper"])
    base = build_model(dict(source["base_model_args"]), cfg).to(dtype=torch.float64)
    model = FixedBoxProjection(
        base,
        cfg.umax,
        float(wrapper["scale"]),
        temperature=float(wrapper.get("temperature", 1.0)),
        learn_temperature=bool(wrapper.get("learn_temperature", False)),
    ).to(dtype=torch.float64)
    model.load_state_dict(source["model_state"], strict=True)
    return model


def linear_model_from_payload(
    source: dict[str, Any], cfg: ProblemConfig
) -> LinearRawBoxProjection:
    base = build_model(dict(source["base_model_args"]), cfg).to(dtype=torch.float64)
    model = LinearRawBoxProjection(base, cfg.umax).to(dtype=torch.float64)
    model.load_state_dict(source["model_state"], strict=True)
    return model


def initialize_linear_model(
    source_model: FixedBoxProjection,
    source: dict[str, Any],
    cfg: ProblemConfig,
    normalized_t: torch.Tensor,
    method: str,
) -> tuple[LinearRawBoxProjection, dict[str, float]]:
    """Copy the backbone and least-squares calibrate the new affine head."""

    base = build_model(dict(source["base_model_args"]), cfg).to(dtype=torch.float64)
    base.load_state_dict(source_model.base.state_dict(), strict=True)
    model = LinearRawBoxProjection(base, cfg.umax).to(dtype=torch.float64)

    with torch.no_grad():
        hidden = model.hidden(normalized_t)
        source_logits = source_model.base.output(hidden).squeeze(-1)
        # Match the quantity immediately before the source hard projection.
        # Fitting this rather than the already-clamped u preserves an outward
        # margin on active upper-bound nodes.
        pre_projection = (
            source_model.scale
            * cfg.umax
            * torch.sigmoid(source_logits / source_model.temperature_value())
        )
        source_control = source_model(normalized_t)
        if method == "hidden_ols":
            design = torch.cat(
                [
                    hidden,
                    torch.ones(
                        hidden.shape[0],
                        1,
                        dtype=hidden.dtype,
                        device=hidden.device,
                    ),
                ],
                dim=1,
            )
            # SVD least squares is deterministic for the correlated feature
            # columns.  This is the closest curve transfer, but can yield a
            # large head norm; ``logit_affine`` is the safer alternative.
            coefficients = torch.linalg.lstsq(
                design,
                pre_projection.unsqueeze(-1),
                driver="gelsd",
            ).solution.squeeze(-1)
            model.base.output.weight.copy_(coefficients[:-1].unsqueeze(0))
            model.base.output.bias.copy_(coefficients[-1:])
            affine_scale = float("nan")
            affine_offset = float("nan")
        elif method == "logit_affine":
            # Restrict the new head to the old logit direction.  Only two
            # scalars are fitted: raw=a*z+c.  This keeps the head norm modest
            # and prevents a tiny backbone update from being amplified by an
            # ill-conditioned 64-feature OLS solution.
            centered_logits = source_logits - source_logits.mean()
            centered_target = pre_projection - pre_projection.mean()
            scale = (centered_logits @ centered_target) / (
                centered_logits.square().sum().clamp_min(1.0e-24)
            )
            offset = pre_projection.mean() - scale * source_logits.mean()
            model.base.output.weight.copy_(
                scale * source_model.base.output.weight
            )
            model.base.output.bias.copy_(
                scale * source_model.base.output.bias + offset
            )
            affine_scale = float(scale)
            affine_offset = float(offset)
        else:
            raise ValueError(method)
        fitted_raw = model.raw_control(normalized_t)
        fitted_control = model(normalized_t)
        error = fitted_control - source_control
        diagnostics = {
            "least_squares_control_rmse": float(error.square().mean().sqrt()),
            "least_squares_control_linf": float(error.abs().max()),
            "least_squares_raw_min": float(fitted_raw.min()),
            "least_squares_raw_max": float(fitted_raw.max()),
            "source_preprojection_min": float(pre_projection.min()),
            "source_preprojection_max": float(pre_projection.max()),
            "initialization_method": method,
            "logit_affine_scale": affine_scale,
            "logit_affine_offset": affine_offset,
            "initialized_head_l2_norm": float(model.base.output.weight.norm()),
            "initialized_head_bias": float(model.base.output.bias),
        }
    return model, diagnostics


def residual_pack(
    model: nn.Module,
    normalized_t: torch.Tensor,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
    *,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    control = model(normalized_t)
    interval = control[:-1]
    objective = rk4_reduced_objective(interval, cfg, params)
    gradient = torch.autograd.grad(
        objective,
        interval,
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    mapping = interval - torch.clamp(interval - gradient, 0.0, cfg.umax)
    return control, gradient, mapping


def shape_metrics(
    physical_t: np.ndarray,
    control: np.ndarray,
    cfg: ProblemConfig,
) -> dict[str, Any]:
    interval = np.asarray(control[:-1], dtype=np.float64)
    edge = max(2, cfg.n // 20)
    interior = interval[edge:-edge]
    early = float(np.median(interval[:edge]))
    late = float(np.median(interval[-edge:]))
    interior_low = float(np.quantile(interior, 0.10))
    dynamic_range = float(np.ptp(interval))
    upper = int(np.count_nonzero(interval == cfg.umax))
    lower = int(np.count_nonzero(interval == 0.0))
    nondegenerate = bool(
        dynamic_range >= 0.8
        and early >= interior_low + 0.5
        and late >= interior_low + 0.5
        and upper < int(0.95 * cfg.n)
        and lower < int(0.95 * cfg.n)
    )
    return {
        "nondegenerate_high_low_high": nondegenerate,
        "early_width_10_90": float(
            interval_crossing_width(physical_t, control, "early")
        ),
        "late_width_10_90": float(
            interval_crossing_width(physical_t, control, "late")
        ),
        "total_variation": float(np.abs(np.diff(interval)).sum()),
        "second_difference_l1": float(np.abs(np.diff(interval, n=2)).sum()),
        "u_min": float(interval.min()),
        "u_max": float(interval.max()),
        "exact_upper_bound_count": upper,
        "exact_lower_bound_count": lower,
        "raw_head_min": float("nan"),
        "raw_head_max": float("nan"),
    }


def residual_metrics(
    model: nn.Module,
    normalized_t: torch.Tensor,
    physical_t: np.ndarray,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    with torch.enable_grad():
        control, gradient, mapping = residual_pack(
            model, normalized_t, cfg, params, create_graph=False
        )
    u = control.detach().cpu().numpy()
    g = gradient.detach().cpu().numpy()
    residual = mapping.detach().cpu().numpy()
    row = {
        "projected_gradient_linf": float(np.max(np.abs(residual))),
        "projected_gradient_rms": float(np.sqrt(np.mean(residual**2))),
        "raw_gradient_linf": float(np.max(np.abs(g))),
        **shape_metrics(physical_t, u, cfg),
    }
    if isinstance(model, LinearRawBoxProjection):
        with torch.no_grad():
            raw = model.raw_control(normalized_t).detach().cpu().numpy()[:-1]
        row["raw_head_min"] = float(raw.min())
        row["raw_head_max"] = float(raw.max())
    model.train()
    return row, u, residual


def passes_shape_guard(
    row: dict[str, Any],
    source: dict[str, Any],
    width_tolerance: float,
    variation_fraction_tolerance: float,
    minimum_upper_fraction: float,
) -> bool:
    return bool(
        row["nondegenerate_high_low_high"]
        and math.isfinite(row["early_width_10_90"])
        and math.isfinite(row["late_width_10_90"])
        and row["early_width_10_90"]
        <= source["early_width_10_90"] + width_tolerance
        and row["late_width_10_90"]
        <= source["late_width_10_90"] + width_tolerance
        and row["total_variation"]
        <= source["total_variation"] * (1.0 + variation_fraction_tolerance)
        and row["second_difference_l1"]
        <= source["second_difference_l1"]
        * (1.0 + variation_fraction_tolerance)
        and row["exact_upper_bound_count"]
        >= math.floor(
            minimum_upper_fraction * source["exact_upper_bound_count"]
        )
    )


def passes_selection_guard(
    row: dict[str, Any],
    source: dict[str, Any],
    args: argparse.Namespace,
) -> bool:
    if args.selection_guard == "residual-only":
        return bool(
            math.isfinite(row["projected_gradient_linf"])
            and math.isfinite(row["projected_gradient_rms"])
        )
    return passes_shape_guard(
        row,
        source,
        args.width_tolerance,
        args.variation_fraction_tolerance,
        args.minimum_upper_fraction,
    )


def residual_loss(
    model: nn.Module,
    normalized_t: torch.Tensor,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
    p: float,
    high_p_weight: float,
    solver_mode: str,
    detached_step_multiplier: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if solver_mode == "exact":
        _, _, mapping = residual_pack(
            model, normalized_t, cfg, params, create_graph=True
        )
    elif solver_mode == "detached_fixed_point":
        control = model(normalized_t)
        interval = control[:-1]
        objective = rk4_reduced_objective(interval, cfg, params)
        gradient = torch.autograd.grad(objective, interval)[0]
        with torch.no_grad():
            target = torch.clamp(
                interval.detach()
                - detached_step_multiplier * gradient.detach(),
                0.0,
                cfg.umax,
            )
        # Divide out the step so logged training scales remain comparable to
        # the canonical unit-step mapping.  The fixed points are unchanged.
        mapping = (interval - target) / detached_step_multiplier
    else:
        raise ValueError(solver_mode)
    scaled = mapping / (cfg.T / cfg.n)
    rms_square = scaled.square().mean()
    epsilon = torch.finfo(scaled.dtype).tiny
    p_mean_square = (
        scaled.abs().pow(p).mean() + epsilon
    ).pow(2.0 / p)
    loss = rms_square + high_p_weight * p_mean_square
    return loss, {
        "training_residual_rms_square": float(rms_square.detach()),
        "training_residual_pmean_square": float(p_mean_square.detach()),
        "training_loss": float(loss.detach()),
        "solver_mode": solver_mode,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def inference_benchmark(
    model: nn.Module,
    normalized_t: torch.Tensor,
    warmup: int,
    repeats: int,
) -> dict[str, float | int]:
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(normalized_t)
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            model(normalized_t)
            samples.append(1000.0 * (time.perf_counter() - started))
    values = np.asarray(samples, dtype=np.float64)
    return {
        "warmup_forwards": warmup,
        "timed_forwards": repeats,
        "mean_milliseconds": float(values.mean()),
        "median_milliseconds": float(np.median(values)),
        "p95_milliseconds": float(np.quantile(values, 0.95)),
        "min_milliseconds": float(values.min()),
        "max_milliseconds": float(values.max()),
    }


def set_train_scope(model: LinearRawBoxProjection, scope: str) -> list[nn.Parameter]:
    """Enable either the stable affine-head subspace or all parameters."""

    if scope not in {"head", "all"}:
        raise ValueError(scope)
    for parameter in model.parameters():
        parameter.requires_grad_(scope == "all")
    if scope == "head":
        for parameter in model.base.output.parameters():
            parameter.requires_grad_(True)
    selected = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not selected:
        raise RuntimeError("train scope selected no parameters")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=731)
    parser.add_argument(
        "--initialization-method",
        choices=("hidden_ols", "logit_affine"),
        default="hidden_ols",
    )
    parser.add_argument("--adam-epochs", type=int, default=60)
    parser.add_argument("--adam-learning-rate", type=float, default=2.0e-6)
    parser.add_argument("--adam-final-learning-rate", type=float, default=2.0e-7)
    parser.add_argument("--adam-eval-every", type=int, default=10)
    parser.add_argument(
        "--adam-train-scope",
        choices=("head", "all"),
        default="head",
        help=(
            "The calibrated OLS head can amplify tiny backbone moves; head-only "
            "is therefore the safe default."
        ),
    )
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--lbfgs-outer-steps", type=int, default=12)
    parser.add_argument("--lbfgs-inner-iterations", type=int, default=5)
    parser.add_argument("--lbfgs-learning-rate", type=float, default=0.3)
    parser.add_argument("--lbfgs-history-size", type=int, default=30)
    parser.add_argument(
        "--lbfgs-train-scope", choices=("head", "all"), default="head"
    )
    parser.add_argument("--p", type=float, default=16.0)
    parser.add_argument("--high-p-weight", type=float, default=1.0)
    parser.add_argument(
        "--solver-mode",
        choices=("exact", "detached_fixed_point"),
        default="exact",
    )
    parser.add_argument(
        "--parameter-jacobian-mode",
        choices=("exact_hard_clamp", "semismooth_identity"),
        default="exact_hard_clamp",
        help=(
            "The legacy-named semismooth option keeps exact hard-box forward "
            "values but uses an identity straight-through derivative to release "
            "incorrectly saturated raw-control nodes during parameter optimization."
        ),
    )
    parser.add_argument("--detached-step-multiplier", type=float, default=20.0)
    parser.add_argument("--width-tolerance", type=float, default=0.005)
    parser.add_argument(
        "--variation-fraction-tolerance", type=float, default=0.10
    )
    parser.add_argument("--minimum-upper-fraction", type=float, default=0.75)
    parser.add_argument(
        "--selection-guard",
        choices=("source-shape", "residual-only"),
        default="source-shape",
        help="Use residual-only when changed objective weights alter the control topology.",
    )
    parser.add_argument("--benchmark-warmup", type=int, default=20)
    parser.add_argument("--benchmark-repeats", type=int, default=100)
    args = parser.parse_args()

    if args.p <= 2.0:
        raise ValueError("p must exceed 2")
    if args.detached_step_multiplier <= 0.0:
        raise ValueError("detached-step-multiplier must be positive")
    if not 0.0 <= args.minimum_upper_fraction <= 1.0:
        raise ValueError("minimum-upper-fraction must lie in [0, 1]")
    set_seed(args.seed)
    torch.set_num_threads(1)
    checkpoint_path = resolve(args.checkpoint)
    out_dir = resolve(args.out_dir)
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True)
    prohibited = ("direct", "manual", "supervised", "distill")
    if any(token in str(checkpoint_path).lower() for token in prohibited):
        raise ValueError(f"prohibited teacher-like checkpoint: {checkpoint_path}")

    source = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if source.get("teacher_free") is not True:
        raise ValueError("source checkpoint lacks teacher_free provenance")
    if source.get("direct_or_manual_solution_used") is not False:
        raise ValueError("source checkpoint does not deny direct/manual use")
    if source.get("objective_value_used_as_loss_or_selection") is not False:
        raise ValueError("source checkpoint does not deny objective loss/selection")
    if source.get("switching_time_or_mask_used") is not False:
        raise ValueError("source checkpoint does not deny switch-time/mask use")
    cfg = ProblemConfig(**source["problem"])
    if cfg.n != 800:
        raise ValueError(f"expected n=800, received n={cfg.n}")

    normalized_t = torch.linspace(0.0, 1.0, cfg.n + 1, dtype=torch.float64)
    physical_t = np.linspace(0.0, cfg.T, cfg.n + 1)
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    source_wrapper_class = str(source.get("wrapper", {}).get("class", ""))
    if source_wrapper_class == "FixedBoxProjection":
        source_model = source_model_from_payload(source, cfg)
        model, initialization = initialize_linear_model(
            source_model, source, cfg, normalized_t, args.initialization_method
        )
    elif source_wrapper_class == "LinearRawBoxProjection":
        source_model = linear_model_from_payload(source, cfg)
        model = linear_model_from_payload(source, cfg)
        with torch.no_grad():
            raw = model.raw_control(normalized_t)
        initialization = {
            "initialization_method": "identity_from_linear_raw_source",
            "least_squares_control_rmse": 0.0,
            "least_squares_control_linf": 0.0,
            "least_squares_raw_min": float(raw.min()),
            "least_squares_raw_max": float(raw.max()),
        }
    else:
        raise ValueError(f"unsupported source wrapper: {source_wrapper_class!r}")

    if args.parameter_jacobian_mode == "semismooth_identity":
        semismooth_model = SemismoothLinearRawBoxProjection(
            model.base, cfg.umax
        ).to(dtype=torch.float64)
        with torch.no_grad():
            hard_values = model(normalized_t)
            semismooth_values = semismooth_model(normalized_t)
        if not torch.equal(hard_values, semismooth_values):
            raise RuntimeError(
                "semismooth wrapper changed the hard-projection forward values"
            )
        model = semismooth_model
    source_row, source_u, source_residual = residual_metrics(
        source_model, normalized_t, physical_t, cfg, params
    )
    source_row.update(
        {
            "phase": "source_reference",
            "step": -1,
            "shape_guard_pass": True,
            "elapsed_seconds": 0.0,
        }
    )
    initial_row, initial_u, initial_residual = residual_metrics(
        model, normalized_t, physical_t, cfg, params
    )
    initial_row.update(
        {
            "phase": "linear_initialization",
            "step": 0,
            "shape_guard_pass": passes_selection_guard(
                initial_row, source_row, args
            ),
            "elapsed_seconds": 0.0,
        }
    )
    history: list[dict[str, Any]] = [dict(source_row), dict(initial_row)]
    linear_candidates: list[
        tuple[dict[str, Any], dict[str, torch.Tensor], np.ndarray, np.ndarray]
    ] = []
    if initial_row["shape_guard_pass"]:
        linear_candidates.append(
            (dict(initial_row), clone_state(model), initial_u, initial_residual)
        )
    started = time.perf_counter()

    print(
        f"[source] PGinf={source_row['projected_gradient_linf']:.3e} "
        f"PGrms={source_row['projected_gradient_rms']:.3e} "
        f"upper={source_row['exact_upper_bound_count']}",
        flush=True,
    )
    print(
        f"[linear-init] PGinf={initial_row['projected_gradient_linf']:.3e} "
        f"PGrms={initial_row['projected_gradient_rms']:.3e} "
        f"upper={initial_row['exact_upper_bound_count']} "
        f"guard={initial_row['shape_guard_pass']}",
        flush=True,
    )

    if args.adam_epochs > 0:
        adam_parameters = set_train_scope(model, args.adam_train_scope)
        optimizer = torch.optim.AdamW(
            adam_parameters,
            lr=args.adam_learning_rate,
            weight_decay=0.0,
            eps=1.0e-12,
        )
        for epoch in range(1, args.adam_epochs + 1):
            fraction = epoch / max(args.adam_epochs, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * fraction))
            learning_rate = args.adam_final_learning_rate + cosine * (
                args.adam_learning_rate - args.adam_final_learning_rate
            )
            optimizer.param_groups[0]["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            loss, pieces = residual_loss(
                model,
                normalized_t,
                cfg,
                params,
                args.p,
                args.high_p_weight,
                args.solver_mode,
                args.detached_step_multiplier,
            )
            loss.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(adam_parameters, args.grad_clip)
            )
            optimizer.step()
            history.append(
                {
                    "phase": "adam_train",
                    "step": epoch,
                    "learning_rate": learning_rate,
                    "parameter_gradient_norm_before_clip": gradient_norm,
                    "elapsed_seconds": time.perf_counter() - started,
                    **pieces,
                }
            )
            if epoch % args.adam_eval_every == 0 or epoch == args.adam_epochs:
                row, control, residual = residual_metrics(
                    model, normalized_t, physical_t, cfg, params
                )
                row.update(
                    {
                        "phase": "adam_evaluation",
                        "step": epoch,
                        "learning_rate": learning_rate,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                )
                row["shape_guard_pass"] = passes_selection_guard(
                    row, source_row, args
                )
                history.append(dict(row))
                if row["shape_guard_pass"]:
                    linear_candidates.append(
                        (dict(row), clone_state(model), control, residual)
                    )
                print(
                    f"[adam={epoch:03d}] "
                    f"PGinf={row['projected_gradient_linf']:.3e} "
                    f"PGrms={row['projected_gradient_rms']:.3e} "
                    f"width={row['early_width_10_90']:.5f}/"
                    f"{row['late_width_10_90']:.5f} "
                    f"upper={row['exact_upper_bound_count']} "
                    f"guard={row['shape_guard_pass']}",
                    flush=True,
                )

    if not linear_candidates:
        raise RuntimeError("no linear-head candidate passed the checkpoint selection guard")
    best_before_lbfgs = min(
        linear_candidates,
        key=lambda item: (
            item[0]["projected_gradient_linf"],
            item[0]["projected_gradient_rms"],
        ),
    )
    model.load_state_dict(best_before_lbfgs[1])

    closure_calls = 0
    if args.lbfgs_outer_steps > 0:
        lbfgs_parameters = set_train_scope(model, args.lbfgs_train_scope)
        optimizer = torch.optim.LBFGS(
            lbfgs_parameters,
            lr=args.lbfgs_learning_rate,
            max_iter=args.lbfgs_inner_iterations,
            max_eval=max(
                args.lbfgs_inner_iterations * 2,
                args.lbfgs_inner_iterations + 2,
            ),
            tolerance_grad=1.0e-12,
            tolerance_change=1.0e-15,
            history_size=args.lbfgs_history_size,
            line_search_fn="strong_wolfe",
        )
        for outer_step in range(1, args.lbfgs_outer_steps + 1):

            def closure() -> torch.Tensor:
                nonlocal closure_calls
                optimizer.zero_grad(set_to_none=True)
                loss, _ = residual_loss(
                    model,
                    normalized_t,
                    cfg,
                    params,
                    args.p,
                    args.high_p_weight,
                    args.solver_mode,
                    args.detached_step_multiplier,
                )
                loss.backward()
                closure_calls += 1
                return loss

            reported_loss = float(optimizer.step(closure).detach())
            row, control, residual = residual_metrics(
                model, normalized_t, physical_t, cfg, params
            )
            row.update(
                {
                    "phase": "lbfgs_evaluation",
                    "step": outer_step,
                    "lbfgs_reported_loss_before_step": reported_loss,
                    "closure_calls": closure_calls,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            row["shape_guard_pass"] = passes_selection_guard(
                row, source_row, args
            )
            history.append(dict(row))
            if row["shape_guard_pass"]:
                linear_candidates.append(
                    (dict(row), clone_state(model), control, residual)
                )
            print(
                f"[lbfgs={outer_step:02d}] "
                f"PGinf={row['projected_gradient_linf']:.3e} "
                f"PGrms={row['projected_gradient_rms']:.3e} "
                f"width={row['early_width_10_90']:.5f}/"
                f"{row['late_width_10_90']:.5f} "
                f"upper={row['exact_upper_bound_count']} "
                f"guard={row['shape_guard_pass']} closures={closure_calls}",
                flush=True,
            )

    best_linear_row, best_linear_state, best_linear_u, best_linear_residual = min(
        linear_candidates,
        key=lambda item: (
            item[0]["projected_gradient_linf"],
            item[0]["projected_gradient_rms"],
        ),
    )
    model.load_state_dict(best_linear_state)
    verify_row, verify_u, verify_residual = residual_metrics(
        model, normalized_t, physical_t, cfg, params
    )
    if np.max(np.abs(verify_u - best_linear_u)) > 1.0e-12:
        raise RuntimeError("best linear checkpoint reload mismatch")
    if np.max(np.abs(verify_residual - best_linear_residual)) > 1.0e-12:
        raise RuntimeError("best linear residual reload mismatch")

    hard_verify_model = LinearRawBoxProjection(
        build_model(dict(source["base_model_args"]), cfg).to(dtype=torch.float64),
        cfg.umax,
    ).to(dtype=torch.float64)
    hard_verify_model.load_state_dict(best_linear_state, strict=True)
    hard_verify_row, hard_verify_u, hard_verify_residual = residual_metrics(
        hard_verify_model, normalized_t, physical_t, cfg, params
    )
    if np.max(np.abs(hard_verify_u - verify_u)) > 1.0e-12:
        raise RuntimeError("hard-wrapper control differs after semismooth training")
    if np.max(np.abs(hard_verify_residual - verify_residual)) > 1.0e-12:
        raise RuntimeError("hard-wrapper residual differs after semismooth training")
    verify_row = hard_verify_row
    verify_u = hard_verify_u
    verify_residual = hard_verify_residual

    # Freeze selection before computing any diagnostic objective value.
    source_key = (
        source_row["projected_gradient_linf"],
        source_row["projected_gradient_rms"],
    )
    linear_key = (
        verify_row["projected_gradient_linf"],
        verify_row["projected_gradient_rms"],
    )
    overall_variant = "linear_raw_box_projection" if linear_key < source_key else "source"
    selection_frozen_at = time.time()

    linear_final_metrics, linear_final_u = evaluate(
        model, cfg, normalized_t, params, high_accuracy=True
    )
    source_final_metrics, _ = evaluate(
        source_model, cfg, normalized_t, params, high_accuracy=True
    )
    benchmark = inference_benchmark(
        model,
        normalized_t,
        args.benchmark_warmup,
        args.benchmark_repeats,
    )

    linear_payload = {
        "model_state": best_linear_state,
        "base_model_args": source["base_model_args"],
        "problem": source["problem"],
        "wrapper": {
            "class": "LinearRawBoxProjection",
            "forward": "clamp(raw_affine_transformer_head, 0, umax)",
            "contains_sigmoid": False,
            "contains_temperature": False,
            "contains_switch_parameter_or_mask": False,
        },
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "teacher_free": True,
        "method": (
            "pure Transformer affine raw-control head plus exact piecewise-linear "
            "box projection; complete projected-gradient residual only"
        ),
        "objective_value_used_as_loss_or_selection": False,
        "direct_or_manual_solution_used": False,
        "switching_time_or_mask_used": False,
        "initialization_only_uses_source_teacher_free_preprojection": True,
        "initialization_method": initialization["initialization_method"],
        "parameter_optimization_jacobian": args.parameter_jacobian_mode,
        "selected_checkpoint_forward_wrapper": "literal_hard_box_projection",
        "selected_phase": best_linear_row["phase"],
        "selected_step": int(best_linear_row["step"]),
        "selection_residual_metrics": verify_row,
        "post_selection_diagnostics": linear_final_metrics,
        "inference_benchmark": benchmark,
    }
    torch.save(linear_payload, out_dir / "best_linear_checkpoint.pt")

    if overall_variant == "linear_raw_box_projection":
        selected_payload = dict(linear_payload)
    else:
        selected_payload = dict(source)
        selected_payload["ablation_selection"] = {
            "selected_variant": "source",
            "reason": "source lexicographically beat the best guarded linear candidate",
            "selection_frozen_unix_time": selection_frozen_at,
            "best_linear_residual_metrics": verify_row,
        }
    torch.save(selected_payload, out_dir / "selected_checkpoint.pt")
    selected_u = linear_final_u if overall_variant == "linear_raw_box_projection" else source_u
    selected_residual = (
        verify_residual
        if overall_variant == "linear_raw_box_projection"
        else source_residual
    )
    np.savez_compressed(
        out_dir / "selected_solution.npz",
        t=physical_t,
        u=selected_u,
        projected_gradient_mapping=selected_residual,
    )
    np.savez_compressed(
        out_dir / "best_linear_solution.npz",
        t=physical_t,
        u=linear_final_u,
        projected_gradient_mapping=verify_residual,
    )
    np.savez_compressed(
        out_dir / "source_candidate_solution.npz",
        t=physical_t,
        u=source_u,
        projected_gradient_mapping=source_residual,
    )
    write_csv(out_dir / "history.csv", history)
    make_plot(out_dir / "linear_teacher_free_control", physical_t, linear_final_u)

    provenance = {
        "teacher_free": True,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "source_is_explicit_selection_candidate": True,
        "training_reads_direct_or_manual_solution": False,
        "training_uses_switching_time_or_mask": False,
        "training_uses_objective_value_as_loss_or_selection": False,
        "full_gradient_includes_state_dependence": True,
        "initialization_transfer": (
            "identity load from an already hard-box LinearRawBoxProjection "
            "checkpoint; no calibration or fitted target was used"
            if source_wrapper_class == "LinearRawBoxProjection"
            else (
                f"{args.initialization_method} calibration from the frozen source "
                "Transformer to the same teacher-free source pre-projection "
                "control; used once before residual-only optimization"
            )
        ),
        "linear_forward": "clamp(affine_head(transformer_hidden(t)), 0, umax)",
        "linear_forward_is_single_transformer_forward": True,
        "linear_forward_has_no_sigmoid_or_temperature": True,
        "parameter_optimization_jacobian": args.parameter_jacobian_mode,
        "selected_checkpoint_reverified_with_literal_hard_projection": True,
        "selection": (
            "lexicographic projected-gradient Linf then RMS among finite linear "
            "candidates and the unmodified source candidate"
            if args.selection_guard == "residual-only"
            else "lexicographic projected-gradient Linf then RMS among guarded linear "
            "candidates and the unmodified source candidate"
        ),
        "selection_frozen_before_objective_diagnostics": True,
        "selection_frozen_unix_time": selection_frozen_at,
        "arguments": vars(args),
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "completed",
        "goal_projected_gradient_linf": 1.0e-5,
        "goal_met_by_linear_head": bool(
            verify_row["projected_gradient_linf"] < 1.0e-5
        ),
        "overall_selected_variant": overall_variant,
        "selection_frozen_before_objective_diagnostics": True,
        "initialization": initialization,
        "source_residual_metrics": source_row,
        "best_linear_selected_phase": best_linear_row["phase"],
        "best_linear_selected_step": int(best_linear_row["step"]),
        "best_linear_residual_metrics": verify_row,
        "best_linear_post_selection_diagnostics": linear_final_metrics,
        "source_post_selection_diagnostics": source_final_metrics,
        "inference_benchmark": benchmark,
        "closure_calls": closure_calls,
        "wall_seconds": time.perf_counter() - started,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "COMPLETED.json").write_text(
        json.dumps(
            {
                "completed_unix_time": time.time(),
                "direct_solution_read_during_training": False,
                "selection_frozen_before_objective_diagnostics": True,
                "overall_selected_variant": overall_variant,
                "best_linear_checkpoint_sha256": sha256(
                    out_dir / "best_linear_checkpoint.pt"
                ),
                "selected_checkpoint_sha256": sha256(
                    out_dir / "selected_checkpoint.pt"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
