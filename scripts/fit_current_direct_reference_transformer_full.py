#!/usr/bin/env python3
"""Full-network direct-reference fit for the current n=800 objective.

This reproduces the *kind* of direct-assisted pipeline used by the earlier
strong diagnostic: all Transformer parameters are supervised on the direct
schedule, followed by an optional projected-KKT continuation.  The current
objective ends at the lower control bound, so the boundary projection is
two-sided:

    u(t) = clamp(scale * u_base(t) - offset, 0, umax).

The result is a diagnostic method-choice endpoint, not a teacher-free method.
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

from scripts.continue_teacher_free_linear_box_projection import (  # noqa: E402
    clone_state,
    source_model_from_payload,
)
from scripts.fit_current_direct_reference_transformer import falling_width  # noqa: E402
from scripts.fixed_nominal_opt_gap import (  # noqa: E402
    TimeOnlyPolicyAdapter,
    evaluate_fixed_nominal_der,
    metric_metadata,
)
from scripts.refine_time_only_singular_plateau import rk4_reduced_objective  # noqa: E402
from scripts.train_teacher_free_resolution_curriculum import evaluate  # noqa: E402
from train_paper_pmp_kkt import ProblemConfig, build_params, set_seed  # noqa: E402


DEFAULT_SOURCE = (
    ROOT
    / "outputs/new_objective_alpha1_grid_20260721_v2/a1_b40_g8000/"
    "time_only/curriculum/stage_3_n800/selected_checkpoint.pt"
)
DEFAULT_DIRECT = (
    ROOT
    / "outputs/new_objective_alpha1_grid_20260721_v2/a1_b40_g8000/"
    "direct_nominal/n800_strict_final/scale_1_direct_solution.npz"
)
DEFAULT_OUT = (
    ROOT
    / "outputs/time_only_refinement_choice_20260726/"
    "direct_reference_full_fit_v1"
)


class AffineBoundaryProjectedControl(nn.Module):
    """Transformer control with exact access to both box boundaries."""

    def __init__(
        self,
        base: nn.Module,
        *,
        umax: float,
        scale: float,
        offset: float,
    ) -> None:
        super().__init__()
        self.base = base
        self.umax = float(umax)
        self.register_buffer(
            "scale", torch.tensor(float(scale), dtype=torch.float64)
        )
        self.register_buffer(
            "offset", torch.tensor(float(offset), dtype=torch.float64)
        )

    def forward(self, normalized_t: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            self.scale * self.base(normalized_t) - self.offset,
            0.0,
            self.umax,
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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_json(value: Any) -> Any:
    """Replace non-finite diagnostic floats with JSON null."""

    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return value


def switch_windows(
    physical_t: np.ndarray,
    target: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    interval_t = physical_t[:-1]
    differences = np.diff(target[:-1])
    centers: list[float] = []
    for candidates in (
        np.flatnonzero(interval_t[:-1] < 0.5 * interval_t[-1]),
        np.flatnonzero(interval_t[:-1] >= 0.5 * interval_t[-1]),
    ):
        index = int(candidates[np.argmin(differences[candidates])])
        centers.append(float(interval_t[index + 1]))
    point_mask = np.zeros_like(physical_t, dtype=bool)
    difference_mask = np.zeros(physical_t.size - 1, dtype=bool)
    for center in centers:
        point_mask |= np.abs(physical_t - center) <= radius
        difference_mask |= np.abs(physical_t[1:] - center) <= radius
    return point_mask, difference_mask, centers


def direct_fit_metrics(
    model: nn.Module,
    normalized_t: torch.Tensor,
    target: torch.Tensor,
) -> tuple[float, float, np.ndarray]:
    model.eval()
    # Keep the same Transformer execution path used by the differentiable
    # objective evaluation.  CUDA inference mode may select a fused attention
    # kernel whose roundoff differs noticeably from the gradient-enabled path.
    with torch.enable_grad():
        control = model(normalized_t)
        difference = control - target
        rmse = float(difference.detach().square().mean().sqrt())
        linf = float(difference.detach().abs().max())
        control_numpy = control.detach().cpu().numpy().copy()
    model.train()
    return rmse, linf, control_numpy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", default=str(DEFAULT_SOURCE))
    parser.add_argument("--direct-solution", default=str(DEFAULT_DIRECT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=2607)
    parser.add_argument("--scale", type=float, default=1.04)
    parser.add_argument("--offset", type=float, default=0.06)
    parser.add_argument("--supervised-epochs", type=int, default=4000)
    parser.add_argument("--supervised-learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--supervised-lbfgs-iterations", type=int, default=0)
    parser.add_argument("--supervised-lbfgs-history-size", type=int, default=100)
    parser.add_argument("--global-point-weight", type=float, default=1.0e5)
    parser.add_argument("--switch-point-weight", type=float, default=30.0)
    parser.add_argument("--switch-difference-weight", type=float, default=50.0)
    parser.add_argument("--difference-loss-weight", type=float, default=10.0)
    parser.add_argument("--switch-radius", type=float, default=0.35)
    parser.add_argument("--continuation-epochs", type=int, default=30)
    parser.add_argument("--continuation-learning-rate", type=float, default=3.0e-7)
    parser.add_argument("--anchor-weight", type=float, default=1.0e5)
    parser.add_argument(
        "--continuation-switch-difference-weight", type=float, default=2.0e3
    )
    parser.add_argument("--projected-linf-weight", type=float, default=100.0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument(
        "--fixed-validation-every",
        type=int,
        default=50,
        help=(
            "evaluate the common nominal DER optimality gap every this many "
            "AdamW updates; 0 disables the trace"
        ),
    )
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--direct-rmse-guard-factor", type=float, default=1.25)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device. 'auto' selects CUDA when available.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cpu":
        torch.set_num_threads(1)
    source_path = resolve(args.source_checkpoint)
    direct_path = resolve(args.direct_solution)
    out_dir = resolve(args.out_dir)
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True)

    source = torch.load(source_path, map_location="cpu", weights_only=False)
    if source.get("wrapper", {}).get("class") != "FixedBoxProjection":
        raise ValueError("source must be the current FixedBoxProjection checkpoint")
    cfg = ProblemConfig(**source["problem"])
    if cfg.n != 800:
        raise ValueError(f"expected n=800, received {cfg.n}")
    source_model = source_model_from_payload(source, cfg).to(device=device)
    model = AffineBoundaryProjectedControl(
        source_model.base,
        umax=cfg.umax,
        scale=args.scale,
        offset=args.offset,
    ).to(device=device, dtype=torch.float64)
    normalized_t = torch.linspace(
        0.0, 1.0, cfg.n + 1, device=device, dtype=torch.float64
    )
    physical_t = np.linspace(0.0, cfg.T, cfg.n + 1)
    params = build_params(cfg, device, torch.float64)
    fixed_validation_model = TimeOnlyPolicyAdapter(model, cfg)

    direct = np.load(direct_path)
    direct_t = np.asarray(direct["t"], dtype=np.float64)
    direct_u = np.asarray(direct["u"], dtype=np.float64)
    if not np.array_equal(direct_t, physical_t):
        raise ValueError("direct and Transformer time grids do not match")
    target = torch.tensor(direct_u, device=device, dtype=torch.float64)
    point_mask, difference_mask, switch_centers = switch_windows(
        physical_t, direct_u, args.switch_radius
    )
    point_weights = torch.ones_like(target)
    point_weights[
        torch.tensor(point_mask, device=device, dtype=torch.bool)
    ] = args.switch_point_weight
    difference_weights = torch.ones(
        cfg.n, device=device, dtype=torch.float64
    )
    difference_weights[
        torch.tensor(difference_mask, device=device, dtype=torch.bool)
    ] = (
        args.switch_difference_weight
    )

    history: list[dict[str, Any]] = []
    fixed_validation_history: list[dict[str, Any]] = []
    candidates: list[
        tuple[float, float, dict[str, torch.Tensor], np.ndarray, str, int]
    ] = []
    started = time.perf_counter()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.supervised_learning_rate,
        weight_decay=0.0,
    )

    def record_fixed_validation(epoch: int) -> None:
        if args.fixed_validation_every <= 0:
            return
        fixed_validation_history.append(
            {
                "phase": "time_only_initialization",
                "phase_step": epoch,
                "optimizer_step": epoch,
                **evaluate_fixed_nominal_der(
                    fixed_validation_model,
                    cfg,
                    params,
                    state_mode="w_zero",
                ),
            }
        )

    record_fixed_validation(0)

    def supervised_objective() -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        prediction = model(normalized_t)
        point_error = (prediction - target) / cfg.umax
        point_loss = (
            point_weights * point_error.square()
        ).sum() / point_weights.sum()
        difference_error = (
            torch.diff(prediction) - torch.diff(target)
        ) / cfg.umax
        difference_loss = (
            difference_weights * difference_error.square()
        ).sum() / difference_weights.sum()
        global_point_loss = point_error.square().mean()
        loss = (
            point_loss
            + args.difference_loss_weight * difference_loss
            + args.global_point_weight * global_point_loss
        )
        return loss, point_loss, difference_loss, global_point_loss

    for epoch in range(1, args.supervised_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, point_loss, difference_loss, global_point_loss = (
            supervised_objective()
        )
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        )
        optimizer.step()
        if args.fixed_validation_every > 0 and (
            epoch % args.fixed_validation_every == 0
            or epoch == args.supervised_epochs
        ):
            record_fixed_validation(epoch)
        history.append(
            {
                "phase": "full_network_direct_supervision",
                "epoch": epoch,
                "loss": float(loss.detach()),
                "point_loss": float(point_loss.detach()),
                "difference_loss": float(difference_loss.detach()),
                "global_point_loss": float(global_point_loss.detach()),
                "gradient_norm_before_clip": gradient_norm,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        if epoch == 1 or epoch % 100 == 0 or epoch == args.supervised_epochs:
            rmse, linf, control = direct_fit_metrics(
                model, normalized_t, target
            )
            candidates.append(
                (
                    rmse,
                    linf,
                    clone_state(model),
                    control,
                    "supervised",
                    epoch,
                )
            )
            print(
                f"[supervised {epoch:04d}] RMSE={rmse:.3e} "
                f"Linf={linf:.3e}",
                flush=True,
            )
    if args.supervised_lbfgs_iterations > 0:
        best_before_lbfgs = min(
            candidates, key=lambda item: (item[0], item[1])
        )
        model.load_state_dict(best_before_lbfgs[2], strict=True)
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=args.supervised_lbfgs_iterations,
            history_size=args.supervised_lbfgs_history_size,
            tolerance_grad=1.0e-12,
            tolerance_change=1.0e-15,
            line_search_fn="strong_wolfe",
        )
        closure_calls = 0

        def closure() -> torch.Tensor:
            nonlocal closure_calls
            lbfgs.zero_grad(set_to_none=True)
            objective, _, _, _ = supervised_objective()
            objective.backward()
            closure_calls += 1
            return objective

        lbfgs.step(closure)
        rmse, linf, control = direct_fit_metrics(
            model, normalized_t, target
        )
        candidates.append(
            (
                rmse,
                linf,
                clone_state(model),
                control,
                "supervised_lbfgs",
                args.supervised_lbfgs_iterations,
            )
        )
        with torch.enable_grad():
            loss, point_loss, difference_loss, global_point_loss = (
                supervised_objective()
            )
        history.append(
            {
                "phase": "full_network_direct_supervision_lbfgs",
                "epoch": args.supervised_lbfgs_iterations,
                "loss": float(loss.detach()),
                "point_loss": float(point_loss.detach()),
                "difference_loss": float(difference_loss.detach()),
                "global_point_loss": float(global_point_loss.detach()),
                "gradient_norm_before_clip": None,
                "closure_calls": closure_calls,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        print(
            f"[supervised LBFGS {args.supervised_lbfgs_iterations:04d}] "
            f"RMSE={rmse:.3e} Linf={linf:.3e} "
            f"closures={closure_calls}",
            flush=True,
        )

    best_supervised = min(candidates, key=lambda item: (item[0], item[1]))
    model.load_state_dict(best_supervised[2], strict=True)
    supervised_control = best_supervised[3].copy()
    supervised_rmse = best_supervised[0]
    anchor = torch.tensor(
        supervised_control, device=device, dtype=torch.float64
    )

    continuation_candidates: list[
        tuple[
            float,
            float,
            float,
            float,
            dict[str, torch.Tensor],
            np.ndarray,
            int,
        ]
    ] = []

    def record_continuation(epoch: int) -> None:
        rmse, linf, control = direct_fit_metrics(model, normalized_t, target)
        metrics, rechecked = evaluate(
            model, cfg, normalized_t, params, high_accuracy=False
        )
        reload_tolerance = 5.0e-7 if device.type == "cuda" else 5.0e-9
        if np.max(np.abs(control - rechecked)) > reload_tolerance:
            raise RuntimeError("evaluation changed the saved forward values")
        early = falling_width(physical_t, control, "early")
        late = falling_width(physical_t, control, "late")
        continuation_candidates.append(
            (
                float(metrics["projected_gradient_linf"]),
                float(metrics["projected_gradient_rms"]),
                rmse,
                linf,
                clone_state(model),
                control,
                epoch,
            )
        )
        print(
            f"[continuation {epoch:03d}] PGinf="
            f"{metrics['projected_gradient_linf']:.3e} "
            f"RMSE={rmse:.3e} width={early:.5f}/{late:.5f}",
            flush=True,
        )

    record_continuation(0)
    if args.continuation_epochs > 0:
        continuation_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.continuation_learning_rate,
            weight_decay=0.0,
        )
        dt = cfg.T / cfg.n
        switch_difference_mask = torch.tensor(
            difference_mask, device=device, dtype=torch.float64
        )
        for epoch in range(1, args.continuation_epochs + 1):
            continuation_optimizer.zero_grad(set_to_none=True)
            control = model(normalized_t)
            interval_control = control[:-1]
            objective = rk4_reduced_objective(interval_control, cfg, params)
            gradient = torch.autograd.grad(
                objective,
                interval_control,
                create_graph=True,
            )[0]
            mapping = interval_control - torch.clamp(
                interval_control - gradient, 0.0, cfg.umax
            )
            projected_loss = (mapping / dt).square().mean()
            projected_linf_loss = (mapping.abs().max() / dt).square()
            anchor_loss = ((control - anchor) / cfg.umax).square().mean()
            difference_error = (
                torch.diff(control) - torch.diff(anchor)
            ) / cfg.umax
            switch_difference_loss = (
                switch_difference_mask * difference_error.square()
            ).sum() / switch_difference_mask.sum().clamp_min(1.0)
            loss = (
                projected_loss
                + args.projected_linf_weight * projected_linf_loss
                + args.anchor_weight * anchor_loss
                + args.continuation_switch_difference_weight
                * switch_difference_loss
            )
            loss.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )
            )
            continuation_optimizer.step()
            history.append(
                {
                    "phase": "anchored_projected_kkt_continuation",
                    "epoch": epoch,
                    "loss": float(loss.detach()),
                    "projected_loss": float(projected_loss.detach()),
                    "projected_linf_loss": float(projected_linf_loss.detach()),
                    "anchor_loss": float(anchor_loss.detach()),
                    "switch_difference_loss": float(
                        switch_difference_loss.detach()
                    ),
                    "gradient_norm_before_clip": gradient_norm,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            if epoch % args.eval_every == 0 or epoch == args.continuation_epochs:
                record_continuation(epoch)

    guarded = [
        item
        for item in continuation_candidates
        if item[2] <= args.direct_rmse_guard_factor * supervised_rmse
    ]
    pool = guarded or continuation_candidates
    selected = min(pool, key=lambda item: (item[0], item[1], item[2]))
    model.load_state_dict(selected[4], strict=True)
    final_metrics, selected_control = evaluate(
        model, cfg, normalized_t, params, high_accuracy=True
    )
    selected_rmse = float(
        np.sqrt(np.mean((selected_control - direct_u) ** 2))
    )
    selected_linf = float(np.max(np.abs(selected_control - direct_u)))
    final_metrics.update(
        {
            "physical_objective": 400.0
            * float(final_metrics["high_accuracy_J_diagnostic_only"]),
            "direct_target_rmse": selected_rmse,
            "direct_target_linf": selected_linf,
            "early_falling_width_10_90": falling_width(
                physical_t, selected_control, "early"
            ),
            "late_falling_width_10_90": falling_width(
                physical_t, selected_control, "late"
            ),
        }
    )

    torch.save(
        {
            "model_state": selected[4],
            "base_model_args": source["base_model_args"],
            "problem": source["problem"],
            "wrapper": {
                "class": "AffineBoundaryProjectedControl",
                "scale": args.scale,
                "offset": args.offset,
                "forward": "clamp(scale * base(t) - offset, 0, umax)",
            },
            "source_checkpoint": str(source_path),
            "direct_reference": str(direct_path),
            "teacher_free": False,
            "direct_or_manual_solution_used": True,
            "method": (
                "full-Transformer direct-reference supervision followed by "
                "anchored projected-KKT continuation"
            ),
            "selected_continuation_epoch": selected[6],
            "metrics": final_metrics,
        },
        out_dir / "selected_checkpoint.pt",
    )
    np.savez(
        out_dir / "solution.npz",
        t=physical_t,
        u=selected_control,
        direct_u=direct_u,
        supervised_u=supervised_control,
    )
    write_csv(out_dir / "history.csv", history)
    if fixed_validation_history:
        write_csv(
            out_dir / "fixed_validation_opt_gap.csv",
            fixed_validation_history,
        )
    summary = {
        "status": "completed",
        "scope": "direct-assisted diagnostic; not teacher-free training",
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": sha256(source_path),
        "direct_reference": str(direct_path),
        "direct_reference_sha256": sha256(direct_path),
        "training_uses_direct_transcription_schedule": True,
        "full_transformer_parameters_trained": True,
        "switch_centers_detected_from_direct_target": switch_centers,
        "training": vars(args),
        "fixed_validation_metric": metric_metadata(),
        "device": str(device),
        "best_supervised": {
            "epoch": best_supervised[5],
            "direct_target_rmse": best_supervised[0],
            "direct_target_linf": best_supervised[1],
        },
        "selected_continuation_epoch": selected[6],
        "metrics": final_metrics,
        "wall_seconds": time.perf_counter() - started,
    }
    serializable_summary = finite_json(summary)
    (out_dir / "summary.json").write_text(
        json.dumps(serializable_summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(serializable_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
