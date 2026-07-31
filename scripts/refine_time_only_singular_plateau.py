#!/usr/bin/env python3
"""Warm-start refinement experiments for the time-only singular plateau.

This script deliberately leaves the canonical training artifacts untouched.  It
loads the selected time-only Transformer weights, starts a fresh optimizer, and
compares controlled alternatives for assigning the singular and boundary KKT
losses.  An optional direct-mesh curriculum is supported, but its weight can be
annealed to zero so the final phase is driven by the optimality conditions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_paper_pmp_kkt import (  # noqa: E402
    ParamControl,
    ProblemConfig,
    TimeCNN,
    TimeMLP,
    TimeTransformer,
    build_params,
    dynamics,
    parse_hidden,
    pmp_kkt_loss,
    set_seed,
)
from tumor_problem import (  # noqa: E402
    TumorProblem,
    evaluate_zoh_control,
    serializable_metrics,
)


DEFAULT_CHECKPOINT = (
    ROOT / "paper_runs/smoothness_weight_sweep/w3/seed_4/best_pmp_kkt.pt"
)
DEFAULT_BASELINE = (
    ROOT / "paper_runs/canonical_results/solutions/direct_time_mesh_n200.npz"
)


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = ROOT / value
    return value.resolve()


def build_model(checkpoint_args: dict[str, Any], cfg: ProblemConfig) -> torch.nn.Module:
    model_name = str(checkpoint_args.get("model", "transformer"))
    if model_name == "transformer":
        return TimeTransformer(
            int(checkpoint_args.get("d_model", 64)),
            int(checkpoint_args.get("heads", 4)),
            int(checkpoint_args.get("layers", 2)),
            cfg.umax,
            float(checkpoint_args.get("init_u", 1.5)),
        )
    if model_name == "mlp":
        return TimeMLP(
            parse_hidden(str(checkpoint_args.get("hidden", "128,128"))),
            cfg.umax,
            float(checkpoint_args.get("init_u", 1.5)),
        )
    if model_name == "cnn":
        return TimeCNN(
            int(checkpoint_args.get("cnn_channels", 96)),
            int(checkpoint_args.get("cnn_layers", 3)),
            int(checkpoint_args.get("cnn_kernel_size", 5)),
            cfg.umax,
            float(checkpoint_args.get("init_u", 1.5)),
        )
    if model_name == "param":
        return ParamControl(
            cfg.n + 1,
            cfg.umax,
            float(checkpoint_args.get("init_u", 1.5)),
        )
    raise ValueError(f"unsupported checkpoint model {model_name!r}")


def conditional_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (weights * values).sum() / weights.sum().clamp_min(1.0e-12)


def rk4_reduced_objective(
    controls: torch.Tensor,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Differentiable RK4 reduced objective for intervalwise-ZOH controls."""

    if controls.numel() != cfg.n:
        raise ValueError(f"expected {cfg.n} controls, got {controls.numel()}")
    dt = cfg.T / cfg.n
    state = params["N0"]
    accumulated = torch.zeros((), dtype=controls.dtype, device=controls.device)
    for index in range(cfg.n):
        control = controls[index]
        k1 = dynamics(state, control, params)
        r1 = (params["beta"] * state).sum() + params["gamma"] * control

        state2 = torch.clamp(state + 0.5 * dt * k1, min=1.0e-10)
        k2 = dynamics(state2, control, params)
        r2 = (params["beta"] * state2).sum() + params["gamma"] * control

        state3 = torch.clamp(state + 0.5 * dt * k2, min=1.0e-10)
        k3 = dynamics(state3, control, params)
        r3 = (params["beta"] * state3).sum() + params["gamma"] * control

        state4 = torch.clamp(state + dt * k3, min=1.0e-10)
        k4 = dynamics(state4, control, params)
        r4 = (params["beta"] * state4).sum() + params["gamma"] * control

        state = torch.clamp(
            state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0,
            min=1.0e-10,
        )
        accumulated = accumulated + dt * (r1 + 2.0 * r2 + 2.0 * r3 + r4) / 6.0
    return (params["alpha"] * state).sum() + accumulated


def transition_width(
    t: np.ndarray,
    u: np.ndarray,
    *,
    early_high: float = 2.7,
    interior: float = 1.4,
) -> tuple[float, float]:
    interval_t = np.asarray(t[:-1], dtype=np.float64)
    interval_u = np.asarray(u[: len(interval_t)], dtype=np.float64)
    early_high_ids = np.flatnonzero(interval_u <= early_high)
    early_interior_ids = np.flatnonzero(interval_u <= interior)
    early = float("nan")
    if early_high_ids.size and early_interior_ids.size:
        early = float(
            interval_t[early_interior_ids[0]] - interval_t[early_high_ids[0]]
        )

    late_region = np.flatnonzero(interval_t >= 0.5 * interval_t[-1])
    late_interior = np.flatnonzero(interval_u[late_region] >= interior)
    late_high = np.flatnonzero(interval_u[late_region] >= early_high)
    late = float("nan")
    if late_interior.size and late_high.size:
        late = float(
            interval_t[late_region[late_high[0]]]
            - interval_t[late_region[late_interior[0]]]
        )
    return early, late


def local_metrics(
    t: np.ndarray,
    u: np.ndarray,
    N: np.ndarray,
    u_sing: np.ndarray,
    *,
    plateau_start: float,
    plateau_end: float,
) -> dict[str, float]:
    state_t = np.asarray(t[: len(N)], dtype=np.float64)
    total = np.asarray(N, dtype=np.float64).sum(axis=1)
    plateau = (state_t >= plateau_start) & (state_t <= plateau_end)
    if not np.any(plateau):
        raise ValueError("plateau metric window contains no state points")
    plateau_values = total[plateau]
    dt = np.diff(state_t)
    slopes = np.diff(total) / dt
    slope_mask = (state_t[:-1] >= plateau_start) & (
        state_t[:-1] < plateau_end
    )
    common = min(len(u), len(u_sing), len(state_t))
    candidate_mask = (state_t[:common] >= plateau_start) & (
        state_t[:common] <= plateau_end
    )
    candidate_error = np.abs(
        np.asarray(u[:common])[candidate_mask]
        - np.asarray(u_sing[:common])[candidate_mask]
    )
    early_width, late_width = transition_width(state_t, u)
    return {
        "plateau_total_min": float(plateau_values.min()),
        "plateau_total_max": float(plateau_values.max()),
        "plateau_total_range": float(np.ptp(plateau_values)),
        "plateau_total_relative_range": float(
            np.ptp(plateau_values) / plateau_values.mean()
        ),
        "plateau_dtotal_rms": float(np.sqrt(np.mean(slopes[slope_mask] ** 2))),
        "plateau_dtotal_max_abs": float(np.max(np.abs(slopes[slope_mask]))),
        "plateau_candidate_mae": float(np.mean(candidate_error)),
        "plateau_candidate_max_abs": float(np.max(candidate_error)),
        "early_transition_width": early_width,
        "late_transition_width": late_width,
    }


def high_accuracy_metrics(
    t: np.ndarray,
    u: np.ndarray,
    *,
    plateau_start: float,
    plateau_end: float,
    problem: TumorProblem | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    result = evaluate_zoh_control(
        t,
        u,
        problem=problem if problem is not None else TumorProblem(),
        diagnostic_points=4001,
    )
    diagnostic_t = np.asarray(result["diagnostic_t"])
    diagnostic_N = np.asarray(result["diagnostic_N"])
    diagnostic_u = np.asarray(result["diagnostic_u"])
    diagnostic_u_singular = np.asarray(result["diagnostic_u_singular"])
    metrics = dict(serializable_metrics(result))
    metrics.update(
        local_metrics(
            diagnostic_t,
            diagnostic_u,
            diagnostic_N,
            diagnostic_u_singular,
            plateau_start=plateau_start,
            plateau_end=plateau_end,
        )
    )
    arrays = {
        "t": diagnostic_t,
        "N": diagnostic_N,
        "u": diagnostic_u,
        "lambda": np.asarray(result["diagnostic_lambda"]),
        "psi": np.asarray(result["diagnostic_psi"]),
        "u_singular": diagnostic_u_singular,
        "singular_weight": np.asarray(result["diagnostic_singular_weight"]),
    }
    return metrics, arrays


def write_history(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    checkpoint_path = resolve(args.checkpoint)
    baseline_path = resolve(args.baseline_solution)
    out_dir = resolve(args.out_dir)
    if out_dir == checkpoint_path.parent or checkpoint_path.is_relative_to(out_dir):
        raise ValueError("refinement output must not contain or overwrite the source checkpoint")
    out_dir.mkdir(parents=True, exist_ok=False)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    problem = checkpoint["problem"]
    cfg = ProblemConfig(**problem)
    checkpoint_args = dict(checkpoint.get("args", {}))
    device = torch.device(args.device)
    dtype = torch.float64 if args.float64 else torch.float32
    params = build_params(cfg, device, dtype)
    normalized_t = torch.linspace(
        0.0, 1.0, cfg.n + 1, device=device, dtype=dtype
    )
    physical_t = cfg.T * normalized_t

    model = build_model(checkpoint_args, cfg).to(device=device, dtype=dtype)
    model.load_state_dict(checkpoint["model_state"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
    )

    baseline_npz = np.load(baseline_path)
    baseline_u = np.asarray(baseline_npz["u"], dtype=np.float64)
    if baseline_u.size < cfg.n:
        raise ValueError(
            f"baseline has {baseline_u.size} controls, expected at least {cfg.n}"
        )
    baseline_tensor = torch.as_tensor(
        baseline_u[: cfg.n], device=device, dtype=dtype
    )

    fixed_mask = (
        (physical_t[:-1] >= args.fixed_start)
        & (physical_t[:-1] <= args.fixed_end)
    ).to(dtype)
    if not bool(torch.any(fixed_mask)) or not bool(torch.any(1.0 - fixed_mask)):
        raise ValueError("fixed mask must contain both singular and non-singular points")

    best_metric = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        u = model(normalized_t)
        pack = pmp_kkt_loss(
            u,
            cfg,
            params,
            args.singular_eps,
            args.singular_tau,
            detach_gate=args.detach_gate,
        )
        l_sing = (u[:-1] - pack["u_sing"][:-1]).square()
        psi_interval = pack["psi"][:-1]
        l_boundary = (
            torch.relu(psi_interval) * u[:-1]
            + torch.relu(-psi_interval) * (params["umax"] - u[:-1])
        ).square()

        if args.mask_mode == "original":
            optimality_loss = pack["opt_gap"]
            singular_conditioned = conditional_mean(
                l_sing, pack["q"][:-1].detach()
            )
            boundary_conditioned = conditional_mean(
                l_boundary, 1.0 - pack["q"][:-1].detach()
            )
        elif args.mask_mode == "dynamic_detached":
            dynamic_mask = pack["q"][:-1].detach()
            singular_conditioned = conditional_mean(l_sing, dynamic_mask)
            boundary_conditioned = conditional_mean(
                l_boundary, 1.0 - dynamic_mask
            )
            optimality_loss = (
                args.singular_weight * singular_conditioned
                + boundary_conditioned
            )
        elif args.mask_mode == "fixed_replace":
            singular_conditioned = conditional_mean(l_sing, fixed_mask)
            boundary_conditioned = conditional_mean(
                l_boundary, 1.0 - fixed_mask
            )
            optimality_loss = (
                args.singular_weight * singular_conditioned
                + boundary_conditioned
            )
        elif args.mask_mode == "fixed_add":
            singular_conditioned = conditional_mean(l_sing, fixed_mask)
            boundary_conditioned = conditional_mean(
                l_boundary, 1.0 - fixed_mask
            )
            optimality_loss = (
                pack["opt_gap"]
                + args.singular_weight * singular_conditioned
            )
        elif args.mask_mode == "rk4_full_gradient":
            # dF_h/du_k is an interval integral.  Division by dt expresses the
            # same residual on the H_u scale and avoids a mesh-dependent loss.
            interval_controls = u[:-1]
            objective_rk4 = rk4_reduced_objective(interval_controls, cfg, params)
            reduced_gradient = torch.autograd.grad(
                objective_rk4, interval_controls, create_graph=True
            )[0]
            scaled_gradient = reduced_gradient / (cfg.T / cfg.n)
            # The sigmoid output keeps all controls strictly inside the box.
            # Therefore the applicable KKT equation is the interior condition
            # dF_h/du_k = 0 at every interval; projection must not hide a large
            # derivative merely because an output happens to lie near a bound.
            optimality_loss = scaled_gradient.square().mean()
            singular_conditioned = conditional_mean(
                l_sing, pack["q"][:-1].detach()
            )
            boundary_conditioned = conditional_mean(
                l_boundary, 1.0 - pack["q"][:-1].detach()
            )
        else:
            raise ValueError(f"unknown mask mode {args.mask_mode}")

        progress = 1.0 if args.epochs <= 1 else (epoch - 1) / (args.epochs - 1)
        baseline_weight = (
            args.baseline_weight_start
            + progress
            * (args.baseline_weight_end - args.baseline_weight_start)
        )
        baseline_mse = (
            (u[:-1] - baseline_tensor).square().mean()
            / (cfg.umax * cfg.umax)
        )
        loss = (
            optimality_loss
            + args.smooth_weight * pack["smooth"]
            + baseline_weight * baseline_mse
        )
        loss.backward()
        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step(float(loss.detach().cpu()))

        with torch.no_grad():
            total = pack["N"].sum(dim=1)
            plateau = (
                (physical_t >= args.plateau_start)
                & (physical_t <= args.plateau_end)
            )
            plateau_values = total[plateau]
            plateau_relative = (
                (plateau_values.max() - plateau_values.min())
                / plateau_values.mean()
            )
            row = {
                "epoch": float(epoch),
                "loss": float(loss.detach().cpu()),
                "optimality_loss": float(optimality_loss.detach().cpu()),
                "original_opt_gap": float(pack["opt_gap"].detach().cpu()),
                "singular_conditioned": float(
                    singular_conditioned.detach().cpu()
                ),
                "boundary_conditioned": float(
                    boundary_conditioned.detach().cpu()
                ),
                "smooth": float(pack["smooth"].detach().cpu()),
                "baseline_weight": float(baseline_weight),
                "baseline_mse_normalized": float(baseline_mse.detach().cpu()),
                "objective_euler": float(pack["objective"].detach().cpu()),
                "plateau_relative_range_euler": float(
                    plateau_relative.detach().cpu()
                ),
                "q_mean": float(pack["q"].mean().detach().cpu()),
                "u_min": float(u.min().detach().cpu()),
                "u_max": float(u.max().detach().cpu()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": float(time.perf_counter() - start_time),
            }
            if args.mask_mode == "rk4_full_gradient":
                row["rk4_reduced_gradient_linf"] = float(
                    reduced_gradient.detach().abs().max().cpu()
                )
                row["rk4_reduced_gradient_rms"] = float(
                    reduced_gradient.detach().square().mean().sqrt().cpu()
                )
            history.append(row)
            metric = row["loss"]
            if metric < best_metric:
                best_metric = metric
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(
                f"[{epoch:04d}] loss={row['loss']:.6g} "
                f"opt={row['optimality_loss']:.6g} "
                f"flat={100.0 * row['plateau_relative_range_euler']:.4f}% "
                f"base={row['baseline_mse_normalized']:.6g} "
                f"u=({row['u_min']:.3f},{row['u_max']:.3f})"
            )

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    training_seconds = float(time.perf_counter() - start_time)
    model.load_state_dict(best_state)
    with torch.no_grad():
        u = model(normalized_t)
        pack = pmp_kkt_loss(
            u,
            cfg,
            params,
            args.singular_eps,
            args.singular_tau,
            detach_gate=args.detach_gate,
        )
    t_numpy = physical_t.detach().cpu().numpy()
    u_numpy = u.detach().cpu().numpy()
    N_numpy = pack["N"].detach().cpu().numpy()
    local = local_metrics(
        t_numpy,
        u_numpy,
        N_numpy,
        pack["u_sing"].detach().cpu().numpy(),
        plateau_start=args.plateau_start,
        plateau_end=args.plateau_end,
    )
    high_accuracy, diagnostic_arrays = high_accuracy_metrics(
        t_numpy,
        u_numpy,
        plateau_start=args.plateau_start,
        plateau_end=args.plateau_end,
    )

    checkpoint_payload = {
        "model_state": best_state,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": best_epoch,
        "best_metric": best_metric,
        "args": vars(args),
        "problem": cfg.__dict__,
        "source_checkpoint": str(checkpoint_path),
        "training_seconds": training_seconds,
    }
    torch.save(checkpoint_payload, out_dir / "best_refined.pt")
    np.savez(
        out_dir / "solution.npz",
        t=t_numpy,
        u=u_numpy,
        N=N_numpy,
        lam=pack["lambda"].detach().cpu().numpy(),
        psi=pack["psi"].detach().cpu().numpy(),
        q=pack["q"].detach().cpu().numpy(),
        u_sing=pack["u_sing"].detach().cpu().numpy(),
    )
    np.savez(out_dir / "high_accuracy_diagnostics.npz", **diagnostic_arrays)
    write_history(out_dir / "history.csv", history)
    summary = {
        "mask_mode": args.mask_mode,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "training_seconds": training_seconds,
        "source_checkpoint": str(checkpoint_path),
        "baseline_solution": str(baseline_path),
        "local_euler_metrics": local,
        "high_accuracy_metrics": high_accuracy,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved refinement experiment to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--baseline-solution", default=str(DEFAULT_BASELINE))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--mask-mode",
        choices=(
            "original",
            "dynamic_detached",
            "fixed_replace",
            "fixed_add",
            "rk4_full_gradient",
        ),
        default="fixed_replace",
    )
    parser.add_argument("--fixed-start", type=float, default=0.7)
    parser.add_argument("--fixed-end", type=float, default=8.8)
    parser.add_argument("--plateau-start", type=float, default=1.0)
    parser.add_argument("--plateau-end", type=float, default=8.5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-patience", type=int, default=80)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--singular-eps", type=float, default=0.1)
    parser.add_argument("--singular-tau", type=float, default=0.03)
    parser.add_argument("--singular-weight", type=float, default=10.0)
    parser.add_argument("--smooth-weight", type=float, default=0.3)
    parser.add_argument("--baseline-weight-start", type=float, default=0.0)
    parser.add_argument("--baseline-weight-end", type=float, default=0.0)
    parser.add_argument("--detach-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--float64", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-every", type=int, default=25)
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.fixed_start >= args.fixed_end:
        parser.error("--fixed-start must be less than --fixed-end")
    if args.plateau_start >= args.plateau_end:
        parser.error("--plateau-start must be less than --plateau-end")
    if min(args.singular_weight, args.smooth_weight) < 0.0:
        parser.error("loss weights must be nonnegative")
    train(args)


if __name__ == "__main__":
    main()
