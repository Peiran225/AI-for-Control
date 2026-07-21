#!/usr/bin/env python3
"""Teacher-free detached projected-fixed-point continuation.

Given a checkpoint produced by ``train_teacher_free_resolution_curriculum.py``,
this script repeatedly forms the full reduced-gradient target

    target = projection_[0,umax](u - step * grad F_h(u))

and fits the same Transformer to that detached target.  The target is
recomputed at every epoch.  This is an optimality-fixed-point iteration, not
direct-solution supervision: no direct control, manual target, switching time,
or switching mask is read, and F_h itself is not used as the training loss.
The positive step multiplier does not change the KKT fixed points.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


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
    clone_state,
    evaluate,
    make_plot,
    write_csv,
)
from train_paper_pmp_kkt import ProblemConfig, build_params, set_seed  # noqa: E402


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=301)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument(
        "--step-multiplier",
        type=float,
        default=80.0,
        help="Positive projected-gradient step; 80 equals 1/dt for n=800.",
    )
    parser.add_argument("--linf-weight", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--learn-temperature", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(1)
    checkpoint_path = resolve(args.checkpoint)
    out_dir = resolve(args.out_dir)
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True)
    lowered = str(checkpoint_path).lower()
    if any(token in lowered for token in ("direct", "manual_target", "supervised", "distill")):
        raise ValueError(f"prohibited teacher-like checkpoint: {checkpoint_path}")
    source = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not source.get("teacher_free", False):
        raise ValueError("source checkpoint is not marked teacher_free")
    cfg = ProblemConfig(**source["problem"])
    if cfg.n != 800:
        raise ValueError(f"continuation expects n=800, received n={cfg.n}")
    wrapper = dict(source["wrapper"])
    base = build_model(dict(source["base_model_args"]), cfg).to(dtype=torch.float64)
    model = FixedBoxProjection(
        base,
        cfg.umax,
        float(wrapper["scale"]),
        temperature=float(wrapper.get("temperature", 1.0)),
        learn_temperature=args.learn_temperature,
    ).to(dtype=torch.float64)
    missing, unexpected = model.load_state_dict(source["model_state"], strict=False)
    acceptable_missing = {"raw_temperature"} if args.learn_temperature else set()
    if set(missing) - acceptable_missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")

    normalized_t = torch.linspace(0.0, 1.0, cfg.n + 1, dtype=torch.float64)
    physical_t = np.linspace(0.0, cfg.T, cfg.n + 1)
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.0
    )
    history: list[dict[str, Any]] = []
    candidates: list[tuple[dict[str, Any], dict[str, torch.Tensor], np.ndarray, int]] = []
    started = time.perf_counter()

    def record(epoch: int) -> None:
        metrics, control = evaluate(
            model, cfg, normalized_t, params, high_accuracy=False
        )
        history.append(
            {
                "epoch": epoch,
                "event": "evaluation",
                **metrics,
                "elapsed_seconds": time.perf_counter() - started,
                "temperature": float(model.temperature_value().detach()),
            }
        )
        if metrics["nondegenerate_high_low_high"]:
            candidates.append((metrics, clone_state(model), control, epoch))
        print(
            f"[e={epoch}] PGinf={metrics['projected_gradient_linf']:.3e} "
            f"PGrms={metrics['projected_gradient_rms']:.3e} "
            f"J(diag)={metrics['rk4_objective_diagnostic_only']:.9f} "
            f"width={metrics['early_width_10_90']:.4f}/"
            f"{metrics['late_width_10_90']:.4f} "
            f"tau={float(model.temperature_value().detach()):.4f}",
            flush=True,
        )

    record(0)
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        all_control = model(normalized_t)
        interval_control = all_control[:-1]
        objective = rk4_reduced_objective(interval_control, cfg, params)
        gradient = torch.autograd.grad(objective, interval_control)[0]
        with torch.no_grad():
            target = torch.clamp(
                interval_control.detach() - args.step_multiplier * gradient.detach(),
                0.0,
                cfg.umax,
            )
        residual = (interval_control - target) / cfg.umax
        rms_loss = residual.square().mean()
        linf_loss = residual.abs().max().square()
        loss = rms_loss + args.linf_weight * linf_loss
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        )
        optimizer.step()
        history.append(
            {
                "epoch": epoch,
                "event": "train",
                "loss": float(loss.detach()),
                "detached_fixed_point_rms_loss": float(rms_loss.detach()),
                "detached_fixed_point_linf_loss": float(linf_loss.detach()),
                "full_gradient_linf_before_step": float(gradient.detach().abs().max()),
                "target_update_linf": float(
                    (target - interval_control.detach()).abs().max()
                ),
                "gradient_norm_before_clip": gradient_norm,
                "temperature": float(model.temperature_value().detach()),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            record(epoch)

    if not candidates:
        raise RuntimeError("no candidate passed the generic nondegeneracy guard")
    selected_metrics, selected_state, selected_control, selected_epoch = min(
        candidates,
        key=lambda item: (
            item[0]["projected_gradient_linf"],
            item[0]["projected_gradient_rms"],
        ),
    )
    model.load_state_dict(selected_state)
    final_metrics, final_control = evaluate(
        model, cfg, normalized_t, params, high_accuracy=True
    )
    if np.max(np.abs(final_control - selected_control)) > 1.0e-12:
        raise RuntimeError("selected checkpoint reload mismatch")
    wall_seconds = time.perf_counter() - started
    torch.save(
        {
            "model_state": selected_state,
            "base_model_args": source["base_model_args"],
            "problem": source["problem"],
            "wrapper": {
                "class": "FixedBoxProjection",
                "scale": float(wrapper["scale"]),
                "temperature": float(model.temperature_value().detach()),
                "learn_temperature": args.learn_temperature,
                "forward": wrapper["forward"],
            },
            "source_checkpoint": str(checkpoint_path),
            "teacher_free": True,
            "method": "detached projected full-gradient fixed-point continuation",
            "selected_epoch": selected_epoch,
            "selection_metrics": final_metrics,
        },
        out_dir / "selected_checkpoint.pt",
    )
    np.savez(out_dir / "solution.npz", t=physical_t, u=final_control)
    write_csv(out_dir / "history.csv", history)
    make_plot(out_dir / "teacher_free_control", physical_t, final_control)
    summary = {
        "status": "completed_before_any_direct_blind_test",
        "teacher_free": True,
        "source_checkpoint": str(checkpoint_path),
        "training_uses_direct_solution": False,
        "training_uses_manual_target": False,
        "training_uses_switch_time_or_mask": False,
        "training_uses_objective_value_as_loss": False,
        "method": (
            "detached projected fixed-point using the full RK4 reduced gradient "
            "including N=N(u)"
        ),
        "training": vars(args),
        "selected_epoch": selected_epoch,
        "wall_seconds": wall_seconds,
        "metrics": final_metrics,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "COMPLETED.json").write_text(
        json.dumps(
            {
                "completed_unix_time": time.time(),
                "direct_solution_read_during_training": False,
                "selected_epoch": selected_epoch,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(final_metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
