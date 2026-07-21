#!/usr/bin/env python3
"""Strict teacher-free projected-KKT continuation for the n=800 Transformer.

The script never reads a direct/manual control, switching time, switching mask,
or objective target.  Checkpoint selection uses projected-gradient residuals
subject to generic shape-preservation guards; the objective value is recorded
only as a diagnostic because its full derivative is needed to form the KKT
mapping.

Two training modes are supported:

``detached``
    Recompute ``Pi(u - s grad F_h(u))`` and fit this detached fixed-point target.

``exact``
    Differentiate the unit-step projected-gradient mapping itself.  This uses
    second derivatives of the reduced objective and is intended only for a
    short final continuation.
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generalized_mean_square(values: torch.Tensor, p: float) -> torch.Tensor:
    """Return ||values||_p^2 with a mean rather than a sum normalization."""

    epsilon = torch.finfo(values.dtype).tiny
    return (values.abs().pow(p).mean() + epsilon).pow(2.0 / p)


def shape_guard(metrics: dict[str, Any], args: argparse.Namespace) -> bool:
    return bool(
        metrics["nondegenerate_high_low_high"]
        and abs(metrics["u_max"] - 3.0) <= 1.0e-14
        and metrics["exact_upper_bound_count"] >= args.min_upper_nodes
        and metrics["early_width_10_90"] <= args.max_early_width
        and metrics["late_width_10_90"] <= args.max_late_width
        and metrics["plateau_relative_percent"] <= args.max_plateau_percent
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3.0e-6)
    parser.add_argument("--final-learning-rate", type=float, default=3.0e-7)
    parser.add_argument(
        "--schedule",
        choices=("constant", "cosine", "linear"),
        default="cosine",
    )
    parser.add_argument("--mode", choices=("detached", "exact"), default="detached")
    parser.add_argument("--step-multiplier", type=float, default=20.0)
    parser.add_argument("--rms-weight", type=float, default=1.0)
    parser.add_argument("--linf-weight", type=float, default=0.0)
    parser.add_argument("--pnorm-weight", type=float, default=1.0)
    parser.add_argument("--p", type=float, default=12.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--adam-eps", type=float, default=1.0e-12)
    parser.add_argument(
        "--learn-temperature",
        action="store_true",
        help="also refine the single global, time-agnostic output temperature",
    )
    parser.add_argument("--max-early-width", type=float, default=0.030)
    parser.add_argument("--max-late-width", type=float, default=0.050)
    parser.add_argument("--max-plateau-percent", type=float, default=0.45)
    parser.add_argument("--min-upper-nodes", type=int, default=100)
    args = parser.parse_args()

    if args.mode == "exact" and args.step_multiplier != 1.0:
        raise ValueError("exact mode uses the canonical unit-step KKT mapping")
    if args.p <= 2.0:
        raise ValueError("p must be greater than 2")

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
        raise ValueError(f"strict continuation expects n=800, received n={cfg.n}")
    wrapper = dict(source["wrapper"])
    base = build_model(dict(source["base_model_args"]), cfg).to(dtype=torch.float64)
    model = FixedBoxProjection(
        base,
        cfg.umax,
        float(wrapper["scale"]),
        temperature=float(wrapper.get("temperature", 1.0)),
        learn_temperature=args.learn_temperature,
    ).to(dtype=torch.float64)
    model.load_state_dict(source["model_state"], strict=True)

    normalized_t = torch.linspace(0.0, 1.0, cfg.n + 1, dtype=torch.float64)
    physical_t = np.linspace(0.0, cfg.T, cfg.n + 1)
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
        eps=args.adam_eps,
    )
    history: list[dict[str, Any]] = []
    candidates: list[tuple[dict[str, Any], dict[str, torch.Tensor], np.ndarray, int]] = []
    started = time.perf_counter()

    provenance = {
        "teacher_free": True,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "training_uses_direct_solution": False,
        "training_uses_manual_target": False,
        "training_uses_switch_time_or_mask": False,
        "training_uses_objective_value_as_loss": False,
        "selection_uses_objective_value": False,
        "selection_rule": (
            "minimum projected-gradient Linf, then RMS, among candidates "
            "passing predeclared generic shape-preservation guards"
        ),
        "direct_comparison_allowed_only_after_checkpoint_is_frozen": True,
        "training": vars(args),
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    def record(epoch: int) -> None:
        metrics, control = evaluate(model, cfg, normalized_t, params, high_accuracy=True)
        passed = shape_guard(metrics, args)
        row = {
            "epoch": epoch,
            "event": "evaluation",
            **metrics,
            "shape_guard_passed": passed,
            "elapsed_seconds": time.perf_counter() - started,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        if passed:
            candidates.append((metrics, clone_state(model), control, epoch))
        print(
            f"[e={epoch}] PGinf={metrics['projected_gradient_linf']:.3e} "
            f"PGrms={metrics['projected_gradient_rms']:.3e} "
            f"width={metrics['early_width_10_90']:.4f}/"
            f"{metrics['late_width_10_90']:.4f} "
            f"plateau={metrics['plateau_relative_percent']:.4f}% "
            f"upper={metrics['exact_upper_bound_count']} guard={passed}",
            flush=True,
        )

    record(0)
    for epoch in range(1, args.epochs + 1):
        fraction = epoch / max(args.epochs, 1)
        if args.schedule == "constant":
            learning_rate = args.learning_rate
        elif args.schedule == "linear":
            learning_rate = args.learning_rate + fraction * (
                args.final_learning_rate - args.learning_rate
            )
        else:
            cosine = 0.5 * (1.0 + math.cos(math.pi * fraction))
            learning_rate = args.final_learning_rate + cosine * (
                args.learning_rate - args.final_learning_rate
            )
        optimizer.param_groups[0]["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        all_control = model(normalized_t)
        interval_control = all_control[:-1]
        objective = rk4_reduced_objective(interval_control, cfg, params)
        create_graph = args.mode == "exact"
        gradient = torch.autograd.grad(
            objective, interval_control, create_graph=create_graph
        )[0]
        if args.mode == "detached":
            with torch.no_grad():
                target = torch.clamp(
                    interval_control.detach()
                    - args.step_multiplier * gradient.detach(),
                    0.0,
                    cfg.umax,
                )
            residual = (interval_control - target) / cfg.umax
        else:
            mapping = interval_control - torch.clamp(
                interval_control - gradient, 0.0, cfg.umax
            )
            residual = mapping / cfg.umax
        rms_loss = residual.square().mean()
        linf_loss = residual.abs().max().square()
        pnorm_loss = generalized_mean_square(residual, args.p)
        loss = (
            args.rms_weight * rms_loss
            + args.linf_weight * linf_loss
            + args.pnorm_weight * pnorm_loss
        )
        loss.backward()
        parameter_gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        )
        optimizer.step()
        history.append(
            {
                "epoch": epoch,
                "event": "train",
                "loss": float(loss.detach()),
                "rms_loss": float(rms_loss.detach()),
                "linf_loss": float(linf_loss.detach()),
                "pnorm_loss": float(pnorm_loss.detach()),
                "p": args.p,
                "full_gradient_linf_before_step": float(gradient.detach().abs().max()),
                "parameter_gradient_norm_before_clip": parameter_gradient_norm,
                "learning_rate": learning_rate,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            record(epoch)

    if not candidates:
        raise RuntimeError("no candidate passed the predeclared shape guard")
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
    selected_wrapper = dict(wrapper)
    selected_wrapper["temperature"] = float(model.temperature_value().detach())
    selected_wrapper["learn_temperature"] = args.learn_temperature
    checkpoint = {
        "model_state": selected_state,
        "base_model_args": source["base_model_args"],
        "problem": source["problem"],
        "wrapper": selected_wrapper,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "teacher_free": True,
        "method": f"strict projected-KKT continuation ({args.mode})",
        "selected_epoch": selected_epoch,
        "selection_metrics": final_metrics,
        "learned_global_temperature": float(model.temperature_value().detach()),
    }
    torch.save(checkpoint, out_dir / "selected_checkpoint.pt")
    np.savez(out_dir / "solution.npz", t=physical_t, u=final_control)
    write_csv(out_dir / "history.csv", history)
    make_plot(out_dir / "teacher_free_control", physical_t, final_control)
    summary = {
        "status": "checkpoint_frozen_before_any_direct_comparison",
        **provenance,
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
                "selected_checkpoint_sha256": sha256(out_dir / "selected_checkpoint.pt"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(final_metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
