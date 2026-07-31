#!/usr/bin/env python3
"""Teacher-free optimality-condition curriculum for a time-only neural policy.

The training path deliberately has no direct-solution, manual-target, switching-
time, switching-mask, or objective-value supervision.  It starts from the
manuscript PMP/KKT-only checkpoint and minimizes the projected full reduced-
gradient mapping

    G_h(u) = u - projection_[0,umax](u - grad F_h(u)).

The differentiable RK4 rollout inside ``F_h`` includes the complete N=N(u)
dependence.  Resolution is increased from n=200 to n=400 and then n=800 while
retaining the same policy-network weights.  Optional PMP/KKT and global
smoothness terms are generic optimality/curriculum terms; neither uses a target
control or a prescribed switching region, and smoothness is zero in the final
part of every stage.

Direct-solution comparison is intentionally implemented in a separate script
and may only be run after this script has finished and written COMPLETED.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT, ROOT / "scripts"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts.refine_time_only_singular_plateau import (  # noqa: E402
    build_model,
    high_accuracy_metrics,
    rk4_reduced_objective,
)
from train_paper_pmp_kkt import (  # noqa: E402
    ProblemConfig,
    build_params,
    pmp_kkt_loss,
    set_seed,
)
from tumor_problem import TumorProblem  # noqa: E402


DEFAULT_START = (
    ROOT / "paper_runs/smoothness_weight_sweep/w3/seed_4/best_pmp_kkt.pt"
)


class FixedBoxProjection(nn.Module):
    """Boundary-capable output without time- or case-specific information."""

    def __init__(
        self,
        base: nn.Module,
        umax: float,
        scale: float,
        temperature: float = 1.0,
        learn_temperature: bool = False,
    ) -> None:
        super().__init__()
        self.base = base
        self.umax = float(umax)
        self.register_buffer("scale", torch.tensor(float(scale), dtype=torch.float64))
        self.learn_temperature = bool(learn_temperature)
        raw_temperature = torch.tensor(
            math.log(float(temperature)), dtype=torch.float64
        )
        if learn_temperature:
            self.raw_temperature = nn.Parameter(raw_temperature)
        else:
            self.register_buffer("raw_temperature", raw_temperature)

    def temperature_value(self) -> torch.Tensor:
        # Global architecture parameter: it contains no time, state, mask, or
        # switching-location information.  The bounds only prevent numerical
        # saturation of the inverse-sigmoid transform.
        return torch.exp(self.raw_temperature).clamp(0.25, 1.5)

    def forward(self, normalized_t: torch.Tensor) -> torch.Tensor:
        base_control = self.base(normalized_t)
        probability = torch.clamp(base_control / self.umax, 1.0e-8, 1.0 - 1.0e-8)
        reshaped = self.umax * torch.sigmoid(
            torch.logit(probability) / self.temperature_value()
        )
        return torch.clamp(self.scale * reshaped, 0.0, self.umax)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def linear_anneal(start: float, end: float, fraction: float) -> float:
    return float(start + (end - start) * min(max(fraction, 0.0), 1.0))


def plateau_summary(high: dict[str, Any]) -> dict[str, float]:
    return {
        "high_accuracy_J_diagnostic_only": float(high["J"]),
        "continuous_projected_kkt_rms": float(high["projected_kkt_rms"]),
        "plateau_relative_percent": 100.0
        * float(high["plateau_total_relative_range"]),
        "plateau_range": float(high["plateau_total_range"]),
    }


def interval_crossing_width(
    physical_t: np.ndarray, control: np.ndarray, side: str
) -> float:
    """Generic 10--90 width located from the largest early/late jump."""

    t = np.asarray(physical_t[:-1], dtype=np.float64)
    u = np.asarray(control[:-1], dtype=np.float64)
    differences = np.diff(u)
    if side == "early":
        candidates = np.flatnonzero(t[:-1] < 0.5 * t[-1])
        jump = int(candidates[np.argmin(differences[candidates])])
    elif side == "late":
        candidates = np.flatnonzero(t[:-1] >= 0.5 * t[-1])
        jump = int(candidates[np.argmax(differences[candidates])])
    else:
        raise ValueError(side)
    junction = float(t[jump + 1])
    radius = max(0.35, 8.0 * float(np.median(np.diff(t))))
    pre_mask = (t >= junction - radius) & (t < junction - 0.55 * radius)
    post_mask = (t > junction + 0.55 * radius) & (t <= junction + radius)
    if np.count_nonzero(pre_mask) < 2 or np.count_nonzero(post_mask) < 2:
        return float("nan")
    pre = float(np.median(u[pre_mask]))
    post = float(np.median(u[post_mask]))
    if abs(post - pre) < 1.0e-8:
        return float("nan")
    progress = (u - pre) / (post - pre)
    local = np.flatnonzero((t >= junction - radius) & (t <= junction + radius))

    def crossing(level: float) -> float:
        values: list[float] = []
        for left in local[:-1]:
            right = left + 1
            p0, p1 = progress[left], progress[right]
            if p0 <= level <= p1 and p1 > p0:
                values.append(
                    float(
                        t[left]
                        + (level - p0) / (p1 - p0) * (t[right] - t[left])
                    )
                )
        if not values:
            return float("nan")
        return min(values, key=lambda value: abs(value - junction))

    t10, t90 = crossing(0.1), crossing(0.9)
    return float(t90 - t10)


def evaluate(
    model: nn.Module,
    cfg: ProblemConfig,
    normalized_t: torch.Tensor,
    params: dict[str, torch.Tensor],
    *,
    high_accuracy: bool,
) -> tuple[dict[str, Any], np.ndarray]:
    model.eval()
    with torch.enable_grad():
        all_control = model(normalized_t)
        interval_control = all_control[:-1]
        objective = rk4_reduced_objective(interval_control, cfg, params)
        gradient = torch.autograd.grad(objective, interval_control)[0]
        mapping = interval_control - torch.clamp(
            interval_control - gradient, 0.0, cfg.umax
        )
    physical_t = (cfg.T * normalized_t).detach().cpu().numpy()
    u = all_control.detach().cpu().numpy()
    interval_u_numpy = u[:-1]
    edge_count = max(2, cfg.n // 20)
    early_edge_median = float(np.median(interval_u_numpy[:edge_count]))
    late_edge_median = float(np.median(interval_u_numpy[-edge_count:]))
    interior = interval_u_numpy[edge_count:-edge_count]
    interior_low_quantile = float(np.quantile(interior, 0.10))
    dynamic_range = float(np.ptp(interval_u_numpy))
    # This is only an anti-degeneracy guard.  It specifies neither a switching
    # time nor a candidate mask, and it is never included in the loss.
    nondegenerate_high_low_high = bool(
        dynamic_range >= 0.8
        and early_edge_median >= interior_low_quantile + 0.5
        and late_edge_median >= interior_low_quantile + 0.5
        and np.count_nonzero(interval_u_numpy == cfg.umax) < int(0.95 * cfg.n)
        and np.count_nonzero(interval_u_numpy == 0.0) < int(0.95 * cfg.n)
    )
    dt = cfg.T / cfg.n
    metrics: dict[str, Any] = {
        "n": cfg.n,
        "rk4_objective_diagnostic_only": float(objective.detach()),
        "projected_gradient_linf": float(mapping.detach().abs().max()),
        "projected_gradient_rms": float(mapping.detach().square().mean().sqrt()),
        "projected_gradient_over_dt_linf": float(mapping.detach().abs().max() / dt),
        "projected_gradient_over_dt_rms": float(
            mapping.detach().square().mean().sqrt() / dt
        ),
        "raw_gradient_linf": float(gradient.detach().abs().max()),
        "u_min": float(np.min(u[:-1])),
        "u_max": float(np.max(u[:-1])),
        "exact_upper_bound_count": int(np.count_nonzero(u[:-1] == cfg.umax)),
        "exact_lower_bound_count": int(np.count_nonzero(u[:-1] == 0.0)),
        "early_edge_median": early_edge_median,
        "late_edge_median": late_edge_median,
        "interior_low_quantile": interior_low_quantile,
        "control_dynamic_range": dynamic_range,
        "nondegenerate_high_low_high": nondegenerate_high_low_high,
        "early_width_10_90": interval_crossing_width(physical_t, u, "early"),
        "late_width_10_90": interval_crossing_width(physical_t, u, "late"),
        "control_total_variation": float(np.abs(np.diff(u[:-1])).sum()),
        "control_second_difference_l1": float(np.abs(np.diff(u[:-1], n=2)).sum()),
    }
    with torch.no_grad():
        pack = pmp_kkt_loss(
            all_control, cfg, params, 0.1, 0.03, detach_gate=True
        )
    metrics.update(
        {
            "euler_pmp_kkt_opt_gap": float(pack["opt_gap"]),
            "euler_smoothness": float(pack["smooth"]),
        }
    )
    if high_accuracy:
        high, _ = high_accuracy_metrics(
            physical_t,
            u,
            plateau_start=1.0,
            plateau_end=8.5,
            problem=TumorProblem(
                T=cfg.T,
                m=cfg.m,
                umax=cfg.umax,
                beta=cfg.beta,
                alpha=cfg.alpha,
                gamma=cfg.gamma,
                n0=cfg.n0,
                m_suppression=cfg.m_suppression,
            ),
        )
        metrics.update(plateau_summary(high))
    model.train()
    return metrics, u


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


def make_plot(path: Path, physical_t: np.ndarray, control: np.ndarray) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.15))
    axes[0].step(physical_t, control, where="post", color="#225B8A", lw=1.55)
    axes[0].set(xlim=(0.0, 10.0), ylim=(-0.05, 3.08), title="Teacher-free n=800", xlabel="time", ylabel="control")
    axes[1].step(physical_t, control, where="post", color="#225B8A", lw=1.7)
    axes[1].set(xlim=(0.15, 0.80), ylim=(-0.05, 3.08), title="Early transition", xlabel="time")
    axes[2].step(physical_t, control, where="post", color="#225B8A", lw=1.7)
    axes[2].set(xlim=(8.55, 9.30), ylim=(-0.05, 3.08), title="Late transition", xlabel="time")
    for axis in axes:
        axis.grid(alpha=0.18)
    figure.tight_layout()
    figure.savefig(path.with_suffix(".png"), dpi=220)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--start-checkpoint", default=str(DEFAULT_START))
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--scale", type=float, default=1.02)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learn-temperature", action="store_true")
    parser.add_argument("--perturbation-std", type=float, default=0.0)
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Use the checkpoint architecture/problem but not its model weights.",
    )
    parser.add_argument("--epochs", default="160,120,100")
    parser.add_argument("--learning-rates", default="2e-5,8e-6,3e-6")
    parser.add_argument("--linf-weights", default="0.05,0.10,0.20")
    parser.add_argument("--pmp-weights", default="0.02,0.005,0.0")
    parser.add_argument("--smooth-weights", default="0.01,0.002,0.0005")
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument(
        "--selection-guard",
        choices=("high-low-high", "residual-only"),
        default="high-low-high",
        help=(
            "Checkpoint admissibility rule.  Use residual-only when changing "
            "objective weights because the old high-low-high topology need not persist."
        ),
    )
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(1)
    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir = out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True)
    checkpoint_path = Path(args.start_checkpoint).expanduser().resolve()
    lowered = str(checkpoint_path).lower()
    prohibited = ("direct", "manual_target", "supervised", "distill")
    if any(token in lowered for token in prohibited):
        raise ValueError(f"prohibited teacher-like starting checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = dict(checkpoint["args"])
    base_problem = dict(checkpoint["problem"])
    cfg0 = ProblemConfig(**base_problem)
    base = build_model(checkpoint_args, cfg0).to(dtype=torch.float64)
    if not args.random_init:
        base.load_state_dict(checkpoint["model_state"])
    if args.perturbation_std > 0:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        with torch.no_grad():
            for parameter in base.parameters():
                parameter.add_(
                    args.perturbation_std
                    * torch.randn(
                        parameter.shape,
                        dtype=parameter.dtype,
                        device=parameter.device,
                        generator=generator,
                    )
                )
    model = FixedBoxProjection(
        base,
        cfg0.umax,
        args.scale,
        temperature=args.temperature,
        learn_temperature=args.learn_temperature,
    ).to(dtype=torch.float64)

    resolutions = (200, 400, 800)
    epochs = tuple(int(value) for value in args.epochs.split(","))
    learning_rates = tuple(float(value) for value in args.learning_rates.split(","))
    linf_weights = tuple(float(value) for value in args.linf_weights.split(","))
    pmp_weights = tuple(float(value) for value in args.pmp_weights.split(","))
    smooth_weights = tuple(float(value) for value in args.smooth_weights.split(","))
    if not all(len(values) == 3 for values in (epochs, learning_rates, linf_weights, pmp_weights, smooth_weights)):
        raise ValueError("all curriculum schedules must contain three comma-separated values")

    lineage = {
        "method": "teacher-free resolution curriculum",
        "training_uses_direct_solution": False,
        "training_uses_manual_target": False,
        "training_uses_switch_time_or_mask": False,
        "training_uses_objective_value_as_loss": False,
        "permitted_start": (
            "random initialization"
            if args.random_init
            else "manuscript PMP/KKT-only checkpoint"
        ),
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "source_checkpoint_args": checkpoint_args,
        "source_checkpoint_problem": base_problem,
        "curriculum": {
            "resolutions": resolutions,
            "epochs": epochs,
            "learning_rates": learning_rates,
            "projected_linf_weights": linf_weights,
            "pmp_weights_at_stage_start": pmp_weights,
            "smooth_weights_at_stage_start": smooth_weights,
            "pmp_and_smooth_anneal_to_zero_by_fraction": 0.70,
            "fixed_box_projection_scale": args.scale,
            "global_logit_temperature_initial": args.temperature,
            "global_logit_temperature_learned": args.learn_temperature,
        },
        "selection_before_blind_test": (
            "minimum projected-gradient Linf at each stage; objective values are diagnostics only"
        ),
        "seed": args.seed,
        "perturbation_std": args.perturbation_std,
        "random_init": args.random_init,
        "selection_guard": args.selection_guard,
    }
    (out_dir / "lineage.json").write_text(json.dumps(lineage, indent=2) + "\n", encoding="utf-8")

    all_history: list[dict[str, Any]] = []
    stage_summaries: list[dict[str, Any]] = []
    total_started = time.perf_counter()
    final_t: np.ndarray | None = None
    final_u: np.ndarray | None = None

    for stage_index, n in enumerate(resolutions):
        cfg = ProblemConfig(**{**base_problem, "n": n})
        params = build_params(cfg, torch.device("cpu"), torch.float64)
        normalized_t = torch.linspace(0.0, 1.0, n + 1, dtype=torch.float64)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rates[stage_index], weight_decay=0.0
        )
        best_state = clone_state(model)
        best_metrics, best_u = evaluate(
            model, cfg, normalized_t, params, high_accuracy=False
        )
        best_epoch = 0
        stage_started = time.perf_counter()
        stage_history: list[dict[str, Any]] = [
            {"stage": stage_index + 1, "epoch": 0, "event": "evaluation", **best_metrics}
        ]
        print(
            f"[n={n} e=0] PGinf={best_metrics['projected_gradient_linf']:.3e} "
            f"PGrms={best_metrics['projected_gradient_rms']:.3e} "
            f"J(diag)={best_metrics['rk4_objective_diagnostic_only']:.8f}",
            flush=True,
        )

        for epoch in range(1, epochs[stage_index] + 1):
            optimizer.zero_grad(set_to_none=True)
            all_control = model(normalized_t)
            interval_control = all_control[:-1]
            objective = rk4_reduced_objective(interval_control, cfg, params)
            full_gradient = torch.autograd.grad(
                objective, interval_control, create_graph=True
            )[0]
            mapping = interval_control - torch.clamp(
                interval_control - full_gradient, 0.0, cfg.umax
            )
            dt = cfg.T / cfg.n
            projected_rms_loss = (mapping / dt).square().mean()
            projected_linf_loss = (mapping.abs().max() / dt).square()
            fraction = epoch / max(1, epochs[stage_index])
            anneal = max(0.0, 1.0 - fraction / 0.70)
            pmp_weight = pmp_weights[stage_index] * anneal
            smooth_weight = smooth_weights[stage_index] * anneal
            if pmp_weight > 0.0:
                pack = pmp_kkt_loss(
                    all_control, cfg, params, 0.1, 0.03, detach_gate=True
                )
                pmp_term = pack["opt_gap"]
            else:
                pmp_term = torch.zeros((), dtype=torch.float64)
            normalized_slope = torch.diff(all_control) / dt
            smooth_term = normalized_slope.square().mean()
            loss = (
                projected_rms_loss
                + linf_weights[stage_index] * projected_linf_loss
                + pmp_weight * pmp_term
                + smooth_weight * smooth_term
            )
            loss.backward()
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            )
            optimizer.step()
            row: dict[str, Any] = {
                "stage": stage_index + 1,
                "n": n,
                "epoch": epoch,
                "event": "train",
                "loss": float(loss.detach()),
                "projected_rms_loss": float(projected_rms_loss.detach()),
                "projected_linf_loss": float(projected_linf_loss.detach()),
                "pmp_term": float(pmp_term.detach()),
                "smooth_term": float(smooth_term.detach()),
                "pmp_weight": pmp_weight,
                "smooth_weight": smooth_weight,
                "gradient_norm_before_clip": gradient_norm,
                "elapsed_stage_seconds": time.perf_counter() - stage_started,
                "elapsed_total_seconds": time.perf_counter() - total_started,
            }
            stage_history.append(row)
            if epoch % args.eval_every == 0 or epoch == epochs[stage_index]:
                metrics, u = evaluate(
                    model, cfg, normalized_t, params, high_accuracy=False
                )
                eval_row = {
                    "stage": stage_index + 1,
                    "epoch": epoch,
                    "event": "evaluation",
                    **metrics,
                    "elapsed_stage_seconds": time.perf_counter() - stage_started,
                    "elapsed_total_seconds": time.perf_counter() - total_started,
                }
                stage_history.append(eval_row)
                admissible = (
                    metrics["nondegenerate_high_low_high"]
                    if args.selection_guard == "high-low-high"
                    else math.isfinite(metrics["projected_gradient_linf"])
                    and math.isfinite(metrics["projected_gradient_rms"])
                )
                if admissible and (
                    metrics["projected_gradient_linf"],
                    metrics["projected_gradient_rms"],
                ) < (
                    best_metrics["projected_gradient_linf"],
                    best_metrics["projected_gradient_rms"],
                ):
                    best_state = clone_state(model)
                    best_metrics = metrics
                    best_u = u
                    best_epoch = epoch
                print(
                    f"[n={n} e={epoch}] PGinf={metrics['projected_gradient_linf']:.3e} "
                    f"PGrms={metrics['projected_gradient_rms']:.3e} "
                    f"J(diag)={metrics['rk4_objective_diagnostic_only']:.8f} "
                    f"u=[{metrics['u_min']:.3f},{metrics['u_max']:.3f}]",
                    flush=True,
                )

        model.load_state_dict(best_state)
        selected_metrics, selected_u = evaluate(
            model, cfg, normalized_t, params, high_accuracy=True
        )
        stage_elapsed = time.perf_counter() - stage_started
        stage_dir = out_dir / f"stage_{stage_index + 1}_n{n}"
        stage_dir.mkdir()
        torch.save(
            {
                "model_state": clone_state(model),
                "base_model_args": checkpoint_args,
                "problem": asdict(cfg),
                "wrapper": {
                    "class": "FixedBoxProjection",
                    "scale": args.scale,
                    "temperature": float(model.temperature_value().detach()),
                    "learn_temperature": args.learn_temperature,
                    "forward": (
                        "clamp(scale * umax * sigmoid(logit(base/umax) / "
                        "global_temperature), 0, umax)"
                    ),
                },
                "source_checkpoint": str(checkpoint_path),
                "teacher_free": True,
                "direct_or_manual_solution_used": False,
                "objective_value_used_as_loss_or_selection": False,
                "switching_time_or_mask_used": False,
                "full_gradient_includes_state_dependence": True,
                "stage": stage_index + 1,
                "selected_epoch": best_epoch,
                "selection_metrics": selected_metrics,
            },
            stage_dir / "selected_checkpoint.pt",
        )
        physical_t = np.linspace(0.0, cfg.T, cfg.n + 1)
        np.savez(stage_dir / "solution.npz", t=physical_t, u=selected_u)
        write_csv(stage_dir / "history.csv", stage_history)
        stage_summary = {
            "stage": stage_index + 1,
            "n": n,
            "selected_epoch": best_epoch,
            "wall_seconds": stage_elapsed,
            "metrics": selected_metrics,
        }
        (stage_dir / "summary.json").write_text(
            json.dumps(stage_summary, indent=2) + "\n", encoding="utf-8"
        )
        stage_summaries.append(stage_summary)
        all_history.extend(stage_history)
        final_t, final_u = physical_t, selected_u

    assert final_t is not None and final_u is not None
    total_elapsed = time.perf_counter() - total_started
    write_csv(out_dir / "history_all_stages.csv", all_history)
    make_plot(out_dir / "teacher_free_control", final_t, final_u)
    summary = {
        "status": "completed_before_any_direct_blind_test",
        "teacher_free_training": True,
        "total_training_wall_seconds": total_elapsed,
        "stages": stage_summaries,
        "final_metrics": stage_summaries[-1]["metrics"],
        "selection_rule": lineage["selection_before_blind_test"],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "COMPLETED.json").write_text(
        json.dumps(
            {
                "completed_unix_time": time.time(),
                "total_training_wall_seconds": total_elapsed,
                "direct_solution_read_during_training": False,
                "final_checkpoint": str(out_dir / "stage_3_n800/selected_checkpoint.pt"),
                "final_solution": str(out_dir / "stage_3_n800/solution.npz"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["final_metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
