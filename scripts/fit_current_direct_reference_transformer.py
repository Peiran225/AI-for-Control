#!/usr/bin/env python3
"""Fit the current-weight time-only Transformer to the strict direct schedule.

This is a diagnostic direct-assisted endpoint for method selection.  It is
deliberately kept separate from teacher-free training artifacts and records
that the direct-transcription control is used as a supervised target.
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

from scripts.continue_teacher_free_linear_box_projection import (  # noqa: E402
    LinearRawBoxProjection,
    clone_state,
    linear_model_from_payload,
)
from scripts.train_teacher_free_resolution_curriculum import evaluate  # noqa: E402
from train_paper_pmp_kkt import ProblemConfig, build_params, set_seed  # noqa: E402


DEFAULT_SOURCE = (
    ROOT
    / "outputs/clean_hard_box_pmp_20260725/seed_23/"
    "refinement_cont20_final/selected_checkpoint.pt"
)
DEFAULT_DIRECT = (
    ROOT
    / "outputs/new_objective_alpha1_grid_20260721_v2/a1_b40_g8000/"
    "direct_nominal/n800_strict_final/scale_1_direct_solution.npz"
)
DEFAULT_OUT = (
    ROOT
    / "outputs/time_only_refinement_choice_20260726/"
    "direct_reference_fit"
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


def falling_width(t: np.ndarray, u: np.ndarray, half: str) -> float:
    """Local 10--90 width around the largest falling jump in a temporal half."""

    ti = np.asarray(t[:-1], dtype=np.float64)
    ui = np.asarray(u[:-1], dtype=np.float64)
    differences = np.diff(ui)
    if half == "early":
        candidates = np.flatnonzero(ti[:-1] < 0.5 * ti[-1])
    elif half == "late":
        candidates = np.flatnonzero(ti[:-1] >= 0.5 * ti[-1])
    else:
        raise ValueError(half)
    center_index = int(candidates[np.argmin(differences[candidates])])
    center = float(ti[center_index + 1])
    radius = max(0.35, 10.0 * float(np.median(np.diff(ti))))
    left = (ti >= center - radius) & (ti < center - 0.55 * radius)
    right = (ti > center + 0.55 * radius) & (ti <= center + radius)
    if np.count_nonzero(left) < 2 or np.count_nonzero(right) < 2:
        return float("nan")
    pre, post = float(np.median(ui[left])), float(np.median(ui[right]))
    if abs(post - pre) < 1.0e-10:
        return float("nan")
    progress = (ui - pre) / (post - pre)
    local = np.flatnonzero((ti >= center - radius) & (ti <= center + radius))

    def crossing(level: float) -> float:
        hits: list[float] = []
        for index in local[:-1]:
            p0, p1 = float(progress[index]), float(progress[index + 1])
            if p0 <= level <= p1 and p1 > p0:
                hits.append(
                    float(
                        ti[index]
                        + (level - p0)
                        / (p1 - p0)
                        * (ti[index + 1] - ti[index])
                    )
                )
        if not hits:
            return float("nan")
        return min(hits, key=lambda value: abs(value - center))

    return float(crossing(0.9) - crossing(0.1))


def fit_loss(
    model: LinearRawBoxProjection,
    normalized_t: torch.Tensor,
    target: torch.Tensor,
    raw_target: torch.Tensor,
    point_weights: torch.Tensor,
    *,
    control_weight: float,
    difference_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    raw = model.raw_control(normalized_t)
    control = torch.clamp(raw, 0.0, model.umax)
    raw_loss = (point_weights * (raw - raw_target).square()).mean()
    control_loss = (point_weights * (control - target).square()).mean()
    difference_loss = (
        torch.diff(control) - torch.diff(target)
    ).square().mean()
    total = raw_loss + control_weight * control_loss + difference_weight * difference_loss
    pieces = {
        "loss": float(total.detach()),
        "raw_target_mse": float(raw_loss.detach()),
        "control_target_mse": float(control_loss.detach()),
        "difference_target_mse": float(difference_loss.detach()),
        "control_linf": float((control - target).detach().abs().max()),
    }
    return total, pieces


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", default=str(DEFAULT_SOURCE))
    parser.add_argument("--direct-solution", default=str(DEFAULT_DIRECT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=2607)
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help=(
            "Optional Adam fine-tuning after the deterministic weighted "
            "least-squares fit. Zero keeps the reproducible head-only diagnostic."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=2.0e-6)
    parser.add_argument("--final-learning-rate", type=float, default=5.0e-8)
    parser.add_argument("--switch-window", type=int, default=14)
    parser.add_argument("--switch-weight", type=float, default=25.0)
    parser.add_argument("--bound-margin", type=float, default=0.20)
    parser.add_argument("--control-weight", type=float, default=2.0)
    parser.add_argument("--difference-weight", type=float, default=8.0)
    parser.add_argument("--eval-every", type=int, default=25)
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(1)
    source_path = resolve(args.source_checkpoint)
    direct_path = resolve(args.direct_solution)
    out_dir = resolve(args.out_dir)
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True)

    source = torch.load(source_path, map_location="cpu", weights_only=False)
    if source.get("wrapper", {}).get("class") != "LinearRawBoxProjection":
        raise ValueError("source checkpoint must use LinearRawBoxProjection")
    cfg = ProblemConfig(**source["problem"])
    if cfg.n != 800:
        raise ValueError(f"expected n=800, received {cfg.n}")
    model = linear_model_from_payload(source, cfg).to(dtype=torch.float64)
    normalized_t = torch.linspace(0.0, 1.0, cfg.n + 1, dtype=torch.float64)
    physical_t = np.linspace(0.0, cfg.T, cfg.n + 1)
    params = build_params(cfg, torch.device("cpu"), torch.float64)

    direct = np.load(direct_path)
    direct_t = np.asarray(direct["t"], dtype=np.float64)
    direct_u = np.asarray(direct["u"], dtype=np.float64)
    if direct_t.shape != physical_t.shape or direct_u.shape != physical_t.shape:
        raise ValueError("direct reference must contain n+1 matching nodes")
    if not np.allclose(direct_t, physical_t, atol=1.0e-12, rtol=0.0):
        raise ValueError("direct-reference time grid does not match the network grid")
    target = torch.tensor(direct_u, dtype=torch.float64)
    raw_target = target.clone()
    raw_target[target <= 1.0e-12] = -args.bound_margin
    raw_target[target >= cfg.umax - 1.0e-12] = cfg.umax + args.bound_margin

    differences = torch.abs(torch.diff(target, prepend=target[:1]))
    point_weights = torch.ones_like(target)
    switch_nodes = torch.nonzero(differences > 0.05, as_tuple=False).flatten()
    for node in switch_nodes.tolist():
        left = max(0, node - args.switch_window)
        right = min(cfg.n + 1, node + args.switch_window + 1)
        point_weights[left:right] = torch.maximum(
            point_weights[left:right],
            torch.tensor(args.switch_weight, dtype=torch.float64),
        )

    # Closed-form weighted least-squares calibration of the existing affine
    # output head gives a reproducible starting point before gradient fitting.
    with torch.no_grad():
        hidden = model.hidden(normalized_t)
        design = torch.cat(
            [hidden, torch.ones((cfg.n + 1, 1), dtype=torch.float64)], dim=1
        )
        root_weight = torch.sqrt(point_weights)
        solution = torch.linalg.lstsq(
            design * root_weight[:, None],
            raw_target * root_weight,
        ).solution
        model.base.output.weight.copy_(solution[:-1][None, :])
        model.base.output.bias.copy_(solution[-1:])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
        eps=1.0e-12,
    )
    history: list[dict[str, Any]] = []
    candidates: list[
        tuple[float, float, dict[str, torch.Tensor], np.ndarray, int]
    ] = []
    started = time.perf_counter()

    def record(epoch: int) -> None:
        model.eval()
        with torch.no_grad():
            control = model(normalized_t)
            mse = float((control - target).square().mean())
            linf = float((control - target).abs().max())
            candidates.append(
                (
                    mse,
                    linf,
                    clone_state(model),
                    control.detach().cpu().numpy().copy(),
                    epoch,
                )
            )
        model.train()
        print(
            f"[epoch={epoch:04d}] target_rmse={math.sqrt(mse):.3e} "
            f"target_linf={linf:.3e}",
            flush=True,
        )

    record(0)
    for epoch in range(1, args.epochs + 1):
        fraction = epoch / max(args.epochs, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * fraction))
        learning_rate = args.final_learning_rate + cosine * (
            args.learning_rate - args.final_learning_rate
        )
        optimizer.param_groups[0]["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        loss, pieces = fit_loss(
            model,
            normalized_t,
            target,
            raw_target,
            point_weights,
            control_weight=args.control_weight,
            difference_weight=args.difference_weight,
        )
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        history.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "gradient_norm_before_clip": gradient_norm,
                "elapsed_seconds": time.perf_counter() - started,
                **pieces,
            }
        )
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            record(epoch)

    selected_mse, selected_linf, selected_state, selected_u, selected_epoch = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    model.load_state_dict(selected_state, strict=True)
    metrics, reloaded_u = evaluate(
        model, cfg, normalized_t, params, high_accuracy=True
    )
    if np.max(np.abs(reloaded_u - selected_u)) > 5.0e-9:
        raise RuntimeError("selected checkpoint reload mismatch")
    metrics.update(
        {
            "physical_objective": 400.0
            * float(metrics["high_accuracy_J_diagnostic_only"]),
            "direct_target_rmse": math.sqrt(selected_mse),
            "direct_target_linf": selected_linf,
            "early_falling_width_10_90": falling_width(
                physical_t, selected_u, "early"
            ),
            "late_falling_width_10_90": falling_width(
                physical_t, selected_u, "late"
            ),
        }
    )

    torch.save(
        {
            "model_state": selected_state,
            "base_model_args": source["base_model_args"],
            "problem": source["problem"],
            "wrapper": source["wrapper"],
            "source_checkpoint": str(source_path),
            "direct_reference": str(direct_path),
            "teacher_free": False,
            "direct_or_manual_solution_used": True,
            "method": "direct-reference supervised diagnostic fit",
            "selected_epoch": selected_epoch,
            "metrics": metrics,
        },
        out_dir / "selected_checkpoint.pt",
    )
    np.savez(
        out_dir / "solution.npz",
        t=physical_t,
        u=selected_u,
        direct_u=direct_u,
    )
    if not history:
        history.append(
            {
                "epoch": 0,
                "learning_rate": 0.0,
                "gradient_norm_before_clip": 0.0,
                "elapsed_seconds": 0.0,
                "loss": float("nan"),
                "raw_target_mse": float("nan"),
                "control_target_mse": selected_mse,
                "difference_target_mse": float("nan"),
                "control_linf": selected_linf,
            }
        )
    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "status": "completed",
        "scope": "diagnostic method-choice comparison; not teacher-free training",
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": sha256(source_path),
        "direct_reference": str(direct_path),
        "direct_reference_sha256": sha256(direct_path),
        "training_uses_direct_transcription_schedule": True,
        "training": vars(args),
        "selected_epoch": selected_epoch,
        "metrics": metrics,
        "wall_seconds": time.perf_counter() - started,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
