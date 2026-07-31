#!/usr/bin/env python3
"""Fit direct ZOH targets at strict off-grid query times near both switches."""

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
for search_path in (ROOT, ROOT / "scripts"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from refine_direct_guided_scalar_time import (  # noqa: E402
    clone_state,
    load_model,
)
from train_paper_pmp_kkt import build_params, set_seed, time_features  # noqa: E402
from train_teacher_free_resolution_curriculum import evaluate  # noqa: E402


def resolve(path: str | Path) -> Path:
    expanded = Path(path).expanduser()
    return (expanded if expanded.is_absolute() else ROOT / expanded).resolve()


def fixed_support_query_control(
    model: torch.nn.Module,
    support_time: torch.Tensor,
    query_time: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    """Differentiable fixed-support query used by the off-grid evaluator."""

    support_count = int(support_time.numel())
    sequence_length = support_count + 1
    mask = torch.zeros(
        (sequence_length, sequence_length),
        dtype=torch.bool,
        device=support_time.device,
    )
    mask[:support_count, support_count] = True
    mask[support_count, support_count] = True
    support_features = time_features(support_time)
    controls: list[torch.Tensor] = []
    for start in range(0, int(query_time.numel()), batch_size):
        current = query_time[start : start + batch_size]
        batch = int(current.numel())
        features = torch.cat(
            [
                support_features.unsqueeze(0).expand(batch, -1, -1),
                time_features(current).unsqueeze(1),
            ],
            dim=1,
        )
        hidden = model.base.input(features)
        encoded = model.base.encoder(hidden, mask=mask)
        raw = model.base.output(encoded[:, -1, :]).squeeze(-1)
        base_control = model.umax * torch.sigmoid(raw)
        controls.append(
            torch.clamp(
                model.scale * base_control - model.offset,
                0.0,
                model.umax,
            )
        )
    return torch.cat(controls, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-checkpoint", required=True)
    parser.add_argument("--direct-solution", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2607)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--final-learning-rate", type=float, default=1.0e-8)
    parser.add_argument("--query-radius", type=float, default=0.20)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument(
        "--query-target-mode",
        choices=("linear", "zoh-left"),
        default="linear",
        help=(
            "Interpolation used to supervise strict off-grid queries. "
            "'zoh-left' preserves a direct transcription's sharp switches."
        ),
    )
    parser.add_argument("--query-weight", type=float, default=100.0)
    parser.add_argument("--support-weight", type=float, default=1.0)
    parser.add_argument("--start-anchor-weight", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--support-rmse-guard-factor", type=float, default=2.0)
    parser.add_argument(
        "--support-restoration-steps",
        type=int,
        default=0,
        help="Support-only AdamW steps applied after every joint query step.",
    )
    parser.add_argument(
        "--support-restoration-learning-rate",
        type=float,
        default=1.0e-6,
    )
    args = parser.parse_args()

    set_seed(args.seed)
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    start_checkpoint = resolve(args.start_checkpoint)
    direct_solution = resolve(args.direct_solution)
    out_dir = resolve(args.out_dir)
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True)
    model, cfg, source = load_model(start_checkpoint, device)
    support_time = torch.linspace(
        0.0, 1.0, cfg.n + 1, device=device, dtype=torch.float64
    )
    physical_t = np.linspace(0.0, cfg.T, cfg.n + 1)
    direct = np.load(direct_solution)
    direct_t = np.asarray(direct["t"], dtype=np.float64)
    direct_u = np.asarray(direct["u"], dtype=np.float64)
    if (
        direct_t.ndim != 1
        or direct_u.shape != direct_t.shape
        or abs(float(direct_t[0])) > 1.0e-12
        or abs(float(direct_t[-1]) - cfg.T) > 1.0e-12
    ):
        raise ValueError("direct solution must span the full treatment horizon")
    support_target_numpy = np.interp(physical_t, direct_t, direct_u)
    support_target = torch.tensor(
        support_target_numpy, device=device, dtype=torch.float64
    )

    interval_midpoints = 0.5 * (physical_t[:-1] + physical_t[1:])
    differences = np.diff(support_target_numpy[:-1])
    early_candidates = np.flatnonzero(physical_t[:-2] < 0.5 * cfg.T)
    late_candidates = np.flatnonzero(physical_t[:-2] >= 0.5 * cfg.T)
    early_switch = float(
        physical_t[
            int(early_candidates[np.argmin(differences[early_candidates])]) + 1
        ]
    )
    late_switch = float(
        physical_t[
            int(late_candidates[np.argmin(differences[late_candidates])]) + 1
        ]
    )
    query_mask = (
        (np.abs(interval_midpoints - early_switch) <= args.query_radius)
        | (np.abs(interval_midpoints - late_switch) <= args.query_radius)
    )
    query_physical = interval_midpoints[query_mask]
    query_time = torch.tensor(
        query_physical / cfg.T, device=device, dtype=torch.float64
    )
    # A higher-resolution direct solution supplies genuine values at strict
    # off-grid query coordinates; an n=800 reference reduces to interpolation.
    if args.query_target_mode == "linear":
        query_target_numpy = np.interp(
            query_physical, direct_t, direct_u
        )
    else:
        query_indices = np.searchsorted(
            direct_t, query_physical, side="right"
        ) - 1
        query_indices = np.clip(query_indices, 0, direct_u.size - 2)
        query_target_numpy = direct_u[query_indices]
    query_target = torch.tensor(
        query_target_numpy,
        device=device,
        dtype=torch.float64,
    )
    model.eval()
    with torch.enable_grad():
        start_support = model(support_time).detach()
        start_query = fixed_support_query_control(
            model,
            support_time,
            query_time,
            batch_size=args.query_batch_size,
        ).detach()

    def loss_pack() -> dict[str, torch.Tensor]:
        support_control = model(support_time)
        query_control = fixed_support_query_control(
            model,
            support_time,
            query_time,
            batch_size=args.query_batch_size,
        )
        support_loss = (
            (support_control - support_target) / cfg.umax
        ).square().mean()
        query_loss = (
            (query_control - query_target) / cfg.umax
        ).square().mean()
        start_anchor = (
            (support_control - start_support) / cfg.umax
        ).square().mean() + (
            (query_control - start_query) / cfg.umax
        ).square().mean()
        total = (
            args.support_weight * support_loss
            + args.query_weight * query_loss
            + args.start_anchor_weight * start_anchor
        )
        return {
            "loss": total,
            "support_rmse": (support_control - support_target)
            .square()
            .mean()
            .sqrt(),
            "support_linf": (support_control - support_target).abs().max(),
            "query_rmse": (query_control - query_target)
            .square()
            .mean()
            .sqrt(),
            "query_linf": (query_control - query_target).abs().max(),
            "support_loss": support_loss,
            "query_loss": query_loss,
            "start_anchor": start_anchor,
        }

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
        eps=1.0e-12,
    )
    restoration_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.support_restoration_learning_rate,
        weight_decay=0.0,
        eps=1.0e-12,
    )
    history: list[dict[str, Any]] = []
    candidates: list[
        tuple[float, float, float, dict[str, torch.Tensor], int]
    ] = []
    started = time.perf_counter()

    def record(epoch: int) -> None:
        with torch.enable_grad():
            pack = loss_pack()
        row = {
            key: float(value.detach().cpu())
            for key, value in pack.items()
        }
        row.update(
            {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        history.append(row)
        candidates.append(
            (
                row["query_rmse"],
                row["support_rmse"],
                row["query_linf"],
                clone_state(model),
                epoch,
            )
        )
        print(
            f"[epoch {epoch:04d}] query RMSE/Linf="
            f"{row['query_rmse']:.3e}/{row['query_linf']:.3e} "
            f"support RMSE={row['support_rmse']:.3e}",
            flush=True,
        )

    record(0)
    for epoch in range(1, args.epochs + 1):
        fraction = epoch / max(args.epochs, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * fraction))
        optimizer.param_groups[0]["lr"] = (
            args.final_learning_rate
            + cosine * (args.learning_rate - args.final_learning_rate)
        )
        restoration_optimizer.param_groups[0]["lr"] = (
            max(args.final_learning_rate, 1.0e-12)
            + cosine
            * (
                args.support_restoration_learning_rate
                - max(args.final_learning_rate, 1.0e-12)
            )
        )
        optimizer.zero_grad(set_to_none=True)
        pack = loss_pack()
        pack["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        for _ in range(max(args.support_restoration_steps, 0)):
            restoration_optimizer.zero_grad(set_to_none=True)
            restored_support = model(support_time)
            restoration_loss = (
                (restored_support - support_target) / cfg.umax
            ).square().mean()
            restoration_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip
            )
            restoration_optimizer.step()
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            record(epoch)

    initial_support_rmse = candidates[0][1]
    guarded = [
        item
        for item in candidates
        if item[1]
        <= args.support_rmse_guard_factor * initial_support_rmse
    ]
    selected = min(guarded or candidates, key=lambda item: item[:3])
    model.load_state_dict(selected[3], strict=True)
    params = build_params(cfg, device, torch.float64)
    grid_metrics, selected_control = evaluate(
        model, cfg, support_time, params, high_accuracy=True
    )
    payload = {
        **source,
        "model_state": selected[3],
        "source_checkpoint": str(start_checkpoint),
        "direct_reference": str(direct_solution),
        "teacher_free": False,
        "direct_or_manual_solution_used": True,
        "method": (
            "direct-reference initialization with strict off-grid ZOH query "
            "supervision"
        ),
        "selected_query_epoch": selected[4],
        "metrics": grid_metrics,
    }
    torch.save(payload, out_dir / "selected_checkpoint.pt")
    np.savez_compressed(
        out_dir / "solution.npz",
        t=physical_t,
        u=selected_control,
        direct_t=direct_t,
        direct_u=direct_u,
        support_target=support_target_numpy,
        query_t=query_physical,
        query_target=query_target.detach().cpu().numpy(),
    )
    fields: list[str] = []
    for row in history:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (out_dir / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "status": "completed",
        "device": str(device),
        "start_checkpoint": str(start_checkpoint),
        "direct_reference": str(direct_solution),
        "switch_times": [early_switch, late_switch],
        "query_count": int(query_time.numel()),
        "query_target_mode": args.query_target_mode,
        "training": vars(args),
        "selected_epoch": selected[4],
        "selected_query_rmse": selected[0],
        "selected_support_rmse": selected[1],
        "selected_query_linf": selected[2],
        "grid_metrics": grid_metrics,
        "wall_seconds": time.perf_counter() - started,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
