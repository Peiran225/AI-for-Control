#!/usr/bin/env python3
"""Refine a direct-initialized time-only Transformer on scalar singular conditions."""

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

from fit_current_direct_reference_transformer_full import (  # noqa: E402
    AffineBoundaryProjectedControl,
)
from refine_time_only_singular_plateau import build_model  # noqa: E402
from train_feedback_section5 import (  # noqa: E402
    compute_costate_rk4,
    simulate_open_loop,
    singular_quantities,
)
from train_paper_pmp_kkt import (  # noqa: E402
    ProblemConfig,
    build_params,
    set_seed,
)
from train_teacher_free_resolution_curriculum import evaluate  # noqa: E402


def resolve(path: str | Path) -> Path:
    expanded = Path(path).expanduser()
    return (expanded if expanded.is_absolute() else ROOT / expanded).resolve()


def clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def load_model(
    checkpoint: Path,
    device: torch.device,
) -> tuple[AffineBoundaryProjectedControl, ProblemConfig, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    wrapper = dict(payload.get("wrapper", {}))
    if wrapper.get("class") != "AffineBoundaryProjectedControl":
        raise ValueError(
            "start checkpoint must use AffineBoundaryProjectedControl"
        )
    cfg = ProblemConfig(**payload["problem"])
    base = build_model(dict(payload["base_model_args"]), cfg)
    model = AffineBoundaryProjectedControl(
        base,
        umax=cfg.umax,
        scale=float(wrapper["scale"]),
        offset=float(wrapper["offset"]),
    ).to(device=device, dtype=torch.float64)
    model.load_state_dict(payload["model_state"], strict=True)
    return model, cfg, payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-checkpoint", required=True)
    parser.add_argument("--direct-solution", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=2607)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1.0e-7)
    parser.add_argument("--final-learning-rate", type=float, default=1.0e-9)
    parser.add_argument(
        "--psi-scale",
        type=float,
        default=2.5e-6,
        help="Training-objective scale (physical scale divided by 400).",
    )
    parser.add_argument(
        "--dot-scale",
        type=float,
        default=1.25e-5,
        help="Training-objective scale (physical scale divided by 400).",
    )
    parser.add_argument(
        "--ddot-scale",
        type=float,
        default=1.25e-4,
        help="Training-objective scale (physical scale divided by 400).",
    )
    parser.add_argument("--w0", type=float, default=1.0)
    parser.add_argument("--w1", type=float, default=1.0)
    parser.add_argument("--w2", type=float, default=1.0)
    parser.add_argument("--direct-anchor-weight", type=float, default=1.0)
    parser.add_argument("--start-anchor-weight", type=float, default=0.0)
    parser.add_argument("--interior-start", type=float, default=1.5)
    parser.add_argument("--interior-end", type=float, default=8.0)
    parser.add_argument(
        "--costate-pairing", choices=("current", "next"), default="current"
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--report-scale-factor", type=float, default=400.0)
    args = parser.parse_args()

    set_seed(args.seed)
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
    if cfg.n != 800:
        raise ValueError(f"expected n=800, received n={cfg.n}")
    params = build_params(cfg, device, torch.float64)
    normalized_t = torch.linspace(
        0.0, 1.0, cfg.n + 1, device=device, dtype=torch.float64
    )
    physical_interval_t = (
        cfg.T * normalized_t[:-1].detach().cpu().numpy()
    )
    interior_numpy = (
        (physical_interval_t >= args.interior_start)
        & (physical_interval_t < args.interior_end)
    )
    interior = torch.tensor(
        interior_numpy, device=device, dtype=torch.bool
    ).unsqueeze(0)
    initial_state = torch.full(
        (1, cfg.m),
        cfg.n0,
        device=device,
        dtype=torch.float64,
    )
    direct = np.load(direct_solution)
    direct_t = np.asarray(direct["t"], dtype=np.float64)
    direct_u = np.asarray(direct["u"], dtype=np.float64)
    expected_t = np.linspace(0.0, cfg.T, cfg.n + 1)
    if not np.allclose(direct_t, expected_t, atol=1.0e-12, rtol=0.0):
        raise ValueError("direct solution grid does not match checkpoint")
    direct_target = torch.tensor(
        direct_u, device=device, dtype=torch.float64
    )
    model.eval()
    with torch.enable_grad():
        start_control = model(normalized_t).detach()

    def loss_pack() -> dict[str, torch.Tensor]:
        all_control = model(normalized_t)
        controls = all_control[:-1].unsqueeze(0)
        states = simulate_open_loop(
            controls,
            initial_state,
            cfg,
            params,
            integrator="rk4",
        )
        costates, _ = compute_costate_rk4(
            states, controls, cfg, params
        )
        quantities = singular_quantities(
            states,
            controls,
            costates,
            cfg,
            params,
            costate_pairing=args.costate_pairing,
        )
        psi = quantities["psi"][interior]
        dot = quantities["dot_psi"][interior]
        ddot = quantities["ddot_psi"][interior]
        psi_term = (psi / args.psi_scale).square().mean()
        dot_term = (dot / args.dot_scale).square().mean()
        ddot_term = (ddot / args.ddot_scale).square().mean()
        scalar = (
            args.w0 * psi_term
            + args.w1 * dot_term
            + args.w2 * ddot_term
        )
        direct_anchor = (
            (all_control - direct_target) / cfg.umax
        ).square().mean()
        start_anchor = (
            (all_control - start_control) / cfg.umax
        ).square().mean()
        total = (
            scalar
            + args.direct_anchor_weight * direct_anchor
            + args.start_anchor_weight * start_anchor
        )
        return {
            "loss": total,
            "scalar_loss": scalar,
            "psi_term": psi_term,
            "dot_term": dot_term,
            "ddot_term": ddot_term,
            "direct_anchor": direct_anchor,
            "start_anchor": start_anchor,
            "psi_rms": psi.square().mean().sqrt(),
            "dot_rms": dot.square().mean().sqrt(),
            "ddot_rms": ddot.square().mean().sqrt(),
            "direct_rmse": (all_control - direct_target).square().mean().sqrt(),
        }

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
        eps=1.0e-12,
    )
    history: list[dict[str, Any]] = []
    candidates: list[
        tuple[float, float, float, float, dict[str, torch.Tensor], int]
    ] = []
    started = time.perf_counter()

    def record(epoch: int) -> None:
        model.eval()
        with torch.enable_grad():
            pack = loss_pack()
        row = {
            key: float(value.detach().cpu())
            for key, value in pack.items()
        }
        score = math.sqrt(
            row["psi_rms"] ** 2
            + row["dot_rms"] ** 2
            + row["ddot_rms"] ** 2
        )
        candidates.append(
            (
                score,
                row["psi_rms"],
                row["dot_rms"],
                row["ddot_rms"],
                clone_state(model),
                epoch,
            )
        )
        row.update(
            {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        history.append(row)
        print(
            f"[epoch {epoch:04d}] physical raw RMS "
            f"{args.report_scale_factor * row['psi_rms']:.3e}/"
            f"{args.report_scale_factor * row['dot_rms']:.3e}/"
            f"{args.report_scale_factor * row['ddot_rms']:.3e} directRMSE="
            f"{row['direct_rmse']:.3e}",
            flush=True,
        )
        model.train()

    record(0)
    for epoch in range(1, args.epochs + 1):
        fraction = epoch / max(args.epochs, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * fraction))
        learning_rate = args.final_learning_rate + cosine * (
            args.learning_rate - args.final_learning_rate
        )
        optimizer.param_groups[0]["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        pack = loss_pack()
        pack["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            record(epoch)

    selected = min(candidates, key=lambda item: item[:4])
    model.load_state_dict(selected[4], strict=True)
    grid_metrics, selected_control = evaluate(
        model, cfg, normalized_t, params, high_accuracy=True
    )
    output_payload = {
        **source,
        "model_state": selected[4],
        "source_checkpoint": str(start_checkpoint),
        "direct_reference": str(direct_solution),
        "teacher_free": False,
        "direct_or_manual_solution_used": True,
        "method": (
            "direct-reference initialization followed by scalar singular-"
            "condition refinement"
        ),
        "selected_scalar_epoch": selected[5],
        "scalar_grid_rms": {
            "H_u": args.report_scale_factor * selected[1],
            "dH_u_dt": args.report_scale_factor * selected[2],
            "d2H_u_dt2": args.report_scale_factor * selected[3],
        },
        "metrics": grid_metrics,
    }
    torch.save(output_payload, out_dir / "selected_checkpoint.pt")
    np.savez_compressed(
        out_dir / "solution.npz",
        t=expected_t,
        u=selected_control,
        direct_u=direct_u,
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
        "start_checkpoint": str(start_checkpoint),
        "direct_reference": str(direct_solution),
        "device": str(device),
        "training": vars(args),
        "selected_epoch": selected[5],
        "selected_scalar_grid_rms": output_payload["scalar_grid_rms"],
        "grid_metrics": grid_metrics,
        "wall_seconds": time.perf_counter() - started,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
