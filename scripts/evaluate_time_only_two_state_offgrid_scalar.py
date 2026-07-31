#!/usr/bin/env python3
"""Evaluate one time-only checkpoint on two states with dense policy queries.

This is the two-state companion to ``evaluate_time_only_offgrid_scalar.py``.
It deliberately reuses that evaluator's fixed-support Transformer queries,
continuous DOP853 rollout, and analytic scalar Hamiltonian identities.  The
only added operation is replaying the same time-only policy from the nominal
and resistant-heavy initial states.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for search_path in (ROOT, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from generate_offgrid_policy_switching_diagnostics import (  # noqa: E402
    DensePolicy,
    build_grid_flags,
    fixed_support_query_logits,
    integrate_trajectory,
    load_time_model,
    raw_time_logits,
)
QUANTITIES = ("H_u", "dH_u_dt", "d2H_u_dt2")


def resolve(path: Path) -> Path:
    expanded = path.expanduser()
    return (expanded if expanded.is_absolute() else ROOT / expanded).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(
    values: np.ndarray,
    mask: np.ndarray,
    *,
    scale: float,
) -> dict[str, float | int | None]:
    selected = scale * np.asarray(values[mask], dtype=np.float64)
    if selected.size == 0:
        return {"count": 0, "rms": None, "mean_abs": None, "max_abs": None}
    return {
        "count": int(selected.size),
        "rms": float(np.sqrt(np.mean(selected**2))),
        "mean_abs": float(np.mean(np.abs(selected))),
        "max_abs": float(np.max(np.abs(selected))),
    }


def run(args: argparse.Namespace) -> None:
    if args.dense_points < 3:
        raise ValueError("--dense-points must be at least 3")
    if args.refinement_multiplier < 1:
        raise ValueError("--refinement-multiplier must be positive")
    if not 0.0 <= args.resistant_radius < 1.0:
        raise ValueError("--resistant-radius must lie in [0,1)")
    torch.set_num_threads(args.torch_threads)
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)

    checkpoint = resolve(args.checkpoint)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    model, cfg, wrapper = load_time_model(checkpoint)
    model = model.to(device=device, dtype=torch.float64)
    dense_time = np.linspace(0.0, cfg.T, args.dense_points, dtype=np.float64)
    dense_normalized = torch.from_numpy(dense_time / cfg.T).to(
        device=device, dtype=torch.float64
    )
    support = torch.linspace(
        0.0, 1.0, cfg.n + 1, device=device, dtype=torch.float64
    )
    on_support, nearest_index, _ = build_grid_flags(dense_time, cfg)
    if int(on_support.sum()) != cfg.n + 1:
        raise RuntimeError(
            f"expected {cfg.n + 1} support points, found {int(on_support.sum())}"
        )

    dense_intervals = args.dense_points - 1
    refinement_intervals = cfg.n * args.refinement_multiplier
    if dense_intervals % refinement_intervals:
        raise ValueError(
            "dense-points - 1 must be divisible by "
            "n * refinement-multiplier"
        )
    refinement_stride = dense_intervals // refinement_intervals
    dense_index = np.arange(args.dense_points)
    on_refinement_grid = dense_index % refinement_stride == 0

    with torch.inference_mode():
        support_raw = raw_time_logits(model, support)
    dense_raw = torch.empty_like(dense_normalized)
    on_support_tensor = torch.from_numpy(on_support).to(device=device)
    dense_raw[on_support_tensor] = support_raw[
        torch.from_numpy(nearest_index[on_support]).to(device=device)
    ]
    off_support_tensor = ~on_support_tensor
    dense_raw[off_support_tensor] = fixed_support_query_logits(
        model,
        support,
        dense_normalized[off_support_tensor],
        batch_size=args.query_batch_size,
    )
    reconstruction_error = float(
        (
            dense_raw[on_support_tensor]
            - support_raw[
                torch.from_numpy(nearest_index[on_support]).to(device=device)
            ]
        )
        .abs()
        .max()
        .cpu()
    )

    policy = DensePolicy(
        "time_only",
        cfg,
        dense_time,
        dense_raw.cpu().numpy(),
        support_raw.cpu().numpy(),
        checkpoint,
        time_wrapper=wrapper,
    )
    nominal = np.full(cfg.m, cfg.n0, dtype=np.float64)
    resistant_direction = np.linspace(-1.0, 1.0, cfg.m, dtype=np.float64)
    resistant = cfg.n0 * (
        1.0 + args.resistant_radius * resistant_direction
    )
    states = {
        "nominal": nominal,
        "resistant_heavy": resistant,
    }
    interior = (
        (dense_time >= args.interior_start) & (dense_time < args.interior_end)
    )
    masks = {
        "full_horizon": np.ones_like(interior),
        "singular_interior": interior,
        "singular_interior_support": interior & on_support,
        "singular_interior_strict_off_grid": interior & ~on_support,
        "singular_interior_refinement_grid": interior & on_refinement_grid,
        "singular_interior_held_out_from_refinement": (
            interior & ~on_refinement_grid
        ),
    }

    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "problem": vars(cfg),
        "wrapper": wrapper,
        "dense_points": args.dense_points,
        "refinement_multiplier": args.refinement_multiplier,
        "interior": [args.interior_start, args.interior_end],
        "report_scale_factor": args.report_scale_factor,
        "query_device": str(device),
        "time_context_tokens": cfg.n + 1,
        "structured_resistant_state": {
            "radius": args.resistant_radius,
            "direction": resistant_direction.tolist(),
            "definition": (
                "N_i(0)=n0*(1+r*z_i), with z_i linearly spaced "
                "from -1 to 1"
            ),
            "initial_state": resistant.tolist(),
            "initial_total": float(resistant.sum()),
        },
        "fixed_support_on_grid_reconstruction_max_abs": reconstruction_error,
        "states": {},
    }
    saved: dict[str, np.ndarray] = {
        "t": dense_time,
        "is_transformer_support": on_support,
        "is_refinement_grid": on_refinement_grid,
        "is_singular_interior": interior,
    }
    for name, initial_state in states.items():
        result = integrate_trajectory(
            policy,
            initial_state,
            dense_time,
            rtol=args.rtol,
            atol=args.atol,
            max_step=cfg.T / (args.dense_points - 1),
        )
        state_summary: dict[str, Any] = {
            "initial_state": initial_state.tolist(),
            "normalized_running_cost": result.normalized_running_cost,
            "normalized_objective": result.normalized_objective,
            "physical_running_cost": (
                args.report_scale_factor * result.normalized_running_cost
            ),
            "physical_objective": (
                args.report_scale_factor * result.normalized_objective
            ),
            "identity_errors": result.identity_errors,
            "metrics": {},
        }
        for mask_name, mask in masks.items():
            state_summary["metrics"][mask_name] = {
                quantity: summarize(
                    result.quantities[quantity],
                    mask,
                    scale=args.report_scale_factor,
                )
                for quantity in QUANTITIES
            }
        summary["states"][name] = state_summary

        prefix = f"{name}__"
        saved[prefix + "u"] = result.control
        saved[prefix + "N"] = result.state
        saved[prefix + "costate"] = result.costate
        saved[prefix + "H"] = (
            args.report_scale_factor * result.hamiltonian
        )
        for quantity in QUANTITIES:
            saved[prefix + quantity] = (
                args.report_scale_factor * result.quantities[quantity]
            )

    np.savez_compressed(out_dir / "timeseries.npz", **saved)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dense-points", type=int, default=12801)
    parser.add_argument("--refinement-multiplier", type=int, default=8)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--interior-start", type=float, default=1.5)
    parser.add_argument("--interior-end", type=float, default=8.0)
    parser.add_argument("--resistant-radius", type=float, default=0.10)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--report-scale-factor", type=float, default=400.0)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--device", default="auto")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
