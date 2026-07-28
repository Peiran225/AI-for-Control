#!/usr/bin/env python3
"""Evaluate one time-only Transformer on strict off-grid PMP scalar diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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


def resolve(path: Path) -> Path:
    expanded = path.expanduser()
    return (expanded if expanded.is_absolute() else ROOT / expanded).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dense-points", type=int, default=1601)
    parser.add_argument(
        "--refinement-multiplier",
        type=int,
        default=1,
        help=(
            "number of scalar-refinement subintervals per original Transformer "
            "interval; points between this grid and the dense evaluation grid "
            "are reported as held out"
        ),
    )
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--interior-start", type=float, default=1.5)
    parser.add_argument("--interior-end", type=float, default=8.0)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--report-scale-factor",
        type=float,
        default=400.0,
        help="Convert the normalized training objective to physical weights.",
    )
    args = parser.parse_args()
    if args.refinement_multiplier < 1:
        raise ValueError("--refinement-multiplier must be positive")

    torch.set_num_threads(args.torch_threads)
    # The fused inference-only MHA kernel can differ from the differentiable
    # Transformer path enough to perturb this highly sensitive control
    # problem.  Disable it so diagnostic queries use the same arithmetic path
    # as training and checkpoint selection.
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
    on_grid, nearest_index, _ = build_grid_flags(dense_time, cfg)
    if int(on_grid.sum()) != cfg.n + 1:
        raise RuntimeError(
            f"dense grid has {int(on_grid.sum())} training coordinates; "
            f"expected {cfg.n + 1}"
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
    refinement_query = on_refinement_grid & ~on_grid
    held_out_from_refinement = ~on_refinement_grid

    with torch.inference_mode():
        coarse_raw = raw_time_logits(model, support)
    dense_raw = torch.empty_like(dense_normalized)
    on_grid_tensor = torch.from_numpy(on_grid).to(device=device)
    dense_raw[on_grid_tensor] = coarse_raw[
        torch.from_numpy(nearest_index[on_grid]).to(device=device)
    ]
    off_grid_tensor = torch.from_numpy(~on_grid).to(device=device)
    dense_raw[off_grid_tensor] = fixed_support_query_logits(
        model,
        support,
        dense_normalized[off_grid_tensor],
        batch_size=args.query_batch_size,
    )

    policy = DensePolicy(
        "time_only",
        cfg,
        dense_time,
        dense_raw.cpu().numpy(),
        coarse_raw.cpu().numpy(),
        checkpoint,
        time_wrapper=wrapper,
    )
    initial_state = np.full(cfg.m, cfg.n0, dtype=np.float64)
    result = integrate_trajectory(
        policy,
        initial_state,
        dense_time,
        rtol=args.rtol,
        atol=args.atol,
        max_step=cfg.T / (args.dense_points - 1),
    )
    if not hasattr(result, "normalized_running_cost"):
        raise RuntimeError(
            "the loaded trajectory evaluator returned an incompatible result: "
            f"type={type(result)!r}, fields={sorted(vars(result))}"
        )

    interior = (
        (result.time >= args.interior_start)
        & (result.time < args.interior_end)
    )
    subsets = {
        "all": interior,
        "transformer_support": interior & on_grid,
        "refinement_queries": interior & refinement_query,
        "refinement_grid": interior & on_refinement_grid,
        "held_out_from_refinement": interior & held_out_from_refinement,
        # Backward-compatible names retained for earlier audit scripts.
        "training_grid": interior & on_grid,
        "strict_off_grid": interior & ~on_grid,
    }
    summary: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "problem": vars(cfg),
        "wrapper": wrapper,
        "dense_points": args.dense_points,
        "refinement_multiplier": args.refinement_multiplier,
        "query_device": str(device),
        "interior": [args.interior_start, args.interior_end],
        "report_scale_factor": args.report_scale_factor,
        "on_grid_reconstruction_max_abs": float(
            np.max(
                np.abs(
                    dense_raw[on_grid_tensor].cpu().numpy()
                    - coarse_raw[
                        torch.from_numpy(nearest_index[on_grid]).to(
                            device=device
                        )
                    ]
                    .cpu()
                    .numpy()
                )
            )
        ),
        "identity_errors": result.identity_errors,
        "objective": {
            "normalized_running_cost": result.normalized_running_cost,
            "normalized_objective": result.normalized_objective,
            "physical_running_cost": (
                args.report_scale_factor * result.normalized_running_cost
            ),
            "physical_objective": (
                args.report_scale_factor * result.normalized_objective
            ),
            "method": (
                "augmented-state DOP853 under the same continuously queried "
                "policy used for the scalar diagnostics"
            ),
        },
        "metrics": {},
    }
    metrics = summary["metrics"]
    assert isinstance(metrics, dict)
    for subset, mask in subsets.items():
        metrics[subset] = {"count": int(mask.sum())}
        for quantity, values in result.quantities.items():
            selected = args.report_scale_factor * np.asarray(
                values[mask], dtype=np.float64
            )
            if selected.size:
                metrics[subset][quantity] = {
                    "rms": float(np.sqrt(np.mean(selected**2))),
                    "mean_abs": float(np.mean(np.abs(selected))),
                    "max_abs": float(np.max(np.abs(selected))),
                }
            else:
                metrics[subset][quantity] = {
                    "rms": None,
                    "mean_abs": None,
                    "max_abs": None,
                }

    np.savez_compressed(
        out_dir / "timeseries.npz",
        t=result.time,
        u=result.control,
        N=result.state,
        costate=result.costate,
        H=args.report_scale_factor * result.hamiltonian,
        H_u=args.report_scale_factor * result.quantities["H_u"],
        dH_u_dt=args.report_scale_factor * result.quantities["dH_u_dt"],
        d2H_u_dt2=args.report_scale_factor
        * result.quantities["d2H_u_dt2"],
        is_training_grid=on_grid,
        is_transformer_support=on_grid,
        is_refinement_query=refinement_query,
        is_held_out_from_refinement=held_out_from_refinement,
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
