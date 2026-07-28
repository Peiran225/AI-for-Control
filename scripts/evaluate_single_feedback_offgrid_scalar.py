#!/usr/bin/env python3
"""Fresh continuous-policy scalar diagnostics for one feedback checkpoint.

The original 801-token Transformer support is kept fixed.  Additional time
queries are evaluated directly, their raw logits are interpolated by PCHIP,
and the state correction is queried continuously along a DOP853 rollout.
No query cache is read or written.
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
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.evaluate_feedback_section5 import load_feedback_checkpoint  # noqa: E402
from scripts.generate_offgrid_policy_switching_diagnostics import (  # noqa: E402
    DensePolicy,
    build_grid_flags,
    fixed_support_query_logits,
    integrate_trajectory,
    raw_time_logits,
    validate_numpy_feedback_action,
)
from scripts.generate_three_case_hamiltonian_results import (  # noqa: E402
    problem_from_config,
    resistant_heavy_initial_state,
)
from scripts.make_canonical_report_figures import (  # noqa: E402
    strict_singular_quantities,
)


QUANTITIES = ("H_u", "dH_u_dt", "d2H_u_dt2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
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
        return {
            "count": 0,
            "rms": None,
            "mean_abs": None,
            "max_abs": None,
        }
    return {
        "count": int(selected.size),
        "rms": float(np.sqrt(np.mean(selected**2))),
        "mean_abs": float(np.mean(np.abs(selected))),
        "max_abs": float(np.max(np.abs(selected))),
    }


def run(args: argparse.Namespace) -> None:
    if args.dense_points < 3:
        raise ValueError("--dense-points must be at least 3")
    if args.refinement_multiplier < 0:
        raise ValueError("--refinement-multiplier must be nonnegative")
    torch.set_num_threads(args.torch_threads)
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    checkpoint = args.checkpoint.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    checkpoint_payload = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    model, cfg, source_args = load_feedback_checkpoint(checkpoint)
    refinement_payload = checkpoint_payload.get("refinement")
    recorded_multiplier: int | None = None
    if isinstance(refinement_payload, dict) and refinement_payload.get(
        "multiplier"
    ) is not None:
        recorded_multiplier = int(refinement_payload["multiplier"])
    elif getattr(source_args, "offgrid_refinement_multiplier", None) is not None:
        recorded_multiplier = int(source_args.offgrid_refinement_multiplier)
    if args.refinement_multiplier:
        refinement_multiplier = int(args.refinement_multiplier)
        if (
            recorded_multiplier is not None
            and recorded_multiplier != refinement_multiplier
        ):
            raise ValueError(
                "requested refinement multiplier does not match checkpoint: "
                f"{refinement_multiplier} != {recorded_multiplier}"
            )
    else:
        refinement_multiplier = recorded_multiplier or 1
    out_dir.mkdir(parents=True, exist_ok=False)
    model = model.to(device=device, dtype=torch.float64)
    support = torch.linspace(
        0.0, 1.0, cfg.n + 1, device=device, dtype=torch.float64
    )
    dense_time = np.linspace(
        0.0, cfg.T, args.dense_points, dtype=np.float64
    )
    dense_normalized = torch.from_numpy(dense_time / cfg.T).to(
        device=device, dtype=torch.float64
    )
    on_support, nearest_index, _ = build_grid_flags(dense_time, cfg)
    if int(on_support.sum()) != cfg.n + 1:
        raise RuntimeError(
            f"expected {cfg.n + 1} support points, found "
            f"{int(on_support.sum())}"
        )
    dense_intervals = args.dense_points - 1
    refinement_intervals = cfg.n * refinement_multiplier
    if dense_intervals % refinement_intervals:
        raise ValueError(
            "dense-points - 1 must be divisible by "
            "n * refinement-multiplier"
        )
    refinement_stride = dense_intervals // refinement_intervals
    dense_index = np.arange(args.dense_points)
    on_refinement_grid = dense_index % refinement_stride == 0
    on_support_tensor = torch.from_numpy(on_support).to(device=device)
    support_indices = torch.from_numpy(
        nearest_index[on_support]
    ).to(device=device)
    with torch.inference_mode():
        support_raw = raw_time_logits(model.time_branch, support)
    dense_raw = torch.empty_like(dense_normalized)
    dense_raw[on_support_tensor] = support_raw[support_indices]
    off_support_tensor = ~on_support_tensor
    dense_raw[off_support_tensor] = fixed_support_query_logits(
        model.time_branch,
        support,
        dense_normalized[off_support_tensor],
        batch_size=args.query_batch_size,
    )
    reconstruction_error = float(
        (
            dense_raw[on_support_tensor] - support_raw[support_indices]
        )
        .abs()
        .max()
        .cpu()
    )

    model = model.cpu()
    policy = DensePolicy(
        "feedback_der",
        cfg,
        dense_time,
        dense_raw.cpu().numpy(),
        support_raw.cpu().numpy(),
        checkpoint,
        feedback_model=model,
        feedback_args=source_args,
    )
    action_validation_error = validate_numpy_feedback_action(policy)
    states = {
        "nominal": np.full(cfg.m, cfg.n0, dtype=np.float64),
        "resistant_heavy": resistant_heavy_initial_state(cfg),
    }
    interior = (
        (dense_time >= args.interior_start)
        & (dense_time < args.interior_end)
    )
    full_horizon = np.ones_like(interior)
    masks = {
        "full_horizon": full_horizon,
        "singular_interior": interior,
        "singular_interior_support": interior & on_support,
        "singular_interior_strict_off_grid": interior & ~on_support,
        "singular_interior_refinement_grid": (
            interior & on_refinement_grid
        ),
        "singular_interior_held_out_from_refinement": (
            interior & ~on_refinement_grid
        ),
    }

    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "initialization_checkpoint": checkpoint_payload.get(
            "initialization_checkpoint"
        ),
        "refinement": checkpoint_payload.get("refinement"),
        "selection_metric": checkpoint_payload.get("selection_metric"),
        "problem": vars(cfg),
        "dense_points": args.dense_points,
        "refinement_multiplier": refinement_multiplier,
        "checkpoint_refinement_multiplier": recorded_multiplier,
        "interior": [args.interior_start, args.interior_end],
        "report_scale_factor": args.report_scale_factor,
        "query_device": str(device),
        "time_context_tokens": cfg.n + 1,
        "fixed_support_on_grid_reconstruction_max_abs": (
            reconstruction_error
        ),
        "numpy_vs_torch_action_max_abs": action_validation_error,
        "states": {},
    }
    saved: dict[str, np.ndarray] = {"t": dense_time}
    for name, initial_state in states.items():
        result = integrate_trajectory(
            policy,
            initial_state,
            dense_time,
            rtol=args.rtol,
            atol=args.atol,
            max_step=cfg.T / (args.dense_points - 1),
        )
        strict = strict_singular_quantities(
            {
                "diagnostic_N": result.state,
                "diagnostic_lambda": result.costate,
                "diagnostic_u": result.control,
            },
            problem_from_config(cfg),
        )
        strict_control = np.asarray(
            strict["strict_control"], dtype=np.float64
        )
        closed_form_error = result.control - strict_control
        finite_closed_form = np.isfinite(strict_control)
        state_summary: dict[str, Any] = {
            "initial_state": initial_state.tolist(),
            "normalized_objective": result.normalized_objective,
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
            state_summary["metrics"][mask_name][
                "closed_form_control_error"
            ] = summarize(
                closed_form_error,
                mask & finite_closed_form,
                scale=1.0,
            )
        summary["states"][name] = state_summary
        prefix = f"{name}__"
        saved[prefix + "u"] = result.control
        saved[prefix + "closed_form_u"] = strict_control
        saved[prefix + "closed_form_control_error"] = closed_form_error
        saved[prefix + "B"] = np.asarray(strict["B"], dtype=np.float64)
        saved[prefix + "N"] = result.state
        saved[prefix + "costate"] = result.costate
        saved[prefix + "H"] = (
            args.report_scale_factor * result.hamiltonian
        )
        for quantity in QUANTITIES:
            saved[prefix + quantity] = (
                args.report_scale_factor * result.quantities[quantity]
            )
    saved["is_transformer_support"] = on_support
    saved["is_refinement_grid"] = on_refinement_grid
    saved["is_held_out_from_refinement"] = ~on_refinement_grid
    saved["is_singular_interior"] = interior
    np.savez_compressed(out_dir / "timeseries.npz", **saved)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dense-points", type=int, default=12801)
    parser.add_argument(
        "--refinement-multiplier",
        type=int,
        default=0,
        help=(
            "number of refinement subintervals per original Transformer "
            "interval; 0 reads the value from checkpoint metadata"
        ),
    )
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--interior-start", type=float, default=1.5)
    parser.add_argument("--interior-end", type=float, default=8.0)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--report-scale-factor", type=float, default=400.0)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--device", default="auto")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
