#!/usr/bin/env python3
"""Strict teacher-free residual continuation for the n=800 Transformer.

This program never reads a direct/manual control, a switching time/mask, or an
objective value for supervision or checkpoint selection.  It differentiates
the RK4 reduced objective only to form the complete projected-gradient mapping

    G(u) = u - projection_[0,umax](u - grad F_h(u)),

where the rollout inside ``F_h`` retains the full N=N(u) dependence.  Network
parameters are refined with L-BFGS on a smooth high-p residual objective.
Checkpoint selection is lexicographic in ||G||_inf and ||G||_2 subject to a
generic non-degeneracy/shape guard measured relative to the input checkpoint.
Diagnostic objective values are deliberately withheld until after selection.
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
    evaluate,
    interval_crossing_width,
    make_plot,
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


def clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def residual_pack(
    model: torch.nn.Module,
    normalized_t: torch.Tensor,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
    *,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    control = model(normalized_t)
    interval = control[:-1]
    objective = rk4_reduced_objective(interval, cfg, params)
    gradient = torch.autograd.grad(
        objective,
        interval,
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    mapping = interval - torch.clamp(interval - gradient, 0.0, cfg.umax)
    return control, gradient, mapping


def shape_metrics(
    physical_t: np.ndarray,
    control: np.ndarray,
    cfg: ProblemConfig,
) -> dict[str, Any]:
    interval = np.asarray(control[:-1], dtype=np.float64)
    edge = max(2, cfg.n // 20)
    interior = interval[edge:-edge]
    early = float(np.median(interval[:edge]))
    late = float(np.median(interval[-edge:]))
    interior_low = float(np.quantile(interior, 0.10))
    dynamic_range = float(np.ptp(interval))
    upper = int(np.count_nonzero(interval == cfg.umax))
    lower = int(np.count_nonzero(interval == 0.0))
    nondegenerate = bool(
        dynamic_range >= 0.8
        and early >= interior_low + 0.5
        and late >= interior_low + 0.5
        and upper < int(0.95 * cfg.n)
        and lower < int(0.95 * cfg.n)
    )
    return {
        "nondegenerate_high_low_high": nondegenerate,
        "early_width_10_90": float(
            interval_crossing_width(physical_t, control, "early")
        ),
        "late_width_10_90": float(
            interval_crossing_width(physical_t, control, "late")
        ),
        "total_variation": float(np.abs(np.diff(interval)).sum()),
        "second_difference_l1": float(np.abs(np.diff(interval, n=2)).sum()),
        "u_min": float(interval.min()),
        "u_max": float(interval.max()),
        "exact_upper_bound_count": upper,
        "exact_lower_bound_count": lower,
    }


def residual_metrics(
    model: torch.nn.Module,
    normalized_t: torch.Tensor,
    physical_t: np.ndarray,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    with torch.enable_grad():
        control, gradient, mapping = residual_pack(
            model, normalized_t, cfg, params, create_graph=False
        )
    u = control.detach().cpu().numpy()
    g = gradient.detach().cpu().numpy()
    residual = mapping.detach().cpu().numpy()
    row = {
        "projected_gradient_linf": float(np.max(np.abs(residual))),
        "projected_gradient_rms": float(np.sqrt(np.mean(residual**2))),
        "raw_gradient_linf": float(np.max(np.abs(g))),
        **shape_metrics(physical_t, u, cfg),
    }
    model.train()
    return row, u, residual


def passes_shape_guard(
    row: dict[str, Any],
    baseline: dict[str, Any],
    width_tolerance: float,
    variation_fraction_tolerance: float,
) -> bool:
    return bool(
        row["nondegenerate_high_low_high"]
        and math.isfinite(row["early_width_10_90"])
        and math.isfinite(row["late_width_10_90"])
        and row["early_width_10_90"]
        <= baseline["early_width_10_90"] + width_tolerance
        and row["late_width_10_90"]
        <= baseline["late_width_10_90"] + width_tolerance
        and row["total_variation"]
        <= baseline["total_variation"] * (1.0 + variation_fraction_tolerance)
        and row["second_difference_l1"]
        <= baseline["second_difference_l1"]
        * (1.0 + variation_fraction_tolerance)
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--outer-steps", type=int, default=12)
    parser.add_argument("--inner-iterations", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.3)
    parser.add_argument("--history-size", type=int, default=30)
    parser.add_argument("--p", type=float, default=12.0)
    parser.add_argument("--high-p-weight", type=float, default=0.5)
    parser.add_argument(
        "--fixed-temperature",
        type=float,
        default=None,
        help=(
            "Optional global output temperature imposed before continuation. "
            "It contains no time, switching-location, or target-control information."
        ),
    )
    parser.add_argument(
        "--learn-temperature",
        action="store_true",
        help=(
            "Optimize one global, time-independent output temperature together "
            "with the network parameters.  This adds no switch location or mask."
        ),
    )
    parser.add_argument("--width-tolerance", type=float, default=0.005)
    parser.add_argument("--variation-fraction-tolerance", type=float, default=0.05)
    args = parser.parse_args()

    if args.p <= 2.0:
        raise ValueError("p must exceed 2")
    if args.fixed_temperature is not None and not (
        0.25 <= args.fixed_temperature <= 1.5
    ):
        raise ValueError("fixed temperature must lie in [0.25, 1.5]")
    if args.fixed_temperature is not None and args.learn_temperature:
        raise ValueError("choose either a fixed or a learnable temperature")
    set_seed(args.seed)
    torch.set_num_threads(1)
    checkpoint_path = resolve(args.checkpoint)
    out_dir = resolve(args.out_dir)
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True)
    prohibited = ("direct", "manual", "supervised", "distill")
    if any(token in str(checkpoint_path).lower() for token in prohibited):
        raise ValueError(f"prohibited teacher-like checkpoint: {checkpoint_path}")
    source = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if source.get("teacher_free") is not True:
        raise ValueError("source checkpoint lacks teacher_free provenance")
    cfg = ProblemConfig(**source["problem"])
    if cfg.n != 800:
        raise ValueError(f"expected n=800, received n={cfg.n}")
    wrapper = dict(source["wrapper"])
    base = build_model(dict(source["base_model_args"]), cfg).to(dtype=torch.float64)
    model = FixedBoxProjection(
        base,
        cfg.umax,
        float(wrapper["scale"]),
        temperature=float(wrapper.get("temperature", 1.0)),
        learn_temperature=args.learn_temperature,
    ).to(dtype=torch.float64)
    model.load_state_dict(source["model_state"])
    if args.fixed_temperature is not None:
        with torch.no_grad():
            model.raw_temperature.fill_(math.log(args.fixed_temperature))
        wrapper["temperature"] = float(args.fixed_temperature)
        wrapper["learn_temperature"] = False
    model.train()

    normalized_t = torch.linspace(0.0, 1.0, cfg.n + 1, dtype=torch.float64)
    physical_t = np.linspace(0.0, cfg.T, cfg.n + 1)
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    dt = cfg.T / cfg.n
    baseline, baseline_u, baseline_residual = residual_metrics(
        model, normalized_t, physical_t, cfg, params
    )
    baseline["global_output_temperature"] = float(
        model.temperature_value().detach()
    )
    baseline["shape_guard_pass"] = True
    baseline["outer_step"] = 0
    history: list[dict[str, Any]] = [dict(baseline)]
    candidates: list[
        tuple[dict[str, Any], dict[str, torch.Tensor], np.ndarray, np.ndarray]
    ] = [(dict(baseline), clone_state(model), baseline_u, baseline_residual)]

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=args.learning_rate,
        max_iter=args.inner_iterations,
        max_eval=max(args.inner_iterations * 2, args.inner_iterations + 2),
        tolerance_grad=1.0e-12,
        tolerance_change=1.0e-15,
        history_size=args.history_size,
        line_search_fn="strong_wolfe",
    )
    closure_calls = 0
    started = time.perf_counter()

    for outer_step in range(1, args.outer_steps + 1):
        def closure() -> torch.Tensor:
            nonlocal closure_calls
            optimizer.zero_grad(set_to_none=True)
            _, _, mapping = residual_pack(
                model, normalized_t, cfg, params, create_graph=True
            )
            scaled = mapping / dt
            rms_square = scaled.square().mean()
            # The normalized p-mean approaches max(abs(scaled)) as p grows,
            # but remains smooth away from exact zero.
            p_mean_square = (
                (scaled.square() + 1.0e-24).pow(0.5 * args.p).mean()
            ).pow(2.0 / args.p)
            loss = rms_square + args.high_p_weight * p_mean_square
            loss.backward()
            closure_calls += 1
            return loss

        loss = float(optimizer.step(closure).detach())
        row, control, residual = residual_metrics(
            model, normalized_t, physical_t, cfg, params
        )
        row["global_output_temperature"] = float(
            model.temperature_value().detach()
        )
        row.update(
            {
                "outer_step": outer_step,
                "lbfgs_reported_loss_before_step": loss,
                "closure_calls": closure_calls,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        row["shape_guard_pass"] = passes_shape_guard(
            row,
            baseline,
            args.width_tolerance,
            args.variation_fraction_tolerance,
        )
        history.append(dict(row))
        if row["shape_guard_pass"]:
            candidates.append((dict(row), clone_state(model), control, residual))
        print(
            f"[outer={outer_step:02d}] "
            f"PGinf={row['projected_gradient_linf']:.3e} "
            f"PGrms={row['projected_gradient_rms']:.3e} "
            f"width={row['early_width_10_90']:.5f}/"
            f"{row['late_width_10_90']:.5f} "
            f"guard={row['shape_guard_pass']} closures={closure_calls}",
            flush=True,
        )

    selected_row, selected_state, selected_u, selected_residual = min(
        candidates,
        key=lambda item: (
            item[0]["projected_gradient_linf"],
            item[0]["projected_gradient_rms"],
        ),
    )
    model.load_state_dict(selected_state)
    verify_row, verify_u, verify_residual = residual_metrics(
        model, normalized_t, physical_t, cfg, params
    )
    verify_row["global_output_temperature"] = float(
        model.temperature_value().detach()
    )
    if np.max(np.abs(verify_u - selected_u)) > 1.0e-12:
        raise RuntimeError("selected checkpoint reload mismatch")
    if np.max(np.abs(verify_residual - selected_residual)) > 1.0e-12:
        raise RuntimeError("selected residual reload mismatch")

    # Selection is now frozen.  Only at this point compute the standard
    # high-accuracy diagnostics, which include J as a reporting-only quantity.
    final_metrics, final_u = evaluate(
        model, cfg, normalized_t, params, high_accuracy=True
    )
    selected_outer = int(selected_row["outer_step"])
    wrapper["temperature"] = float(model.temperature_value().detach())
    wrapper["learn_temperature"] = bool(args.learn_temperature)
    payload = {
        "model_state": selected_state,
        "base_model_args": source["base_model_args"],
        "problem": source["problem"],
        "wrapper": wrapper,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "teacher_free": True,
        "method": "network-parameter L-BFGS on complete projected full-gradient residual",
        "objective_value_used_as_loss_or_selection": False,
        "direct_or_manual_solution_used": False,
        "switching_time_or_mask_used": False,
        "selected_outer_step": selected_outer,
        "selection_residual_metrics": verify_row,
        "post_selection_diagnostics": final_metrics,
    }
    torch.save(payload, out_dir / "selected_checkpoint.pt")
    np.savez_compressed(
        out_dir / "solution.npz",
        t=physical_t,
        u=final_u,
        projected_gradient_mapping=verify_residual,
    )
    write_csv(out_dir / "history.csv", history)
    make_plot(out_dir / "teacher_free_control", physical_t, final_u)
    provenance = {
        "teacher_free": True,
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256(checkpoint_path),
        "training_reads_direct_or_manual_solution": False,
        "training_uses_switching_time_or_mask": False,
        "training_uses_objective_value_as_loss_or_selection": False,
        "full_gradient_includes_state_dependence": True,
        "selection": (
            "minimum projected-gradient Linf then RMS among candidates passing "
            "the source-relative generic shape guard"
        ),
        "post_selection_diagnostic_objective_only": True,
        "arguments": vars(args),
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "status": "completed",
        "method": payload["method"],
        "selected_outer_step": selected_outer,
        "closure_calls": closure_calls,
        "wall_seconds": time.perf_counter() - started,
        "baseline_residual_metrics": baseline,
        "selected_residual_metrics": verify_row,
        "post_selection_diagnostics": final_metrics,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "COMPLETED.json").write_text(
        json.dumps(
            {
                "completed_unix_time": time.time(),
                "direct_solution_read_during_training": False,
                "selection_frozen_before_objective_diagnostics": True,
                "selected_outer_step": selected_outer,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
