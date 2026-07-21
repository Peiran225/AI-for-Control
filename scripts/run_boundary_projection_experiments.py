#!/usr/bin/env python3
"""Boundary-capable Transformer refinements with projection in the forward pass.

The starting point is the manual-target Stage-B Transformer.  The wrapper
computes

    u(t) = clamp(scale * u_sigmoid(t), 0, 3),

and is therefore capable of returning the control bound exactly.  Training
uses the projected full reduced-gradient mapping, the saved manual switch
target, and an anchor on all non-switch nodes.  No switch time is encoded in
the model or wrapper.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/boundary_capable_transformer_20260720"
sys.path.insert(0, str(ROOT))

from scripts.diagnose_reduced_objective_hessian import (  # noqa: E402
    make_reduced_objective,
)
from scripts.boundary_control import BoundaryProjectedControl  # noqa: E402
from scripts.control_switch_metrics import switch_metrics  # noqa: E402
from scripts.refine_time_only_singular_plateau import (  # noqa: E402
    build_model,
    high_accuracy_metrics,
    rk4_reduced_objective,
)
from train_paper_pmp_kkt import ProblemConfig, build_params  # noqa: E402


START_CHECKPOINT = (
    ROOT
    / "outputs/time_only_switch_sharpening_20260720"
    / "manual_target_two_stage_w1000_300_anchor1e5/stage_b_best.pt"
)
MANUAL_TARGET = START_CHECKPOINT.parent / "manual_target.npz"


@dataclass(frozen=True)
class Experiment:
    name: str
    scale_mode: str
    scale: float
    epochs: int = 350
    learning_rate: float = 1.0e-5
    target_weight: float = 300.0
    anchor_weight: float = 1.0e5
    projected_gradient_weight: float = 1.0


EXPERIMENTS = (
    Experiment("fixed_scale_1p01", "fixed", 1.01),
    Experiment("fixed_scale_1p02", "fixed", 1.02),
    Experiment("fixed_scale_1p03", "fixed", 1.03),
    Experiment("learnable_scale_1p02", "learnable", 1.02),
)


def build_wrapper(
    checkpoint: dict,
    cfg: ProblemConfig,
    experiment: Experiment,
) -> BoundaryProjectedControl:
    base = build_model(dict(checkpoint["args"]), cfg).to(dtype=torch.float64)
    base.load_state_dict(checkpoint["model_state"])
    return BoundaryProjectedControl(
        base,
        umax=cfg.umax,
        scale_mode=experiment.scale_mode,
        initial_scale=experiment.scale,
    ).to(dtype=torch.float64)


def discrete_diagnostics(
    interval_u: np.ndarray,
    objective,
) -> tuple[dict[str, float], np.ndarray]:
    variable = torch.tensor(interval_u, dtype=torch.float64, requires_grad=True)
    value = objective(variable)
    gradient = torch.autograd.grad(value, variable)[0].detach().cpu().numpy()
    mapping = interval_u - np.clip(interval_u - gradient, 0.0, 3.0)
    return {
        "rk4_reduced_objective": float(value.detach()),
        "full_gradient_linf": float(np.max(np.abs(gradient))),
        "full_gradient_rms": float(np.sqrt(np.mean(gradient**2))),
        "projected_gradient_linf": float(np.max(np.abs(mapping))),
        "projected_gradient_rms": float(np.sqrt(np.mean(mapping**2))),
    }, gradient


def evaluate_control(
    label: str,
    t: np.ndarray,
    u: np.ndarray,
    objective,
) -> tuple[dict[str, object], np.ndarray]:
    discrete, gradient = discrete_diagnostics(u[:200], objective)
    high, _ = high_accuracy_metrics(
        t,
        u,
        plateau_start=1.0,
        plateau_end=8.5,
    )
    early = switch_metrics(t, u, "early")
    late = switch_metrics(t, u, "late")
    metrics: dict[str, object] = {
        "label": label,
        **discrete,
        "high_accuracy_objective": float(high["J"]),
        "continuous_projected_kkt_mean": float(high["projected_kkt_mean"]),
        "continuous_projected_kkt_rms": float(high["projected_kkt_rms"]),
        "continuous_pmp_merit_mean": float(high["pmp_merit_mean"]),
        "continuous_pmp_merit_rms": float(high["pmp_merit_rms"]),
        "continuous_psi_relative_abs_mean": float(high["psi_relative_abs_mean"]),
        "continuous_singular_control_mae": float(high["singular_control_mae"]),
        "continuous_singular_control_rms": float(high["singular_control_rms"]),
        "plateau_total_range": float(high["plateau_total_range"]),
        "plateau_relative_range_percent": float(
            100.0 * high["plateau_total_relative_range"]
        ),
        "early_10_90_width": float(early["width_10_90_linear"]),
        "late_10_90_width": float(late["width_10_90_linear"]),
        "early_largest_jump": float(early["largest_abs_jump_at_junction"]),
        "late_largest_jump": float(late["largest_abs_jump_at_junction"]),
        "u_min": float(np.min(u[:200])),
        "u_max": float(np.max(u[:200])),
        "upper_bound_node_count": int(np.count_nonzero(u[:200] == 3.0)),
        "lower_bound_node_count": int(np.count_nonzero(u[:200] == 0.0)),
    }
    return metrics, gradient


def checkpoint_payload(
    model: BoundaryProjectedControl,
    starting_checkpoint: dict,
    experiment: Experiment,
    *,
    epoch: int,
    metrics: dict[str, object],
) -> dict:
    return {
        "model_state": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "base_model_args": dict(starting_checkpoint["args"]),
        "problem": dict(starting_checkpoint["problem"]),
        "wrapper": {
            "class": "BoundaryProjectedControl",
            "forward": "clamp(scale * base_sigmoid_control(t), 0, umax)",
            "scale_mode": experiment.scale_mode,
            "initial_scale": experiment.scale,
            "scale_min": model.scale_min,
            "scale_max": model.scale_max,
            "realized_scale": float(model.scale_value().detach()),
        },
        "source_checkpoint": str(START_CHECKPOINT),
        "manual_target": str(MANUAL_TARGET),
        "experiment": experiment.__dict__,
        "selected_epoch": epoch,
        "selection_metrics": metrics,
        "loader": str(Path(__file__).resolve()),
    }


def run_experiment(
    experiment: Experiment,
    checkpoint: dict,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
    normalized_t: torch.Tensor,
    physical_t: np.ndarray,
    target_u: torch.Tensor,
    source_u: torch.Tensor,
    switch_mask: torch.Tensor,
    anchor_mask: torch.Tensor,
    objective,
) -> dict[str, object]:
    destination = OUT / experiment.name
    destination.mkdir(parents=True, exist_ok=True)
    model = build_wrapper(checkpoint, cfg, experiment)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=experiment.learning_rate, weight_decay=0.0
    )
    dt = cfg.T / cfg.n
    scale_squared = cfg.umax * cfg.umax
    history: list[dict[str, float]] = []
    candidate_states: list[tuple[int, dict[str, torch.Tensor]]] = []
    started = time.perf_counter()

    for epoch in range(1, experiment.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        control = model(normalized_t)
        interval_control = control[:-1]
        value = rk4_reduced_objective(interval_control, cfg, params)
        full_gradient = torch.autograd.grad(
            value, interval_control, create_graph=True
        )[0]
        projected_mapping = interval_control - torch.clamp(
            interval_control - full_gradient, 0.0, cfg.umax
        )
        projected_loss = (projected_mapping / dt).square().mean()
        target_loss = (
            (control[switch_mask] - target_u[switch_mask]).square().mean()
            / scale_squared
        )
        anchor_loss = (
            (control[anchor_mask] - source_u[anchor_mask]).square().mean()
            / scale_squared
        )
        loss = (
            experiment.projected_gradient_weight * projected_loss
            + experiment.target_weight * target_loss
            + experiment.anchor_weight * anchor_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        row = {
            "epoch": float(epoch),
            "loss": float(loss.detach()),
            "projected_loss": float(projected_loss.detach()),
            "target_loss": float(target_loss.detach()),
            "anchor_loss": float(anchor_loss.detach()),
            "scale": float(model.scale_value().detach()),
            "elapsed_seconds": float(time.perf_counter() - started),
        }
        history.append(row)
        if epoch == 1 or epoch % 25 == 0 or epoch == experiment.epochs:
            candidate_states.append(
                (
                    epoch,
                    {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    },
                )
            )
        if epoch == 1 or epoch % 50 == 0 or epoch == experiment.epochs:
            print(
                f"{experiment.name} {epoch:03d}: loss={row['loss']:.6g} "
                f"proj={row['projected_loss']:.4g} target={row['target_loss']:.3g} "
                f"anchor={row['anchor_loss']:.3g} scale={row['scale']:.6f}",
                flush=True,
            )

    # Evaluate saved epochs with the exact high-accuracy protocol.  Feasible
    # candidates are ranked by projected Linf, then projected RMS, then J.
    evaluated: list[tuple[int, dict[str, object], np.ndarray, dict[str, torch.Tensor]]] = []
    for epoch, state in candidate_states:
        model.load_state_dict(state)
        with torch.no_grad():
            control = model(normalized_t).detach().cpu().numpy()
        metrics, gradient = evaluate_control(
            f"{experiment.name}_epoch_{epoch}", physical_t, control, objective
        )
        metrics["epoch"] = epoch
        metrics["realized_scale"] = float(model.scale_value().detach())
        metrics["passes_width_screen"] = bool(
            float(metrics["early_10_90_width"]) <= 0.045
            and float(metrics["late_10_90_width"]) <= 0.045
        )
        metrics["passes_flatness_screen"] = bool(
            float(metrics["plateau_relative_range_percent"]) <= 0.7063
        )
        metrics["passes_all_screens"] = bool(
            metrics["passes_width_screen"] and metrics["passes_flatness_screen"]
        )
        evaluated.append((epoch, metrics, gradient, state))

    feasible = [item for item in evaluated if item[1]["passes_all_screens"]]
    pool = feasible if feasible else evaluated
    selected = min(
        pool,
        key=lambda item: (
            float(item[1]["projected_gradient_linf"]),
            float(item[1]["projected_gradient_rms"]),
            float(item[1]["high_accuracy_objective"]),
        ),
    )
    selected_epoch, selected_metrics, selected_gradient, selected_state = selected
    model.load_state_dict(selected_state)
    with torch.no_grad():
        selected_u = model(normalized_t).detach().cpu().numpy()

    torch.save(
        checkpoint_payload(
            model,
            checkpoint,
            experiment,
            epoch=selected_epoch,
            metrics=selected_metrics,
        ),
        destination / "selected_checkpoint.pt",
    )
    np.savez(
        destination / "selected_solution.npz",
        t=physical_t,
        u=selected_u,
        full_gradient=selected_gradient,
        switch_mask=switch_mask.detach().cpu().numpy(),
        source_u=source_u.detach().cpu().numpy(),
        target_u=target_u.detach().cpu().numpy(),
    )
    (destination / "selected_metrics.json").write_text(
        json.dumps(selected_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "checkpoint_evaluations.json").write_text(
        json.dumps([item[1] for item in evaluated], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (destination / "history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    return {
        "experiment": experiment.__dict__,
        "training_seconds": float(time.perf_counter() - started),
        "feasible_checkpoint_count": len(feasible),
        "selected_metrics": selected_metrics,
        "checkpoint": str(destination / "selected_checkpoint.pt"),
        "solution": str(destination / "selected_solution.npz"),
    }


def main() -> None:
    torch.manual_seed(4)
    torch.set_num_threads(1)
    checkpoint = torch.load(
        START_CHECKPOINT, map_location="cpu", weights_only=False
    )
    cfg = ProblemConfig(**checkpoint["problem"])
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    normalized_t = torch.linspace(0.0, 1.0, cfg.n + 1, dtype=torch.float64)
    physical_t = (cfg.T * normalized_t).detach().cpu().numpy()

    target = np.load(MANUAL_TARGET)
    target_u = torch.as_tensor(target["u"], dtype=torch.float64)
    source_u = torch.as_tensor(target["source_u"], dtype=torch.float64)
    switch_interval_mask = np.asarray(target["switch_mask"], dtype=bool)
    switch_mask = torch.as_tensor(
        np.r_[switch_interval_mask, switch_interval_mask[-1]], dtype=torch.bool
    )
    anchor_mask = ~switch_mask
    objective = make_reduced_objective(cfg, params)

    results: list[dict[str, object]] = []
    for experiment in EXPERIMENTS:
        results.append(
            run_experiment(
                experiment,
                checkpoint,
                cfg,
                params,
                normalized_t,
                physical_t,
                target_u,
                source_u,
                switch_mask,
                anchor_mask,
                objective,
            )
        )

    feasible = [
        result
        for result in results
        if result["selected_metrics"]["passes_all_screens"]
    ]
    pool = feasible if feasible else results
    winner = min(
        pool,
        key=lambda result: (
            float(result["selected_metrics"]["projected_gradient_linf"]),
            float(result["selected_metrics"]["projected_gradient_rms"]),
            float(result["selected_metrics"]["high_accuracy_objective"]),
        ),
    )
    summary = {
        "source_checkpoint": str(START_CHECKPOINT),
        "manual_target": str(MANUAL_TARGET),
        "screen": {
            "early_10_90_width_max": 0.045,
            "late_10_90_width_max": 0.045,
            "plateau_relative_range_percent_max": 0.7063,
        },
        "results": results,
        "feasible_experiment_count": len(feasible),
        "winner": winner,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
