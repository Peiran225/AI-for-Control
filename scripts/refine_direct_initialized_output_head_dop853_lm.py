#!/usr/bin/env python3
"""DOP853-calibrated output-head LM refinement of scalar PMP/KKT residuals.

The direct solution is represented only by the input checkpoint.  After that
checkpoint is loaded, this script freezes the Transformer input projection and
encoder, evaluates the original linear output head at every support node and
strict midpoint query, and minimizes only ``H_u``, ``dH_u/dt``,
``d^2H_u/dt^2``, and source-box projected KKT residuals.  State and costate
trajectories use the same high-accuracy DOP853 evaluator as the final
off-grid audit.  A finite-difference Jacobian of these residuals with respect
to the 65 original output-head parameters drives a damped Gauss--Newton
(Levenberg--Marquardt) update.  No physical-objective loss and no auxiliary
control correction are used.
"""

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

from continue_direct_initialized_scalar_transformer import (  # noqa: E402
    clone_state,
    load_model,
    sha256,
)
from generate_offgrid_policy_switching_diagnostics import (  # noqa: E402
    DensePolicy,
    integrate_trajectory,
)
from train_paper_pmp_kkt import set_seed, time_features  # noqa: E402


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else ROOT / value).resolve()


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2607)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--dense-points", type=int, default=3201)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--interior-start", type=float, default=1.5)
    parser.add_argument("--interior-end", type=float, default=8.0)
    parser.add_argument("--boundary-margin", type=float, default=0.05)
    parser.add_argument("--w0", type=float, default=1.0)
    parser.add_argument("--w1", type=float, default=1.0)
    parser.add_argument("--w2", type=float, default=4.0)
    parser.add_argument("--boundary-weight", type=float, default=10.0)
    parser.add_argument("--psi-scale", type=float, default=1.0)
    parser.add_argument("--dot-scale", type=float, default=1.0)
    parser.add_argument("--ddot-scale", type=float, default=1.0)
    parser.add_argument("--boundary-scale", type=float, default=1.0)
    parser.add_argument("--report-scale-factor", type=float, default=400.0)
    parser.add_argument("--fd-relative-step", type=float, default=1.0e-4)
    parser.add_argument("--initial-damping", type=float, default=1.0e-2)
    parser.add_argument("--minimum-damping", type=float, default=1.0e-10)
    parser.add_argument("--maximum-damping", type=float, default=1.0e12)
    parser.add_argument("--damping-decrease", type=float, default=0.3)
    parser.add_argument("--damping-increase", type=float, default=10.0)
    parser.add_argument("--maximum-attempts", type=int, default=8)
    parser.add_argument("--maximum-step-norm", type=float, default=0.5)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    args = parser.parse_args()

    if not (
        args.iterations >= 0
        and args.dense_points >= 3
        and args.query_batch_size > 0
        and 0.0 <= args.interior_start < args.interior_end <= 10.0
        and args.boundary_margin >= 0.0
        and args.fd_relative_step > 0.0
        and args.maximum_attempts > 0
        and args.maximum_step_norm > 0.0
    ):
        raise ValueError("invalid grid, interval, or LM setting")
    for name in (
        "psi_scale",
        "dot_scale",
        "ddot_scale",
        "boundary_scale",
        "initial_damping",
        "minimum_damping",
        "maximum_damping",
        "damping_decrease",
        "damping_increase",
        "rtol",
        "atol",
    ):
        if getattr(args, name) <= 0.0:
            raise ValueError(f"{name} must be positive")
    for name in ("w0", "w1", "w2", "boundary_weight"):
        if getattr(args, name) < 0.0:
            raise ValueError(f"{name} must be nonnegative")

    set_seed(args.seed)
    torch.set_num_threads(8)
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    start_checkpoint = resolve(args.start_checkpoint)
    out_dir = resolve(args.out_dir)
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True)

    model, cfg, source = load_model(start_checkpoint, device)
    model.eval()
    if not isinstance(model.base.output, torch.nn.Linear):
        raise TypeError("the Transformer must have a linear output head")
    expected_points = 2 * cfg.n + 1
    if args.dense_points != expected_points:
        raise ValueError(
            "this fixed support/midpoint implementation requires "
            f"--dense-points={expected_points}"
        )
    dense_time = np.linspace(0.0, cfg.T, args.dense_points)
    support_time = torch.linspace(
        0.0, 1.0, cfg.n + 1, device=device, dtype=torch.float64
    )
    query_time = (
        torch.arange(cfg.n, device=device, dtype=torch.float64) + 0.5
    ) / cfg.n
    support_count = int(support_time.numel())
    with torch.inference_mode():
        support_features = time_features(support_time)
        support_encoded = model.base.encoder(
            model.base.input(support_features).unsqueeze(0)
        ).squeeze(0)
        pointwise_mask = torch.zeros(
            (support_count + 1, support_count + 1),
            device=device,
            dtype=torch.bool,
        )
        pointwise_mask[:support_count, support_count] = True
        pointwise_mask[support_count, support_count] = True
        query_batches = []
        for start in range(0, cfg.n, args.query_batch_size):
            current = query_time[start : start + args.query_batch_size]
            batch = int(current.numel())
            features = torch.cat(
                (
                    support_features.unsqueeze(0).expand(batch, -1, -1),
                    time_features(current).unsqueeze(1),
                ),
                dim=1,
            )
            encoded = model.base.encoder(
                model.base.input(features),
                mask=pointwise_mask,
            )
            query_batches.append(encoded[:, -1, :])
        query_encoded = torch.cat(query_batches, dim=0)
        interleaved_encoded = torch.empty(
            (args.dense_points, support_encoded.shape[-1]),
            device=device,
            dtype=torch.float64,
        )
        interleaved_encoded[0::2] = support_encoded
        interleaved_encoded[1::2] = query_encoded
    encoded_numpy = interleaved_encoded.cpu().numpy()
    head_width = int(model.base.output.in_features)
    initial_theta = np.concatenate(
        (
            model.base.output.weight.detach().cpu().numpy().reshape(-1),
            model.base.output.bias.detach().cpu().numpy().reshape(-1),
        )
    )
    if initial_theta.size != head_width + 1:
        raise RuntimeError("unexpected output-head size")
    wrapper = dict(source.get("wrapper", {}))
    initial_state = np.full(cfg.m, cfg.n0, dtype=np.float64)
    interior = (
        (dense_time >= args.interior_start)
        & (dense_time < args.interior_end)
    )

    def raw_from_theta(theta: np.ndarray) -> np.ndarray:
        return encoded_numpy @ theta[:head_width] + theta[head_width]

    def policy_from_theta(theta: np.ndarray) -> DensePolicy:
        dense_raw = raw_from_theta(theta)
        return DensePolicy(
            "time_only",
            cfg,
            dense_time,
            dense_raw,
            dense_raw[0::2],
            start_checkpoint,
            time_wrapper=wrapper,
        )

    starting_policy = policy_from_theta(initial_theta)
    starting_control = np.asarray(
        [
            starting_policy.action(float(value), initial_state)
            for value in dense_time
        ],
        dtype=np.float64,
    )
    boundary_gate = (
        (starting_control <= args.boundary_margin)
        | (starting_control >= cfg.umax - args.boundary_margin)
    )
    if not np.any(interior) or not np.any(boundary_gate):
        raise RuntimeError("the interior or source-box KKT gate is empty")

    def evaluate_theta(
        theta: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float], Any]:
        result = integrate_trajectory(
            policy_from_theta(theta),
            initial_state,
            dense_time,
            rtol=args.rtol,
            atol=args.atol,
            max_step=cfg.T / (args.dense_points - 1),
        )
        psi_normalized = np.asarray(result.quantities["H_u"])
        psi = args.report_scale_factor * psi_normalized[interior]
        dot = args.report_scale_factor * np.asarray(
            result.quantities["dH_u_dt"]
        )[interior]
        ddot = args.report_scale_factor * np.asarray(
            result.quantities["d2H_u_dt2"]
        )[interior]
        projected = result.control - np.clip(
            result.control - psi_normalized,
            0.0,
            cfg.umax,
        )
        boundary = args.report_scale_factor * projected[boundary_gate]
        components = (
            ("psi", psi, args.w0, args.psi_scale),
            ("dot", dot, args.w1, args.dot_scale),
            ("ddot", ddot, args.w2, args.ddot_scale),
            (
                "boundary",
                boundary,
                args.boundary_weight,
                args.boundary_scale,
            ),
        )
        residuals = [
            math.sqrt(weight / value.size) * value / scale
            for _name, value, weight, scale in components
            if weight > 0.0
        ]
        residual = np.concatenate(residuals)
        metrics: dict[str, float] = {}
        for name, value, _weight, _scale in components:
            metrics[f"{name}_rms"] = float(np.sqrt(np.mean(value**2)))
            metrics[f"{name}_linf"] = float(np.max(np.abs(value)))
        metrics["control_drift_rms"] = float(
            np.sqrt(np.mean((result.control - starting_control) ** 2))
        )
        metrics["control_drift_linf"] = float(
            np.max(np.abs(result.control - starting_control))
        )
        metrics["physical_component_max"] = max(
            metrics["psi_rms"],
            metrics["dot_rms"],
            metrics["ddot_rms"],
        )
        metrics["physical_joint_rms"] = math.sqrt(
            metrics["psi_rms"] ** 2
            + metrics["dot_rms"] ** 2
            + metrics["ddot_rms"] ** 2
        )
        return residual, metrics, result

    def set_head(theta: np.ndarray) -> None:
        with torch.no_grad():
            model.base.output.weight.copy_(
                torch.from_numpy(theta[:head_width])
                .to(device=device, dtype=torch.float64)
                .reshape_as(model.base.output.weight)
            )
            model.base.output.bias.copy_(
                torch.from_numpy(theta[head_width:])
                .to(device=device, dtype=torch.float64)
                .reshape_as(model.base.output.bias)
            )

    theta = initial_theta.copy()
    damping = args.initial_damping
    history: list[dict[str, Any]] = []
    candidates: list[
        tuple[float, float, np.ndarray, int, dict[str, float], Any]
    ] = []
    started = time.perf_counter()

    def record(
        iteration: int,
        accepted: bool,
        residual: np.ndarray,
        metrics: dict[str, float],
        result: Any,
    ) -> None:
        objective = 0.5 * float(residual @ residual)
        row = {
            "iteration": iteration,
            "accepted": accepted,
            "weighted_objective": objective,
            "damping": damping,
            "elapsed_seconds": time.perf_counter() - started,
            **metrics,
        }
        history.append(row)
        candidates.append(
            (
                metrics["physical_component_max"],
                metrics["physical_joint_rms"],
                theta.copy(),
                iteration,
                metrics,
                result,
            )
        )
        print(
            f"[DOP853 LM {iteration:03d}] RMS "
            f"{metrics['psi_rms']:.6e}/"
            f"{metrics['dot_rms']:.6e}/"
            f"{metrics['ddot_rms']:.6e} "
            f"max={metrics['physical_component_max']:.6e} "
            f"objective={objective:.6e} damping={damping:.3e} "
            f"accepted={accepted}",
            flush=True,
        )

    residual, current_metrics, current_result = evaluate_theta(theta)
    objective = 0.5 * float(residual @ residual)
    record(0, True, residual, current_metrics, current_result)

    for iteration in range(1, args.iterations + 1):
        jacobian = np.empty((residual.size, theta.size), dtype=np.float64)
        for index in range(theta.size):
            step = args.fd_relative_step * max(1.0, abs(theta[index]))
            perturbed = theta.copy()
            perturbed[index] += step
            perturbed_residual, _metrics, _result = evaluate_theta(perturbed)
            jacobian[:, index] = (perturbed_residual - residual) / step
            if (index + 1) % 8 == 0 or index + 1 == theta.size:
                print(
                    f"  Jacobian column {index + 1}/{theta.size}",
                    flush=True,
                )
        gradient = jacobian.T @ residual
        normal = jacobian.T @ jacobian
        diagonal = np.maximum(np.diag(normal), np.finfo(np.float64).eps)
        accepted = False
        best = None
        for _attempt in range(args.maximum_attempts):
            system = normal + damping * np.diag(diagonal)
            try:
                update = np.linalg.solve(system, -gradient)
            except np.linalg.LinAlgError:
                damping = min(
                    args.maximum_damping,
                    damping * args.damping_increase,
                )
                continue
            norm = float(np.linalg.norm(update))
            if norm > args.maximum_step_norm:
                update *= args.maximum_step_norm / norm
            trial_theta = theta + update
            trial_residual, trial_metrics, trial_result = evaluate_theta(
                trial_theta
            )
            trial_objective = 0.5 * float(trial_residual @ trial_residual)
            if trial_objective < objective:
                accepted = True
                best = (
                    trial_theta,
                    trial_residual,
                    trial_metrics,
                    trial_result,
                    trial_objective,
                )
                damping = max(
                    args.minimum_damping,
                    damping * args.damping_decrease,
                )
                break
            damping = min(
                args.maximum_damping,
                damping * args.damping_increase,
            )
        if accepted and best is not None:
            theta, residual, current_metrics, current_result, objective = best
        record(
            iteration,
            accepted,
            residual,
            current_metrics,
            current_result,
        )
        if not accepted and damping >= args.maximum_damping:
            break

    selected = min(candidates, key=lambda item: item[:2])
    selected_theta = selected[2]
    selected_iteration = selected[3]
    selected_metrics = selected[4]
    selected_result = selected[5]
    set_head(selected_theta)
    payload = {
        **source,
        "model_state": clone_state(model),
        "source_checkpoint": str(start_checkpoint),
        "source_checkpoint_sha256": sha256(start_checkpoint),
        "teacher_free": False,
        "direct_or_manual_solution_used": True,
        "continuation_reads_direct_or_manual_solution": False,
        "continuation_uses_objective_value_as_loss_or_selection": False,
        "external_correction_head": False,
        "method": (
            "direct-initialized Transformer followed by output-head-only "
            "DOP853-calibrated damped Gauss-Newton refinement of scalar "
            "PMP/KKT residuals"
        ),
        "selected_lm_iteration": selected_iteration,
        "scalar_dense_grid_metrics": selected_metrics,
        "lm_args": vars(args),
    }
    torch.save(payload, out_dir / "selected_checkpoint.pt")
    np.savez_compressed(
        out_dir / "timeseries.npz",
        t=selected_result.time,
        u=selected_result.control,
        N=selected_result.state,
        costate=selected_result.costate,
        H=args.report_scale_factor * selected_result.hamiltonian,
        H_u=args.report_scale_factor
        * selected_result.quantities["H_u"],
        dH_u_dt=args.report_scale_factor
        * selected_result.quantities["dH_u_dt"],
        d2H_u_dt2=args.report_scale_factor
        * selected_result.quantities["d2H_u_dt2"],
    )
    write_csv(out_dir / "history.csv", history)
    summary = {
        "status": "completed",
        "start_checkpoint": str(start_checkpoint),
        "start_checkpoint_sha256": sha256(start_checkpoint),
        "device": str(device),
        "trainable_parameter_count": int(initial_theta.size),
        "continuation_reads_direct_or_manual_solution": False,
        "continuation_uses_objective_value_as_loss_or_selection": False,
        "external_correction_head": False,
        "selected_iteration": selected_iteration,
        "selected_scalar_dense_grid_metrics": selected_metrics,
        "wall_seconds": time.perf_counter() - started,
        "training": vars(args),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
