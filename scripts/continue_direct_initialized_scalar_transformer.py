#!/usr/bin/env python3
"""Continue a direct-initialized Transformer using only PMP/KKT scalar residuals.

The continuation never opens a direct or manually supplied control.  A direct
solution may have been used to construct the input checkpoint, but after that
checkpoint is loaded every update and every checkpoint-selection decision uses
only the current policy, the state/costate equations, and the scalar
optimality conditions.
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
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=2607)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=1.0e-7)
    parser.add_argument("--final-learning-rate", type=float, default=1.0e-9)
    parser.add_argument(
        "--psi-scale",
        type=float,
        default=2.5e-3,
        help="Scale in the normalized training units.",
    )
    parser.add_argument(
        "--dot-scale",
        type=float,
        default=2.5e-3,
        help="Scale in the normalized training units.",
    )
    parser.add_argument(
        "--ddot-scale",
        type=float,
        default=2.5e-3,
        help="Scale in the normalized training units.",
    )
    parser.add_argument("--w0", type=float, default=1.0)
    parser.add_argument("--w1", type=float, default=1.0)
    parser.add_argument("--w2", type=float, default=1.0)
    parser.add_argument(
        "--boundary-kkt-weight",
        type=float,
        default=1.0,
        help="Weight of the projected Hamiltonian KKT residual off the interior.",
    )
    parser.add_argument("--interior-start", type=float, default=1.5)
    parser.add_argument("--interior-end", type=float, default=8.0)
    parser.add_argument(
        "--costate-pairing", choices=("current", "next"), default="current"
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--report-scale-factor", type=float, default=400.0)
    parser.add_argument(
        "--save-eval-checkpoints",
        action="store_true",
        help="Save every evaluated candidate for independent off-grid selection.",
    )
    args = parser.parse_args()

    if not (
        0.0 <= args.interior_start < args.interior_end <= 10.0
        and args.epochs >= 0
        and args.eval_every > 0
    ):
        raise ValueError("invalid interval or training schedule")
    for name in ("psi_scale", "dot_scale", "ddot_scale"):
        if getattr(args, name) <= 0.0:
            raise ValueError(f"{name} must be positive")

    set_seed(args.seed)
    # Keep inference and differentiable Transformer arithmetic on the same path.
    # The fused inference-only MHA kernel can otherwise change this sensitive
    # control by roughly 1e-2 even when the parameters are identical.
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
    candidate_dir = out_dir / "candidates"
    if args.save_eval_checkpoints:
        candidate_dir.mkdir()

    model, cfg, source = load_model(start_checkpoint, device)
    params = build_params(cfg, device, torch.float64)
    normalized_t = torch.linspace(
        0.0, 1.0, cfg.n + 1, device=device, dtype=torch.float64
    )
    interval_t = cfg.T * normalized_t[:-1]
    interior = (
        (interval_t >= args.interior_start)
        & (interval_t < args.interior_end)
    ).unsqueeze(0)
    exterior = ~interior
    initial_state = torch.full(
        (1, cfg.m),
        cfg.n0,
        device=device,
        dtype=torch.float64,
    )
    model.eval()
    with torch.enable_grad():
        initial_control = model(normalized_t).detach()

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
        costates, _ = compute_costate_rk4(states, controls, cfg, params)
        quantities = singular_quantities(
            states,
            controls,
            costates,
            cfg,
            params,
            costate_pairing=args.costate_pairing,
        )
        psi_all = quantities["psi"]
        psi = psi_all[interior]
        dot = quantities["dot_psi"][interior]
        ddot = quantities["ddot_psi"][interior]
        psi_term = (psi / args.psi_scale).square().mean()
        dot_term = (dot / args.dot_scale).square().mean()
        ddot_term = (ddot / args.ddot_scale).square().mean()
        scalar_loss = (
            args.w0 * psi_term
            + args.w1 * dot_term
            + args.w2 * ddot_term
        )
        projected = controls - torch.clamp(
            controls - psi_all, 0.0, cfg.umax
        )
        boundary_kkt = projected[exterior].square().mean()
        total = scalar_loss + args.boundary_kkt_weight * boundary_kkt
        return {
            "loss": total,
            "scalar_loss": scalar_loss,
            "psi_term": psi_term,
            "dot_term": dot_term,
            "ddot_term": ddot_term,
            "boundary_kkt": boundary_kkt,
            "psi_rms": psi.square().mean().sqrt(),
            "dot_rms": dot.square().mean().sqrt(),
            "ddot_rms": ddot.square().mean().sqrt(),
            "control_drift_rms": (
                all_control - initial_control
            ).square().mean().sqrt(),
            "control_drift_linf": (
                all_control - initial_control
            ).abs().max(),
        }

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
        eps=1.0e-12,
    )
    history: list[dict[str, Any]] = []
    candidates: list[
        tuple[
            float,
            float,
            float,
            float,
            float,
            dict[str, torch.Tensor],
            int,
        ]
    ] = []
    started = time.perf_counter()

    def make_payload(
        state: dict[str, torch.Tensor],
        epoch: int,
        scalar_rms: dict[str, float],
    ) -> dict[str, Any]:
        return {
            **source,
            "model_state": state,
            "source_checkpoint": str(start_checkpoint),
            "source_checkpoint_sha256": sha256(start_checkpoint),
            "teacher_free": False,
            "direct_or_manual_solution_used": True,
            "continuation_reads_direct_or_manual_solution": False,
            "continuation_uses_objective_value_as_loss_or_selection": False,
            "external_correction_head": False,
            "method": (
                "direct-initialized Transformer followed by PMP/KKT scalar-"
                "condition continuation"
            ),
            "selected_scalar_epoch": epoch,
            "scalar_grid_rms": scalar_rms,
            "continuation_args": vars(args),
        }

    def record(epoch: int) -> None:
        model.eval()
        with torch.enable_grad():
            pack = loss_pack()
        row = {
            key: float(value.detach().cpu())
            for key, value in pack.items()
        }
        physical = [
            args.report_scale_factor * row["psi_rms"],
            args.report_scale_factor * row["dot_rms"],
            args.report_scale_factor * row["ddot_rms"],
        ]
        maximum = max(physical)
        joint = math.sqrt(sum(value * value for value in physical))
        state = clone_state(model)
        candidates.append(
            (
                maximum,
                joint,
                physical[0],
                physical[1],
                physical[2],
                state,
                epoch,
            )
        )
        row.update(
            {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "physical_psi_rms": physical[0],
                "physical_dot_rms": physical[1],
                "physical_ddot_rms": physical[2],
                "physical_component_max": maximum,
                "physical_joint_rms": joint,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        history.append(row)
        if args.save_eval_checkpoints:
            payload = make_payload(
                state,
                epoch,
                {
                    "H_u": physical[0],
                    "dH_u_dt": physical[1],
                    "d2H_u_dt2": physical[2],
                },
            )
            torch.save(
                payload, candidate_dir / f"epoch_{epoch:05d}.pt"
            )
        print(
            f"[epoch {epoch:05d}] physical RMS "
            f"{physical[0]:.6e}/{physical[1]:.6e}/{physical[2]:.6e} "
            f"max={maximum:.6e} drift={row['control_drift_linf']:.3e}",
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

    selected = min(candidates, key=lambda item: item[:5])
    model.load_state_dict(selected[5], strict=True)
    grid_metrics, selected_control = evaluate(
        model, cfg, normalized_t, params, high_accuracy=True
    )
    scalar_rms = {
        "H_u": selected[2],
        "dH_u_dt": selected[3],
        "d2H_u_dt2": selected[4],
    }
    output_payload = make_payload(
        selected[5], selected[6], scalar_rms
    )
    output_payload["metrics"] = grid_metrics
    torch.save(output_payload, out_dir / "selected_checkpoint.pt")
    np.savez_compressed(
        out_dir / "solution.npz",
        t=np.linspace(0.0, cfg.T, cfg.n + 1),
        u=selected_control,
        initial_u=initial_control.detach().cpu().numpy(),
    )
    write_csv(out_dir / "history.csv", history)
    summary = {
        "status": "completed",
        "start_checkpoint": str(start_checkpoint),
        "start_checkpoint_sha256": sha256(start_checkpoint),
        "device": str(device),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "continuation_reads_direct_or_manual_solution": False,
        "continuation_uses_objective_value_as_loss_or_selection": False,
        "external_correction_head": False,
        "training": vars(args),
        "selected_epoch": selected[6],
        "selected_scalar_grid_rms": scalar_rms,
        "grid_metrics": grid_metrics,
        "wall_seconds": time.perf_counter() - started,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
