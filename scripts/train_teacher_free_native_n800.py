#!/usr/bin/env python3
"""Teacher-free optimality-condition training for a native n=800 Transformer.

This program intentionally has no command-line option for a reference control,
checkpoint, direct solution, manual target, switching time, or switching mask.
The only training signal is the discrete optimality system: the projected full
reduced gradient (including the state dependence N=N(u)), with optional generic
PMP/KKT and smoothness regularizers.  The reduced objective is differentiated
to construct the gradient residual; its value is never part of the loss or
checkpoint-selection rule.

The same randomly initialized time-only Transformer is continued over a list
of successively finer uniform grids.  Grid continuation is generic and does not
encode where a switch should occur.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.refine_time_only_singular_plateau import rk4_reduced_objective  # noqa: E402
from train_paper_pmp_kkt import (  # noqa: E402
    ProblemConfig,
    TimeTransformer,
    build_params,
    pmp_kkt_loss,
    set_seed,
)


class GenericBoxProjectedControl(nn.Module):
    """A boundary-capable output map without case-specific structure."""

    def __init__(self, base: nn.Module, umax: float, scale: float) -> None:
        super().__init__()
        self.base = base
        self.umax = float(umax)
        self.register_buffer("scale", torch.tensor(float(scale), dtype=torch.float64))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.scale * self.base(t), 0.0, self.umax)


@dataclass(frozen=True)
class GridStage:
    n: int
    epochs: int
    learning_rate: float
    smooth_start: float
    smooth_end: float
    pmp_start: float
    pmp_end: float


def parse_stages(text: str) -> list[GridStage]:
    """Parse n:epochs:lr:smooth0:smooth1:pmp0:pmp1 comma-separated stages."""

    stages: list[GridStage] = []
    for item in text.split(","):
        fields = item.strip().split(":")
        if len(fields) != 7:
            raise ValueError(
                "each stage must be n:epochs:lr:smooth_start:smooth_end:pmp_start:pmp_end"
            )
        stage = GridStage(
            n=int(fields[0]),
            epochs=int(fields[1]),
            learning_rate=float(fields[2]),
            smooth_start=float(fields[3]),
            smooth_end=float(fields[4]),
            pmp_start=float(fields[5]),
            pmp_end=float(fields[6]),
        )
        if stage.n <= 0 or stage.epochs <= 0 or stage.learning_rate <= 0.0:
            raise ValueError(f"invalid stage: {stage}")
        stages.append(stage)
    if not stages or stages[-1].n != 800:
        raise ValueError("the final teacher-free training grid must be n=800")
    return stages


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def linear_weight(start: float, end: float, epoch: int, epochs: int) -> float:
    fraction = 1.0 if epochs <= 1 else (epoch - 1) / (epochs - 1)
    return start + fraction * (end - start)


def full_gradient_pack(
    model: nn.Module,
    normalized_t: torch.Tensor,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
    *,
    create_graph: bool,
) -> dict[str, torch.Tensor]:
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
    dt = cfg.T / cfg.n
    return {
        "control": control,
        "interval": interval,
        "objective_for_differentiation_only": objective,
        "gradient": gradient,
        "mapping": mapping,
        "scaled_mapping": mapping / dt,
    }


def raw_metrics(pack: dict[str, torch.Tensor], cfg: ProblemConfig) -> dict[str, float]:
    with torch.no_grad():
        u = pack["interval"]
        g = pack["gradient"]
        m = pack["mapping"]
        scaled = pack["scaled_mapping"]
        return {
            # Logged only; not used in the loss or selection.
            "J_diagnostic_only": float(pack["objective_for_differentiation_only"].detach()),
            "projected_gradient_linf": float(m.detach().abs().max()),
            "projected_gradient_rms": float(m.detach().square().mean().sqrt()),
            "projected_gradient_over_dt_linf": float(scaled.detach().abs().max()),
            "projected_gradient_over_dt_rms": float(scaled.detach().square().mean().sqrt()),
            "raw_gradient_linf": float(g.detach().abs().max()),
            "u_min": float(u.min()),
            "u_max": float(u.max()),
            "exact_upper_bound_count": int(torch.count_nonzero(u == cfg.umax)),
            "exact_lower_bound_count": int(torch.count_nonzero(u == 0.0)),
            "first_difference_rms": float(torch.diff(u).square().mean().sqrt()),
            "second_difference_rms": float(torch.diff(u, n=2).square().mean().sqrt()),
            "total_variation": float(torch.diff(u).abs().sum()),
        }


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
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
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--resume-teacher-free",
        default="",
        help=(
            "Optional checkpoint produced by this teacher-free program. The "
            "checkpoint is rejected unless its provenance explicitly confirms "
            "that no reference, objective-value loss, or switching mask was used."
        ),
    )
    parser.add_argument(
        "--stages",
        default=(
            "100:160:3e-4:0.03:0.003:0.2:0.05,"
            "200:140:1.5e-4:0.003:0.0003:0.05:0.01,"
            "400:100:7e-5:0.0003:0:0.01:0,"
            "800:100:3e-5:0:0:0:0"
        ),
    )
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--m", type=int, default=21)
    parser.add_argument("--umax", type=float, default=3.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=20.0)
    parser.add_argument("--n0", type=float, default=10.0)
    parser.add_argument("--m-suppression", type=float, default=0.5)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--init-u", type=float, default=1.5)
    parser.add_argument("--boundary-scale", type=float, default=1.02)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=("adamw", "lbfgs"), default="adamw")
    parser.add_argument("--lbfgs-max-iter", type=int, default=12)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--projected-linf-weight", type=float, default=0.05)
    parser.add_argument(
        "--solver-mode",
        choices=("residual_squared", "fixed_point_detached"),
        default="residual_squared",
        help=(
            "residual_squared differentiates the squared KKT residual through "
            "the full gradient; fixed_point_detached uses a detached projected "
            "KKT update as the per-epoch target. Both have the same fixed points."
        ),
    )
    parser.add_argument(
        "--projection-step-over-dt",
        type=float,
        default=0.1,
        help="Positive dimensionless projected fixed-point step for fixed_point_detached.",
    )
    parser.add_argument("--singular-eps", type=float, default=0.1)
    parser.add_argument("--singular-tau", type=float, default=0.03)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--print-every", type=int, default=20)
    args = parser.parse_args()

    stages = parse_stages(args.stages)
    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir = out_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True)

    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    dtype = torch.float64
    device = torch.device("cpu")
    base = TimeTransformer(
        args.d_model,
        args.heads,
        args.layers,
        args.umax,
        args.init_u,
    ).to(device=device, dtype=dtype)
    model = GenericBoxProjectedControl(base, args.umax, args.boundary_scale).to(
        device=device, dtype=dtype
    )
    resume_path: Path | None = None
    if args.resume_teacher_free:
        resume_path = Path(args.resume_teacher_free).expanduser()
        if not resume_path.is_absolute():
            resume_path = ROOT / resume_path
        resume_path = resume_path.resolve()
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        provenance_ok = (
            resume.get("reference_solution_used") is False
            and resume.get("objective_value_used_as_loss") is False
            and resume.get("switching_time_or_mask_used") is False
            and str(resume.get("training_rule", "")).startswith("teacher-free")
        )
        if not provenance_ok:
            raise ValueError(
                "resume checkpoint does not carry strict teacher-free provenance"
            )
        resume_arch = resume["architecture"]
        requested_arch = {
            "d_model": args.d_model,
            "heads": args.heads,
            "layers": args.layers,
            "init_u": args.init_u,
            "boundary_scale": args.boundary_scale,
        }
        if any(
            not math.isclose(float(resume_arch[key]), float(value), rel_tol=0.0, abs_tol=1.0e-12)
            for key, value in requested_arch.items()
        ):
            raise ValueError("resume architecture does not match requested architecture")
        model.load_state_dict(resume["model_state"])

    problem_common = {
        "T": args.T,
        "m": args.m,
        "umax": args.umax,
        "beta": args.beta,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "n0": args.n0,
        "m_suppression": args.m_suppression,
    }
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    best_final_state: dict[str, torch.Tensor] | None = None
    best_final_metric = math.inf
    best_final_epoch = 0

    for stage_index, stage in enumerate(stages, start=1):
        cfg = ProblemConfig(n=stage.n, **problem_common)
        params = build_params(cfg, device, dtype)
        normalized_t = torch.linspace(0.0, 1.0, cfg.n + 1, device=device, dtype=dtype)
        if args.optimizer == "adamw":
            optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                model.parameters(), lr=stage.learning_rate, weight_decay=args.weight_decay
            )
        else:
            if args.solver_mode != "residual_squared":
                raise ValueError("LBFGS is supported only for residual_squared mode")
            optimizer = torch.optim.LBFGS(
                model.parameters(),
                lr=stage.learning_rate,
                max_iter=args.lbfgs_max_iter,
                history_size=50,
                line_search_fn="strong_wolfe",
                tolerance_grad=1.0e-12,
                tolerance_change=1.0e-14,
            )
        print(f"[seed {args.seed}] stage {stage_index}/{len(stages)} n={stage.n}", flush=True)

        for epoch in range(1, stage.epochs + 1):
            smooth_weight = linear_weight(
                stage.smooth_start, stage.smooth_end, epoch, stage.epochs
            )
            pmp_weight = linear_weight(
                stage.pmp_start, stage.pmp_end, epoch, stage.epochs
            )

            def compute_training_loss() -> tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]:
                pack_local = full_gradient_pack(
                    model,
                    normalized_t,
                    cfg,
                    params,
                    create_graph=(args.solver_mode == "residual_squared"),
                )
                if args.solver_mode == "residual_squared":
                    projected_residual = pack_local["scaled_mapping"]
                else:
                    # The fixed points of u=Pi(u-a*dJ/du) are independent of
                    # each positive a. Scaling by 1/dt removes mesh dependence.
                    projected_target = torch.clamp(
                        pack_local["interval"].detach()
                        - args.projection_step_over_dt
                        * pack_local["gradient"].detach()
                        / (cfg.T / cfg.n),
                        0.0,
                        cfg.umax,
                    )
                    projected_residual = (
                        pack_local["interval"] - projected_target
                    ) / args.projection_step_over_dt
                projected_mean_local = projected_residual.square().mean()
                projected_linf_local = projected_residual.abs().max().square()
                smooth_local = torch.diff(pack_local["control"]).square().mean()
                pmp_local = pmp_kkt_loss(
                    pack_local["control"],
                    cfg,
                    params,
                    args.singular_eps,
                    args.singular_tau,
                    detach_gate=True,
                )["opt_gap"]
                # No term proportional to the objective value is present here.
                loss_local = (
                    projected_mean_local
                    + args.projected_linf_weight * projected_linf_local
                    + smooth_weight * smooth_local
                    + pmp_weight * pmp_local
                )
                return loss_local, projected_mean_local, pmp_local, smooth_local

            if args.optimizer == "lbfgs":
                closure_values: dict[str, torch.Tensor] = {}

                def closure() -> torch.Tensor:
                    optimizer.zero_grad(set_to_none=True)
                    values = compute_training_loss()
                    values[0].backward()
                    closure_values["loss"] = values[0].detach()
                    closure_values["projected"] = values[1].detach()
                    closure_values["pmp"] = values[2].detach()
                    closure_values["smooth"] = values[3].detach()
                    return values[0]

                optimizer.step(closure)
                loss = closure_values["loss"]
                projected_mean = closure_values["projected"]
                pmp = closure_values["pmp"]
                smooth = closure_values["smooth"]
            else:
                optimizer.zero_grad(set_to_none=True)
                loss, projected_mean, pmp, smooth = compute_training_loss()
                loss.backward()
                if args.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            should_record = (
                epoch == 1
                or epoch % args.print_every == 0
                or epoch == stage.epochs
            )
            if should_record:
                eval_pack = full_gradient_pack(
                    model, normalized_t, cfg, params, create_graph=False
                )
                row: dict[str, Any] = {
                    "seed": args.seed,
                    "stage_index": stage_index,
                    "n": stage.n,
                    "epoch": epoch,
                    "global_epoch": sum(item.epochs for item in stages[: stage_index - 1]) + epoch,
                    "loss_before_step": float(loss.detach()),
                    "projected_loss_before_step": float(projected_mean.detach()),
                    "pmp_loss_before_step": float(pmp.detach()),
                    "smooth_loss_before_step": float(smooth.detach()),
                    "smooth_weight": smooth_weight,
                    "pmp_weight": pmp_weight,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "elapsed_seconds": time.perf_counter() - started,
                    **raw_metrics(eval_pack, cfg),
                }
                history.append(row)
                print(
                    f"  [{epoch:04d}] PGinf={row['projected_gradient_linf']:.3e} "
                    f"PGrms={row['projected_gradient_rms']:.3e} "
                    f"u={row['u_min']:.3f}..{row['u_max']:.3f} "
                    f"TV={row['total_variation']:.3f}",
                    flush=True,
                )

                # Final-grid selection uses only the optimality residual.
                if stage_index == len(stages):
                    selection_metric = row["projected_gradient_linf"]
                    if selection_metric < best_final_metric:
                        best_final_metric = selection_metric
                        best_final_epoch = epoch
                        best_final_state = clone_state(model)

        torch.save(
            {
                "model_state": clone_state(model),
                "seed": args.seed,
                "completed_stage": asdict(stage),
                "problem": asdict(cfg),
                "architecture": {
                    "d_model": args.d_model,
                    "heads": args.heads,
                    "layers": args.layers,
                    "init_u": args.init_u,
                    "boundary_scale": args.boundary_scale,
                },
                "training_rule": "teacher-free projected full reduced-gradient with generic regularization",
                "solver_mode": args.solver_mode,
                "optimizer": args.optimizer,
                "projection_step_over_dt": args.projection_step_over_dt,
                "objective_value_used_as_loss": False,
                "reference_solution_used": False,
                "switching_time_or_mask_used": False,
                "resume_teacher_free_checkpoint": str(resume_path) if resume_path else None,
                "elapsed_seconds": time.perf_counter() - started,
            },
            out_dir / f"stage_{stage_index}_n{stage.n}_end.pt",
        )

    if best_final_state is None:
        raise RuntimeError("no final-grid candidate was recorded")
    final_cfg = ProblemConfig(n=800, **problem_common)
    final_params = build_params(final_cfg, device, dtype)
    final_t = torch.linspace(0.0, 1.0, 801, device=device, dtype=dtype)
    model.load_state_dict(best_final_state)
    final_pack = full_gradient_pack(model, final_t, final_cfg, final_params, create_graph=False)
    final_metrics = raw_metrics(final_pack, final_cfg)
    elapsed = time.perf_counter() - started
    payload = {
        "model_state": best_final_state,
        "seed": args.seed,
        "selected_epoch_on_n800": best_final_epoch,
        "selection_metric": "projected_gradient_linf",
        "selection_value": best_final_metric,
        "problem": asdict(final_cfg),
        "architecture": {
            "d_model": args.d_model,
            "heads": args.heads,
            "layers": args.layers,
            "init_u": args.init_u,
            "boundary_scale": args.boundary_scale,
        },
        "stages": [asdict(stage) for stage in stages],
        "training_rule": "teacher-free projected full reduced-gradient with generic PMP/KKT and smoothness continuation",
        "solver_mode": args.solver_mode,
        "optimizer": args.optimizer,
        "projection_step_over_dt": args.projection_step_over_dt,
        "objective_value_used_as_loss": False,
        "reference_solution_used": False,
        "switching_time_or_mask_used": False,
        "resume_teacher_free_checkpoint": str(resume_path) if resume_path else None,
        "wall_time_seconds": elapsed,
        "final_training_grid_metrics": final_metrics,
    }
    torch.save(payload, out_dir / "selected_teacher_free.pt")
    control = final_pack["control"].detach().cpu().numpy()
    np.savez(
        out_dir / "teacher_free_solution.npz",
        t=np.linspace(0.0, args.T, 801),
        u=control,
    )
    write_history(out_dir / "history.csv", history)
    (out_dir / "training_summary.json").write_text(
        json.dumps(payload | {"model_state": "stored in selected_teacher_free.pt"}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    start_line = (
        f"- Resumed only from the provenance-validated teacher-free checkpoint `{resume_path}`.\n"
        if resume_path
        else "- Training began from random initialization; no checkpoint or control solution was loaded.\n"
    )
    provenance_text = (
        "# Teacher-free training provenance\n\n"
        f"- Seed: `{args.seed}`\n"
        f"- Wall time: `{elapsed:.6f} s`\n"
        f"{start_line}"
        "- No direct/manual control values, switching times, or switching masks were used.\n"
        "- The reduced objective was differentiated only to construct its complete projected gradient; its value was not a loss or selection term.\n"
        "- Selection on the final grid used projected-gradient Linf only.\n"
    )
    (out_dir / "TRAINING_PROVENANCE.md").write_text(
        provenance_text,
        encoding="utf-8",
    )
    print(f"saved {out_dir} in {elapsed:.2f} s", flush=True)


if __name__ == "__main__":
    main()
