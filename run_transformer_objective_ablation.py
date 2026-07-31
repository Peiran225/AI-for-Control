#!/usr/bin/env python3
"""Transformer u(t) ablation: PMP residual versus direct objective training."""

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch

from train_paper_pmp_kkt import (
    ProblemConfig,
    TimeTransformer,
    build_params,
    dynamics,
    pmp_kkt_loss,
    set_seed,
)
from tumor_problem import TumorProblem, evaluate_zoh_control


@dataclass
class LossConfig:
    name: str
    residual_weight: float
    objective_weight: float
    smooth_weight: float
    objective_kind: str = "euler"


def parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_modes(text: str) -> List[LossConfig]:
    presets = {
        "residual": LossConfig("residual", residual_weight=1.0, objective_weight=0.0, smooth_weight=1e-4),
        "objective": LossConfig("objective", residual_weight=0.0, objective_weight=1.0, smooth_weight=0.0),
        "mixed001": LossConfig("mixed001", residual_weight=0.01, objective_weight=1.0, smooth_weight=1e-4),
        "mixed01": LossConfig("mixed01", residual_weight=0.1, objective_weight=1.0, smooth_weight=1e-4),
        "objective_rk4": LossConfig(
            "objective_rk4", residual_weight=0.0, objective_weight=1.0, smooth_weight=0.0, objective_kind="rk4"
        ),
        "mixedrk4_001": LossConfig(
            "mixedrk4_001", residual_weight=0.01, objective_weight=1.0, smooth_weight=1e-4, objective_kind="rk4"
        ),
        "mixedrk4_01": LossConfig(
            "mixedrk4_01", residual_weight=0.1, objective_weight=1.0, smooth_weight=1e-4, objective_kind="rk4"
        ),
    }
    modes = []
    for item in text.split(","):
        key = item.strip()
        if not key:
            continue
        if key not in presets:
            raise ValueError(f"Unknown mode {key}. Available: {', '.join(presets)}")
        modes.append(presets[key])
    return modes


def make_problem(args: argparse.Namespace) -> ProblemConfig:
    return ProblemConfig(
        T=args.T,
        n=args.n,
        m=args.m,
        umax=args.umax,
        beta=args.beta,
        alpha=args.alpha,
        gamma=args.gamma,
        n0=args.n0,
        m_suppression=args.m_suppression,
    )


def write_dict_csv(path: Path, rows: Iterable[Dict[str, object]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_solution(path: Path, t: torch.Tensor, u: torch.Tensor, pack: Dict[str, torch.Tensor], cfg: ProblemConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        t=(cfg.T * t).detach().cpu().numpy(),
        u=u.detach().cpu().numpy(),
        N=pack["N"].detach().cpu().numpy(),
        lam=pack["lambda"].detach().cpu().numpy(),
        psi=pack["psi"].detach().cpu().numpy(),
        q=pack["q"].detach().cpu().numpy(),
        u_sing=pack["u_sing"].detach().cpu().numpy(),
    )


def rk4_objective_value(u: torch.Tensor, cfg: ProblemConfig, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    dt = cfg.T / cfg.n
    N = params["N0"]
    accumulated = torch.zeros((), device=N.device, dtype=N.dtype)
    beta = params["beta"]
    gamma = params["gamma"]
    for k in range(cfg.n):
        uk = u[k]
        k1_N = dynamics(N, uk, params)
        k1_J = (beta * N).sum() + gamma * uk

        N2 = torch.clamp(N + 0.5 * dt * k1_N, min=1e-8)
        k2_N = dynamics(N2, uk, params)
        k2_J = (beta * N2).sum() + gamma * uk

        N3 = torch.clamp(N + 0.5 * dt * k2_N, min=1e-8)
        k3_N = dynamics(N3, uk, params)
        k3_J = (beta * N3).sum() + gamma * uk

        N4 = torch.clamp(N + dt * k3_N, min=1e-8)
        k4_N = dynamics(N4, uk, params)
        k4_J = (beta * N4).sum() + gamma * uk

        N = torch.clamp(N + dt * (k1_N + 2.0 * k2_N + 2.0 * k3_N + k4_N) / 6.0, min=1e-8)
        accumulated = accumulated + dt * (k1_J + 2.0 * k2_J + 2.0 * k3_J + k4_J) / 6.0
    return accumulated + (params["alpha"] * N).sum()


def train_one(
    cfg_loss: LossConfig,
    seed: int,
    args: argparse.Namespace,
    cfg: ProblemConfig,
    params: Dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    out_dir: Path,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    set_seed(seed)
    t = torch.linspace(0.0, 1.0, cfg.n + 1, device=device, dtype=dtype)
    model = TimeTransformer(args.d_model, args.heads, args.layers, cfg.umax, args.init_u).to(device=device, dtype=dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=args.lr_patience)

    best_loss = math.inf
    best_obj = math.inf
    best_loss_state = None
    best_obj_state = None
    best_loss_epoch = 0
    best_obj_epoch = 0
    history: List[Dict[str, object]] = []
    efficient_objective_only_active = bool(
        args.efficient_objective_only
        and cfg_loss.name == "objective_rk4"
        and cfg_loss.residual_weight == 0.0
        and cfg_loss.smooth_weight == 0.0
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        diagnostics_due = epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs
        opt.zero_grad(set_to_none=True)
        u = model(t)
        if efficient_objective_only_active:
            # The objective-only mode does not use the PMP/KKT or smoothness terms.
            # Avoid building those graphs on every epoch; diagnostics are evaluated
            # separately, without autograd, only when they will be logged.
            pack = None
            objective_for_loss = rk4_objective_value(u, cfg, params)
        else:
            pack = pmp_kkt_loss(u, cfg, params, args.singular_eps, args.singular_tau)
            objective_for_loss = (
                rk4_objective_value(u, cfg, params) if cfg_loss.objective_kind == "rk4" else pack["objective"]
            )
        objective_scaled = objective_for_loss / (cfg.n + 1)
        if efficient_objective_only_active:
            loss = cfg_loss.objective_weight * objective_scaled
        else:
            loss = (
                cfg_loss.residual_weight * pack["opt_gap"]
                + cfg_loss.objective_weight * objective_scaled
                + cfg_loss.smooth_weight * pack["smooth"]
            )
        loss.backward()
        with torch.no_grad():
            loss_value = float(loss.detach().cpu())
            objective_value = float(objective_for_loss.detach().cpu())
            if loss_value < best_loss:
                best_loss = loss_value
                best_loss_epoch = epoch
                best_loss_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if objective_value < best_obj:
                best_obj = objective_value
                best_obj_epoch = epoch
                best_obj_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if diagnostics_due:
                if pack is None:
                    pack = pmp_kkt_loss(u, cfg, params, args.singular_eps, args.singular_tau)
                row = {
                    "mode": cfg_loss.name,
                    "seed": seed,
                    "epoch": epoch,
                    "loss": loss_value,
                    "objective": objective_value,
                    "objective_euler": float(pack["objective"].detach().cpu()),
                    "objective_scaled": float(objective_scaled.detach().cpu()),
                    "opt_gap": float(pack["opt_gap"].detach().cpu()),
                    "singular_component": float(pack["singular_component"].detach().cpu()),
                    "nonsingular_component": float(pack["nonsingular_component"].detach().cpu()),
                    "smooth": float(pack["smooth"].detach().cpu()),
                    "u_min": float(u.min().detach().cpu()),
                    "u_max": float(u.max().detach().cpu()),
                    "u_mean": float(u.mean().detach().cpu()),
                    "final_mean_N": float(pack["N"][-1].mean().detach().cpu()),
                    "psi_mean_abs": float(pack["psi"].abs().mean().detach().cpu()),
                    "q_mean": float(pack["q"].mean().detach().cpu()),
                    "lr": float(opt.param_groups[0]["lr"]),
                    "efficient_objective_only_requested": bool(args.efficient_objective_only),
                    "efficient_objective_only_active": efficient_objective_only_active,
                }
                history.append(row)
                print(
                    f"{cfg_loss.name} seed={seed} ep={epoch} loss={row['loss']:.5g} "
                    f"J={row['objective']:.5f} gap={row['opt_gap']:.5g} "
                    f"u=({row['u_min']:.3f},{row['u_max']:.3f},{row['u_mean']:.3f}) "
                    f"finalN={row['final_mean_N']:.4f}",
                    flush=True,
                )
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        scheduler.step(loss_value)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_wall_seconds = time.perf_counter() - training_started

    result = {
        "mode": cfg_loss.name,
        "seed": seed,
        "best_loss": best_loss,
        "best_loss_epoch": best_loss_epoch,
        "best_objective": best_obj,
        "best_objective_epoch": best_obj_epoch,
        "training_wall_seconds": training_wall_seconds,
        "efficient_objective_only_requested": bool(args.efficient_objective_only),
        "efficient_objective_only_active": efficient_objective_only_active,
    }
    run_dir = out_dir / cfg_loss.name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for tag, state in [("best_loss", best_loss_state), ("best_objective", best_obj_state)]:
        if state is None:
            continue
        model.load_state_dict(state)
        with torch.no_grad():
            u = model(t)
            pack = pmp_kkt_loss(u, cfg, params, args.singular_eps, args.singular_tau)
            objective_for_loss = rk4_objective_value(u, cfg, params) if cfg_loss.objective_kind == "rk4" else pack["objective"]
            save_solution(run_dir / f"{tag}.npz", t, u, pack, cfg)
            torch.save(
                {
                    "model_state": state,
                    "problem": cfg.__dict__,
                    "args": vars(args),
                    "loss_config": cfg_loss.__dict__,
                    "selection": tag,
                    "seed": seed,
                    "training_wall_seconds": training_wall_seconds,
                    "efficient_objective_only_requested": bool(args.efficient_objective_only),
                    "efficient_objective_only_active": efficient_objective_only_active,
                },
                run_dir / f"{tag}.pt",
            )
            result[f"{tag}_train_J"] = float(objective_for_loss.detach().cpu())
            result[f"{tag}_native_J"] = float(pack["objective"].detach().cpu())
            result[f"{tag}_native_gap"] = float(pack["opt_gap"].detach().cpu())
            result[f"{tag}_u_min"] = float(u.min().detach().cpu())
            result[f"{tag}_u_max"] = float(u.max().detach().cpu())
            result[f"{tag}_u_mean"] = float(u.mean().detach().cpu())
            result[f"{tag}_final_mean_N"] = float(pack["N"][-1].mean().detach().cpu())
            result[f"{tag}_tv"] = float((u[1:] - u[:-1]).abs().sum().detach().cpu())

    return result, history


def eval_rk4(u_src: np.ndarray, problem: TumorProblem, n_ref: int = 25600) -> Dict[str, float]:
    del n_ref
    t_src = np.linspace(0.0, problem.T, len(u_src), dtype=np.float64)
    metrics = evaluate_zoh_control(t_src, u_src, problem, include_diagnostics=False)
    return {
        "J_ref_25600": metrics["J"],
        "terminal_ref": metrics["terminal_cost"],
        "running_integral_ref": metrics["running_cost"],
        "final_mean_N_ref": metrics["final_mean_N"],
        "u_mean_ref": metrics["u_mean_time"],
    }


def eval_common_grid(u_src: np.ndarray, problem: TumorProblem, n: int = 1000) -> Dict[str, float]:
    t_src = np.linspace(0.0, problem.T, len(u_src), dtype=np.float64)
    metrics = evaluate_zoh_control(t_src, u_src, problem, diagnostic_points=max(2001, n + 1))
    singular_rms = metrics["singular_control_rms"]
    return {
        "common_J": metrics["J"],
        "common_gap": metrics["pmp_merit_mean"],
        "common_singular_component": float(singular_rms**2) if np.isfinite(singular_rms) else float("nan"),
        "common_nonsingular_component": metrics["projected_kkt_rms"] ** 2,
        "common_psi_abs_mean": metrics["psi_relative_abs_mean"],
        "common_q_mean": metrics["singular_fraction"],
    }


def evaluate_saved_solutions(out_dir: Path, problem: TumorProblem) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted(out_dir.glob("*/seed_*/*.npz")):
        mode = path.parts[-3]
        seed = int(path.parts[-2].split("_")[-1])
        selection = path.stem
        data = np.load(path)
        u = data["u"].astype(np.float64)
        row: Dict[str, object] = {
            "mode": mode,
            "seed": seed,
            "selection": selection,
            "source": str(path),
            "u_min": float(u.min()),
            "u_max": float(u.max()),
            "u_mean": float(u.mean()),
            "tv": float(np.abs(np.diff(u)).sum()),
        }
        row.update(eval_common_grid(u, problem))
        row.update(eval_rk4(u, problem))
        rows.append(row)
    fields = [
        "mode",
        "seed",
        "selection",
        "source",
        "u_min",
        "u_max",
        "u_mean",
        "tv",
        "common_J",
        "common_gap",
        "common_singular_component",
        "common_nonsingular_component",
        "common_psi_abs_mean",
        "common_q_mean",
        "J_ref_25600",
        "terminal_ref",
        "running_integral_ref",
        "final_mean_N_ref",
        "u_mean_ref",
    ]
    write_dict_csv(out_dir / "evaluation_summary.csv", rows, fields)
    return rows


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float64 if args.float64 else torch.float32
    cfg = make_problem(args)
    params = build_params(cfg, device, dtype)
    problem = TumorProblem(
        T=cfg.T,
        m=cfg.m,
        umax=cfg.umax,
        beta=cfg.beta,
        alpha=cfg.alpha,
        gamma=cfg.gamma,
        n0=cfg.n0,
        m_suppression=cfg.m_suppression,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = parse_modes(args.modes)
    seeds = parse_ints(args.seeds)

    print(
        f"device={device} dtype={dtype} modes={[m.name for m in modes]} seeds={seeds} "
        f"efficient_objective_only_requested={bool(args.efficient_objective_only)}",
        flush=True,
    )
    run_rows: List[Dict[str, object]] = []
    history_rows: List[Dict[str, object]] = []
    for mode in modes:
        for seed in seeds:
            result, hist = train_one(mode, seed, args, cfg, params, device, dtype, out_dir)
            run_rows.append(result)
            history_rows.extend(hist)
    run_fields = sorted({k for row in run_rows for k in row.keys()})
    hist_fields = [
        "mode",
        "seed",
        "epoch",
        "loss",
        "objective",
        "objective_euler",
        "objective_scaled",
        "opt_gap",
        "singular_component",
        "nonsingular_component",
        "smooth",
        "u_min",
        "u_max",
        "u_mean",
        "final_mean_N",
        "psi_mean_abs",
        "q_mean",
        "lr",
        "efficient_objective_only_requested",
        "efficient_objective_only_active",
    ]
    write_dict_csv(out_dir / "run_summary.csv", run_rows, run_fields)
    write_dict_csv(out_dir / "history.csv", history_rows, hist_fields)
    eval_rows = evaluate_saved_solutions(out_dir, problem)
    print("Evaluation summary:")
    for row in sorted(eval_rows, key=lambda r: float(r["J_ref_25600"])):
        print(
            f"{row['mode']} seed={row['seed']} {row['selection']} "
            f"Jref={float(row['J_ref_25600']):.6f} commonJ={float(row['common_J']):.6f} "
            f"gap={float(row['common_gap']):.5g} tv={float(row['tv']):.3f}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Transformer u(t) objective/residual ablation.")
    parser.add_argument("--modes", type=str, default="objective,mixed001,mixed01,residual")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_patience", type=int, default=250)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--init_u", type=float, default=1.5)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--m", type=int, default=21)
    parser.add_argument("--umax", type=float, default=3.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=20.0)
    parser.add_argument("--n0", type=float, default=10.0)
    parser.add_argument("--m_suppression", type=float, default=0.5)
    parser.add_argument("--singular_eps", type=float, default=0.05)
    parser.add_argument("--singular_tau", type=float, default=0.02)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--float64", action="store_true")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument(
        "--efficient_objective_only",
        action="store_true",
        help=(
            "For objective_rk4 only, skip zero-weight PMP/KKT and smoothness graphs between logging epochs. "
            "The default retains the historical training path."
        ),
    )
    parser.add_argument("--out_dir", type=str, default="paper_runs/transformer_objective_ablation_beta01")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
