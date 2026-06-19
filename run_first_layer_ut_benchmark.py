#!/usr/bin/env python3
"""Publishable first-layer u(t) benchmark.

This script keeps the comparison in the manuscript's fixed-initial-condition,
time-dependent-control setting.  It aggregates:

* Transformer u_theta(t) trained by the paper PMP/KKT loss, over multiple seeds.
* Neural-PMP / PMP-gradient controller-stage baseline [3].
* Direct grid-cost reference.
* Repository demo / teacher checkpoint outputs.
* Constant-control anchor.

All controls are re-evaluated by the same RK4 reference integrator.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "paper_runs" / "first_layer_ut_benchmark"


@dataclass
class Config:
    T: float = 10.0
    m: int = 21
    umax: float = 3.0
    beta: float = 0.1
    alpha: float = 1.0
    gamma: float = 20.0
    n0: float = 10.0
    M: float = 0.5


def build_params(cfg: Config) -> Dict[str, np.ndarray]:
    x = np.linspace(0.0, 1.0, cfg.m, dtype=np.float64)
    return {
        "r": 2.0 / (1.0 + 3.0 * x**4),
        "phi": 1.0 / (1.0 + x**2),
        "M": np.full(cfg.m, cfg.M, dtype=np.float64),
        "beta": np.full(cfg.m, cfg.beta, dtype=np.float64),
        "alpha": np.full(cfg.m, cfg.alpha, dtype=np.float64),
        "N0": np.full(cfg.m, cfg.n0, dtype=np.float64),
    }


def dynamics(N: np.ndarray, u: float, p: Dict[str, np.ndarray]) -> np.ndarray:
    G = np.log1p(np.mean(N))
    return (p["r"] - p["phi"] * u - p["M"] * G) * N


def piecewise_constant_u(t_src: np.ndarray, u_src: np.ndarray, t: float) -> float:
    """Hold u[k] on [t[k], t[k+1]); this matches the grid-control baselines."""
    if t >= t_src[-1]:
        return float(u_src[-1])
    k = int(np.searchsorted(t_src, t, side="right") - 1)
    k = max(0, min(k, len(u_src) - 2))
    return float(u_src[k])


def rk4_rollout(t_src: np.ndarray, u_src: np.ndarray, cfg: Config, p: Dict[str, np.ndarray], n_ref: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0.0, cfg.T, n_ref + 1)
    dt = cfg.T / n_ref
    N = np.zeros((n_ref + 1, cfg.m), dtype=np.float64)
    u = np.zeros(n_ref + 1, dtype=np.float64)
    N[0] = p["N0"]
    for k in range(n_ref):
        tk = t[k]
        Nk = N[k]
        u1 = np.clip(piecewise_constant_u(t_src, u_src, tk), 0.0, cfg.umax)
        k1 = dynamics(Nk, u1, p)
        k2 = dynamics(np.maximum(Nk + 0.5 * dt * k1, 1e-10), u1, p)
        k3 = dynamics(np.maximum(Nk + 0.5 * dt * k2, 1e-10), u1, p)
        k4 = dynamics(np.maximum(Nk + dt * k3, 1e-10), u1, p)
        u[k] = u1
        N[k + 1] = np.maximum(Nk + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0, 1e-10)
    u[-1] = np.clip(piecewise_constant_u(t_src, u_src, cfg.T), 0.0, cfg.umax)
    return t, N, u


def objective(t: np.ndarray, N: np.ndarray, u: np.ndarray, cfg: Config, p: Dict[str, np.ndarray]) -> float:
    running = N @ p["beta"] + cfg.gamma * u
    return float(p["alpha"] @ N[-1] + np.trapezoid(running, t))


def costate(t: np.ndarray, N: np.ndarray, u: np.ndarray, cfg: Config, p: Dict[str, np.ndarray]) -> np.ndarray:
    dt = t[1] - t[0]
    lam = np.zeros_like(N)
    lam[-1] = p["alpha"]
    for k in range(len(t) - 2, -1, -1):
        Nk = N[k]
        uk = u[k]
        G = np.log1p(np.mean(Nk))
        a = p["r"] - p["phi"] * uk - p["M"] * G
        D = cfg.m + Nk.sum()
        coupling = (p["M"] * lam[k + 1] * Nk).sum()
        dH_dN = p["beta"] + lam[k + 1] * a - coupling / D
        lam[k] = lam[k + 1] + dt * dH_dN
    return lam


def singular_control(N: np.ndarray, cfg: Config, p: Dict[str, np.ndarray]) -> np.ndarray:
    G = np.log1p(N.mean(axis=1))
    num = (p["beta"] * (p["r"][None, :] - p["M"][None, :] * G[:, None]) * N).sum(axis=1)
    den = (p["beta"] * p["phi"] * N).sum(axis=1)
    return np.clip(num / np.maximum(den, 1e-12), 0.0, cfg.umax)


def diagnostics(t: np.ndarray, N: np.ndarray, u: np.ndarray, cfg: Config, p: Dict[str, np.ndarray]) -> Dict[str, float]:
    lam = costate(t, N, u, cfg, p)
    psi = cfg.gamma - (lam * p["phi"][None, :] * N).sum(axis=1)
    u_sing = singular_control(N, cfg, p)
    q = 1.0 / (1.0 + np.exp(-np.clip((0.1 - np.abs(psi)) / 0.03, -60.0, 60.0)))
    comp = np.maximum(psi, 0.0) * u + np.maximum(-psi, 0.0) * (cfg.umax - u)
    gap = (q * (u - u_sing) ** 2 + (1.0 - q) * comp**2).mean()
    return {
        "J_ref": objective(t, N, u, cfg, p),
        "pmp_kkt_gap": float(gap),
        "kkt_mean": float((comp / (cfg.gamma * cfg.umax)).mean()),
        "u_min": float(u.min()),
        "u_max": float(u.max()),
        "u_mean": float(u.mean()),
        "terminal_total_N": float(N[-1].sum()),
        "terminal_mean_N": float(N[-1].mean()),
    }


def read_solution_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    return np.asarray(z["t"], dtype=np.float64), np.asarray(z["u"], dtype=np.float64)


def read_u_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(path.open()))
    return np.array([float(r["t"]) for r in rows]), np.array([float(r["u"]) for r in rows])


def evaluate_control(label: str, group: str, t_src: np.ndarray, u_src: np.ndarray, cfg: Config, p: Dict[str, np.ndarray], n_ref: int, note: str = "") -> Dict[str, object]:
    t, N, u = rk4_rollout(t_src, u_src, cfg, p, n_ref)
    row = {"method": label, "group": group, **diagnostics(t, N, u, cfg, p), "note": note}
    return row


def run_transformer_seed(seed: int, args: argparse.Namespace) -> Path:
    out_dir = OUT_DIR / "transformer_seeds" / f"seed_{seed}"
    if (out_dir / "solution.npz").exists() and not args.force:
        return out_dir
    cmd = [
        sys.executable,
        "train_paper_pmp_kkt.py",
        "--model",
        "transformer",
        "--n",
        str(args.n_train),
        "--beta",
        "0.1",
        "--epochs",
        str(args.transformer_epochs),
        "--lr",
        str(args.transformer_lr),
        "--d_model",
        "64",
        "--heads",
        "4",
        "--layers",
        "2",
        "--init_u",
        "1.5",
        "--singular_eps",
        "0.1",
        "--singular_tau",
        "0.03",
        "--smooth_weight",
        "1e-4",
        "--seed",
        str(seed),
        "--float64",
        "--print_every",
        str(max(100, args.transformer_epochs // 4)),
        "--out_dir",
        str(out_dir),
    ]
    print("running", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    return out_dir


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def summarize_group(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out = []
    groups = []
    for r in rows:
        if r["group"] not in groups:
            groups.append(str(r["group"]))
    for group in groups:
        vals = [r for r in rows if r["group"] == group]
        J = np.array([float(r["J_ref"]) for r in vals])
        gap = np.array([float(r["pmp_kkt_gap"]) for r in vals])
        out.append(
            {
                "group": group,
                "n": len(vals),
                "J_mean": float(J.mean()),
                "J_std": float(J.std(ddof=1)) if len(J) > 1 else 0.0,
                "J_min": float(J.min()),
                "J_max": float(J.max()),
                "gap_mean": float(gap.mean()),
                "gap_std": float(gap.std(ddof=1)) if len(gap) > 1 else 0.0,
                "gap_min": float(gap.min()),
                "gap_max": float(gap.max()),
            }
        )
    return out


def plot_outputs(rows: List[Dict[str, object]], summary: List[Dict[str, object]], out_dir: Path) -> None:
    groups = [r["group"] for r in summary]
    J_mean = np.array([float(r["J_mean"]) for r in summary])
    J_std = np.array([float(r["J_std"]) for r in summary])
    gap_mean = np.array([float(r["gap_mean"]) for r in summary])
    gap_std = np.array([float(r["gap_std"]) for r in summary])
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#54A24B", "#E45756", "#B279A2"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=190)
    axes[0].bar(groups, J_mean, yerr=J_std, color=colors[: len(groups)], capsize=3)
    axes[0].set_ylabel("RK4 reference objective J")
    axes[0].set_title("First-layer u(t) benchmark")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(groups, gap_mean, yerr=gap_std, color=colors[: len(groups)], capsize=3)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("PMP/KKT diagnostic gap")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "first_layer_summary.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=190)
    for group in groups:
        y = [float(r["J_ref"]) for r in rows if r["group"] == group]
        x = np.full(len(y), groups.index(group), dtype=float) + np.linspace(-0.08, 0.08, len(y))
        ax.scatter(x, y, s=36, label=group)
    ax.set_xticks(range(len(groups)), groups, rotation=20)
    ax.set_ylabel("RK4 reference objective J")
    ax.set_title("Per-run objective values")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "first_layer_per_run_J.png", bbox_inches="tight")
    plt.close(fig)

    direct_vals = [float(r["J_ref"]) for r in rows if r["group"] == "direct reference"]
    if direct_vals:
        direct = direct_vals[0]
        close_groups = ["direct reference", "Transformer", "Neural-PMP [3]"]
        fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=190)
        palette = {"direct reference": "#4C78A8", "Transformer": "#72B7B2", "Neural-PMP [3]": "#F58518"}
        for i, group in enumerate(close_groups):
            vals = [float(r["J_ref"]) - direct for r in rows if r["group"] == group]
            if not vals:
                continue
            x = np.full(len(vals), i, dtype=float)
            if len(vals) > 1:
                x += np.linspace(-0.08, 0.08, len(vals))
            ax.scatter(x, vals, s=42, color=palette[group], label=group, zorder=3)
            ax.hlines(np.mean(vals), i - 0.18, i + 0.18, color=palette[group], lw=2.0)
        ax.axhline(0.0, color="#333333", lw=0.9)
        ax.set_xticks(range(len(close_groups)), close_groups, rotation=12)
        ax.set_ylabel("objective gap relative to direct reference")
        ax.set_title("Main u(t) objective comparison")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "first_layer_objective_gap_closeup.png", bbox_inches="tight")
        plt.close(fig)


def plot_transformer_seed_losses(out_dir: Path, seeds: List[int]) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=190)
    for seed in seeds:
        path = out_dir / "transformer_seeds" / f"seed_{seed}" / "history.csv"
        if not path.exists():
            continue
        data = np.genfromtxt(path, delimiter=",", names=True)
        if data.size == 0:
            continue
        best_so_far = np.minimum.accumulate(data["loss"])
        ax.plot(data["epoch"], best_so_far, lw=1.9, alpha=0.9, label=f"seed {seed}")
    ax.set_yscale("log")
    ax.set_xlabel("training epoch")
    ax.set_ylabel("best-so-far PMP/KKT training loss")
    ax.set_title("Transformer u(t) multi-seed convergence")
    ax.grid(True, which="both", axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "transformer_seed_loss_trajectories.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--transformer-epochs", type=int, default=1200)
    parser.add_argument("--transformer-lr", type=float, default=5e-4)
    parser.add_argument("--n-ref", type=int, default=6400)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    p = build_params(cfg)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    rows: List[Dict[str, object]] = []

    direct_path = ROOT / "paper_runs" / "direct_openloop_cost_beta01_n800_refined" / "scale_1_direct_solution.npz"
    if direct_path.exists():
        t, u = read_solution_npz(direct_path)
        rows.append(evaluate_control("direct n800 refined", "direct reference", t, u, cfg, p, args.n_ref, "independent direct cost reference"))

    # Main clean Transformer run from the current report.
    report_path = ROOT / "paper_runs" / "open_loop_ut_report" / "solution.npz"
    if report_path.exists():
        t, u = read_solution_npz(report_path)
        rows.append(evaluate_control("Transformer report run", "Transformer", t, u, cfg, p, args.n_ref, "current report checkpoint"))

    for seed in seeds:
        out = run_transformer_seed(seed, args)
        t, u = read_solution_npz(out / "solution.npz")
        rows.append(evaluate_control(f"Transformer seed {seed}", "Transformer", t, u, cfg, p, args.n_ref, f"seed={seed}"))

    neural_path = ROOT / "paper_runs" / "neural_pmp_baseline_beta01" / "best_neural_pmp_solution.npz"
    if neural_path.exists():
        t, u = read_solution_npz(neural_path)
        rows.append(evaluate_control("Neural-PMP best", "Neural-PMP [3]", t, u, cfg, p, args.n_ref, "best existing oracle-dynamics run"))

    # Constant anchor.
    t_const = np.linspace(0.0, cfg.T, args.n_train + 1)
    u_const = np.full_like(t_const, 1.5)
    rows.append(evaluate_control("constant u=1.5", "constant anchor", t_const, u_const, cfg, p, args.n_ref, "scale check"))

    # Repository demo outputs, if present.
    demo_s = ROOT / "s.csv"
    if demo_s.exists():
        try:
            arr = np.loadtxt(demo_s, delimiter=",")
            u = arr.reshape(-1)
            t = np.linspace(0.0, cfg.T, len(u))
            rows.append(evaluate_control("demo singular fixed-point s.csv", "repo demo", t, u, cfg, p, args.n_ref, "repository output"))
        except Exception as exc:
            print(f"skipping s.csv: {exc}", flush=True)
    demo_u = ROOT / "u_vec.csv"
    if demo_u.exists():
        try:
            arr = np.loadtxt(demo_u, delimiter=",")
            u = arr.reshape(-1)
            t = np.linspace(0.0, cfg.T, len(u))
            rows.append(evaluate_control("demo u_vec.csv", "repo demo", t, u, cfg, p, args.n_ref, "repository output"))
        except Exception as exc:
            print(f"skipping u_vec.csv: {exc}", flush=True)

    fields = [
        "method",
        "group",
        "J_ref",
        "pmp_kkt_gap",
        "kkt_mean",
        "u_min",
        "u_max",
        "u_mean",
        "terminal_total_N",
        "terminal_mean_N",
        "note",
    ]
    write_csv(OUT_DIR / "first_layer_runs.csv", rows, fields)
    summary = summarize_group(rows)
    write_csv(
        OUT_DIR / "first_layer_summary.csv",
        summary,
        ["group", "n", "J_mean", "J_std", "J_min", "J_max", "gap_mean", "gap_std", "gap_min", "gap_max"],
    )
    plot_outputs(rows, summary, OUT_DIR)
    plot_transformer_seed_losses(OUT_DIR, seeds)
    print(OUT_DIR / "first_layer_summary.csv")
    for r in summary:
        print(r)


if __name__ == "__main__":
    main()
