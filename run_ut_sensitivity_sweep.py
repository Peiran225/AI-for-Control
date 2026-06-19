#!/usr/bin/env python3
"""One-factor-at-a-time sensitivity sweep for the first-layer u(t) experiment."""

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
OUT_DIR = ROOT / "paper_runs" / "ut_sensitivity_sweep"


@dataclass(frozen=True)
class Condition:
    tag: str
    family: str
    value_label: str
    beta: float = 0.1
    gamma: float = 20.0
    alpha: float = 1.0
    n0: float = 10.0


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


def all_conditions() -> List[Condition]:
    return [
        Condition("baseline", "baseline", "baseline"),
        Condition("beta_0p05", "beta", "0.05", beta=0.05),
        Condition("beta_0p20", "beta", "0.20", beta=0.20),
        Condition("gamma_10", "gamma", "10", gamma=10.0),
        Condition("gamma_40", "gamma", "40", gamma=40.0),
        Condition("alpha_0p5", "alpha", "0.5", alpha=0.5),
        Condition("alpha_2", "alpha", "2", alpha=2.0),
        Condition("n0_5", "N0", "5", n0=5.0),
        Condition("n0_20", "N0", "20", n0=20.0),
    ]


def cfg_from_condition(cond: Condition) -> Config:
    return Config(beta=cond.beta, gamma=cond.gamma, alpha=cond.alpha, n0=cond.n0)


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
        uk = np.clip(piecewise_constant_u(t_src, u_src, tk), 0.0, cfg.umax)
        x = N[k]
        k1 = dynamics(x, uk, p)
        k2 = dynamics(np.maximum(x + 0.5 * dt * k1, 1e-10), uk, p)
        k3 = dynamics(np.maximum(x + 0.5 * dt * k2, 1e-10), uk, p)
        k4 = dynamics(np.maximum(x + dt * k3, 1e-10), uk, p)
        u[k] = uk
        N[k + 1] = np.maximum(x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0, 1e-10)
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
        "kkt_mean": float((comp / (cfg.gamma * cfg.umax + 1e-12)).mean()),
        "u_min": float(u.min()),
        "u_max": float(u.max()),
        "u_mean": float(u.mean()),
        "terminal_total_N": float(N[-1].sum()),
        "terminal_mean_N": float(N[-1].mean()),
    }


def read_solution_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    return np.asarray(z["t"], dtype=np.float64), np.asarray(z["u"], dtype=np.float64)


def run_cmd(cmd: List[str], cwd: Path, force: bool = False, sentinel: Path | None = None) -> None:
    if sentinel is not None and sentinel.exists() and not force:
        print(f"skip existing {sentinel}", flush=True)
        return
    print("running", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def run_direct(cond: Condition, args: argparse.Namespace) -> Path:
    out = OUT_DIR / cond.tag / "direct_n400"
    sol = out / "scale_1_direct_solution.npz"
    cmd = [
        sys.executable,
        "run_direct_openloop_cost.py",
        "--out_dir",
        str(out),
        "--scales",
        "1.0",
        "--n",
        str(args.direct_n),
        "--beta",
        str(cond.beta),
        "--gamma",
        str(cond.gamma),
        "--alpha",
        str(cond.alpha),
        "--n0",
        str(cond.n0),
        "--singular_eps",
        "0.1",
        "--singular_tau",
        "0.03",
        "--random_starts",
        str(args.direct_random_starts),
        "--maxiter",
        str(args.direct_maxiter),
        "--maxfun",
        str(args.direct_maxfun),
        "--seed",
        "0",
    ]
    run_cmd(cmd, ROOT, args.force, sol)
    return sol


def run_transformer(cond: Condition, seed: int, args: argparse.Namespace) -> Path:
    out = OUT_DIR / cond.tag / "transformer" / f"seed_{seed}"
    sol = out / "solution.npz"
    cmd = [
        sys.executable,
        "train_paper_pmp_kkt.py",
        "--model",
        "transformer",
        "--n",
        str(args.train_n),
        "--beta",
        str(cond.beta),
        "--gamma",
        str(cond.gamma),
        "--alpha",
        str(cond.alpha),
        "--n0",
        str(cond.n0),
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
        str(max(200, args.transformer_epochs // 2)),
        "--out_dir",
        str(out),
    ]
    run_cmd(cmd, ROOT, args.force, sol)
    return sol


def run_neural_pmp(cond: Condition, args: argparse.Namespace) -> Path:
    out = OUT_DIR / cond.tag / "neural_pmp"
    sol = out / "best_neural_pmp_solution.npz"
    cmd = [
        sys.executable,
        "run_neural_pmp_baseline.py",
        "--out_dir",
        str(out),
        "--n",
        str(args.train_n),
        "--beta",
        str(cond.beta),
        "--gamma",
        str(cond.gamma),
        "--alpha",
        str(cond.alpha),
        "--n0",
        str(cond.n0),
        "--seeds",
        args.neural_seeds,
        "--starts",
        args.neural_starts,
        "--lrs",
        args.neural_lrs,
        "--iters",
        str(args.neural_iters),
        "--log_every",
        str(max(200, args.neural_iters // 4)),
        "--singular_eps",
        "0.1",
        "--singular_tau",
        "0.03",
    ]
    run_cmd(cmd, ROOT, args.force, sol)
    return sol


def evaluate_control(cond: Condition, method: str, run_id: str, t_src: np.ndarray, u_src: np.ndarray, n_ref: int, direct_J: float | None) -> Dict[str, object]:
    cfg = cfg_from_condition(cond)
    p = build_params(cfg)
    t, N, u = rk4_rollout(t_src, u_src, cfg, p, n_ref)
    d = diagnostics(t, N, u, cfg, p)
    gap = "" if direct_J is None else d["J_ref"] - direct_J
    rel = "" if direct_J is None else (d["J_ref"] - direct_J) / (abs(direct_J) + 1e-12)
    return {
        "condition": cond.tag,
        "family": cond.family,
        "value": cond.value_label,
        "beta": cond.beta,
        "gamma": cond.gamma,
        "alpha": cond.alpha,
        "n0": cond.n0,
        "method": method,
        "run_id": run_id,
        **d,
        "J_gap_to_direct": gap,
        "relative_gap_to_direct": rel,
    }


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out = []
    keys = []
    for row in rows:
        key = (row["condition"], row["method"])
        if key not in keys:
            keys.append(key)
    for condition, method in keys:
        vals = [r for r in rows if r["condition"] == condition and r["method"] == method]
        first = vals[0]
        J = np.array([float(r["J_ref"]) for r in vals])
        gap = np.array([float(r["pmp_kkt_gap"]) for r in vals])
        rel_vals = [r["relative_gap_to_direct"] for r in vals if r["relative_gap_to_direct"] != ""]
        rel = np.array([float(x) for x in rel_vals]) if rel_vals else np.array([])
        out.append(
            {
                "condition": condition,
                "family": first["family"],
                "value": first["value"],
                "beta": first["beta"],
                "gamma": first["gamma"],
                "alpha": first["alpha"],
                "n0": first["n0"],
                "method": method,
                "n": len(vals),
                "J_mean": float(J.mean()),
                "J_std": float(J.std(ddof=1)) if len(J) > 1 else 0.0,
                "J_min": float(J.min()),
                "J_max": float(J.max()),
                "relative_gap_mean": float(rel.mean()) if len(rel) else "",
                "relative_gap_std": float(rel.std(ddof=1)) if len(rel) > 1 else 0.0 if len(rel) else "",
                "pmp_gap_mean": float(gap.mean()),
                "pmp_gap_std": float(gap.std(ddof=1)) if len(gap) > 1 else 0.0,
            }
        )
    return out


def plot(summary: List[Dict[str, object]], out_dir: Path) -> None:
    method_order = ["Transformer", "Neural-PMP [3]", "constant u=1.5"]
    cond_order = []
    for row in summary:
        if row["condition"] not in cond_order:
            cond_order.append(str(row["condition"]))
    label_map = {
        "baseline": "baseline",
        "beta_0p05": "beta=0.05",
        "beta_0p20": "beta=0.20",
        "gamma_10": "gamma=10",
        "gamma_40": "gamma=40",
        "alpha_0p5": "alpha=0.5",
        "alpha_2": "alpha=2",
        "n0_5": "N0=5",
        "n0_20": "N0=20",
    }
    cond_labels = [label_map.get(c, c.replace("_", "\n")) for c in cond_order]
    colors = {"Transformer": "#72B7B2", "Neural-PMP [3]": "#F58518", "constant u=1.5": "#54A24B"}

    fig, ax = plt.subplots(figsize=(10.2, 4.4), dpi=190)
    width = 0.24
    x = np.arange(len(cond_order))
    for j, method in enumerate(method_order):
        vals = []
        errs = []
        for cond in cond_order:
            row = next((r for r in summary if r["condition"] == cond and r["method"] == method), None)
            vals.append(np.nan if row is None or row["relative_gap_mean"] == "" else 100.0 * float(row["relative_gap_mean"]))
            errs.append(0.0 if row is None or row["relative_gap_std"] == "" else 100.0 * float(row["relative_gap_std"]))
        ax.bar(x + (j - 1) * width, vals, width=width, yerr=errs, capsize=2, label=method, color=colors[method])
    ax.axhline(0.0, color="#333333", lw=0.8)
    ax.set_xticks(x, cond_labels, rotation=0)
    ax.set_ylabel("relative objective gap to direct reference (%)")
    ax.set_title("u(t) sensitivity sweep")
    ax.legend(frameon=False, ncol=3)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "sensitivity_relative_gap.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.2, 4.4), dpi=190)
    zoom_methods = ["Transformer", "Neural-PMP [3]"]
    zoom_width = 0.28
    for j, method in enumerate(zoom_methods):
        vals = []
        errs = []
        for cond in cond_order:
            row = next((r for r in summary if r["condition"] == cond and r["method"] == method), None)
            vals.append(np.nan if row is None or row["relative_gap_mean"] == "" else 100.0 * float(row["relative_gap_mean"]))
            errs.append(0.0 if row is None or row["relative_gap_std"] == "" else 100.0 * float(row["relative_gap_std"]))
        ax.bar(x + (j - 0.5) * zoom_width, vals, width=zoom_width, yerr=errs, capsize=2, label=method, color=colors[method])
    ax.axhline(0.0, color="#333333", lw=0.8)
    ax.set_xticks(x, cond_labels, rotation=0)
    ax.set_ylabel("relative objective gap to direct reference (%)")
    ax.set_title("u(t) sensitivity sweep: close-up")
    ax.legend(frameon=False, ncol=2)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "sensitivity_relative_gap_zoom.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.2, 4.4), dpi=190)
    for method in ["direct reference", "Transformer", "Neural-PMP [3]"]:
        vals = []
        for cond in cond_order:
            row = next((r for r in summary if r["condition"] == cond and r["method"] == method), None)
            vals.append(np.nan if row is None else float(row["pmp_gap_mean"]))
        ax.plot(cond_labels, vals, marker="o", lw=2.0, label=method)
    ax.set_yscale("log")
    ax.set_ylabel("PMP/KKT diagnostic gap")
    ax.set_title("Optimality-gap diagnostic across sensitivity conditions")
    ax.legend(frameon=False)
    ax.grid(True, which="both", axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "sensitivity_pmp_gap.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--train-n", type=int, default=200)
    parser.add_argument("--direct-n", type=int, default=400)
    parser.add_argument("--n-ref", type=int, default=6400)
    parser.add_argument("--transformer-epochs", type=int, default=900)
    parser.add_argument("--transformer-lr", type=float, default=5e-4)
    parser.add_argument("--direct-random-starts", type=int, default=1)
    parser.add_argument("--direct-maxiter", type=int, default=450)
    parser.add_argument("--direct-maxfun", type=int, default=1600)
    parser.add_argument("--neural-seeds", type=str, default="0")
    parser.add_argument("--neural-starts", type=str, default="mid,front,back,random")
    parser.add_argument("--neural-lrs", type=str, default="0.02,0.05,0.1")
    parser.add_argument("--neural-iters", type=int, default=1200)
    parser.add_argument("--skip-neural-pmp", action="store_true")
    parser.add_argument("--only", type=str, default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    rows: List[Dict[str, object]] = []

    selected = all_conditions()
    if args.only.strip():
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        selected = [c for c in selected if c.tag in wanted]
        missing = wanted - {c.tag for c in selected}
        if missing:
            raise ValueError(f"Unknown condition(s): {sorted(missing)}")

    for cond in selected:
        print(f"=== condition {cond.tag} beta={cond.beta} gamma={cond.gamma} alpha={cond.alpha} n0={cond.n0} ===", flush=True)
        direct_sol = run_direct(cond, args)
        t_d, u_d = read_solution_npz(direct_sol)
        direct_row = evaluate_control(cond, "direct reference", "direct_n400", t_d, u_d, args.n_ref, None)
        direct_J = float(direct_row["J_ref"])
        direct_row["J_gap_to_direct"] = 0.0
        direct_row["relative_gap_to_direct"] = 0.0
        rows.append(direct_row)

        for seed in seeds:
            sol = run_transformer(cond, seed, args)
            t, u = read_solution_npz(sol)
            rows.append(evaluate_control(cond, "Transformer", f"seed_{seed}", t, u, args.n_ref, direct_J))

        if not args.skip_neural_pmp:
            sol = run_neural_pmp(cond, args)
            t, u = read_solution_npz(sol)
            rows.append(evaluate_control(cond, "Neural-PMP [3]", "best", t, u, args.n_ref, direct_J))

        t_const = np.linspace(0.0, 10.0, args.train_n + 1)
        u_const = np.full_like(t_const, 1.5)
        rows.append(evaluate_control(cond, "constant u=1.5", "constant", t_const, u_const, args.n_ref, direct_J))

    fields = [
        "condition",
        "family",
        "value",
        "beta",
        "gamma",
        "alpha",
        "n0",
        "method",
        "run_id",
        "J_ref",
        "J_gap_to_direct",
        "relative_gap_to_direct",
        "pmp_kkt_gap",
        "kkt_mean",
        "u_min",
        "u_max",
        "u_mean",
        "terminal_total_N",
        "terminal_mean_N",
    ]
    write_csv(OUT_DIR / "sensitivity_runs.csv", rows, fields)
    summary = summarize(rows)
    write_csv(
        OUT_DIR / "sensitivity_summary.csv",
        summary,
        [
            "condition",
            "family",
            "value",
            "beta",
            "gamma",
            "alpha",
            "n0",
            "method",
            "n",
            "J_mean",
            "J_std",
            "J_min",
            "J_max",
            "relative_gap_mean",
            "relative_gap_std",
            "pmp_gap_mean",
            "pmp_gap_std",
        ],
    )
    plot(summary, OUT_DIR)
    print(OUT_DIR / "sensitivity_summary.csv")


if __name__ == "__main__":
    main()
