#!/usr/bin/env python3
"""Neural-PMP / PMP-Gradient baseline for the tumor-control problem.

This script implements the controller-optimization stage of
"Pontryagin Optimal Control via Neural Networks" on the local tumor-control
benchmark.  The paper first learns a differentiable dynamics model and then
updates the action sequence by a PMP costate recursion.  Here the tumor
dynamics are known, so the baseline uses the exact differentiable dynamics.
This is the favorable oracle-dynamics version of the PMP-Gradient controller.
"""

import argparse
import csv
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class ProblemConfig:
    T: float
    n: int
    m: int
    umax: float
    beta: float
    alpha: float
    gamma: float
    n0: float
    m_suppression: float


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def parse_csv_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


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


def build_params(cfg: ProblemConfig) -> Dict[str, np.ndarray]:
    x = np.linspace(0.0, 1.0, cfg.m, dtype=np.float64)
    return {
        "x": x,
        "r": 2.0 / (1.0 + 3.0 * x**4),
        "phi": 1.0 / (1.0 + x**2),
        "M": np.full(cfg.m, cfg.m_suppression, dtype=np.float64),
        "beta": np.full(cfg.m, cfg.beta, dtype=np.float64),
        "alpha": np.full(cfg.m, cfg.alpha, dtype=np.float64),
        "N0": np.full(cfg.m, cfg.n0, dtype=np.float64),
    }


def tumor_g(N: np.ndarray) -> float:
    return float(np.log1p(np.mean(N)))


def dynamics(N: np.ndarray, u: float, params: Dict[str, np.ndarray]) -> np.ndarray:
    G = tumor_g(N)
    return (params["r"] - params["phi"] * u - params["M"] * G) * N


def rollout(u: np.ndarray, cfg: ProblemConfig, params: Dict[str, np.ndarray]) -> np.ndarray:
    dt = cfg.T / cfg.n
    N = np.zeros((cfg.n + 1, cfg.m), dtype=np.float64)
    N[0] = params["N0"]
    for k in range(cfg.n):
        N[k + 1] = np.maximum(N[k] + dt * dynamics(N[k], float(u[k]), params), 1e-10)
    return N


def objective_value(N: np.ndarray, u: np.ndarray, cfg: ProblemConfig, params: Dict[str, np.ndarray]) -> float:
    dt = cfg.T / cfg.n
    running = N @ params["beta"] + cfg.gamma * u
    integral = dt * (0.5 * running[0] + running[1:-1].sum() + 0.5 * running[-1])
    terminal = float(params["alpha"] @ N[-1])
    return terminal + float(integral)


def f_n_transpose_times(vec: np.ndarray, N: np.ndarray, u: float, params: Dict[str, np.ndarray]) -> np.ndarray:
    G = tumor_g(N)
    a = params["r"] - params["phi"] * u - params["M"] * G
    D = len(N) + float(N.sum())
    coupling = float((vec * params["M"] * N).sum())
    return vec * a - coupling / D


def pmp_gradient(N: np.ndarray, u: np.ndarray, cfg: ProblemConfig, params: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Return the discrete costate and dJ/du from the PMP adjoint recursion."""
    dt = cfg.T / cfg.n
    weights = np.ones(cfg.n + 1, dtype=np.float64)
    weights[0] = 0.5
    weights[-1] = 0.5

    lam = np.zeros_like(N)
    grad = np.zeros(cfg.n + 1, dtype=np.float64)
    lam[-1] = params["alpha"] + dt * weights[-1] * params["beta"]
    grad[-1] = dt * weights[-1] * cfg.gamma

    for k in range(cfg.n - 1, -1, -1):
        grad[k] = dt * weights[k] * cfg.gamma - dt * float((params["phi"] * N[k] * lam[k + 1]).sum())
        lam[k] = dt * weights[k] * params["beta"] + lam[k + 1] + dt * f_n_transpose_times(
            lam[k + 1], N[k], float(u[k]), params
        )
    return lam, grad


def continuous_costate(N: np.ndarray, u: np.ndarray, cfg: ProblemConfig, params: Dict[str, np.ndarray]) -> np.ndarray:
    dt = cfg.T / cfg.n
    lam = np.zeros_like(N)
    lam[-1] = params["alpha"]
    for k in range(cfg.n - 1, -1, -1):
        lam[k] = lam[k + 1] + dt * (params["beta"] + f_n_transpose_times(lam[k + 1], N[k], float(u[k]), params))
    return lam


def singular_control(N: np.ndarray, cfg: ProblemConfig, params: Dict[str, np.ndarray]) -> np.ndarray:
    G = np.log1p(N.mean(axis=1))
    numerator = (params["beta"] * (params["r"][None, :] - G[:, None] * params["M"][None, :]) * N).sum(axis=1)
    denominator = (params["beta"] * params["phi"] * N).sum(axis=1)
    return numerator / np.maximum(denominator, 1e-12)


def diagnostics(u: np.ndarray, cfg: ProblemConfig, params: Dict[str, np.ndarray], singular_eps: float, singular_tau: float) -> Dict[str, float]:
    N = rollout(u, cfg, params)
    lam = continuous_costate(N, u, cfg, params)
    psi = cfg.gamma - (lam * params["phi"][None, :] * N).sum(axis=1)
    u_sing = singular_control(N, cfg, params)
    admissible = ((u_sing >= 0.0) & (u_sing <= cfg.umax)).astype(np.float64)
    q = sigmoid((singular_eps - np.abs(psi)) / singular_tau) * admissible
    l_sing = (u - u_sing) ** 2
    l_ns = (np.maximum(psi, 0.0) * u + np.maximum(-psi, 0.0) * (cfg.umax - u)) ** 2
    legacy_gap = float((q * l_sing + (1.0 - q) * l_ns).mean())
    comp = np.maximum(psi, 0.0) * u + np.maximum(-psi, 0.0) * (cfg.umax - u)
    f = np.vstack([dynamics(N[k], float(u[min(k, cfg.n - 1)]), params) for k in range(cfg.n + 1)])
    running = N @ params["beta"] + cfg.gamma * u
    H = running + (lam * f).sum(axis=1)
    H_drift = float(H.max() - H.min())
    proj = np.clip(u - pmp_gradient(N, u, cfg, params)[1], 0.0, cfg.umax)
    return {
        "J": objective_value(N, u, cfg, params),
        "legacy_pmp_gap": legacy_gap,
        "normalized_kkt_mean": float((comp / (cfg.gamma * cfg.umax + 1e-12)).mean()),
        "normalized_kkt_max": float((comp / (cfg.gamma * cfg.umax + 1e-12)).max()),
        "projected_gradient_residual": float(np.abs(u - proj).max()),
        "hamiltonian_drift": H_drift,
        "hamiltonian_rel_drift": float(H_drift / max(float(np.abs(H).mean()), 1e-12)),
        "u_min": float(u.min()),
        "u_max": float(u.max()),
        "u_mean": float(u.mean()),
        "final_mean_N": float(N[-1].mean()),
        "psi_abs_mean": float(np.abs(psi).mean()),
        "q_mean": float(q.mean()),
    }


def init_control(start: str, cfg: ProblemConfig, rng: np.random.Generator) -> np.ndarray:
    if start == "zero":
        return np.zeros(cfg.n + 1, dtype=np.float64)
    if start == "max":
        return np.full(cfg.n + 1, cfg.umax, dtype=np.float64)
    if start == "mid":
        return np.full(cfg.n + 1, 0.5 * cfg.umax, dtype=np.float64)
    if start == "front":
        return np.linspace(cfg.umax, 0.0, cfg.n + 1, dtype=np.float64)
    if start == "back":
        return np.linspace(0.0, cfg.umax, cfg.n + 1, dtype=np.float64)
    if start == "random":
        raw = rng.uniform(0.0, cfg.umax, size=cfg.n + 1)
        kernel = np.ones(9, dtype=np.float64) / 9.0
        return np.convolve(np.pad(raw, (4, 4), mode="edge"), kernel, mode="valid")
    raise ValueError(f"Unknown start: {start}")


def run_one(
    run_id: int,
    seed: int,
    start: str,
    lr: float,
    cfg: ProblemConfig,
    params: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> Tuple[Dict[str, object], List[Dict[str, object]], Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed * 10007 + run_id)
    u = np.clip(init_control(start, cfg, rng), 0.0, cfg.umax)
    m = np.zeros_like(u)
    v = np.zeros_like(u)
    best_u = u.copy()
    best_diag = diagnostics(best_u, cfg, params, args.singular_eps, args.singular_tau)
    best_iter = 0
    history: List[Dict[str, object]] = []

    for it in range(1, args.iters + 1):
        N = rollout(u, cfg, params)
        J = objective_value(N, u, cfg, params)
        _, grad = pmp_gradient(N, u, cfg, params)
        if args.grad_clip > 0.0:
            norm = float(np.linalg.norm(grad))
            if norm > args.grad_clip:
                grad = grad * (args.grad_clip / norm)

        m = args.adam_beta1 * m + (1.0 - args.adam_beta1) * grad
        v = args.adam_beta2 * v + (1.0 - args.adam_beta2) * (grad * grad)
        m_hat = m / (1.0 - args.adam_beta1**it)
        v_hat = v / (1.0 - args.adam_beta2**it)
        u = np.clip(u - lr * m_hat / (np.sqrt(v_hat) + args.adam_eps), 0.0, cfg.umax)

        if J < best_diag["J"]:
            best_u = u.copy()
            best_diag = diagnostics(best_u, cfg, params, args.singular_eps, args.singular_tau)
            best_iter = it

        if it == 1 or it % args.log_every == 0 or it == args.iters:
            diag = diagnostics(u, cfg, params, args.singular_eps, args.singular_tau)
            history.append(
                {
                    "run_id": run_id,
                    "seed": seed,
                    "start": start,
                    "lr": lr,
                    "iter": it,
                    "J": diag["J"],
                    "best_J": best_diag["J"],
                    "legacy_pmp_gap": diag["legacy_pmp_gap"],
                    "projected_gradient_residual": diag["projected_gradient_residual"],
                    "u_min": diag["u_min"],
                    "u_max": diag["u_max"],
                    "u_mean": diag["u_mean"],
                }
            )

    row: Dict[str, object] = {
        "run_id": run_id,
        "seed": seed,
        "start": start,
        "lr": lr,
        "iters": args.iters,
        "best_iter": best_iter,
        **best_diag,
    }
    artifacts = {
        "u": best_u,
        "N": rollout(best_u, cfg, params),
        "lambda": continuous_costate(rollout(best_u, cfg, params), best_u, cfg, params),
    }
    return row, history, artifacts


def write_dict_csv(path: Path, rows: Iterable[Dict[str, object]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_best_outputs(out_dir: Path, best_row: Dict[str, object], best_artifacts: Dict[str, np.ndarray], cfg: ProblemConfig, params: Dict[str, np.ndarray]) -> None:
    t = np.linspace(0.0, cfg.T, cfg.n + 1, dtype=np.float64)
    u = best_artifacts["u"]
    N = best_artifacts["N"]
    lam = best_artifacts["lambda"]
    psi = cfg.gamma - (lam * params["phi"][None, :] * N).sum(axis=1)
    u_sing = singular_control(N, cfg, params)
    H = N @ params["beta"] + cfg.gamma * u + np.vstack(
        [dynamics(N[k], float(u[min(k, cfg.n - 1)]), params) for k in range(cfg.n + 1)]
    ).__mul__(lam).sum(axis=1)

    np.savez(out_dir / "best_neural_pmp_solution.npz", t=t, u=u, N=N, lambda_=lam, psi=psi, u_sing=u_sing, H=H)
    with (out_dir / "best_neural_pmp_trajectory.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "u", "u_sing", "psi", "H", "mean_N"])
        for k in range(cfg.n + 1):
            writer.writerow([t[k], u[k], u_sing[k], psi[k], H[k], N[k].mean()])


def make_pil_plots(out_dir: Path, history_rows: List[Dict[str, object]], best_artifacts: Dict[str, np.ndarray], cfg: ProblemConfig) -> None:
    from PIL import Image, ImageDraw, ImageFont

    try:
        small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 12)
        title = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 16)
    except Exception:
        small = title = ImageFont.load_default()

    def line_chart(draw, box, xs, ys, title_text, ylabel, color, ylim=None):
        x0, y0, x1, y1 = box
        draw.rectangle(box, fill="white", outline="#D0D0D0")
        pad_l, pad_r, pad_t, pad_b = 58, 18, 34, 42
        px0, px1 = x0 + pad_l, x1 - pad_r
        py0, py1 = y0 + pad_t, y1 - pad_b
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = (min(ys), max(ys)) if ylim is None else ylim
        if ymax - ymin < 1e-9:
            ymax = ymin + 1.0
        margin = 0.06 * (ymax - ymin)
        ymin, ymax = ymin - margin, ymax + margin
        for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
            yy = py1 - frac * (py1 - py0)
            xx = px0 + frac * (px1 - px0)
            draw.line([(px0, yy), (px1, yy)], fill="#EEEEEE")
            draw.line([(xx, py0), (xx, py1)], fill="#F2F2F2")
            draw.text((x0 + 8, yy - 7), f"{ymin + frac * (ymax - ymin):.3g}", font=small, fill="#555555")
        draw.line([(px0, py1), (px1, py1)], fill="#333333")
        draw.line([(px0, py0), (px0, py1)], fill="#333333")
        denom_x = xmax - xmin if xmax > xmin else 1.0
        pts = [
            (
                px0 + (x - xmin) / denom_x * (px1 - px0),
                py1 - (y - ymin) / (ymax - ymin) * (py1 - py0),
            )
            for x, y in zip(xs, ys)
        ]
        if len(pts) > 1:
            draw.line(pts, fill=color, width=3)
        draw.text((x0 + pad_l, y0 + 8), title_text, font=title, fill="#222222")
        draw.text((x0 + 8, y0 + 8), ylabel, font=small, fill="#333333")
        draw.text(((px0 + px1) / 2 - 20, y1 - 26), "iteration", font=small, fill="#333333")

    if history_rows:
        iters = sorted({int(r["iter"]) for r in history_rows})
        min_J = []
        min_gap = []
        for it in iters:
            pts = [r for r in history_rows if int(r["iter"]) == it]
            min_J.append(min(float(r["best_J"]) for r in pts))
            min_gap.append(min(float(r["legacy_pmp_gap"]) for r in pts))
        img = Image.new("RGB", (1100, 520), "white")
        draw = ImageDraw.Draw(img)
        line_chart(draw, (40, 40, 530, 480), iters, min_J, "Neural-PMP best objective", "J", "#4C78A8")
        line_chart(draw, (570, 40, 1060, 480), iters, min_gap, "PMP/KKT diagnostic gap", "gap", "#F58518")
        img.save(out_dir / "neural_pmp_training_curve.png")

    t = np.linspace(0.0, cfg.T, cfg.n + 1, dtype=np.float64)
    u = best_artifacts["u"]
    N = best_artifacts["N"]
    img = Image.new("RGB", (1100, 720), "white")
    draw = ImageDraw.Draw(img)
    line_chart(draw, (40, 40, 1060, 330), list(t), list(u), "Best Neural-PMP control", "u", "#4C78A8", (-0.05, cfg.umax + 0.05))
    line_chart(draw, (40, 370, 1060, 680), list(t), list(N.mean(axis=1)), "State rollout", "mean N", "#D62728", (0.0, max(12.0, float(N.max()) * 1.05)))
    img.save(out_dir / "neural_pmp_ut_nt.png")


def make_plots(out_dir: Path, history_rows: List[Dict[str, object]], best_artifacts: Dict[str, np.ndarray], cfg: ProblemConfig) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Matplotlib unavailable ({exc}); using PIL fallback plots.", flush=True)
        make_pil_plots(out_dir, history_rows, best_artifacts, cfg)
        return

    if history_rows:
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), dpi=170)
        for run_id in sorted({int(r["run_id"]) for r in history_rows}):
            pts = [r for r in history_rows if int(r["run_id"]) == run_id]
            pts.sort(key=lambda r: int(r["iter"]))
            label = f"run {run_id}" if run_id < 8 else None
            axes[0].plot([int(r["iter"]) for r in pts], [float(r["best_J"]) for r in pts], alpha=0.55, linewidth=1.1, label=label)
            axes[1].plot(
                [int(r["iter"]) for r in pts],
                [float(r["legacy_pmp_gap"]) for r in pts],
                alpha=0.55,
                linewidth=1.1,
                label=label,
            )
        axes[0].set_title("Neural-PMP best objective")
        axes[0].set_xlabel("iteration")
        axes[0].set_ylabel("J")
        axes[1].set_title("PMP/KKT diagnostic gap")
        axes[1].set_xlabel("iteration")
        axes[1].set_yscale("log")
        axes[1].set_ylabel("gap")
        for ax in axes:
            ax.grid(True, color="#E6E6E6")
        axes[0].legend(frameon=False, fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / "neural_pmp_training_curve.png")
        plt.close(fig)

    t = np.linspace(0.0, cfg.T, cfg.n + 1, dtype=np.float64)
    u = best_artifacts["u"]
    N = best_artifacts["N"]
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.0), dpi=170, sharex=True)
    axes[0].plot(t, u, color="#4C78A8", linewidth=2.0)
    axes[0].set_ylabel("u(t)")
    axes[0].set_ylim(-0.05, cfg.umax + 0.05)
    for j in range(cfg.m):
        axes[1].plot(t, N[:, j], color="#888888", alpha=0.2, linewidth=0.7)
    axes[1].plot(t, N.mean(axis=1), color="#D62728", linewidth=2.0, label="mean N")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("N(t)")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(True, color="#E6E6E6")
    fig.suptitle("Best Neural-PMP control and state rollout")
    fig.tight_layout()
    fig.savefig(out_dir / "neural_pmp_ut_nt.png")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    cfg = make_problem(args)
    params = build_params(cfg)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_csv_ints(args.seeds)
    starts = [x.strip() for x in args.starts.split(",") if x.strip()]
    lrs = parse_csv_floats(args.lrs)
    configs = list(itertools.product(seeds, starts, lrs))
    print(f"Running Neural-PMP oracle-dynamics baseline: {len(configs)} configs, n={cfg.n}, iters={args.iters}", flush=True)

    run_rows: List[Dict[str, object]] = []
    history_rows: List[Dict[str, object]] = []
    best_row: Dict[str, object] = {}
    best_artifacts: Dict[str, np.ndarray] = {}
    for run_id, (seed, start, lr) in enumerate(configs):
        row, hist, artifacts = run_one(run_id, seed, start, lr, cfg, params, args)
        run_rows.append(row)
        history_rows.extend(hist)
        if not best_row or float(row["J"]) < float(best_row["J"]):
            best_row = row
            best_artifacts = artifacts
        print(
            f"run={run_id:03d} seed={seed} start={start} lr={lr:g} "
            f"J={float(row['J']):.6f} gap={float(row['legacy_pmp_gap']):.4g} "
            f"proj={float(row['projected_gradient_residual']):.4g}",
            flush=True,
        )

    run_fields = [
        "run_id",
        "seed",
        "start",
        "lr",
        "iters",
        "best_iter",
        "J",
        "legacy_pmp_gap",
        "normalized_kkt_mean",
        "normalized_kkt_max",
        "projected_gradient_residual",
        "hamiltonian_drift",
        "hamiltonian_rel_drift",
        "u_min",
        "u_max",
        "u_mean",
        "final_mean_N",
        "psi_abs_mean",
        "q_mean",
    ]
    hist_fields = [
        "run_id",
        "seed",
        "start",
        "lr",
        "iter",
        "J",
        "best_J",
        "legacy_pmp_gap",
        "projected_gradient_residual",
        "u_min",
        "u_max",
        "u_mean",
    ]
    write_dict_csv(out_dir / "neural_pmp_runs.csv", run_rows, run_fields)
    write_dict_csv(out_dir / "neural_pmp_history.csv", history_rows, hist_fields)
    save_best_outputs(out_dir, best_row, best_artifacts, cfg, params)
    make_plots(out_dir, history_rows, best_artifacts, cfg)
    with (out_dir / "neural_pmp_summary.txt").open("w") as f:
        f.write("Best Neural-PMP oracle-dynamics run\n")
        for key in run_fields:
            f.write(f"{key}: {best_row.get(key)}\n")
    print(
        f"BEST Neural-PMP J={float(best_row['J']):.6f} "
        f"gap={float(best_row['legacy_pmp_gap']):.4g} "
        f"run={best_row['run_id']} seed={best_row['seed']} start={best_row['start']} lr={best_row['lr']}",
        flush=True,
    )
    print(f"Saved outputs to {out_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Neural-PMP/PMP-Gradient baseline on tumor-control.")
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--m", type=int, default=21)
    parser.add_argument("--umax", type=float, default=3.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=20.0)
    parser.add_argument("--n0", type=float, default=10.0)
    parser.add_argument("--m_suppression", type=float, default=0.5)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--starts", type=str, default="mid,random,front,back")
    parser.add_argument("--lrs", type=str, default="0.02,0.05,0.1,0.2")
    parser.add_argument("--iters", type=int, default=2500)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--singular_eps", type=float, default=0.05)
    parser.add_argument("--singular_tau", type=float, default=0.02)
    parser.add_argument("--out_dir", type=str, default="paper_runs/neural_pmp_baseline_beta01")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
