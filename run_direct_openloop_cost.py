#!/usr/bin/env python3
"""Direct time-mesh minimization of the tumor-control objective.

The optimizer acts on interval-wise zero-order-hold controls. PMP/KKT
quantities are recomputed afterward as independent diagnostics.
"""

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import minimize

from train_paper_pmp_kkt import ProblemConfig, build_params, set_seed
from tumor_problem import TumorProblem, evaluate_zoh_control


def parse_scales(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_label_paths(items: List[str]) -> List[Tuple[str, Path]]:
    pairs = []
    for item in items:
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Expected label:path, got {item}")
        label, path = item.split(":", 1)
        pairs.append((label.strip(), Path(path.strip())))
    return pairs


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


def make_initial_state(scale: float, cfg: ProblemConfig, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.full((cfg.m,), cfg.n0 * scale, device=device, dtype=dtype)


def tumor_g(N: torch.Tensor) -> torch.Tensor:
    return torch.log1p(N.mean(dim=-1))


def dynamics_single(N: torch.Tensor, u: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    G = tumor_g(N)
    return (params["r"] - params["phi"] * u - params["M"] * G) * N


def rk4_objective_interval_controls(
    u: torch.Tensor,
    N0: torch.Tensor,
    cfg: ProblemConfig,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """High-order discrete objective for interval-wise ZOH controls."""

    if u.numel() != cfg.n:
        raise ValueError(f"expected {cfg.n} interval controls, got {u.numel()}")
    dt = cfg.T / cfg.n
    N = N0
    accumulated = torch.zeros((), device=N.device, dtype=N.dtype)
    for k in range(cfg.n):
        uk = u[k]
        k1_N = dynamics_single(N, uk, params)
        k1_J = (params["beta"] * N).sum() + params["gamma"] * uk

        N2 = torch.clamp(N + 0.5 * dt * k1_N, min=1e-10)
        k2_N = dynamics_single(N2, uk, params)
        k2_J = (params["beta"] * N2).sum() + params["gamma"] * uk

        N3 = torch.clamp(N + 0.5 * dt * k2_N, min=1e-10)
        k3_N = dynamics_single(N3, uk, params)
        k3_J = (params["beta"] * N3).sum() + params["gamma"] * uk

        N4 = torch.clamp(N + dt * k3_N, min=1e-10)
        k4_N = dynamics_single(N4, uk, params)
        k4_J = (params["beta"] * N4).sum() + params["gamma"] * uk

        N = torch.clamp(N + dt * (k1_N + 2.0 * k2_N + 2.0 * k3_N + k4_N) / 6.0, min=1e-10)
        accumulated = accumulated + dt * (k1_J + 2.0 * k2_J + 2.0 * k3_J + k4_J) / 6.0
    return accumulated + (params["alpha"] * N).sum()


def diagnostics(u: torch.Tensor, N0: torch.Tensor, cfg: ProblemConfig) -> Dict[str, float]:
    u_np = u.detach().cpu().numpy().astype(np.float64)
    t = np.linspace(0.0, cfg.T, cfg.n + 1, dtype=np.float64)
    problem = TumorProblem(
        T=cfg.T,
        m=cfg.m,
        umax=cfg.umax,
        beta=cfg.beta,
        alpha=cfg.alpha,
        gamma=cfg.gamma,
        n0=float(N0.mean().detach().cpu()),
        m_suppression=cfg.m_suppression,
    )
    result = evaluate_zoh_control(t, u_np, problem, diagnostic_points=max(2001, cfg.n + 1))
    psi = result["diagnostic_psi"]
    return {
        "J": result["J"],
        "pmp_merit_mean": result["pmp_merit_mean"],
        "pmp_merit_rms": result["pmp_merit_rms"],
        "normalized_kkt_mean": result["projected_kkt_mean"],
        "normalized_kkt_max": float(np.max(np.abs(result["diagnostic_u"] / cfg.umax - np.clip(
            result["diagnostic_u"] / cfg.umax - psi / cfg.gamma, 0.0, 1.0
        )))),
        "hamiltonian_drift": float("nan"),
        "hamiltonian_rel_drift": float("nan"),
        "u_min": result["u_min"],
        "u_max": result["u_max"],
        "u_mean": result["u_mean_time"],
        "final_mean_N": result["final_mean_N"],
        "psi_min": float(np.min(psi)),
        "psi_max": float(np.max(psi)),
        "psi_abs_mean": float(np.mean(np.abs(psi))),
        "q_mean": result["singular_fraction"],
        "sign_bad_frac": float("nan"),
    }


def solve_direct_scale(
    scale: float,
    cfg: ProblemConfig,
    params: Dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
    out_dir: Path,
    previous_u: Optional[np.ndarray],
) -> Tuple[np.ndarray, Dict[str, float], List[Dict[str, object]]]:
    N0 = make_initial_state(scale, cfg, device, dtype)
    n_points = cfg.n
    rng = np.random.default_rng(args.seed + int(round(scale * 1000)))

    starts: List[Tuple[str, np.ndarray]] = [
        ("zero", np.zeros(n_points, dtype=np.float64)),
        ("max", np.full(n_points, cfg.umax, dtype=np.float64)),
        ("mid", np.full(n_points, 0.5 * cfg.umax, dtype=np.float64)),
        ("front_loaded", np.linspace(cfg.umax, 0.0, n_points, dtype=np.float64)),
        ("back_loaded", np.linspace(0.0, cfg.umax, n_points, dtype=np.float64)),
    ]
    if previous_u is not None:
        previous = previous_u[:-1] if previous_u.size == cfg.n + 1 else previous_u
        starts.insert(0, ("previous_scale_best", previous.astype(np.float64).copy()))
        if args.initial_only:
            starts = starts[:1]
    for i in range(args.random_starts):
        raw = rng.uniform(0.0, cfg.umax, size=n_points)
        kernel = np.ones(9, dtype=np.float64) / 9.0
        smooth = np.convolve(np.pad(raw, (4, 4), mode="edge"), kernel, mode="valid")
        starts.append((f"random_{i}", smooth.astype(np.float64)))

    def fun_and_grad(u_np: np.ndarray) -> Tuple[float, np.ndarray]:
        u = torch.tensor(u_np, device=device, dtype=dtype, requires_grad=True)
        J_physical = rk4_objective_interval_controls(u, N0, cfg, params)
        J_optimization = J_physical / args.objective_scale
        J_optimization.backward()
        grad = u.grad.detach().cpu().numpy().astype(np.float64)
        return float(J_optimization.detach().cpu()), grad

    start_rows: List[Dict[str, object]] = []
    best_u: Optional[np.ndarray] = None
    best_J = math.inf
    best_label = ""
    for label, u0 in starts:
        t0 = time.time()
        result = minimize(
            fun_and_grad,
            np.clip(u0, 0.0, cfg.umax),
            method="L-BFGS-B",
            jac=True,
            bounds=[(0.0, cfg.umax)] * n_points,
            options={
                "maxiter": args.maxiter,
                "maxfun": args.maxfun,
                "ftol": args.ftol,
                "gtol": args.gtol,
                "maxls": 50,
            },
        )
        elapsed = time.time() - t0
        row = {
            "scale": scale,
            "start": label,
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nit": int(result.nit),
            "nfev": int(result.nfev),
            "J": float(result.fun) * args.objective_scale,
            "J_optimization": float(result.fun),
            "elapsed_sec": elapsed,
        }
        start_rows.append(row)
        print(
            f"scale={scale:g} start={label} "
            f"J={result.fun * args.objective_scale:.8g} "
            f"nit={result.nit} success={result.success} time={elapsed:.1f}s",
            flush=True,
        )
        if float(result.fun) < best_J:
            best_J = float(result.fun)
            best_u = np.clip(result.x.astype(np.float64), 0.0, cfg.umax)
            best_label = label

    if best_u is None:
        raise RuntimeError(f"No optimization result for scale={scale}")

    u_tensor = torch.tensor(np.concatenate([best_u, best_u[-1:]]), device=device, dtype=dtype)
    diag = diagnostics(u_tensor, N0, cfg)
    diag.update({"scale": scale, "best_start": best_label})
    save_scale_outputs(scale, u_tensor, N0, cfg, params, out_dir)
    return best_u, diag, start_rows


def save_scale_outputs(scale: float, u: torch.Tensor, N0: torch.Tensor, cfg: ProblemConfig, params: Dict[str, torch.Tensor], out_dir: Path) -> None:
    del params
    problem = TumorProblem(
        T=cfg.T,
        m=cfg.m,
        umax=cfg.umax,
        beta=cfg.beta,
        alpha=cfg.alpha,
        gamma=cfg.gamma,
        n0=float(N0.mean().detach().cpu()),
        m_suppression=cfg.m_suppression,
    )
    t = np.linspace(0.0, cfg.T, cfg.n + 1, dtype=np.float64)
    u_np = u.detach().cpu().numpy().astype(np.float64)
    result = evaluate_zoh_control(t, u_np, problem, diagnostic_points=cfg.n + 1)
    N = result["diagnostic_N"]
    lam = result["diagnostic_lambda"]
    psi = result["diagnostic_psi"]
    sampled_u = result["diagnostic_u"]
    p = problem.vectors()
    G = np.log1p(N.mean(axis=1))
    f = (p["r"][None, :] - p["phi"][None, :] * sampled_u[:, None] - p["M"][None, :] * G[:, None]) * N
    running = N @ p["beta"] + problem.gamma * sampled_u
    H = running + (lam * f).sum(axis=1)
    tag = f"scale_{scale:g}".replace(".", "p")
    np.savez(
        out_dir / f"{tag}_direct_solution.npz",
        t=t,
        u=u_np,
        N=N,
        lambda_=lam,
        psi=psi,
        H=H,
    )
    with (out_dir / f"{tag}_trajectory.csv").open("w", newline="") as fcsv:
        writer = csv.writer(fcsv, lineterminator="\n")
        writer.writerow(["t", "u", "psi", "H", "mean_N"])
        for k in range(cfg.n + 1):
            writer.writerow([t[k], sampled_u[k], psi[k], H[k], N[k].mean()])


def write_csv(path: Path, fields: List[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_openloop_checkpoint(path: Path, cfg: ProblemConfig, device: torch.device, dtype: torch.dtype):
    from train_paper_pmp_kkt import ParamControl, TimeMLP, TimeTransformer, parse_hidden

    ckpt = torch.load(path, map_location="cpu")
    args = ckpt.get("args", {})
    problem = ckpt.get("problem", cfg.__dict__)
    model_name = args.get("model", "transformer")
    if "open_loop" in ckpt.get("method_key", ""):
        model_name = "transformer"
    if model_name == "transformer":
        d_model = int(args.get("d_model", args.get("open_d_model", 32)))
        heads = int(args.get("heads", args.get("open_heads", 4)))
        layers = int(args.get("layers", args.get("open_layers", 1)))
        model = TimeTransformer(d_model, heads, layers, float(problem.get("umax", cfg.umax)), float(args.get("init_u", 1.5)))
    elif model_name == "mlp":
        model = TimeMLP(parse_hidden(args.get("hidden", "128,128")), float(problem.get("umax", cfg.umax)), float(args.get("init_u", 1.5)))
    elif model_name == "param":
        model = ParamControl(cfg.n + 1, float(problem.get("umax", cfg.umax)), float(args.get("init_u", 1.5)))
    else:
        raise ValueError(f"Unsupported open-loop checkpoint model: {model_name}")
    model.load_state_dict(ckpt["model_state"])
    return model.to(device=device, dtype=dtype).eval()


@torch.no_grad()
def evaluate_open_loop_u(
    u_np: np.ndarray,
    scale: float,
    cfg: ProblemConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, float]:
    u = torch.tensor(u_np, device=device, dtype=dtype)
    N0 = make_initial_state(scale, cfg, device, dtype)
    return diagnostics(u, N0, cfg)


def make_plots(out_dir: Path, scales: List[float], direct_us: Dict[float, np.ndarray], summary_rows: List[Dict[str, object]], comparison_rows: List[Dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots: {exc}", flush=True)
        return

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=170)
    t = np.linspace(0.0, 10.0, len(next(iter(direct_us.values()))), endpoint=False)
    for scale in scales:
        ax.plot(t, direct_us[scale], linewidth=1.8, label=f"s={scale:g}")
    ax.set_xlabel("t")
    ax.set_ylabel("u*(t)")
    ax.set_title("Per-initial-condition direct open-loop controls")
    ax.set_ylim(-0.05, 3.05)
    ax.grid(True, color="#E6E6E6")
    ax.legend(frameon=False, ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "per_ic_direct_controls.png")
    plt.close(fig)

    nominal_tag = "scale_1".replace(".", "p")
    nominal_npz = out_dir / f"{nominal_tag}_direct_solution.npz"
    if nominal_npz.exists():
        data = np.load(nominal_npz)
        fig, axes = plt.subplots(3, 1, figsize=(8.2, 7.0), dpi=170, sharex=True)
        axes[0].plot(data["t"], data["u"], color="#4C78A8", linewidth=2.0)
        axes[0].set_ylabel("u")
        axes[1].plot(data["t"], data["psi"], color="#F58518", linewidth=1.8)
        axes[1].axhline(0.0, color="#333333", linewidth=0.8)
        axes[1].set_ylabel("switching Phi")
        H = data["H"]
        axes[2].plot(data["t"], H - H.mean(), color="#54A24B", linewidth=1.8)
        axes[2].axhline(0.0, color="#333333", linewidth=0.8)
        axes[2].set_ylabel("H - mean(H)")
        axes[2].set_xlabel("t")
        for ax in axes:
            ax.grid(True, color="#E6E6E6")
        fig.suptitle("Nominal direct time-mesh PMP diagnostics")
        fig.tight_layout()
        fig.savefig(out_dir / "nominal_pmp_diagnostics.png")
        plt.close(fig)

    methods = sorted({str(r["method"]) for r in comparison_rows if str(r["method"]) != "per-condition direct reference"})
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=170)
    for method in methods:
        pts = sorted((float(r["scale"]), float(r["relative_gap"])) for r in comparison_rows if r["method"] == method)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", linewidth=2.0, label=method)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xlabel("initial-condition scale")
    ax.set_ylabel("relative objective difference")
    ax.set_title("Cost differences from per-condition direct references")
    ax.grid(True, color="#E6E6E6")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "cost_gap_vs_scale.png")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    if not math.isfinite(args.objective_scale) or args.objective_scale <= 0.0:
        raise ValueError("objective_scale must be positive and finite")
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float64 if args.float64 else torch.float32
    cfg = make_problem(args)
    params = build_params(cfg, device, dtype)
    scales = parse_scales(args.scales)
    if 1.0 not in scales:
        scales.append(1.0)
        scales = sorted(scales)

    print(f"device={device} dtype={dtype} out_dir={out_dir}", flush=True)
    print(f"scales={scales}", flush=True)

    direct_us: Dict[float, np.ndarray] = {}
    summary_rows: List[Dict[str, object]] = []
    start_rows_all: List[Dict[str, object]] = []
    previous_u: Optional[np.ndarray] = None
    if args.initial_control:
        source = np.load(args.initial_control)
        source_u = np.asarray(source["u"], dtype=np.float64).reshape(-1)
        source_t = np.asarray(source["t"], dtype=np.float64).reshape(-1)
        if source_u.size == source_t.size:
            source_u = source_u[:-1]
            source_t = source_t[:-1]
        target_t = np.linspace(0.0, cfg.T, cfg.n, endpoint=False)
        previous_u = np.interp(target_t, source_t, source_u)
    solve_order = [1.0] + [s for s in scales if s != 1.0]
    for scale in solve_order:
        print(f"=== direct cost solve scale={scale:g} ===", flush=True)
        best_u, diag, start_rows = solve_direct_scale(scale, cfg, params, device, dtype, args, out_dir, previous_u)
        direct_us[scale] = best_u
        previous_u = best_u
        summary_rows.append(diag)
        start_rows_all.extend(start_rows)
        print(
            f"BEST scale={scale:g} J_ref={diag['J']:.8g} PMP={diag['pmp_merit_mean']:.5g} "
            f"u=({diag['u_min']:.3f},{diag['u_max']:.3f},{diag['u_mean']:.3f})",
            flush=True,
        )

    summary_rows = sorted(summary_rows, key=lambda r: float(r["scale"]))
    write_csv(
        out_dir / "per_ic_direct_summary.csv",
        [
            "scale",
            "best_start",
            "J",
            "pmp_merit_mean",
            "pmp_merit_rms",
            "normalized_kkt_mean",
            "normalized_kkt_max",
            "hamiltonian_drift",
            "hamiltonian_rel_drift",
            "u_min",
            "u_max",
            "u_mean",
            "final_mean_N",
            "psi_min",
            "psi_max",
            "psi_abs_mean",
            "q_mean",
            "sign_bad_frac",
        ],
        summary_rows,
    )
    write_csv(
        out_dir / "direct_start_results.csv",
        [
            "scale",
            "start",
            "success",
            "status",
            "message",
            "nit",
            "nfev",
            "J",
            "J_optimization",
            "elapsed_sec",
        ],
        start_rows_all,
    )

    direct_J = {float(r["scale"]): float(r["J"]) for r in summary_rows}
    nominal_u = direct_us[1.0]
    comparison_rows: List[Dict[str, object]] = []
    for scale in sorted(scales):
        floor_J = direct_J[scale]
        direct_diag = evaluate_open_loop_u(direct_us[scale], scale, cfg, device, dtype)
        comparison_rows.append({
            "scale": scale,
            "method": "per-condition direct reference",
            "J": direct_diag["J"],
            "gap_to_direct": direct_diag["J"] - floor_J,
            "relative_gap": (direct_diag["J"] - floor_J) / (abs(floor_J) + 1e-12),
            "u_min": direct_diag["u_min"],
            "u_max": direct_diag["u_max"],
            "u_mean": direct_diag["u_mean"],
            "final_mean_N": direct_diag["final_mean_N"],
        })
        frozen_diag = evaluate_open_loop_u(nominal_u, scale, cfg, device, dtype)
        comparison_rows.append({
            "scale": scale,
            "method": "fixed nominal time profile",
            "J": frozen_diag["J"],
            "gap_to_direct": frozen_diag["J"] - floor_J,
            "relative_gap": (frozen_diag["J"] - floor_J) / (abs(floor_J) + 1e-12),
            "u_min": frozen_diag["u_min"],
            "u_max": frozen_diag["u_max"],
            "u_mean": frozen_diag["u_mean"],
            "final_mean_N": frozen_diag["final_mean_N"],
        })

    t_grid = torch.linspace(0.0, 1.0, cfg.n + 1, device=device, dtype=dtype)
    for label, path in parse_label_paths(args.openloop_ckpt):
        if not path.exists():
            print(f"Skipping missing open-loop checkpoint {path}", flush=True)
            continue
        model = load_openloop_checkpoint(path, cfg, device, dtype)
        with torch.no_grad():
            u_np = model(t_grid).detach().cpu().numpy().astype(np.float64)
        for scale in sorted(scales):
            floor_J = direct_J[scale]
            diag = evaluate_open_loop_u(u_np, scale, cfg, device, dtype)
            comparison_rows.append({
                "scale": scale,
                "method": label,
                "J": diag["J"],
                "gap_to_direct": diag["J"] - floor_J,
                "relative_gap": (diag["J"] - floor_J) / (abs(floor_J) + 1e-12),
                "u_min": diag["u_min"],
                "u_max": diag["u_max"],
                "u_mean": diag["u_mean"],
                "final_mean_N": diag["final_mean_N"],
            })

    comparison_rows = sorted(comparison_rows, key=lambda r: (float(r["scale"]), str(r["method"])))
    write_csv(
        out_dir / "cost_comparison_summary.csv",
        ["scale", "method", "J", "gap_to_direct", "relative_gap", "u_min", "u_max", "u_mean", "final_mean_N"],
        comparison_rows,
    )
    make_plots(out_dir, sorted(scales), direct_us, summary_rows, comparison_rows)

    print("=== direct time-mesh summary ===", flush=True)
    for row in summary_rows:
        print(
            f"scale={float(row['scale']):g} J_ref={float(row['J']):.8g} "
            f"PMP={float(row['pmp_merit_mean']):.5g} "
            f"u=({float(row['u_min']):.3f},{float(row['u_max']):.3f},{float(row['u_mean']):.3f})",
            flush=True,
        )
    print(f"Saved outputs to {out_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Direct time-mesh minimization of the tumor objective.")
    parser.add_argument("--out_dir", type=str, default="paper_runs/direct_openloop_cost_beta01")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--float64", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scales", type=str, default="0.25,0.5,0.75,1.0,1.25,1.5,2.0")

    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--m", type=int, default=21)
    parser.add_argument("--umax", type=float, default=3.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=20.0)
    parser.add_argument("--n0", type=float, default=10.0)
    parser.add_argument("--m_suppression", type=float, default=0.5)
    parser.add_argument("--random_starts", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=600)
    parser.add_argument("--maxfun", type=int, default=2000)
    parser.add_argument("--ftol", type=float, default=1e-10)
    parser.add_argument("--gtol", type=float, default=1e-7)
    parser.add_argument(
        "--objective_scale",
        type=float,
        default=1.0,
        help="positive constant dividing J and its gradient inside L-BFGS-B",
    )
    parser.add_argument("--openloop_ckpt", action="append", default=[])
    parser.add_argument("--initial_control", type=str, default="")
    parser.add_argument("--initial_only", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
