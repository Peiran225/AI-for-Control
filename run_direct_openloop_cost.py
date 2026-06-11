#!/usr/bin/env python3
"""Direct open-loop cost minimization and cost-based comparisons.

This runner is for the numerical optimal-control audit prompted by the feedback
discussion.  It treats PMP/KKT quantities as a posteriori diagnostics and uses
the actual objective J for comparisons.
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

from train_feedback_pmp_kkt import build_feedback_model, simulate_feedback
from train_paper_pmp_kkt import ProblemConfig, build_params, set_seed


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


def simulate_open_loop(u: torch.Tensor, N0: torch.Tensor, cfg: ProblemConfig, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    dt = cfg.T / cfg.n
    N = N0
    states = [N]
    for k in range(cfg.n):
        N = N + dt * dynamics_single(N, u[k], params)
        N = torch.clamp(N, min=1e-10)
        states.append(N)
    return torch.stack(states, dim=0)


def objective_value(N: torch.Tensor, u: torch.Tensor, cfg: ProblemConfig, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    dt = cfg.T / cfg.n
    running = (N * params["beta"].unsqueeze(0)).sum(dim=-1) + params["gamma"] * u
    integral = dt * (0.5 * running[0] + running[1:-1].sum() + 0.5 * running[-1])
    terminal = (params["alpha"] * N[-1]).sum()
    return terminal + integral


def dH_dN(N: torch.Tensor, lam: torch.Tensor, u: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    G = tumor_g(N)
    mean_N = N.mean()
    dG = (1.0 / N.numel()) / (1.0 + mean_N)
    a = params["r"] - params["phi"] * u - params["M"] * G
    coupling = (lam * params["M"] * N).sum()
    return params["beta"] + lam * a - dG * coupling


def compute_costate(N: torch.Tensor, u: torch.Tensor, cfg: ProblemConfig, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    dt = cfg.T / cfg.n
    lams: List[Optional[torch.Tensor]] = [None] * (cfg.n + 1)
    lam = params["alpha"]
    lams[cfg.n] = lam
    for k in range(cfg.n - 1, -1, -1):
        lam = lam + dt * dH_dN(N[k], lam, u[k], params)
        lams[k] = lam
    return torch.stack([x for x in lams if x is not None], dim=0)


def singular_control(N: torch.Tensor, params: Dict[str, torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
    G = tumor_g(N)
    numerator = (params["beta"] * (params["r"].unsqueeze(0) - G.unsqueeze(-1) * params["M"].unsqueeze(0)) * N).sum(dim=-1)
    denominator = (params["beta"] * params["phi"] * N).sum(dim=-1).clamp_min(eps)
    return numerator / denominator


def diagnostics(u: torch.Tensor, N0: torch.Tensor, cfg: ProblemConfig, params: Dict[str, torch.Tensor], args: argparse.Namespace) -> Dict[str, float]:
    with torch.no_grad():
        N = simulate_open_loop(u, N0, cfg, params)
        lam = compute_costate(N, u, cfg, params)
        psi = params["gamma"] - (lam * params["phi"].unsqueeze(0) * N).sum(dim=-1)
        u_sing = singular_control(N, params)
        admissible = ((u_sing >= 0.0) & (u_sing <= params["umax"])).to(u.dtype)
        q = torch.sigmoid((args.singular_eps - psi.abs()) / args.singular_tau) * admissible
        l_sing = (u - u_sing).pow(2)
        l_ns = (torch.relu(psi) * u + torch.relu(-psi) * (params["umax"] - u)).pow(2)
        legacy_gap = (q * l_sing + (1.0 - q) * l_ns).mean()
        comp = torch.relu(psi) * u + torch.relu(-psi) * (params["umax"] - u)
        norm_comp = comp / (float(cfg.gamma) * float(cfg.umax) + 1e-12)
        f = torch.stack([dynamics_single(N[k], u[k], params) for k in range(cfg.n + 1)], dim=0)
        running = (N * params["beta"].unsqueeze(0)).sum(dim=-1) + params["gamma"] * u
        H = running + (lam * f).sum(dim=-1)
        H_drift = H.max() - H.min()
        H_mean_abs = H.abs().mean().clamp_min(1e-12)
        sign_bad = ((psi > 0.0) & (u > 1e-3)) | ((psi < 0.0) & (u < cfg.umax - 1e-3))
        return {
            "J": float(objective_value(N, u, cfg, params).detach().cpu()),
            "legacy_pmp_gap": float(legacy_gap.detach().cpu()),
            "normalized_kkt_mean": float(norm_comp.mean().detach().cpu()),
            "normalized_kkt_max": float(norm_comp.max().detach().cpu()),
            "hamiltonian_drift": float(H_drift.detach().cpu()),
            "hamiltonian_rel_drift": float((H_drift / H_mean_abs).detach().cpu()),
            "u_min": float(u.min().detach().cpu()),
            "u_max": float(u.max().detach().cpu()),
            "u_mean": float(u.mean().detach().cpu()),
            "final_mean_N": float(N[-1].mean().detach().cpu()),
            "psi_min": float(psi.min().detach().cpu()),
            "psi_max": float(psi.max().detach().cpu()),
            "psi_abs_mean": float(psi.abs().mean().detach().cpu()),
            "q_mean": float(q.mean().detach().cpu()),
            "sign_bad_frac": float(sign_bad.to(u.dtype).mean().detach().cpu()),
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
    n_points = cfg.n + 1
    rng = np.random.default_rng(args.seed + int(round(scale * 1000)))

    starts: List[Tuple[str, np.ndarray]] = [
        ("zero", np.zeros(n_points, dtype=np.float64)),
        ("max", np.full(n_points, cfg.umax, dtype=np.float64)),
        ("mid", np.full(n_points, 0.5 * cfg.umax, dtype=np.float64)),
        ("front_loaded", np.linspace(cfg.umax, 0.0, n_points, dtype=np.float64)),
        ("back_loaded", np.linspace(0.0, cfg.umax, n_points, dtype=np.float64)),
    ]
    if previous_u is not None:
        starts.insert(0, ("previous_scale_best", previous_u.astype(np.float64).copy()))
    for i in range(args.random_starts):
        raw = rng.uniform(0.0, cfg.umax, size=n_points)
        kernel = np.ones(9, dtype=np.float64) / 9.0
        smooth = np.convolve(np.pad(raw, (4, 4), mode="edge"), kernel, mode="valid")
        starts.append((f"random_{i}", smooth.astype(np.float64)))

    def fun_and_grad(u_np: np.ndarray) -> Tuple[float, np.ndarray]:
        u = torch.tensor(u_np, device=device, dtype=dtype, requires_grad=True)
        N = simulate_open_loop(u, N0, cfg, params)
        J = objective_value(N, u, cfg, params)
        J.backward()
        grad = u.grad.detach().cpu().numpy().astype(np.float64)
        return float(J.detach().cpu()), grad

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
            "J": float(result.fun),
            "elapsed_sec": elapsed,
        }
        start_rows.append(row)
        print(
            f"scale={scale:g} start={label} J={result.fun:.8g} "
            f"nit={result.nit} success={result.success} time={elapsed:.1f}s",
            flush=True,
        )
        if float(result.fun) < best_J:
            best_J = float(result.fun)
            best_u = np.clip(result.x.astype(np.float64), 0.0, cfg.umax)
            best_label = label

    if best_u is None:
        raise RuntimeError(f"No optimization result for scale={scale}")

    u_tensor = torch.tensor(best_u, device=device, dtype=dtype)
    diag = diagnostics(u_tensor, N0, cfg, params, args)
    diag.update({"scale": scale, "best_start": best_label})
    save_scale_outputs(scale, u_tensor, N0, cfg, params, out_dir)
    return best_u, diag, start_rows


def save_scale_outputs(scale: float, u: torch.Tensor, N0: torch.Tensor, cfg: ProblemConfig, params: Dict[str, torch.Tensor], out_dir: Path) -> None:
    with torch.no_grad():
        N = simulate_open_loop(u, N0, cfg, params)
        lam = compute_costate(N, u, cfg, params)
        psi = params["gamma"] - (lam * params["phi"].unsqueeze(0) * N).sum(dim=-1)
        f = torch.stack([dynamics_single(N[k], u[k], params) for k in range(cfg.n + 1)], dim=0)
        running = (N * params["beta"].unsqueeze(0)).sum(dim=-1) + params["gamma"] * u
        H = running + (lam * f).sum(dim=-1)
        t = torch.linspace(0.0, cfg.T, cfg.n + 1, device=u.device, dtype=u.dtype)
        tag = f"scale_{scale:g}".replace(".", "p")
        np.savez(
            out_dir / f"{tag}_direct_solution.npz",
            t=t.detach().cpu().numpy(),
            u=u.detach().cpu().numpy(),
            N=N.detach().cpu().numpy(),
            lambda_=lam.detach().cpu().numpy(),
            psi=psi.detach().cpu().numpy(),
            H=H.detach().cpu().numpy(),
        )
        with (out_dir / f"{tag}_trajectory.csv").open("w", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(["t", "u", "psi", "H", "mean_N"])
            for k in range(cfg.n + 1):
                writer.writerow([
                    float(t[k].detach().cpu()),
                    float(u[k].detach().cpu()),
                    float(psi[k].detach().cpu()),
                    float(H[k].detach().cpu()),
                    float(N[k].mean().detach().cpu()),
                ])


def write_csv(path: Path, fields: List[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
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


def feedback_args_from_checkpoint(ckpt: Dict, cfg: ProblemConfig) -> argparse.Namespace:
    values = dict(ckpt.get("args", {}))
    method_key = ckpt.get("method_key", "")
    if "transformer" in method_key:
        values["model"] = "transformer"
    else:
        values.setdefault("model", "mlp")
    values.setdefault("hidden", "128,128")
    values["d_model"] = int(values.get("d_model", values.get("feedback_d_model", 64)))
    values["heads"] = int(values.get("heads", values.get("feedback_heads", 4)))
    values["layers"] = int(values.get("layers", values.get("feedback_layers", 2)))
    values.setdefault("state_scale", 15.0)
    values.setdefault("init_u", 1.5)
    values.setdefault("umax", cfg.umax)
    return argparse.Namespace(**values)


def load_feedback_checkpoint(path: Path, cfg: ProblemConfig, device: torch.device, dtype: torch.dtype):
    ckpt = torch.load(path, map_location="cpu")
    model_args = feedback_args_from_checkpoint(ckpt, cfg)
    model = build_feedback_model(cfg, model_args)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device=device, dtype=dtype).eval()


@torch.no_grad()
def evaluate_open_loop_u(u_np: np.ndarray, scale: float, cfg: ProblemConfig, params: Dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype, args: argparse.Namespace) -> Dict[str, float]:
    u = torch.tensor(u_np, device=device, dtype=dtype)
    N0 = make_initial_state(scale, cfg, device, dtype)
    return diagnostics(u, N0, cfg, params, args)


@torch.no_grad()
def evaluate_feedback_model(model, scale: float, cfg: ProblemConfig, params: Dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype, args: argparse.Namespace) -> Dict[str, float]:
    N0 = make_initial_state(scale, cfg, device, dtype).unsqueeze(0)
    N, u = simulate_feedback(model, N0, cfg, params)
    running = (N * params["beta"].view(1, 1, -1)).sum(dim=-1) + params["gamma"] * u
    dt = cfg.T / cfg.n
    integral = dt * (0.5 * running[:, 0] + running[:, 1:-1].sum(dim=-1) + 0.5 * running[:, -1])
    terminal = (params["alpha"].view(1, -1) * N[:, -1]).sum(dim=-1)
    J = terminal + integral
    return {
        "J": float(J.mean().detach().cpu()),
        "u_min": float(u.min().detach().cpu()),
        "u_max": float(u.max().detach().cpu()),
        "u_mean": float(u.mean().detach().cpu()),
        "final_mean_N": float(N[:, -1].mean().detach().cpu()),
    }


def make_plots(out_dir: Path, scales: List[float], direct_us: Dict[float, np.ndarray], summary_rows: List[Dict[str, object]], comparison_rows: List[Dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plots: {exc}", flush=True)
        return

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=170)
    t = np.linspace(0.0, 10.0, len(next(iter(direct_us.values()))))
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
        fig.suptitle("Nominal direct open-loop PMP diagnostics")
        fig.tight_layout()
        fig.savefig(out_dir / "nominal_pmp_diagnostics.png")
        plt.close(fig)

    methods = sorted({str(r["method"]) for r in comparison_rows if str(r["method"]) != "per-IC direct open-loop"})
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=170)
    for method in methods:
        pts = sorted((float(r["scale"]), float(r["relative_gap"])) for r in comparison_rows if r["method"] == method)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", linewidth=2.0, label=method)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xlabel("initial-condition scale")
    ax.set_ylabel("(J - J*_OL) / |J*_OL|")
    ax.set_title("Cost gaps relative to per-IC direct open-loop optimum")
    ax.grid(True, color="#E6E6E6")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "cost_gap_vs_scale.png")
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
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
    solve_order = [1.0] + [s for s in scales if s != 1.0]
    for scale in solve_order:
        print(f"=== direct cost solve scale={scale:g} ===", flush=True)
        best_u, diag, start_rows = solve_direct_scale(scale, cfg, params, device, dtype, args, out_dir, previous_u)
        direct_us[scale] = best_u
        previous_u = best_u
        summary_rows.append(diag)
        start_rows_all.extend(start_rows)
        print(
            f"BEST scale={scale:g} J={diag['J']:.8g} legacy_gap={diag['legacy_pmp_gap']:.5g} "
            f"H_rel_drift={diag['hamiltonian_rel_drift']:.5g} u=({diag['u_min']:.3f},{diag['u_max']:.3f},{diag['u_mean']:.3f})",
            flush=True,
        )

    summary_rows = sorted(summary_rows, key=lambda r: float(r["scale"]))
    write_csv(
        out_dir / "per_ic_direct_summary.csv",
        [
            "scale",
            "best_start",
            "J",
            "legacy_pmp_gap",
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
        ["scale", "start", "success", "status", "message", "nit", "nfev", "J", "elapsed_sec"],
        start_rows_all,
    )

    direct_J = {float(r["scale"]): float(r["J"]) for r in summary_rows}
    nominal_u = direct_us[1.0]
    comparison_rows: List[Dict[str, object]] = []
    for scale in sorted(scales):
        floor_J = direct_J[scale]
        direct_diag = evaluate_open_loop_u(direct_us[scale], scale, cfg, params, device, dtype, args)
        comparison_rows.append({
            "scale": scale,
            "method": "per-IC direct open-loop",
            "J": direct_diag["J"],
            "gap_to_direct": direct_diag["J"] - floor_J,
            "relative_gap": (direct_diag["J"] - floor_J) / (abs(floor_J) + 1e-12),
            "u_min": direct_diag["u_min"],
            "u_max": direct_diag["u_max"],
            "u_mean": direct_diag["u_mean"],
            "final_mean_N": direct_diag["final_mean_N"],
        })
        frozen_diag = evaluate_open_loop_u(nominal_u, scale, cfg, params, device, dtype, args)
        comparison_rows.append({
            "scale": scale,
            "method": "frozen nominal direct open-loop",
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
            diag = evaluate_open_loop_u(u_np, scale, cfg, params, device, dtype, args)
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

    for label, path in parse_label_paths(args.feedback_ckpt):
        if not path.exists():
            print(f"Skipping missing feedback checkpoint {path}", flush=True)
            continue
        model = load_feedback_checkpoint(path, cfg, device, dtype)
        for scale in sorted(scales):
            floor_J = direct_J[scale]
            diag = evaluate_feedback_model(model, scale, cfg, params, device, dtype, args)
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

    print("=== direct open-loop summary ===", flush=True)
    for row in summary_rows:
        print(
            f"scale={float(row['scale']):g} J*={float(row['J']):.8g} "
            f"legacy_gap={float(row['legacy_pmp_gap']):.5g} Hrel={float(row['hamiltonian_rel_drift']):.5g} "
            f"u=({float(row['u_min']):.3f},{float(row['u_max']):.3f},{float(row['u_mean']):.3f})",
            flush=True,
        )
    print(f"Saved outputs to {out_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Direct open-loop cost minimization audit.")
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
    parser.add_argument("--singular_eps", type=float, default=0.05)
    parser.add_argument("--singular_tau", type=float, default=0.02)

    parser.add_argument("--random_starts", type=int, default=3)
    parser.add_argument("--maxiter", type=int, default=600)
    parser.add_argument("--maxfun", type=int, default=2000)
    parser.add_argument("--ftol", type=float, default=1e-10)
    parser.add_argument("--gtol", type=float, default=1e-7)
    parser.add_argument("--openloop_ckpt", action="append", default=[])
    parser.add_argument("--feedback_ckpt", action="append", default=[])
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
