import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from train_feedback import build_feedback_model, pmp_kkt_loss as feedback_loss
from train_openloop import (
    ParamControl,
    ProblemConfig,
    TimeMLP,
    TimeTransformer,
    build_params,
    parse_hidden,
)


def parse_scales(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def make_initial_states(scales: List[float], cfg: ProblemConfig, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    s = torch.tensor(scales, device=device, dtype=dtype).view(-1, 1)
    return cfg.n0 * s.expand(-1, cfg.m)


def tumor_g(N: torch.Tensor) -> torch.Tensor:
    return torch.log1p(N.mean(dim=-1))


def dynamics(N: torch.Tensor, u: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    G = tumor_g(N)
    return (params["r"].unsqueeze(0) - params["phi"].unsqueeze(0) * u.unsqueeze(-1) - params["M"].unsqueeze(0) * G.unsqueeze(-1)) * N


def simulate_open_loop(u: torch.Tensor, N0: torch.Tensor, cfg: ProblemConfig, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    dt = cfg.T / cfg.n
    N = N0
    states = [N]
    for k in range(cfg.n):
        uk = u[k].expand(N0.shape[0])
        N = N + dt * dynamics(N, uk, params)
        N = torch.clamp(N, min=1e-8)
        states.append(N)
    return torch.stack(states, dim=1)


def dH_dN(N: torch.Tensor, lam: torch.Tensor, u: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    G = tumor_g(N)
    mean_N = N.mean(dim=-1)
    dG = (1.0 / N.shape[-1]) / (1.0 + mean_N)
    a = params["r"].unsqueeze(0) - params["phi"].unsqueeze(0) * u.unsqueeze(-1) - params["M"].unsqueeze(0) * G.unsqueeze(-1)
    coupling = (lam * params["M"].unsqueeze(0) * N).sum(dim=-1)
    return params["beta"].unsqueeze(0) + lam * a - dG.unsqueeze(-1) * coupling.unsqueeze(-1)


def compute_costate(N: torch.Tensor, u: torch.Tensor, cfg: ProblemConfig, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    dt = cfg.T / cfg.n
    B = N.shape[0]
    lam = params["alpha"].unsqueeze(0).expand(B, -1)
    lams = [None] * (cfg.n + 1)
    lams[cfg.n] = lam
    for k in range(cfg.n - 1, -1, -1):
        uk = u[:, k] if u.ndim == 2 else u[k].expand(B)
        lam = lam + dt * dH_dN(N[:, k], lam, uk, params)
        lams[k] = lam
    return torch.stack(lams, dim=1)


def singular_control(N: torch.Tensor, params: Dict[str, torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
    G = tumor_g(N)
    numerator = (params["beta"] * (params["r"].view(1, 1, -1) - G.unsqueeze(-1) * params["M"].view(1, 1, -1)) * N).sum(dim=-1)
    denominator = (params["beta"] * params["phi"] * N).sum(dim=-1).clamp_min(eps)
    return numerator / denominator


def objective_value(N: torch.Tensor, u: torch.Tensor, cfg: ProblemConfig, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    dt = cfg.T / cfg.n
    if u.ndim == 1:
        u = u.unsqueeze(0).expand(N.shape[0], -1)
    running = (N * params["beta"].view(1, 1, -1)).sum(dim=-1) + params["gamma"] * u
    integral = dt * (0.5 * running[:, 0] + running[:, 1:-1].sum(dim=-1) + 0.5 * running[:, -1])
    terminal = (params["alpha"].view(1, -1) * N[:, -1]).sum(dim=-1)
    return terminal + integral


def open_loop_metrics(u: torch.Tensor, N0: torch.Tensor, cfg: ProblemConfig, params: Dict[str, torch.Tensor], eps: float, tau: float) -> Dict[str, torch.Tensor]:
    N = simulate_open_loop(u, N0, cfg, params)
    lam = compute_costate(N, u, cfg, params)
    uu = u.unsqueeze(0).expand(N0.shape[0], -1)
    psi = params["gamma"] - (lam * params["phi"].view(1, 1, -1) * N).sum(dim=-1)
    u_sing = singular_control(N, params)
    admissible = ((u_sing >= 0.0) & (u_sing <= params["umax"])).to(u.dtype)
    q = torch.sigmoid((eps - psi.abs()) / tau) * admissible
    l_sing = (uu - u_sing).pow(2)
    l_ns = (torch.relu(psi) * uu + torch.relu(-psi) * (params["umax"] - uu)).pow(2)
    gap_by_sample = (q * l_sing + (1.0 - q) * l_ns).mean(dim=1)
    J = objective_value(N, uu, cfg, params)
    return {"N": N, "u": uu, "gap": gap_by_sample, "objective": J, "psi": psi, "q": q}


def build_open_loop_model(ckpt: Dict, device: torch.device, dtype: torch.dtype):
    args = ckpt["args"]
    problem = ckpt["problem"]
    n_points = int(problem["n"]) + 1
    model_name = args.get("model", "transformer")
    if model_name == "transformer":
        model = TimeTransformer(int(args.get("d_model", 32)), int(args.get("heads", 4)), int(args.get("layers", 1)), float(problem["umax"]), float(args.get("init_u", 1.5)))
    elif model_name == "mlp":
        model = TimeMLP(parse_hidden(args.get("hidden", "128,128")), float(problem["umax"]), float(args.get("init_u", 1.5)))
    elif model_name == "param":
        model = ParamControl(n_points, float(problem["umax"]), float(args.get("init_u", 1.5)))
    else:
        raise ValueError(f"Unsupported open-loop model: {model_name}")
    model.load_state_dict(ckpt["model_state"])
    return model.to(device=device, dtype=dtype).eval()


class ArgsDict:
    def __init__(self, values: Dict):
        self.__dict__.update(values)


def load_feedback(path: Path, device: torch.device, dtype: torch.dtype) -> Tuple[nn.Module, ProblemConfig, Dict]:
    ckpt = torch.load(path, map_location="cpu")
    p = ckpt["problem"]
    a = ckpt["args"]
    cfg = ProblemConfig(**p)
    defaults = {
        "model": "mlp",
        "hidden": "128,128",
        "d_model": 64,
        "heads": 4,
        "layers": 2,
        "state_scale": 15.0,
        "init_u": 1.5,
    }
    model_args = ArgsDict({**defaults, **a})
    model = build_feedback_model(cfg, model_args)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device=device, dtype=dtype).eval(), cfg, a


def load_open_loop(path: Path, device: torch.device, dtype: torch.dtype):
    ckpt = torch.load(path, map_location="cpu")
    cfg = ProblemConfig(**ckpt["problem"])
    model = build_open_loop_model(ckpt, device, dtype)
    return model, cfg, ckpt["args"]


def evaluate_checkpoint(kind: str, path: Path, label: str, scales: List[float], out_rows: List[Dict], args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    dtype = torch.float64 if args.float64 else torch.float32
    if kind == "feedback":
        model, cfg, train_args = load_feedback(path, device, dtype)
        params = build_params(cfg, device, dtype)
        N0 = make_initial_states(scales, cfg, device, dtype)
        with torch.no_grad():
            pack = feedback_loss(model, N0, cfg, params, args.singular_eps, args.singular_tau)
            N = pack["N"]
            u = pack["u"]
            gap_by_sample = ((pack["q"] * (u - pack["u_sing"]).pow(2)) + (1.0 - pack["q"]) * (torch.relu(pack["psi"]) * u + torch.relu(-pack["psi"]) * (params["umax"] - u)).pow(2)).mean(dim=1)
            J = (N * params["beta"].view(1, 1, -1)).sum()
            obj = None
            # Use objective helper from module through pack-level value only for mean; recompute per sample via open-loop-style helper.
            obj_by_sample = objective_value(N, u, cfg, params)
            q = pack["q"]
    elif kind == "openloop":
        model, cfg, train_args = load_open_loop(path, device, dtype)
        params = build_params(cfg, device, dtype)
        N0 = make_initial_states(scales, cfg, device, dtype)
        t = torch.linspace(0.0, 1.0, cfg.n + 1, device=device, dtype=dtype)
        with torch.no_grad():
            u_traj = model(t)
            pack = open_loop_metrics(u_traj, N0, cfg, params, args.singular_eps, args.singular_tau)
            N = pack["N"]
            u = pack["u"]
            gap_by_sample = pack["gap"]
            obj_by_sample = pack["objective"]
            q = pack["q"]
    else:
        raise ValueError(kind)

    for idx, scale in enumerate(scales):
        out_rows.append(
            {
                "label": label,
                "kind": kind,
                "checkpoint": str(path),
                "scale": scale,
                "n": cfg.n,
                "gamma": cfg.gamma,
                "alpha": cfg.alpha,
                "beta": cfg.beta,
                "gap": float(gap_by_sample[idx].detach().cpu()),
                "objective": float(obj_by_sample[idx].detach().cpu()),
                "u_min": float(u[idx].min().detach().cpu()),
                "u_max": float(u[idx].max().detach().cpu()),
                "u_mean": float(u[idx].mean().detach().cpu()),
                "final_mean_N": float(N[idx, -1].mean().detach().cpu()),
                "min_mean_N": float(N[idx].mean(dim=-1).min().detach().cpu()),
                "q_mean": float(q[idx].mean().detach().cpu()),
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate feedback/open-loop policies across N0 scales.")
    parser.add_argument("--feedback", action="append", default=[], help="label:path to feedback checkpoint")
    parser.add_argument("--openloop", action="append", default=[], help="label:path to open-loop checkpoint")
    parser.add_argument("--scales", type=str, default="0.25,0.5,0.75,1,1.25,1.5,2")
    parser.add_argument("--singular_eps", type=float, default=0.05)
    parser.add_argument("--singular_tau", type=float, default=0.02)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--float64", action="store_true")
    parser.add_argument("--out_csv", type=str, default="runs/generalization.csv")
    args = parser.parse_args()
    rows: List[Dict] = []
    scales = parse_scales(args.scales)
    for item in args.feedback:
        label, path = item.split(":", 1)
        evaluate_checkpoint("feedback", Path(path), label, scales, rows, args)
    for item in args.openloop:
        label, path = item.split(":", 1)
        evaluate_checkpoint("openloop", Path(path), label, scales, rows, args)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out} with {len(rows)} rows")


if __name__ == "__main__":
    main()
