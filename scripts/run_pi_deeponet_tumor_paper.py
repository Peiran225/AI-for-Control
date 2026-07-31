#!/usr/bin/env python3
"""Paper-algorithm PI-DeepONet adaptation for the tumor OCP.

The paper [5] has no author code release that we could locate, so this script
implements its Algorithm-1 structure directly: branch/trunk DeepONet,
physics-informed policy-evaluation loss, repeated policy improvement, and a
terminal-function family supplied through branch sensors.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_transformer_objective_ablation import eval_common_grid, eval_rk4  # noqa: E402
from tumor_problem import NOMINAL_TUMOR_PROBLEM  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


@dataclass
class TumorConfig:
    T: float = NOMINAL_TUMOR_PROBLEM.T
    m: int = NOMINAL_TUMOR_PROBLEM.m
    umax: float = NOMINAL_TUMOR_PROBLEM.umax
    beta: float = NOMINAL_TUMOR_PROBLEM.beta
    alpha: float = NOMINAL_TUMOR_PROBLEM.alpha
    gamma: float = NOMINAL_TUMOR_PROBLEM.gamma
    n0: float = NOMINAL_TUMOR_PROBLEM.n0
    m_suppression: float = NOMINAL_TUMOR_PROBLEM.m_suppression
    state_scale: float = 220.0
    value_scale: float = 400.0


def build_params(cfg: TumorConfig, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    x = torch.linspace(0.0, 1.0, cfg.m, device=device, dtype=dtype)
    return {
        "r": 2.0 / (1.0 + 3.0 * x.pow(4)),
        "phi": 1.0 / (1.0 + x.pow(2)),
        "M": torch.full((cfg.m,), cfg.m_suppression, device=device, dtype=dtype),
        "beta": torch.full((cfg.m,), cfg.beta, device=device, dtype=dtype),
        "alpha": torch.full((cfg.m,), cfg.alpha, device=device, dtype=dtype),
        "N0": torch.full((cfg.m,), cfg.n0, device=device, dtype=dtype),
    }


def np_params(cfg: TumorConfig) -> dict[str, np.ndarray]:
    x = np.linspace(0.0, 1.0, cfg.m)
    return {
        "r": 2.0 / (1.0 + 3.0 * x**4),
        "phi": 1.0 / (1.0 + x**2),
        "M": np.full(cfg.m, cfg.m_suppression),
        "beta": np.full(cfg.m, cfg.beta),
        "alpha": np.full(cfg.m, cfg.alpha),
    }


def normalize_trunk(t: torch.Tensor, N: torch.Tensor, cfg: TumorConfig) -> torch.Tensor:
    t_col = (t / cfg.T).unsqueeze(-1)
    N_col = torch.log1p(torch.clamp(N, min=1e-8)) / math.log1p(cfg.state_scale)
    return torch.cat([t_col, N_col], dim=-1)


def dynamics(N: torch.Tensor, u: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
    G = torch.log1p(N.mean(dim=-1, keepdim=True))
    return (params["r"].unsqueeze(0) - params["phi"].unsqueeze(0) * u - params["M"].unsqueeze(0) * G) * N


def singular_control(N: torch.Tensor, params: dict[str, torch.Tensor], cfg: TumorConfig) -> torch.Tensor:
    G = torch.log1p(N.mean(dim=-1, keepdim=True))
    num = (params["beta"].unsqueeze(0) * (params["r"].unsqueeze(0) - params["M"].unsqueeze(0) * G) * N).sum(
        dim=-1, keepdim=True
    )
    den = (params["beta"].unsqueeze(0) * params["phi"].unsqueeze(0) * N).sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.clamp(num / den, 0.0, cfg.umax)


class PDeepONet(nn.Module):
    def __init__(self, cfg: TumorConfig, sensor_count: int, width: int, branch_depth: int, trunk_depth: int):
        super().__init__()
        self.cfg = cfg

        def mlp(in_dim: int, depth: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            last = in_dim
            for _ in range(depth):
                layers.append(nn.Linear(last, width))
                layers.append(nn.Tanh())
                last = width
            layers.append(nn.Linear(last, width))
            return nn.Sequential(*layers)

        self.branch = mlp(sensor_count, branch_depth)
        self.trunk = mlp(cfg.m + 1, trunk_depth)
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, branch: torch.Tensor, t: torch.Tensor, N: torch.Tensor) -> torch.Tensor:
        b = self.branch(branch)
        h = self.trunk(normalize_trunk(t, N, self.cfg))
        return self.cfg.value_scale * ((b * h).sum(dim=-1) / math.sqrt(h.shape[-1]) + self.bias)


def make_sensor_states(cfg: TumorConfig, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sensors = []
    base_path = ROOT / "paper_runs" / "direct_openloop_cost_beta01_n800_refined" / "scale_1_direct_solution.npz"
    if base_path.exists():
        data = np.load(base_path)
        N = np.asarray(data["N"], dtype=np.float64)
        idx = np.linspace(0, len(N) - 1, min(count // 2, len(N))).astype(int)
        sensors.extend(N[idx])
    while len(sensors) < count:
        sensors.append(np.clip(rng.lognormal(mean=np.log(6.0), sigma=0.75, size=cfg.m), 0.02, cfg.state_scale))
    return np.asarray(sensors[:count], dtype=np.float64)


def branch_from_alpha(alpha_mult: torch.Tensor, sensor_states: torch.Tensor, cfg: TumorConfig) -> torch.Tensor:
    terminal = alpha_mult.unsqueeze(-1) * sensor_states.sum(dim=-1).unsqueeze(0)
    return terminal / cfg.value_scale


def terminal_cost(alpha_mult: torch.Tensor, N: torch.Tensor, cfg: TumorConfig) -> torch.Tensor:
    return alpha_mult * cfg.alpha * N.sum(dim=-1)


def load_base_trajectory(cfg: TumorConfig) -> tuple[np.ndarray, np.ndarray]:
    direct = ROOT / "paper_runs" / "direct_openloop_cost_beta01_n800_refined" / "scale_1_direct_solution.npz"
    if direct.exists():
        data = np.load(direct)
        return np.asarray(data["t"], dtype=np.float64), np.asarray(data["N"], dtype=np.float64)
    t = np.linspace(0.0, cfg.T, 401)
    N = np.empty((len(t), cfg.m), dtype=np.float64)
    N[0] = cfg.n0
    p = np_params(cfg)
    dt = t[1] - t[0]
    for k in range(len(t) - 1):
        u = 1.5
        G = np.log1p(N[k].mean())
        N[k + 1] = np.maximum(N[k] + dt * (p["r"] - p["phi"] * u - p["M"] * G) * N[k], 1e-8)
    return t, N


def sample_batch(
    base_t: np.ndarray,
    base_N: np.ndarray,
    train_alphas: torch.Tensor,
    sensor_states: torch.Tensor,
    cfg: TumorConfig,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = np.random.randint(0, len(base_t), size=batch)
    t_np = np.clip(base_t[idx] + np.random.normal(0.0, 0.25, size=batch), 0.0, cfg.T)
    N_np = base_N[idx] * np.exp(np.random.normal(0.0, 0.18, size=(batch, cfg.m)))
    broad = np.random.rand(batch) < 0.20
    if broad.any():
        N_np[broad] = np.random.lognormal(mean=np.log(6.0), sigma=0.8, size=(broad.sum(), cfg.m))
    N_np = np.clip(N_np, 1e-6, cfg.state_scale)
    alpha_idx = torch.randint(0, train_alphas.numel(), (batch,), device=device)
    alpha_mult = train_alphas[alpha_idx]
    branch = branch_from_alpha(alpha_mult, sensor_states, cfg)
    return (
        branch.to(device=device, dtype=dtype),
        torch.tensor(t_np, device=device, dtype=dtype),
        torch.tensor(N_np, device=device, dtype=dtype),
        alpha_mult.to(device=device, dtype=dtype),
    )


def value_and_grad(
    model: PDeepONet, branch: torch.Tensor, t: torch.Tensor, N: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t_req = t.detach().clone().requires_grad_(True)
    N_req = N.detach().clone().requires_grad_(True)
    V = model(branch, t_req, N_req)
    grad_t, grad_N = torch.autograd.grad(V.sum(), (t_req, N_req), create_graph=True)
    return V, grad_t, grad_N


def policy_from_model(
    model: PDeepONet,
    branch: torch.Tensor,
    t: torch.Tensor,
    N: torch.Tensor,
    params: dict[str, torch.Tensor],
    cfg: TumorConfig,
    temp: float,
) -> torch.Tensor:
    with torch.enable_grad():
        _, _, V_N = value_and_grad(model, branch, t, N)
    psi = cfg.gamma - (params["phi"].unsqueeze(0) * N * V_N).sum(dim=-1, keepdim=True)
    u_bang = cfg.umax * torch.sigmoid(-psi / max(temp, 1e-6))
    u_sing = singular_control(N, params, cfg)
    q = torch.sigmoid((0.1 - torch.abs(psi)) / 0.03)
    return torch.clamp(q * u_sing + (1.0 - q) * u_bang, 0.0, cfg.umax).detach()


def constant_policy(value: float) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    def _policy(branch: torch.Tensor, t: torch.Tensor, N: torch.Tensor) -> torch.Tensor:
        return torch.full((N.shape[0], 1), float(value), device=N.device, dtype=N.dtype)

    return _policy


def model_policy(
    model: PDeepONet,
    params: dict[str, torch.Tensor],
    cfg: TumorConfig,
    temp: float,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    def _policy(branch: torch.Tensor, t: torch.Tensor, N: torch.Tensor) -> torch.Tensor:
        return policy_from_model(model, branch, t, N, params, cfg, temp)

    return _policy


def pde_loss(
    model: PDeepONet,
    branch: torch.Tensor,
    t: torch.Tensor,
    N: torch.Tensor,
    alpha_mult: torch.Tensor,
    current_policy: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    params: dict[str, torch.Tensor],
    cfg: TumorConfig,
    sensor_states: torch.Tensor,
    terminal_weight: float,
    terminal_batch: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    V, V_t, V_N = value_and_grad(model, branch, t, N)
    u = current_policy(branch, t, N)
    f = dynamics(N, u, params)
    running = (params["beta"].unsqueeze(0) * N).sum(dim=-1) + cfg.gamma * u.squeeze(-1)
    residual = V_t + running + (V_N * f).sum(dim=-1)

    idx = torch.randint(0, N.shape[0], (terminal_batch,), device=N.device)
    NT = N[idx].detach()
    alpha_T = alpha_mult[idx].detach()
    branch_T = branch_from_alpha(alpha_T, sensor_states, cfg)
    tT = torch.full((terminal_batch,), cfg.T, device=N.device, dtype=N.dtype)
    VT = model(branch_T, tT, NT)
    target = terminal_cost(alpha_T, NT, cfg)
    terminal = VT - target
    loss = residual.square().mean() + terminal_weight * terminal.square().mean()
    return loss, {
        "loss": float(loss.detach().cpu()),
        "pde_residual": float(residual.square().mean().detach().cpu()),
        "terminal_loss": float(terminal.square().mean().detach().cpu()),
        "u_mean": float(u.mean().detach().cpu()),
        "V_mean": float(V.mean().detach().cpu()),
    }


def train_policy_iteration(args: argparse.Namespace) -> tuple[PDeepONet, list[dict[str, object]], TumorConfig, torch.Tensor]:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float64 if args.float64 else torch.float32
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = TumorConfig()
    params = build_params(cfg, device, dtype)
    sensor_np = make_sensor_states(cfg, args.sensors, args.seed)
    sensor_states = torch.tensor(sensor_np, device=device, dtype=dtype)
    train_alphas = torch.tensor(parse_float_list(args.train_alpha_multipliers), device=device, dtype=dtype)
    base_t, base_N = load_base_trajectory(cfg)

    model = PDeepONet(cfg, args.sensors, args.width, args.branch_depth, args.trunk_depth).to(device=device, dtype=dtype)
    history: list[dict[str, object]] = []
    current_policy: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] = constant_policy(args.initial_u)

    for outer in range(1, args.outer + 1):
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for step in range(1, args.steps_per_outer + 1):
            branch, t, N, alpha_mult = sample_batch(base_t, base_N, train_alphas, sensor_states, cfg, args.batch, device, dtype)
            loss, metrics = pde_loss(
                model,
                branch,
                t,
                N,
                alpha_mult,
                current_policy,
                params,
                cfg,
                sensor_states,
                args.terminal_weight,
                min(args.terminal_batch, args.batch),
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            if step == 1 or step % args.log_every == 0 or step == args.steps_per_outer:
                row = {"outer": outer, "step": step, **metrics}
                history.append(row)
                print(
                    f"[PI-DeepONet outer={outer}] step={step} loss={metrics['loss']:.4g} "
                    f"pde={metrics['pde_residual']:.4g} term={metrics['terminal_loss']:.4g} u={metrics['u_mean']:.3f}",
                    flush=True,
                )
        fraction = outer / max(args.outer - 1, 1)
        policy_temp = args.policy_temp_start * (args.policy_temp / args.policy_temp_start) ** fraction
        current_policy = model_policy(model, params, cfg, policy_temp)
    return model, history, cfg, sensor_states


def evaluate_model(
    model: PDeepONet,
    cfg: TumorConfig,
    sensor_states: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    p_np = np_params(cfg)
    t = np.linspace(0.0, cfg.T, args.n_eval + 1)
    dt = cfg.T / args.n_eval
    N = np.empty((args.n_eval + 1, cfg.m), dtype=np.float64)
    u = np.empty(args.n_eval + 1, dtype=np.float64)
    N[0] = cfg.n0

    params = build_params(cfg, device, dtype)
    alpha = torch.ones(1, device=device, dtype=dtype)

    def u_of(t_val: float, N_val: np.ndarray) -> float:
        branch = branch_from_alpha(alpha, sensor_states, cfg)
        tt = torch.tensor([t_val], device=device, dtype=dtype)
        NN = torch.tensor(N_val[None, :], device=device, dtype=dtype)
        return float(policy_from_model(model, branch, tt, NN, params, cfg, args.policy_temp).detach().cpu().numpy().reshape(-1)[0])

    def dyn(x: np.ndarray, u_val: float) -> np.ndarray:
        x_pos = np.maximum(x, 1e-12)
        G = np.log1p(x_pos.mean())
        return (p_np["r"] - p_np["phi"] * u_val - p_np["M"] * G) * x_pos

    for k in range(args.n_eval):
        uk = u_of(t[k], N[k])
        u[k] = uk
        x = N[k]
        k1 = dyn(x, uk)
        k2 = dyn(np.maximum(x + 0.5 * dt * k1, 1e-12), uk)
        k3 = dyn(np.maximum(x + 0.5 * dt * k2, 1e-12), uk)
        k4 = dyn(np.maximum(x + dt * k3, 1e-12), uk)
        N[k + 1] = np.maximum(x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0, 1e-12)
    u[-1] = u_of(t[-1], N[-1])
    branch0 = branch_from_alpha(alpha, sensor_states, cfg)
    t0 = torch.zeros(1, device=device, dtype=dtype)
    N0 = torch.full((1, cfg.m), cfg.n0, device=device, dtype=dtype)
    native = float(model(branch0, t0, N0).detach().cpu().numpy().reshape(-1)[0])
    return t, N, u, native


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=str(ROOT / "paper_runs" / "official_related_work_adaptations"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--float64", action="store_true")
    parser.add_argument("--outer", type=int, default=4)
    parser.add_argument("--steps_per_outer", type=int, default=700)
    parser.add_argument("--batch", type=int, default=384)
    parser.add_argument("--terminal_batch", type=int, default=192)
    parser.add_argument("--sensors", type=int, default=32)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--branch_depth", type=int, default=2)
    parser.add_argument("--trunk_depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=50.0)
    parser.add_argument("--terminal_weight", type=float, default=2.0)
    parser.add_argument("--initial_u", type=float, default=1.5)
    parser.add_argument("--policy_temp", type=float, default=0.5)
    parser.add_argument("--policy_temp_start", type=float, default=2.0)
    parser.add_argument("--train_alpha_multipliers", default="0.8,1.0,1.2")
    parser.add_argument("--n_eval", type=int, default=400)
    parser.add_argument("--log_every", type=int, default=100)
    args = parser.parse_args()

    model, history, cfg, sensor_states = train_policy_iteration(args)
    t, N, u, native = evaluate_model(model, cfg, sensor_states, args)
    ref = eval_rk4(u)
    common = eval_common_grid(u)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "pi_deeponet_tumor_paper"
    np.savez(
        prefix.with_suffix(".npz"),
        t=t,
        N=N,
        u=u,
        native_estimate=native,
        history=np.array(history, dtype=object),
        train_alpha_multipliers=np.array(parse_float_list(args.train_alpha_multipliers)),
        policy_temp=args.policy_temp,
        policy_temp_start=args.policy_temp_start,
        m_suppression=cfg.m_suppression,
    )
    if history:
        write_csv(prefix.with_name(prefix.name + "_history.csv"), history, list(history[0].keys()))
    rows = [
        {
            "method": "PI-DeepONet [5] paper-algorithm tumor adaptation",
            "native_estimate": native,
            "realized_J": ref["J_ref_25600"],
            "J_gap_to_direct": "",
            "relative_gap_percent": "",
            "pmp_kkt_gap": common["common_gap"],
            "u_min": float(u.min()),
            "u_max": float(u.max()),
            "u_mean": float(u.mean()),
            "final_mean_N": ref["final_mean_N_ref"],
            "source": str(prefix.with_suffix(".npz").relative_to(ROOT)),
            "note": f"paper-algorithm PI-DeepONet implementation; M={cfg.m_suppression:g}; no author code release found",
        }
    ]
    write_csv(prefix.with_name(prefix.name + "_summary.csv"), rows, list(rows[0].keys()))
    print(prefix.with_suffix(".npz"))
    print(rows[0])


if __name__ == "__main__":
    main()
