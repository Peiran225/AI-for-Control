import argparse
import csv
import math
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from train_openloop import ProblemConfig, build_params, parse_hidden, set_seed, time_features


class FeedbackMLP(nn.Module):
    def __init__(self, m: int, hidden: Tuple[int, ...], umax: float, state_scale: float, init_u: float):
        super().__init__()
        self.m = m
        self.umax = float(umax)
        self.state_log_scale = math.log1p(float(state_scale))
        input_dim = 6 + m + 3
        layers = []
        prev = input_dim
        for width in hidden:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.Tanh())
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.init_u = float(init_u)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        linears = [m for m in self.modules() if isinstance(m, nn.Linear)]
        for layer in linears[:-1]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        last = linears[-1]
        nn.init.normal_(last.weight, mean=0.0, std=1e-3)
        init = min(max(self.init_u / self.umax, 1e-4), 1.0 - 1e-4)
        with torch.no_grad():
            last.bias.fill_(math.log(init / (1.0 - init)))

    def forward(self, t: torch.Tensor, N: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(N.shape[0])
        tf = time_features(t)
        logN = torch.log1p(N) / self.state_log_scale
        mean = logN.mean(dim=-1, keepdim=True)
        std = logN.std(dim=-1, keepdim=True, unbiased=False)
        maxv = logN.max(dim=-1, keepdim=True).values
        x = torch.cat([tf, logN, mean, std, maxv], dim=-1)
        return self.umax * torch.sigmoid(self.net(x).squeeze(-1))


class FeedbackTransformer(nn.Module):
    def __init__(self, m: int, d_model: int, heads: int, layers: int, umax: float, state_scale: float, init_u: float):
        super().__init__()
        self.m = m
        self.umax = float(umax)
        self.state_log_scale = math.log1p(float(state_scale))
        x = torch.linspace(0.0, 1.0, m)
        self.register_buffer("trait_x", x)
        self.register_buffer("trait_r", 2.0 / (1.0 + 3.0 * x.pow(4)))
        self.register_buffer("trait_phi", 1.0 / (1.0 + x.pow(2)))
        token_dim = 6 + 1 + 3 + 3
        self.input = nn.Linear(token_dim, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.output = nn.Linear(d_model, 1)
        self.init_u = float(init_u)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.zeros_(self.input.bias)
        nn.init.zeros_(self.cls)
        nn.init.normal_(self.output.weight, mean=0.0, std=1e-3)
        init = min(max(self.init_u / self.umax, 1e-4), 1.0 - 1e-4)
        with torch.no_grad():
            self.output.bias.fill_(math.log(init / (1.0 - init)))

    def forward(self, t: torch.Tensor, N: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(N.shape[0])
        B = N.shape[0]
        tf = time_features(t).unsqueeze(1).expand(B, self.m, -1)
        logN = torch.log1p(N) / self.state_log_scale
        mean = logN.mean(dim=-1, keepdim=True).unsqueeze(1).expand(B, self.m, -1)
        std = logN.std(dim=-1, keepdim=True, unbiased=False).unsqueeze(1).expand(B, self.m, -1)
        maxv = logN.max(dim=-1, keepdim=True).values.unsqueeze(1).expand(B, self.m, -1)
        trait = torch.stack([self.trait_x, self.trait_r, self.trait_phi], dim=-1).unsqueeze(0).expand(B, -1, -1)
        tokens = torch.cat([tf, logN.unsqueeze(-1), trait, mean, std, maxv], dim=-1)
        h = self.input(tokens)
        cls = self.cls.expand(B, -1, -1)
        h = self.encoder(torch.cat([cls, h], dim=1))
        logits = self.output(h[:, 0]).squeeze(-1)
        return self.umax * torch.sigmoid(logits)


def build_feedback_model(cfg: ProblemConfig, args: argparse.Namespace) -> nn.Module:
    model_name = getattr(args, "model", "mlp")
    if model_name == "mlp":
        return FeedbackMLP(cfg.m, parse_hidden(args.hidden), args.umax, args.state_scale, args.init_u)
    if model_name == "transformer":
        return FeedbackTransformer(cfg.m, args.d_model, args.heads, args.layers, args.umax, args.state_scale, args.init_u)
    raise ValueError(f"Unsupported feedback model: {model_name}")


def tumor_g(N: torch.Tensor) -> torch.Tensor:
    return torch.log1p(N.mean(dim=-1))


def dynamics(N: torch.Tensor, u: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    G = tumor_g(N)
    return (params["r"].unsqueeze(0) - params["phi"].unsqueeze(0) * u.unsqueeze(-1) - params["M"].unsqueeze(0) * G.unsqueeze(-1)) * N


def sample_initial_states(
    batch_size: int,
    cfg: ProblemConfig,
    low: float,
    high: float,
    trait_noise: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    scale = low + (high - low) * torch.rand(batch_size, 1, device=device, dtype=dtype)
    if trait_noise > 0:
        noise = 1.0 + trait_noise * torch.randn(batch_size, cfg.m, device=device, dtype=dtype)
        noise = torch.clamp(noise, min=0.2)
    else:
        noise = torch.ones(batch_size, cfg.m, device=device, dtype=dtype)
    return torch.clamp(cfg.n0 * scale * noise, min=1e-6)


def simulate_feedback(
    model: nn.Module,
    N0: torch.Tensor,
    cfg: ProblemConfig,
    params: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    dt = cfg.T / cfg.n
    B = N0.shape[0]
    N = N0
    states = []
    controls = []
    for k in range(cfg.n + 1):
        t = torch.full((B,), k / cfg.n, device=N0.device, dtype=N0.dtype)
        u = model(t, N)
        states.append(N)
        controls.append(u)
        if k < cfg.n:
            N = N + dt * dynamics(N, u, params)
            N = torch.clamp(N, min=1e-8)
    return torch.stack(states, dim=1), torch.stack(controls, dim=1)


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
        lam = lam + dt * dH_dN(N[:, k], lam, u[:, k], params)
        lams[k] = lam
    return torch.stack(lams, dim=1)


def singular_control(N: torch.Tensor, params: Dict[str, torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
    G = tumor_g(N)
    numerator = (params["beta"] * (params["r"].view(1, 1, -1) - G.unsqueeze(-1) * params["M"].view(1, 1, -1)) * N).sum(dim=-1)
    denominator = (params["beta"] * params["phi"] * N).sum(dim=-1).clamp_min(eps)
    return numerator / denominator


def objective_value(N: torch.Tensor, u: torch.Tensor, cfg: ProblemConfig, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    dt = cfg.T / cfg.n
    running = (N * params["beta"].view(1, 1, -1)).sum(dim=-1) + params["gamma"] * u
    integral = dt * (0.5 * running[:, 0] + running[:, 1:-1].sum(dim=-1) + 0.5 * running[:, -1])
    terminal = (params["alpha"].view(1, -1) * N[:, -1]).sum(dim=-1)
    return terminal + integral


def pmp_kkt_loss(
    model: nn.Module,
    N0: torch.Tensor,
    cfg: ProblemConfig,
    params: Dict[str, torch.Tensor],
    singular_eps: float,
    singular_tau: float,
) -> Dict[str, torch.Tensor]:
    N, u = simulate_feedback(model, N0, cfg, params)
    lam = compute_costate(N, u, cfg, params)
    psi = params["gamma"] - (lam * params["phi"].view(1, 1, -1) * N).sum(dim=-1)
    u_sing = singular_control(N, params)
    admissible = ((u_sing >= 0.0) & (u_sing <= params["umax"])).to(u.dtype)
    q = torch.sigmoid((singular_eps - psi.abs()) / singular_tau) * admissible
    l_sing = (u - u_sing).pow(2)
    l_ns = (torch.relu(psi) * u + torch.relu(-psi) * (params["umax"] - u)).pow(2)
    opt_gap = (q * l_sing + (1.0 - q) * l_ns).mean()
    smooth = (u[:, 1:] - u[:, :-1]).pow(2).mean()
    J = objective_value(N, u, cfg, params).mean()
    return {
        "opt_gap": opt_gap,
        "smooth": smooth,
        "objective": J,
        "N": N,
        "u": u,
        "lambda": lam,
        "psi": psi,
        "u_sing": u_sing,
        "q": q,
    }


def make_eval_initials(cfg: ProblemConfig, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    scales = torch.tensor([0.5, 0.75, 1.0, 1.25, 1.5], device=device, dtype=dtype).view(-1, 1)
    return cfg.n0 * scales.expand(-1, cfg.m)


def write_trajectory(path: Path, pack: Dict[str, torch.Tensor], cfg: ProblemConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    N = pack["N"].detach().cpu().numpy()
    u = pack["u"].detach().cpu().numpy()
    psi = pack["psi"].detach().cpu().numpy()
    q = pack["q"].detach().cpu().numpy()
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample", "t", "u", "psi", "q", "mean_N"])
        for b in range(N.shape[0]):
            for k in range(N.shape[1]):
                writer.writerow([b, cfg.T * k / cfg.n, u[b, k], psi[b, k], q[b, k], N[b, k].mean()])


def evaluate(model, N0, cfg, params, args) -> Dict[str, float]:
    with torch.no_grad():
        pack = pmp_kkt_loss(model, N0, cfg, params, args.singular_eps, args.singular_tau)
        N = pack["N"]
        u = pack["u"]
        return {
            "gap": float(pack["opt_gap"].detach().cpu()),
            "objective": float(pack["objective"].detach().cpu()),
            "u_min": float(u.min().detach().cpu()),
            "u_max": float(u.max().detach().cpu()),
            "u_mean": float(u.mean().detach().cpu()),
            "final_mean_N": float(N[:, -1].mean().detach().cpu()),
            "psi_mean_abs": float(pack["psi"].abs().mean().detach().cpu()),
            "q_mean": float(pack["q"].mean().detach().cpu()),
        }


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float64 if args.float64 else torch.float32
    cfg = ProblemConfig(
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
    params = build_params(cfg, device, dtype)
    model = build_feedback_model(cfg, args).to(device=device, dtype=dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=args.lr_patience, min_lr=args.min_lr)
    val_N0 = make_eval_initials(cfg, device, dtype)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = [["epoch", "train_loss", "train_gap", "val_gap", "val_objective", "u_min", "u_max", "u_mean", "final_mean_N", "q_mean"]]
    best_state = None
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        N0 = sample_initial_states(args.batch_size, cfg, args.train_low, args.train_high, args.trait_noise, device, dtype)
        opt.zero_grad(set_to_none=True)
        pack = pmp_kkt_loss(model, N0, cfg, params, args.singular_eps, args.singular_tau)
        loss = pack["opt_gap"] + args.smooth_weight * pack["smooth"] + args.objective_weight * pack["objective"] / (cfg.n + 1)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            metrics = evaluate(model, val_N0, cfg, params, args)
            train_gap = float(pack["opt_gap"].detach().cpu())
            row = [
                epoch,
                float(loss.detach().cpu()),
                train_gap,
                metrics["gap"],
                metrics["objective"],
                metrics["u_min"],
                metrics["u_max"],
                metrics["u_mean"],
                metrics["final_mean_N"],
                metrics["q_mean"],
            ]
            history.append(row)
            scheduler.step(metrics["gap"])
            if metrics["gap"] < best_val:
                best_val = metrics["gap"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(
                f"[{epoch:04d}] train_gap={train_gap:.5g} val_gap={metrics['gap']:.5g} "
                f"J={metrics['objective']:.3f} u=({metrics['u_min']:.3f},{metrics['u_max']:.3f},{metrics['u_mean']:.3f}) "
                f"finalN={metrics['final_mean_N']:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    eval_N0 = make_eval_initials(cfg, device, dtype)
    eval_pack = pmp_kkt_loss(model, eval_N0, cfg, params, args.singular_eps, args.singular_tau)
    torch.save(
        {
            "model_state": best_state,
            "args": vars(args),
            "problem": cfg.__dict__,
            "best_val_gap": best_val,
        },
        out_dir / "best_feedback_pmp_kkt.pt",
    )
    with (out_dir / "history.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(history)
    write_trajectory(out_dir / "eval_trajectories.csv", eval_pack, cfg)
    np.savez(
        out_dir / "eval_solution.npz",
        N=eval_pack["N"].detach().cpu().numpy(),
        u=eval_pack["u"].detach().cpu().numpy(),
        psi=eval_pack["psi"].detach().cpu().numpy(),
        q=eval_pack["q"].detach().cpu().numpy(),
        u_sing=eval_pack["u_sing"].detach().cpu().numpy(),
    )
    print(f"Saved feedback prototype outputs to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a feedback policy u_theta(t, N) with PMP/KKT gap.")
    parser.add_argument("--model", choices=["mlp", "transformer"], default="transformer")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--m", type=int, default=21)
    parser.add_argument("--umax", type=float, default=3.0)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=20.0)
    parser.add_argument("--n0", type=float, default=10.0)
    parser.add_argument("--m_suppression", type=float, default=0.5)
    parser.add_argument("--hidden", type=str, default="128,128")
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--state_scale", type=float, default=15.0)
    parser.add_argument("--init_u", type=float, default=1.5)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--train_low", type=float, default=0.5)
    parser.add_argument("--train_high", type=float, default=1.5)
    parser.add_argument("--trait_noise", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_patience", type=int, default=20)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--singular_eps", type=float, default=0.05)
    parser.add_argument("--singular_tau", type=float, default=0.02)
    parser.add_argument("--smooth_weight", type=float, default=1e-4)
    parser.add_argument("--objective_weight", type=float, default=0.0)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--float64", action="store_true")
    parser.add_argument("--out_dir", type=str, default="runs/feedback")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
