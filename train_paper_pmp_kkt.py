import argparse
import csv
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_hidden(text: str) -> Tuple[int, ...]:
    if not text.strip():
        return ()
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def make_vector(value: float, m: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.full((m,), float(value), device=device, dtype=dtype)


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


def build_params(cfg: ProblemConfig, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    x = torch.linspace(0.0, 1.0, cfg.m, device=device, dtype=dtype)
    return {
        "x": x,
        "r": 2.0 / (1.0 + 3.0 * x.pow(4)),
        "phi": 1.0 / (1.0 + x.pow(2)),
        "M": torch.full((cfg.m,), cfg.m_suppression, device=device, dtype=dtype),
        "beta": make_vector(cfg.beta, cfg.m, device, dtype),
        "alpha": make_vector(cfg.alpha, cfg.m, device, dtype),
        "N0": make_vector(cfg.n0, cfg.m, device, dtype),
        "gamma": torch.tensor(float(cfg.gamma), device=device, dtype=dtype),
        "umax": torch.tensor(float(cfg.umax), device=device, dtype=dtype),
    }


def time_features(t: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            t,
            t.pow(2),
            torch.sin(2.0 * math.pi * t),
            torch.cos(2.0 * math.pi * t),
            torch.sin(4.0 * math.pi * t),
            torch.cos(4.0 * math.pi * t),
        ],
        dim=-1,
    )


class TimeMLP(nn.Module):
    def __init__(self, hidden: Tuple[int, ...], umax: float, init_u: float):
        super().__init__()
        layers = []
        prev = 6
        for width in hidden:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.Tanh())
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.umax = float(umax)
        self.init_u = float(init_u)
        self.reset_last_layer()

    def reset_last_layer(self) -> None:
        linears = [m for m in self.modules() if isinstance(m, nn.Linear)]
        for layer in linears[:-1]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        last = linears[-1]
        nn.init.normal_(last.weight, mean=0.0, std=1e-3)
        init = min(max(self.init_u / self.umax, 1e-4), 1.0 - 1e-4)
        with torch.no_grad():
            last.bias.fill_(math.log(init / (1.0 - init)))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        logits = self.net(time_features(t)).squeeze(-1)
        return self.umax * torch.sigmoid(logits)


class TimeTransformer(nn.Module):
    def __init__(self, d_model: int, heads: int, layers: int, umax: float, init_u: float):
        super().__init__()
        self.input = nn.Linear(6, d_model)
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
        self.umax = float(umax)
        self.init_u = float(init_u)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.zeros_(self.input.bias)
        nn.init.normal_(self.output.weight, mean=0.0, std=1e-3)
        init = min(max(self.init_u / self.umax, 1e-4), 1.0 - 1e-4)
        with torch.no_grad():
            self.output.bias.fill_(math.log(init / (1.0 - init)))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        h = self.input(time_features(t)).unsqueeze(0)
        h = self.encoder(h).squeeze(0)
        logits = self.output(h).squeeze(-1)
        return self.umax * torch.sigmoid(logits)


class TimeCNN(nn.Module):
    """Local temporal baseline with the same pointwise time features.

    The model applies a short stack of same-padded one-dimensional
    convolutions along the sampled treatment horizon.  Unlike the Transformer,
    its receptive field is fixed by the kernel width and depth.
    """

    def __init__(
        self,
        channels: int,
        layers: int,
        kernel_size: int,
        umax: float,
        init_u: float,
    ):
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if layers <= 0:
            raise ValueError("layers must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        blocks = []
        in_channels = 6
        padding = kernel_size // 2
        for _ in range(layers):
            blocks.append(
                nn.Conv1d(
                    in_channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )
            blocks.append(nn.GELU())
            in_channels = channels
        self.features = nn.Sequential(*blocks)
        self.output = nn.Conv1d(channels, 1, kernel_size=1)
        self.umax = float(umax)
        self.init_u = float(init_u)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.features.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.output.weight, mean=0.0, std=1e-3)
        init = min(max(self.init_u / self.umax, 1e-4), 1.0 - 1e-4)
        with torch.no_grad():
            self.output.bias.fill_(math.log(init / (1.0 - init)))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        features = time_features(t).transpose(0, 1).unsqueeze(0)
        logits = self.output(self.features(features)).squeeze(0).squeeze(0)
        return self.umax * torch.sigmoid(logits)


class ParamControl(nn.Module):
    def __init__(self, n_points: int, umax: float, init_u: float):
        super().__init__()
        init = min(max(init_u / umax, 1e-4), 1.0 - 1e-4)
        logit = math.log(init / (1.0 - init))
        self.logits = nn.Parameter(torch.full((n_points,), logit))
        self.umax = float(umax)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.umax * torch.sigmoid(self.logits)


def tumor_g(N: torch.Tensor) -> torch.Tensor:
    return torch.log1p(N.mean(dim=-1))


def dynamics(N: torch.Tensor, u: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    G = tumor_g(N)
    return (params["r"] - params["phi"] * u - params["M"] * G) * N


def simulate_state(u: torch.Tensor, cfg: ProblemConfig, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    dt = cfg.T / cfg.n
    states = [params["N0"]]
    N = params["N0"]
    for k in range(cfg.n):
        N = N + dt * dynamics(N, u[k], params)
        N = torch.clamp(N, min=1e-8)
        states.append(N)
    return torch.stack(states, dim=0)


def dH_dN(
    N: torch.Tensor,
    lam: torch.Tensor,
    u: torch.Tensor,
    params: Dict[str, torch.Tensor],
) -> torch.Tensor:
    G = tumor_g(N)
    mean_N = N.mean()
    dG = (1.0 / N.numel()) / (1.0 + mean_N)
    a = params["r"] - params["phi"] * u - params["M"] * G
    coupling = (lam * params["M"] * N).sum()
    return params["beta"] + lam * a - dG * coupling


def compute_costate(N: torch.Tensor, u: torch.Tensor, cfg: ProblemConfig, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    dt = cfg.T / cfg.n
    lams = [None] * (cfg.n + 1)
    lam = params["alpha"]
    lams[cfg.n] = lam
    for k in range(cfg.n - 1, -1, -1):
        lam = lam + dt * dH_dN(N[k], lam, u[k], params)
        lams[k] = lam
    return torch.stack(lams, dim=0)


def singular_control(N: torch.Tensor, params: Dict[str, torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
    G = tumor_g(N)
    numerator = (params["beta"] * (params["r"].unsqueeze(0) - G.unsqueeze(-1) * params["M"].unsqueeze(0)) * N).sum(dim=-1)
    denominator = (params["beta"] * params["phi"] * N).sum(dim=-1).clamp_min(eps)
    return numerator / denominator


def objective_value(N: torch.Tensor, u: torch.Tensor, cfg: ProblemConfig, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    dt = cfg.T / cfg.n
    running = (N[:-1] * params["beta"].unsqueeze(0)).sum(dim=-1) + params["gamma"] * u[:-1]
    integral = dt * running.sum()
    terminal = (params["alpha"] * N[-1]).sum()
    return terminal + integral


def pmp_kkt_loss(
    u: torch.Tensor,
    cfg: ProblemConfig,
    params: Dict[str, torch.Tensor],
    singular_eps: float,
    singular_tau: float,
    detach_gate: bool = False,
) -> Dict[str, torch.Tensor]:
    N = simulate_state(u, cfg, params)
    lam = compute_costate(N, u, cfg, params)
    # For the forward-Euler transcription, u_k is paired with lambda_{k+1}.
    # The terminal value is retained only for plotting the continuous endpoint.
    psi_interval = params["gamma"] - (lam[1:] * params["phi"].unsqueeze(0) * N[:-1]).sum(dim=-1)
    psi_terminal = params["gamma"] - (lam[-1] * params["phi"] * N[-1]).sum()
    psi = torch.cat([psi_interval, psi_terminal.unsqueeze(0)])
    u_sing = singular_control(N, params)
    admissible = ((u_sing >= 0.0) & (u_sing <= params["umax"])).to(u.dtype)
    q = torch.sigmoid((singular_eps - psi.abs()) / singular_tau) * admissible

    l_sing = (u[:-1] - u_sing[:-1]).pow(2)
    l_ns = (
        torch.relu(psi_interval) * u[:-1]
        + torch.relu(-psi_interval) * (params["umax"] - u[:-1])
    ).pow(2)
    q_for_loss = q[:-1].detach() if detach_gate else q[:-1]
    singular_component = (q_for_loss * l_sing).mean()
    nonsingular_component = ((1.0 - q_for_loss) * l_ns).mean()
    opt_gap = singular_component + nonsingular_component
    smooth = (u[1:] - u[:-1]).pow(2).mean()
    J = objective_value(N, u, cfg, params)

    return {
        "opt_gap": opt_gap,
        "singular_loss": l_sing.mean(),
        "nonsingular_loss": l_ns.mean(),
        "singular_component": singular_component,
        "nonsingular_component": nonsingular_component,
        "smooth": smooth,
        "objective": J,
        "N": N,
        "lambda": lam,
        "psi": psi,
        "u_sing": u_sing,
        "q": q,
    }


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(rows)


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
    t = torch.linspace(0.0, 1.0, cfg.n + 1, device=device, dtype=dtype)

    if args.model == "mlp":
        model = TimeMLP(parse_hidden(args.hidden), args.umax, args.init_u).to(device=device, dtype=dtype)
    elif args.model == "transformer":
        model = TimeTransformer(args.d_model, args.heads, args.layers, args.umax, args.init_u).to(device=device, dtype=dtype)
    elif args.model == "cnn":
        model = TimeCNN(
            args.cnn_channels,
            args.cnn_layers,
            args.cnn_kernel_size,
            args.umax,
            args.init_u,
        ).to(device=device, dtype=dtype)
    elif args.model == "param":
        model = ParamControl(cfg.n + 1, args.umax, args.init_u).to(device=device, dtype=dtype)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=args.lr_patience)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_metric = float("inf")
    best_state = None
    history = [[
        "epoch",
        "loss",
        "opt_gap",
        "objective",
        "smooth",
        "u_min",
        "u_max",
        "u_mean",
        "final_mean_N",
        "psi_mean_abs",
        "q_mean",
        "singular_component",
        "nonsingular_component",
    ]]

    for epoch in range(1, args.epochs + 1):
        opt.zero_grad(set_to_none=True)
        u = model(t)
        pack = pmp_kkt_loss(u, cfg, params, args.singular_eps, args.singular_tau, args.detach_gate)
        loss = pack["opt_gap"] + args.smooth_weight * pack["smooth"] + args.objective_weight * pack["objective"] / (cfg.n + 1)
        loss.backward()

        with torch.no_grad():
            N = pack["N"]
            row = [
                epoch,
                float(loss.detach().cpu()),
                float(pack["opt_gap"].detach().cpu()),
                float(pack["objective"].detach().cpu()),
                float(pack["smooth"].detach().cpu()),
                float(u.min().detach().cpu()),
                float(u.max().detach().cpu()),
                float(u.mean().detach().cpu()),
                float(N[-1].mean().detach().cpu()),
                float(pack["psi"].abs().mean().detach().cpu()),
                float(pack["q"].mean().detach().cpu()),
                float(pack["singular_component"].detach().cpu()),
                float(pack["nonsingular_component"].detach().cpu()),
            ]
            history.append(row)
            metric = row[1]
            if metric < best_metric:
                best_metric = metric
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        scheduler.step(row[1])
        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.epochs:
            print(
                f"[{epoch:04d}] loss={row[1]:.6g} gap={row[2]:.6g} J={row[3]:.6g} "
                f"u=({row[5]:.3f},{row[6]:.3f},{row[7]:.3f}) finalN={row[8]:.6g} q={row[10]:.3f}"
            )

    if best_state is not None:
        torch.save(
            {
                "model_state": best_state,
                "args": vars(args),
                "problem": cfg.__dict__,
                "best_metric": best_metric,
            },
            out_dir / "best_pmp_kkt.pt",
        )

    model.load_state_dict(best_state)
    with torch.no_grad():
        u = model(t)
        pack = pmp_kkt_loss(u, cfg, params, args.singular_eps, args.singular_tau, args.detach_gate)
        N = pack["N"]
        lam = pack["lambda"]
        rows = [["t", "u", "u_sing", "psi", "q", "mean_N"]]
        for i in range(cfg.n + 1):
            rows.append(
                [
                    float((args.T * t[i]).detach().cpu()),
                    float(u[i].detach().cpu()),
                    float(pack["u_sing"][i].detach().cpu()),
                    float(pack["psi"][i].detach().cpu()),
                    float(pack["q"][i].detach().cpu()),
                    float(N[i].mean().detach().cpu()),
                ]
            )
        write_csv(out_dir / "trajectory.csv", rows)
        write_csv(out_dir / "history.csv", history)
        np.savez(
            out_dir / "solution.npz",
            t=(args.T * t).detach().cpu().numpy(),
            u=u.detach().cpu().numpy(),
            N=N.detach().cpu().numpy(),
            lam=lam.detach().cpu().numpy(),
            psi=pack["psi"].detach().cpu().numpy(),
            q=pack["q"].detach().cpu().numpy(),
            u_sing=pack["u_sing"].detach().cpu().numpy(),
        )

    print(f"Saved best checkpoint and CSV outputs to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the paper-style PMP/KKT optimality-gap control.")
    parser.add_argument("--model", choices=["mlp", "transformer", "cnn", "param"], default="mlp")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--m", type=int, default=21)
    parser.add_argument("--umax", type=float, default=3.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=20.0)
    parser.add_argument("--n0", type=float, default=10.0)
    parser.add_argument("--m_suppression", type=float, default=0.5)
    parser.add_argument("--hidden", type=str, default="128,128")
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--cnn_channels", type=int, default=96)
    parser.add_argument("--cnn_layers", type=int, default=3)
    parser.add_argument("--cnn_kernel_size", type=int, default=5)
    parser.add_argument("--init_u", type=float, default=1.5)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_patience", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--singular_eps", type=float, default=0.05)
    parser.add_argument("--singular_tau", type=float, default=0.02)
    parser.add_argument("--detach_gate", action="store_true")
    parser.add_argument("--smooth_weight", type=float, default=1e-4)
    parser.add_argument("--objective_weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--float64", action="store_true")
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--out_dir", type=str, default="paper_runs/pmp_kkt_mlp")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
