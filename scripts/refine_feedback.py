import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from train_feedback import (
    build_feedback_model,
    evaluate,
    pmp_kkt_loss,
    sample_initial_states,
    write_trajectory,
)
from train_openloop import ProblemConfig, build_params, parse_hidden, set_seed


def parse_scales(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def make_eval_initials_from_scales(scales, cfg, device, dtype):
    s = torch.tensor(scales, device=device, dtype=dtype).view(-1, 1)
    return cfg.n0 * s.expand(-1, cfg.m)


def train(args):
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
    if not args.init_feedback:
        raise ValueError("--init_feedback is required for refinement")
    if args.init_feedback:
        ckpt = torch.load(args.init_feedback, map_location="cpu")
        ckpt_cfg = ProblemConfig(**ckpt["problem"])
        if ckpt_cfg.n != cfg.n or ckpt_cfg.m != cfg.m:
            raise ValueError(f"Feedback checkpoint grid mismatch: got n={ckpt_cfg.n}, m={ckpt_cfg.m}")
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded initial feedback checkpoint from {args.init_feedback}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=args.lr_patience, min_lr=args.min_lr)
    history = [["epoch", "train_loss", "train_gap", "val_gap", "val_objective", "u_min", "u_max", "u_mean", "final_mean_N", "q_mean"]]
    best_state = None
    best_val = float("inf")
    eval_scales = parse_scales(args.eval_scales)
    val_N0 = make_eval_initials_from_scales(eval_scales, cfg, device, dtype)
    if args.init_feedback:
        model.eval()
        metrics = evaluate(model, val_N0, cfg, params, args)
        best_val = metrics["gap"]
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append([
            0,
            0.0,
            metrics["gap"],
            metrics["gap"],
            metrics["objective"],
            metrics["u_min"],
            metrics["u_max"],
            metrics["u_mean"],
            metrics["final_mean_N"],
            metrics["q_mean"],
        ])
        print(
            f"[init] val_gap={metrics['gap']:.5g} "
            f"J={metrics['objective']:.3f} u=({metrics['u_min']:.3f},{metrics['u_max']:.3f},{metrics['u_mean']:.3f}) "
            f"finalN={metrics['final_mean_N']:.4f}"
        )

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.curriculum_epochs > 0 and epoch <= args.curriculum_epochs:
            train_low = args.curriculum_train_low
            train_high = args.curriculum_train_high
            trait_noise = args.curriculum_trait_noise
        else:
            train_low = args.train_low
            train_high = args.train_high
            trait_noise = args.trait_noise
        N0 = sample_initial_states(args.batch_size, cfg, train_low, train_high, trait_noise, device, dtype)
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
    eval_N0 = make_eval_initials_from_scales(eval_scales, cfg, device, dtype)
    eval_pack = pmp_kkt_loss(model, eval_N0, cfg, params, args.singular_eps, args.singular_tau)
    torch.save(
        {
            "model_state": best_state,
            "args": vars(args),
            "problem": cfg.__dict__,
            "eval_scales": eval_scales,
            "best_val_gap": best_val,
        },
        out_dir / "best_feedback_pmp_kkt.pt",
    )
    with (out_dir / "history.csv").open("w", newline="") as f:
        csv.writer(f).writerows(history)
    write_trajectory(out_dir / "eval_trajectories.csv", eval_pack, cfg)
    np.savez(
        out_dir / "eval_solution.npz",
        N=eval_pack["N"].detach().cpu().numpy(),
        u=eval_pack["u"].detach().cpu().numpy(),
        psi=eval_pack["psi"].detach().cpu().numpy(),
        q=eval_pack["q"].detach().cpu().numpy(),
        u_sing=eval_pack["u_sing"].detach().cpu().numpy(),
    )
    print(f"Saved refined feedback outputs to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Refine a feedback policy checkpoint with PMP/KKT gap.")
    parser.add_argument("--init_feedback", type=str, default="")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--m", type=int, default=21)
    parser.add_argument("--umax", type=float, default=3.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=20.0)
    parser.add_argument("--n0", type=float, default=10.0)
    parser.add_argument("--m_suppression", type=float, default=0.5)
    parser.add_argument("--model", choices=["mlp", "transformer"], default="transformer")
    parser.add_argument("--hidden", type=str, default="128,128")
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--state_scale", type=float, default=15.0)
    parser.add_argument("--init_u", type=float, default=1.5)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--train_low", type=float, default=0.5)
    parser.add_argument("--train_high", type=float, default=1.5)
    parser.add_argument("--trait_noise", type=float, default=0.05)
    parser.add_argument("--eval_scales", type=str, default="0.5,0.75,1.0,1.25,1.5")
    parser.add_argument("--curriculum_epochs", type=int, default=0)
    parser.add_argument("--curriculum_train_low", type=float, default=0.5)
    parser.add_argument("--curriculum_train_high", type=float, default=1.5)
    parser.add_argument("--curriculum_trait_noise", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_patience", type=int, default=20)
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--singular_eps", type=float, default=0.05)
    parser.add_argument("--singular_tau", type=float, default=0.02)
    parser.add_argument("--smooth_weight", type=float, default=1e-3)
    parser.add_argument("--objective_weight", type=float, default=0.0)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--float64", action="store_true")
    parser.add_argument("--out_dir", type=str, default="runs/refined_feedback")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
