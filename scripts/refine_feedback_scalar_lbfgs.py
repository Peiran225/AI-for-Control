#!/usr/bin/env python3
"""Deterministic last-layer feedback refinement with scalar PMP residuals.

This is the second stage of the direct-initialized feedback protocol.  The
incoming checkpoint may have used direct trajectories only for initialization.
This script removes that supervision entirely: it freezes the time branch and
the hidden state layers, then refines only the final state-correction layer
using H_u, dH_u/dt, and d2H_u/dt2 on a fixed singular interval.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.train_feedback_section5 import (  # noqa: E402
    NestedFeedbackTransformer,
    fixed_interval_candidate_mask,
    make_fixed_directions,
    scalar_metrics,
    section5_loss,
    test_states_from_directions,
)
from train_paper_pmp_kkt import ProblemConfig, build_params, parse_hidden  # noqa: E402


def load_feedback_checkpoint(
    path: Path,
) -> tuple[NestedFeedbackTransformer, ProblemConfig, argparse.Namespace]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    argument_values = dict(checkpoint.get("args", {}))
    summary_path = path.parent / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for key, value in summary.get("protocol", {}).items():
            argument_values.setdefault(key, value)
    defaults = {
        "state_scale": 15.0,
        "state_hidden": "128,128",
        "d_model": 64,
        "heads": 4,
        "layers": 2,
        "init_u": 1.5,
        "correction_gain": 1.0,
        "state_feature_mode": "burden_composition",
        "center_state_correction": True,
        "action_temperature": 1.0,
        "action_scale": 1.0,
        "action_parameterization": "logit-temperature",
        "action_offset": 0.0,
        "option": "der",
        "loss_variant": "lc_live",
        "state_mode": "feedback",
        "training_integrator": "rk4",
    }
    for key, value in defaults.items():
        argument_values.setdefault(key, value)
    source_args = argparse.Namespace(**argument_values)
    cfg = ProblemConfig(**checkpoint["problem"])
    model = NestedFeedbackTransformer(
        cfg.m,
        cfg.umax,
        source_args.state_scale,
        parse_hidden(source_args.state_hidden),
        source_args.d_model,
        source_args.heads,
        source_args.layers,
        source_args.init_u,
        source_args.correction_gain,
        getattr(source_args, "state_feature_mode", "log_absolute"),
        bool(getattr(source_args, "center_state_correction", False)),
        float(getattr(source_args, "action_temperature", 1.0)),
        float(getattr(source_args, "action_scale", 1.0)),
        str(getattr(source_args, "action_parameterization", "logit-temperature")),
        float(getattr(source_args, "action_offset", 0.0)),
    ).double()
    if checkpoint.get("time_checkpoint"):
        model.load_time_checkpoint(Path(checkpoint["time_checkpoint"]))
        source_args.action_temperature = model.action_temperature
        source_args.action_scale = model.action_scale
        source_args.action_parameterization = model.action_parameterization
        source_args.action_offset = model.action_offset
    model.load_state_dict(checkpoint["model_state"])
    if "nominal_reference" in checkpoint:
        model.set_nominal_reference(
            checkpoint["nominal_reference"].to(dtype=torch.float64)
        )
    return model, cfg, source_args


def resistant_heavy_state(
    n0: float,
    m: int,
    radius: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    direction = torch.linspace(-1.0, 1.0, m, device=device, dtype=dtype)
    return n0 * (1.0 + radius * direction)


def anchored_states(
    count: int,
    seed: int,
    radius: float,
    cfg: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if count < 0:
        raise ValueError("random state count must be nonnegative")
    random_states = test_states_from_directions(
        make_fixed_directions(count, cfg.m, seed, dtype),
        radius,
        cfg,
        device,
        dtype,
    )
    nominal = torch.full(
        (1, cfg.m), cfg.n0, device=device, dtype=dtype
    )
    resistant = resistant_heavy_state(
        cfg.n0, cfg.m, radius, device=device, dtype=dtype
    ).unsqueeze(0)
    return torch.cat((nominal, resistant, random_states), dim=0)


def configure_scalar_loss(
    source: argparse.Namespace,
    cfg: Any,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> argparse.Namespace:
    values = vars(source).copy()
    values.update(
        {
            "option": "der",
            "loss_variant": "lc_live",
            "state_mode": "feedback",
            "training_integrator": "rk4",
            "singular_eps": float(values.get("singular_eps", 0.1)),
            "singular_tau": float(values.get("singular_tau", 0.03)),
            "dot_eps": float(values.get("dot_eps", 0.1)),
            "dot_tau": float(values.get("dot_tau", 0.03)),
            "b_min": float(values.get("b_min", 1.0e-8)),
            "gate_gradient_mode": "live",
            "w0": 1.0,
            "w1": 1.0,
            "w2": 1.0,
            "w_lc": 0.0,
            "psi_scale": 1.0,
            "dot_scale": 1.0,
            "ddot_scale": 1.0,
            "B_scale": 1.0,
            "singular_loss_weight": 1.0,
            "nonsingular_loss_weight": 0.0,
            "smooth_weight": 0.0,
            "smooth_second_weight": 0.0,
            "smooth_max_weight": 0.0,
            "full_gradient_weight": 0.0,
            "full_gradient_max_weight": 0.0,
        }
    )
    configured = argparse.Namespace(**values)
    configured._fixed_candidate_mask_tensor = fixed_interval_candidate_mask(
        cfg,
        integrator="rk4",
        start=args.interval_start,
        end=args.interval_end,
        device=device,
        dtype=dtype,
    )
    return configured


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = torch.float64
    checkpoint_path = args.checkpoint.expanduser().resolve()
    model, cfg, source_args = load_feedback_checkpoint(checkpoint_path)
    if getattr(source_args, "option", "der") != "der":
        raise ValueError("scalar DER refinement requires a DER feedback checkpoint")
    model.to(device=device, dtype=dtype)
    params = build_params(cfg, device, dtype)
    model.set_feature_vectors(params["r"], params["phi"])
    loss_args = configure_scalar_loss(
        source_args, cfg, args, device, dtype
    )

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    linears = [
        module
        for module in model.state_branch.modules()
        if isinstance(module, torch.nn.Linear)
    ]
    if not linears:
        raise ValueError("state branch contains no linear layers")
    for parameter in linears[-1].parameters():
        parameter.requires_grad_(True)
    trainable = list(linears[-1].parameters())

    train_states = anchored_states(
        args.train_random_states,
        args.train_seed,
        args.radius,
        cfg,
        device,
        dtype,
    )
    validation_states = anchored_states(
        args.validation_random_states,
        args.validation_seed,
        args.radius,
        cfg,
        device,
        dtype,
    )
    optimizer = torch.optim.LBFGS(
        trainable,
        lr=args.lr,
        max_iter=args.inner_iterations,
        max_eval=args.max_eval,
        tolerance_grad=args.tolerance_grad,
        tolerance_change=args.tolerance_change,
        history_size=args.history_size,
        line_search_fn="strong_wolfe",
    )

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_loss = math.inf
    best_step = 0
    best_state = copy.deepcopy(model.state_dict())

    def evaluate(step: int) -> float:
        nonlocal best_loss, best_step, best_state
        model.eval()
        with torch.no_grad():
            train_pack = section5_loss(
                model, train_states, cfg, params, loss_args
            )
            validation_pack = section5_loss(
                model, validation_states, cfg, params, loss_args
            )
        train_metrics = scalar_metrics(train_pack, cfg, loss_args)
        validation_metrics = scalar_metrics(
            validation_pack, cfg, loss_args
        )
        row = {
            "outer_step": step,
            "train_loss": train_metrics["loss"],
            "validation_loss": validation_metrics["loss"],
            "validation_q_H_u_rms": validation_metrics["q_psi_rms"],
            "validation_q_dH_u_dt_rms": validation_metrics[
                "q_dotpsi_rms"
            ],
            "validation_q_d2H_u_dt2_rms": validation_metrics[
                "q_ddotpsi_rms"
            ],
            "validation_objective": validation_metrics["objective"],
        }
        history.append(row)
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            best_step = step
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"[{step:03d}] train={row['train_loss']:.8g} "
            f"val={row['validation_loss']:.8g} "
            f"rms=({row['validation_q_H_u_rms']:.4g},"
            f"{row['validation_q_dH_u_dt_rms']:.4g},"
            f"{row['validation_q_d2H_u_dt2_rms']:.4g})",
            flush=True,
        )
        return validation_metrics["loss"]

    evaluate(0)
    for outer_step in range(1, args.outer_steps + 1):
        model.train()

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            pack = section5_loss(
                model, train_states, cfg, params, loss_args
            )
            pack["loss"].backward()
            return pack["loss"]

        optimizer.step(closure)
        evaluate(outer_step)

    model.load_state_dict(best_state)
    output_args = vars(source_args).copy()
    output_args.update(
        {
            "option": "der",
            "loss_variant": "lc_live",
            "candidate_mask_mode": "fixed_interval",
            "fixed_interval_start": args.interval_start,
            "fixed_interval_end": args.interval_end,
            "w0": 1.0,
            "w1": 1.0,
            "w2": 1.0,
            "w_lc": 0.0,
            "singular_loss_weight": 1.0,
            "nonsingular_loss_weight": 0.0,
            "smooth_weight": 0.0,
            "smooth_second_weight": 0.0,
            "smooth_max_weight": 0.0,
            "full_gradient_weight": 0.0,
            "full_gradient_max_weight": 0.0,
        }
    )
    payload = {
        "model_state": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "args": output_args,
        "problem": cfg.__dict__,
        "best_validation_loss": best_loss,
        "best_epoch": best_step,
        "selection_metric": "fixed_validation_scalar_residual_loss",
        "nominal_reference": model.nominal_reference.detach().cpu(),
        "initialization_checkpoint": str(checkpoint_path),
        "refinement": {
            "optimizer": "LBFGS",
            "train_scope": "state_branch_final_linear",
            "direct_supervision_used": False,
            "physical_objective_used": False,
            "interval": [args.interval_start, args.interval_end],
            "train_random_states": args.train_random_states,
            "validation_random_states": args.validation_random_states,
            "radius": args.radius,
        },
    }
    torch.save(payload, out_dir / "best_feedback_section5.pt")
    write_history(out_dir / "history.csv", history)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "best_validation_loss": best_loss,
                "best_outer_step": best_step,
                "initialization_checkpoint": str(checkpoint_path),
                "history": history,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--radius", type=float, default=0.10)
    parser.add_argument("--interval-start", type=float, default=1.5)
    parser.add_argument("--interval-end", type=float, default=8.0)
    parser.add_argument("--train-random-states", type=int, default=8)
    parser.add_argument("--validation-random-states", type=int, default=32)
    parser.add_argument("--train-seed", type=int, default=20260721)
    parser.add_argument("--validation-seed", type=int, default=20260719)
    parser.add_argument("--outer-steps", type=int, default=12)
    parser.add_argument("--inner-iterations", type=int, default=5)
    parser.add_argument("--max-eval", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--history-size", type=int, default=50)
    parser.add_argument("--tolerance-grad", type=float, default=1.0e-12)
    parser.add_argument("--tolerance-change", type=float, default=1.0e-14)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
