#!/usr/bin/env python3
"""Initialize the nested feedback correction from direct trajectories.

The direct controls are used only in this supervised initialization stage.
The output checkpoint contains the original nested feedback policy, without
an auxiliary correction head, and can be passed to ``train_feedback_section5``
through ``--state_checkpoint`` for subsequent scalar PMP/KKT refinement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_feedback_section5 import (  # noqa: E402
    NestedFeedbackTransformer,
    rk4_state_step,
    simulate_feedback,
)
from scripts.generate_offgrid_policy_switching_diagnostics import (  # noqa: E402
    DensePolicy,
    integrate_trajectory,
)
from scripts.refine_feedback_offgrid_scalar import (  # noqa: E402
    fixed_support_dense_logits,
)
from train_paper_pmp_kkt import (  # noqa: E402
    ProblemConfig,
    build_params,
    parse_hidden,
    set_seed,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def load_direct_teacher(
    path: Path,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor | str]:
    source = np.load(path)
    time_key = "t" if "t" in source.files else "time"
    control_key = "u" if "u" in source.files else "control"
    if time_key not in source.files or control_key not in source.files:
        raise ValueError(f"{path} must contain time and control arrays")
    if "initial_state" not in source.files:
        raise ValueError(f"{path} must contain initial_state")

    times_np = np.asarray(source[time_key], dtype=np.float64).reshape(-1)
    controls_np = np.asarray(source[control_key], dtype=np.float64).reshape(-1)
    initial_np = np.asarray(source["initial_state"], dtype=np.float64).reshape(-1)
    if times_np.size == cfg.n + 1 and controls_np.size == cfg.n + 1:
        controls_np = controls_np[:-1]
    if times_np.size != cfg.n + 1 or controls_np.size != cfg.n:
        raise ValueError(
            f"{path} has incompatible grid: t={times_np.size}, "
            f"u={controls_np.size}, expected {cfg.n + 1}/{cfg.n}"
        )
    expected = np.linspace(0.0, cfg.T, cfg.n + 1)
    if not np.allclose(times_np, expected, atol=1.0e-11, rtol=0.0):
        raise ValueError(f"{path} is not on the declared uniform n={cfg.n} grid")
    if initial_np.size != cfg.m:
        raise ValueError(
            f"{path} initial state has {initial_np.size} components; expected {cfg.m}"
        )
    if np.any(controls_np < -1.0e-12) or np.any(controls_np > cfg.umax + 1.0e-12):
        raise ValueError(f"{path} control violates [0, {cfg.umax}]")

    initial = torch.as_tensor(initial_np, device=device, dtype=dtype)
    controls = torch.as_tensor(
        np.clip(controls_np, 0.0, cfg.umax), device=device, dtype=dtype
    )
    state = initial
    states = [state]
    step = cfg.T / cfg.n
    with torch.no_grad():
        for control in controls:
            state = rk4_state_step(
                state.unsqueeze(0),
                control.reshape(1),
                step,
                params,
            )[0]
            states.append(state)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "times": torch.as_tensor(
            times_np[:-1] / cfg.T, device=device, dtype=dtype
        ),
        "states": torch.stack(states[:-1]),
        "controls": controls,
        "initial_state": initial,
    }


def weighted_control_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    normalized_times: torch.Tensor,
    *,
    T: float,
    interior_start: float,
    interior_end: float,
    interior_weight: float,
) -> torch.Tensor:
    physical_time = normalized_times * T
    interior = (
        (physical_time >= interior_start) & (physical_time < interior_end)
    ).to(prediction.dtype)
    weights = 1.0 + (interior_weight - 1.0) * interior
    return (weights * (prediction - target).square()).sum() / weights.sum()


def prediction_for_samples(
    model: NestedFeedbackTransformer,
    normalized_times: torch.Tensor,
    states: torch.Tensor,
    base_logits: torch.Tensor,
) -> torch.Tensor:
    return model.interval_action(
        base_logits,
        normalized_times,
        states,
        state_mode="feedback",
    )


def control_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    normalized_times: torch.Tensor,
    *,
    T: float,
    interior_start: float,
    interior_end: float,
) -> dict[str, float]:
    difference = prediction - target
    physical_time = normalized_times * T
    interior = (physical_time >= interior_start) & (physical_time < interior_end)

    def metrics_for(values: torch.Tensor, prefix: str) -> dict[str, float]:
        if values.numel() == 0:
            return {
                f"{prefix}_rms": float("nan"),
                f"{prefix}_linf": float("nan"),
            }
        return {
            f"{prefix}_rms": float(values.square().mean().sqrt().cpu()),
            f"{prefix}_linf": float(values.abs().max().cpu()),
        }

    return {
        **metrics_for(difference, "full"),
        **metrics_for(difference[interior], "interior"),
    }


def write_csv(path: Path, rows: Iterable[dict[str, float | int]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_nominal_reference(
    model: NestedFeedbackTransformer,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
    time_checkpoint: Path,
    args: argparse.Namespace,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Construct the feature/centering reference used throughout Stage 1."""

    nominal_initial = torch.full(
        (1, cfg.m), cfg.n0, device=device, dtype=dtype
    )
    if args.nominal_reference_method == "zoh":
        with torch.no_grad():
            nominal_states, _, _ = simulate_feedback(
                model,
                nominal_initial,
                cfg,
                params,
                state_mode="w_zero",
                integrator="rk4",
            )
        return nominal_states[0]

    multiplier = int(args.nominal_reference_multiplier)
    dense_normalized, dense_raw = fixed_support_dense_logits(
        model,
        cfg,
        multiplier,
        query_batch_size=args.nominal_reference_query_batch_size,
    )
    if model.action_parameterization == "linear-raw-box":
        wrapper: dict[str, object] = {
            "class": "LinearRawBoxProjection"
        }
    elif model.action_offset:
        if not math.isclose(model.action_temperature, 1.0):
            raise ValueError(
                "the DOP853 reference does not support a simultaneous "
                "nonunit action temperature and nonzero offset"
            )
        wrapper = {
            "class": "AffineBoundaryProjectedControl",
            "scale": model.action_scale,
            "offset": model.action_offset,
        }
    else:
        wrapper = {
            "class": "FixedBoxProjection",
            "scale": model.action_scale,
            "temperature": model.action_temperature,
        }
    dense_time = dense_normalized.detach().cpu().numpy() * cfg.T
    dense_raw_numpy = dense_raw.detach().cpu().numpy()
    policy = DensePolicy(
        "time_only_reference",
        cfg,
        dense_time,
        dense_raw_numpy,
        dense_raw_numpy[::multiplier],
        time_checkpoint,
        time_wrapper=wrapper,
    )
    result = integrate_trajectory(
        policy,
        np.full(cfg.m, cfg.n0, dtype=np.float64),
        dense_time,
        rtol=args.nominal_reference_rtol,
        atol=args.nominal_reference_atol,
        max_step=cfg.T / (cfg.n * multiplier),
    )
    return torch.from_numpy(result.state).to(device=device, dtype=dtype)


def train(args: argparse.Namespace) -> None:
    if args.nominal_reference_multiplier < 1:
        raise ValueError("nominal reference multiplier must be positive")
    if args.nominal_reference_query_batch_size < 1:
        raise ValueError("nominal reference query batch size must be positive")
    if min(
        args.nominal_reference_rtol,
        args.nominal_reference_atol,
    ) <= 0.0:
        raise ValueError("nominal reference tolerances must be positive")
    set_seed(args.seed)
    # Use the same deterministic attention implementation during checkpoint
    # selection as in refinement and the final continuous-policy evaluation.
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = torch.float64 if args.float64 else torch.float32
    if device.type == "mps" and dtype == torch.float64:
        raise ValueError("MPS does not support float64")

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
    model = NestedFeedbackTransformer(
        cfg.m,
        cfg.umax,
        args.state_scale,
        parse_hidden(args.state_hidden),
        args.d_model,
        args.heads,
        args.layers,
        args.init_u,
        args.correction_gain,
        args.state_feature_mode,
        True,
        args.action_temperature,
        args.action_scale,
        args.action_parameterization,
        args.action_offset,
    ).to(device=device, dtype=dtype)
    time_checkpoint = Path(args.time_checkpoint).resolve()
    checkpoint = model.load_time_checkpoint(time_checkpoint)
    checkpoint_problem = checkpoint.get("problem", {})
    for key in ("n", "m"):
        if int(checkpoint_problem.get(key, getattr(cfg, key))) != int(
            getattr(cfg, key)
        ):
            raise ValueError(f"time checkpoint {key} mismatch")
    for key in ("T", "umax", "beta", "alpha", "gamma", "n0", "m_suppression"):
        if not math.isclose(
            float(checkpoint_problem.get(key, getattr(cfg, key))),
            float(getattr(cfg, key)),
        ):
            raise ValueError(f"time checkpoint {key} mismatch")
    model.to(device=device, dtype=dtype)
    model.set_feature_vectors(params["r"], params["phi"])

    nominal_reference = build_nominal_reference(
        model,
        cfg,
        params,
        time_checkpoint,
        args,
        device=device,
        dtype=dtype,
    )
    model.set_nominal_reference(nominal_reference)

    teachers = [
        load_direct_teacher(
            Path(path).resolve(),
            cfg,
            params,
            device=device,
            dtype=dtype,
        )
        for path in args.teacher
    ]
    if not teachers:
        raise ValueError("at least one --teacher trajectory is required")

    for parameter in model.time_branch.parameters():
        parameter.requires_grad_(False)
    trainable = [parameter for parameter in model.state_branch.parameters()]
    def make_optimizer() -> tuple[
        torch.optim.AdamW, torch.optim.lr_scheduler.ReduceLROnPlateau
    ]:
        optimizer = torch.optim.AdamW(
            trainable,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=args.lr_patience,
            min_lr=args.min_lr,
        )
        return optimizer, scheduler

    optimizer, scheduler = make_optimizer()

    grid = torch.linspace(0.0, 1.0, cfg.n + 1, device=device, dtype=dtype)
    with torch.no_grad():
        base_grid_logits = model.time_logits(grid)[: cfg.n].detach()

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_round = -1
    best_epoch = -1
    start = time.time()

    # The first round uses direct-trajectory states. Later rounds add the
    # policy's own closed-loop states while retaining the same direct actions.
    sample_state_sets = [
        [teacher["states"].detach().clone()]  # type: ignore[union-attr]
        for teacher in teachers
    ]

    for dagger_round in range(args.dagger_rounds + 1):
        if dagger_round > 0 and args.reset_optimizer_each_round:
            optimizer, scheduler = make_optimizer()
        time_blocks = []
        target_blocks = []
        state_blocks = []
        for teacher, teacher_state_sets in zip(teachers, sample_state_sets):
            for state_set in teacher_state_sets:
                time_blocks.append(teacher["times"])
                target_blocks.append(teacher["controls"])
                state_blocks.append(state_set)
        times = torch.cat(time_blocks)  # type: ignore[arg-type]
        targets = torch.cat(target_blocks)  # type: ignore[arg-type]
        states = torch.cat(state_blocks)
        base_logits = base_grid_logits.repeat(len(state_blocks))
        dataset_size = times.numel()

        for epoch in range(1, args.epochs_per_round + 1):
            model.train()
            permutation = torch.randperm(dataset_size, device=device)
            for start_index in range(0, dataset_size, args.batch_size):
                indices = permutation[
                    start_index : start_index + args.batch_size
                ]
                optimizer.zero_grad(set_to_none=True)
                prediction = prediction_for_samples(
                    model,
                    times[indices],
                    states[indices],
                    base_logits[indices],
                )
                loss = weighted_control_loss(
                    prediction,
                    targets[indices],
                    times[indices],
                    T=cfg.T,
                    interior_start=args.interior_start,
                    interior_end=args.interior_end,
                    interior_weight=args.interior_weight,
                )
                loss.backward()
                if args.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                optimizer.step()

            should_evaluate = (
                epoch == 1
                or epoch % args.eval_every == 0
                or epoch == args.epochs_per_round
            )
            if not should_evaluate:
                continue
            model.eval()
            with torch.no_grad():
                prediction = prediction_for_samples(
                    model, times, states, base_logits
                )
                validation_loss = weighted_control_loss(
                    prediction,
                    targets,
                    times,
                    T=cfg.T,
                    interior_start=args.interior_start,
                    interior_end=args.interior_end,
                    interior_weight=args.interior_weight,
                )
                metrics = control_metrics(
                    prediction,
                    targets,
                    times,
                    T=cfg.T,
                    interior_start=args.interior_start,
                    interior_end=args.interior_end,
                )
                closed_loop_metrics = []
                for teacher in teachers:
                    _, rollout_controls, _ = simulate_feedback(
                        model,
                        teacher["initial_state"].reshape(1, -1),  # type: ignore[union-attr]
                        cfg,
                        params,
                        state_mode="feedback",
                        integrator="rk4",
                    )
                    closed_loop_metrics.append(
                        control_metrics(
                            rollout_controls[0],
                            teacher["controls"],  # type: ignore[arg-type]
                            teacher["times"],  # type: ignore[arg-type]
                            T=cfg.T,
                            interior_start=args.interior_start,
                            interior_end=args.interior_end,
                        )
                    )
            score = float(
                np.mean(
                    [
                        metric["interior_rms"]
                        for metric in closed_loop_metrics
                    ]
                )
            )
            scheduler.step(score)
            row: dict[str, float | int] = {
                "dagger_round": dagger_round,
                "epoch": epoch,
                "teacher_forced_loss": float(validation_loss.cpu()),
                "closed_loop_interior_rms": score,
                "closed_loop_interior_linf": float(
                    max(
                        metric["interior_linf"]
                        for metric in closed_loop_metrics
                    )
                ),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": time.time() - start,
                **metrics,
            }
            history.append(row)
            if score < best_loss:
                best_loss = score
                best_state = cpu_state_dict(model)
                best_round = dagger_round
                best_epoch = epoch
            print(
                f"[round={dagger_round} epoch={epoch:04d}] "
                f"teacher_loss={float(validation_loss.cpu()):.6g} "
                f"closed_loop_rms={score:.6g} "
                f"closed_loop_linf={row['closed_loop_interior_linf']:.6g}",
                flush=True,
            )

        if dagger_round < args.dagger_rounds:
            model.eval()
            with torch.no_grad():
                for teacher, teacher_state_sets in zip(
                    teachers, sample_state_sets
                ):
                    rollout_states, _, _ = simulate_feedback(
                        model,
                        teacher["initial_state"].reshape(1, -1),  # type: ignore[union-attr]
                        cfg,
                        params,
                        state_mode="feedback",
                        integrator="rk4",
                    )
                    teacher_state_sets.append(
                        rollout_states[0, :-1].detach()
                    )

    if best_state is None:
        raise RuntimeError("supervised initialization produced no checkpoint")
    model.load_state_dict(best_state)
    model.set_nominal_reference(nominal_reference)
    model.set_feature_vectors(params["r"], params["phi"])
    model.eval()

    teacher_summaries = []
    rollout_arrays: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for index, teacher in enumerate(teachers):
            rollout_states, rollout_controls, _ = simulate_feedback(
                model,
                teacher["initial_state"].reshape(1, -1),  # type: ignore[union-attr]
                cfg,
                params,
                state_mode="feedback",
                integrator="rk4",
            )
            normalized_times = teacher["times"]  # type: ignore[assignment]
            target = teacher["controls"]  # type: ignore[assignment]
            metrics = control_metrics(
                rollout_controls[0],
                target,
                normalized_times,
                T=cfg.T,
                interior_start=args.interior_start,
                interior_end=args.interior_end,
            )
            teacher_summaries.append(
                {
                    "path": teacher["path"],
                    "sha256": teacher["sha256"],
                    **metrics,
                }
            )
            rollout_arrays[f"teacher_{index}_initial_state"] = (
                teacher["initial_state"].detach().cpu().numpy()  # type: ignore[union-attr]
            )
            rollout_arrays[f"teacher_{index}_target_control"] = (
                target.detach().cpu().numpy()
            )
            rollout_arrays[f"teacher_{index}_policy_control"] = (
                rollout_controls[0].detach().cpu().numpy()
            )
            rollout_arrays[f"teacher_{index}_policy_state"] = (
                rollout_states[0].detach().cpu().numpy()
            )

    checkpoint_payload = {
        "model_state": cpu_state_dict(model),
        "args": {
            "d_model": args.d_model,
            "heads": args.heads,
            "layers": args.layers,
            "init_u": args.init_u,
            "state_hidden": args.state_hidden,
            "state_scale": args.state_scale,
            "correction_gain": args.correction_gain,
            "state_feature_mode": args.state_feature_mode,
            "center_state_correction": True,
            "action_temperature": float(model.action_temperature),
            "action_scale": float(model.action_scale),
            "action_parameterization": str(model.action_parameterization),
            "action_offset": float(model.action_offset),
            "state_mode": "feedback",
            "training_integrator": "rk4",
            "initialization": "direct-trajectory supervised state-branch fit",
            "nominal_reference_method": args.nominal_reference_method,
            "nominal_reference_multiplier": (
                args.nominal_reference_multiplier
            ),
        },
        "problem": asdict(cfg),
        "time_checkpoint": str(time_checkpoint),
        "teacher_trajectories": teacher_summaries,
        "best_loss": best_loss,
        "selection_metric": "closed_loop_direct_control_interior_rms",
        "best_dagger_round": best_round,
        "best_epoch": best_epoch,
        "nominal_reference": model.nominal_reference.detach().cpu().clone(),
    }
    torch.save(checkpoint_payload, output / "feedback_direct_initialization.pt")
    np.savez(output / "closed_loop_comparison.npz", **rollout_arrays)
    write_csv(output / "training_history.csv", history)
    summary = {
        "method": (
            "direct trajectories initialize only the nested state branch; "
            "the saved policy has no auxiliary correction head"
        ),
        "physical_objective_used_as_loss": False,
        "direct_targets_used_after_initialization": False,
        "problem": asdict(cfg),
        "protocol": {
            **vars(args),
            "time_checkpoint": str(time_checkpoint),
            "teacher": [str(Path(path).resolve()) for path in args.teacher],
        },
        "best_loss": best_loss,
        "selection_metric": "closed_loop_direct_control_interior_rms",
        "best_dagger_round": best_round,
        "best_epoch": best_epoch,
        "elapsed_seconds": time.time() - start,
        "teacher_trajectories": teacher_summaries,
        "checkpoint": "feedback_direct_initialization.pt",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-checkpoint", required=True)
    parser.add_argument("--teacher", action="append", default=[])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--float64", action="store_true")
    parser.add_argument("--seed", type=int, default=31)

    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--m", type=int, default=21)
    parser.add_argument("--umax", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.0025)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=20.0)
    parser.add_argument("--n0", type=float, default=10.0)
    parser.add_argument("--m-suppression", type=float, default=0.5)

    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--init-u", type=float, default=1.5)
    parser.add_argument("--state-hidden", default="128,128")
    parser.add_argument("--state-scale", type=float, default=15.0)
    parser.add_argument("--correction-gain", type=float, default=1.0)
    parser.add_argument(
        "--state-feature-mode",
        choices=("relative_nominal", "burden_composition", "log_absolute"),
        default="burden_composition",
    )
    parser.add_argument("--action-temperature", type=float, default=1.0)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument(
        "--action-parameterization",
        choices=("logit-temperature", "linear-raw-box"),
        default="logit-temperature",
    )
    parser.add_argument("--action-offset", type=float, default=0.0)

    parser.add_argument("--epochs-per-round", type=int, default=600)
    parser.add_argument("--dagger-rounds", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--min-lr", type=float, default=1.0e-6)
    parser.add_argument("--lr-patience", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--reset-optimizer-each-round",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "restart AdamW and its scheduler after each DAgger state refresh; "
            "this prevents a learning rate reduced on the previous state "
            "distribution from carrying into the next round"
        ),
    )
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--interior-start", type=float, default=1.5)
    parser.add_argument("--interior-end", type=float, default=8.0)
    parser.add_argument("--interior-weight", type=float, default=4.0)
    parser.add_argument(
        "--nominal-reference-method",
        choices=("zoh", "dop853"),
        default="zoh",
    )
    parser.add_argument(
        "--nominal-reference-multiplier",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--nominal-reference-query-batch-size",
        type=int,
        default=16,
    )
    parser.add_argument("--nominal-reference-rtol", type=float, default=1.0e-10)
    parser.add_argument("--nominal-reference-atol", type=float, default=1.0e-12)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
