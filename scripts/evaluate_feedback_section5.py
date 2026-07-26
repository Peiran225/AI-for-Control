#!/usr/bin/env python3
"""Common float64 RK4-ZOH evaluation for Section-5 feedback checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_feedback_section5 import (  # noqa: E402
    NestedFeedbackTransformer,
    dynamics,
    load_operational_time_control,
    make_fixed_directions,
    test_states_from_directions,
)
from train_paper_pmp_kkt import ProblemConfig, build_params, parse_hidden  # noqa: E402


def load_feedback_checkpoint(path: Path) -> tuple[NestedFeedbackTransformer, ProblemConfig, argparse.Namespace]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    args = argparse.Namespace(**checkpoint["args"])
    cfg = ProblemConfig(**checkpoint["problem"])
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
        getattr(args, "state_feature_mode", "log_absolute"),
        bool(getattr(args, "center_state_correction", False)),
        float(getattr(args, "action_temperature", 1.0)),
        float(getattr(args, "action_scale", 1.0)),
        str(getattr(args, "action_parameterization", "logit-temperature")),
    ).double()
    model.load_state_dict(checkpoint["model_state"])
    feature_params = build_params(cfg, torch.device("cpu"), torch.float64)
    model.set_feature_vectors(feature_params["r"], feature_params["phi"])
    if "nominal_reference" in checkpoint:
        model.set_nominal_reference(
            checkpoint["nominal_reference"].to(dtype=torch.float64)
        )
    model.eval()
    return model, cfg, args


def running_cost(
    state: torch.Tensor,
    control: torch.Tensor,
    params: dict[str, torch.Tensor],
) -> torch.Tensor:
    return (state * params["beta"]).sum(dim=-1) + params["gamma"] * control


def rk4_zoh_feedback(
    model: NestedFeedbackTransformer,
    initial_state: torch.Tensor,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
    *,
    substeps: int,
    state_blind: bool,
    state_mode: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grid = torch.linspace(0.0, 1.0, cfg.n + 1, dtype=initial_state.dtype)
    base_logits = model.time_logits(grid)[: cfg.n]
    state = initial_state
    accumulated = torch.zeros(initial_state.shape[0], dtype=initial_state.dtype)
    controls = []
    step = cfg.T / cfg.n / substeps
    for index in range(cfg.n):
        time = torch.full(
            (initial_state.shape[0],), index / cfg.n, dtype=initial_state.dtype
        )
        control = model.interval_action(
            base_logits[index],
            time,
            state,
            state_blind=state_blind,
            state_mode=state_mode,
        )
        controls.append(control)
        for _ in range(substeps):
            k1 = dynamics(state, control, params)
            state2 = state + 0.5 * step * k1
            k2 = dynamics(state2, control, params)
            state3 = state + 0.5 * step * k2
            k3 = dynamics(state3, control, params)
            state4 = state + step * k3
            k4 = dynamics(state4, control, params)
            accumulated = accumulated + (step / 6.0) * (
                running_cost(state, control, params)
                + 2.0 * running_cost(state2, control, params)
                + 2.0 * running_cost(state3, control, params)
                + running_cost(state4, control, params)
            )
            state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    objective = accumulated + (state * params["alpha"]).sum(dim=-1)
    return objective, torch.stack(controls, dim=1), state


def rk4_zoh_open_loop(
    interval_control: torch.Tensor,
    initial_state: torch.Tensor,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
    *,
    substeps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = initial_state
    accumulated = torch.zeros(initial_state.shape[0], dtype=initial_state.dtype)
    step = cfg.T / cfg.n / substeps
    for index in range(cfg.n):
        control = (
            interval_control[:, index]
            if interval_control.ndim == 2
            else interval_control[index].expand(initial_state.shape[0])
        )
        for _ in range(substeps):
            k1 = dynamics(state, control, params)
            state2 = state + 0.5 * step * k1
            k2 = dynamics(state2, control, params)
            state3 = state + 0.5 * step * k2
            k3 = dynamics(state3, control, params)
            state4 = state + step * k3
            k4 = dynamics(state4, control, params)
            accumulated = accumulated + (step / 6.0) * (
                running_cost(state, control, params)
                + 2.0 * running_cost(state2, control, params)
                + 2.0 * running_cost(state3, control, params)
                + running_cost(state4, control, params)
            )
            state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    objective = accumulated + (state * params["alpha"]).sum(dim=-1)
    return objective, state


def projected_gradient_linf(
    interval_controls: torch.Tensor,
    initial_state: torch.Tensor,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
    *,
    substeps: int,
) -> torch.Tensor:
    with torch.enable_grad():
        controls = interval_controls.detach().clone().requires_grad_(True)
        objective, _ = rk4_zoh_open_loop(
            controls, initial_state, cfg, params, substeps=substeps
        )
        gradient = torch.autograd.grad(objective.sum(), controls)[0]
        mapping = controls - torch.clamp(controls - gradient, 0.0, cfg.umax)
        return mapping.abs().amax(dim=-1).detach()


def summarize(values: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def configured_state_mode(args: argparse.Namespace) -> str:
    return getattr(
        args,
        "state_mode",
        "fixed_query" if bool(getattr(args, "state_blind", False)) else "feedback",
    )


def evaluate(args: argparse.Namespace) -> None:
    feedback_model, cfg, feedback_args = load_feedback_checkpoint(Path(args.feedback))
    blind_model = None
    blind_args = None
    if args.blind:
        blind_model, blind_cfg, blind_args = load_feedback_checkpoint(Path(args.blind))
        if blind_cfg != cfg:
            raise ValueError("feedback and state-blind checkpoints use different problems")

    device = torch.device("cpu")
    params = build_params(cfg, device, torch.float64)
    directions = make_fixed_directions(args.test_size, cfg.m, args.test_seed, torch.float64)
    time_checkpoint = Path(args.time_checkpoint or feedback_args.time_checkpoint)
    time_control = load_operational_time_control(
        time_checkpoint, cfg, device, torch.float64
    )
    direct_data = np.load(args.direct_control_npz)
    direct_control = torch.tensor(
        np.asarray(direct_data["u"], dtype=np.float64).reshape(-1)[: cfg.n],
        dtype=torch.float64,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    summary: dict[str, dict] = {}
    with torch.no_grad():
        for radius in args.radii:
            initial = test_states_from_directions(
                directions, radius, cfg, device, torch.float64
            )
            feedback_J, feedback_u, _ = rk4_zoh_feedback(
                feedback_model,
                initial,
                cfg,
                params,
                substeps=args.substeps,
                state_blind=bool(feedback_args.state_blind),
                state_mode=configured_state_mode(feedback_args),
            )
            same_checkpoint_w_zero_J, same_checkpoint_w_zero_u, _ = (
                rk4_zoh_feedback(
                    feedback_model,
                    initial,
                    cfg,
                    params,
                    substeps=args.substeps,
                    state_blind=False,
                    state_mode="w_zero",
                )
            )
            time_J, _ = rk4_zoh_open_loop(
                time_control, initial, cfg, params, substeps=args.substeps
            )
            direct_J, _ = rk4_zoh_open_loop(
                direct_control, initial, cfg, params, substeps=args.substeps
            )
            blind_J = None
            if blind_model is not None and blind_args is not None:
                blind_J, _, _ = rk4_zoh_feedback(
                    blind_model,
                    initial,
                    cfg,
                    params,
                    substeps=args.substeps,
                    state_blind=bool(blind_args.state_blind),
                    state_mode=configured_state_mode(blind_args),
                )

            key = f"radius_{radius:.2f}"
            entry = {
                "feedback_J": summarize(feedback_J),
                "time_only_J": summarize(time_J),
                "nominal_direct_mesh_J": summarize(direct_J),
                "time_only_minus_feedback": summarize(time_J - feedback_J),
                "feedback_minus_nominal_direct_mesh": summarize(feedback_J - direct_J),
                "same_checkpoint_w_zero_J": summarize(same_checkpoint_w_zero_J),
                "same_checkpoint_w_zero_minus_feedback": summarize(
                    same_checkpoint_w_zero_J - feedback_J
                ),
                "feedback_win_fraction_vs_time": float((feedback_J < time_J).double().mean()),
                "feedback_win_fraction_vs_same_checkpoint_w_zero": float(
                    (feedback_J < same_checkpoint_w_zero_J).double().mean()
                ),
                "feedback_cross_sample_u_std": float(
                    feedback_u.std(dim=0, unbiased=False).mean()
                ),
            }
            if blind_J is not None:
                entry["blind_J"] = summarize(blind_J)
                entry["blind_minus_feedback"] = summarize(blind_J - feedback_J)
                entry["feedback_win_fraction_vs_blind"] = float(
                    (feedback_J < blind_J).double().mean()
                )

            stationarity_count = min(args.stationarity_size, initial.shape[0])
            stationarity_initial = initial[:stationarity_count]
            stationarity_feedback_u = feedback_u[:stationarity_count]
            feedback_pg = projected_gradient_linf(
                stationarity_feedback_u,
                stationarity_initial,
                cfg,
                params,
                substeps=args.substeps,
            )
            same_checkpoint_w_zero_pg = projected_gradient_linf(
                same_checkpoint_w_zero_u[:stationarity_count],
                stationarity_initial,
                cfg,
                params,
                substeps=args.substeps,
            )
            time_pg = projected_gradient_linf(
                time_control.expand(stationarity_count, -1),
                stationarity_initial,
                cfg,
                params,
                substeps=args.substeps,
            )
            entry["feedback_projected_gradient_linf"] = summarize(feedback_pg)
            entry["same_checkpoint_w_zero_projected_gradient_linf"] = summarize(
                same_checkpoint_w_zero_pg
            )
            entry["time_only_projected_gradient_linf"] = summarize(time_pg)
            direct_pg = projected_gradient_linf(
                direct_control.expand(stationarity_count, -1),
                stationarity_initial,
                cfg,
                params,
                substeps=args.substeps,
            )
            entry["nominal_direct_mesh_projected_gradient_linf"] = summarize(direct_pg)
            if blind_model is not None and blind_args is not None:
                with torch.no_grad():
                    _, stationarity_blind_u, _ = rk4_zoh_feedback(
                        blind_model,
                        stationarity_initial,
                        cfg,
                        params,
                        substeps=args.substeps,
                        state_blind=bool(blind_args.state_blind),
                        state_mode=configured_state_mode(blind_args),
                    )
                blind_pg = projected_gradient_linf(
                    stationarity_blind_u,
                    stationarity_initial,
                    cfg,
                    params,
                    substeps=args.substeps,
                )
                entry["blind_projected_gradient_linf"] = summarize(blind_pg)
            summary[key] = entry

            for index in range(initial.shape[0]):
                row = {
                    "radius": radius,
                    "sample": index,
                    "kind": "random",
                    "feedback_J": float(feedback_J[index]),
                    "time_only_J": float(time_J[index]),
                    "nominal_direct_mesh_J": float(direct_J[index]),
                    "time_only_minus_feedback": float(time_J[index] - feedback_J[index]),
                    "feedback_minus_nominal_direct_mesh": float(
                        feedback_J[index] - direct_J[index]
                    ),
                    "same_checkpoint_w_zero_J": float(
                        same_checkpoint_w_zero_J[index]
                    ),
                    "same_checkpoint_w_zero_minus_feedback": float(
                        same_checkpoint_w_zero_J[index] - feedback_J[index]
                    ),
                }
                if blind_J is not None:
                    row["blind_J"] = float(blind_J[index])
                    row["blind_minus_feedback"] = float(blind_J[index] - feedback_J[index])
                rows.append(row)

            trait = torch.linspace(-1.0, 1.0, cfg.m, dtype=torch.float64)
            stress_directions = torch.stack(
                [
                    torch.ones(cfg.m, dtype=torch.float64),
                    -torch.ones(cfg.m, dtype=torch.float64),
                    trait,
                    -trait,
                ]
            )
            stress_names = [
                "all_plus",
                "all_minus",
                "resistant_heavy",
                "sensitive_heavy",
            ]
            stress_initial = test_states_from_directions(
                stress_directions, radius, cfg, device, torch.float64
            )
            stress_feedback_J, _, _ = rk4_zoh_feedback(
                feedback_model,
                stress_initial,
                cfg,
                params,
                substeps=args.substeps,
                state_blind=bool(feedback_args.state_blind),
                state_mode=configured_state_mode(feedback_args),
            )
            stress_same_checkpoint_w_zero_J, _, _ = rk4_zoh_feedback(
                feedback_model,
                stress_initial,
                cfg,
                params,
                substeps=args.substeps,
                state_blind=False,
                state_mode="w_zero",
            )
            stress_time_J, _ = rk4_zoh_open_loop(
                time_control, stress_initial, cfg, params, substeps=args.substeps
            )
            stress_direct_J, _ = rk4_zoh_open_loop(
                direct_control, stress_initial, cfg, params, substeps=args.substeps
            )
            stress_blind_J = None
            if blind_model is not None and blind_args is not None:
                stress_blind_J, _, _ = rk4_zoh_feedback(
                    blind_model,
                    stress_initial,
                    cfg,
                    params,
                    substeps=args.substeps,
                    state_blind=bool(blind_args.state_blind),
                    state_mode=configured_state_mode(blind_args),
                )
            for index, name in enumerate(stress_names):
                row = {
                    "radius": radius,
                    "sample": index,
                    "kind": name,
                    "feedback_J": float(stress_feedback_J[index]),
                    "time_only_J": float(stress_time_J[index]),
                    "nominal_direct_mesh_J": float(stress_direct_J[index]),
                    "time_only_minus_feedback": float(
                        stress_time_J[index] - stress_feedback_J[index]
                    ),
                    "feedback_minus_nominal_direct_mesh": float(
                        stress_feedback_J[index] - stress_direct_J[index]
                    ),
                    "same_checkpoint_w_zero_J": float(
                        stress_same_checkpoint_w_zero_J[index]
                    ),
                    "same_checkpoint_w_zero_minus_feedback": float(
                        stress_same_checkpoint_w_zero_J[index]
                        - stress_feedback_J[index]
                    ),
                }
                if stress_blind_J is not None:
                    row["blind_J"] = float(stress_blind_J[index])
                    row["blind_minus_feedback"] = float(
                        stress_blind_J[index] - stress_feedback_J[index]
                    )
                rows.append(row)

    # Independent integration convergence check on the widest test set.
    widest = test_states_from_directions(
        directions, max(args.radii), cfg, device, torch.float64
    )
    with torch.no_grad():
        fine_J, _, _ = rk4_zoh_feedback(
            feedback_model,
            widest,
            cfg,
            params,
            substeps=args.convergence_substeps,
            state_blind=bool(feedback_args.state_blind),
            state_mode=configured_state_mode(feedback_args),
        )
        coarse_J, _, _ = rk4_zoh_feedback(
            feedback_model,
            widest,
            cfg,
            params,
            substeps=args.substeps,
            state_blind=bool(feedback_args.state_blind),
            state_mode=configured_state_mode(feedback_args),
        )
    summary["integration_convergence"] = {
        "coarse_substeps": args.substeps,
        "fine_substeps": args.convergence_substeps,
        "max_absolute_J_difference": float((fine_J - coarse_J).abs().max()),
        "max_relative_J_difference": float(
            ((fine_J - coarse_J).abs() / fine_J.abs().clamp_min(1e-12)).max()
        ),
    }

    with (out_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with (out_dir / "per_sample.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "protocol.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_floats(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", required=True)
    parser.add_argument("--blind", default="")
    parser.add_argument("--time_checkpoint", default="")
    parser.add_argument(
        "--direct_control_npz",
        default=str(ROOT / "paper_runs/canonical_results/solutions/direct_time_mesh_n200.npz"),
    )
    parser.add_argument("--radii", type=parse_floats, default=parse_floats("0.05,0.10,0.20"))
    parser.add_argument("--test_size", type=int, default=256)
    parser.add_argument("--test_seed", type=int, default=20260720)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--convergence_substeps", type=int, default=8)
    parser.add_argument("--stationarity_size", type=int, default=8)
    parser.add_argument("--out_dir", required=True)
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
