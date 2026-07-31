#!/usr/bin/env python3
"""Off-grid scalar continuation of a direct-initialized Transformer.

The input checkpoint may have been initialized from direct transcription.
After it is loaded, this script never opens a direct solution, never matches a
teacher control, and never uses the physical objective for training or model
selection.  It updates only the original Transformer parameters using
PMP/KKT scalar conditions on the support grid and randomly sampled strict
midpoints.  No external control correction is introduced.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT, ROOT / "scripts"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from continue_direct_initialized_scalar_transformer import (  # noqa: E402
    clone_state,
    load_model,
    sha256,
)
from refine_direct_offgrid_query_fit import (  # noqa: E402
    fixed_support_query_control,
)
from train_feedback_section5 import (  # noqa: E402
    compute_costate_rk4,
    dH_dN,
    dynamics,
    simulate_open_loop,
    singular_quantities,
)
from train_paper_pmp_kkt import (  # noqa: E402
    build_params,
    set_seed,
    time_features,
)
from train_teacher_free_resolution_curriculum import evaluate  # noqa: E402


def resolve(path: str | Path) -> Path:
    expanded = Path(path).expanduser()
    return (expanded if expanded.is_absolute() else ROOT / expanded).resolve()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dense_control(
    model: torch.nn.Module,
    support_time: torch.Tensor,
    query_time: torch.Tensor,
    *,
    query_batch_size: int,
    train_query_indices: torch.Tensor | None = None,
    query_baseline: torch.Tensor | None = None,
) -> torch.Tensor:
    """Interleave support values with fixed-context strict midpoint queries."""

    support = model(support_time)
    if query_baseline is None:
        with torch.no_grad():
            query = fixed_support_query_control(
                model,
                support_time,
                query_time,
                batch_size=query_batch_size,
            )
    else:
        query = query_baseline.detach()
    if train_query_indices is not None and train_query_indices.numel() > 0:
        selected = fixed_support_query_control(
            model,
            support_time,
            query_time[train_query_indices],
            batch_size=query_batch_size,
        )
        query = query.index_copy(0, train_query_indices, selected)
    output = torch.empty(
        support.numel() + query.numel(),
        device=support.device,
        dtype=support.dtype,
    )
    output[0::2] = support
    output[1::2] = query
    return output


def full_fixed_support_dense_control(
    model: torch.nn.Module,
    support_time: torch.Tensor,
    query_time: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Evaluate all fixed-context midpoint queries in one differentiable pass."""

    support_count = int(support_time.numel())
    combined_time = torch.cat((support_time, query_time))
    embedded = model.base.input(time_features(combined_time)).unsqueeze(0)
    encoded = model.base.encoder(
        embedded,
        mask=attention_mask,
    ).squeeze(0)
    raw = model.base.output(encoded).squeeze(-1)
    base_control = model.umax * torch.sigmoid(raw)
    control = torch.clamp(
        model.scale * base_control - model.offset,
        0.0,
        model.umax,
    )
    support_control = control[:support_count]
    query_control = control[support_count:]
    output = torch.empty(
        support_count + query_time.numel(),
        device=control.device,
        dtype=control.dtype,
    )
    output[0::2] = support_control
    output[1::2] = query_control
    return output


def simulate_continuous_control_rk4(
    node_control: torch.Tensor,
    initial_state: torch.Tensor,
    cfg: Any,
    params: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """RK4 state solve with linearly varying control between time nodes."""

    step = cfg.T / cfg.n
    state = initial_state
    states = [state]
    midpoint_states = []
    for index in range(cfg.n):
        control_left = node_control[:, index]
        control_right = node_control[:, index + 1]
        control_mid = 0.5 * (control_left + control_right)
        k1 = dynamics(state, control_left, params)
        state2 = state + 0.5 * step * k1
        k2 = dynamics(state2, control_mid, params)
        state3 = state + 0.5 * step * k2
        k3 = dynamics(state3, control_mid, params)
        state4 = state + step * k3
        k4 = dynamics(state4, control_right, params)
        next_state = state + (step / 6.0) * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )
        midpoint_states.append(0.5 * (state2 + state3))
        state = next_state.clamp_min(1.0e-8)
        states.append(state)
    return torch.stack(states, dim=1), torch.stack(midpoint_states, dim=1)


def compute_continuous_costate_rk4(
    states: torch.Tensor,
    midpoint_states: torch.Tensor,
    node_control: torch.Tensor,
    cfg: Any,
    params: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Backward RK4 solve of the continuous PMP costate equation."""

    step = cfg.T / cfg.n
    batch = states.shape[0]
    costate = params["alpha"].expand(batch, -1)
    values: list[torch.Tensor | None] = [None] * (cfg.n + 1)
    values[cfg.n] = costate

    def rhs(
        state: torch.Tensor,
        current_costate: torch.Tensor,
        control: torch.Tensor,
    ) -> torch.Tensor:
        return -dH_dN(state, current_costate, control, params)

    for index in range(cfg.n - 1, -1, -1):
        state_left = states[:, index]
        state_mid = midpoint_states[:, index]
        state_right = states[:, index + 1]
        control_left = node_control[:, index]
        control_right = node_control[:, index + 1]
        control_mid = 0.5 * (control_left + control_right)
        k1 = rhs(state_right, costate, control_right)
        k2 = rhs(state_mid, costate - 0.5 * step * k1, control_mid)
        k3 = rhs(state_mid, costate - 0.5 * step * k2, control_mid)
        k4 = rhs(state_left, costate - step * k3, control_left)
        costate = costate - (step / 6.0) * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )
        values[index] = costate
    return torch.stack(values, dim=1)  # type: ignore[arg-type]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2607)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--optimizer",
        choices=("adamw", "lbfgs"),
        default="adamw",
        help="Optimizer used for scalar-condition continuation.",
    )
    parser.add_argument(
        "--trainable-scope",
        choices=(
            "all",
            "output-head",
            "input-output",
            "last-block-output",
        ),
        default="all",
        help=(
            "Optimize all Transformer parameters, only its final linear head, "
            "the input/output projections, or the final encoder block plus "
            "the output head."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=5.0e-8)
    parser.add_argument("--final-learning-rate", type=float, default=1.0e-9)
    parser.add_argument("--lbfgs-inner-iterations", type=int, default=5)
    parser.add_argument("--lbfgs-history-size", type=int, default=20)
    parser.add_argument("--lbfgs-tolerance-grad", type=float, default=1.0e-12)
    parser.add_argument("--lbfgs-tolerance-change", type=float, default=1.0e-15)
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument(
        "--full-query-gradient",
        action="store_true",
        help=(
            "Evaluate every strict midpoint with the exact fixed-support "
            "attention context and retain gradients through all queries."
        ),
    )
    parser.add_argument(
        "--train-query-count",
        type=int,
        default=24,
        help="Number of strict midpoint queries carrying gradient per update.",
    )
    parser.add_argument("--refresh-query-every", type=int, default=5)
    parser.add_argument(
        "--psi-scale",
        type=float,
        default=2.5e-3,
        help="Scale in normalized training units.",
    )
    parser.add_argument(
        "--dot-scale",
        type=float,
        default=2.5e-3,
        help="Scale in normalized training units.",
    )
    parser.add_argument(
        "--ddot-scale",
        type=float,
        default=2.5e-3,
        help="Scale in normalized training units.",
    )
    parser.add_argument("--w0", type=float, default=1.0)
    parser.add_argument("--w1", type=float, default=1.0)
    parser.add_argument("--w2", type=float, default=1.0)
    parser.add_argument("--boundary-kkt-weight", type=float, default=1.0)
    parser.add_argument(
        "--boundary-gate-mode",
        choices=("interval-complement", "source-box"),
        default="interval-complement",
        help=(
            "Apply boundary KKT either outside the scalar interval or only "
            "where the starting control lies near a box boundary."
        ),
    )
    parser.add_argument(
        "--boundary-margin",
        type=float,
        default=0.05,
        help="Control-space margin used by --boundary-gate-mode=source-box.",
    )
    parser.add_argument("--interior-start", type=float, default=1.5)
    parser.add_argument("--interior-end", type=float, default=8.0)
    parser.add_argument(
        "--costate-pairing", choices=("current", "next"), default="current"
    )
    parser.add_argument(
        "--trajectory-mode",
        choices=("continuous-rk4", "zoh-discrete-adjoint"),
        default="continuous-rk4",
        help=(
            "Use the continuous PMP state/costate equations or the legacy "
            "ZOH transcription and matching discrete adjoint."
        ),
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--report-scale-factor", type=float, default=400.0)
    parser.add_argument("--save-eval-checkpoints", action="store_true")
    parser.add_argument(
        "--diagnose-gradients",
        action="store_true",
        help=(
            "Report parameter-gradient norms and pairwise cosines of the "
            "three physical scalar residual losses at the starting checkpoint, "
            "then exit without training."
        ),
    )
    args = parser.parse_args()

    if not (
        args.epochs >= 0
        and args.eval_every > 0
        and args.query_batch_size > 0
        and args.train_query_count >= 0
        and 0.0 <= args.interior_start < args.interior_end <= 10.0
    ):
        raise ValueError("invalid training or interval setting")
    for name in ("psi_scale", "dot_scale", "ddot_scale"):
        if getattr(args, name) <= 0.0:
            raise ValueError(f"{name} must be positive")

    set_seed(args.seed)
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    start_checkpoint = resolve(args.start_checkpoint)
    out_dir = resolve(args.out_dir)
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True)
    candidate_dir = out_dir / "candidates"
    if args.save_eval_checkpoints:
        candidate_dir.mkdir()

    model, cfg, source = load_model(start_checkpoint, device)
    if args.trainable_scope != "all":
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    if args.trainable_scope == "output-head":
        for parameter in model.base.output.parameters():
            parameter.requires_grad_(True)
    elif args.trainable_scope == "input-output":
        for module in (model.base.input, model.base.output):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    elif args.trainable_scope == "last-block-output":
        for module in (model.base.encoder.layers[-1], model.base.output):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    optimization_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not optimization_parameters:
        raise RuntimeError("no trainable Transformer parameters were selected")
    dense_cfg = replace(cfg, n=2 * cfg.n)
    support_time = torch.linspace(
        0.0, 1.0, cfg.n + 1, device=device, dtype=torch.float64
    )
    query_time = (
        torch.arange(cfg.n, device=device, dtype=torch.float64) + 0.5
    ) / cfg.n
    full_attention_mask: torch.Tensor | None = None
    if args.full_query_gradient:
        sequence_length = support_time.numel() + query_time.numel()
        support_count = support_time.numel()
        full_attention_mask = torch.zeros(
            (sequence_length, sequence_length),
            device=device,
            dtype=torch.bool,
        )
        full_attention_mask[:support_count, support_count:] = True
        full_attention_mask[support_count:, support_count:] = True
    query_physical_time = query_time.detach().cpu().numpy() * cfg.T
    eligible_query_indices = torch.tensor(
        np.flatnonzero(
            (query_physical_time >= args.interior_start)
            & (query_physical_time < args.interior_end)
        ),
        device=device,
        dtype=torch.long,
    )
    dense_physical_time = np.linspace(0.0, cfg.T, dense_cfg.n + 1)
    interval_t = dense_physical_time[:-1]
    interior = torch.tensor(
        (interval_t >= args.interior_start)
        & (interval_t < args.interior_end),
        device=device,
        dtype=torch.bool,
    ).unsqueeze(0)
    interval_exterior = ~interior

    params = build_params(dense_cfg, device, torch.float64)
    original_params = build_params(cfg, device, torch.float64)
    initial_state = torch.full(
        (1, cfg.m),
        cfg.n0,
        device=device,
        dtype=torch.float64,
    )
    model.eval()
    with torch.no_grad():
        if args.full_query_gradient:
            if full_attention_mask is None:
                raise RuntimeError("full-query attention mask was not built")
            starting_dense = full_fixed_support_dense_control(
                model,
                support_time,
                query_time,
                full_attention_mask,
            ).detach()
            cached_query = starting_dense[1::2].detach()
            standalone_support = model(support_time)
            support_error = float(
                (starting_dense[0::2] - standalone_support).abs().max().cpu()
            )
            check_indices = torch.linspace(
                0,
                query_time.numel() - 1,
                min(32, query_time.numel()),
                device=device,
                dtype=torch.float64,
            ).round().long()
            pointwise_queries = fixed_support_query_control(
                model,
                support_time,
                query_time[check_indices],
                batch_size=args.query_batch_size,
            )
            query_error = float(
                (
                    starting_dense[1::2][check_indices]
                    - pointwise_queries
                )
                .abs()
                .max()
                .cpu()
            )
            if max(support_error, query_error) > 2.0e-11:
                raise RuntimeError(
                    "full-query construction changed fixed-context outputs: "
                    f"support={support_error:.3e}, query={query_error:.3e}"
                )
        else:
            cached_query = fixed_support_query_control(
                model,
                support_time,
                query_time,
                batch_size=args.query_batch_size,
            ).detach()
            starting_dense = dense_control(
                model,
                support_time,
                query_time,
                query_batch_size=args.query_batch_size,
                query_baseline=cached_query,
            ).detach()
    if args.boundary_gate_mode == "source-box":
        source_interval_control = starting_dense[:-1].unsqueeze(0)
        boundary_gate = (
            (source_interval_control <= args.boundary_margin)
            | (
                source_interval_control
                >= dense_cfg.umax - args.boundary_margin
            )
        )
    else:
        boundary_gate = interval_exterior
    if not bool(boundary_gate.any()):
        raise RuntimeError("boundary KKT gate is empty")

    def refresh_query_cache() -> None:
        nonlocal cached_query
        if args.full_query_gradient:
            return
        model.eval()
        with torch.no_grad():
            cached_query = fixed_support_query_control(
                model,
                support_time,
                query_time,
                batch_size=args.query_batch_size,
            ).detach()

    def raw_pack(
        train_query_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if args.full_query_gradient:
            if full_attention_mask is None:
                raise RuntimeError("full-query attention mask was not built")
            all_control = full_fixed_support_dense_control(
                model,
                support_time,
                query_time,
                full_attention_mask,
            )
        else:
            all_control = dense_control(
                model,
                support_time,
                query_time,
                query_batch_size=args.query_batch_size,
                train_query_indices=train_query_indices,
                query_baseline=cached_query,
            )
        node_controls = all_control.unsqueeze(0)
        controls = node_controls[:, :-1]
        if args.trajectory_mode == "continuous-rk4":
            states, midpoint_states = simulate_continuous_control_rk4(
                node_controls,
                initial_state,
                dense_cfg,
                params,
            )
            costates = compute_continuous_costate_rk4(
                states,
                midpoint_states,
                node_controls,
                dense_cfg,
                params,
            )
        else:
            states = simulate_open_loop(
                controls,
                initial_state,
                dense_cfg,
                params,
                integrator="rk4",
            )
            costates, _ = compute_costate_rk4(
                states, controls, dense_cfg, params
            )
        quantities = singular_quantities(
            states,
            controls,
            costates,
            dense_cfg,
            params,
            costate_pairing=args.costate_pairing,
        )
        psi_all = quantities["psi"]
        psi = psi_all[interior]
        dot = quantities["dot_psi"][interior]
        ddot = quantities["ddot_psi"][interior]
        projected = controls - torch.clamp(
            controls - psi_all, 0.0, dense_cfg.umax
        )
        return {
            "all_control": all_control,
            "psi": psi,
            "dot": dot,
            "ddot": ddot,
            "psi_rms": psi.square().mean().sqrt(),
            "dot_rms": dot.square().mean().sqrt(),
            "ddot_rms": ddot.square().mean().sqrt(),
            "boundary_kkt": projected[boundary_gate].square().mean(),
            "start_drift_rms": (
                all_control - starting_dense
            ).square().mean().sqrt(),
            "start_drift_linf": (
                all_control - starting_dense
            ).abs().max(),
        }

    def loss_pack(
        train_query_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pack = raw_pack(train_query_indices)
        psi_term = (pack["psi"] / args.psi_scale).square().mean()
        dot_term = (pack["dot"] / args.dot_scale).square().mean()
        ddot_term = (pack["ddot"] / args.ddot_scale).square().mean()
        scalar_loss = (
            args.w0 * psi_term
            + args.w1 * dot_term
            + args.w2 * ddot_term
        )
        loss = scalar_loss + args.boundary_kkt_weight * pack["boundary_kkt"]
        return {
            **pack,
            "loss": loss,
            "scalar_loss": scalar_loss,
            "psi_term": psi_term,
            "dot_term": dot_term,
            "ddot_term": ddot_term,
        }

    if args.optimizer == "adamw":
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            optimization_parameters,
            lr=args.learning_rate,
            weight_decay=0.0,
            eps=1.0e-12,
        )
    else:
        optimizer = torch.optim.LBFGS(
            optimization_parameters,
            lr=args.learning_rate,
            max_iter=args.lbfgs_inner_iterations,
            history_size=args.lbfgs_history_size,
            tolerance_grad=args.lbfgs_tolerance_grad,
            tolerance_change=args.lbfgs_tolerance_change,
            line_search_fn="strong_wolfe",
        )
    if args.diagnose_gradients:
        query_count = min(
            args.train_query_count, int(eligible_query_indices.numel())
        )
        if query_count:
            positions = torch.linspace(
                0,
                int(eligible_query_indices.numel()) - 1,
                query_count,
                device=device,
                dtype=torch.float64,
            ).round().long()
            diagnostic_query_indices = eligible_query_indices[positions]
        else:
            diagnostic_query_indices = torch.empty(
                0, device=device, dtype=torch.long
            )
        model.train()
        diagnostic_pack = raw_pack(diagnostic_query_indices)
        physical_losses = {
            "H_u": diagnostic_pack["psi"].square().mean()
            * args.report_scale_factor**2,
            "dH_u_dt": diagnostic_pack["dot"].square().mean()
            * args.report_scale_factor**2,
            "d2H_u_dt2": diagnostic_pack["ddot"].square().mean()
            * args.report_scale_factor**2,
            "boundary_KKT": diagnostic_pack["boundary_kkt"],
        }
        parameters = optimization_parameters
        gradient_vectors: dict[str, torch.Tensor] = {}
        for name, component_loss in physical_losses.items():
            gradients = torch.autograd.grad(
                component_loss,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            gradient_vectors[name] = torch.cat(
                [
                    (
                        gradient.reshape(-1)
                        if gradient is not None
                        else torch.zeros_like(parameter).reshape(-1)
                    )
                    for parameter, gradient in zip(parameters, gradients)
                ]
            )
        gradient_norms = {
            name: float(vector.norm().detach().cpu())
            for name, vector in gradient_vectors.items()
        }
        pairwise_cosines: dict[str, float] = {}
        names = list(gradient_vectors)
        for left_index, left_name in enumerate(names):
            for right_name in names[left_index + 1 :]:
                left = gradient_vectors[left_name]
                right = gradient_vectors[right_name]
                denominator = left.norm() * right.norm()
                cosine = (
                    left.dot(right) / denominator
                    if float(denominator.detach()) > 0.0
                    else torch.tensor(float("nan"), device=device)
                )
                pairwise_cosines[f"{left_name}__{right_name}"] = float(
                    cosine.detach().cpu()
                )
        diagnostics = {
            "checkpoint": str(start_checkpoint),
            "query_count": query_count,
            "physical_rms": {
                "H_u": args.report_scale_factor
                * float(diagnostic_pack["psi_rms"].detach()),
                "dH_u_dt": args.report_scale_factor
                * float(diagnostic_pack["dot_rms"].detach()),
                "d2H_u_dt2": args.report_scale_factor
                * float(diagnostic_pack["ddot_rms"].detach()),
            },
            "physical_squared_loss_gradient_norms": gradient_norms,
            "pairwise_gradient_cosines": pairwise_cosines,
        }
        (out_dir / "gradient_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2), encoding="utf-8"
        )
        print(json.dumps(diagnostics, indent=2), flush=True)
        return

    history: list[dict[str, Any]] = []
    candidates: list[
        tuple[
            float,
            float,
            float,
            float,
            float,
            dict[str, torch.Tensor],
            int,
        ]
    ] = []
    started = time.perf_counter()

    def make_payload(
        state: dict[str, torch.Tensor],
        epoch: int,
        scalar_rms: dict[str, float],
    ) -> dict[str, Any]:
        return {
            **source,
            "model_state": state,
            "source_checkpoint": str(start_checkpoint),
            "source_checkpoint_sha256": sha256(start_checkpoint),
            "teacher_free": False,
            "direct_or_manual_solution_used": True,
            "continuation_reads_direct_or_manual_solution": False,
            "continuation_uses_objective_value_as_loss_or_selection": False,
            "continuation_uses_direct_based_gate_or_sampling": False,
            "external_correction_head": False,
            "method": (
                "direct-initialized Transformer followed by support-and-off-"
                "grid PMP/KKT scalar-condition continuation"
            ),
            "selected_scalar_epoch": epoch,
            "scalar_dense_grid_rms": scalar_rms,
            "continuation_args": vars(args),
        }

    def record(epoch: int) -> None:
        model.eval()
        refresh_query_cache()
        with torch.no_grad():
            pack = raw_pack(
                torch.empty(0, device=device, dtype=torch.long)
            )
        row = {
            key: float(value.detach().cpu())
            for key, value in pack.items()
            if value.numel() == 1
        }
        physical = [
            args.report_scale_factor * row["psi_rms"],
            args.report_scale_factor * row["dot_rms"],
            args.report_scale_factor * row["ddot_rms"],
        ]
        maximum = max(physical)
        joint = math.sqrt(sum(value * value for value in physical))
        state = clone_state(model)
        candidates.append(
            (
                maximum,
                joint,
                physical[0],
                physical[1],
                physical[2],
                state,
                epoch,
            )
        )
        row.update(
            {
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "physical_psi_rms": physical[0],
                "physical_dot_rms": physical[1],
                "physical_ddot_rms": physical[2],
                "physical_component_max": maximum,
                "physical_joint_rms": joint,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        history.append(row)
        if args.save_eval_checkpoints:
            torch.save(
                make_payload(
                    state,
                    epoch,
                    {
                        "H_u": physical[0],
                        "dH_u_dt": physical[1],
                        "d2H_u_dt2": physical[2],
                    },
                ),
                candidate_dir / f"epoch_{epoch:05d}.pt",
            )
        print(
            f"[epoch {epoch:05d}] dense physical RMS "
            f"{physical[0]:.6e}/{physical[1]:.6e}/{physical[2]:.6e} "
            f"max={maximum:.6e} drift={row['start_drift_linf']:.3e}",
            flush=True,
        )
        model.train()

    record(0)
    if args.optimizer == "adamw":
        for epoch in range(1, args.epochs + 1):
            if (
                epoch == 1
                or (epoch - 1) % max(args.refresh_query_every, 1) == 0
            ):
                refresh_query_cache()
            fraction = epoch / max(args.epochs, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * fraction))
            optimizer.param_groups[0]["lr"] = (
                args.final_learning_rate
                + cosine * (args.learning_rate - args.final_learning_rate)
            )
            optimizer.zero_grad(set_to_none=True)
            query_count = min(
                args.train_query_count, int(eligible_query_indices.numel())
            )
            if query_count:
                order = torch.randperm(
                    int(eligible_query_indices.numel()), device=device
                )[:query_count]
                train_query_indices = eligible_query_indices[order]
            else:
                train_query_indices = torch.empty(
                    0, device=device, dtype=torch.long
                )
            pack = loss_pack(train_query_indices)
            pack["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                optimization_parameters, args.grad_clip
            )
            optimizer.step()
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                record(epoch)
    else:
        query_count = min(
            args.train_query_count, int(eligible_query_indices.numel())
        )
        if query_count:
            positions = torch.linspace(
                0,
                int(eligible_query_indices.numel()) - 1,
                query_count,
                device=device,
                dtype=torch.float64,
            ).round().long()
            fixed_query_indices = eligible_query_indices[positions]
        else:
            fixed_query_indices = torch.empty(
                0, device=device, dtype=torch.long
            )
        for epoch in range(1, args.epochs + 1):
            refresh_query_cache()

            def closure() -> torch.Tensor:
                optimizer.zero_grad(set_to_none=True)
                current = loss_pack(fixed_query_indices)
                current["loss"].backward()
                return current["loss"]

            optimizer.step(closure)
            if epoch % args.eval_every == 0 or epoch == args.epochs:
                record(epoch)

    selected = min(candidates, key=lambda item: item[:5])
    model.load_state_dict(selected[5], strict=True)
    refresh_query_cache()
    grid_metrics, selected_support = evaluate(
        model,
        cfg,
        support_time,
        original_params,
        high_accuracy=True,
    )
    with torch.no_grad():
        selected_pack = raw_pack()
    selected_dense = (
        selected_pack["all_control"].detach().cpu().numpy().copy()
    )
    scalar_rms = {
        "H_u": args.report_scale_factor
        * float(selected_pack["psi_rms"].detach()),
        "dH_u_dt": args.report_scale_factor
        * float(selected_pack["dot_rms"].detach()),
        "d2H_u_dt2": args.report_scale_factor
        * float(selected_pack["ddot_rms"].detach()),
    }
    payload = make_payload(selected[5], selected[6], scalar_rms)
    payload["metrics"] = grid_metrics
    torch.save(payload, out_dir / "selected_checkpoint.pt")
    np.savez_compressed(
        out_dir / "solution.npz",
        t=dense_physical_time,
        u=selected_dense,
        support_t=dense_physical_time[0::2],
        support_u=selected_support,
        starting_u=starting_dense.detach().cpu().numpy(),
    )
    write_csv(out_dir / "history.csv", history)
    summary = {
        "status": "completed",
        "start_checkpoint": str(start_checkpoint),
        "start_checkpoint_sha256": sha256(start_checkpoint),
        "device": str(device),
        "dense_intervals": dense_cfg.n,
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "continuation_reads_direct_or_manual_solution": False,
        "continuation_uses_objective_value_as_loss_or_selection": False,
        "continuation_uses_direct_based_gate_or_sampling": False,
        "external_correction_head": False,
        "training": vars(args),
        "selected_epoch": selected[6],
        "selected_scalar_dense_grid_rms": scalar_rms,
        "grid_metrics": grid_metrics,
        "wall_seconds": time.perf_counter() - started,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
