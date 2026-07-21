#!/usr/bin/env python3
"""Matched n=800 benchmark for feedback versus repeated direct recomputation.

For each held-out initial state, the script compares

1. replaying the nominal direct control;
2. the loaded Case-2 feedback policy; and
3. a state-specific box-constrained direct solve.

All objective values use the same float64 RK4-ZOH evaluator.  The direct solve
starts from the nominal direct solution and is restarted from the feedback
control only when that is needed to obtain a reference no worse than the
feedback candidate or to satisfy the projected-gradient tolerance.  Results
are checkpointed per state so the benchmark can be resumed.
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
from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for search_path in (ROOT, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from evaluate_feedback_refinement import (  # noqa: E402
    make_held_out_direction_families,
    prepare,
)
from evaluate_feedback_section5 import (  # noqa: E402
    configured_state_mode,
    rk4_zoh_feedback,
    rk4_zoh_open_loop,
)
from train_feedback_section5 import dynamics, test_states_from_directions  # noqa: E402
from train_paper_pmp_kkt import ProblemConfig, build_params  # noqa: E402


DEFAULT_FEEDBACK = (
    ROOT
    / "outputs/feedback_n800_state_refinement_20260720/der_moderate/"
    "best_feedback_section5_full_gradient.pt"
)
DEFAULT_DIRECT = (
    ROOT
    / "outputs/direct_n800_strict_refinement_20260720/active_set_newton/"
    "strict_direct_n800.npz"
)
DEFAULT_OUT = ROOT / "outputs/feedback_vs_state_direct_20260721"
FAMILIES = ("random", "total", "composition")


def safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def selected_pool_indices(pool_size: int, selected_count: int) -> list[int]:
    if selected_count <= 0 or selected_count > pool_size:
        raise ValueError("selected_per_family must lie in [1, family_pool_size]")
    if selected_count == 1:
        return [0]
    if selected_count == 2:
        return [0, 1]
    remaining = selected_count - 2
    tail = np.rint(np.linspace(2, pool_size - 1, remaining)).astype(int).tolist()
    return [0, 1, *tail]


def projected_gradient(control: np.ndarray, gradient: np.ndarray, upper: float) -> np.ndarray:
    return control - np.clip(control - gradient, 0.0, upper)


def value_and_gradient(
    control: np.ndarray,
    initial_state: torch.Tensor,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
    substeps: int,
) -> tuple[float, np.ndarray]:
    values = torch.tensor(control, dtype=torch.float64, requires_grad=True)
    objective, _ = rk4_zoh_open_loop(
        values.unsqueeze(0),
        initial_state.unsqueeze(0),
        cfg,
        params,
        substeps=substeps,
    )
    gradient = torch.autograd.grad(objective.sum(), values)[0]
    return float(objective.detach()[0]), gradient.detach().cpu().numpy()


@torch.no_grad()
def value_only(
    control: np.ndarray,
    initial_state: torch.Tensor,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
    substeps: int,
) -> float:
    values = torch.as_tensor(control, dtype=torch.float64)
    objective, _ = rk4_zoh_open_loop(
        values.unsqueeze(0),
        initial_state.unsqueeze(0),
        cfg,
        params,
        substeps=substeps,
    )
    return float(objective[0])


def run_lbfgsb(
    start: np.ndarray,
    initial_state: torch.Tensor,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
    args: argparse.Namespace,
    *,
    substeps: int,
    maxiter: int,
    maxfun: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    result = minimize(
        lambda control: value_and_gradient(
            control, initial_state, cfg, params, substeps
        ),
        np.clip(start, 0.0, cfg.umax),
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, cfg.umax)] * cfg.n,
        options={
            "maxiter": maxiter,
            "maxfun": maxfun,
            "ftol": args.ftol,
            "gtol": args.gtol,
            "maxls": args.maxls,
            "maxcor": args.maxcor,
        },
    )
    elapsed = time.perf_counter() - started
    control = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, cfg.umax)
    objective, gradient = value_and_gradient(
        control, initial_state, cfg, params, substeps
    )
    mapping = projected_gradient(control, gradient, cfg.umax)
    detail = {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "nit": int(result.nit),
        "nfev": int(result.nfev),
        "njev": int(getattr(result, "njev", result.nfev)),
        "elapsed_seconds": elapsed,
        "substeps": substeps,
        "J": objective,
        "gradient": gradient,
        "projected_gradient": mapping,
        "projected_gradient_linf": float(np.max(np.abs(mapping))),
        "projected_gradient_rms": float(np.sqrt(np.mean(mapping * mapping))),
    }
    return control, detail


def run_direct_chain(
    start: np.ndarray,
    start_label: str,
    initial_state: torch.Tensor,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    coarse_control, coarse = run_lbfgsb(
        start,
        initial_state,
        cfg,
        params,
        args,
        substeps=args.coarse_substeps,
        maxiter=args.maxiter,
        maxfun=args.maxfun,
    )
    coarse = {"start": start_label, "stage": "coarse", **coarse}
    fine_control, fine = run_lbfgsb(
        coarse_control,
        initial_state,
        cfg,
        params,
        args,
        substeps=args.substeps,
        maxiter=args.fine_maxiter,
        maxfun=args.fine_maxfun,
    )
    fine = {"start": start_label, "stage": "fine", **fine}
    return fine_control, fine, [coarse, fine]


@torch.no_grad()
def feedback_control_and_value(
    model: torch.nn.Module,
    feedback_args: argparse.Namespace,
    initial_state: torch.Tensor,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
    substeps: int,
) -> tuple[float, np.ndarray]:
    objective, controls, _ = rk4_zoh_feedback(
        model,
        initial_state.unsqueeze(0),
        cfg,
        params,
        substeps=substeps,
        state_blind=bool(getattr(feedback_args, "state_blind", False)),
        state_mode=configured_state_mode(feedback_args),
    )
    return float(objective[0]), controls[0].detach().cpu().numpy()


@torch.no_grad()
def integrate_interval(
    state: torch.Tensor,
    control: torch.Tensor,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
    substeps: int,
) -> torch.Tensor:
    step = cfg.T / cfg.n / substeps
    for _ in range(substeps):
        k1 = dynamics(state, control, params)
        state2 = state + 0.5 * step * k1
        k2 = dynamics(state2, control, params)
        state3 = state + 0.5 * step * k2
        k3 = dynamics(state3, control, params)
        state4 = state + step * k3
        k4 = dynamics(state4, control, params)
        state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return state


@torch.no_grad()
def open_loop_trajectory(
    control: np.ndarray,
    initial_state: torch.Tensor,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
    substeps: int,
) -> np.ndarray:
    state = initial_state.clone()
    states = [state.detach().cpu().numpy()]
    values = torch.as_tensor(control, dtype=torch.float64)
    for index in range(cfg.n):
        state = integrate_interval(
            state, values[index], cfg, params, substeps
        )
        states.append(state.detach().cpu().numpy())
    return np.stack(states)


@torch.no_grad()
def feedback_trajectory(
    model: torch.nn.Module,
    feedback_args: argparse.Namespace,
    initial_state: torch.Tensor,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
    substeps: int,
) -> tuple[np.ndarray, np.ndarray]:
    state = initial_state.unsqueeze(0).clone()
    grid = torch.linspace(0.0, 1.0, cfg.n + 1, dtype=torch.float64)
    base_logits = model.time_logits(grid)[: cfg.n]
    controls: list[np.ndarray] = []
    states = [state[0].detach().cpu().numpy()]
    for index in range(cfg.n):
        time_value = torch.full((1,), index / cfg.n, dtype=torch.float64)
        control = model.interval_action(
            base_logits[index],
            time_value,
            state,
            state_blind=bool(getattr(feedback_args, "state_blind", False)),
            state_mode=configured_state_mode(feedback_args),
        )
        controls.append(control[0].detach().cpu().numpy())
        state = integrate_interval(state, control, cfg, params, substeps)
        states.append(state[0].detach().cpu().numpy())
    return np.asarray(controls, dtype=np.float64), np.stack(states)


def median_runtime(function, repeats: int) -> tuple[float, list[float]]:
    function()
    times: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        times.append(time.perf_counter() - started)
    return float(np.median(times)), times


def solve_one(
    sample_id: str,
    family: str,
    name: str,
    pool_index: int,
    direction: np.ndarray,
    initial_state: torch.Tensor,
    model: torch.nn.Module,
    feedback_args: argparse.Namespace,
    nominal_direct: np.ndarray,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
    args: argparse.Namespace,
    sample_dir: Path,
    signature: str,
) -> dict[str, Any]:
    json_path = sample_dir / f"{sample_id}.json"
    npz_path = sample_dir / f"{sample_id}.npz"
    if args.resume and json_path.exists() and npz_path.exists():
        cached = json.loads(json_path.read_text())
        if cached.get("protocol_signature") == signature:
            return cached

    replay_J = value_only(
        nominal_direct, initial_state, cfg, params, args.substeps
    )
    feedback_J, feedback_control = feedback_control_and_value(
        model, feedback_args, initial_state, cfg, params, args.substeps
    )

    direct_control, nominal_run, runs = run_direct_chain(
        nominal_direct,
        "nominal_direct",
        initial_state,
        cfg,
        params,
        args,
    )
    selected_control = direct_control
    selected_run = nominal_run
    restart_needed = (
        nominal_run["projected_gradient_linf"] > args.direct_pg_tolerance
        or nominal_run["J"] > feedback_J + args.reference_margin
    )
    if restart_needed:
        feedback_start_control, feedback_run, feedback_runs = run_direct_chain(
            feedback_control,
            "feedback",
            initial_state,
            cfg,
            params,
            args,
        )
        runs.extend(feedback_runs)
        if feedback_run["J"] < selected_run["J"]:
            selected_control = feedback_start_control
            selected_run = feedback_run

    direct_J = float(selected_run["J"])
    direct_pg = float(selected_run["projected_gradient_linf"])
    if direct_J > min(replay_J, feedback_J) + args.reference_margin:
        raise RuntimeError(
            f"{sample_id}: state-specific direct result J={direct_J:.12g} is worse "
            f"than an evaluated candidate {min(replay_J, feedback_J):.12g}"
        )

    replay_time, replay_times = median_runtime(
        lambda: value_only(
            nominal_direct, initial_state, cfg, params, args.substeps
        ),
        args.timing_repeats,
    )
    feedback_time, feedback_times = median_runtime(
        lambda: feedback_control_and_value(
            model, feedback_args, initial_state, cfg, params, args.substeps
        ),
        args.timing_repeats,
    )

    replay_states = open_loop_trajectory(
        nominal_direct, initial_state, cfg, params, args.substeps
    )
    checked_feedback_control, feedback_states = feedback_trajectory(
        model, feedback_args, initial_state, cfg, params, args.substeps
    )
    if not np.allclose(checked_feedback_control, feedback_control, rtol=0.0, atol=1e-11):
        raise RuntimeError(f"{sample_id}: feedback trajectory control mismatch")
    direct_states = open_loop_trajectory(
        selected_control, initial_state, cfg, params, args.substeps
    )

    np.savez_compressed(
        npz_path,
        direction=np.asarray(direction, dtype=np.float64),
        N0=initial_state.detach().cpu().numpy(),
        t=np.linspace(0.0, cfg.T, cfg.n + 1),
        nominal_replay_control=nominal_direct,
        feedback_control=feedback_control,
        state_direct_control=selected_control,
        nominal_replay_states=replay_states,
        feedback_states=feedback_states,
        state_direct_states=direct_states,
        state_direct_gradient=np.asarray(selected_run["gradient"]),
        state_direct_projected_gradient=np.asarray(
            selected_run["projected_gradient"]
        ),
    )

    direct_total_seconds = float(sum(float(run["elapsed_seconds"]) for run in runs))
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "family": family,
        "name": name,
        "pool_index": pool_index,
        "protocol_signature": signature,
        "N0_mean": float(initial_state.mean()),
        "N0_std": float(initial_state.std(unbiased=False)),
        "J_nominal_direct_replay": replay_J,
        "J_feedback": feedback_J,
        "J_state_direct": direct_J,
        "feedback_advantage_vs_replay": replay_J - feedback_J,
        "replay_excess_vs_state_direct": replay_J - direct_J,
        "feedback_excess_vs_state_direct": feedback_J - direct_J,
        "benefit_recovered": (
            (replay_J - feedback_J) / (replay_J - direct_J)
            if replay_J - direct_J > args.recovery_denominator_tolerance
            else None
        ),
        "state_direct_projected_gradient_linf": direct_pg,
        "state_direct_projected_gradient_rms": float(
            selected_run["projected_gradient_rms"]
        ),
        "state_direct_kkt_pass": bool(direct_pg <= args.direct_pg_tolerance),
        "state_direct_selected_start": str(selected_run["start"]),
        "state_direct_optimizer_success": bool(selected_run["success"]),
        "state_direct_optimizer_nit": int(selected_run["nit"]),
        "state_direct_optimizer_nfev": int(selected_run["nfev"]),
        "state_direct_optimizer_message": str(selected_run["message"]),
        "state_direct_restarts": len({str(run["start"]) for run in runs}),
        "state_direct_solve_seconds": direct_total_seconds,
        "nominal_replay_rollout_median_seconds": replay_time,
        "feedback_rollout_median_seconds": feedback_time,
        "nominal_replay_rollout_times_seconds": replay_times,
        "feedback_rollout_times_seconds": feedback_times,
        "artifact": npz_path.name,
        "direct_runs": [
            {
                key: value
                for key, value in run.items()
                if key not in {"gradient", "projected_gradient"}
            }
            for run in runs
        ],
    }
    json_path.write_text(json.dumps(safe(row), indent=2, sort_keys=True) + "\n")
    return row


def summarize_results(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    cfg: ProblemConfig,
    signature: str,
    out_dir: Path,
) -> dict[str, Any]:
    flat_rows = [
        {
            key: value
            for key, value in row.items()
            if key
            not in {
                "direct_runs",
                "nominal_replay_rollout_times_seconds",
                "feedback_rollout_times_seconds",
            }
        }
        for row in rows
    ]
    with (out_dir / "per_state.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    def array(key: str) -> np.ndarray:
        return np.asarray([row[key] for row in rows], dtype=np.float64)

    recovered = np.asarray(
        [row["benefit_recovered"] for row in rows if row["benefit_recovered"] is not None],
        dtype=np.float64,
    )
    by_family: dict[str, Any] = {}
    for family in FAMILIES:
        family_rows = [row for row in rows if row["family"] == family]
        by_family[family] = {
            "count": len(family_rows),
            "feedback_advantage_vs_replay": stats(
                np.asarray(
                    [row["feedback_advantage_vs_replay"] for row in family_rows]
                )
            ),
            "benefit_recovered": stats(
                np.asarray(
                    [
                        row["benefit_recovered"]
                        for row in family_rows
                        if row["benefit_recovered"] is not None
                    ]
                )
            ),
        }

    summary = {
        "comparison": (
            "Nominal direct replay versus loaded Case-2 feedback versus "
            "state-specific direct recomputation"
        ),
        "problem": asdict(cfg),
        "protocol_signature": signature,
        "protocol": safe(vars(args)),
        "sample_count": len(rows),
        "state_direct_kkt_pass_count": int(
            sum(bool(row["state_direct_kkt_pass"]) for row in rows)
        ),
        "feedback_win_fraction_vs_nominal_direct_replay": float(
            np.mean(array("feedback_advantage_vs_replay") > 0.0)
        ),
        "feedback_advantage_vs_nominal_direct_replay": stats(
            array("feedback_advantage_vs_replay")
        ),
        "nominal_replay_excess_vs_state_direct": stats(
            array("replay_excess_vs_state_direct")
        ),
        "feedback_excess_vs_state_direct": stats(
            array("feedback_excess_vs_state_direct")
        ),
        "benefit_recovered": stats(recovered) if recovered.size else None,
        "benefit_recovered_count": int(recovered.size),
        "state_direct_projected_gradient_linf": stats(
            array("state_direct_projected_gradient_linf")
        ),
        "nominal_replay_rollout_seconds": stats(
            array("nominal_replay_rollout_median_seconds")
        ),
        "feedback_rollout_seconds": stats(
            array("feedback_rollout_median_seconds")
        ),
        "state_direct_solve_seconds": stats(array("state_direct_solve_seconds")),
        "median_speedup_feedback_vs_state_direct": float(
            np.median(array("state_direct_solve_seconds"))
            / np.median(array("feedback_rollout_median_seconds"))
        ),
        "by_family": by_family,
        "artifacts": {
            "per_state": "per_state.csv",
            "sample_directory": "samples",
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(safe(summary), indent=2, sort_keys=True) + "\n"
    )
    return summary


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    feedback_path = args.feedback.expanduser().resolve()
    direct_path = args.nominal_direct.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    sample_dir = out_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    model, cfg, feedback_args = prepare(feedback_path, args.n)
    if cfg.n != args.n:
        raise ValueError("feedback checkpoint could not be prepared at the requested n")
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    direct_pack = np.load(direct_path)
    direct_key = "interval_u" if "interval_u" in direct_pack else "u"
    nominal_direct = np.asarray(direct_pack[direct_key], dtype=np.float64).reshape(-1)[: cfg.n]
    if nominal_direct.size != cfg.n:
        raise ValueError("nominal direct artifact has the wrong control length")

    all_directions, all_labels, all_names = make_held_out_direction_families(
        args.family_pool_size,
        cfg.m,
        args.direction_seed,
        torch.float64,
        list(FAMILIES),
    )
    pool_indices = selected_pool_indices(
        args.family_pool_size, args.selected_per_family
    )
    selected_global: list[int] = []
    for family_index in range(len(FAMILIES)):
        selected_global.extend(
            family_index * args.family_pool_size + index for index in pool_indices
        )
    selected_directions = all_directions[selected_global]
    initial_states = test_states_from_directions(
        selected_directions,
        args.radius,
        cfg,
        torch.device("cpu"),
        torch.float64,
    )

    signature_payload = {
        "feedback_sha256": sha256(feedback_path),
        "direct_sha256": sha256(direct_path),
        "problem": asdict(cfg),
        "radius": args.radius,
        "direction_seed": args.direction_seed,
        "family_pool_size": args.family_pool_size,
        "selected_per_family": args.selected_per_family,
        "pool_indices": pool_indices,
        "substeps": args.substeps,
        "coarse_substeps": args.coarse_substeps,
        "maxiter": args.maxiter,
        "maxfun": args.maxfun,
        "fine_maxiter": args.fine_maxiter,
        "fine_maxfun": args.fine_maxfun,
        "ftol": args.ftol,
        "gtol": args.gtol,
        "direct_pg_tolerance": args.direct_pg_tolerance,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True).encode()
    ).hexdigest()
    (out_dir / "protocol.json").write_text(
        json.dumps(safe(signature_payload | {"protocol_signature": signature}), indent=2)
        + "\n"
    )

    rows: list[dict[str, Any]] = []
    for position, global_index in enumerate(selected_global):
        family = all_labels[global_index]
        name = all_names[global_index]
        pool_index = global_index % args.family_pool_size
        sample_id = f"{family}_{pool_index:04d}"
        print(
            f"[{position + 1}/{len(selected_global)}] {sample_id} ({name})",
            flush=True,
        )
        row = solve_one(
            sample_id,
            family,
            name,
            pool_index,
            selected_directions[position].detach().cpu().numpy(),
            initial_states[position],
            model,
            feedback_args,
            nominal_direct,
            cfg,
            params,
            args,
            sample_dir,
            signature,
        )
        rows.append(row)
        print(
            "  "
            f"feedback gain={row['feedback_advantage_vs_replay']:.6g}, "
            f"recovered={row['benefit_recovered']}, "
            f"direct PG={row['state_direct_projected_gradient_linf']:.3e}, "
            f"direct={row['state_direct_solve_seconds']:.2f}s, "
            f"feedback={1e3 * row['feedback_rollout_median_seconds']:.2f}ms",
            flush=True,
        )

    summary = summarize_results(rows, args, cfg, signature, out_dir)
    print(json.dumps(safe(summary), indent=2, sort_keys=True), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    result.add_argument("--nominal-direct", type=Path, default=DEFAULT_DIRECT)
    result.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    result.add_argument("--n", type=int, default=800)
    result.add_argument("--radius", type=float, default=0.10)
    result.add_argument("--direction-seed", type=int, default=20260723)
    result.add_argument("--family-pool-size", type=int, default=128)
    result.add_argument("--selected-per-family", type=int, default=8)
    result.add_argument("--substeps", type=int, default=4)
    result.add_argument("--coarse-substeps", type=int, default=1)
    result.add_argument("--maxiter", type=int, default=120)
    result.add_argument("--maxfun", type=int, default=260)
    result.add_argument("--fine-maxiter", type=int, default=10)
    result.add_argument("--fine-maxfun", type=int, default=80)
    result.add_argument("--ftol", type=float, default=0.0)
    result.add_argument("--gtol", type=float, default=1.0e-9)
    result.add_argument("--maxls", type=int, default=50)
    result.add_argument("--maxcor", type=int, default=30)
    result.add_argument("--direct-pg-tolerance", type=float, default=1.0e-4)
    result.add_argument("--reference-margin", type=float, default=1.0e-9)
    result.add_argument(
        "--recovery-denominator-tolerance", type=float, default=1.0e-10
    )
    result.add_argument("--timing-repeats", type=int, default=3)
    result.add_argument("--threads", type=int, default=1)
    result.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    return result


if __name__ == "__main__":
    run(parser().parse_args())
