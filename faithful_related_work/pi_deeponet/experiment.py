"""Reproducible experiment and artifact layer for the faithful implementation."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .core import (
    ImprovedPolicy,
    TrainConfig,
    build_model,
    terminal_branch_values,
    train_policy_iteration,
)
from .problems import ControlProblem, TumorAdaptation


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: tuple[int, ...]
    terminal_parameter_family: tuple[float, ...]
    target_terminal_parameter: float
    initial_control: tuple[float, ...]
    initial_state: tuple[float, ...]
    evaluation_intervals: int
    output_dir: Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        raise ValueError("history must not be empty")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _rk4_step(problem: ControlProblem, t: float, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    dtype = torch.float64

    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        tt = torch.tensor([time], dtype=dtype)
        xx = torch.as_tensor(state[None, :], dtype=dtype)
        uu = torch.as_tensor(u[None, :], dtype=dtype)
        return problem.dynamics(tt, xx, uu).detach().cpu().numpy()[0]

    k1 = rhs(t, x)
    k2 = rhs(t + 0.5 * dt, x + 0.5 * dt * k1)
    k3 = rhs(t + 0.5 * dt, x + 0.5 * dt * k2)
    k4 = rhs(t + dt, x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def feedback_rollout(
    problem: ControlProblem,
    model: torch.nn.Module,
    sensor_states: np.ndarray,
    terminal_parameter: float,
    initial_state: np.ndarray,
    *,
    h: float,
    tie_tolerance: float,
    intervals: int,
) -> dict[str, np.ndarray | float | int | bool]:
    """Export a breakpoint-aligned ZOH trajectory from the exact policy."""

    if intervals < 1:
        raise ValueError("evaluation_intervals must be positive")
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    sensors = torch.as_tensor(sensor_states, device=device, dtype=dtype)
    parameter = torch.tensor([terminal_parameter], device=device, dtype=dtype)
    branch = terminal_branch_values(problem, parameter, sensors)
    policy = ImprovedPolicy(model, problem, h, tie_tolerance)
    t = np.linspace(0.0, problem.T, intervals + 1, dtype=np.float64)
    x = np.empty((intervals + 1, problem.state_dim), dtype=np.float64)
    u = np.empty((intervals, problem.control_dim), dtype=np.float64)
    nonunique = np.zeros(intervals, dtype=bool)
    x[0] = np.asarray(initial_state, dtype=np.float64)
    for index in range(intervals):
        tt = torch.tensor([t[index]], device=device, dtype=dtype)
        xx = torch.as_tensor(x[index][None, :], device=device, dtype=dtype)
        result = policy(branch, tt, xx)
        u[index] = result.control.detach().cpu().numpy()[0]
        nonunique[index] = bool(result.nonunique.any().detach().cpu())
        x[index + 1] = _rk4_step(problem, t[index], x[index], u[index], t[index + 1] - t[index])
    domain_exit = bool(np.any(x < problem.state_lower[None, :] - 1.0e-12) or np.any(x > problem.state_upper[None, :] + 1.0e-12))
    return {
        "t": t,
        "u": u,
        "x": x,
        "nonunique": nonunique,
        "nonunique_count": int(nonunique.sum()),
        "domain_exit": domain_exit,
    }


def _native_lqr_cost(
    problem: ControlProblem,
    trajectory: dict[str, np.ndarray | float | int | bool],
    terminal_parameter: float,
) -> float:
    t = np.asarray(trajectory["t"])
    x = np.asarray(trajectory["x"])
    u = np.asarray(trajectory["u"])
    tt = torch.as_tensor(t[:-1], dtype=torch.float64)
    xx = torch.as_tensor(x[:-1], dtype=torch.float64)
    uu = torch.as_tensor(u, dtype=torch.float64)
    running = problem.running_cost(tt, xx, uu).detach().cpu().numpy()
    terminal = problem.terminal_cost(
        torch.as_tensor(x[-1][None, :], dtype=torch.float64),
        torch.tensor([terminal_parameter], dtype=torch.float64),
    ).item()
    return float(np.sum(running * np.diff(t)) + terminal)


def _save_solution(
    path: Path,
    problem: ControlProblem,
    trajectory: dict[str, np.ndarray | float | int | bool],
    *,
    seed: int,
    outer: int,
    terminal_parameter: float,
) -> dict[str, float | int | bool]:
    t = np.asarray(trajectory["t"], dtype=np.float64)
    u_matrix = np.asarray(trajectory["u"], dtype=np.float64)
    x = np.asarray(trajectory["x"], dtype=np.float64)
    u_export = u_matrix[:, 0] if problem.control_dim == 1 else u_matrix
    payload: dict[str, object] = {
        "t": t,
        "u": u_export,
        "x": x,
        "seed": np.int64(seed),
        "outer": np.int64(outer),
        "target_terminal_parameter": np.float64(terminal_parameter),
        "nonunique": np.asarray(trajectory["nonunique"], dtype=bool),
        "control_execution": np.array("zero_order_hold"),
        "selection_rule": np.array("final_predeclared_outer_iteration"),
    }
    if isinstance(problem, TumorAdaptation):
        from tumor_problem import TumorProblem, evaluate_zoh_control

        physical_N = problem.state_scale * x
        evaluation_problem = TumorProblem(
            T=problem.T,
            m=problem.state_dim,
            umax=problem.umax,
            beta=problem.beta,
            alpha=problem.alpha,
            gamma=problem.gamma,
            n0=problem.initial_N,
            m_suppression=problem.suppression,
        )
        realized = evaluate_zoh_control(
            t,
            u_export,
            evaluation_problem,
            include_diagnostics=False,
        )
        payload.update(
            {
                "N": physical_N,
                "J_realized": np.float64(realized["J"]),
                "running_cost_realized": np.float64(realized["running_cost"]),
                "terminal_cost_realized": np.float64(realized["terminal_cost"]),
            }
        )
        metric = float(realized["J"])
        metric_name = "J_realized"
    else:
        metric = _native_lqr_cost(problem, trajectory, terminal_parameter)
        metric_name = "J_native_left_rule"
        payload[metric_name] = np.float64(metric)
    np.savez_compressed(path, **payload)
    return {
        metric_name: metric,
        "nonunique_count": int(trajectory["nonunique_count"]),
        "domain_exit": bool(trajectory["domain_exit"]),
    }


def run_experiment(
    problem: ControlProblem,
    train_config: TrainConfig,
    experiment: ExperimentConfig,
    *,
    paper_pdf: Path | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, object]:
    """Train all predeclared seeds and save every outer-iteration candidate."""

    experiment_started = time.perf_counter()
    experiment.output_dir.mkdir(parents=True, exist_ok=True)
    seed_records: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for seed in experiment.seeds:
        seed_started = time.perf_counter()
        seed_dir = experiment.output_dir / f"seed_{seed}"

        def progress(row: dict[str, float | int]) -> None:
            print(
                f"[{problem.name} seed={seed} outer={row['outer']} step={row['step']}] "
                f"loss={row['loss']:.6g} pde={row['pde_mse']:.6g} terminal={row['terminal_mse']:.6g}",
                flush=True,
            )

        result = train_policy_iteration(
            problem,
            experiment.terminal_parameter_family,
            train_config,
            seed=seed,
            output_dir=seed_dir,
            initial_control=np.asarray(experiment.initial_control, dtype=np.float64),
            progress=progress,
        )
        history_path = seed_dir / "history.csv"
        _write_history(history_path, result.history)
        candidate_records: list[dict[str, object]] = []
        final_solution_path: Path | None = None
        for checkpoint_path in result.checkpoints:
            checkpoint = torch.load(checkpoint_path, map_location=train_config.device, weights_only=False)
            model = build_model(problem, train_config).to(
                device=torch.device(train_config.device),
                dtype=torch.float64 if train_config.dtype == "float64" else torch.float32,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            outer = int(checkpoint["outer"])
            trajectory = feedback_rollout(
                problem,
                model,
                result.sensor_states,
                experiment.target_terminal_parameter,
                np.asarray(experiment.initial_state, dtype=np.float64),
                h=train_config.h,
                tie_tolerance=train_config.tie_tolerance,
                intervals=experiment.evaluation_intervals,
            )
            solution_path = seed_dir / f"solution_outer_{outer:03d}.npz"
            metrics = _save_solution(
                solution_path,
                problem,
                trajectory,
                seed=seed,
                outer=outer,
                terminal_parameter=experiment.target_terminal_parameter,
            )
            candidate_records.append(
                {
                    "outer": outer,
                    "checkpoint": checkpoint_path.name,
                    "checkpoint_sha256": file_sha256(checkpoint_path),
                    "solution": solution_path.name,
                    "solution_sha256": file_sha256(solution_path),
                    **metrics,
                }
            )
            final_solution_path = solution_path
        if final_solution_path is None:
            raise RuntimeError("training produced no checkpoint")
        canonical_solution = seed_dir / "solution.npz"
        shutil.copyfile(final_solution_path, canonical_solution)
        seed_records.append(
            {
                "seed": seed,
                "wall_time_seconds": time.perf_counter() - seed_started,
                "history": history_path.name,
                "history_sha256": file_sha256(history_path),
                "selection_rule": "final_predeclared_outer_iteration",
                "selected_solution": canonical_solution.name,
                "selected_solution_sha256": file_sha256(canonical_solution),
                "candidates": candidate_records,
            }
        )
        summary_rows.append({"seed": seed, **candidate_records[-1]})

    summary_path = experiment.output_dir / "summary.csv"
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    metric_name = "J_realized" if isinstance(problem, TumorAdaptation) else "J_native_left_rule"
    selected_values = np.asarray([float(row[metric_name]) for row in summary_rows], dtype=np.float64)
    aggregate = {
        "metric": metric_name,
        "count": int(selected_values.size),
        "mean": float(selected_values.mean()),
        "sample_std": float(selected_values.std(ddof=1)) if selected_values.size > 1 else 0.0,
        "min": float(selected_values.min()),
        "max": float(selected_values.max()),
    }

    manifest: dict[str, object] = {
        "schema_version": 1,
        "method": "Hamilton-Jacobi based policy iteration via PI-DeepONet",
        "citation": "Lee and Kim, arXiv:2406.10920",
        "paper_equations": ["2.3", "2.4", "2.5", "2.6"],
        "algorithm": "Algorithm 1",
        "upstream": {
            "kind": "paper specification",
            "author_code_used": False,
            "repository_revision": None,
        },
        "experiment_label": (
            "original-paper benchmark implementation"
            if not isinstance(problem, TumorAdaptation)
            else "tumor adaptation"
        ),
        "claim_scope": (
            "paper-specified equations, Algorithm 1, LQR matrices, terminal family, h, N, and M; "
            "not an exact numerical reproduction of the published figures because the paper and source archive "
            "do not provide author code or the omitted training hyperparameters"
            if not isinstance(problem, TumorAdaptation)
            else "paper-specified PI-DeepONet mechanism adapted to the separately declared tumor problem"
        ),
        "fidelity_invariants": {
            "spatial_gradient": "central finite difference nabla^h",
            "laplacian": "central discrete Delta^h",
            "artificial_viscosity": "N h Delta^h",
            "monotonicity_check": "N >= max(1, ||f||_inf/2)",
            "policy_improvement": "exact pointwise Hamiltonian argmin",
            "policy_frozen_during_evaluation": True,
            "terminal_function_usage": "all declared functions at every Adam step",
            "deeponet_output": "unnormalized branch/trunk inner product (paper Eq. 2.6)",
            "gradient_clipping": (
                "disabled (not reported in paper Algorithm 1)"
                if train_config.gradient_clip is None
                else f"enabled ablation at norm {train_config.gradient_clip}"
            ),
            "optimizer": (
                "Adam as specified by Algorithm 1; learning rate, Adam moments across policy iterations, "
                "and number of steps are implementation choices because they are not reported"
            ),
            "paper_pseudocode_resolution": (
                "minimize the positive mean of squared pointwise Eq. (2.3) residuals, including L; "
                "Algorithm 1's displayed L1 omits L, places the sample sum inside the square, and writes "
                "Adam(-alpha1 L1-alpha2 L2), which is inconsistent with Eq. (2.3) and loss minimization"
            ),
            "direct_reference_used": False,
            "terminal_function_family": list(experiment.terminal_parameter_family),
        },
        "problem": problem.metadata(),
        "train_config": train_config.to_dict(),
        "experiment": {
            "seeds": list(experiment.seeds),
            "target_terminal_parameter": experiment.target_terminal_parameter,
            "initial_control": list(experiment.initial_control),
            "initial_state": list(experiment.initial_state),
            "evaluation_intervals": experiment.evaluation_intervals,
            "selection_rule": "final_predeclared_outer_iteration",
            "stopping_rule": "fixed_predeclared_outer_iterations_and_steps; no early stopping",
            "selection_uses_nominal_realized_objective": False,
            "paper_omitted_hyperparameters": [
                "branch/trunk widths and depths",
                "activation, initialization, and input/value scaling",
                "latent dimension and sensor locations/count",
                "state/time collocation distribution and batch sizes",
                "Adam learning rate and whether optimizer moments persist across policy iterations",
                "number of Adam steps and convergence tolerance",
                "loss weights alpha_1 and alpha_2",
            ],
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "command": list(command if command is not None else sys.argv),
        },
        "wall_time_seconds": time.perf_counter() - experiment_started,
        "paper_pdf": None if paper_pdf is None else {"path": str(paper_pdf), "sha256": file_sha256(paper_pdf)},
        "summary": aggregate,
        "summary_csv": {"path": summary_path.name, "sha256": file_sha256(summary_path)},
        "seeds": seed_records,
    }
    manifest_path = experiment.output_dir / "manifest.json"
    with manifest_path.open("w") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest
