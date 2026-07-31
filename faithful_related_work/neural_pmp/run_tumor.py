#!/usr/bin/env python3
"""Run the faithful Neural-PMP adaptation on the tumor problem.

Selection protocol (fixed before execution): a run can either use one
predeclared start or select a start separately within each seed using the
controller-dynamics validation objective. Seeds are never ranked by
performance. The canonical artifact uses a predeclared fixed seed. True and
realized objectives are computed only after all selections are frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from faithful_related_work.neural_pmp.core import objective, rollout, solve_neural_pmp
from faithful_related_work.neural_pmp.dynamics import (
    DynamicsTrainingConfig,
    array_sha256,
    seed_everything,
    train_dynamics_model,
)
from faithful_related_work.neural_pmp.tumor import (
    ExactTumorEulerMap,
    TumorConfig,
    generate_dynamics_dataset,
    initial_control,
    make_costs,
    true_discrete_objective,
    validation_initial_states,
)
from tumor_problem import TumorProblem, evaluate_zoh_control, serializable_metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _parse_ints(text: str) -> list[int]:
    return [int(value.strip()) for value in text.split(",") if value.strip()]


def _parse_strings(text: str) -> list[str]:
    return [value.strip() for value in text.split(",") if value.strip()]


def _validation_score(model, states, control, stage_cost, terminal_cost) -> float:
    with torch.no_grad():
        values = [
            objective(rollout(model, state, control), control, stage_cost, terminal_cost)
            for state in states
        ]
    return float(torch.stack(values).mean())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="paper_runs/faithful_neural_pmp_smoke")
    parser.add_argument("--dynamics-mode", choices=("learned", "exact"), default="learned")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--control-device",
        default=None,
        help="device for the explicit PMP updates; defaults to --device",
    )
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--starts", default="zero,mid,front,back,random")
    parser.add_argument("--control-lr", type=float, default=1e-3)
    parser.add_argument("--control-iters", type=int, default=3000)
    parser.add_argument("--control-eval-interval", type=int, default=25)
    parser.add_argument("--control-patience", type=int, default=80)
    parser.add_argument("--projected-tolerance", type=float, default=1e-8)
    parser.add_argument("--dynamics-train-samples", type=int, default=2000)
    parser.add_argument("--dynamics-validation-samples", type=int, default=500)
    parser.add_argument("--dynamics-epochs", type=int, default=50_000)
    parser.add_argument("--dynamics-validation-interval", type=int, default=100)
    parser.add_argument("--dynamics-patience", type=int, default=100)
    parser.add_argument("--dynamics-hidden", type=int, default=128)
    parser.add_argument("--dynamics-state-low", type=float, default=0.1)
    parser.add_argument("--dynamics-state-high", type=float, default=20.0)
    parser.add_argument(
        "--dynamics-checkpoint-policy",
        choices=("best_validation", "fixed_final"),
        default="best_validation",
    )
    parser.add_argument(
        "--control-return-policy",
        choices=("best_validation", "fixed_final"),
        default="best_validation",
    )
    parser.add_argument("--selection-seed", type=int, default=20260712)
    parser.add_argument("--selection-states", type=int, default=8)
    parser.add_argument("--canonical-seed", type=int, default=0)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--m", type=int, default=21)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=20.0)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_started = time.perf_counter()
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {out_dir}")
    dynamics_dir = out_dir / "dynamics"
    controls_dir = out_dir / "controls"
    datasets_dir = out_dir / "datasets"
    for directory in (out_dir, dynamics_dir, controls_dir, datasets_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if args.alpha <= 0.0 or args.beta <= 0.0 or args.gamma <= 0.0:
        raise ValueError("alpha, beta, and gamma must be positive")
    device = torch.device(args.device)
    control_device = torch.device(args.control_device or args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if control_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for control updates but is not available")
    tumor = TumorConfig(
        final_time=args.T,
        intervals=args.n,
        state_dim=args.m,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
    )
    dtype = torch.float32 if args.dynamics_mode == "learned" else torch.float64
    stage_cost, terminal_cost = make_costs(tumor, dtype=dtype)
    fixed_validation_states = validation_initial_states(
        tumor,
        count=args.selection_states,
        seed=args.selection_seed,
        dtype=dtype,
    )
    fixed_validation_states = [state.to(control_device) for state in fixed_validation_states]
    validation_array = np.stack([state.detach().cpu().numpy() for state in fixed_validation_states])
    np.savez(datasets_dir / "control_validation_initial_states.npz", states=validation_array)

    seeds = _parse_ints(args.seeds)
    starts = _parse_strings(args.starts)
    if not seeds or not starts:
        raise ValueError("at least one seed and start are required")
    if args.canonical_seed not in seeds:
        raise ValueError("--canonical-seed must be included in --seeds")

    run_records: list[dict[str, Any]] = []
    history_records: list[dict[str, Any]] = []
    controls: list[np.ndarray] = []
    for seed in seeds:
        seed_everything(seed)
        checkpoint_path = dynamics_dir / f"dynamics_seed_{seed}.pt"
        dynamics_history: list[dict[str, Any]] = []
        if args.dynamics_mode == "learned":
            train_x, train_y = generate_dynamics_dataset(
                tumor,
                samples=args.dynamics_train_samples,
                seed=seed * 2 + 101,
                state_low=args.dynamics_state_low,
                state_high=args.dynamics_state_high,
            )
            validation_x, validation_y = generate_dynamics_dataset(
                tumor,
                samples=args.dynamics_validation_samples,
                seed=seed * 2 + 102,
                state_low=args.dynamics_state_low,
                state_high=args.dynamics_state_high,
            )
            dataset_path = datasets_dir / f"dynamics_seed_{seed}.npz"
            np.savez(
                dataset_path,
                train_inputs=train_x,
                train_targets=train_y,
                validation_inputs=validation_x,
                validation_targets=validation_y,
            )
            training_config = DynamicsTrainingConfig(
                epochs=args.dynamics_epochs,
                validation_interval=args.dynamics_validation_interval,
                validation_patience=args.dynamics_patience,
            )
            trained = train_dynamics_model(
                train_inputs=train_x,
                train_targets=train_y,
                validation_inputs=validation_x,
                validation_targets=validation_y,
                state_dim=tumor.state_dim,
                action_dim=1,
                hidden_dim=args.dynamics_hidden,
                seed=seed,
                config=training_config,
                dtype=dtype,
                checkpoint_policy=args.dynamics_checkpoint_policy,
                device=device,
            )
            model = trained.model
            dynamics_history = [dict(seed=seed, **row) for row in trained.history]
            checkpoint = {
                "paper": "arXiv:2212.14566",
                "mode": "learned",
                "seed": seed,
                "tumor_config": tumor.to_dict(),
                "model_config": {
                    "state_dim": tumor.state_dim,
                    "action_dim": 1,
                    "hidden_dim": args.dynamics_hidden,
                    "dtype": str(dtype),
                },
                "training_config": asdict(training_config),
                "checkpoint_policy": trained.checkpoint_policy,
                "dataset_sha256": array_sha256(train_x, train_y, validation_x, validation_y),
                "best_epoch": trained.best_epoch,
                "selected_epoch": trained.selected_epoch,
                "best_validation_mse": trained.best_validation_mse,
                "stop_reason": trained.stop_reason,
                "state_dict": {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                },
            }
        else:
            model = ExactTumorEulerMap(tumor, dtype=dtype).to(control_device)
            checkpoint = {
                "paper": "arXiv:2212.14566",
                "mode": "exact_known_dynamics",
                "seed": seed,
                "tumor_config": tumor.to_dict(),
                "state_dict": {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                },
            }
        torch.save(checkpoint, checkpoint_path)
        model = model.to(control_device)
        dynamics_hash = _sha256(checkpoint_path)
        if dynamics_history:
            _write_csv(dynamics_dir / f"dynamics_seed_{seed}_history.csv", dynamics_history)

        # The same in-memory model and the same checkpoint hash are used for
        # every start associated with this seed.
        for start_index, start in enumerate(starts):
            start_seed = seed * 1_000_003 + start_index * 10_007 + 17
            control0 = initial_control(tumor, start, seed=start_seed, dtype=dtype).to(control_device)
            initial_state = torch.full(
                (tumor.state_dim,),
                tumor.initial_state,
                dtype=dtype,
                device=control_device,
            )
            result = solve_neural_pmp(
                dynamics=model,
                initial_state=initial_state,
                initial_control=control0,
                stage_cost=stage_cost,
                terminal_cost=terminal_cost,
                action_lower=tumor.action_lower,
                action_upper=tumor.action_upper,
                learning_rate=args.control_lr,
                max_iterations=args.control_iters,
                validation_initial_states=fixed_validation_states,
                evaluation_interval=args.control_eval_interval,
                projected_tolerance=args.projected_tolerance,
                patience_evaluations=args.control_patience,
                return_policy=args.control_return_policy,
            )
            selected_validation = _validation_score(
                model, fixed_validation_states, result.control, stage_cost, terminal_cost
            )
            control_array = result.control.detach().cpu().numpy().reshape(tumor.intervals)
            controls.append(control_array)
            control_path = controls_dir / f"control_seed_{seed}_start_{start}.npz"
            np.savez(
                control_path,
                t=np.linspace(0.0, tumor.final_time, tumor.intervals + 1),
                u=control_array,
                learned_states=result.states.detach().cpu().numpy(),
                learned_costates=result.costates.detach().cpu().numpy(),
                raw_hamiltonian_gradient=result.raw_gradient.detach().cpu().numpy(),
                dynamics_checkpoint_sha256=dynamics_hash,
                validation_objective=selected_validation,
                best_iteration=result.best_iteration,
                stop_reason=result.stop_reason,
            )
            record = {
                "run_index": len(run_records),
                "seed": seed,
                "start": start,
                "dynamics_checkpoint": str(checkpoint_path.relative_to(out_dir)),
                "dynamics_checkpoint_sha256": dynamics_hash,
                "selection_metric": (
                    "predeclared_single_start; validation_objective_diagnostic_only"
                    if len(starts) == 1
                    else f"{args.dynamics_mode}_dynamics_validation_objective"
                ),
                "selection_value": selected_validation,
                "best_iteration": result.best_iteration,
                "stop_reason": result.stop_reason,
                "control_checkpoint": str(control_path.relative_to(out_dir)),
                "control_min": float(control_array.min()),
                "control_max": float(control_array.max()),
            }
            run_records.append(record)
            for row in result.history:
                history_records.append(
                    {
                        "run_index": record["run_index"],
                        "seed": seed,
                        "start": start,
                        **row,
                    }
                )

    # Freeze one start independently within every seed.  There is deliberately
    # no cross-seed performance selection.
    selected_by_seed: dict[int, int] = {}
    for seed in seeds:
        candidates = [index for index, row in enumerate(run_records) if int(row["seed"]) == seed]
        if len(starts) == 1:
            selected_by_seed[seed] = candidates[0]
        else:
            selected_by_seed[seed] = min(
                candidates,
                key=lambda index: (
                    float(run_records[index]["selection_value"]),
                    str(run_records[index]["start"]),
                ),
            )
    for seed, index in selected_by_seed.items():
        run_records[index]["selected_within_seed"] = True
    selected_index = selected_by_seed[args.canonical_seed]
    single_start = len(starts) == 1
    selection_snapshot = {
        "selected_run_index_by_seed": {str(seed): index for seed, index in selected_by_seed.items()},
        "rule": (
            "predeclared single start within every seed; canonical seed fixed before execution"
            if single_start
            else "within each seed, minimum controller-dynamics validation objective; canonical seed fixed before execution"
        ),
        "within_seed_rule": (
            "predeclared single start; no performance selection"
            if single_start
            else "minimum controller-dynamics validation objective; ties by start"
        ),
        "cross_seed_performance_selection": False,
        "canonical_seed": args.canonical_seed,
        "canonical_run_index": selected_index,
        "true_or_realized_objective_available_to_rule": False,
    }

    # Post-selection diagnostics only.  They are saved but never used above.
    for index, control_array in enumerate(controls):
        run_records[index]["true_discrete_objective_post_selection"] = true_discrete_objective(tumor, control_array)

    selected = run_records[selected_index]
    selected_control = controls[selected_index]
    canonical_t = np.linspace(0.0, tumor.final_time, tumor.intervals + 1)
    canonical_problem = TumorProblem(
        T=tumor.final_time,
        m=tumor.state_dim,
        alpha=tumor.alpha,
        beta=tumor.beta,
        gamma=tumor.gamma,
    )
    realized = evaluate_zoh_control(
        canonical_t,
        selected_control,
        canonical_problem,
        include_diagnostics=False,
    )
    selected["canonical_realized_J_post_selection"] = float(realized["J"])
    np.savez(
        out_dir / "canonical_control.npz",
        t=canonical_t,
        u=selected_control,
        selected_run_index=selected_index,
        seed=int(selected["seed"]),
        start=str(selected["start"]),
        selection_value=float(selected["selection_value"]),
        canonical_realized_J=float(realized["J"]),
    )

    _write_csv(out_dir / "all_runs.csv", run_records)
    _write_csv(out_dir / "history.csv", history_records)
    configuration = {
        "paper": "Pontryagin Optimal Control via Neural Networks, arXiv:2212.14566",
        "algorithm_invariant": "raw Hamiltonian gradient -> gradient step -> action projection",
        "gradient_clipping": False,
        "tumor": tumor.to_dict(),
        "runner_arguments": vars(args),
        "selection": selection_snapshot,
        "selected_run": selected,
        "selected_realized_metrics": serializable_metrics(realized),
        "wall_time_seconds": time.perf_counter() - run_started,
    }
    with (out_dir / "run_config_and_summary.json").open("w") as stream:
        json.dump(configuration, stream, indent=2, sort_keys=True)

    artifact_hashes = {
        str(path.relative_to(out_dir)): _sha256(path)
        for path in sorted(out_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "schema_version": 1,
        "paper": "arXiv:2212.14566",
        "selection_frozen_before_post_selection_diagnostics": True,
        "cross_seed_performance_selection": False,
        "canonical_seed": args.canonical_seed,
        "canonical_run_index": selected_index,
        "artifacts": artifact_hashes,
    }
    with (out_dir / "manifest.json").open("w") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
    return configuration


def main() -> None:
    configuration = run(build_parser().parse_args())
    selected = configuration["selected_run"]
    print(json.dumps(selected, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
