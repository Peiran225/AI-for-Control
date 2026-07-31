#!/usr/bin/env python3
"""Learned-dynamics reproduction of the paper's original LQR experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
from faithful_related_work.neural_pmp.lqr import (
    LQR_ACTION_HIGH,
    LQR_ACTION_LOW,
    LQR_DYNAMICS_EPOCHS,
    LQR_DYNAMICS_SAMPLES,
    LQR_HORIZON,
    LQR_PMP_ITERATIONS,
    LQR_PMP_LEARNING_RATE,
    LQR_PAPER_TEXT_SAMPLE_ACTION_HIGH,
    LQR_PAPER_TEXT_SAMPLE_ACTION_LOW,
    LQR_REPRODUCTION_RUNS,
    LQR_OFFICIAL_SAMPLE_ACTION_HIGH,
    LQR_OFFICIAL_SAMPLE_ACTION_LOW,
    LQR_STATE_HIGH,
    LQR_STATE_LOW,
    generate_lqr_dynamics_dataset,
    lqr_initial_control,
    original_lqr_components,
)
from faithful_related_work.neural_pmp.provenance import collect_upstream_provenance


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


def build_parser(profile: str = "full") -> argparse.ArgumentParser:
    if profile not in {"smoke", "full"}:
        raise ValueError("profile must be smoke or full")
    full = profile == "full"
    seeds = ",".join(str(seed) for seed in range(LQR_REPRODUCTION_RUNS if full else 2))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", default=seeds)
    parser.add_argument("--starts", default="zero")
    parser.add_argument("--canonical-seed", type=int, default=0)
    parser.add_argument(
        "--sampling-profile",
        choices=("official-code", "paper-text-sensitivity"),
        default="official-code",
        help="Dynamics-data action domain; controller bounds always remain +/-100000.",
    )
    parser.add_argument("--dynamics-train-samples", type=int, default=LQR_DYNAMICS_SAMPLES if full else 128)
    parser.add_argument("--dynamics-validation-samples", type=int, default=500 if full else 64)
    parser.add_argument("--dynamics-epochs", type=int, default=LQR_DYNAMICS_EPOCHS if full else 25)
    parser.add_argument("--dynamics-validation-interval", type=int, default=100 if full else 5)
    parser.add_argument("--dynamics-hidden", type=int, default=64)
    parser.add_argument("--control-iters", type=int, default=LQR_PMP_ITERATIONS if full else 20)
    parser.add_argument("--control-lr", type=float, default=LQR_PMP_LEARNING_RATE)
    parser.add_argument("--control-eval-interval", type=int, default=100 if full else 2)
    parser.add_argument("--control-patience", type=int, default=1_000_000)
    parser.add_argument("--projected-tolerance", type=float, default=1e-8)
    parser.set_defaults(profile=profile)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {out_dir}")
    dynamics_dir = out_dir / "dynamics"
    controls_dir = out_dir / "controls"
    datasets_dir = out_dir / "datasets"
    for directory in (out_dir, dynamics_dir, controls_dir, datasets_dir):
        directory.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    upstream_provenance = collect_upstream_provenance(repo_root)

    seeds = _parse_ints(args.seeds)
    starts = _parse_strings(args.starts)
    if not seeds or not starts:
        raise ValueError("at least one seed and start are required")
    if args.canonical_seed not in seeds:
        raise ValueError("--canonical-seed must be included in --seeds")
    if args.sampling_profile == "official-code":
        sample_action_low = LQR_OFFICIAL_SAMPLE_ACTION_LOW
        sample_action_high = LQR_OFFICIAL_SAMPLE_ACTION_HIGH
    else:
        sample_action_low = LQR_PAPER_TEXT_SAMPLE_ACTION_LOW
        sample_action_high = LQR_PAPER_TEXT_SAMPLE_ACTION_HIGH

    dtype = torch.float32
    _, original_initial_state, stage_cost, terminal_cost = original_lqr_components(dtype=dtype)
    run_records: list[dict[str, Any]] = []
    history_records: list[dict[str, Any]] = []
    control_arrays: list[np.ndarray] = []

    for seed in seeds:
        seed_everything(seed)
        train_x, train_y = generate_lqr_dynamics_dataset(
            samples=args.dynamics_train_samples,
            seed=seed * 2 + 101,
            action_sample_low=sample_action_low,
            action_sample_high=sample_action_high,
        )
        validation_x, validation_y = generate_lqr_dynamics_dataset(
            samples=args.dynamics_validation_samples,
            seed=seed * 2 + 102,
            action_sample_low=sample_action_low,
            action_sample_high=sample_action_high,
        )
        dataset_path = datasets_dir / f"lqr_dynamics_seed_{seed}.npz"
        np.savez(
            dataset_path,
            train_inputs=train_x,
            train_targets=train_y,
            validation_inputs=validation_x,
            validation_targets=validation_y,
            state_domain=np.array([LQR_STATE_LOW, LQR_STATE_HIGH]),
            dynamics_sample_action_domain=np.array([sample_action_low, sample_action_high]),
            controller_action_bounds=np.array([LQR_ACTION_LOW, LQR_ACTION_HIGH]),
            sampling_profile=np.array(args.sampling_profile),
        )
        training_config = DynamicsTrainingConfig(
            epochs=args.dynamics_epochs,
            learning_rate=5e-3,
            weight_decay=1e-4,
            validation_interval=args.dynamics_validation_interval,
            validation_patience=1_000_000,
        )
        trained = train_dynamics_model(
            train_inputs=train_x,
            train_targets=train_y,
            validation_inputs=validation_x,
            validation_targets=validation_y,
            state_dim=5,
            action_dim=3,
            hidden_dim=args.dynamics_hidden,
            seed=seed,
            config=training_config,
            dtype=dtype,
            checkpoint_policy="fixed_final",
        )
        model = trained.model
        checkpoint_path = dynamics_dir / f"lqr_dynamics_seed_{seed}.pt"
        torch.save(
            {
                "paper": "arXiv:2212.14566 Appendix C LQR",
                "upstream_nested_head": upstream_provenance["nested_head"],
                "upstream_env_head_blob_sha256": upstream_provenance["files"]["env"]["head_blob_sha256"],
                "upstream_solver_head_blob_sha256": upstream_provenance["files"]["solver"]["head_blob_sha256"],
                "independent_runner_does_not_execute_upstream_worktree": True,
                "seed": seed,
                "state_domain": [LQR_STATE_LOW, LQR_STATE_HIGH],
                "sampling_profile": args.sampling_profile,
                "dynamics_sample_state_domain": [LQR_STATE_LOW, LQR_STATE_HIGH],
                "dynamics_sample_action_domain": [sample_action_low, sample_action_high],
                "controller_action_bounds": [LQR_ACTION_LOW, LQR_ACTION_HIGH],
                "paper_code_sampling_conflict": {
                    "official_code_dynamics_sample_action_domain": [LQR_OFFICIAL_SAMPLE_ACTION_LOW, LQR_OFFICIAL_SAMPLE_ACTION_HIGH],
                    "appendix_c_text_dynamics_sample_action_domain": [LQR_PAPER_TEXT_SAMPLE_ACTION_LOW, LQR_PAPER_TEXT_SAMPLE_ACTION_HIGH],
                },
                "dataset_sha256": array_sha256(train_x, train_y, validation_x, validation_y),
                "model_config": {"state_dim": 5, "action_dim": 3, "hidden_dim": args.dynamics_hidden, "hidden_layers": 2},
                "training_config": asdict(training_config),
                "checkpoint_policy": trained.checkpoint_policy,
                "selected_epoch": trained.selected_epoch,
                "best_validation_epoch_diagnostic_only": trained.best_epoch,
                "best_validation_mse_diagnostic_only": trained.best_validation_mse,
                "stop_reason": trained.stop_reason,
                "state_dict": model.state_dict(),
            },
            checkpoint_path,
        )
        dynamics_hash = _sha256(checkpoint_path)
        _write_csv(
            dynamics_dir / f"lqr_dynamics_seed_{seed}_history.csv",
            [dict(seed=seed, **row) for row in trained.history],
        )

        for start_index, start in enumerate(starts):
            start_seed = seed * 1_000_003 + start_index * 10_007 + 17
            control0 = lqr_initial_control(start, seed=start_seed, dtype=dtype)
            result = solve_neural_pmp(
                dynamics=model,
                initial_state=original_initial_state,
                initial_control=control0,
                stage_cost=stage_cost,
                terminal_cost=terminal_cost,
                action_lower=LQR_ACTION_LOW,
                action_upper=LQR_ACTION_HIGH,
                learning_rate=args.control_lr,
                max_iterations=args.control_iters,
                validation_initial_states=[original_initial_state],
                evaluation_interval=args.control_eval_interval,
                projected_tolerance=args.projected_tolerance,
                patience_evaluations=args.control_patience,
                return_policy="fixed_final",
            )
            control = result.control.detach().cpu().numpy()
            control_arrays.append(control)
            validation_value = float(objective(result.states, result.control, stage_cost, terminal_cost))
            control_path = controls_dir / f"lqr_control_seed_{seed}_start_{start}.npz"
            np.savez(
                control_path,
                t=np.arange(LQR_HORIZON + 1, dtype=np.float64),
                u=control,
                learned_states=result.states.detach().cpu().numpy(),
                learned_costates=result.costates.detach().cpu().numpy(),
                raw_hamiltonian_gradient=result.raw_gradient.detach().cpu().numpy(),
                dynamics_checkpoint_sha256=dynamics_hash,
                learned_objective_diagnostic=validation_value,
                best_iteration=result.best_iteration,
                stop_reason=result.stop_reason,
            )
            record = {
                "run_index": len(run_records),
                "seed": seed,
                "start": start,
                "dynamics_checkpoint": str(checkpoint_path.relative_to(out_dir)),
                "dynamics_checkpoint_sha256": dynamics_hash,
                "selection_metric": "none_fixed_final",
                "learned_objective_diagnostic": validation_value,
                "control_return_policy": "fixed_final",
                "best_iteration": result.best_iteration,
                "stop_reason": result.stop_reason,
                "control_checkpoint": str(control_path.relative_to(out_dir)),
            }
            run_records.append(record)
            for row in result.history:
                history_records.append({"run_index": record["run_index"], "seed": seed, "start": start, **row})

    # The official default has one zero start. If additional sensitivity starts
    # are requested, retain them all but use the first predeclared start rather
    # than an objective-based choice. No true objective has been computed yet.
    selected_by_seed: dict[int, int] = {}
    for seed in seeds:
        candidates = [index for index, row in enumerate(run_records) if int(row["seed"]) == seed]
        selected_by_seed[seed] = candidates[0]
        run_records[selected_by_seed[seed]]["selected_within_seed"] = True
    canonical_index = selected_by_seed[args.canonical_seed]
    selection = {
        "selected_run_index_by_seed": {str(seed): index for seed, index in selected_by_seed.items()},
        "rule": "fixed final checkpoint and final control; first predeclared start per seed; canonical seed fixed before execution",
        "within_seed_rule": "first predeclared start; no objective selection",
        "dynamics_checkpoint_policy": "fixed_final",
        "control_return_policy": "fixed_final",
        "uses_validation_for_selection": False,
        "cross_seed_performance_selection": False,
        "canonical_seed": args.canonical_seed,
        "canonical_run_index": canonical_index,
        "true_objective_available_to_selection": False,
    }

    # Only now evaluate the selected controls under the true LQR dynamics.
    true_dynamics, true_initial, true_stage_cost, true_terminal_cost = original_lqr_components(dtype=torch.float64)
    selected_true_values: list[float] = []
    for seed in seeds:
        index = selected_by_seed[seed]
        control = torch.as_tensor(control_arrays[index], dtype=torch.float64)
        states = rollout(true_dynamics, true_initial, control)
        true_value = float(objective(states, control, true_stage_cost, true_terminal_cost))
        run_records[index]["true_objective_post_selection"] = true_value
        selected_true_values.append(true_value)

    canonical_control = control_arrays[canonical_index]
    np.savez(
        out_dir / "canonical_control.npz",
        t=np.arange(LQR_HORIZON + 1, dtype=np.float64),
        u=canonical_control,
        canonical_seed=args.canonical_seed,
        canonical_run_index=canonical_index,
        true_objective_post_selection=float(run_records[canonical_index]["true_objective_post_selection"]),
    )
    aggregate = {
        "runs": len(seeds),
        "seeds": seeds,
        "mean_true_objective_post_selection": float(np.mean(selected_true_values)),
        "sample_std_true_objective_post_selection": float(np.std(selected_true_values, ddof=1)) if len(seeds) > 1 else 0.0,
        "minimum_true_objective_diagnostic_only": float(np.min(selected_true_values)),
        "maximum_true_objective_diagnostic_only": float(np.max(selected_true_values)),
        "cross_seed_performance_selection": False,
    }
    _write_csv(out_dir / "all_runs.csv", run_records)
    _write_csv(out_dir / "history.csv", history_records)
    summary = {
        "paper": "Pontryagin Optimal Control via Neural Networks, arXiv:2212.14566",
        "experiment": "Appendix C learned-dynamics LQR",
        "profile": args.profile,
        "paper_defaults": {
            "horizon": LQR_HORIZON,
            "dynamics_samples": LQR_DYNAMICS_SAMPLES,
            "dynamics_epochs": LQR_DYNAMICS_EPOCHS,
            "pmp_iterations": LQR_PMP_ITERATIONS,
            "pmp_learning_rate": LQR_PMP_LEARNING_RATE,
            "reproduction_runs": LQR_REPRODUCTION_RUNS,
            "dynamics_sample_state_domain": [LQR_STATE_LOW, LQR_STATE_HIGH],
            "official_code_dynamics_sample_action_domain": [LQR_OFFICIAL_SAMPLE_ACTION_LOW, LQR_OFFICIAL_SAMPLE_ACTION_HIGH],
            "appendix_c_text_dynamics_sample_action_domain": [LQR_PAPER_TEXT_SAMPLE_ACTION_LOW, LQR_PAPER_TEXT_SAMPLE_ACTION_HIGH],
            "controller_action_bounds": [LQR_ACTION_LOW, LQR_ACTION_HIGH],
            "default_sampling_profile": "official-code",
            "dynamics_checkpoint_policy": "fixed_final",
            "control_return_policy": "fixed_final",
            "dynamics_hidden_layers": 2,
            "dynamics_activation": "ReLU",
        },
        "arguments": vars(args),
        "upstream_provenance": upstream_provenance,
        "selection": selection,
        "aggregate": aggregate,
    }
    with (out_dir / "run_config_and_summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    hashes = {
        str(path.relative_to(out_dir)): _sha256(path)
        for path in sorted(out_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }
    with (out_dir / "manifest.json").open("w") as stream:
        json.dump(
            {
                "schema_version": 1,
                "paper": "arXiv:2212.14566 Appendix C LQR",
                "upstream_provenance": upstream_provenance,
                "sampling_profile": args.sampling_profile,
                "dynamics_sample_action_domain": [sample_action_low, sample_action_high],
                "controller_action_bounds": [LQR_ACTION_LOW, LQR_ACTION_HIGH],
                "paper_code_sampling_conflict_recorded": True,
                "dynamics_checkpoint_policy": "fixed_final",
                "control_return_policy": "fixed_final",
                "uses_validation_for_selection": False,
                "selection_frozen_before_post_selection_diagnostics": True,
                "true_objective_available_to_selection": False,
                "cross_seed_performance_selection": False,
                "canonical_seed": args.canonical_seed,
                "selected_run_index_by_seed": selection["selected_run_index_by_seed"],
                "artifacts": hashes,
            },
            stream,
            indent=2,
            sort_keys=True,
        )
    return summary


def main(profile: str = "full") -> None:
    summary = run(build_parser(profile).parse_args())
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main("full")
