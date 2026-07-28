#!/usr/bin/env python3
"""Run one alpha/beta/gamma objective-weight experiment.

The script trains the three methods used in the report:

* the teacher-free time-only Transformer;
* feedback Case 1 (CF); and
* feedback Case 2 (DER).

For numerical conditioning, training minimizes the same positive rescaling of
the physical objective in every term.  With ``s = gamma / 20`` the training
coefficients are ``(alpha, beta, gamma) / s``.  This leaves the minimizer and
all first-order KKT zeros unchanged.  Final checkpoints are rebound to the
unscaled physical coefficients before diagnostics and reporting.

This is a new, standalone experiment tree.  It never overwrites an existing
report PDF or a completed run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"
SOURCE_TIME = (
    ROOT
    / "outputs/teacher_free_n800_strict_20260720/"
    "network_lbfgs_after_learn_tau/selected_checkpoint.pt"
)
DEFAULT_ROOT = ROOT / "outputs/new_objective_alpha1_grid_20260721_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_command(
    command: list[str],
    *,
    expected: Path,
    command_log: list[dict[str, Any]],
) -> None:
    if expected.exists():
        print(f"[skip] {expected}", flush=True)
        command_log.append(
            {"command": command, "status": "skipped_existing", "expected": str(expected)}
        )
        return
    started = time.perf_counter()
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    elapsed = time.perf_counter() - started
    if not expected.exists():
        raise FileNotFoundError(f"command completed without expected artifact: {expected}")
    command_log.append(
        {
            "command": command,
            "status": "completed",
            "elapsed_seconds": elapsed,
            "expected": str(expected),
        }
    )


def physical_problem(
    beta: float,
    gamma: float,
    *,
    alpha: float = 1.0,
) -> dict[str, Any]:
    return {
        "T": 10.0,
        "n": 800,
        "m": 21,
        "umax": 3.0,
        "beta": float(beta),
        "alpha": float(alpha),
        "gamma": float(gamma),
        "n0": 10.0,
        "m_suppression": 0.5,
    }


def normalized_problem(
    beta: float,
    gamma: float,
    *,
    alpha: float = 1.0,
) -> tuple[dict[str, Any], float]:
    scale = float(gamma) / 20.0
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("gamma must be positive and finite")
    problem = physical_problem(beta, gamma, alpha=alpha)
    problem.update(
        {
            "beta": float(beta) / scale,
            "alpha": float(alpha) / scale,
            "gamma": float(gamma) / scale,
        }
    )
    return problem, scale


def prepare_time_start(
    destination: Path,
    normalized: dict[str, Any],
    physical: dict[str, Any],
    scale: float,
) -> None:
    if destination.exists():
        return
    source = torch.load(SOURCE_TIME, map_location="cpu", weights_only=False)
    source_state = dict(source["model_state"])
    base_state = {
        key.removeprefix("base."): value.detach().cpu().clone()
        for key, value in source_state.items()
        if key.startswith("base.")
    }
    if not base_state:
        raise ValueError("source time checkpoint has no wrapped base-model parameters")
    base_args = dict(source["base_model_args"])
    for key in ("n", "alpha", "beta", "gamma"):
        base_args[key] = normalized[key]
    payload = {
        "model_state": base_state,
        "args": base_args,
        "problem": normalized,
        "teacher_free": True,
        "method": "cross-objective warm start followed by complete retraining",
        "source_checkpoint": str(SOURCE_TIME),
        "source_checkpoint_sha256": sha256(SOURCE_TIME),
        "training_normalization": {
            "physical_problem": physical,
            "normalized_problem": normalized,
            "positive_objective_scale": scale,
            "identity": "J_training = J_physical / positive_objective_scale",
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def rebind_time_checkpoint(
    source: Path,
    destination: Path,
    normalized: dict[str, Any],
    physical: dict[str, Any],
    scale: float,
) -> None:
    if destination.exists():
        return
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("problem") != normalized:
        raise ValueError("normalized time checkpoint problem mismatch")
    rebound = dict(payload)
    normalized_metrics = {}
    for key in (
        "selection_metrics",
        "selection_residual_metrics",
        "post_selection_diagnostics",
    ):
        if key in rebound:
            normalized_metrics[key] = rebound.pop(key)
    rebound["problem"] = physical
    base_args = dict(rebound["base_model_args"])
    for key in ("n", "alpha", "beta", "gamma"):
        base_args[key] = physical[key]
    rebound["base_model_args"] = base_args
    rebound["training_normalization"] = {
        "normalized_problem": normalized,
        "physical_problem": physical,
        "positive_objective_scale": scale,
        "argmin_preserved": True,
    }
    rebound["normalized_training_metrics"] = normalized_metrics
    rebound["normalized_source_checkpoint"] = str(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rebound, destination)


def rebind_feedback_checkpoint(
    source: Path,
    destination: Path,
    physical_time_checkpoint: Path,
    normalized: dict[str, Any],
    physical: dict[str, Any],
    scale: float,
) -> None:
    if destination.exists():
        return
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("problem") != normalized:
        raise ValueError("normalized feedback checkpoint problem mismatch")
    rebound = dict(payload)
    normalized_metrics = {}
    for key in (
        "best_validation_metrics",
        "selection_metrics",
        "post_selection_diagnostics",
    ):
        if key in rebound:
            normalized_metrics[key] = rebound.pop(key)
    rebound["problem"] = physical
    checkpoint_args = dict(rebound["args"])
    for key in ("n", "alpha", "beta", "gamma"):
        checkpoint_args[key] = physical[key]
    checkpoint_args["time_checkpoint"] = str(physical_time_checkpoint)
    rebound["args"] = checkpoint_args
    rebound["time_checkpoint"] = str(physical_time_checkpoint)
    rebound["normalized_source_checkpoint"] = str(source)
    rebound["training_normalization"] = {
        "normalized_problem": normalized,
        "physical_problem": physical,
        "positive_objective_scale": scale,
        "argmin_preserved": True,
    }
    rebound["normalized_training_metrics"] = normalized_metrics
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rebound, destination)


def feedback_command(
    *,
    option: str,
    time_checkpoint: Path,
    output: Path,
    normalized: dict[str, Any],
    smoke: bool,
) -> list[str]:
    common = [
        str(PYTHON),
        "scripts/train_feedback_section5.py",
        "--option",
        option,
        "--loss_variant",
        "literal",
        "--time_checkpoint",
        str(time_checkpoint),
        "--n",
        "800",
        "--alpha",
        str(normalized["alpha"]),
        "--beta",
        str(normalized["beta"]),
        "--gamma",
        str(normalized["gamma"]),
        "--center_state_correction",
        "--state_mode",
        "feedback",
        "--training_integrator",
        "rk4",
        "--freeze_time_branch",
        "--train_radius",
        "0.10",
        "--training_sample_seed_base",
        "20260721",
        "--full_gradient_weight",
        "50",
        "--full_gradient_residual",
        "projected",
        "--full_gradient_projection_step",
        "1",
        "--full_gradient_ramp_epochs",
        "1" if smoke else "20",
        "--device",
        "cpu",
        "--out_dir",
        str(output),
    ]
    if option == "cf":
        common.extend(
            [
                "--state_feature_mode",
                "relative_nominal",
                "--epochs",
                "100" if not smoke else "3",
                "--batch_size",
                "12" if not smoke else "3",
                "--validation_size",
                "48" if not smoke else "4",
                "--validation_seed",
                "20260719",
                "--test_size",
                "32" if not smoke else "4",
                "--test_seed",
                "20260720",
                "--test_radii",
                "0.05,0.10,0.20",
                "--eval_every",
                "5" if not smoke else "1",
                "--lr",
                "1e-4",
                "--lr_patience",
                "12",
                "--smooth_weight",
                "0",
                "--smooth_second_weight",
                "0",
                "--smooth_max_weight",
                "0",
                "--full_gradient_max_weight",
                "5",
                "--full_gradient_max_tau",
                "0.1",
                "--seed",
                "21",
            ]
        )
    elif option == "der":
        common.extend(
            [
                "--state_feature_mode",
                "burden_composition",
                "--epochs",
                "80" if not smoke else "3",
                "--batch_size",
                "12" if not smoke else "3",
                "--validation_size",
                "48" if not smoke else "4",
                "--validation_seed",
                "20260719",
                "--test_size",
                "64" if not smoke else "4",
                "--test_seed",
                "20260720",
                "--test_radii",
                "0.05,0.10,0.20",
                "--eval_every",
                "5" if not smoke else "1",
                "--lr",
                "5e-5",
                "--lr_patience",
                "5",
                "--psi_scale",
                "0.1160686131",
                "--dot_scale",
                "0.07238542893",
                "--ddot_scale",
                "0.5160189033",
                "--B_scale",
                "5.344734186",
                "--auto_residual_scales",
                "--singular_loss_weight",
                "0.094995694",
                "--smooth_weight",
                "3",
                "--smooth_second_weight",
                "35",
                "--smooth_max_weight",
                "0.5",
                "--smooth_max_tau",
                "0.02",
                "--full_gradient_max_weight",
                "0.5",
                "--full_gradient_max_tau",
                "0.1",
                "--seed",
                "2",
            ]
        )
    else:
        raise ValueError(option)
    return common


def build(args: argparse.Namespace) -> None:
    for name in ("alpha", "beta", "gamma"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    physical = physical_problem(args.beta, args.gamma, alpha=args.alpha)
    normalized, scale = normalized_problem(
        args.beta,
        args.gamma,
        alpha=args.alpha,
    )
    tag = (
        f"a{scalar_tag(args.alpha)}_"
        f"b{scalar_tag(args.beta)}_g{scalar_tag(args.gamma)}"
    )
    setting_dir = args.root.expanduser().resolve() / tag
    setting_dir.mkdir(parents=True, exist_ok=True)
    mode_path = setting_dir / "run_mode.json"
    requested_mode = "smoke" if args.smoke else "formal"
    if mode_path.exists():
        existing_mode = json.loads(mode_path.read_text(encoding="utf-8")).get("mode")
        if existing_mode != requested_mode:
            raise RuntimeError(
                f"refusing to mix {requested_mode} artifacts with {existing_mode} run: "
                f"{setting_dir}"
            )
    else:
        write_json(mode_path, {"mode": requested_mode})
    command_log_path = setting_dir / "command_log.json"
    command_log: list[dict[str, Any]] = (
        json.loads(command_log_path.read_text(encoding="utf-8"))
        if command_log_path.exists()
        else []
    )
    write_json(
        setting_dir / "problem.json",
        {
            "tag": tag,
            "physical_problem": physical,
            "normalized_training_problem": normalized,
            "positive_objective_scale": scale,
            "normalization_identity": "J_training = J_physical / positive_objective_scale",
        },
    )

    time_root = setting_dir / "time_only"
    start = time_root / "normalized_start.pt"
    prepare_time_start(start, normalized, physical, scale)
    curriculum = time_root / "curriculum"
    curriculum_epochs = "8,6,4" if args.smoke else "160,120,100"
    run_command(
        [
            str(PYTHON),
            "scripts/train_teacher_free_resolution_curriculum.py",
            "--out-dir",
            str(curriculum),
            "--start-checkpoint",
            str(start),
            "--seed",
            "101",
            "--scale",
            "1.08",
            "--temperature",
            "0.6991330744962188",
            "--learn-temperature",
            "--epochs",
            curriculum_epochs,
            "--eval-every",
            "2" if args.smoke else "20",
            "--selection-guard",
            "residual-only",
        ],
        expected=curriculum / "COMPLETED.json",
        command_log=command_log,
    )

    kkt = time_root / "projected_kkt"
    run_command(
        [
            str(PYTHON),
            "scripts/continue_teacher_free_strict_kkt.py",
            "--checkpoint",
            str(curriculum / "stage_3_n800/selected_checkpoint.pt"),
            "--out-dir",
            str(kkt),
            "--seed",
            "701",
            "--epochs",
            "5" if args.smoke else "120",
            "--eval-every",
            "1" if args.smoke else "10",
            "--learning-rate",
            "1e-6" if args.smoke else "5e-7",
            "--final-learning-rate",
            "5e-7" if args.smoke else "5e-8",
            "--schedule",
            "cosine",
            "--mode",
            "detached",
            "--step-multiplier",
            "20",
            "--pnorm-weight",
            "3",
            "--p",
            "20",
            "--learn-temperature",
            "--max-early-width",
            "1.0",
            "--max-late-width",
            "1.0",
            "--max-plateau-percent",
            "100",
            "--min-upper-nodes",
            "1",
            "--selection-guard",
            "residual-only",
        ],
        expected=kkt / "COMPLETED.json",
        command_log=command_log,
    )

    linear = time_root / "linear_box"
    run_command(
        [
            str(PYTHON),
            "scripts/continue_teacher_free_linear_box_projection.py",
            "--checkpoint",
            str(kkt / "selected_checkpoint.pt"),
            "--out-dir",
            str(linear),
            "--seed",
            "731",
            "--initialization-method",
            "logit_affine",
            "--adam-epochs",
            "3" if args.smoke else "160",
            "--adam-learning-rate",
            "1e-6",
            "--adam-final-learning-rate",
            "1e-7",
            "--adam-eval-every",
            "1" if args.smoke else "10",
            "--adam-train-scope",
            "all",
            "--lbfgs-outer-steps",
            "2" if args.smoke else "18",
            "--lbfgs-inner-iterations",
            "2" if args.smoke else "10",
            "--lbfgs-learning-rate",
            "0.3",
            "--lbfgs-history-size",
            "30",
            "--lbfgs-train-scope",
            "all",
            "--p",
            "16",
            "--high-p-weight",
            "2",
            "--solver-mode",
            "detached_fixed_point",
            "--detached-step-multiplier",
            "20",
            "--width-tolerance",
            "0.2",
            "--variation-fraction-tolerance",
            "0.5",
            "--selection-guard",
            "residual-only",
        ],
        expected=linear / "COMPLETED.json",
        command_log=command_log,
    )

    linear_exact = time_root / "linear_box_exact"
    run_command(
        [
            str(PYTHON),
            "scripts/continue_teacher_free_linear_box_projection.py",
            "--checkpoint",
            str(linear / "selected_checkpoint.pt"),
            "--out-dir",
            str(linear_exact),
            "--seed",
            "733",
            "--adam-epochs",
            "2" if args.smoke else "80",
            "--adam-learning-rate",
            "2e-7",
            "--adam-final-learning-rate",
            "2e-8",
            "--adam-eval-every",
            "1" if args.smoke else "10",
            "--adam-train-scope",
            "all",
            "--lbfgs-outer-steps",
            "2" if args.smoke else "12",
            "--lbfgs-inner-iterations",
            "2" if args.smoke else "5",
            "--lbfgs-learning-rate",
            "0.2",
            "--lbfgs-history-size",
            "30",
            "--lbfgs-train-scope",
            "all",
            "--p",
            "16",
            "--high-p-weight",
            "2",
            "--solver-mode",
            "exact",
            "--selection-guard",
            "residual-only",
        ],
        expected=linear_exact / "COMPLETED.json",
        command_log=command_log,
    )

    time_physical = time_root / "selected_checkpoint_exact_physical.pt"
    rebind_time_checkpoint(
        linear_exact / "selected_checkpoint.pt",
        time_physical,
        normalized,
        physical,
        scale,
    )

    feedback_root = setting_dir / "feedback"
    cf_train = feedback_root / "case1_cf_exact_train"
    der_train = feedback_root / "case2_der_exact_train"
    run_command(
        feedback_command(
            option="cf",
            time_checkpoint=linear_exact / "selected_checkpoint.pt",
            output=cf_train,
            normalized=normalized,
            smoke=args.smoke,
        ),
        expected=cf_train / "best_feedback_section5_full_gradient.pt",
        command_log=command_log,
    )
    run_command(
        feedback_command(
            option="der",
            time_checkpoint=linear_exact / "selected_checkpoint.pt",
            output=der_train,
            normalized=normalized,
            smoke=args.smoke,
        ),
        expected=der_train / "best_feedback_section5_full_gradient.pt",
        command_log=command_log,
    )
    cf_physical = feedback_root / "case1_cf_exact_physical.pt"
    der_physical = feedback_root / "case2_der_exact_physical.pt"
    rebind_feedback_checkpoint(
        cf_train / "best_feedback_section5_full_gradient.pt",
        cf_physical,
        time_physical,
        normalized,
        physical,
        scale,
    )
    rebind_feedback_checkpoint(
        der_train / "best_feedback_section5_full_gradient.pt",
        der_physical,
        time_physical,
        normalized,
        physical,
        scale,
    )

    if not args.smoke:
        direct_root = setting_dir / "direct_nominal"
        initial = linear_exact / "selected_solution.npz"
        stages = (
            (200, direct_root / "n200", initial, False, 600, 2000, 1),
            (400, direct_root / "n400", direct_root / "n200/scale_1_direct_solution.npz", True, 800, 2500, 0),
            (800, direct_root / "n800", direct_root / "n400/scale_1_direct_solution.npz", True, 1200, 4000, 0),
            (800, direct_root / "n800_strict_final", direct_root / "n800/scale_1_direct_solution.npz", True, 2500, 6000, 0),
        )
        for n, output, initial_control, initial_only, maxiter, maxfun, random_starts in stages:
            command = [
                str(PYTHON),
                "run_direct_openloop_cost.py",
                "--out_dir",
                str(output),
                "--n",
                str(n),
                "--scales",
                "1",
                "--alpha",
                str(args.alpha),
                "--beta",
                str(args.beta),
                "--gamma",
                str(args.gamma),
                "--objective_scale",
                str(scale),
                "--initial_control",
                str(initial_control),
                "--random_starts",
                str(random_starts),
                "--maxiter",
                str(maxiter),
                "--maxfun",
                str(maxfun),
                "--ftol",
                "0" if n == 800 else "1e-12",
                "--gtol",
                "1e-12" if n == 800 else "1e-9",
            ]
            if initial_only:
                command.append("--initial_only")
            run_command(
                command,
                expected=output / "scale_1_direct_solution.npz",
                command_log=command_log,
            )

        diagnostics = setting_dir / "two_state_diagnostics"
        run_command(
            [
                str(PYTHON),
                "scripts/generate_two_state_three_case_main_figures.py",
                "--time-checkpoint",
                str(time_physical),
                "--cf-checkpoint",
                str(cf_physical),
                "--der-checkpoint",
                str(der_physical),
                "--out-dir",
                str(diagnostics),
                "--n",
                "800",
                "--kkt-tolerance",
                str(scale * 1.0e-4),
            ],
            expected=diagnostics / "metadata.json",
            command_log=command_log,
        )

    write_json(command_log_path, command_log)
    write_json(
        setting_dir / "COMPLETED.json",
        {
            "status": "smoke_completed" if args.smoke else "completed",
            "alpha": args.alpha,
            "beta": args.beta,
            "gamma": args.gamma,
            "physical_problem": physical,
            "normalized_training_problem": normalized,
            "positive_objective_scale": scale,
            "time_checkpoint": str(time_physical),
            "case1_checkpoint": str(cf_physical),
            "case2_checkpoint": str(der_physical),
            "direct_reference": (
                None
                if args.smoke
                else str(setting_dir / "direct_nominal/n800_strict_final/scale_1_direct_solution.npz")
            ),
            "diagnostics": None if args.smoke else str(setting_dir / "two_state_diagnostics"),
        },
    )
    print(f"completed {setting_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--smoke", action="store_true")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
