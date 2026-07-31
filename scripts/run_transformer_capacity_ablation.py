#!/usr/bin/env python3
"""Run a width-only Transformer capacity ablation.

The experiment keeps the two-layer, four-head architecture, PMP/KKT
pretraining, resolution curriculum, and projected-gradient L-BFGS refinement
fixed.  Only ``d_model`` (and therefore the feed-forward width ``4*d_model``)
changes.  Direct-transcription artifacts and objective values are not read
during training or checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "outputs/transformer_capacity_ablation_20260724"


def parse_int_list(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("values must be unique")
    return result


def run_command(command: list[str], *, log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write(" ".join(command) + "\n\n")
        stream.flush()
        subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return time.perf_counter() - started


def run_one(
    *,
    experiment_root: Path,
    d_model: int,
    seed: int,
    pretrain_epochs: int,
    curriculum_epochs: str,
) -> dict[str, Any]:
    if d_model <= 0 or d_model % 4:
        raise ValueError(f"d_model must be a positive multiple of four: {d_model}")

    run_root = experiment_root / f"d{d_model}" / f"seed_{seed}"
    pretrain_dir = run_root / "pretrain"
    curriculum_dir = run_root / "curriculum"
    lbfgs_dir = run_root / "lbfgs"
    timings: dict[str, float] = {}

    pretrain_checkpoint = pretrain_dir / "best_pmp_kkt.pt"
    if pretrain_checkpoint.exists():
        print(f"[skip] pretrain d={d_model} seed={seed}")
    else:
        if pretrain_dir.exists():
            raise FileExistsError(
                f"partial pretraining directory requires inspection: {pretrain_dir}"
            )
        command = [
            sys.executable,
            "scripts/train_openloop.py",
            "--model",
            "transformer",
            "--n",
            "800",
            "--T",
            "10",
            "--m",
            "21",
            "--umax",
            "3",
            "--beta",
            "0.1",
            "--alpha",
            "0.0025",
            "--gamma",
            "20",
            "--n0",
            "10",
            "--m_suppression",
            "0.5",
            "--d_model",
            str(d_model),
            "--heads",
            "4",
            "--layers",
            "2",
            "--init_u",
            "1.5",
            "--epochs",
            str(pretrain_epochs),
            "--lr",
            "0.0005",
            "--weight_decay",
            "0",
            "--lr_patience",
            "100",
            "--grad_clip",
            "10",
            "--singular_eps",
            "0.1",
            "--singular_tau",
            "0.03",
            "--smooth_weight",
            "3",
            "--objective_weight",
            "0",
            "--seed",
            str(seed),
            "--device",
            "cpu",
            "--float64",
            "--print_every",
            "300",
            "--out_dir",
            str(pretrain_dir),
        ]
        print(f"[run] pretrain d={d_model} seed={seed}")
        timings["pretrain"] = run_command(
            command, log_path=run_root / "pretrain.log"
        )

    curriculum_checkpoint = curriculum_dir / "stage_3_n800/selected_checkpoint.pt"
    curriculum_complete = curriculum_dir / "COMPLETED.json"
    if curriculum_complete.exists() and curriculum_checkpoint.exists():
        print(f"[skip] curriculum d={d_model} seed={seed}")
    else:
        if curriculum_dir.exists():
            raise FileExistsError(
                f"partial curriculum directory requires inspection: {curriculum_dir}"
            )
        command = [
            sys.executable,
            "scripts/train_teacher_free_resolution_curriculum.py",
            "--out-dir",
            str(curriculum_dir),
            "--start-checkpoint",
            str(pretrain_checkpoint),
            "--seed",
            str(1100 + seed),
            "--scale",
            "1.08",
            "--temperature",
            "0.6991330744962188",
            "--learn-temperature",
            "--epochs",
            curriculum_epochs,
            "--learning-rates",
            "4e-5,1.6e-5,6e-6",
            "--linf-weights",
            "0.05,0.10,0.20",
            "--pmp-weights",
            "0.02,0.005,0.0",
            "--smooth-weights",
            "0.01,0.002,0.0005",
            "--eval-every",
            "20",
            "--grad-clip",
            "5",
            "--selection-guard",
            "residual-only",
        ]
        print(f"[run] curriculum d={d_model} seed={seed}")
        timings["curriculum"] = run_command(
            command, log_path=run_root / "curriculum.log"
        )

    lbfgs_checkpoint = lbfgs_dir / "selected_checkpoint.pt"
    lbfgs_complete = lbfgs_dir / "COMPLETED.json"
    if lbfgs_complete.exists() and lbfgs_checkpoint.exists():
        print(f"[skip] L-BFGS d={d_model} seed={seed}")
    else:
        if lbfgs_dir.exists():
            raise FileExistsError(
                f"partial L-BFGS directory requires inspection: {lbfgs_dir}"
            )
        command = [
            sys.executable,
            "scripts/continue_teacher_free_strict_optimality.py",
            "--checkpoint",
            str(curriculum_checkpoint),
            "--out-dir",
            str(lbfgs_dir),
            "--seed",
            str(2100 + seed),
            "--outer-steps",
            "6",
            "--inner-iterations",
            "5",
            "--learning-rate",
            "0.2",
            "--history-size",
            "20",
            "--p",
            "20",
            "--high-p-weight",
            "3",
            "--learn-temperature",
            "--width-tolerance",
            "0.005",
            "--variation-fraction-tolerance",
            "0.05",
            "--selection-guard",
            "residual-only",
        ]
        print(f"[run] L-BFGS d={d_model} seed={seed}")
        timings["lbfgs"] = run_command(command, log_path=run_root / "lbfgs.log")

    summary = json.loads(
        (lbfgs_dir / "summary.json").read_text(encoding="utf-8")
    )
    diagnostics = dict(summary["post_selection_diagnostics"])
    result = {
        "d_model": d_model,
        "heads": 4,
        "layers": 2,
        "feedforward_dimension": 4 * d_model,
        "seed": seed,
        "projected_gradient_linf": float(
            diagnostics["projected_gradient_linf"]
        ),
        "projected_gradient_rms": float(
            diagnostics["projected_gradient_rms"]
        ),
        "normalized_objective_diagnostic": float(
            diagnostics["high_accuracy_J_diagnostic_only"]
        ),
        "u_min": float(diagnostics["u_min"]),
        "u_max": float(diagnostics["u_max"]),
        "checkpoint": str(lbfgs_checkpoint.resolve()),
        "new_wall_seconds": timings,
    }
    (run_root / "run_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-models", type=parse_int_list, default=[16, 32, 48])
    parser.add_argument("--seeds", type=parse_int_list, default=[1])
    parser.add_argument("--out-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--pretrain-epochs", type=int, default=1200)
    parser.add_argument("--curriculum-epochs", default="160,120,100")
    args = parser.parse_args()

    experiment_root = Path(args.out_root).expanduser()
    if not experiment_root.is_absolute():
        experiment_root = ROOT / experiment_root
    experiment_root.mkdir(parents=True, exist_ok=True)

    protocol = {
        "ablation_variable": "Transformer model dimension",
        "d_models": args.d_models,
        "seeds": args.seeds,
        "fixed_architecture": {
            "layers": 2,
            "heads": 4,
            "feedforward_dimension": "4*d_model",
            "time_features": 6,
        },
        "pretraining_epochs": args.pretrain_epochs,
        "curriculum_resolutions": [200, 400, 800],
        "curriculum_epochs": [
            int(value) for value in args.curriculum_epochs.split(",")
        ],
        "selection": "projected-gradient Linf, then RMS",
        "training_uses_direct_solution": False,
        "training_uses_objective_value_as_loss_or_selection": False,
    }
    (experiment_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )

    results = []
    for d_model in args.d_models:
        for seed in args.seeds:
            results.append(
                run_one(
                    experiment_root=experiment_root,
                    d_model=d_model,
                    seed=seed,
                    pretrain_epochs=args.pretrain_epochs,
                    curriculum_epochs=args.curriculum_epochs,
                )
            )
    (experiment_root / "latest_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
