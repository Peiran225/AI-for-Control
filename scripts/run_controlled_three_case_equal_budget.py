#!/usr/bin/env python3
"""Run one branch of the equal-budget three-policy comparison.

The comparison starts every branch from the same time-only checkpoint and
uses the same optimizer, update count, minibatch stream, validation set,
architecture, numerical integrator, and residual settings.  The controlled
differences are:

* ``time`` disables the state correction and uses the derivative singular
  residual;
* ``cf`` enables the state correction and uses the closed-form singular
  residual;
* ``der`` enables the state correction and uses the derivative singular
  residual.

The learning rate is supplied explicitly because the three losses can have
different numerical scales.  The state branch is zero-initialized, but its
correction is not constrained to remain zero on the nominal trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TIME_CHECKPOINT = (
    ROOT
    / "outputs/server_direct_guided_20260727/n800_clean_lm_20260727/"
    "main_m8_equal6/selected_checkpoint.pt"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_command(
    *,
    case: str,
    learning_rate: float,
    epochs: int,
    seed: int,
    time_checkpoint: Path,
    out_dir: Path,
    device: str,
) -> list[str]:
    option = "cf" if case == "cf" else "der"
    state_mode = "w_zero" if case == "time" else "feedback"
    return [
        sys.executable,
        str(ROOT / "scripts/train_feedback_section5.py"),
        "--option",
        option,
        "--loss_variant",
        "literal",
        "--time_checkpoint",
        str(time_checkpoint),
        "--n",
        "800",
        "--T",
        "10",
        "--m",
        "21",
        "--umax",
        "3",
        "--alpha",
        "0.0025",
        "--beta",
        "0.1",
        "--gamma",
        "20",
        "--n0",
        "10",
        "--m_suppression",
        "0.5",
        "--d_model",
        "64",
        "--heads",
        "4",
        "--layers",
        "2",
        "--init_u",
        "1.5",
        "--state_hidden",
        "128,128",
        "--state_scale",
        "15",
        "--correction_gain",
        "1",
        "--no-center_state_correction",
        "--state_feature_mode",
        "burden_composition",
        "--state_mode",
        state_mode,
        "--training_integrator",
        "rk4",
        "--train_radius",
        "0.10",
        "--include_resistant_heavy",
        "--epochs",
        str(epochs),
        "--batch_size",
        "12",
        "--validation_size",
        "48",
        "--validation_seed",
        "20260719",
        "--test_size",
        "8",
        "--test_seed",
        "20260720",
        "--test_radii",
        "0.10",
        "--lr",
        f"{learning_rate:.12g}",
        "--time_lr_scale",
        "1",
        "--selection_start_epoch",
        str(epochs),
        "--training_sample_seed_base",
        "20260721",
        "--min_lr",
        "1e-12",
        "--lr_patience",
        str(epochs + 1),
        "--weight_decay",
        "0",
        "--grad_clip",
        "10",
        "--singular_eps",
        "0.1",
        "--singular_tau",
        "0.03",
        "--dot_eps",
        "0.1",
        "--dot_tau",
        "0.03",
        "--b_min",
        "1e-8",
        "--persistence_window",
        "3",
        "--gate_gradient_mode",
        "live",
        "--candidate_mask_mode",
        "dynamic",
        "--auto_residual_scales",
        "--singular_loss_weight",
        "1",
        "--nonsingular_loss_weight",
        "1",
        "--smooth_weight",
        "0",
        "--smooth_second_weight",
        "0",
        "--smooth_max_weight",
        "0",
        "--full_gradient_weight",
        "0",
        "--full_gradient_max_weight",
        "0",
        "--eval_every",
        "10",
        "--seed",
        str(seed),
        "--device",
        device,
        "--out_dir",
        str(out_dir),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("time", "cf", "der"), required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--time-checkpoint", type=Path, default=DEFAULT_TIME_CHECKPOINT
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "outputs/controlled_three_case_equal_budget_20260728",
    )
    args = parser.parse_args()

    time_checkpoint = args.time_checkpoint.expanduser().resolve()
    if not time_checkpoint.is_file():
        raise FileNotFoundError(time_checkpoint)
    out_root = args.out_root.expanduser().resolve()
    run_name = (
        f"{args.case}_lr{args.lr:.0e}_seed{args.seed}_epochs{args.epochs}"
        .replace("+", "")
    )
    out_dir = out_root / run_name
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    command = training_command(
        case=args.case,
        learning_rate=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        time_checkpoint=time_checkpoint,
        out_dir=out_dir,
        device=args.device,
    )
    protocol = {
        "schema": "controlled-three-case-equal-budget-v1",
        "case": args.case,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "seed": args.seed,
        "time_checkpoint": {
            "path": str(time_checkpoint),
            "sha256": sha256(time_checkpoint),
        },
        "controlled_differences": {
            "time": "DER singular loss with the state correction disabled",
            "cf": "closed-form singular loss with feedback enabled",
            "der": "derivative singular loss with feedback enabled",
            "learning_rate": "explicitly allowed to differ by the study design",
        },
        "shared_settings": {
            "optimizer": "AdamW",
            "updates": args.epochs,
            "minibatch_stream_seed": 20260721 + args.seed,
            "validation_seed": 20260719,
            "state_branch_initialization": "zero final layer",
            "nominal_centering_after_initialization": False,
            "time_branch_frozen": False,
            "training_integrator": "RK4",
            "control_intervals": 800,
        },
        "command": command,
    }
    (out_dir / "controlled_protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
