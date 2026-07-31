#!/usr/bin/env python3
"""One-shot blind audit for a selected gated residual adapter.

This script never trains or selects a checkpoint.  It loads a previously
selected adapter, verifies its protected-trajectory identity, evaluates one
fresh componentwise-random state set, and writes an immutable JSON summary.
"""

from __future__ import annotations

import argparse
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

from scripts.refine_feedback_gated_residual_adapter import (  # noqa: E402
    load_adapter_artifact,
    protected_identity_metrics,
    sha256,
)
from scripts.refine_feedback_svd_null_projected_kkt import (  # noqa: E402
    antithetic_initial_states,
    policy_rollout,
    posthoc_test_metrics,
    structured_initial_states,
)
from train_paper_pmp_kkt import build_params  # noqa: E402


def paired_bootstrap_mean(
    values: torch.Tensor,
    *,
    seed: int,
    repeats: int,
) -> dict[str, float]:
    array = values.detach().cpu().numpy().astype(np.float64, copy=False)
    generator = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=np.float64)
    block = 1000
    for start in range(0, repeats, block):
        stop = min(start + block, repeats)
        indices = generator.integers(
            0, array.size, size=(stop - start, array.size)
        )
        means[start:stop] = array[indices].mean(axis=1)
    return {
        "repeats": int(repeats),
        "seed": int(seed),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
    }


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    artifact_path = args.artifact.expanduser().resolve()
    payload, source, model, cfg = load_adapter_artifact(
        artifact_path, device=device
    )
    params = build_params(cfg, device, torch.float64)
    protected_initial = structured_initial_states(
        payload["protected_radii"],
        cfg,
        device,
        torch.float64,
    )
    identity = protected_identity_metrics(
        source, model, protected_initial, cfg, params
    )
    if identity["gate_max_abs"] > 1.0e-12:
        raise RuntimeError("protected gate exceeds 1e-12")
    if identity["control_max_abs"] > 1.0e-10:
        raise RuntimeError("protected controls drift by more than 1e-10")
    initial = antithetic_initial_states(
        args.count,
        args.seed,
        args.radius,
        cfg,
        device,
        torch.float64,
        antithetic=False,
    )
    with torch.no_grad():
        candidate = policy_rollout(
            model, initial, cfg, params, state_mode="feedback"
        )
        source_feedback = policy_rollout(
            source, initial, cfg, params, state_mode="feedback"
        )
        frozen_time = policy_rollout(
            source, initial, cfg, params, state_mode="w_zero"
        )
    scale = 1.0 / cfg.alpha
    time_advantage = (
        frozen_time.objectives - candidate.objectives
    ) * scale
    source_advantage = (
        source_feedback.objectives - candidate.objectives
    ) * scale
    result = {
        "protocol": "fresh_blind_gated_residual_adapter_v1",
        "artifact": str(artifact_path),
        "artifact_sha256": sha256(artifact_path),
        "gate_transition": payload.get("gate_transition"),
        "seed": args.seed,
        "count": args.count,
        "radius": args.radius,
        "used_for_training": False,
        "used_for_checkpoint_selection": False,
        "used_for_hyperparameter_tuning": False,
        "protected_identity": identity,
        **posthoc_test_metrics(
            candidate,
            source_feedback,
            frozen_time,
            physical_scale_factor=scale,
        ),
        "paired_bootstrap": {
            "candidate_advantage_over_frozen_time": paired_bootstrap_mean(
                time_advantage,
                seed=args.bootstrap_seed,
                repeats=args.bootstrap_repeats,
            ),
            "candidate_advantage_over_source_feedback": paired_bootstrap_mean(
                source_advantage,
                seed=args.bootstrap_seed + 1,
                repeats=args.bootstrap_repeats,
            ),
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--radius", type=float, default=0.20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--bootstrap-repeats", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260831)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
