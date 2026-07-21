#!/usr/bin/env python3
"""Benchmark pretrained time-only control generation on a fixed CPU setup."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.refine_time_only_singular_plateau import build_model  # noqa: E402
from train_paper_pmp_kkt import ProblemConfig  # noqa: E402
from scripts.boundary_control import BoundaryProjectedControl  # noqa: E402


def resolve(path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (ROOT / path).resolve()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 10 or args.threads < 1:
        parser.error("warmup, repeats, and threads must be positive")

    checkpoint_path = resolve(args.checkpoint)
    output_path = resolve(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.threads)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ProblemConfig(**checkpoint["problem"])
    wrapper = checkpoint.get("wrapper")
    if wrapper:
        base_args = checkpoint.get("base_model_args")
        if base_args is None:
            source = torch.load(
                Path(checkpoint["source_checkpoint"]),
                map_location="cpu",
                weights_only=False,
            )
            base_args = source["args"]
        base = build_model(dict(base_args), cfg).to(device="cpu", dtype=torch.float64)
        scale = float(wrapper.get("scale", wrapper.get("realized_scale", 1.0)))
        model = BoundaryProjectedControl(
            base,
            umax=cfg.umax,
            scale_mode="fixed",
            initial_scale=scale,
        ).to(device="cpu", dtype=torch.float64)
    else:
        model = build_model(dict(checkpoint.get("args", {})), cfg).to(
            device="cpu", dtype=torch.float64
        )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    time_grid = torch.linspace(0.0, 1.0, cfg.n + 1, dtype=torch.float64)

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(time_grid)
        durations_ms: list[float] = []
        for _ in range(args.repeats):
            started = time.perf_counter_ns()
            control = model(time_grid)
            durations_ms.append((time.perf_counter_ns() - started) / 1.0e6)

    result = {
        "checkpoint": str(checkpoint_path),
        "control_nodes": int(control.numel()),
        "warmup_runs": args.warmup,
        "timed_runs": args.repeats,
        "torch_threads": args.threads,
        "forward_time_ms": {
            "median": statistics.median(durations_ms),
            "mean": statistics.fmean(durations_ms),
            "p05": percentile(durations_ms, 0.05),
            "p95": percentile(durations_ms, 0.95),
            "minimum": min(durations_ms),
            "maximum": max(durations_ms),
        },
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "scope": "pretrained forward pass only; checkpoint loading and offline training excluded",
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
