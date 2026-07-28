#!/usr/bin/env python3
"""Benchmark loaded time-only policy generation with the production loader."""

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
SCRIPTS = ROOT / "scripts"
for search_path in (ROOT, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from generate_offgrid_policy_switching_diagnostics import (  # noqa: E402
    load_time_model,
    raw_time_logits,
)


def resolve(path: Path) -> Path:
    expanded = path.expanduser()
    return (expanded if expanded.is_absolute() else ROOT / expanded).resolve()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def apply_wrapper(
    raw: torch.Tensor,
    *,
    umax: float,
    wrapper: dict[str, object],
) -> torch.Tensor:
    wrapper_class = str(wrapper.get("class", ""))
    if wrapper_class == "LinearRawBoxProjection":
        return raw.clamp(0.0, umax)
    probability = raw.sigmoid()
    if wrapper_class == "AffineBoundaryProjectedControl":
        return (
            float(wrapper["scale"]) * umax * probability
            - float(wrapper["offset"])
        ).clamp(0.0, umax)
    if wrapper_class == "FixedBoxProjection":
        scale = float(wrapper["scale"])
        temperature = float(wrapper.get("temperature", 1.0))
        return (
            scale * umax * torch.sigmoid(torch.logit(probability) / temperature)
        ).clamp(0.0, umax)
    if wrapper_class == "BoundaryProjectedControl":
        scale = float(wrapper.get("scale", wrapper.get("initial_scale", 1.0)))
        return (scale * umax * probability).clamp(0.0, umax)
    return umax * probability


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

    checkpoint = resolve(args.checkpoint)
    output = resolve(args.out)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)

    model, cfg, wrapper = load_time_model(checkpoint)
    model = model.to(device="cpu", dtype=torch.float64)
    normalized_time = torch.linspace(
        0.0, 1.0, cfg.n + 1, dtype=torch.float64
    )

    def forward() -> torch.Tensor:
        raw = raw_time_logits(model, normalized_time)
        return apply_wrapper(raw, umax=float(cfg.umax), wrapper=wrapper)

    durations_ms: list[float] = []
    with torch.inference_mode():
        for _ in range(args.warmup):
            control = forward()
        for _ in range(args.repeats):
            started = time.perf_counter_ns()
            control = forward()
            durations_ms.append((time.perf_counter_ns() - started) / 1.0e6)
    if int(control.numel()) != cfg.n + 1:
        raise RuntimeError("loaded policy returned the wrong number of nodes")
    if float(control.min()) < -1.0e-12 or float(control.max()) > cfg.umax + 1.0e-12:
        raise RuntimeError("loaded policy returned an infeasible control")

    payload = {
        "checkpoint": str(checkpoint),
        "control_nodes": int(control.numel()),
        "wrapper": wrapper,
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
        "scope": (
            "preloaded network forward plus the checkpoint's recorded output "
            "wrapper for 801 nodes; loading and offline training excluded"
        ),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
