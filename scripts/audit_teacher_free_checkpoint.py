#!/usr/bin/env python3
"""Reload, lineage, and CPU-inference audit for a teacher-free checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT, ROOT / "scripts"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts.refine_time_only_singular_plateau import build_model  # noqa: E402
from scripts.train_teacher_free_resolution_curriculum import FixedBoxProjection  # noqa: E402
from train_paper_pmp_kkt import ProblemConfig  # noqa: E402


def resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(checkpoint: dict[str, Any]) -> tuple[torch.nn.Module, ProblemConfig]:
    cfg = ProblemConfig(**checkpoint["problem"])
    architecture = dict(checkpoint.get("base_model_args", checkpoint.get("args", {})))
    base = build_model(architecture, cfg).to(dtype=torch.float64)
    wrapper = checkpoint.get("wrapper")
    if wrapper is None:
        base.load_state_dict(checkpoint["model_state"])
        return base, cfg
    model = FixedBoxProjection(
        base,
        cfg.umax,
        float(wrapper["scale"]),
        temperature=float(wrapper.get("temperature", 1.0)),
        learn_temperature=bool(wrapper.get("learn_temperature", False)),
    ).to(dtype=torch.float64)
    model.load_state_dict(checkpoint["model_state"])
    return model, cfg


def lineage(checkpoint_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current = checkpoint_path
    seen: set[Path] = set()
    while current.exists() and current not in seen:
        seen.add(current)
        payload = torch.load(current, map_location="cpu", weights_only=False)
        rows.append(
            {
                "path": str(current),
                "sha256": sha256(current),
                "teacher_free_marker": payload.get("teacher_free"),
                "method": payload.get("method"),
                "stage": payload.get("stage"),
                "selected_epoch": payload.get("selected_epoch"),
                "source_checkpoint": payload.get("source_checkpoint"),
            }
        )
        source = payload.get("source_checkpoint")
        if not source:
            break
        current = resolve(source)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=500)
    args = parser.parse_args()
    run_dir = resolve(args.run_dir)
    checkpoint_path = run_dir / "selected_checkpoint.pt"
    solution_path = run_dir / "solution.npz"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, cfg = load_model(checkpoint)
    model.eval()
    torch.set_num_threads(1)
    normalized_t = torch.linspace(0.0, 1.0, cfg.n + 1, dtype=torch.float64)
    with np.load(solution_path, allow_pickle=False) as data:
        saved_t = np.asarray(data["t"], dtype=np.float64)
        saved_u = np.asarray(data["u"], dtype=np.float64)
    with torch.inference_mode():
        reloaded_u = model(normalized_t).cpu().numpy()
    reload_linf = float(np.max(np.abs(reloaded_u - saved_u)))
    if reload_linf > 1.0e-12:
        raise RuntimeError(f"checkpoint reload mismatch: {reload_linf:.3e}")

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(normalized_t)
        durations: list[float] = []
        for _ in range(args.runs):
            started = time.perf_counter_ns()
            model(normalized_t)
            durations.append((time.perf_counter_ns() - started) / 1.0e6)
    ordered = sorted(durations)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]

    lineage_rows = lineage(checkpoint_path)
    summary_paths = [
        run_dir / "summary.json",
        resolve(checkpoint["source_checkpoint"]).parent / "summary.json",
    ]
    source2 = torch.load(
        resolve(checkpoint["source_checkpoint"]), map_location="cpu", weights_only=False
    )
    if source2.get("source_checkpoint"):
        curriculum_stage = resolve(source2["source_checkpoint"]).parent
        curriculum_root = curriculum_stage.parent
        summary_paths.append(curriculum_root / "summary.json")
    wall_components: list[dict[str, Any]] = []
    for path in summary_paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        seconds = payload.get("wall_seconds", payload.get("total_training_wall_seconds"))
        if seconds is not None:
            wall_components.append({"summary": str(path), "seconds": float(seconds)})

    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "solution": str(solution_path),
        "solution_sha256": sha256(solution_path),
        "checkpoint_reload_control_linf": reload_linf,
        "saved_node_count": int(saved_t.size),
        "exact_upper_bound_count": int(np.count_nonzero(saved_u[:-1] == cfg.umax)),
        "u_min": float(saved_u[:-1].min()),
        "u_max": float(saved_u[:-1].max()),
        "inference": {
            "device": "CPU",
            "torch_threads": 1,
            "node_count": cfg.n + 1,
            "warmup_runs": args.warmup,
            "timed_runs": args.runs,
            "median_ms": float(statistics.median(durations)),
            "mean_ms": float(statistics.mean(durations)),
            "p95_ms": float(p95),
            "min_ms": float(min(durations)),
            "max_ms": float(max(durations)),
            "scope": "loaded network forward pass only",
        },
        "lineage": lineage_rows,
        "training_wall_components": wall_components,
        "training_wall_total_seconds": float(
            sum(component["seconds"] for component in wall_components)
        ),
        "prohibited_teacher_token_in_lineage": any(
            any(token in row["path"].lower() for token in ("direct", "manual_target", "supervised", "distill"))
            for row in lineage_rows
        ),
    }
    (run_dir / "checkpoint_reload_and_inference.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
