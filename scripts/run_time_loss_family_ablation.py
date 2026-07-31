#!/usr/bin/env python3
"""Controlled loss-family study for the time-only Transformer.

This runner compares four training losses while holding fixed:

* the two-layer, four-head, d_model=64 Transformer;
* the exact seed-specific parameter initialization for seeds 21--23;
* the normalized coefficient setting (alpha, beta, gamma)=(0.0025, 0.1, 20);
* the 1200-epoch n=800 warm-up and n=200 -> 400 -> 800 continuation budget;
* AdamW settings, grids, precision, output wrapper, and final checkpoint rule.

The four loss families are:

``j_only``
    The differentiable RK4 scalar reduced objective.
``pmp_kkt_only``
    The local Euler PMP/KKT singular/boundary optimality loss.
``projected_gradient_only``
    The complete RK4 box projected-gradient residual.
``normalized_staged_pg_pmp``
    A scheduled optimality arm: PMP/KKT warm-up, followed by the
    normalized projected-gradient loss with normalized PMP/KKT auxiliary weights
    0.02/0.005/0 that anneal to zero over the first 70% of each continuation
    stage.

An additional ``balanced_hybrid_exploratory`` arm is available explicitly for
diagnostics, but is not part of the controlled four-arm study.

Raw losses have incompatible units.  For a controlled comparison, each raw
component is divided by the L2 norm of its parameter gradient at the shared
initialization, separately for the smooth-output warm-up and the clipped-output
continuation grids.  Thus every single-component arm has unit initial
parameter-gradient norm before clipping.  These constants are computed once
from the common initialization and are shared by all arms.  No
direct-transcription artifact is read until every neural and refinement
checkpoint has been frozen.

The neural phase uses the final fixed-budget iterate, avoiding a checkpoint
selection rule that would favor one loss family.  A common residual-only
L-BFGS refinement is then launched for every arm with identical settings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
for search_path in (ROOT, ROOT / "scripts"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from scripts.refine_time_only_singular_plateau import (  # noqa: E402
    rk4_reduced_objective,
)
from scripts.train_teacher_free_resolution_curriculum import (  # noqa: E402
    FixedBoxProjection,
    evaluate,
    make_plot,
)
from train_paper_pmp_kkt import (  # noqa: E402
    ProblemConfig,
    TimeTransformer,
    build_params,
    pmp_kkt_loss,
    set_seed,
)


FAMILIES = (
    "j_only",
    "pmp_kkt_only",
    "projected_gradient_only",
    "normalized_staged_pg_pmp",
)
OPTIONAL_FAMILIES = ("balanced_hybrid_exploratory",)
DEFAULT_OUT_ROOT = (
    ROOT / "outputs/paper_ablation_20260725/time_loss_family_v2/seed_21"
)
DEFAULT_DIRECT_SUMMARY = (
    ROOT
    / "outputs/new_objective_alpha1_grid_20260721_v2"
    / "a1_b40_g8000/direct_nominal/n800_strict_final"
    / "cost_comparison_summary.csv"
)
PHYSICAL_OBJECTIVE_SCALE = 400.0
STUDY_SCHEMA_VERSION = 2
SOURCE_FILES = {
    "runner": Path(__file__).resolve(),
    "common_time_model_and_pmp_loss": ROOT / "train_paper_pmp_kkt.py",
    "rk4_objective_and_high_accuracy_evaluator": (
        ROOT / "scripts/refine_time_only_singular_plateau.py"
    ),
    "output_wrapper_and_common_metrics": (
        ROOT / "scripts/train_teacher_free_resolution_curriculum.py"
    ),
    "common_refinement": (
        ROOT / "scripts/continue_teacher_free_strict_optimality.py"
    ),
    "continuous_evaluator": ROOT / "tumor_problem.py",
}


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else ROOT / value).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def parse_int_tuple(value: str, *, expected: int | None = None) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    if expected is not None and len(result) != expected:
        raise argparse.ArgumentTypeError(f"expected exactly {expected} integers")
    return result


def parse_float_tuple(
    value: str, *, expected: int | None = None
) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(not math.isfinite(item) or item <= 0.0 for item in result):
        raise argparse.ArgumentTypeError("expected positive comma-separated numbers")
    if expected is not None and len(result) != expected:
        raise argparse.ArgumentTypeError(f"expected exactly {expected} numbers")
    return result


def parse_families(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one loss family is required")
    unknown = sorted(set(result) - set(FAMILIES) - set(OPTIONAL_FAMILIES))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown loss families: {unknown}")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("loss families must be unique")
    return result


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if math.isfinite(scalar) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            json_safe(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def environment_record() -> dict[str, Any]:
    return {
        "python_version": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }


def source_hashes() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for label, path in SOURCE_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen source {label}: {path}")
        result[label] = {
            "path": str(path),
            "sha256": sha256(path),
        }
    return result


def canonical_args_record(
    args: argparse.Namespace,
    out_root: Path,
) -> dict[str, Any]:
    values = dict(vars(args))
    values["out_root"] = str(out_root)
    values["direct_summary"] = str(resolve(args.direct_summary))
    values["families"] = list(args.families)
    values["continuation_epochs"] = list(args.continuation_epochs)
    values["continuation_lrs"] = list(args.continuation_lrs)
    values["linf_weights"] = list(args.linf_weights)
    values["staged_aux_pmp_weights"] = list(
        args.staged_aux_pmp_weights
    )
    return json_safe(values)


def frozen_config_record(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "study": "controlled time-only training-loss family study",
        "schema_version": STUDY_SCHEMA_VERSION,
        "families": list(args.families),
        "problem": asdict(make_cfg(800)),
        "physical_objective_scale": PHYSICAL_OBJECTIVE_SCALE,
        "architecture": {
            "model": "TimeTransformer",
            "d_model": args.d_model,
            "heads": args.heads,
            "layers": args.layers,
            "feedforward_dimension": 4 * args.d_model,
            "init_u": args.init_u,
            "dtype": "torch.float64",
        },
        "losses": {
            "j_only": "differentiable RK4 scalar reduced objective",
            "pmp_kkt_only": (
                "Euler local singular/boundary PMP/KKT residual with "
                "trajectory-dependent detached gate"
            ),
            "projected_gradient_only": (
                "complete RK4 box projected-gradient mean-square plus "
                "stage-specific max-square term"
            ),
            "normalized_staged_pg_pmp": {
                "pretrain": "normalized PMP/KKT only",
                "continuation": (
                    "normalized projected-gradient plus annealed normalized "
                    "PMP/KKT auxiliary"
                ),
                "auxiliary_weights": list(args.staged_aux_pmp_weights),
                "anneal": "linear to zero by 70% of each continuation stage",
            },
            "normalization": (
                "each raw component uses a fixed divisor equal to its "
                "parameter-gradient L2 norm at the common seed-specific "
                "reference initialization for that parameterization/grid; "
                "divisors are not re-estimated at any arm's stage start"
            ),
        },
        "training": {
            "pretrain": {
                "n": 800,
                "epochs": args.pretrain_epochs,
                "learning_rate": args.pretrain_lr,
                "grad_clip": args.pretrain_grad_clip,
            },
            "continuation": {
                "n": [200, 400, 800],
                "epochs": list(args.continuation_epochs),
                "learning_rates": list(args.continuation_lrs),
                "linf_weights": list(args.linf_weights),
                "grad_clip": args.continuation_grad_clip,
            },
            "checkpoint_rule": "fixed final iterate",
            "optimizer": "AdamW, weight_decay=0, no scheduler",
        },
        "common_refinement_procedure": {
            "implementation": "continue_teacher_free_strict_optimality.py",
            "outer_steps": args.refine_outer_steps,
            "inner_iterations": args.refine_inner_iterations,
            "learning_rate": args.refine_lr,
            "history_size": args.refine_history_size,
            "p": args.refine_p,
            "high_p_weight": args.refine_high_p_weight,
            "selection": "projected-gradient Linf then RMS",
        },
    }


def snapshot_sources(
    out_root: Path,
    hashes: dict[str, dict[str, str]],
) -> dict[str, dict[str, str]]:
    snapshot_dir = out_root / "source_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    inventory: dict[str, dict[str, str]] = {}
    for label, record in hashes.items():
        source = Path(record["path"])
        destination = snapshot_dir / f"{label}{source.suffix}"
        shutil.copy2(source, destination)
        copied_sha = sha256(destination)
        if copied_sha != record["sha256"]:
            raise RuntimeError(f"source snapshot mismatch for {label}")
        inventory[label] = {
            "source_path": str(source),
            "source_sha256": record["sha256"],
            "snapshot_path": str(destination),
            "snapshot_sha256": copied_sha,
        }
    return inventory


def validate_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA mismatch: expected {expected}, received {actual}"
        )


def relative_inventory(
    directory: Path,
    *,
    excluded_names: set[str],
    excluded_top_dirs: set[str] | None = None,
) -> dict[str, str]:
    excluded_top_dirs = excluded_top_dirs or set()
    result: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name in excluded_names:
            continue
        relative = path.relative_to(directory)
        if relative.parts and relative.parts[0] in excluded_top_dirs:
            continue
        result[str(relative)] = sha256(path)
    return result


def validate_inventory(
    directory: Path,
    expected: dict[str, str],
    *,
    excluded_names: set[str],
    excluded_top_dirs: set[str] | None = None,
) -> None:
    actual = relative_inventory(
        directory,
        excluded_names=excluded_names,
        excluded_top_dirs=excluded_top_dirs,
    )
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(
            key
            for key in set(expected) & set(actual)
            if expected[key] != actual[key]
        )
        raise RuntimeError(
            f"artifact inventory mismatch under {directory}; "
            f"missing={missing}, extra={extra}, changed={changed}"
        )


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def write_or_validate_json(path: Path, value: Any) -> None:
    """Create a JSON artifact once, or require exact canonical agreement."""
    safe_value = json_safe(value)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if canonical_hash(existing) != canonical_hash(safe_value):
            raise RuntimeError(f"immutable JSON artifact mismatch: {path}")
        return
    write_json(path, safe_value)


def write_or_validate_csv(
    path: Path,
    rows: Iterable[dict[str, Any]],
) -> None:
    """Create a CSV artifact once, or require byte-for-byte agreement."""
    import io

    materialized = list(rows)
    if not materialized:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(materialized)
    content = stream.getvalue()
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"immutable CSV artifact mismatch: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def model_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": "transformer",
        "n": 800,
        "T": 10.0,
        "m": 21,
        "umax": 3.0,
        "beta": 0.1,
        "alpha": 0.0025,
        "gamma": 20.0,
        "n0": 10.0,
        "m_suppression": 0.5,
        "d_model": args.d_model,
        "heads": args.heads,
        "layers": args.layers,
        "init_u": args.init_u,
        "seed": args.seed,
        "float64": True,
    }


def make_cfg(n: int) -> ProblemConfig:
    return ProblemConfig(
        T=10.0,
        n=n,
        m=21,
        umax=3.0,
        beta=0.1,
        alpha=0.0025,
        gamma=20.0,
        n0=10.0,
        m_suppression=0.5,
    )


def build_base(args: argparse.Namespace) -> TimeTransformer:
    return TimeTransformer(
        args.d_model,
        args.heads,
        args.layers,
        3.0,
        args.init_u,
    ).to(dtype=torch.float64)


def build_wrapped(
    base: nn.Module,
    args: argparse.Namespace,
) -> FixedBoxProjection:
    return FixedBoxProjection(
        base,
        3.0,
        args.output_scale,
        temperature=args.output_temperature,
        learn_temperature=True,
    ).to(dtype=torch.float64)


def projected_gradient_component(
    all_control: torch.Tensor,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
    *,
    linf_weight: float,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    interval = all_control[:-1]
    objective = rk4_reduced_objective(interval, cfg, params)
    gradient = torch.autograd.grad(
        objective,
        interval,
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    mapping = interval - torch.clamp(interval - gradient, 0.0, cfg.umax)
    scaled = mapping / (cfg.T / cfg.n)
    residual = scaled.square().mean() + linf_weight * scaled.abs().max().square()
    return objective, residual, mapping


def raw_component(
    model: nn.Module,
    normalized_t: torch.Tensor,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
    component: str,
    *,
    linf_weight: float,
    create_graph: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    all_control = model(normalized_t)
    if component == "j":
        value = rk4_reduced_objective(all_control[:-1], cfg, params)
        return value, {"objective": value, "control": all_control}
    if component == "pmp_kkt":
        pack = pmp_kkt_loss(
            all_control,
            cfg,
            params,
            0.1,
            0.03,
            detach_gate=True,
        )
        return pack["opt_gap"], {
            "pmp_kkt": pack["opt_gap"],
            "control": all_control,
        }
    if component == "projected_gradient":
        objective, value, mapping = projected_gradient_component(
            all_control,
            cfg,
            params,
            linf_weight=linf_weight,
            create_graph=create_graph,
        )
        return value, {
            "objective": objective,
            "projected_gradient": value,
            "mapping": mapping,
            "control": all_control,
        }
    raise ValueError(component)


def calibrated_training_loss(
    model: nn.Module,
    normalized_t: torch.Tensor,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
    family: str,
    calibration: dict[str, float],
    *,
    linf_weight: float,
    staged_phase: str,
    staged_aux_weight: float,
    staged_anneal: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if family == "j_only":
        raw, pieces = raw_component(
            model,
            normalized_t,
            cfg,
            params,
            "j",
            linf_weight=linf_weight,
            create_graph=True,
        )
        return raw / calibration["j"], pieces
    if family == "pmp_kkt_only":
        raw, pieces = raw_component(
            model,
            normalized_t,
            cfg,
            params,
            "pmp_kkt",
            linf_weight=linf_weight,
            create_graph=True,
        )
        return raw / calibration["pmp_kkt"], pieces
    if family == "projected_gradient_only":
        raw, pieces = raw_component(
            model,
            normalized_t,
            cfg,
            params,
            "projected_gradient",
            linf_weight=linf_weight,
            create_graph=True,
        )
        return raw / calibration["projected_gradient"], pieces
    if family == "normalized_staged_pg_pmp":
        if staged_phase == "pretrain":
            pmp, pieces = raw_component(
                model,
                normalized_t,
                cfg,
                params,
                "pmp_kkt",
                linf_weight=linf_weight,
                create_graph=True,
            )
            pieces["effective_staged_aux_weight"] = torch.as_tensor(
                0.0,
                dtype=pmp.dtype,
                device=pmp.device,
            )
            return pmp / calibration["pmp_kkt"], pieces
        if staged_phase != "continuation":
            raise ValueError(f"unknown staged phase: {staged_phase}")
        pmp, pmp_pieces = raw_component(
            model,
            normalized_t,
            cfg,
            params,
            "pmp_kkt",
            linf_weight=linf_weight,
            create_graph=True,
        )
        pg, pg_pieces = raw_component(
            model,
            normalized_t,
            cfg,
            params,
            "projected_gradient",
            linf_weight=linf_weight,
            create_graph=True,
        )
        effective_aux = staged_aux_weight * staged_anneal
        pieces = {**pmp_pieces, **pg_pieces}
        pieces["effective_staged_aux_weight"] = torch.as_tensor(
            effective_aux,
            dtype=pg.dtype,
            device=pg.device,
        )
        return (
            pg / calibration["projected_gradient"]
            + effective_aux * pmp / calibration["pmp_kkt"]
        ), pieces
    if family == "balanced_hybrid_exploratory":
        pmp, pmp_pieces = raw_component(
            model,
            normalized_t,
            cfg,
            params,
            "pmp_kkt",
            linf_weight=linf_weight,
            create_graph=True,
        )
        pg, pg_pieces = raw_component(
            model,
            normalized_t,
            cfg,
            params,
            "projected_gradient",
            linf_weight=linf_weight,
            create_graph=True,
        )
        pieces = {**pmp_pieces, **pg_pieces}
        return 0.5 * (
            pmp / calibration["pmp_kkt"]
            + pg / calibration["projected_gradient"]
        ), pieces
    raise ValueError(family)


def parameter_gradient_norm(
    value: torch.Tensor,
    model: nn.Module,
) -> float:
    gradients = torch.autograd.grad(
        value,
        tuple(model.parameters()),
        allow_unused=True,
    )
    squared = torch.zeros((), dtype=torch.float64)
    for gradient in gradients:
        if gradient is not None:
            squared = squared + gradient.detach().square().sum()
    return float(squared.sqrt())


def calibration_for_model(
    model: nn.Module,
    cfg: ProblemConfig,
    *,
    linf_weight: float,
) -> dict[str, Any]:
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    normalized_t = torch.linspace(0.0, 1.0, cfg.n + 1, dtype=torch.float64)
    values: dict[str, float] = {}
    gradient_norms: dict[str, float] = {}
    for component in ("j", "pmp_kkt", "projected_gradient"):
        raw, _ = raw_component(
            model,
            normalized_t,
            cfg,
            params,
            component,
            linf_weight=linf_weight,
            create_graph=True,
        )
        value = float(raw.detach())
        if not math.isfinite(value) or value <= 0.0:
            raise RuntimeError(
                f"invalid {component} calibration at n={cfg.n}: {value}"
            )
        values[component] = value
        gradient_norms[component] = parameter_gradient_norm(raw, model)
    return {
        "n": cfg.n,
        "linf_weight": linf_weight,
        "raw_values": values,
        "raw_parameter_gradient_norms": gradient_norms,
        "parameter_gradient_norms_after_value_normalization": {
            key: gradient_norms[key] / values[key] for key in values
        },
        "training_divisors": gradient_norms,
        "parameter_gradient_norms_after_training_calibration": {
            key: 1.0 for key in values
        },
    }


def common_diagnostics(
    model: nn.Module,
    cfg: ProblemConfig,
    *,
    high_accuracy: bool,
) -> tuple[dict[str, Any], np.ndarray]:
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    normalized_t = torch.linspace(0.0, 1.0, cfg.n + 1, dtype=torch.float64)
    metrics, control = evaluate(
        model,
        cfg,
        normalized_t,
        params,
        high_accuracy=high_accuracy,
    )
    return metrics, control


def train_stage(
    *,
    model: nn.Module,
    family: str,
    stage_name: str,
    cfg: ProblemConfig,
    epochs: int,
    learning_rate: float,
    grad_clip: float,
    linf_weight: float,
    calibration: dict[str, float],
    staged_phase: str,
    staged_aux_weight: float,
    eval_every: int,
    global_step_start: int,
    stage_dir: Path,
) -> tuple[list[dict[str, Any]], int, float]:
    stage_dir.mkdir(parents=True, exist_ok=False)
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    normalized_t = torch.linspace(0.0, 1.0, cfg.n + 1, dtype=torch.float64)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )
    history: list[dict[str, Any]] = []
    global_step = global_step_start
    started = time.perf_counter()

    initial_metrics, _ = common_diagnostics(model, cfg, high_accuracy=False)
    history.append(
        {
            "event": "evaluation",
            "family": family,
            "stage": stage_name,
            "n": cfg.n,
            "epoch": 0,
            "global_step": global_step,
            **initial_metrics,
        }
    )
    print(
        f"[{family} {stage_name} e=0] "
        f"J={initial_metrics['rk4_objective_diagnostic_only']:.8f} "
        f"PGinf={initial_metrics['projected_gradient_linf']:.3e} "
        f"PMP={initial_metrics['euler_pmp_kkt_opt_gap']:.3e}",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, pieces = calibrated_training_loss(
            model,
            normalized_t,
            cfg,
            params,
            family,
            calibration,
            linf_weight=linf_weight,
            staged_phase=staged_phase,
            staged_aux_weight=staged_aux_weight,
            staged_anneal=(
                max(0.0, 1.0 - (epoch / max(1, epochs)) / 0.70)
                if staged_phase == "continuation"
                else 1.0
            ),
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite loss for {family} {stage_name} epoch {epoch}"
            )
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        )
        optimizer.step()
        global_step += 1
        row: dict[str, Any] = {
            "event": "train",
            "family": family,
            "stage": stage_name,
            "n": cfg.n,
            "epoch": epoch,
            "global_step": global_step,
            "loss": float(loss.detach()),
            "learning_rate": learning_rate,
            "gradient_norm_before_clip": gradient_norm,
            "elapsed_stage_seconds": time.perf_counter() - started,
        }
        for key in ("objective", "pmp_kkt", "projected_gradient"):
            if key in pieces:
                row[f"raw_{key}"] = float(pieces[key].detach())
        if "effective_staged_aux_weight" in pieces:
            row["effective_staged_aux_weight"] = float(
                pieces["effective_staged_aux_weight"].detach()
            )
        history.append(row)

        if epoch % eval_every == 0 or epoch == epochs:
            metrics, _ = common_diagnostics(
                model,
                cfg,
                high_accuracy=False,
            )
            history.append(
                {
                    "event": "evaluation",
                    "family": family,
                    "stage": stage_name,
                    "n": cfg.n,
                    "epoch": epoch,
                    "global_step": global_step,
                    "elapsed_stage_seconds": time.perf_counter() - started,
                    **metrics,
                }
            )
            print(
                f"[{family} {stage_name} e={epoch}] "
                f"loss={float(loss.detach()):.4e} "
                f"J={metrics['rk4_objective_diagnostic_only']:.8f} "
                f"PGinf={metrics['projected_gradient_linf']:.3e} "
                f"PMP={metrics['euler_pmp_kkt_opt_gap']:.3e}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    final_metrics, control = common_diagnostics(
        model,
        cfg,
        high_accuracy=True,
    )
    np.savez_compressed(
        stage_dir / "solution.npz",
        t=np.linspace(0.0, cfg.T, cfg.n + 1),
        u=control,
    )
    write_csv(stage_dir / "history.csv", history)
    write_json(
        stage_dir / "summary.json",
        {
            "family": family,
            "stage": stage_name,
            "n": cfg.n,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "linf_weight": linf_weight,
            "staged_phase": staged_phase,
            "staged_aux_weight_at_stage_start": staged_aux_weight,
            "staged_aux_anneal_to_zero_fraction": (
                0.70 if staged_phase == "continuation" else None
            ),
            "calibration": calibration,
            "checkpoint_rule": "fixed final iterate",
            "wall_seconds": elapsed,
            "metrics": final_metrics,
        },
    )
    return history, global_step, elapsed


def save_base_checkpoint(
    path: Path,
    model: nn.Module,
    cfg: ProblemConfig,
    args: argparse.Namespace,
    *,
    family: str,
    stage: str,
    run_identity: dict[str, str],
) -> None:
    torch.save(
        {
            "model_state": clone_state(model),
            "args": model_args(args),
            "problem": asdict(cfg),
            "seed": args.seed,
            "family": family,
            "stage": stage,
            "common_initialization": True,
            "checkpoint_rule": "fixed final iterate",
            "direct_or_manual_solution_used": False,
            "study_identity": run_identity,
        },
        path,
    )


def save_wrapped_checkpoint(
    path: Path,
    model: FixedBoxProjection,
    cfg: ProblemConfig,
    args: argparse.Namespace,
    *,
    family: str,
    run_identity: dict[str, str],
) -> None:
    objective_used = family == "j_only"
    torch.save(
        {
            "model_state": clone_state(model),
            "base_model_args": model_args(args),
            "problem": asdict(cfg),
            "wrapper": {
                "class": "FixedBoxProjection",
                "scale": args.output_scale,
                "temperature": float(model.temperature_value().detach()),
                "learn_temperature": True,
                "forward": (
                    "clamp(scale * umax * sigmoid(logit(base/umax) / "
                    "global_temperature), 0, umax)"
                ),
            },
            "source_checkpoint": "common seed initialization generated by this runner",
            "teacher_free": True,
            "direct_or_manual_solution_used": False,
            "objective_value_used_as_loss": objective_used,
            "objective_value_used_for_checkpoint_selection": False,
            "switching_time_or_mask_used": False,
            "full_gradient_includes_state_dependence": (
                family
                in {
                    "projected_gradient_only",
                    "normalized_staged_pg_pmp",
                    "balanced_hybrid_exploratory",
                }
            ),
            "family": family,
            "seed": args.seed,
            "checkpoint_rule": "fixed final iterate",
            "study_identity": run_identity,
        },
        path,
    )


def validate_neural_arm(
    arm_dir: Path,
    *,
    family: str,
    seed: int,
    run_identity: dict[str, str],
) -> dict[str, Any]:
    marker_path = arm_dir / "NEURAL_COMPLETED.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected_identity = dict(marker.get("study_identity", {}))
    if expected_identity != run_identity:
        raise RuntimeError(
            f"neural reuse identity mismatch for {family}: "
            f"{expected_identity} != {run_identity}"
        )
    if marker.get("family") != family or int(marker.get("seed", -1)) != seed:
        raise RuntimeError(f"neural reuse family/seed mismatch for {arm_dir}")
    validate_inventory(
        arm_dir,
        dict(marker["artifact_inventory"]),
        excluded_names={"NEURAL_COMPLETED.json", "refinement.log"},
        excluded_top_dirs={"refinement"},
    )
    checkpoint_path = arm_dir / "pre_refinement/selected_checkpoint.pt"
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if dict(checkpoint.get("study_identity", {})) != run_identity:
        raise RuntimeError(f"checkpoint identity mismatch for {family}")
    summary_path = arm_dir / "neural_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if dict(summary.get("study_identity", {})) != run_identity:
        raise RuntimeError(f"summary identity mismatch for {family}")
    print(f"[validated skip] completed neural arm {family}", flush=True)
    return summary


def train_family(
    *,
    family: str,
    root: Path,
    common_state: dict[str, torch.Tensor],
    calibrations: dict[str, Any],
    args: argparse.Namespace,
    run_identity: dict[str, str],
) -> dict[str, Any]:
    arm_dir = root / family
    complete = arm_dir / "NEURAL_COMPLETED.json"
    pre_checkpoint = arm_dir / "pre_refinement/selected_checkpoint.pt"
    if complete.exists() and pre_checkpoint.exists():
        return validate_neural_arm(
            arm_dir,
            family=family,
            seed=args.seed,
            run_identity=run_identity,
        )
    if arm_dir.exists():
        raise FileExistsError(
            f"partial arm requires inspection before rerun: {arm_dir}"
        )
    arm_dir.mkdir(parents=True)
    started = time.perf_counter()
    model = build_base(args)
    model.load_state_dict(common_state)
    if state_digest(clone_state(model)) != state_digest(common_state):
        raise RuntimeError(f"common initialization reload mismatch for {family}")

    stage_summaries: list[dict[str, Any]] = []
    all_history: list[dict[str, Any]] = []
    global_step = 0
    pre_cfg = make_cfg(800)
    pre_dir = arm_dir / "pretrain_n800"
    history, global_step, elapsed = train_stage(
        model=model,
        family=family,
        stage_name="pretrain_n800",
        cfg=pre_cfg,
        epochs=args.pretrain_epochs,
        learning_rate=args.pretrain_lr,
        grad_clip=args.pretrain_grad_clip,
        linf_weight=args.linf_weights[2],
        calibration=calibrations["pretrain_n800"]["training_divisors"],
        staged_phase="pretrain",
        staged_aux_weight=0.0,
        eval_every=args.pretrain_eval_every,
        global_step_start=global_step,
        stage_dir=pre_dir,
    )
    all_history.extend(history)
    stage_summaries.append(
        {
            "stage": "pretrain_n800",
            "wall_seconds": elapsed,
            "summary": str(pre_dir / "summary.json"),
        }
    )
    save_base_checkpoint(
        pre_dir / "final_checkpoint.pt",
        model,
        pre_cfg,
        args,
        family=family,
        stage="pretrain_n800",
        run_identity=run_identity,
    )

    wrapped = build_wrapped(model, args)
    for index, n in enumerate((200, 400, 800)):
        cfg = make_cfg(n)
        stage_name = f"continuation_n{n}"
        stage_dir = arm_dir / stage_name
        history, global_step, elapsed = train_stage(
            model=wrapped,
            family=family,
            stage_name=stage_name,
            cfg=cfg,
            epochs=args.continuation_epochs[index],
            learning_rate=args.continuation_lrs[index],
            grad_clip=args.continuation_grad_clip,
            linf_weight=args.linf_weights[index],
            calibration=calibrations[stage_name]["training_divisors"],
            staged_phase="continuation",
            staged_aux_weight=args.staged_aux_pmp_weights[index],
            eval_every=args.continuation_eval_every,
            global_step_start=global_step,
            stage_dir=stage_dir,
        )
        all_history.extend(history)
        stage_summaries.append(
            {
                "stage": stage_name,
                "wall_seconds": elapsed,
                "summary": str(stage_dir / "summary.json"),
            }
        )

    pre_dir = arm_dir / "pre_refinement"
    pre_dir.mkdir()
    final_cfg = make_cfg(800)
    metrics, control = common_diagnostics(
        wrapped,
        final_cfg,
        high_accuracy=True,
    )
    save_wrapped_checkpoint(
        pre_dir / "selected_checkpoint.pt",
        wrapped,
        final_cfg,
        args,
        family=family,
        run_identity=run_identity,
    )
    np.savez_compressed(
        pre_dir / "solution.npz",
        t=np.linspace(0.0, final_cfg.T, final_cfg.n + 1),
        u=control,
    )
    make_plot(
        pre_dir / "control",
        np.linspace(0.0, final_cfg.T, final_cfg.n + 1),
        control,
    )
    write_json(
        pre_dir / "summary.json",
        {
            "family": family,
            "seed": args.seed,
            "checkpoint_rule": "fixed final iterate",
            "metrics": metrics,
        },
    )
    write_csv(arm_dir / "neural_history_all_stages.csv", all_history)
    total = time.perf_counter() - started
    summary = {
        "family": family,
        "seed": args.seed,
        "common_initialization_digest": state_digest(common_state),
        "total_neural_steps": global_step,
        "total_neural_wall_seconds": total,
        "stages": stage_summaries,
        "pre_refinement_checkpoint": str(
            pre_dir / "selected_checkpoint.pt"
        ),
        "pre_refinement_metrics": metrics,
        "study_identity": run_identity,
    }
    write_json(arm_dir / "neural_summary.json", summary)
    artifact_inventory = relative_inventory(
        arm_dir,
        excluded_names={"NEURAL_COMPLETED.json", "refinement.log"},
        excluded_top_dirs={"refinement"},
    )
    write_json(
        complete,
        {
            "completed_unix_time": time.time(),
            "family": family,
            "seed": args.seed,
            "checkpoint": str(pre_dir / "selected_checkpoint.pt"),
            "study_identity": run_identity,
            "artifact_inventory": artifact_inventory,
        },
    )
    return summary


def refinement_command(
    source: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    return [
        sys.executable,
        "scripts/continue_teacher_free_strict_optimality.py",
        "--checkpoint",
        str(source),
        "--out-dir",
        str(out_dir),
        "--seed",
        str(args.seed),
        "--outer-steps",
        str(args.refine_outer_steps),
        "--inner-iterations",
        str(args.refine_inner_iterations),
        "--learning-rate",
        str(args.refine_lr),
        "--history-size",
        str(args.refine_history_size),
        "--p",
        str(args.refine_p),
        "--high-p-weight",
        str(args.refine_high_p_weight),
        "--learn-temperature",
        "--width-tolerance",
        "1.0",
        "--variation-fraction-tolerance",
        "1.0",
        "--selection-guard",
        "residual-only",
    ]


def run_refinement(
    family: str,
    root: Path,
    args: argparse.Namespace,
    run_identity: dict[str, str],
) -> dict[str, Any]:
    arm_dir = root / family
    source = arm_dir / "pre_refinement/selected_checkpoint.pt"
    out_dir = arm_dir / "refinement"
    complete = out_dir / "COMPLETED.json"
    validation_marker = out_dir / "REFINEMENT_VALIDATED.json"
    command = refinement_command(source, out_dir, args)
    if complete.exists() and validation_marker.exists():
        marker = json.loads(validation_marker.read_text(encoding="utf-8"))
        if dict(marker.get("study_identity", {})) != run_identity:
            raise RuntimeError(f"refinement reuse identity mismatch for {family}")
        if marker.get("family") != family:
            raise RuntimeError(f"refinement family mismatch for {family}")
        if marker.get("input_checkpoint_sha256") != sha256(source):
            raise RuntimeError(f"refinement input checkpoint changed for {family}")
        if marker.get("command") != command:
            raise RuntimeError(f"refinement command mismatch for {family}")
        if marker.get("command_hash") != canonical_hash(command):
            raise RuntimeError(f"refinement command hash mismatch for {family}")
        validate_file_hash(
            arm_dir / "refinement.log",
            str(marker["external_log_sha256"]),
            f"{family} refinement log",
        )
        validate_inventory(
            out_dir,
            dict(marker["artifact_inventory"]),
            excluded_names={"REFINEMENT_VALIDATED.json"},
        )
        summary = json.loads(
            (out_dir / "summary.json").read_text(encoding="utf-8")
        )
        if int(summary["closure_calls"]) != int(marker["closure_calls"]):
            raise RuntimeError(f"refinement closure-call mismatch for {family}")
        if float(summary["wall_seconds"]) != float(marker["wall_seconds"]):
            raise RuntimeError(f"refinement wall-time mismatch for {family}")
        print(f"[validated skip] completed refinement {family}", flush=True)
        return summary
    if complete.exists() != validation_marker.exists():
        raise RuntimeError(
            f"incomplete refinement provenance markers for {family}"
        )
    if out_dir.exists():
        raise FileExistsError(
            f"partial refinement requires inspection before rerun: {out_dir}"
        )
    log_path = arm_dir / "refinement.log"
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
    summary = json.loads(
        (out_dir / "summary.json").read_text(encoding="utf-8")
    )
    inventory = relative_inventory(
        out_dir,
        excluded_names={"REFINEMENT_VALIDATED.json"},
    )
    write_json(
        validation_marker,
        {
            "family": family,
            "seed": args.seed,
            "study_identity": run_identity,
            "input_checkpoint": str(source),
            "input_checkpoint_sha256": sha256(source),
            "command": command,
            "command_hash": canonical_hash(command),
            "closure_calls": int(summary["closure_calls"]),
            "wall_seconds": float(summary["wall_seconds"]),
            "external_log_sha256": sha256(log_path),
            "artifact_inventory": inventory,
        },
    )
    return summary


def read_direct_reference(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty direct-reference summary: {path}")
    direct_rows = [
        row
        for row in rows
        if row.get("method") == "per-condition direct reference"
    ]
    selected = direct_rows[0] if direct_rows else rows[0]
    value = float(selected["J"])
    if not math.isfinite(value):
        raise ValueError(f"non-finite direct objective in {path}")
    return value


def comparison_row(
    *,
    family: str,
    phase: str,
    metrics: dict[str, Any],
    direct_physical: float,
    neural_wall: float,
    refinement_wall: float,
    refinement_closure_calls: int,
) -> dict[str, Any]:
    normalized_j = float(metrics["high_accuracy_J_diagnostic_only"])
    physical_j = PHYSICAL_OBJECTIVE_SCALE * normalized_j
    return {
        "family": family,
        "phase": phase,
        "normalized_J": normalized_j,
        "physical_J": physical_j,
        "absolute_gap_vs_direct": physical_j - direct_physical,
        "relative_gap_vs_direct_percent": (
            100.0 * (physical_j - direct_physical) / direct_physical
        ),
        "projected_gradient_linf": float(
            metrics["projected_gradient_linf"]
        ),
        "projected_gradient_rms": float(
            metrics["projected_gradient_rms"]
        ),
        "euler_pmp_kkt_opt_gap": float(
            metrics["euler_pmp_kkt_opt_gap"]
        ),
        "continuous_projected_kkt_rms": float(
            metrics["continuous_projected_kkt_rms"]
        ),
        "u_min": float(metrics["u_min"]),
        "u_max": float(metrics["u_max"]),
        "control_total_variation": float(
            metrics["control_total_variation"]
        ),
        "control_second_difference_l1": float(
            metrics["control_second_difference_l1"]
        ),
        "neural_wall_seconds": neural_wall,
        "refinement_wall_seconds": refinement_wall,
        "refinement_closure_calls": refinement_closure_calls,
        "cumulative_wall_seconds": neural_wall + refinement_wall,
        "cumulative_closure_calls": refinement_closure_calls,
    }


def build_comparison(
    root: Path,
    families: tuple[str, ...],
    direct_summary: Path,
) -> list[dict[str, Any]]:
    # This is intentionally the first function allowed to read a direct
    # transcription artifact.  All requested checkpoints must already exist.
    for family in families:
        required = (
            root / family / "NEURAL_COMPLETED.json",
            root / family / "refinement/COMPLETED.json",
        )
        if not all(path.exists() for path in required):
            raise RuntimeError(
                f"cannot unblind objective comparison before {family} is frozen"
            )
    direct_physical = read_direct_reference(direct_summary)
    rows: list[dict[str, Any]] = []
    for family in families:
        neural = json.loads(
            (root / family / "neural_summary.json").read_text()
        )
        refinement = json.loads(
            (root / family / "refinement/summary.json").read_text()
        )
        neural_wall = float(neural["total_neural_wall_seconds"])
        refinement_wall = float(refinement["wall_seconds"])
        rows.append(
            comparison_row(
                family=family,
                phase="pre_refinement",
                metrics=neural["pre_refinement_metrics"],
                direct_physical=direct_physical,
                neural_wall=neural_wall,
                refinement_wall=0.0,
                refinement_closure_calls=0,
            )
        )
        rows.append(
            comparison_row(
                family=family,
                phase="post_common_refinement",
                metrics=refinement["post_selection_diagnostics"],
                direct_physical=direct_physical,
                neural_wall=neural_wall,
                refinement_wall=refinement_wall,
                refinement_closure_calls=int(refinement["closure_calls"]),
            )
        )
    write_or_validate_csv(root / "comparison.csv", rows)
    write_or_validate_json(
        root / "comparison.json",
        {
            "direct_reference_physical_J": direct_physical,
            "direct_reference_summary": str(direct_summary),
            "direct_reference_sha256": sha256(direct_summary),
            "physical_objective_scale": PHYSICAL_OBJECTIVE_SCALE,
            "rows": rows,
        },
    )
    return rows


def prepare_common_initialization(
    root: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], str]:
    path = root / "common_initialization.pt"
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = dict(payload["model_state"])
        expected = int(payload["seed"])
        if expected != args.seed:
            raise ValueError(
                f"existing common initialization seed {expected} != {args.seed}"
            )
        return state, state_digest(state)
    set_seed(args.seed)
    model = build_base(args)
    state = clone_state(model)
    digest = state_digest(state)
    torch.save(
        {
            "model_state": state,
            "args": model_args(args),
            "problem": asdict(make_cfg(800)),
            "seed": args.seed,
            "state_digest": digest,
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        },
        path,
    )
    return state, digest


def prepare_calibrations(
    root: Path,
    common_state: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, Any]:
    path = root / "loss_calibration.json"
    if path.exists():
        return json.loads(path.read_text())

    base = build_base(args)
    base.load_state_dict(common_state)
    calibrations: dict[str, Any] = {
        "pretrain_n800": calibration_for_model(
            base,
            make_cfg(800),
            linf_weight=args.linf_weights[2],
        )
    }
    wrapped = build_wrapped(base, args)
    for index, n in enumerate((200, 400, 800)):
        calibrations[f"continuation_n{n}"] = calibration_for_model(
            wrapped,
            make_cfg(n),
            linf_weight=args.linf_weights[index],
        )
    write_json(path, calibrations)
    return calibrations


def build_protocol(
    out_root: Path,
    common_state: dict[str, torch.Tensor],
    initialization_digest: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "experiment": "controlled time-only training-loss family study",
        "scope_statement": (
            "This is a controlled loss-family study, not a claim of "
            "command-for-command reproduction of the paper's main, formal, "
            "or current method."
        ),
        "scheduled_optimality_arm": "normalized_staged_pg_pmp",
        "families": list(args.families),
        "problem_training_scale": asdict(make_cfg(800)),
        "physical_problem": {
            "alpha": 1.0,
            "beta": 40.0,
            "gamma": 8000.0,
            "positive_objective_scale": PHYSICAL_OBJECTIVE_SCALE,
        },
        "architecture": {
            "model": "TimeTransformer",
            "d_model": args.d_model,
            "heads": args.heads,
            "layers": args.layers,
            "feedforward_dimension": 4 * args.d_model,
            "parameter_count": sum(
                tensor.numel() for tensor in common_state.values()
            ),
        },
        "common_initialization": {
            "seed": args.seed,
            "state_digest": initialization_digest,
            "checkpoint": str(out_root / "common_initialization.pt"),
        },
        "loss_normalization": (
            "Each raw component is divided by the L2 norm of its parameter "
            "gradient at the common seed-specific reference initialization "
            "for that output parameterization and grid. These divisors are "
            "computed once and shared; they are not re-estimated at any "
            "arm's stage start."
        ),
        "pmp_gate_gradient": "detached; gate values remain trajectory-dependent",
        "checkpoint_rule": "fixed final iterate for every neural arm",
        "warmup": {
            "n": 800,
            "epochs": args.pretrain_epochs,
            "learning_rate": args.pretrain_lr,
            "output": "3*sigmoid(base logit)",
        },
        "continuation": {
            "n": [200, 400, 800],
            "epochs": list(args.continuation_epochs),
            "learning_rates": list(args.continuation_lrs),
            "projected_linf_weights": list(args.linf_weights),
            "scheduled_auxiliary_pmp_weights": list(
                args.staged_aux_pmp_weights
            ),
            "scheduled_auxiliary_pmp_anneal": (
                "linear to zero over the first 70% of each stage"
            ),
            "output_scale": args.output_scale,
            "initial_global_temperature": args.output_temperature,
            "temperature_learned": True,
        },
        "common_refinement_procedure": {
            "scope_statement": (
                "Identical post-training procedure applied to every arm; "
                "not identified as the paper's main/formal/current method."
            ),
            "method": "L-BFGS on complete RK4 projected-gradient residual",
            "outer_steps": args.refine_outer_steps,
            "inner_iterations": args.refine_inner_iterations,
            "learning_rate": args.refine_lr,
            "history_size": args.refine_history_size,
            "p": args.refine_p,
            "high_p_weight": args.refine_high_p_weight,
            "selection": "projected-gradient Linf then RMS",
            "reported_costs": ["closure_calls", "wall_seconds"],
        },
        "direct_reference_policy": (
            "The direct-reference artifact is not read until all requested "
            "neural and refinement completion markers exist."
        ),
        "calibration_file": str(out_root / "loss_calibration.json"),
    }


def validate_json_content(
    path: Path,
    expected: Any,
    *,
    label: str,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    actual = json.loads(path.read_text(encoding="utf-8"))
    if canonical_hash(actual) != canonical_hash(expected):
        raise RuntimeError(f"{label} canonical-content mismatch: {path}")


def initialize_or_validate_study_root(
    out_root: Path,
    args: argparse.Namespace,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, Any],
    dict[str, str],
    dict[str, Any],
]:
    """Create the immutable study identity, or strictly validate it."""
    manifest_path = out_root / "IMMUTABLE_STUDY_MANIFEST.json"
    sources = source_hashes()
    sources_hash = canonical_hash(sources)
    args_record = canonical_args_record(args, out_root)
    args_hash = canonical_hash(args_record)
    config = frozen_config_record(args)
    config_hash = canonical_hash(config)
    environment = environment_record()
    environment_hash = canonical_hash(environment)

    if not out_root.exists():
        out_root.mkdir(parents=True, exist_ok=False)
        command_record = {
            "argv": [sys.executable, *sys.argv],
            "cwd": str(ROOT),
            "canonical_args_hash": args_hash,
            "source_bundle_hash": sources_hash,
        }
        write_json(out_root / "command.json", command_record)
        write_json(out_root / "canonical_args.json", args_record)
        write_json(out_root / "environment.json", environment)
        write_json(out_root / "frozen_config.json", config)
        common_state, initialization_digest = prepare_common_initialization(
            out_root,
            args,
        )
        calibrations = prepare_calibrations(out_root, common_state, args)
        protocol = build_protocol(
            out_root,
            common_state,
            initialization_digest,
            args,
        )
        write_json(out_root / "protocol.json", protocol)
        snapshots = snapshot_sources(out_root, sources)
        metadata_paths = {
            "command": out_root / "command.json",
            "canonical_args": out_root / "canonical_args.json",
            "environment": out_root / "environment.json",
            "frozen_config": out_root / "frozen_config.json",
            "common_initialization": out_root / "common_initialization.pt",
            "loss_calibration": out_root / "loss_calibration.json",
            "protocol": out_root / "protocol.json",
        }
        manifest: dict[str, Any] = {
            "schema_version": STUDY_SCHEMA_VERSION,
            "study": "controlled time-only training-loss family study",
            "created_unix_time": time.time(),
            "root": str(out_root),
            "seed": args.seed,
            "families": list(args.families),
            "runner_sha256": sources["runner"]["sha256"],
            "refinement_sha256": sources["common_refinement"]["sha256"],
            "critical_import_sha256": {
                label: record["sha256"]
                for label, record in sources.items()
                if label not in {"runner", "common_refinement"}
            },
            "source_hashes": sources,
            "source_bundle_hash": sources_hash,
            "source_snapshot_inventory": snapshots,
            "canonical_args": args_record,
            "canonical_args_hash": args_hash,
            "frozen_config_hash": config_hash,
            "environment_hash": environment_hash,
            "initialization": {
                "path": str(out_root / "common_initialization.pt"),
                "file_sha256": sha256(
                    out_root / "common_initialization.pt"
                ),
                "state_digest": initialization_digest,
            },
            "calibration": {
                "path": str(out_root / "loss_calibration.json"),
                "file_sha256": sha256(out_root / "loss_calibration.json"),
                "canonical_hash": canonical_hash(calibrations),
                "reference_policy": (
                    "common reference initialization only; never "
                    "re-estimated at an arm stage start"
                ),
            },
            "protocol": {
                "path": str(out_root / "protocol.json"),
                "file_sha256": sha256(out_root / "protocol.json"),
                "canonical_hash": canonical_hash(protocol),
            },
            "metadata_artifact_sha256": {
                label: sha256(path) for label, path in metadata_paths.items()
            },
        }
        manifest["manifest_hash"] = canonical_hash(manifest)
        write_json(manifest_path, manifest)
    else:
        if not manifest_path.is_file():
            raise FileExistsError(
                "existing output root has no immutable study manifest; "
                f"refusing reuse: {out_root}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        hashed_payload = dict(manifest)
        recorded_manifest_hash = str(hashed_payload.pop("manifest_hash", ""))
        if canonical_hash(hashed_payload) != recorded_manifest_hash:
            raise RuntimeError("immutable study manifest self-hash mismatch")
        if int(manifest.get("schema_version", -1)) != STUDY_SCHEMA_VERSION:
            raise RuntimeError("study schema version mismatch")
        if manifest.get("root") != str(out_root):
            raise RuntimeError("immutable study root path mismatch")
        if int(manifest.get("seed", -1)) != args.seed:
            raise RuntimeError("immutable study seed mismatch")
        if manifest.get("families") != list(args.families):
            raise RuntimeError("immutable study family list mismatch")
        if manifest.get("source_hashes") != sources:
            raise RuntimeError("current source hashes do not match frozen study")
        if manifest.get("source_bundle_hash") != sources_hash:
            raise RuntimeError("current source bundle hash mismatch")
        if manifest.get("canonical_args_hash") != args_hash:
            raise RuntimeError("canonical argument hash mismatch")
        if manifest.get("canonical_args") != args_record:
            raise RuntimeError("canonical argument record mismatch")
        if manifest.get("frozen_config_hash") != config_hash:
            raise RuntimeError("frozen configuration hash mismatch")
        if manifest.get("environment_hash") != environment_hash:
            raise RuntimeError("execution environment hash mismatch")

        metadata_paths = {
            "command": out_root / "command.json",
            "canonical_args": out_root / "canonical_args.json",
            "environment": out_root / "environment.json",
            "frozen_config": out_root / "frozen_config.json",
            "common_initialization": out_root / "common_initialization.pt",
            "loss_calibration": out_root / "loss_calibration.json",
            "protocol": out_root / "protocol.json",
        }
        recorded_metadata = dict(manifest["metadata_artifact_sha256"])
        if set(recorded_metadata) != set(metadata_paths):
            raise RuntimeError("metadata artifact set mismatch")
        for label, path in metadata_paths.items():
            validate_file_hash(path, recorded_metadata[label], label)
        validate_json_content(
            out_root / "canonical_args.json",
            args_record,
            label="canonical arguments",
        )
        validate_json_content(
            out_root / "environment.json",
            environment,
            label="environment record",
        )
        validate_json_content(
            out_root / "frozen_config.json",
            config,
            label="frozen configuration",
        )

        init_payload = torch.load(
            out_root / "common_initialization.pt",
            map_location="cpu",
            weights_only=False,
        )
        common_state = dict(init_payload["model_state"])
        initialization_digest = state_digest(common_state)
        if initialization_digest != manifest["initialization"]["state_digest"]:
            raise RuntimeError("common initialization state digest mismatch")
        if int(init_payload["seed"]) != args.seed:
            raise RuntimeError("common initialization seed mismatch")
        validate_file_hash(
            out_root / "common_initialization.pt",
            manifest["initialization"]["file_sha256"],
            "common initialization",
        )
        calibrations = json.loads(
            (out_root / "loss_calibration.json").read_text(encoding="utf-8")
        )
        if canonical_hash(calibrations) != manifest["calibration"]["canonical_hash"]:
            raise RuntimeError("loss calibration canonical hash mismatch")
        validate_file_hash(
            out_root / "loss_calibration.json",
            manifest["calibration"]["file_sha256"],
            "loss calibration",
        )
        protocol = build_protocol(
            out_root,
            common_state,
            initialization_digest,
            args,
        )
        validate_json_content(
            out_root / "protocol.json",
            protocol,
            label="protocol",
        )
        if canonical_hash(protocol) != manifest["protocol"]["canonical_hash"]:
            raise RuntimeError("protocol canonical hash mismatch")
        validate_file_hash(
            out_root / "protocol.json",
            manifest["protocol"]["file_sha256"],
            "protocol",
        )
        snapshots = dict(manifest["source_snapshot_inventory"])
        if set(snapshots) != set(sources):
            raise RuntimeError("source snapshot set mismatch")
        for label, record in snapshots.items():
            if record["source_path"] != sources[label]["path"]:
                raise RuntimeError(f"snapshot source path mismatch for {label}")
            if record["source_sha256"] != sources[label]["sha256"]:
                raise RuntimeError(f"snapshot source hash mismatch for {label}")
            validate_file_hash(
                Path(record["snapshot_path"]),
                record["snapshot_sha256"],
                f"{label} source snapshot",
            )

    common_payload = torch.load(
        out_root / "common_initialization.pt",
        map_location="cpu",
        weights_only=False,
    )
    common_state = dict(common_payload["model_state"])
    calibrations = json.loads(
        (out_root / "loss_calibration.json").read_text(encoding="utf-8")
    )
    identity = {
        "study_manifest_hash": str(manifest["manifest_hash"]),
        "runner_sha256": str(manifest["runner_sha256"]),
        "refinement_sha256": str(manifest["refinement_sha256"]),
        "source_bundle_hash": str(manifest["source_bundle_hash"]),
        "canonical_args_hash": str(manifest["canonical_args_hash"]),
        "frozen_config_hash": str(manifest["frozen_config_hash"]),
        "environment_hash": str(manifest["environment_hash"]),
        "initialization_digest": str(
            manifest["initialization"]["state_digest"]
        ),
        "calibration_hash": str(manifest["calibration"]["canonical_hash"]),
        "protocol_hash": str(manifest["protocol"]["canonical_hash"]),
    }
    return common_state, calibrations, identity, manifest


def root_artifact_inventory(root: Path) -> dict[str, str]:
    inventory = relative_inventory(root, excluded_names=set())
    inventory.pop("COMPLETED.json", None)
    return inventory


def validate_completed_study(
    root: Path,
    args: argparse.Namespace,
    identity: dict[str, str],
) -> list[dict[str, Any]]:
    marker = json.loads((root / "COMPLETED.json").read_text(encoding="utf-8"))
    if dict(marker.get("study_identity", {})) != identity:
        raise RuntimeError("completed-study identity mismatch")
    if marker.get("families") != list(args.families):
        raise RuntimeError("completed-study family list mismatch")
    if int(marker.get("seed", -1)) != args.seed:
        raise RuntimeError("completed-study seed mismatch")
    actual = root_artifact_inventory(root)
    expected = dict(marker["artifact_inventory"])
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(
            key
            for key in set(expected) & set(actual)
            if expected[key] != actual[key]
        )
        raise RuntimeError(
            "completed-study artifact inventory mismatch; "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    direct_summary = resolve(args.direct_summary)
    validate_file_hash(
        direct_summary,
        str(marker["direct_reference_sha256"]),
        "direct reference",
    )
    comparison = json.loads(
        (root / "comparison.json").read_text(encoding="utf-8")
    )
    print("[validated skip] immutable completed study", flush=True)
    return list(comparison["rows"])


def print_comparison(rows: Iterable[dict[str, Any]]) -> None:
    print("\nUnified post-hoc comparison", flush=True)
    for row in rows:
        print(
            f"{row['family']:>24s} {row['phase']:>22s} "
            f"gap={row['relative_gap_vs_direct_percent']:.6g}% "
            f"PGinf={row['projected_gradient_linf']:.3e} "
            f"PMP={row['euler_pmp_kkt_opt_gap']:.3e} "
            f"closures={int(row['refinement_closure_calls'])} "
            f"wall={row['cumulative_wall_seconds']:.1f}s",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--families", type=parse_families, default=FAMILIES)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--init-u", type=float, default=1.5)
    parser.add_argument("--pretrain-epochs", type=int, default=1200)
    parser.add_argument("--pretrain-lr", type=float, default=5.0e-4)
    parser.add_argument("--pretrain-grad-clip", type=float, default=10.0)
    parser.add_argument("--pretrain-eval-every", type=int, default=100)
    parser.add_argument(
        "--continuation-epochs",
        type=lambda value: parse_int_tuple(value, expected=3),
        default=(160, 120, 100),
    )
    parser.add_argument(
        "--continuation-lrs",
        type=lambda value: parse_float_tuple(value, expected=3),
        default=(2.0e-5, 8.0e-6, 3.0e-6),
    )
    parser.add_argument(
        "--linf-weights",
        type=lambda value: parse_float_tuple(value, expected=3),
        default=(0.05, 0.10, 0.20),
    )
    parser.add_argument(
        "--staged-aux-pmp-weights",
        type=lambda value: tuple(
            float(item.strip()) for item in value.split(",") if item.strip()
        ),
        default=(0.02, 0.005, 0.0),
        help=(
            "PMP/KKT auxiliary weights for normalized_staged_pg_pmp "
            "continuation; "
            "each is linearly annealed to zero by 70%% of its stage"
        ),
    )
    parser.add_argument("--continuation-grad-clip", type=float, default=5.0)
    parser.add_argument("--continuation-eval-every", type=int, default=20)
    parser.add_argument("--output-scale", type=float, default=1.08)
    parser.add_argument(
        "--output-temperature",
        type=float,
        default=0.6991330744962188,
    )
    parser.add_argument("--refine-outer-steps", type=int, default=6)
    parser.add_argument("--refine-inner-iterations", type=int, default=5)
    parser.add_argument("--refine-lr", type=float, default=0.2)
    parser.add_argument("--refine-history-size", type=int, default=20)
    parser.add_argument("--refine-p", type=float, default=20.0)
    parser.add_argument("--refine-high-p-weight", type=float, default=3.0)
    parser.add_argument(
        "--direct-summary",
        default=str(DEFAULT_DIRECT_SUMMARY),
    )
    parser.add_argument(
        "--skip-refinement",
        action="store_true",
        help="Train/freeze neural arms only; comparison remains blinded.",
    )
    args = parser.parse_args()

    if args.d_model != 64 or args.heads != 4 or args.layers != 2:
        parser.error("controlled study requires d_model=64, heads=4, layers=2")
    if args.seed not in {21, 22, 23}:
        parser.error("controlled study is frozen to seeds 21, 22, or 23")
    if (
        len(args.staged_aux_pmp_weights) != 3
        or any(
            not math.isfinite(value) or value < 0.0
            for value in args.staged_aux_pmp_weights
        )
    ):
        parser.error("--staged-aux-pmp-weights requires three nonnegative values")
    if min(
        args.pretrain_epochs,
        *args.continuation_epochs,
        args.pretrain_eval_every,
        args.continuation_eval_every,
    ) <= 0:
        parser.error("all epoch/evaluation counts must be positive")

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out_root = resolve(args.out_root)
    common_state, calibrations, run_identity, manifest = (
        initialize_or_validate_study_root(
            out_root,
            args,
        )
    )
    if (out_root / "COMPLETED.json").exists():
        rows = validate_completed_study(out_root, args, run_identity)
        print_comparison(rows)
        return

    neural_summaries = []
    for family in args.families:
        neural_summaries.append(
            train_family(
                family=family,
                root=out_root,
                common_state=common_state,
                calibrations=calibrations,
                args=args,
                run_identity=run_identity,
            )
        )
    write_or_validate_json(
        out_root / "neural_summaries.json",
        neural_summaries,
    )

    if args.skip_refinement:
        print("Neural arms frozen; refinement and unblinded comparison skipped.")
        return

    refinement_summaries = {}
    for family in args.families:
        refinement_summaries[family] = run_refinement(
            family,
            out_root,
            args,
            run_identity,
        )
    write_or_validate_json(
        out_root / "refinement_summaries.json",
        refinement_summaries,
    )
    direct_summary = resolve(args.direct_summary)
    rows = build_comparison(
        out_root,
        args.families,
        direct_summary,
    )
    comparison_json = json.loads(
        (out_root / "comparison.json").read_text(encoding="utf-8")
    )
    completion = {
        "completed_unix_time": time.time(),
        "study_identity": run_identity,
        "study_manifest_hash": manifest["manifest_hash"],
        "families": list(args.families),
        "seed": args.seed,
        "comparison": str(out_root / "comparison.csv"),
        "comparison_csv_sha256": sha256(out_root / "comparison.csv"),
        "comparison_json_sha256": sha256(out_root / "comparison.json"),
        "direct_reference": str(direct_summary),
        "direct_reference_sha256": comparison_json[
            "direct_reference_sha256"
        ],
        "artifact_inventory": root_artifact_inventory(out_root),
    }
    write_json(out_root / "COMPLETED.json", completion)
    print_comparison(rows)


if __name__ == "__main__":
    main()
