#!/usr/bin/env python3
"""Paired evaluation of pre/post feedback-policy refinement.

The comparison separates a change in the nominal open-loop schedule from the
state-feedback contribution.  For each checkpoint, its control realized from
the nominal initial state is frozen and replayed on every perturbed initial
state.  The feedback advantage is

    A_theta(N0) = J_N0(u_theta,nom) - J_N0(pi_theta).

The physical objective is used only for this held-out evaluation.
"""

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

from evaluate_feedback_section5 import (
    configured_state_mode,
    load_feedback_checkpoint,
    projected_gradient_linf,
    rk4_zoh_feedback,
    rk4_zoh_open_loop,
)
from train_feedback_section5 import make_fixed_directions, test_states_from_directions
from train_paper_pmp_kkt import ProblemConfig, build_params


DIRECTION_FAMILIES = ("random", "total", "composition")
DIRECTION_SEED_OFFSETS = {
    "random": 0,
    "total": 104729,
    "composition": 209759,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cfg_with_n(cfg: ProblemConfig, n: int) -> ProblemConfig:
    values = asdict(cfg)
    values["n"] = int(n)
    return ProblemConfig(**values)


def interpolate_nominal_reference(model: torch.nn.Module, n: int) -> None:
    stored = model.nominal_reference.detach().cpu().numpy()
    source_t = np.linspace(0.0, 1.0, stored.shape[0], dtype=np.float64)
    target_t = np.linspace(0.0, 1.0, n + 1, dtype=np.float64)
    interpolated = np.column_stack(
        [np.interp(target_t, source_t, stored[:, j]) for j in range(stored.shape[1])]
    )
    model.set_nominal_reference(torch.tensor(interpolated, dtype=torch.float64))


def prepare(path: Path, n: int):
    model, source_cfg, args = load_feedback_checkpoint(path)
    cfg = cfg_with_n(source_cfg, n)
    interpolate_nominal_reference(model, n)
    model.eval()
    return model, cfg, args


def stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "positive_fraction": float(np.mean(values > 0.0)),
    }


def advantage_stats(
    values: np.ndarray, *, seed: int, repeats: int
) -> dict[str, Any]:
    """Summarize a paired objective advantage (positive favors feedback)."""

    summary: dict[str, Any] = stats(values)
    summary.update(
        {
            "count": int(np.asarray(values).size),
            "win_fraction": summary["positive_fraction"],
            "worst": summary["min"],
            "best": summary["max"],
            "mean_bootstrap_95ci": bootstrap_mean_ci(
                values, seed=seed, repeats=repeats
            ),
        }
    )
    return summary


def bootstrap_mean_ci(
    values: np.ndarray, *, seed: int, repeats: int
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(repeats, values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def make_held_out_direction_families(
    count: int,
    m: int,
    seed: int,
    dtype: torch.dtype,
    families: list[str],
) -> tuple[torch.Tensor, list[str], list[str]]:
    """Create reproducible random, total-burden, and composition perturbations.

    ``total`` changes every phenotype by the same relative amount.  The first
    two samples are the positive and negative radius endpoints; the remaining
    amplitudes are held-out uniform draws.  ``composition`` has zero component
    sum, so it changes phenotype composition without changing the initial total
    population.  Those directions are normalized to unit L-infinity radius.
    """

    if count <= 0:
        raise ValueError("samples per direction family must be positive")
    if m <= 1:
        raise ValueError("at least two phenotype components are required")
    unknown = sorted(set(families).difference(DIRECTION_FAMILIES))
    if unknown:
        raise ValueError(
            f"unknown direction families {unknown}; choose from {DIRECTION_FAMILIES}"
        )
    if not families:
        raise ValueError("at least one direction family is required")
    if len(set(families)) != len(families):
        raise ValueError("direction families must not be repeated")

    blocks: list[torch.Tensor] = []
    labels: list[str] = []
    names: list[str] = []
    for family in families:
        family_seed = int(seed + DIRECTION_SEED_OFFSETS[family])
        if family == "random":
            block = make_fixed_directions(count, m, family_seed, dtype)
            family_names = [f"random_{index:04d}" for index in range(count)]
        elif family == "total":
            generator = torch.Generator(device="cpu").manual_seed(family_seed)
            amplitudes = (
                2.0 * torch.rand(count, generator=generator, dtype=dtype) - 1.0
            )
            if count >= 1:
                amplitudes[0] = 1.0
            if count >= 2:
                amplitudes[1] = -1.0
            block = amplitudes[:, None].expand(-1, m).clone()
            family_names = [f"total_{index:04d}" for index in range(count)]
            if count >= 1:
                family_names[0] = "total_plus_endpoint"
            if count >= 2:
                family_names[1] = "total_minus_endpoint"
        else:
            generator = torch.Generator(device="cpu").manual_seed(family_seed)
            raw = 2.0 * torch.rand(count, m, generator=generator, dtype=dtype) - 1.0
            block = raw - raw.mean(dim=1, keepdim=True)
            scale = block.abs().amax(dim=1, keepdim=True).clamp_min(1.0e-12)
            block = block / scale
            family_names = [f"composition_{index:04d}" for index in range(count)]
            trait = torch.linspace(-1.0, 1.0, m, dtype=dtype)
            trait = trait - trait.mean()
            trait = trait / trait.abs().max()
            if count >= 1:
                block[0] = trait
                family_names[0] = "resistant_heavy_endpoint"
            if count >= 2:
                block[1] = -trait
                family_names[1] = "sensitive_heavy_endpoint"
        blocks.append(block)
        labels.extend([family] * count)
        names.extend(family_names)
    return torch.cat(blocks, dim=0), labels, names


def evaluate_checkpoint(
    path: Path,
    n: int,
    initial: torch.Tensor,
    *,
    substeps: int,
    pg_count: int,
) -> dict[str, Any]:
    model, cfg, args = prepare(path, n)
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    nominal_initial = torch.full((1, cfg.m), cfg.n0, dtype=torch.float64)
    with torch.no_grad():
        nominal_J, nominal_control_batch, _ = rk4_zoh_feedback(
            model,
            nominal_initial,
            cfg,
            params,
            substeps=substeps,
            state_blind=bool(getattr(args, "state_blind", False)),
            state_mode=configured_state_mode(args),
        )
        feedback_J, feedback_controls, _ = rk4_zoh_feedback(
            model,
            initial,
            cfg,
            params,
            substeps=substeps,
            state_blind=bool(getattr(args, "state_blind", False)),
            state_mode=configured_state_mode(args),
        )
        nominal_control = nominal_control_batch[0]
        frozen_controls = nominal_control.expand(initial.shape[0], -1)
        frozen_J, _ = rk4_zoh_open_loop(
            frozen_controls, initial, cfg, params, substeps=substeps
        )

    selected = min(int(pg_count), int(initial.shape[0]))
    feedback_pg = projected_gradient_linf(
        feedback_controls[:selected],
        initial[:selected],
        cfg,
        params,
        substeps=substeps,
    )
    frozen_pg = projected_gradient_linf(
        frozen_controls[:selected],
        initial[:selected],
        cfg,
        params,
        substeps=substeps,
    )
    nominal_pg = projected_gradient_linf(
        nominal_control.unsqueeze(0),
        nominal_initial,
        cfg,
        params,
        substeps=substeps,
    )[0]
    dt = cfg.T / cfg.n
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "source_n": int(ProblemConfig(**torch.load(path, map_location="cpu", weights_only=False)["problem"]).n),
        "nominal_J": float(nominal_J[0]),
        "nominal_projected_gradient_linf": float(nominal_pg),
        "nominal_projected_gradient_linf_over_dt": float(nominal_pg / dt),
        "nominal_control": nominal_control.detach().cpu().numpy(),
        "feedback_J": feedback_J.detach().cpu().numpy(),
        "frozen_nominal_J": frozen_J.detach().cpu().numpy(),
        "feedback_advantage": (frozen_J - feedback_J).detach().cpu().numpy(),
        "feedback_controls": feedback_controls.detach().cpu().numpy(),
        "feedback_pg": feedback_pg.detach().cpu().numpy(),
        "frozen_pg": frozen_pg.detach().cpu().numpy(),
        "dt": dt,
    }


def paired_summary_block(
    pre: dict[str, Any],
    post: dict[str, Any],
    indices: np.ndarray,
    *,
    bootstrap_seed: int,
    bootstrap_repeats: int,
) -> dict[str, Any]:
    """Build paired statistics for one set of identical initial conditions."""

    pre_feedback = pre["feedback_J"][indices]
    post_feedback = post["feedback_J"][indices]
    pre_frozen = pre["frozen_nominal_J"][indices]
    post_frozen = post["frozen_nominal_J"][indices]
    pre_advantage = pre["feedback_advantage"][indices]
    post_advantage = post["feedback_advantage"][indices]
    total_improvement = pre_feedback - post_feedback
    nominal_schedule_improvement = pre_frozen - post_frozen
    feedback_specific_improvement = post_advantage - pre_advantage
    decomposition_error = (
        total_improvement
        - nominal_schedule_improvement
        - feedback_specific_improvement
    )
    return {
        "count": int(indices.size),
        "pre": {
            "feedback_J": stats(pre_feedback),
            "frozen_nominal_J": stats(pre_frozen),
            "feedback_advantage": advantage_stats(
                pre_advantage,
                seed=bootstrap_seed,
                repeats=bootstrap_repeats,
            ),
        },
        "post": {
            "feedback_J": stats(post_feedback),
            "frozen_nominal_J": stats(post_frozen),
            "feedback_advantage": advantage_stats(
                post_advantage,
                seed=bootstrap_seed + 1,
                repeats=bootstrap_repeats,
            ),
        },
        "total_improvement": advantage_stats(
            total_improvement,
            seed=bootstrap_seed + 2,
            repeats=bootstrap_repeats,
        ),
        "nominal_schedule_improvement": advantage_stats(
            nominal_schedule_improvement,
            seed=bootstrap_seed + 3,
            repeats=bootstrap_repeats,
        ),
        "feedback_specific_improvement": advantage_stats(
            feedback_specific_improvement,
            seed=bootstrap_seed + 4,
            repeats=bootstrap_repeats,
        ),
        "maximum_decomposition_error": float(np.max(np.abs(decomposition_error))),
    }


def compare_pair(
    pre_path: Path,
    post_path: Path,
    n: int,
    directions: torch.Tensor,
    direction_labels: list[str],
    direction_names: list[str],
    radii: list[float],
    *,
    substeps: int,
    pg_count: int,
    bootstrap_repeats: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pre_model, pre_cfg, _ = prepare(pre_path, n)
    post_model, post_cfg, _ = prepare(post_path, n)
    del pre_model, post_model
    if pre_cfg != post_cfg:
        raise ValueError("pre and post checkpoints define different physical problems")
    if directions.shape[0] != len(direction_labels):
        raise ValueError("direction_labels must have one entry per direction")
    if directions.shape[0] != len(direction_names):
        raise ValueError("direction_names must have one entry per direction")
    label_array = np.asarray(direction_labels, dtype=object)
    family_order = list(dict.fromkeys(direction_labels))

    summary: dict[str, Any] = {
        "pre_checkpoint": str(pre_path.resolve()),
        "post_checkpoint": str(post_path.resolve()),
        "radii": {},
    }
    rows: list[dict[str, Any]] = []
    for radius_index, radius in enumerate(radii):
        initial = test_states_from_directions(
            directions, radius, pre_cfg, torch.device("cpu"), torch.float64
        )
        pre = evaluate_checkpoint(
            pre_path, n, initial, substeps=substeps, pg_count=pg_count
        )
        post = evaluate_checkpoint(
            post_path, n, initial, substeps=substeps, pg_count=pg_count
        )
        all_indices = np.arange(initial.shape[0], dtype=np.int64)
        radius_summary = paired_summary_block(
            pre,
            post,
            all_indices,
            bootstrap_seed=bootstrap_seed + radius_index * 100,
            bootstrap_repeats=bootstrap_repeats,
        )
        radius_summary["pre"].update(
            {
                "nominal_J": pre["nominal_J"],
                "nominal_projected_gradient_linf": pre[
                    "nominal_projected_gradient_linf"
                ],
                "feedback_projected_gradient_linf": stats(pre["feedback_pg"]),
                "feedback_projected_gradient_linf_over_dt": stats(
                    pre["feedback_pg"] / pre["dt"]
                ),
            }
        )
        radius_summary["post"].update(
            {
                "nominal_J": post["nominal_J"],
                "nominal_projected_gradient_linf": post[
                    "nominal_projected_gradient_linf"
                ],
                "feedback_projected_gradient_linf": stats(post["feedback_pg"]),
                "feedback_projected_gradient_linf_over_dt": stats(
                    post["feedback_pg"] / post["dt"]
                ),
            }
        )
        radius_summary["by_direction_family"] = {}
        for family_index, family in enumerate(family_order):
            family_indices = np.flatnonzero(label_array == family)
            radius_summary["by_direction_family"][family] = paired_summary_block(
                pre,
                post,
                family_indices,
                bootstrap_seed=(
                    bootstrap_seed + radius_index * 100 + 10 * (family_index + 1)
                ),
                bootstrap_repeats=bootstrap_repeats,
            )

        total_improvement = pre["feedback_J"] - post["feedback_J"]
        nominal_schedule_improvement = (
            pre["frozen_nominal_J"] - post["frozen_nominal_J"]
        )
        feedback_specific_improvement = (
            post["feedback_advantage"] - pre["feedback_advantage"]
        )
        summary["radii"][f"{radius:.2f}"] = radius_summary
        for index in range(initial.shape[0]):
            rows.append(
                {
                    "radius": radius,
                    "sample": index,
                    "direction_family": direction_labels[index],
                    "direction_name": direction_names[index],
                    "direction_linf": float(directions[index].abs().max()),
                    "direction_mean": float(directions[index].mean()),
                    "pre_feedback_J": float(pre["feedback_J"][index]),
                    "post_feedback_J": float(post["feedback_J"][index]),
                    "pre_frozen_nominal_J": float(pre["frozen_nominal_J"][index]),
                    "post_frozen_nominal_J": float(post["frozen_nominal_J"][index]),
                    "pre_feedback_advantage": float(pre["feedback_advantage"][index]),
                    "post_feedback_advantage": float(post["feedback_advantage"][index]),
                    "total_improvement": float(total_improvement[index]),
                    "nominal_schedule_improvement": float(
                        nominal_schedule_improvement[index]
                    ),
                    "feedback_specific_improvement": float(
                        feedback_specific_improvement[index]
                    ),
                }
            )
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre", type=Path, required=True)
    parser.add_argument("--post", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--radii", default="0.05,0.10,0.20,0.30")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument(
        "--direction_families",
        default=",".join(DIRECTION_FAMILIES),
        help="comma-separated subset of random,total,composition",
    )
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--pg_count", type=int, default=16)
    parser.add_argument("--bootstrap_repeats", type=int, default=5000)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    radii = [float(value) for value in args.radii.split(",") if value.strip()]
    families = [
        value.strip()
        for value in args.direction_families.split(",")
        if value.strip()
    ]
    _, direction_cfg, _ = prepare(args.pre.resolve(), args.n)
    directions, direction_labels, direction_names = make_held_out_direction_families(
        args.samples, direction_cfg.m, args.seed, torch.float64, families
    )
    summary, rows = compare_pair(
        args.pre.resolve(),
        args.post.resolve(),
        args.n,
        directions,
        direction_labels,
        direction_names,
        radii,
        substeps=args.substeps,
        pg_count=args.pg_count,
        bootstrap_repeats=args.bootstrap_repeats,
        bootstrap_seed=args.seed + 1000,
    )
    summary["protocol"] = {
        "n": args.n,
        "radii": radii,
        "direction_families": families,
        "samples_per_direction_family": args.samples,
        "total_samples_per_radius": len(direction_labels),
        "direction_seed": args.seed,
        "initial_state_formula": "N0_perturbed = N0_nominal * (1 + radius * direction)",
        "direction_definitions": {
            "random": "independent component-wise Uniform[-1,1] relative perturbations",
            "total": "common-mode relative perturbations; all phenotype components move together",
            "composition": "zero-sum relative perturbations with unit L-infinity direction norm",
        },
        "rk4_substeps_per_control_interval": args.substeps,
        "projected_gradient_samples_per_radius": args.pg_count,
        "bootstrap_repeats": args.bootstrap_repeats,
        "paired_comparison": (
            "for each checkpoint and perturbed initial state, compare closed-loop "
            "feedback with that checkpoint's nominal-state control frozen and replayed"
        ),
        "objective_role": "held-out evaluation only; never a training loss",
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (out_dir / "per_sample.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    np.savez(
        out_dir / "directions.npz",
        directions=directions.numpy(),
        direction_family=np.asarray(direction_labels),
        direction_name=np.asarray(direction_names),
    )
    print(out_dir, flush=True)


if __name__ == "__main__":
    main()
