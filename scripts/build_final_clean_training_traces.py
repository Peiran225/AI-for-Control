#!/usr/bin/env python3
"""Plot the final scalar PMP/KKT refinement traces for the three policies.

The figure deliberately shows one internally consistent refinement stage per
policy.  It does not concatenate stages whose grid density or residual weights
change, because their numerical loss values are not directly comparable.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "time": "#2468A2",
    "cf": "#2F7D5A",
    "der": "#D05A32",
    "validation": "#8D98A3",
    "grid": "#D8DEE5",
    "text": "#25313C",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-history", type=Path, required=True)
    parser.add_argument("--cf-history", type=Path, required=True)
    parser.add_argument("--der-history", type=Path, required=True)
    parser.add_argument("--time-summary", type=Path)
    parser.add_argument("--cf-summary", type=Path)
    parser.add_argument("--der-summary", type=Path)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "mathtext.fontset": "stixsans",
            "font.size": 7.2,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.3,
            "axes.linewidth": 0.62,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.42,
            "grid.alpha": 0.68,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        columns = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                value = row.get(name, "")
                if value in (None, ""):
                    columns[name].append(np.nan)
                elif value in ("True", "False"):
                    columns[name].append(float(value == "True"))
                else:
                    columns[name].append(float(value))
    return {
        name: np.asarray(values, dtype=np.float64)
        for name, values in columns.items()
    }


def require_columns(
    data: dict[str, np.ndarray],
    names: tuple[str, ...],
    path: Path,
) -> None:
    missing = [name for name in names if name not in data]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")


def read_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: summary must be a JSON object")
    return value


def final_rms_text(summary: dict[str, Any] | None, policy: str) -> str | None:
    if summary is None:
        return None
    if policy == "time":
        metrics = summary.get("selected_scalar_dense_grid_metrics", {})
        keys = ("psi_rms", "dot_rms", "ddot_rms")
    else:
        metrics_by_state = summary.get("selected_validation_metrics_by_state", {})
        metrics = metrics_by_state.get("resistant_heavy", {})
        keys = (
            "H_u_physical_rms",
            "dH_u_dt_physical_rms",
            "d2H_u_dt2_physical_rms",
        )
    try:
        values = tuple(float(metrics[key]) for key in keys)
    except (KeyError, TypeError, ValueError):
        return None
    return "final physical RMS\n" + " / ".join(f"{value:.3g}" for value in values)


def positive(values: np.ndarray, path: Path, name: str) -> np.ndarray:
    if np.any(~np.isfinite(values)):
        raise ValueError(f"{path}: {name} contains non-finite values")
    if np.any(values <= 0.0):
        raise ValueError(f"{path}: {name} must be strictly positive for log scale")
    return values


def plot_time(
    axis: plt.Axes,
    path: Path,
    summary: dict[str, Any] | None,
) -> None:
    data = read_csv(path)
    require_columns(data, ("iteration", "weighted_objective"), path)
    step = data["iteration"]
    train = positive(data["weighted_objective"], path, "weighted_objective")
    axis.plot(step, train, "o-", color=COLORS["time"], lw=1.25, ms=3.0)
    axis.set(
        title="(a) PMP/KKT-Time",
        xlabel="LM update",
        ylabel="weighted residual loss",
        yscale="log",
    )


def plot_feedback(
    axis: plt.Axes,
    path: Path,
    summary: dict[str, Any] | None,
    policy: str,
    title: str,
    color: str,
) -> None:
    data = read_csv(path)
    require_columns(data, ("outer_step", "train_loss"), path)
    step = data["outer_step"]
    train = positive(data["train_loss"], path, "train_loss")
    axis.plot(
        step,
        train,
        "o-",
        color=color,
        lw=1.25,
        ms=3.0,
        label="training",
    )
    if "validation_loss" in data:
        validation = positive(data["validation_loss"], path, "validation_loss")
        axis.plot(
            step,
            validation,
            "s--",
            color=COLORS["validation"],
            lw=0.9,
            ms=2.7,
            label="validation",
        )
        axis.legend(frameon=False, loc="lower left")
    axis.set(
        title=title,
        xlabel="L-BFGS outer update",
        ylabel="scalar residual loss",
        yscale="log",
    )


def main() -> None:
    args = parse_args()
    outputs = (
        args.out_prefix.with_suffix(".pdf"),
        args.out_prefix.with_suffix(".png"),
    )
    if not args.overwrite:
        existing = [path for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite {existing}")
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    configure_style()
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(7.02, 2.18),
        gridspec_kw={"wspace": 0.34},
    )
    plot_time(axes[0], args.time_history, read_summary(args.time_summary))
    plot_feedback(
        axes[1],
        args.cf_history,
        read_summary(args.cf_summary),
        "cf",
        "(b) PMP/KKT-CF",
        COLORS["cf"],
    )
    plot_feedback(
        axes[2],
        args.der_history,
        read_summary(args.der_summary),
        "der",
        "(c) PMP/KKT-DER",
        COLORS["der"],
    )
    figure.text(
        0.995,
        0.008,
        "Each panel reports one fixed-grid refinement stage; "
        "loss scales differ across policies.",
        ha="right",
        va="bottom",
        fontsize=5.8,
        color=COLORS["text"],
    )
    figure.subplots_adjust(left=0.075, right=0.995, top=0.90, bottom=0.23)
    figure.savefig(outputs[0], bbox_inches="tight", pad_inches=0.025)
    figure.savefig(outputs[1], dpi=300, bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)
    print(outputs[0])
    print(outputs[1])


if __name__ == "__main__":
    main()
