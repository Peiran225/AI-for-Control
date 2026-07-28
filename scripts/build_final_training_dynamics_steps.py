#!/usr/bin/env python3
"""Plot the complete recorded training traces for the three final policies.

The upper row shows the initialization/adaptation histories at their recorded
optimizer steps.  The lower row shows the scalar PMP refinement stages that
lead to the selected checkpoints.  Losses from stages with different
definitions or query grids are never joined into one curve, and no values are
interpolated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TIME_INITIALIZATION = (
    ROOT
    / "outputs/time_only_refinement_choice_20260726/"
    "direct_reference_full_pipeline_stageab_v5/history.csv"
)
DEFAULT_RAW_DIR = ROOT / "output/training_traces_teacher_redo_20260727/raw"
DEFAULT_OUTPUT = (
    ROOT
    / "output/training_traces_teacher_redo_20260727/"
    "training_dynamics"
)

COLORS = {
    "time": "#2468A2",
    "cf": "#2F7D5A",
    "der": "#D05A32",
    "validation": "#8A96A3",
    "grid": "#D8DEE5",
    "text": "#27333D",
    "stage": "#7B858E",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--time-initialization",
        type=Path,
        default=DEFAULT_TIME_INITIALIZATION,
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "mathtext.fontset": "stixsans",
            "font.size": 6.8,
            "axes.titlesize": 8.2,
            "axes.labelsize": 6.9,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.9,
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
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        columns: dict[str, list[str]] = {
            name: [] for name in reader.fieldnames
        }
        for row in reader:
            for name in reader.fieldnames:
                columns[name].append(row.get(name, ""))
    result: dict[str, np.ndarray] = {}
    for name, values in columns.items():
        numeric: list[float] = []
        numeric_ok = True
        for value in values:
            try:
                numeric.append(float(value))
            except (TypeError, ValueError):
                numeric_ok = False
                break
        result[name] = (
            np.asarray(numeric, dtype=np.float64)
            if numeric_ok
            else np.asarray(values, dtype=object)
        )
    return result


def require_columns(
    data: dict[str, np.ndarray],
    names: Sequence[str],
    path: Path,
) -> None:
    missing = [name for name in names if name not in data]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")


def positive(values: np.ndarray, path: Path, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError(f"{path}: {name} must contain finite positive values")
    return result


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size < 2:
        return values.copy()
    width = max(1, min(int(window), int(values.size)))
    kernel = np.ones(width, dtype=np.float64) / float(width)
    padded = np.pad(values, (width - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plot_time_initialization(axis: plt.Axes, path: Path) -> dict[str, Any]:
    data = read_csv(path)
    require_columns(data, ("stage", "epoch", "loss"), path)
    mask = np.asarray(data["stage"], dtype=object) == "supervised"
    step = np.asarray(data["epoch"][mask], dtype=np.float64)
    loss = positive(data["loss"][mask], path, "loss")
    order = np.argsort(step)
    step, loss = step[order], loss[order]
    if step.size != 1500 or not np.array_equal(
        step, np.arange(1.0, 1501.0)
    ):
        raise ValueError(
            f"{path}: expected the complete 1,...,1500 initialization trace"
        )

    smooth = moving_average(loss, 31)
    marker_mask = (step % 50.0 == 0.0) | (step == step[0])
    axis.plot(step, loss, color=COLORS["time"], lw=0.42, alpha=0.28)
    axis.plot(step, smooth, color=COLORS["time"], lw=1.12)
    axis.plot(
        step[marker_mask],
        loss[marker_mask],
        linestyle="none",
        marker="o",
        ms=2.0,
        markerfacecolor="white",
        markeredgecolor=COLORS["time"],
        markeredgewidth=0.55,
        zorder=4,
    )
    axis.set_yscale("log")
    axis.set_xlabel("optimizer step")
    axis.set_ylabel("initialization loss")
    axis.text(
        0.98,
        0.95,
        "markers every 50 steps",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=5.5,
        color=COLORS["text"],
    )
    return {
        "path": str(path),
        "sha256": sha256(path),
        "recorded_steps": int(step.size),
        "first_step": int(step[0]),
        "last_step": int(step[-1]),
        "displayed_marker_interval": 50,
        "initial_loss": float(loss[0]),
        "final_loss": float(loss[-1]),
    }


def feedback_global_step(data: dict[str, np.ndarray], path: Path) -> np.ndarray:
    require_columns(data, ("dagger_round", "epoch"), path)
    rounds = np.asarray(data["dagger_round"], dtype=np.float64)
    epochs = np.asarray(data["epoch"], dtype=np.float64)
    if np.any(rounds < 0.0) or np.any(epochs <= 0.0):
        raise ValueError(f"{path}: invalid feedback training coordinates")
    maximum_epoch = int(np.max(epochs))
    if maximum_epoch != 1200:
        raise ValueError(f"{path}: expected 1200 optimizer steps per round")
    return rounds * maximum_epoch + epochs


def plot_feedback_initialization(
    axis: plt.Axes,
    path: Path,
    *,
    color: str,
) -> dict[str, Any]:
    data = read_csv(path)
    require_columns(data, ("teacher_forced_loss",), path)
    step = feedback_global_step(data, path)
    loss = positive(
        data["teacher_forced_loss"], path, "teacher_forced_loss"
    )
    order = np.argsort(step)
    step, loss = step[order], loss[order]
    if step.size != np.unique(step).size:
        raise ValueError(f"{path}: duplicate global feedback steps")

    smooth = moving_average(loss, 5)
    axis.plot(
        step,
        loss,
        color=color,
        lw=0.40,
        alpha=0.28,
        marker="o",
        ms=1.15,
        markeredgewidth=0.0,
    )
    axis.plot(step, smooth, color=color, lw=1.08)
    for boundary in np.arange(1200.0, float(step[-1]), 1200.0):
        axis.axvline(
            boundary,
            color=COLORS["stage"],
            lw=0.38,
            alpha=0.28,
            linestyle=(0, (1.5, 2.0)),
        )
    axis.set_yscale("log")
    axis.set_xlabel("optimizer step")
    axis.text(
        0.98,
        0.95,
        "loss recorded every 25 steps",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=5.5,
        color=COLORS["text"],
    )
    return {
        "path": str(path),
        "sha256": sha256(path),
        "recorded_values": int(step.size),
        "first_step": int(step[0]),
        "last_step": int(step[-1]),
        "recording_interval_after_first_step": 25,
        "initial_loss": float(loss[0]),
        "final_loss": float(loss[-1]),
    }


def selected_rows(
    data: dict[str, np.ndarray],
    *,
    step_column: str,
    selected_step: int | None,
) -> np.ndarray:
    step = np.asarray(data[step_column], dtype=np.float64)
    if selected_step is None:
        return np.ones(step.shape, dtype=bool)
    return step <= float(selected_step)


def plot_refinement_chain(
    axis: plt.Axes,
    stages: Sequence[tuple[str, Path, str, str, int | None]],
    *,
    color: str,
) -> dict[str, Any]:
    offset = 0
    stage_records: list[dict[str, Any]] = []
    final_training_x = None
    final_training_y = None
    for stage_index, (
        label,
        path,
        step_column,
        loss_column,
        selected_step,
    ) in enumerate(stages):
        data = read_csv(path)
        require_columns(data, (step_column, loss_column), path)
        mask = selected_rows(
            data,
            step_column=step_column,
            selected_step=selected_step,
        )
        local_step = np.asarray(data[step_column][mask], dtype=np.float64)
        training = positive(data[loss_column][mask], path, loss_column)
        order = np.argsort(local_step)
        local_step, training = local_step[order], training[order]
        x = offset + np.arange(training.size, dtype=np.float64)
        axis.plot(
            x,
            training,
            "o-",
            color=color,
            lw=1.00,
            ms=2.25,
            markerfacecolor="white",
            markeredgewidth=0.55,
        )

        validation_present = "validation_loss" in data
        if validation_present:
            validation = positive(
                data["validation_loss"][mask][order],
                path,
                "validation_loss",
            )
            axis.plot(
                x,
                validation,
                "s--",
                color=COLORS["validation"],
                lw=0.72,
                ms=1.8,
                markeredgewidth=0.0,
            )
        else:
            validation = None

        center = 0.5 * float(x[0] + x[-1])
        axis.text(
            center,
            0.95,
            label,
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=5.2,
            color=COLORS["text"],
        )
        final_training_x = float(x[-1])
        final_training_y = float(training[-1])
        stage_records.append(
            {
                "label": label,
                "path": str(path),
                "sha256": sha256(path),
                "points_plotted": int(training.size),
                "selected_step": selected_step,
                "initial_training_loss": float(training[0]),
                "final_training_loss": float(training[-1]),
                "validation_plotted": bool(validation_present),
                "final_validation_loss": (
                    float(validation[-1])
                    if validation is not None
                    else None
                ),
            }
        )
        offset += int(training.size) + 1
        if stage_index < len(stages) - 1:
            axis.axvline(
                offset - 1.0,
                color=COLORS["stage"],
                lw=0.55,
                alpha=0.65,
                linestyle=(0, (2.0, 2.0)),
            )

    if final_training_x is None or final_training_y is None:
        raise ValueError("refinement chain is empty")
    axis.plot(
        [final_training_x],
        [final_training_y],
        marker="*",
        ms=5.2,
        color=color,
        markeredgecolor="white",
        markeredgewidth=0.35,
        zorder=6,
    )
    axis.set_yscale("log")
    axis.set_xlabel("scalar refinement update")
    axis.set_ylabel("scalar residual loss")
    return {
        "stages": stage_records,
        "final_training_loss": final_training_y,
    }


def main() -> None:
    args = parse_args()
    output_pdf = args.out_prefix.with_suffix(".pdf")
    output_png = args.out_prefix.with_suffix(".png")
    output_json = args.out_prefix.with_name(
        args.out_prefix.name + "_manifest.json"
    )
    outputs = (output_pdf, output_png, output_json)
    if not args.overwrite:
        existing = [path for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite {existing}")
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    raw = args.raw_dir
    time_stages = (
        (
            "LM-1",
            raw / "time_lm_stage1.csv",
            "iteration",
            "weighted_objective",
            20,
        ),
        (
            "LM-2",
            raw / "time_lm_stage2.csv",
            "iteration",
            "weighted_objective",
            12,
        ),
        (
            "LM-3",
            raw / "time_lm_stage3.csv",
            "iteration",
            "weighted_objective",
            3,
        ),
    )
    cf_stages = (
        (
            "2x",
            raw / "cf_m2_initial.csv",
            "outer_step",
            "train_loss",
            2,
        ),
        (
            "2x cont.",
            raw / "cf_m2_continuation.csv",
            "outer_step",
            "train_loss",
            6,
        ),
        (
            "4x",
            raw / "cf_m4_continuation.csv",
            "outer_step",
            "train_loss",
            6,
        ),
        (
            "8x",
            raw / "cf_m8_final.csv",
            "outer_step",
            "train_loss",
            4,
        ),
    )
    der_stages = (
        (
            "2x",
            raw / "der_m2_initial.csv",
            "outer_step",
            "train_loss",
            2,
        ),
        (
            "2x cont.",
            raw / "der_m2_continuation.csv",
            "outer_step",
            "train_loss",
            8,
        ),
        (
            "4x",
            raw / "der_m4_continuation.csv",
            "outer_step",
            "train_loss",
            10,
        ),
        (
            "8x",
            raw / "der_m8_final.csv",
            "outer_step",
            "train_loss",
            6,
        ),
    )

    configure_style()
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(7.12, 4.20),
        gridspec_kw={"hspace": 0.38, "wspace": 0.34},
    )
    titles = (
        "(a) PMP/KKT-Time",
        "(b) PMP/KKT-CF",
        "(c) PMP/KKT-DER",
    )
    for axis, title in zip(axes[0], titles, strict=True):
        axis.set_title(title, loc="left", fontweight="semibold", pad=4.0)

    records: dict[str, Any] = {}
    records["time_initialization"] = plot_time_initialization(
        axes[0, 0],
        args.time_initialization,
    )
    feedback_history = raw / "feedback_initialization_history.csv"
    records["cf_initialization"] = plot_feedback_initialization(
        axes[0, 1],
        feedback_history,
        color=COLORS["cf"],
    )
    records["der_initialization"] = plot_feedback_initialization(
        axes[0, 2],
        feedback_history,
        color=COLORS["der"],
    )
    axes[0, 1].set_ylabel("feedback initialization loss")
    axes[0, 2].set_ylabel("feedback initialization loss")
    axes[0, 1].text(
        0.03,
        0.08,
        "shared initialization",
        transform=axes[0, 1].transAxes,
        fontsize=5.4,
        color=COLORS["text"],
    )
    axes[0, 2].text(
        0.03,
        0.08,
        "shared initialization",
        transform=axes[0, 2].transAxes,
        fontsize=5.4,
        color=COLORS["text"],
    )

    records["time_refinement"] = plot_refinement_chain(
        axes[1, 0],
        time_stages,
        color=COLORS["time"],
    )
    records["cf_refinement"] = plot_refinement_chain(
        axes[1, 1],
        cf_stages,
        color=COLORS["cf"],
    )
    records["der_refinement"] = plot_refinement_chain(
        axes[1, 2],
        der_stages,
        color=COLORS["der"],
    )

    raw_handle = Line2D(
        [0],
        [0],
        color=COLORS["text"],
        lw=0.5,
        alpha=0.35,
        marker="o",
        ms=2.0,
        label="recorded training loss",
    )
    smooth_handle = Line2D(
        [0],
        [0],
        color=COLORS["text"],
        lw=1.15,
        label="moving average",
    )
    validation_handle = Line2D(
        [0],
        [0],
        color=COLORS["validation"],
        lw=0.8,
        linestyle="--",
        marker="s",
        ms=2.2,
        label="validation loss (feedback refinement)",
    )
    figure.legend(
        handles=(raw_handle, smooth_handle, validation_handle),
        loc="upper center",
        bbox_to_anchor=(0.53, 0.995),
        ncol=3,
        frameon=False,
        columnspacing=1.15,
        handlelength=2.0,
    )
    figure.text(
        0.995,
        0.007,
        "All points are recorded optimizer values; no interpolation. "
        "Gaps in the lower row mark changes in query density.",
        ha="right",
        va="bottom",
        fontsize=5.5,
        color=COLORS["text"],
    )
    figure.subplots_adjust(
        left=0.078,
        right=0.994,
        bottom=0.105,
        top=0.905,
    )
    figure.savefig(output_pdf, bbox_inches="tight", pad_inches=0.025)
    figure.savefig(output_png, dpi=320, bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)

    manifest = {
        "schema": "final-training-dynamics-recorded-steps-v1",
        "no_interpolation": True,
        "selected_checkpoints_only": True,
        "records": records,
        "outputs": {
            "pdf": {
                "path": str(output_pdf),
                "sha256": sha256(output_pdf),
                "bytes": output_pdf.stat().st_size,
            },
            "png": {
                "path": str(output_png),
                "sha256": sha256(output_png),
                "bytes": output_png.stat().st_size,
            },
        },
    }
    output_json.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output_pdf)
    print(output_png)
    print(output_json)


if __name__ == "__main__":
    main()
