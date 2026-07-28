#!/usr/bin/env python3
"""Build final nominal and two-state phenotype figures from the DER m32 NPZ."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


STATES = ("nominal", "resistant_heavy")
STATE_LABELS = {
    "nominal": "Nominal",
    "resistant_heavy": "Resistant-heavy",
}
PROFILE_COLORS = (
    "#365E8D",
    "#277F8E",
    "#1FA187",
    "#73D055",
    "#F6C445",
    "#D9553B",
)
PROFILE_MARKERS = ("o", "s", "^", "D", "v", "P")
GRID_COLOR = "#DDE3E8"
TEXT_COLOR = "#303A43"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--der-npz", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "mathtext.fontset": "stixsans",
            "font.size": 7.0,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.3,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.1,
            "axes.linewidth": 0.62,
            "axes.grid": True,
            "grid.color": GRID_COLOR,
            "grid.alpha": 0.68,
            "grid.linewidth": 0.42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "path.simplify": False,
        }
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data(path: Path) -> dict[str, np.ndarray]:
    path = path.expanduser().resolve()
    require(path.is_file(), f"missing DER NPZ: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {"t", "is_transformer_support", "nominal__u"}
        required.update(f"{state}__N" for state in STATES)
        missing = sorted(required - set(archive.files))
        require(not missing, f"{path}: missing arrays {missing}")
        data = {key: np.asarray(archive[key]) for key in required}

    time = np.asarray(data["t"], dtype=np.float64)
    support = np.asarray(data["is_transformer_support"], dtype=bool)
    require(time.shape == (25601,), "DER figure requires the m32 time grid")
    require(np.all(np.diff(time) > 0.0), "time grid must strictly increase")
    require(
        np.isclose(time[0], 0.0) and np.isclose(time[-1], 10.0),
        "expected time horizon [0,10]",
    )
    require(support.shape == time.shape, "invalid support mask")
    require(int(support.sum()) == 801, "expected 801 base-grid display nodes")
    require(
        np.array_equal(np.flatnonzero(support), np.arange(0, time.size, 32)),
        "base-grid support must occur every 32 m32 coordinates",
    )
    for state in STATES:
        values = np.asarray(data[f"{state}__N"], dtype=np.float64)
        require(values.shape == (time.size, 21), f"invalid {state} state shape")
        require(np.all(np.isfinite(values)), f"{state} state is not finite")
        require(np.all(values > 0.0), f"{state} state must be positive")
        data[f"{state}__N"] = values
    data["t"] = time
    data["is_transformer_support"] = support
    control = np.asarray(data["nominal__u"], dtype=np.float64)
    require(control.shape == time.shape, "invalid nominal control shape")
    require(np.all(np.isfinite(control)), "nominal control is not finite")
    data["nominal__u"] = control
    return data


def clean_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#73808B")
    axis.spines["bottom"].set_color("#73808B")
    axis.tick_params(length=2.2, width=0.55, pad=1.5)


def switching_profiles(
    time: np.ndarray,
    control: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    change = np.abs(np.diff(control))
    records: list[dict[str, Any]] = []
    for name, low, high in (
        ("early", 0.0, 2.0),
        ("late", 7.0, 10.0),
    ):
        eligible = np.flatnonzero(
            (time[:-1] >= low) & (time[1:] <= high)
        )
        require(eligible.size > 0, f"empty {name} switching window")
        left_index = int(eligible[np.argmax(change[eligible])])
        right_index = left_index + 1
        records.append(
            {
                "name": name,
                "search_window": [low, high],
                "left_index": left_index,
                "right_index": right_index,
                "bracket": [
                    float(time[left_index]),
                    float(time[right_index]),
                ],
                "midpoint": float(
                    0.5 * (time[left_index] + time[right_index])
                ),
                "absolute_control_change": float(change[left_index]),
                "profile_index": right_index,
                "profile_time": float(time[right_index]),
                "profile_coordinate_rule": (
                    "right m32 coordinate of the maximum adjacent control "
                    "change in the search window"
                ),
            }
        )
    return records[0], records[1]


def profile_indices(
    time: np.ndarray,
    switches: tuple[dict[str, Any], dict[str, Any]],
) -> tuple[int, ...]:
    early, late = switches
    requested = (0.0, 2.0, 8.0, 10.0)
    fixed = {
        value: int(np.argmin(np.abs(time - value)))
        for value in requested
    }
    require(
        all(np.isclose(time[index], value) for value, index in fixed.items()),
        "fixed profile times are absent from the m32 grid",
    )
    return (
        fixed[0.0],
        int(early["profile_index"]),
        fixed[2.0],
        fixed[8.0],
        int(late["profile_index"]),
        fixed[10.0],
    )


def draw_heatmap(
    axis: plt.Axes,
    time: np.ndarray,
    state: np.ndarray,
    support: np.ndarray,
    norm: LogNorm,
    switch_midpoints: tuple[float, float],
) -> Any:
    phenotype = np.linspace(0.0, 1.0, state.shape[1], dtype=np.float64)
    image = axis.pcolormesh(
        time[support],
        phenotype,
        state[support].T,
        shading="nearest",
        cmap="viridis",
        norm=norm,
        rasterized=False,
    )
    axis.set_xlim(0.0, 10.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(np.arange(0.0, 10.1, 2.0))
    axis.set_yticks(np.linspace(0.0, 1.0, 6))
    axis.set_xlabel(r"time $t$")
    axis.set_ylabel(r"phenotype trait $x_i$")
    for value in switch_midpoints:
        axis.axvline(
            value,
            color="white",
            linestyle=(0, (2.0, 2.0)),
            linewidth=0.68,
            alpha=0.9,
        )
    axis.grid(False)
    clean_axis(axis)
    return image


def draw_profiles(
    axis: plt.Axes,
    time: np.ndarray,
    state: np.ndarray,
    indices: tuple[int, ...],
    times: tuple[float, ...],
    y_limits: tuple[float, float],
) -> None:
    phenotype = np.linspace(0.0, 1.0, state.shape[1], dtype=np.float64)
    for position, (index, target, color, marker) in enumerate(zip(
        indices,
        times,
        PROFILE_COLORS,
        PROFILE_MARKERS,
        strict=True,
    )):
        switch_profile = position in (1, 4)
        label = (
            rf"$t={target:.3f}$ (switch)"
            if switch_profile
            else rf"$t={target:g}$"
        )
        axis.plot(
            phenotype,
            state[index],
            color=color,
            linewidth=1.22 if switch_profile else 1.02,
            marker=marker,
            markersize=2.55,
            markeredgewidth=0.35,
            markevery=2,
            label=label,
        )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(*y_limits)
    axis.set_yscale("log")
    axis.set_xticks(np.linspace(0.0, 1.0, 6))
    axis.set_xlabel(r"phenotype trait $x_i$")
    axis.set_ylabel(r"population $N_i(t)$")
    axis.legend(
        frameon=False,
        ncol=2,
        loc="best",
        handlelength=1.55,
        columnspacing=0.85,
        handletextpad=0.45,
    )
    clean_axis(axis)


def save_figure(figure: plt.Figure, pdf: Path, png: Path) -> None:
    figure.savefig(pdf, bbox_inches="tight", pad_inches=0.025)
    figure.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)


def build_main(
    data: dict[str, np.ndarray],
    norm: LogNorm,
    y_limits: tuple[float, float],
    outputs: dict[str, Path],
) -> None:
    time = data["t"]
    support = data["is_transformer_support"]
    state = data["nominal__N"]
    switches = switching_profiles(time, data["nominal__u"])
    indices = profile_indices(time, switches)
    times = tuple(float(time[index]) for index in indices)
    switch_midpoints = tuple(float(item["midpoint"]) for item in switches)
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.05, 2.58),
        gridspec_kw={"width_ratios": (1.18, 1.0), "wspace": 0.31},
    )
    image = draw_heatmap(
        axes[0], time, state, support, norm, switch_midpoints
    )
    axes[0].set_title(
        r"(a) Phenotype evolution under PMP/KKT-DER",
        loc="left",
        pad=3.5,
        fontweight="semibold",
    )
    draw_profiles(axes[1], time, state, indices, times, y_limits)
    axes[1].set_title(
        r"(b) Nominal profiles at fixed and switching times",
        loc="left",
        pad=3.5,
        fontweight="semibold",
    )
    colorbar = figure.colorbar(
        image,
        ax=axes[0],
        orientation="vertical",
        fraction=0.052,
        pad=0.035,
    )
    colorbar.solids.set_rasterized(False)
    colorbar.set_label(r"$N_i(t)$ (log scale)", labelpad=2.8)
    colorbar.ax.tick_params(labelsize=6.0)
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.18, top=0.91)
    save_figure(figure, outputs["main_pdf"], outputs["main_png"])


def build_supplement(
    data: dict[str, np.ndarray],
    norm: LogNorm,
    y_limits: tuple[float, float],
    outputs: dict[str, Path],
) -> None:
    time = data["t"]
    support = data["is_transformer_support"]
    switches = switching_profiles(time, data["nominal__u"])
    indices = profile_indices(time, switches)
    times = tuple(float(time[index]) for index in indices)
    switch_midpoints = tuple(float(item["midpoint"]) for item in switches)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(7.05, 4.90),
        gridspec_kw={
            "width_ratios": (1.18, 1.0),
            "hspace": 0.42,
            "wspace": 0.31,
        },
    )
    panel_letters = (("a", "b"), ("c", "d"))
    image = None
    for row, state_id in enumerate(STATES):
        state = data[f"{state_id}__N"]
        image = draw_heatmap(
            axes[row, 0],
            time,
            state,
            support,
            norm,
            switch_midpoints,
        )
        axes[row, 0].set_title(
            rf"({panel_letters[row][0]}) {STATE_LABELS[state_id]} evolution",
            loc="left",
            pad=3.5,
            fontweight="semibold",
        )
        draw_profiles(
            axes[row, 1],
            time,
            state,
            indices,
            times,
            y_limits,
        )
        axes[row, 1].set_title(
            rf"({panel_letters[row][1]}) {STATE_LABELS[state_id]} profiles",
            loc="left",
            pad=3.5,
            fontweight="semibold",
        )
    assert image is not None
    figure.subplots_adjust(
        left=0.075,
        right=0.99,
        bottom=0.19,
        top=0.955,
    )
    lower_heatmap_box = axes[1, 0].get_position()
    colorbar_axis = figure.add_axes(
        [
            lower_heatmap_box.x0 + 0.12 * lower_heatmap_box.width,
            0.050,
            0.76 * lower_heatmap_box.width,
            0.018,
        ]
    )
    colorbar = figure.colorbar(
        image,
        cax=colorbar_axis,
        orientation="horizontal",
    )
    colorbar.solids.set_rasterized(False)
    colorbar.ax.set_title(r"$N_i(t)$ (log scale)", fontsize=6.3, pad=2.0)
    colorbar.ax.tick_params(labelsize=6.0)
    save_figure(
        figure,
        outputs["supplement_pdf"],
        outputs["supplement_png"],
    )


def main() -> None:
    args = parse_args()
    configure_style()
    source = args.der_npz.expanduser().resolve()
    prefix = args.out_prefix.expanduser().resolve()
    outputs = {
        "main_pdf": prefix.with_name(prefix.name + "_nominal_main.pdf"),
        "main_png": prefix.with_name(prefix.name + "_nominal_main.png"),
        "supplement_pdf": prefix.with_name(
            prefix.name + "_two_state_supplement.pdf"
        ),
        "supplement_png": prefix.with_name(
            prefix.name + "_two_state_supplement.png"
        ),
        "manifest": prefix.with_name(prefix.name + "_manifest.json"),
    }
    if not args.overwrite:
        existing = [path for path in outputs.values() if path.exists()]
        require(
            not existing,
            "refusing to overwrite: " + ", ".join(str(path) for path in existing),
        )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    data = load_data(source)
    all_values = np.concatenate(
        [data[f"{state}__N"].ravel() for state in STATES]
    )
    lower = float(np.min(all_values))
    upper = float(np.max(all_values))
    norm = LogNorm(vmin=lower, vmax=upper)
    y_limits = (lower / 1.08, upper * 1.08)

    build_main(data, norm, y_limits, outputs)
    build_supplement(data, norm, y_limits, outputs)

    time = data["t"]
    switches = switching_profiles(time, data["nominal__u"])
    indices = profile_indices(time, switches)
    times = tuple(float(time[index]) for index in indices)
    manifest = {
        "schema": "final-der-phenotype-figures-v1",
        "source": {
            "path": str(source),
            "sha256": sha256(source),
            "dense_time_points": int(time.size),
            "phenotype_count": 21,
        },
        "figure_contract": {
            "main_state": "nominal",
            "supplement_states": list(STATES),
            "phenotype_trait": "x_i=(i-1)/20 for i=1,...,21",
            "profile_times": list(times),
            "profile_indices": list(indices),
            "switching_neighborhood_profiles": list(switches),
            "switching_profile_legend_suffix": "(switch)",
            "shared_log_normalization": {
                "vmin": lower,
                "vmax": upper,
                "states": list(STATES),
            },
            "profile_y_limits_shared_across_all_panels": list(y_limits),
            "heatmap_display_coordinates": (
                "all 801 base-grid support times sampled exactly from the "
                "m32 DER state trajectory"
            ),
            "profile_coordinates": (
                "exact m32 coordinates at t=0,2,8,10 and at the right "
                "coordinates of the early/late maximum adjacent nominal "
                "control changes"
            ),
            "heatmap_pdf_rendering": "vector pcolormesh; not rasterized",
            "pdf_fonttype": 42,
        },
        "state_summaries": {
            state: {
                "initial_total_population": float(
                    data[f"{state}__N"][0].sum()
                ),
                "terminal_total_population": float(
                    data[f"{state}__N"][-1].sum()
                ),
                "minimum_component": float(
                    data[f"{state}__N"].min()
                ),
                "maximum_component": float(
                    data[f"{state}__N"].max()
                ),
            }
            for state in STATES
        },
        "outputs": {
            name: {
                "path": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in outputs.items()
            if name != "manifest"
        },
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"outputs": manifest["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
