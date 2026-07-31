#!/usr/bin/env python3
"""Build the one-column nominal PMP-DER phenotype figure with a 3D surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
from matplotlib.transforms import Bbox

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from build_final_der_phenotype_3d_figure import (
    load_data,
    require,
    sha256,
    switching_records,
)


PROFILE_COLORS = ("#4C1D95", "#2563A6", "#0F766E", "#D97706", "#B91C1C")
PROFILE_MARKERS = ("o", "s", "^", "D", "v")
GRID_COLOR = "#DDE3E8"
TEXT_COLOR = "#303A43"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--der-npz", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--source-label", type=str, default=None)
    parser.add_argument(
        "--split-panels",
        action="store_true",
        help="also save panels (a), (b), and (c) as separate vector PDFs",
    )
    parser.add_argument(
        "--tex-panel-titles",
        action="store_true",
        help=(
            "omit embedded titles from split-panel PDFs so panel labels and "
            "titles can be supplied by LaTeX"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "mathtext.fontset": "stixsans",
            "font.size": 7.6,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.6,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.3,
            "axes.linewidth": 0.65,
            "axes.grid": True,
            "grid.color": GRID_COLOR,
            "grid.alpha": 0.72,
            "grid.linewidth": 0.42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 320,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "path.simplify": False,
        }
    )


def clean_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#73808B")
    axis.spines["bottom"].set_color("#73808B")
    axis.tick_params(length=2.3, width=0.55, pad=1.5)


def selected_profiles(
    time: np.ndarray,
    switches: tuple[dict[str, Any], dict[str, Any]],
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    entry, exit_ = switches
    midpoint = 0.5 * (
        float(entry["profile_time"]) + float(exit_["profile_time"])
    )
    targets = (
        0.0,
        float(entry["profile_time"]),
        midpoint,
        float(exit_["profile_time"]),
        10.0,
    )
    indices = tuple(int(np.argmin(np.abs(time - target))) for target in targets)
    return indices, tuple(float(time[index]) for index in indices)


def build_figure(
    data: dict[str, np.ndarray],
    output_pdf: Path,
    output_png: Path,
    panel_pdfs: dict[str, Path] | None = None,
    tex_panel_titles: bool = False,
) -> dict[str, Any]:
    time = data["time"]
    support = data["support"]
    control = data["control"]
    state = data["state"]
    trait = np.linspace(0.0, 1.0, state.shape[1], dtype=np.float64)
    switches = switching_records(time, control)
    indices, profile_times = selected_profiles(time, switches)
    transition_times = (profile_times[1], profile_times[3])

    figure = plt.figure(figsize=(3.45, 5.45))
    grid = figure.add_gridspec(
        3,
        1,
        height_ratios=(0.62, 1.40, 1.22),
        hspace=0.52,
        left=0.16,
        right=0.95,
        bottom=0.065,
        top=0.98,
    )

    control_axis = figure.add_subplot(grid[0, 0])
    control_axis.axvspan(
        transition_times[0],
        transition_times[1],
        color="#EAF2F8",
        alpha=0.78,
        linewidth=0.0,
        zorder=0,
    )
    control_axis.plot(time, control, color="#1F5F8B", linewidth=1.25, zorder=2)
    for transition in transition_times:
        control_axis.axvline(
            transition,
            color="#6B7280",
            linestyle="--",
            linewidth=0.72,
            zorder=1,
        )
    control_axis.axvline(
        profile_times[2],
        color="#6B7280",
        linestyle=":",
        linewidth=0.72,
        zorder=1,
    )
    for profile_time, index, color, marker in zip(
        profile_times, indices, PROFILE_COLORS, PROFILE_MARKERS
    ):
        control_axis.plot(
            profile_time,
            control[index],
            linestyle="none",
            marker=marker,
            markersize=3.9,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.45,
            clip_on=False,
            zorder=4,
        )
    control_axis.set_xlim(0.0, 10.0)
    control_axis.set_ylim(-0.12, 3.18)
    control_axis.set_yticks((0, 1, 2, 3))
    control_axis.set_xlabel(r"time $t$")
    control_axis.set_ylabel(r"control $u(t)$")
    control_axis.set_title(
        "(a) Learned control and selected profile times",
        loc="left",
        fontweight="semibold",
        color=TEXT_COLOR,
        pad=3.0,
    )
    clean_axis(control_axis)

    surface_axis = figure.add_subplot(grid[1, 0], projection="3d")
    support_indices = np.flatnonzero(support)
    rendered_indices = support_indices[::4]
    if rendered_indices[-1] != support_indices[-1]:
        rendered_indices = np.append(rendered_indices, support_indices[-1])
    trait_mesh, time_mesh = np.meshgrid(trait, time[rendered_indices])
    surface_axis.plot_surface(
        trait_mesh,
        time_mesh,
        state[rendered_indices],
        cmap="viridis",
        linewidth=0.0,
        antialiased=True,
        alpha=0.98,
        rstride=1,
        cstride=1,
    )
    for profile_time, index, color in zip(
        profile_times, indices, PROFILE_COLORS
    ):
        surface_axis.plot(
            trait,
            np.full_like(trait, profile_time),
            state[index],
            color=color,
            linewidth=1.05,
            zorder=8,
        )
    surface_axis.set_xlim(0.0, 1.0)
    surface_axis.set_ylim(0.0, 10.0)
    surface_axis.set_zlim(0.0, 42.0)
    surface_axis.set_xticks((0.0, 0.5, 1.0))
    surface_axis.set_yticks((0.0, 5.0, 10.0))
    surface_axis.set_zticks((0.0, 20.0, 40.0))
    surface_axis.set_xlabel(r"phenotype trait $x_i$", labelpad=2.0)
    surface_axis.set_ylabel(r"time $t$", labelpad=2.0)
    surface_axis.set_zlabel(r"population $N_i(t)$", labelpad=1.0)
    surface_axis.set_title(
        r"(b) Full phenotype evolution $N(t,x)$",
        loc="left",
        pad=1.0,
        fontweight="semibold",
        color=TEXT_COLOR,
    )
    surface_axis.view_init(elev=27.0, azim=-55.0)
    surface_axis.set_box_aspect((1.05, 1.30, 0.74))
    for pane in (
        surface_axis.xaxis.pane,
        surface_axis.yaxis.pane,
        surface_axis.zaxis.pane,
    ):
        pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        pane.set_edgecolor("#CBD3D9")

    profile_axis = figure.add_subplot(grid[2, 0])
    profile_labels = (
        rf"initial: $t={profile_times[0]:.0f}$",
        rf"entry: $t={profile_times[1]:.3f}$",
        rf"midpoint: $t={profile_times[2]:.3f}$",
        rf"exit: $t={profile_times[3]:.3f}$",
        rf"terminal: $t={profile_times[4]:.0f}$",
    )
    profile_records: list[dict[str, Any]] = []
    profile_handles: list[Any] = []
    for index, label, color, marker in zip(
        indices, profile_labels, PROFILE_COLORS, PROFILE_MARKERS
    ):
        values = state[index]
        (handle,) = profile_axis.plot(
            trait,
            values,
            color=color,
            linewidth=1.18,
            marker=marker,
            markersize=2.6,
            markeredgewidth=0.0,
            label=label,
        )
        profile_handles.append(handle)
        profile_records.append(
            {
                "index": int(index),
                "time": float(time[index]),
                "total_population": float(values.sum()),
                "peak_trait": float(trait[int(np.argmax(values))]),
                "peak_population": float(values.max()),
            }
        )
    profile_axis.set_xlim(0.0, 1.0)
    profile_axis.set_ylim(0.0, 41.0)
    profile_axis.set_xticks((0.0, 0.25, 0.5, 0.75, 1.0))
    profile_axis.set_yticks((0, 10, 20, 30, 40))
    profile_axis.set_xlabel(r"phenotype trait $x_i$")
    profile_axis.set_ylabel(r"population $N_i(t)$")
    profile_axis.set_title(
        "(c) Selected phenotype profiles",
        loc="left",
        fontweight="semibold",
        color=TEXT_COLOR,
        pad=3.0,
    )
    profile_axis.legend(
        profile_handles,
        profile_labels,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="none",
        ncol=1,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.985),
        fontsize=7.0,
        handlelength=1.35,
        handletextpad=0.36,
        borderaxespad=0.15,
    )
    clean_axis(profile_axis)

    panel_records: dict[str, Any] = {}
    if panel_pdfs:
        panel_axes = {
            "a_control": control_axis,
            "b_surface": surface_axis,
            "c_profiles": profile_axis,
        }
        panel_padding = {
            "a_control": (0.035, 0.035, 0.035, 0.035),
            # Matplotlib's 3D tight box underestimates the oblique x label.
            "b_surface": (0.18, 0.22, 0.06, 0.035),
            "c_profiles": (0.035, 0.035, 0.035, 0.035),
        }
        for key, axis in panel_axes.items():
            panel_path = panel_pdfs[key]
            panel_path.parent.mkdir(parents=True, exist_ok=True)
            visibility = {
                other_axis: other_axis.get_visible()
                for other_axis in panel_axes.values()
            }
            for other_axis in panel_axes.values():
                other_axis.set_visible(other_axis is axis)
            original_title = axis.get_title(loc="left")
            if tex_panel_titles:
                axis.set_title("", loc="left")
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            tight = (
                axis.get_tightbbox(renderer)
                .transformed(figure.dpi_scale_trans.inverted())
            )
            left, bottom, right, top = panel_padding[key]
            crop = Bbox.from_extents(
                tight.x0 - left,
                tight.y0 - bottom,
                tight.x1 + right,
                tight.y1 + top,
            )
            figure.savefig(
                panel_path,
                bbox_inches=crop,
                pad_inches=0.0,
            )
            if tex_panel_titles:
                axis.set_title(original_title, loc="left")
            for other_axis, was_visible in visibility.items():
                other_axis.set_visible(was_visible)
            panel_records[key] = {
                "path": str(panel_path),
                "crop_bounds_inches": [float(value) for value in crop.extents],
                "embedded_title": not tex_panel_titles,
            }

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf, bbox_inches="tight", pad_inches=0.035)
    figure.savefig(output_png, dpi=320, bbox_inches="tight", pad_inches=0.035)
    plt.close(figure)

    return {
        "profile_indices": list(indices),
        "profile_times": list(profile_times),
        "profile_records": profile_records,
        "switches": list(switches),
        "surface_source_support_nodes": int(support.sum()),
        "surface_rendered_time_coordinates": int(rendered_indices.size),
        "initial_total_population": float(state[0].sum()),
        "terminal_total_population": float(state[-1].sum()),
        "minimum_component": float(state.min()),
        "maximum_component": float(state.max()),
        "split_panels": panel_records,
    }


def main() -> None:
    args = parse_args()
    configure_style()
    source = args.der_npz.expanduser().resolve()
    prefix = args.out_prefix.expanduser().resolve()
    output_pdf = prefix.with_suffix(".pdf")
    output_png = prefix.with_suffix(".png")
    manifest_path = prefix.with_name(prefix.name + "_manifest.json")
    panel_pdfs = (
        {
            "a_control": prefix.with_name(prefix.name + "_a_control").with_suffix(
                ".pdf"
            ),
            "b_surface": prefix.with_name(prefix.name + "_b_surface").with_suffix(
                ".pdf"
            ),
            "c_profiles": prefix.with_name(
                prefix.name + "_c_profiles"
            ).with_suffix(".pdf"),
        }
        if args.split_panels
        else None
    )
    outputs = (
        output_pdf,
        output_png,
        manifest_path,
        *(panel_pdfs.values() if panel_pdfs else ()),
    )
    if not args.overwrite:
        existing = [path for path in outputs if path.exists()]
        require(
            not existing,
            "refusing to overwrite: " + ", ".join(str(path) for path in existing),
        )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    data = load_data(source)
    figure_contract = build_figure(
        data,
        output_pdf,
        output_png,
        panel_pdfs=panel_pdfs,
        tex_panel_titles=args.tex_panel_titles,
    )
    manifest = {
        "schema": "final-der-phenotype-main-3d-v1",
        "source": {
            "path": args.source_label or str(source),
            "sha256": sha256(source),
            "dense_time_points": int(data["time"].size),
            "transformer_support_nodes": int(data["support"].sum()),
            "phenotype_count": int(data["state"].shape[1]),
        },
        "figure_contract": figure_contract,
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
    if panel_pdfs:
        manifest["outputs"]["split_panels"] = {
            key: {
                "path": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for key, path in panel_pdfs.items()
        }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(output_pdf)
    print(output_png)
    print(manifest_path)


if __name__ == "__main__":
    main()
