#!/usr/bin/env python3
"""Build the final nominal PMP/KKT-DER 3D phenotype-evolution figure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROFILE_COLORS = (
    "#D07A32",
    "#2B7A78",
    "#5B6F9B",
    "#B64E3D",
)
PROFILE_MARKERS = ("s", "^", "D", "v")
GRID_COLOR = "#DDE3E8"
TEXT_COLOR = "#303A43"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--der-npz", type=Path, required=True)
    parser.add_argument(
        "--source-label",
        type=str,
        default=None,
        help="Canonical source path recorded in the manifest.",
    )
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "mathtext.fontset": "stixsans",
            "font.size": 6.8,
            "axes.titlesize": 7.8,
            "axes.labelsize": 6.9,
            "xtick.labelsize": 5.9,
            "ytick.labelsize": 5.9,
            "legend.fontsize": 5.9,
            "axes.linewidth": 0.62,
            "axes.grid": True,
            "grid.color": GRID_COLOR,
            "grid.alpha": 0.72,
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
        required = {
            "t",
            "is_transformer_support",
            "nominal__u",
            "nominal__N",
        }
        missing = sorted(required - set(archive.files))
        require(not missing, f"{path}: missing arrays {missing}")
        data = {key: np.asarray(archive[key]) for key in required}

    time = np.asarray(data["t"], dtype=np.float64)
    support = np.asarray(data["is_transformer_support"], dtype=bool)
    control = np.asarray(data["nominal__u"], dtype=np.float64)
    state = np.asarray(data["nominal__N"], dtype=np.float64)
    require(time.shape == (25601,), "expected the final m32 evaluation grid")
    require(np.all(np.diff(time) > 0.0), "time coordinates must increase")
    require(
        np.isclose(time[0], 0.0) and np.isclose(time[-1], 10.0),
        "expected time horizon [0,10]",
    )
    require(support.shape == time.shape, "invalid support mask")
    require(int(support.sum()) == 801, "expected 801 Transformer support nodes")
    require(control.shape == time.shape, "invalid control shape")
    require(state.shape == (time.size, 21), "invalid state shape")
    require(np.all(np.isfinite(control)), "control contains non-finite values")
    require(np.all(np.isfinite(state)), "state contains non-finite values")
    require(np.all(state > 0.0), "state values must be positive")
    return {
        "time": time,
        "support": support,
        "control": control,
        "state": state,
    }


def switching_records(
    time: np.ndarray,
    control: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    change = np.abs(np.diff(control))
    records: list[dict[str, Any]] = []
    for name, low, high in (("early", 0.0, 2.0), ("late", 7.0, 10.0)):
        eligible = np.flatnonzero(
            (time[:-1] >= low) & (time[1:] <= high)
        )
        require(eligible.size > 0, f"empty {name} switching window")
        left = int(eligible[np.argmax(change[eligible])])
        right = left + 1
        records.append(
            {
                "name": name,
                "left_index": left,
                "right_index": right,
                "bracket": [float(time[left]), float(time[right])],
                "midpoint": float(0.5 * (time[left] + time[right])),
                "control_change": float(control[right] - control[left]),
                "profile_index": right,
                "profile_time": float(time[right]),
            }
        )
    return records[0], records[1]


def profile_indices(
    time: np.ndarray,
    switches: tuple[dict[str, Any], dict[str, Any]],
) -> tuple[int, ...]:
    early, late = switches
    fixed = {
        target: int(np.argmin(np.abs(time - target)))
        for target in (2.0, 8.0)
    }
    require(
        all(np.isclose(time[index], target) for target, index in fixed.items()),
        "a fixed profile time is absent from the dense grid",
    )
    return (
        int(early["profile_index"]),
        fixed[2.0],
        fixed[8.0],
        int(late["profile_index"]),
    )


def clean_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#73808B")
    axis.spines["bottom"].set_color("#73808B")
    axis.tick_params(length=2.2, width=0.55, pad=1.4)


def build_figure(
    data: dict[str, np.ndarray],
    output_pdf: Path,
    output_png: Path,
) -> dict[str, Any]:
    time = data["time"]
    support = data["support"]
    control = data["control"]
    state = data["state"]
    trait = np.linspace(0.0, 1.0, state.shape[1], dtype=np.float64)
    switches = switching_records(time, control)
    indices = profile_indices(time, switches)
    profile_times = tuple(float(time[index]) for index in indices)

    figure = plt.figure(figsize=(7.15, 3.40))
    grid = figure.add_gridspec(
        2,
        3,
        width_ratios=(1.42, 1.0, 1.0),
        hspace=0.46,
        wspace=0.34,
    )

    transition_colors = ("#D07A32", "#B64E3D")
    surface_axis = figure.add_subplot(grid[:, 0], projection="3d")
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
        alpha=0.97,
        rstride=1,
        cstride=1,
    )
    for record, color in zip(switches, transition_colors):
        row = int(record["profile_index"])
        surface_axis.plot(
            trait,
            np.full_like(trait, time[row]),
            state[row],
            color=color,
            linewidth=1.55,
            zorder=8,
        )
    surface_axis.set_xlim(0.0, 1.0)
    surface_axis.set_ylim(0.0, 10.0)
    surface_axis.set_zlim(0.0, 42.0)
    surface_axis.set_xticks((0.0, 0.5, 1.0))
    surface_axis.set_yticks((0.0, 5.0, 10.0))
    surface_axis.set_zticks((0.0, 20.0, 40.0))
    surface_axis.set_xlabel(r"phenotype trait $x_i$", labelpad=1.5)
    surface_axis.set_ylabel(r"time $t$", labelpad=1.5)
    surface_axis.set_zlabel(r"population $N_i(t)$", labelpad=1.0)
    surface_axis.set_title(
        r"(a) Full phenotype evolution $N(t,x)$",
        loc="left",
        pad=0.0,
        fontweight="semibold",
    )
    surface_axis.view_init(elev=27.0, azim=-55.0)
    surface_axis.set_box_aspect((1.05, 1.25, 0.72))
    for pane in (
        surface_axis.xaxis.pane,
        surface_axis.yaxis.pane,
        surface_axis.zaxis.pane,
    ):
        pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        pane.set_edgecolor("#CBD3D9")

    profile_axes = [
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[0, 2]),
        figure.add_subplot(grid[1, 1]),
        figure.add_subplot(grid[1, 2]),
    ]
    for position, (axis, index, target, color, marker) in enumerate(
        zip(
            profile_axes,
            indices,
            profile_times,
            PROFILE_COLORS,
            PROFILE_MARKERS,
        )
    ):
        values = state[index]
        total = float(values.sum())
        peak_trait = float(trait[int(np.argmax(values))])
        is_switch = position in (0, 3)
        panel_letter = chr(ord("b") + position)
        title = (
            rf"({panel_letter}) $t={target:.3f}$ (transition)"
            if is_switch
            else rf"({panel_letter}) $t={target:g}$ (interior)"
        )
        axis.plot(
            trait,
            values,
            color=color,
            linewidth=1.20 if is_switch else 1.05,
            marker=marker,
            markersize=2.15,
            markeredgewidth=0.32,
            markevery=2,
        )
        axis.set_title(
            title,
            color=color if is_switch else TEXT_COLOR,
            pad=2.0,
            fontsize=6.1,
            fontweight="semibold",
        )
        axis.text(
            0.035,
            0.94,
            rf"$\sum_iN_i={total:.1f}$" + "\n" + rf"peak $x={peak_trait:.2f}$",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=5.0,
            color="#52606B",
        )
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 42.0)
        axis.set_xticks((0.0, 0.5, 1.0))
        axis.set_yticks((0.0, 20.0, 40.0))
        if position < 2:
            axis.tick_params(labelbottom=False)
        else:
            axis.set_xlabel(r"trait $x_i$", labelpad=1.0)
        if position % 2 == 0:
            axis.set_ylabel(r"$N_i(t)$", labelpad=1.0)
        else:
            axis.tick_params(labelleft=False)
        clean_axis(axis)

    figure.subplots_adjust(
        left=0.055,
        right=0.992,
        bottom=0.115,
        top=0.925,
    )
    figure.savefig(output_pdf, bbox_inches="tight", pad_inches=0.025)
    figure.savefig(
        output_png,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.025,
    )
    plt.close(figure)

    return {
        "profile_indices": list(indices),
        "profile_times": list(profile_times),
        "switches": list(switches),
        "surface_source_support_nodes": int(support.sum()),
        "surface_rendered_time_coordinates": int(rendered_indices.size),
        "surface_rendering_rule": (
            "every fourth one of the 801 exact Transformer support nodes, "
            "including the terminal node"
        ),
        "phenotype_coordinates": trait.tolist(),
        "initial_total_population": float(state[0].sum()),
        "terminal_total_population": float(state[-1].sum()),
        "minimum_component": float(state.min()),
        "maximum_component": float(state.max()),
    }


def main() -> None:
    args = parse_args()
    configure_style()
    source = args.der_npz.expanduser().resolve()
    prefix = args.out_prefix.expanduser().resolve()
    output_pdf = prefix.with_suffix(".pdf")
    output_png = prefix.with_suffix(".png")
    manifest_path = prefix.with_name(prefix.name + "_manifest.json")
    outputs = (output_pdf, output_png, manifest_path)
    if not args.overwrite:
        existing = [path for path in outputs if path.exists()]
        require(
            not existing,
            "refusing to overwrite: " + ", ".join(str(path) for path in existing),
        )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    data = load_data(source)
    figure_contract = build_figure(data, output_pdf, output_png)
    manifest = {
        "schema": "final-der-phenotype-3d-figure-v1",
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
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output_pdf)
    print(output_png)
    print(manifest_path)


if __name__ == "__main__":
    main()
