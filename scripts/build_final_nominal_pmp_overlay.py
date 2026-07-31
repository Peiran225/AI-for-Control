#!/usr/bin/env python3
"""Build the compact one-row nominal PMP comparison figure.

Each of the five panels overlays PMP-Time, PMP-CF, and PMP-DER.  The plotted
curves contain the 801 n=800 grid coordinates and all 800 interval midpoints;
open circles mark only a sparse representative subset of grid nodes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from build_final_nominal_pmp_figure import (
    CaseData,
    clean_axis,
    configure_style,
    load_case,
    regime_transitions,
    require,
    selected_marker_indices,
    validate_cases,
)


CASE_SPECS = (
    ("time_only", "PMP-Time", "#0072B2", "-"),
    ("cf", "PMP-CF", "#D55E00", (0, (4.0, 1.6))),
    ("der", "PMP-DER", "#009E73", (0, (1.2, 1.25))),
)

PANEL_SPECS = (
    ("u", r"(a) Control $u(t)$"),
    ("population", r"(b) Total population $\sum_i N_i(t)$"),
    ("H_u", r"(c) $\psi(t)$"),
    ("dH_u_dt", r"(d) $\dot{\psi}(t)$"),
    ("d2H_u_dt2", r"(e) $\ddot{\psi}(t)$"),
)

INTERIOR_FIELDS = {"H_u", "dH_u_dt", "d2H_u_dt2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-npz", type=Path, required=True)
    parser.add_argument("--cf-npz", type=Path, required=True)
    parser.add_argument("--der-npz", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--interior-start", type=float, default=1.5)
    parser.add_argument("--interior-end", type=float, default=8.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def method_marker_indices(
    base_grid: np.ndarray,
    domain: np.ndarray,
    method_index: int,
    count: int,
) -> np.ndarray:
    """Choose readable, method-specific markers that remain true grid nodes."""
    eligible = np.flatnonzero(base_grid & domain)
    require(eligible.size >= count + 2, "not enough grid nodes for markers")
    phase = (method_index - 1) * 0.18
    positions = np.linspace(0.08 + phase / count, 0.92 + phase / count, count)
    positions = np.clip(positions, 0.02, 0.98)
    indices = np.rint(positions * (eligible.size - 1)).astype(int)
    return eligible[np.unique(indices)]


def common_transitions(cases: Sequence[CaseData]) -> tuple[float, float]:
    transitions = np.asarray(
        [regime_transitions(case) for case in cases],
        dtype=np.float64,
    )
    require(
        np.max(np.ptp(transitions, axis=0)) <= 2.0e-3,
        "the three cases do not share common transition markers",
    )
    return tuple(np.mean(transitions, axis=0).tolist())


def values_for(case: CaseData, field: str) -> np.ndarray:
    series = case.states["nominal"]
    return np.asarray(getattr(series, field), dtype=np.float64)


def plot_figure(
    cases: Sequence[CaseData],
    interior_start: float,
    interior_end: float,
) -> tuple[plt.Figure, dict[str, Any]]:
    t = cases[0].t
    midpoint = np.zeros(t.shape, dtype=bool)
    midpoint[16:-1:32] = True
    require(int(midpoint.sum()) == 800, "expected 800 interval midpoints")
    curve_support = cases[0].masks.base_grid | midpoint
    full_domain = np.ones(t.shape, dtype=bool)
    singular_domain = (t >= interior_start) & (t < interior_end)
    entry, exit_ = common_transitions(cases)

    configure_style()
    plt.rcParams.update(
        {
            "font.size": 6.5,
            "axes.titlesize": 7.0,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 5.6,
            "ytick.labelsize": 5.6,
            "legend.fontsize": 6.4,
            "axes.linewidth": 0.62,
            "grid.alpha": 0.58,
            "grid.linewidth": 0.38,
        }
    )

    figure, axes = plt.subplots(
        1,
        5,
        figsize=(7.15, 1.80),
        sharex=True,
        squeeze=False,
        gridspec_kw={
            "width_ratios": (1.02, 1.08, 1.0, 1.0, 1.0),
        },
    )
    axes = axes[0]

    for axis, (field, title) in zip(axes, PANEL_SPECS, strict=True):
        domain = singular_domain if field in INTERIOR_FIELDS else full_domain
        plotted_domain = domain & curve_support

        axis.axvspan(
            entry,
            exit_,
            color="#EFF3F5",
            alpha=0.62,
            linewidth=0.0,
            zorder=0,
        )
        for transition in (entry, exit_):
            axis.axvline(
                transition,
                color="#707A83",
                linestyle=(0, (1.2, 1.4)),
                linewidth=0.62,
                zorder=2,
            )

        for method_index, (case, spec) in enumerate(
            zip(cases, CASE_SPECS, strict=True)
        ):
            _, label, color, linestyle = spec
            values = values_for(case, field)
            axis.plot(
                t[plotted_domain],
                values[plotted_domain],
                color=color,
                linestyle=linestyle,
                linewidth=1.02,
                solid_capstyle="round",
                zorder=3 + method_index,
                label=label,
            )
            markers = method_marker_indices(
                case.masks.base_grid,
                domain,
                method_index,
                7 if field in INTERIOR_FIELDS else 8,
            )
            axis.plot(
                t[markers],
                values[markers],
                linestyle="none",
                marker="o",
                markersize=2.25,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=0.62,
                zorder=7 + method_index,
            )

        if field in INTERIOR_FIELDS:
            axis.axhline(
                0.0,
                color="#505B65",
                linewidth=0.58,
                zorder=1,
            )
            axis.set_ylim(-0.055, 0.055)
            axis.set_yticks((-0.05, 0.0, 0.05))
        elif field == "u":
            axis.set_ylim(-0.12, 3.18)
            axis.set_yticks((0.0, 1.5, 3.0))
        else:
            axis.set_ylim(75.0, 335.0)
            axis.set_yticks((100.0, 200.0, 300.0))

        axis.set_xlim(0.0, 10.0)
        axis.set_xticks((0.0, 5.0, 10.0))
        axis.set_title(title, fontweight="semibold", pad=2.8)
        clean_axis(axis)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linestyle=linestyle,
            linewidth=1.2,
            marker="o",
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.65,
            markersize=3.0,
            label=label,
        )
        for _, label, color, linestyle in CASE_SPECS
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        handlelength=2.35,
        columnspacing=1.7,
        handletextpad=0.55,
    )
    figure.supxlabel(r"time $t$", x=0.515, y=0.035, fontsize=6.5)
    figure.subplots_adjust(
        left=0.045,
        right=0.995,
        bottom=0.245,
        top=0.735,
        wspace=0.38,
    )

    metadata = {
        "layout": "one row with five panels",
        "state": "nominal",
        "methods": [spec[1] for spec in CASE_SPECS],
        "panels": [field for field, _ in PANEL_SPECS],
        "notation": {
            "H_u": "psi",
            "dH_u_dt": "dot_psi",
            "d2H_u_dt2": "ddot_psi",
        },
        "curve_coordinates": (
            "801 n=800 grid coordinates plus all 800 interval midpoints"
        ),
        "open_circles": "sparse representative n=800 grid nodes",
        "query_markers": "not displayed",
        "transition_markers": {
            "upper_to_interior": entry,
            "interior_to_lower": exit_,
        },
        "derivative_domain": [interior_start, interior_end],
        "derivative_ylim": [-0.055, 0.055],
    }
    return figure, metadata


def main() -> None:
    args = parse_args()
    paths = [args.time_npz, args.cf_npz, args.der_npz]
    cases = [
        load_case(path, case_id, label)
        for path, (case_id, label, _, _) in zip(
            paths,
            CASE_SPECS,
            strict=True,
        )
    ]
    validate_cases(cases, production=True)

    prefix = args.out_prefix.expanduser().resolve()
    pdf = prefix.with_suffix(".pdf")
    png = prefix.with_suffix(".png")
    manifest = prefix.with_name(prefix.name + "_manifest.json")
    outputs = (pdf, png, manifest)
    if not args.overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        require(not existing, "refusing to overwrite: " + ", ".join(existing))
    prefix.parent.mkdir(parents=True, exist_ok=True)

    figure, metadata = plot_figure(
        cases,
        args.interior_start,
        args.interior_end,
    )
    figure.savefig(pdf, bbox_inches="tight", pad_inches=0.025)
    figure.savefig(png, dpi=360, bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)

    payload = {
        "schema": "nominal-pmp-overlay-v1",
        "inputs": {
            case.case_id: {
                "path": str(case.path),
                "sha256": case.input_sha256,
            }
            for case in cases
        },
        "figure_contract": metadata,
        "outputs": {
            path.suffix.lstrip("."): {
                "path": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in (pdf, png)
        },
    }
    manifest.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["outputs"], indent=2))


if __name__ == "__main__":
    main()
