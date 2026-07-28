#!/usr/bin/env python3
r"""Build the final nominal PMP figure and its two-state supplement.

The production inputs are the dense diagnostic NPZ files for PMP/KKT-Time,
PMP/KKT-CF, and PMP/KKT-DER.  The main-paper figure follows the reporting
contract in the manuscript: its curves use the 801 base-grid coordinates
together with all 800 interval midpoints.  For readability after the figure
is reduced to manuscript size, it displays a representative subset of:

* open circles: the 801 base-grid Transformer support nodes;
* crosses: the 800 direct midpoint queries between adjacent support nodes.

The supplementary figure retains the complete multiplier-32 curves and its
readability markers distinguish base-grid nodes from queries strictly outside
the multiplier-8 scalar-refinement grid.  The latter mask is loaded from
``is_held_out_from_refinement`` when present and verified against the
complement of ``is_refinement_grid``; otherwise it is derived from that
complement.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


CASE_SPECS = (
    ("time_only", "PMP/KKT-Time"),
    ("cf", "PMP/KKT-CF"),
    ("der", "PMP/KKT-DER"),
)
STATES = ("nominal", "resistant_heavy")
STATE_LABELS = {
    "nominal": "Nominal",
    "resistant_heavy": "Resistant-heavy",
}
STATE_COLORS = {
    "nominal": "#1F5A85",
    "resistant_heavy": "#C85A32",
}
STATE_LINESTYLES = {
    "nominal": "-",
    "resistant_heavy": "--",
}
FIELDS = ("u", "N", "H_u", "dH_u_dt", "d2H_u_dt2")
DERIVATIVES = ("H_u", "dH_u_dt", "d2H_u_dt2")
ROW_LABELS = {
    "u": r"control $u(t)$",
    "population": r"total population $\sum_i N_i(t)$",
    "H_u": r"$H_u(t)$",
    "dH_u_dt": r"$\mathrm{d}H_u(t)/\mathrm{d}t$",
    "d2H_u_dt2": r"$\mathrm{d}^2H_u(t)/\mathrm{d}t^2$",
}
MASK_ALIASES = {
    "base_grid": (
        "is_transformer_support",
        "is_base_grid_node",
        "is_training_grid",
    ),
    "refinement_grid": ("is_refinement_grid",),
    "held_out_from_refinement": ("is_held_out_from_refinement",),
    "singular_interior": ("is_singular_interior",),
}
COLORS = {
    "grid": "#DDE3E8",
    "zero": "#56616B",
    "spine": "#73808B",
    "marker_legend": "#30363B",
    "transition": "#6E7781",
    "upper_band": "#EAF2F8",
    "interior_band": "#F5F7F8",
    "lower_band": "#F3F0EA",
}
TARGET_PHYSICAL_WEIGHTS = {
    "alpha": 1.0,
    "beta": 40.0,
    "gamma": 8000.0,
}


@dataclass(frozen=True)
class StateSeries:
    u: np.ndarray
    population: np.ndarray
    H_u: np.ndarray
    dH_u_dt: np.ndarray
    d2H_u_dt2: np.ndarray


@dataclass(frozen=True)
class MaskSet:
    base_grid: np.ndarray
    refinement_grid: np.ndarray
    strict_off_refinement: np.ndarray
    singular_interior: np.ndarray | None
    off_refinement_source: str


@dataclass(frozen=True)
class CaseData:
    case_id: str
    label: str
    path: Path | None
    input_sha256: str | None
    t: np.ndarray
    masks: MaskSet
    states: Mapping[str, StateSeries]
    sibling_summary: Mapping[str, Any] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-npz", type=Path)
    parser.add_argument("--cf-npz", type=Path)
    parser.add_argument("--der-npz", type=Path)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--interior-start", type=float, default=1.5)
    parser.add_argument("--interior-end", type=float, default=8.0)
    parser.add_argument(
        "--full-marker-count",
        type=int,
        default=11,
        help="Maximum displayed markers of each mask on a full-horizon curve.",
    )
    parser.add_argument(
        "--interior-marker-count",
        type=int,
        default=13,
        help="Maximum displayed markers of each mask on an interior curve.",
    )
    parser.add_argument(
        "--layout-smoke",
        action="store_true",
        help="Use deterministic synthetic data to test the complete layout.",
    )
    parser.add_argument(
        "--hide-main-query-markers",
        action="store_true",
        help=(
            "Do not draw interval-midpoint x markers in the main figure. "
            "The midpoint values remain part of every plotted curve and the "
            "supplementary query markers are unchanged."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing outputs.",
    )
    args = parser.parse_args()
    if not args.layout_smoke:
        missing = [
            flag
            for flag, value in (
                ("--time-npz", args.time_npz),
                ("--cf-npz", args.cf_npz),
                ("--der-npz", args.der_npz),
            )
            if value is None
        ]
        if missing:
            parser.error("production mode requires " + ", ".join(missing))
    if not 0.0 <= args.interior_start < args.interior_end <= 10.0:
        parser.error("require 0 <= interior-start < interior-end <= 10")
    if args.full_marker_count < 1 or args.interior_marker_count < 1:
        parser.error("marker counts must be positive")
    return args


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.sans-serif": [
                "DejaVu Sans",
                "Arial",
                "Liberation Sans",
            ],
            "mathtext.fontset": "stixsans",
            "font.size": 6.6,
            "axes.titlesize": 8.1,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 5.9,
            "ytick.labelsize": 5.9,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.62,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.7,
            "grid.linewidth": 0.42,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
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


def mask_sha256(mask: np.ndarray) -> str:
    packed = np.packbits(np.asarray(mask, dtype=np.uint8))
    return hashlib.sha256(packed.tobytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    require(isinstance(payload, dict), f"{path}: expected a JSON object")
    return payload


def first_present(
    archive: Mapping[str, np.ndarray],
    aliases: Iterable[str],
) -> str | None:
    for key in aliases:
        if key in archive:
            return key
    return None


def load_optional_mask(
    archive: Mapping[str, np.ndarray],
    kind: str,
    length: int,
) -> np.ndarray | None:
    key = first_present(archive, MASK_ALIASES[kind])
    if key is None:
        return None
    mask = np.asarray(archive[key], dtype=bool)
    require(mask.shape == (length,), f"{key}: expected shape {(length,)}")
    return mask


def sibling_summary(path: Path) -> dict[str, Any] | None:
    summary_path = path.with_name("summary.json")
    if not summary_path.is_file():
        return None
    payload = load_json(summary_path)
    problem = payload.get("problem")
    factor = payload.get("report_scale_factor")
    if isinstance(problem, dict) and factor is not None:
        for name, target in TARGET_PHYSICAL_WEIGHTS.items():
            source = float(problem[name])
            require(
                np.isclose(
                    source * float(factor),
                    target,
                    rtol=0.0,
                    atol=1.0e-10,
                ),
                f"{summary_path}: {name} does not restore to {target}",
            )
    return payload


def load_case(path: Path, case_id: str, label: str) -> CaseData:
    path = path.expanduser().resolve()
    require(path.is_file(), f"missing input: {path}")
    with np.load(path, allow_pickle=False) as archive:
        require("t" in archive, f"{path}: missing t")
        t = np.asarray(archive["t"], dtype=np.float64)
        require(t.ndim == 1 and t.size > 1, f"{path}: invalid t")
        require(np.all(np.isfinite(t)), f"{path}: t is not finite")
        require(np.all(np.diff(t) > 0.0), f"{path}: t must increase")
        require(
            np.isclose(t[0], 0.0) and np.isclose(t[-1], 10.0),
            f"{path}: expected horizon [0,10]",
        )
        base_grid = load_optional_mask(archive, "base_grid", t.size)
        require(base_grid is not None, f"{path}: missing base-grid mask")
        refinement_grid = load_optional_mask(
            archive,
            "refinement_grid",
            t.size,
        )
        require(
            refinement_grid is not None,
            f"{path}: missing scalar-refinement-grid mask",
        )
        require(
            np.all(~base_grid | refinement_grid),
            f"{path}: every base-grid node must belong to refinement grid",
        )
        derived_off_refinement = ~refinement_grid
        explicit_held_out = load_optional_mask(
            archive,
            "held_out_from_refinement",
            t.size,
        )
        if explicit_held_out is None:
            strict_off_refinement = derived_off_refinement
            off_refinement_source = (
                "derived as the exact complement of is_refinement_grid"
            )
        else:
            require(
                np.array_equal(
                    explicit_held_out,
                    derived_off_refinement,
                ),
                f"{path}: is_held_out_from_refinement must equal the "
                "complement of is_refinement_grid",
            )
            strict_off_refinement = explicit_held_out
            off_refinement_source = (
                "explicit is_held_out_from_refinement mask, verified as "
                "the exact complement of is_refinement_grid"
            )

        masks = MaskSet(
            base_grid=base_grid,
            refinement_grid=refinement_grid,
            strict_off_refinement=strict_off_refinement,
            singular_interior=load_optional_mask(
                archive,
                "singular_interior",
                t.size,
            ),
            off_refinement_source=off_refinement_source,
        )

        states: dict[str, StateSeries] = {}
        for state in STATES:
            arrays: dict[str, np.ndarray] = {}
            for field in FIELDS:
                key = f"{state}__{field}"
                require(key in archive, f"{path}: missing {key}")
                value = np.asarray(archive[key], dtype=np.float64)
                if field == "N":
                    require(
                        value.ndim == 2 and value.shape[0] == t.size,
                        f"{path}: {key} must have shape (time, phenotype)",
                    )
                else:
                    require(
                        value.shape == t.shape,
                        f"{path}: {key} must have shape {t.shape}",
                    )
                require(
                    np.all(np.isfinite(value)),
                    f"{path}: {key} contains non-finite values",
                )
                arrays[field] = value
            require(
                np.min(arrays["u"]) >= -1.0e-8
                and np.max(arrays["u"]) <= 3.0 + 1.0e-8,
                f"{path}: {state} control leaves [0,3]",
            )
            states[state] = StateSeries(
                u=arrays["u"],
                population=np.sum(arrays["N"], axis=1),
                H_u=arrays["H_u"],
                dH_u_dt=arrays["dH_u_dt"],
                d2H_u_dt2=arrays["d2H_u_dt2"],
            )

    return CaseData(
        case_id=case_id,
        label=label,
        path=path,
        input_sha256=sha256(path),
        t=t,
        masks=masks,
        states=states,
        sibling_summary=sibling_summary(path),
    )


def smoke_cases() -> list[CaseData]:
    t = np.linspace(0.0, 10.0, 3201, dtype=np.float64)
    base = np.zeros(t.shape, dtype=bool)
    base[::4] = True
    refinement = np.zeros(t.shape, dtype=bool)
    refinement[::2] = True
    off_refinement = ~refinement
    interior = (t >= 1.5) & (t < 8.0)
    cases: list[CaseData] = []
    for index, (case_id, label) in enumerate(CASE_SPECS):
        states: dict[str, StateSeries] = {}
        for state_index, state in enumerate(STATES):
            phase = 0.12 * index + 0.09 * state_index
            u = 1.05 + 0.12 * np.sin(0.7 * t + phase)
            u[t < 0.55] = 3.0
            u[t > 8.4] = 0.0
            population = (
                210.0 * np.exp(-2.0 * np.minimum(t, 1.0))
                + 82.0
                + 1.5 * state_index
                + 1.2 * np.sin(0.8 * t + phase)
            )
            H_u = 1.6e-3 * np.sin(0.8 * t + phase)
            dH_u = 7.0e-4 * np.cos(1.1 * t + phase)
            d2H_u = 4.5e-3 * np.sin(1.7 * t + phase)
            states[state] = StateSeries(
                u=u,
                population=population,
                H_u=H_u,
                dH_u_dt=dH_u,
                d2H_u_dt2=d2H_u,
            )
        cases.append(
            CaseData(
                case_id=case_id,
                label=label,
                path=None,
                input_sha256=None,
                t=t,
                masks=MaskSet(
                    base_grid=base,
                    refinement_grid=refinement,
                    strict_off_refinement=off_refinement,
                    singular_interior=interior,
                    off_refinement_source=(
                        "layout-smoke complement of synthetic refinement grid"
                    ),
                ),
                states=states,
                sibling_summary=None,
            )
        )
    return cases


def validate_cases(
    cases: Sequence[CaseData],
    *,
    production: bool,
) -> None:
    require(len(cases) == 3, "exactly three cases are required")
    reference = cases[0]
    for case in cases[1:]:
        require(
            np.array_equal(case.t, reference.t),
            "all three inputs must use identical time coordinates",
        )
        require(
            np.array_equal(
                case.masks.base_grid,
                reference.masks.base_grid,
            ),
            "all three inputs must use the same base-grid mask",
        )
        require(
            np.array_equal(
                case.masks.refinement_grid,
                reference.masks.refinement_grid,
            ),
            "all three inputs must use the same refinement-grid mask",
        )
        require(
            np.array_equal(
                case.masks.strict_off_refinement,
                reference.masks.strict_off_refinement,
            ),
            "all three inputs must use the same off-refinement mask",
        )
    require(
        int(reference.masks.base_grid.sum()) == 801,
        "base-grid mask must contain 801 nodes",
    )
    require(
        np.array_equal(
            reference.masks.strict_off_refinement,
            ~reference.masks.refinement_grid,
        ),
        "off-refinement mask must be the exact refinement complement",
    )
    require(
        not np.any(
            reference.masks.base_grid
            & reference.masks.strict_off_refinement
        ),
        "off-refinement queries overlap base-grid nodes",
    )
    if production:
        require(
            reference.t.size == 800 * 32 + 1,
            "production curves must use the multiplier-32 grid",
        )
        require(
            np.array_equal(
                np.flatnonzero(reference.masks.base_grid),
                np.arange(0, reference.t.size, 32),
            ),
            "production base-grid nodes must occur every 32 dense points",
        )
        require(
            np.array_equal(
                np.flatnonzero(reference.masks.refinement_grid),
                np.arange(0, reference.t.size, 4),
            ),
            "production multiplier-8 refinement nodes must occur every "
            "four multiplier-32 points",
        )
        require(
            int(reference.masks.refinement_grid.sum()) == 6401,
            "production multiplier-8 grid must contain 6,401 nodes",
        )
        require(
            int(reference.masks.strict_off_refinement.sum()) == 19200,
            "production off-refinement mask must contain 19,200 points",
        )
        midpoint_indices = np.arange(16, reference.t.size - 1, 32)
        require(
            midpoint_indices.size == 800,
            "production grid must contain 800 base-interval midpoints",
        )
        require(
            not np.any(reference.masks.base_grid[midpoint_indices]),
            "midpoint queries must be outside the base-grid support",
        )


def padded_limits(
    values: np.ndarray,
    *,
    symmetric: bool = False,
    nonnegative: bool = False,
) -> tuple[float, float]:
    require(values.size > 0, "cannot derive limits from an empty array")
    low = float(np.min(values))
    high = float(np.max(values))
    if symmetric:
        bound = max(abs(low), abs(high), 1.0e-12)
        return (-1.12 * bound, 1.12 * bound)
    span = high - low
    if span <= 1.0e-12:
        span = max(abs(low), 1.0) * 0.1
    low -= 0.08 * span
    high += 0.08 * span
    if nonnegative:
        low = max(0.0, low)
    return low, high


def y_limits(
    cases: Sequence[CaseData],
    states: Sequence[str],
    interior: np.ndarray,
    *,
    shared_derivative_scale: bool,
) -> dict[str, tuple[float, float]]:
    values: dict[str, list[np.ndarray]] = {
        "u": [],
        "population": [],
        "H_u": [],
        "dH_u_dt": [],
        "d2H_u_dt2": [],
    }
    for case in cases:
        for state in states:
            series = case.states[state]
            values["u"].append(series.u)
            values["population"].append(series.population)
            for field in DERIVATIVES:
                values[field].append(
                    np.asarray(getattr(series, field))[interior]
                )
    limits = {
        "u": padded_limits(np.concatenate(values["u"])),
        "population": padded_limits(
            np.concatenate(values["population"]),
            nonnegative=True,
        ),
    }
    if shared_derivative_scale:
        shared_limit = padded_limits(
            np.concatenate(
                [
                    values[field][case_state]
                    for field in DERIVATIVES
                    for case_state in range(len(values[field]))
                ]
            ),
            symmetric=True,
        )
        limits.update({field: shared_limit for field in DERIVATIVES})
    else:
        for field in DERIVATIVES:
            limits[field] = padded_limits(
                np.concatenate(values[field]),
                symmetric=True,
            )
    return limits


def selected_marker_indices(
    mask: np.ndarray,
    domain: np.ndarray,
    maximum: int,
    *,
    centered_bins: bool = False,
) -> np.ndarray:
    eligible = np.flatnonzero(mask & domain)
    if eligible.size <= maximum:
        return eligible
    if centered_bins:
        positions = (
            (np.arange(maximum, dtype=np.float64) + 0.5)
            * eligible.size
            / maximum
            - 0.5
        )
    else:
        positions = np.linspace(0, eligible.size - 1, maximum)
    return eligible[np.unique(np.rint(positions).astype(int))]


def clean_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(COLORS["spine"])
    axis.spines["bottom"].set_color(COLORS["spine"])
    axis.tick_params(length=2.2, width=0.55, pad=1.5)


def regime_transitions(case: CaseData) -> tuple[float, float]:
    """Return the nominal upper-to-interior and interior-to-lower times."""
    u = case.states["nominal"].u
    t = case.t
    leave_upper = np.flatnonzero((t > 0.0) & (u < 3.0 - 1.0e-4))
    enter_lower = np.flatnonzero((t > 5.0) & (u <= 1.0e-4))
    require(leave_upper.size > 0, f"{case.case_id}: no upper exit")
    require(enter_lower.size > 0, f"{case.case_id}: no lower entry")
    entry = float(t[leave_upper[0]])
    exit_ = float(t[enter_lower[0]])
    require(0.0 < entry < exit_ < 10.0, "invalid regime transitions")
    return entry, exit_


def add_regime_guides(
    axis: plt.Axes,
    entry: float,
    exit_: float,
    *,
    label_regimes: bool,
) -> None:
    axis.axvspan(
        0.0,
        entry,
        color=COLORS["upper_band"],
        alpha=0.55,
        linewidth=0.0,
        zorder=0,
    )
    axis.axvspan(
        entry,
        exit_,
        color=COLORS["interior_band"],
        alpha=0.42,
        linewidth=0.0,
        zorder=0,
    )
    axis.axvspan(
        exit_,
        10.0,
        color=COLORS["lower_band"],
        alpha=0.52,
        linewidth=0.0,
        zorder=0,
    )
    for switch_time in (entry, exit_):
        axis.axvline(
            switch_time,
            color=COLORS["transition"],
            linestyle=(0, (1.2, 1.4)),
            linewidth=0.55,
            zorder=2,
        )
    if label_regimes:
        axis.text(
            max(0.13, 0.5 * entry),
            0.05,
            "upper",
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=4.4,
            color=COLORS["transition"],
            transform=axis.get_xaxis_transform(),
        )
        axis.text(
            0.5 * (entry + exit_),
            0.05,
            "interior",
            ha="center",
            va="bottom",
            fontsize=5.0,
            color=COLORS["transition"],
            transform=axis.get_xaxis_transform(),
        )
        axis.text(
            0.5 * (exit_ + 10.0),
            0.05,
            "lower",
            ha="center",
            va="bottom",
            fontsize=5.0,
            color=COLORS["transition"],
            transform=axis.get_xaxis_transform(),
        )


def draw_markers(
    axis: plt.Axes,
    t: np.ndarray,
    values: np.ndarray,
    node_indices: np.ndarray,
    query_indices: np.ndarray,
    color: str,
    query_color: str,
    *,
    show_query_markers: bool,
) -> None:
    axis.plot(
        t[node_indices],
        values[node_indices],
        linestyle="none",
        marker="o",
        markersize=2.8,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=0.65,
        alpha=1.0,
        zorder=5,
    )
    if show_query_markers:
        axis.plot(
            t[query_indices],
            values[query_indices],
            linestyle="none",
            marker="x",
            markersize=3.1,
            markeredgewidth=0.75,
            color=query_color,
            alpha=1.0,
            zorder=6,
        )


def plot_state_curve(
    axis: plt.Axes,
    case: CaseData,
    state: str,
    field: str,
    domain: np.ndarray,
    node_indices: np.ndarray,
    query_indices: np.ndarray,
    supplemental: bool,
    show_query_markers: bool,
) -> None:
    series = case.states[state]
    values = np.asarray(getattr(series, field))
    color = STATE_COLORS[state] if supplemental else STATE_COLORS["nominal"]
    axis.plot(
        case.t[domain],
        values[domain],
        color=color,
        linestyle=STATE_LINESTYLES[state],
        linewidth=0.95 if state == "nominal" else 0.85,
        zorder=3,
    )
    draw_markers(
        axis,
        case.t,
        values,
        node_indices,
        query_indices,
        color,
        color if supplemental else STATE_COLORS["resistant_heavy"],
        show_query_markers=show_query_markers,
    )


def figure_legends(
    figure: plt.Figure,
    supplemental: bool,
    show_query_markers: bool,
) -> None:
    marker_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=(
                COLORS["marker_legend"]
                if supplemental
                else STATE_COLORS["nominal"]
            ),
            markeredgewidth=0.7,
            markersize=3.9,
            label=r"$n=800$ node",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            linestyle="none",
            color=(
                COLORS["marker_legend"]
                if supplemental
                else STATE_COLORS["resistant_heavy"]
            ),
            markeredgewidth=0.7,
            markersize=4.0,
            label=(
                r"outside $q=8$ grid"
                if supplemental
                else "interval midpoint"
            ),
        ),
    ]
    if supplemental:
        state_handles = [
            Line2D(
                [0],
                [0],
                color=STATE_COLORS[state],
                linestyle=STATE_LINESTYLES[state],
                linewidth=1.05,
                label=STATE_LABELS[state],
            )
            for state in STATES
        ]
        state_legend = figure.legend(
            handles=state_handles,
            loc="upper left",
            bbox_to_anchor=(0.095, 0.995),
            ncol=2,
            frameon=False,
            handlelength=2.0,
            columnspacing=1.0,
        )
        figure.add_artist(state_legend)
        figure.legend(
            handles=marker_handles,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.995),
            ncol=2,
            frameon=False,
            columnspacing=1.0,
        )
    else:
        figure.legend(
            handles=(
                marker_handles
                if show_query_markers
                else marker_handles[:1]
            ),
            loc="upper center",
            bbox_to_anchor=(0.55, 0.995),
            ncol=2 if show_query_markers else 1,
            frameon=False,
            columnspacing=1.15,
        )


def build_figure(
    cases: Sequence[CaseData],
    *,
    supplemental: bool,
    interior_start: float,
    interior_end: float,
    full_marker_count: int,
    interior_marker_count: int,
    show_query_markers: bool = True,
) -> tuple[plt.Figure, dict[str, Any]]:
    states = STATES if supplemental else ("nominal",)
    t = cases[0].t
    full = np.ones(t.shape, dtype=bool)
    interior = (t >= interior_start) & (t < interior_end)
    midpoint = np.zeros(t.shape, dtype=bool)
    midpoint[16:-1:32] = True
    require(int(midpoint.sum()) == 800, "expected 800 interval midpoints")
    shared_derivative_scale = not supplemental
    limits = y_limits(
        cases,
        states,
        interior,
        shared_derivative_scale=shared_derivative_scale,
    )
    domains = {
        "u": full,
        "population": full,
        "H_u": interior,
        "dH_u_dt": interior,
        "d2H_u_dt2": interior,
    }
    if supplemental:
        curve_support = full
        query_mask = cases[0].masks.strict_off_refinement
        node_indices = {
            "full": selected_marker_indices(
                cases[0].masks.base_grid,
                full,
                full_marker_count,
            ),
            "interior": selected_marker_indices(
                cases[0].masks.base_grid,
                interior,
                interior_marker_count,
            ),
        }
        query_indices = {
            "full": selected_marker_indices(
                query_mask,
                full,
                full_marker_count,
                centered_bins=True,
            ),
            "interior": selected_marker_indices(
                query_mask,
                interior,
                interior_marker_count,
                centered_bins=True,
            ),
        }
    else:
        curve_support = cases[0].masks.base_grid | midpoint
        query_mask = midpoint
        node_indices = {
            "full": selected_marker_indices(
                cases[0].masks.base_grid,
                full,
                full_marker_count,
            ),
            "interior": selected_marker_indices(
                cases[0].masks.base_grid,
                interior,
                interior_marker_count,
            ),
        }
        query_indices = {
            "full": selected_marker_indices(
                query_mask,
                full,
                full_marker_count,
                centered_bins=True,
            ),
            "interior": selected_marker_indices(
                query_mask,
                interior,
                interior_marker_count,
                centered_bins=True,
            ),
        }

    figure, axes = plt.subplots(
        5,
        3,
        figsize=(7.05, 7.45 if supplemental else 4.82),
        gridspec_kw=(
            None
            if supplemental
            else {"height_ratios": (1.0, 1.0, 0.78, 0.78, 0.78)}
        ),
        sharex=False,
        squeeze=False,
    )
    row_fields = ("u", "population", *DERIVATIVES)
    transitions = {
        case.case_id: regime_transitions(case) for case in cases
    }
    for column, case in enumerate(cases):
        entry, exit_ = transitions[case.case_id]
        axes[0, column].set_title(case.label, fontweight="semibold", pad=3.5)
        for row, field in enumerate(row_fields):
            axis = axes[row, column]
            if not supplemental:
                add_regime_guides(
                    axis,
                    entry,
                    exit_,
                    label_regimes=(field == "u"),
                )
            domain = domains[field] & curve_support
            marker_domain = "full" if field in {"u", "population"} else "interior"
            for state in states:
                plot_state_curve(
                    axis,
                    case,
                    state,
                    field,
                    domain,
                    node_indices[marker_domain],
                    query_indices[marker_domain],
                    supplemental,
                    show_query_markers,
                )
            axis.set_ylim(*limits[field])
            if field in DERIVATIVES:
                axis.axhline(
                    0.0,
                    color=COLORS["zero"],
                    linewidth=0.62,
                    zorder=1,
                )
            axis.set_xlim(0.0, 10.0)
            if column == 0:
                axis.set_ylabel(ROW_LABELS[field], labelpad=3.0)
            if row == 4:
                axis.set_xlabel(r"time $t$", labelpad=1.5)
            else:
                axis.tick_params(labelbottom=False)
            clean_axis(axis)

    figure_legends(figure, supplemental, show_query_markers)
    figure.subplots_adjust(
        left=0.105,
        right=0.992,
        bottom=0.055,
        top=0.955 if supplemental else 0.915,
        hspace=0.16,
        wspace=0.23,
    )
    metadata = {
        "states": list(states),
        "derivative_scale_policy": (
            "single shared symmetric y-limit across all nine derivative "
            "panels, computed from every plotted point on 1.5 <= t < 8"
            if shared_derivative_scale
            else
            "row-specific symmetric y-limits shared across the three "
            "cases, computed from every plotted point on 1.5 <= t < 8"
        ),
        "y_limits": {
            field: [float(value) for value in limits[field]]
            for field in row_fields
        },
        "marker_subsampling": {
            "full": {
                "base_grid_indices": node_indices["full"].tolist(),
                "base_grid_times": t[node_indices["full"]].tolist(),
                "query_indices": query_indices["full"].tolist(),
                "query_times": t[query_indices["full"]].tolist(),
            },
            "interior": {
                "base_grid_indices": node_indices["interior"].tolist(),
                "base_grid_times": t[node_indices["interior"]].tolist(),
                "query_indices": query_indices["interior"].tolist(),
                "query_times": t[query_indices["interior"]].tolist(),
            },
            "curves_use_all_coordinates": True,
            "main_curve_support": (
                None
                if supplemental
                else "801 base-grid nodes plus all 800 interval midpoints"
            ),
            "query_semantics": (
                "strict query outside multiplier-8 refinement grid"
                if supplemental
                else "midpoint of each base-grid interval"
            ),
            "query_markers_displayed": bool(show_query_markers),
        },
        "nominal_regime_transitions": {
            case.case_id: {
                "upper_to_interior": transitions[case.case_id][0],
                "interior_to_lower": transitions[case.case_id][1],
            }
            for case in cases
        },
    }
    return figure, metadata


def scalar_summary(values: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(values.size),
        "rms": float(np.sqrt(np.mean(values**2))),
        "mean_abs": float(np.mean(np.abs(values))),
        "max_abs": float(np.max(np.abs(values))),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def mask_record(
    case: CaseData,
    interior: np.ndarray,
) -> dict[str, Any]:
    masks: dict[str, np.ndarray | None] = {
        "base_grid": case.masks.base_grid,
        "refinement_grid": case.masks.refinement_grid,
        "strict_held_out_from_refinement": (
            case.masks.strict_off_refinement
        ),
        "singular_interior": case.masks.singular_interior,
    }
    record: dict[str, Any] = {
        "total_points": int(case.t.size),
        "dense_grid_definition": (
            "multiplier-32 evaluation grid: 32 subintervals per one of "
            "the 800 base-grid intervals"
        ),
        "refinement_grid_definition": (
            "multiplier-8 scalar-refinement grid: every fourth point of "
            "the multiplier-32 evaluation grid"
        ),
        "off_refinement_definition": (
            "strict multiplier-32 query points outside the multiplier-8 "
            "refinement grid"
        ),
        "off_refinement_source": case.masks.off_refinement_source,
    }
    for name, mask in masks.items():
        if mask is None:
            record[name] = None
            continue
        record[name] = {
            "count": int(mask.sum()),
            "interior_count": int((mask & interior).sum()),
            "packed_mask_sha256": mask_sha256(mask),
        }
    refinement = case.masks.refinement_grid
    off_refinement = case.masks.strict_off_refinement
    record["off_refinement_overlap_with_refinement_grid"] = int(
        (off_refinement & refinement).sum()
    )
    record["off_refinement_outside_refinement_grid"] = int(
        (off_refinement & ~refinement).sum()
    )
    record["base_grid_overlap_with_off_refinement"] = int(
        (case.masks.base_grid & off_refinement).sum()
    )
    return record


def metrics_record(
    case: CaseData,
    interior: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    sample_masks = {
        "all_interior": interior,
        "base_grid_interior": interior & case.masks.base_grid,
        "refinement_grid_interior": (
            interior & case.masks.refinement_grid
        ),
        "strict_held_out_from_refinement_interior": (
            interior & case.masks.strict_off_refinement
        ),
    }
    for state in STATES:
        series = case.states[state]
        state_result: dict[str, Any] = {
            "control_full": scalar_summary(series.u),
            "total_population_full": scalar_summary(series.population),
            "scalar_interior": {},
        }
        for field in DERIVATIVES:
            values = np.asarray(getattr(series, field))
            state_result["scalar_interior"][field] = {
                sample_name: scalar_summary(values[mask])
                for sample_name, mask in sample_masks.items()
            }
        result[state] = state_result
    return result


def limits_contain_data(
    cases: Sequence[CaseData],
    states: Sequence[str],
    limits: Mapping[str, Sequence[float]],
    interior: np.ndarray,
) -> bool:
    for case in cases:
        for state in states:
            series = case.states[state]
            for field in ("u", "population", *DERIVATIVES):
                values = np.asarray(getattr(series, field))
                if field in DERIVATIVES:
                    values = values[interior]
                low, high = limits[field]
                if float(np.min(values)) < low or float(np.max(values)) > high:
                    return False
    return True


def output_paths(prefix: Path) -> dict[str, Path]:
    return {
        "main_pdf": prefix.with_name(prefix.name + "_main.pdf"),
        "main_png": prefix.with_name(prefix.name + "_main.png"),
        "supplement_pdf": prefix.with_name(prefix.name + "_supplement.pdf"),
        "supplement_png": prefix.with_name(prefix.name + "_supplement.png"),
        "manifest": prefix.with_name(prefix.name + "_manifest.json"),
    }


def save_figure(figure: plt.Figure, pdf: Path, png: Path) -> None:
    figure.savefig(pdf, bbox_inches="tight", pad_inches=0.025)
    figure.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    configure_style()
    prefix = args.out_prefix.expanduser().resolve()
    paths = output_paths(prefix)
    if not args.overwrite:
        existing = [path for path in paths.values() if path.exists()]
        require(
            not existing,
            "refusing to overwrite: " + ", ".join(str(path) for path in existing),
        )
    prefix.parent.mkdir(parents=True, exist_ok=True)

    if args.layout_smoke:
        cases = smoke_cases()
    else:
        input_paths = (args.time_npz, args.cf_npz, args.der_npz)
        assert all(path is not None for path in input_paths)
        cases = [
            load_case(path, case_id, label)
            for path, (case_id, label) in zip(
                input_paths,
                CASE_SPECS,
                strict=True,
            )
            if path is not None
        ]
    validate_cases(cases, production=not args.layout_smoke)

    interior = (
        (cases[0].t >= args.interior_start)
        & (cases[0].t < args.interior_end)
    )
    require(interior.any(), "interior contains no coordinates")
    for case in cases:
        recorded_interior = case.masks.singular_interior
        if recorded_interior is not None:
            require(
                np.array_equal(recorded_interior, interior),
                f"{case.case_id}: requested interior disagrees with NPZ mask",
            )
    main_figure, main_layout = build_figure(
        cases,
        supplemental=False,
        interior_start=args.interior_start,
        interior_end=args.interior_end,
        full_marker_count=args.full_marker_count,
        interior_marker_count=args.interior_marker_count,
        show_query_markers=not args.hide_main_query_markers,
    )
    supplement_figure, supplement_layout = build_figure(
        cases,
        supplemental=True,
        interior_start=args.interior_start,
        interior_end=args.interior_end,
        full_marker_count=max(7, args.full_marker_count - 4),
        interior_marker_count=max(7, args.interior_marker_count - 3),
        show_query_markers=True,
    )
    save_figure(
        main_figure,
        paths["main_pdf"],
        paths["main_png"],
    )
    save_figure(
        supplement_figure,
        paths["supplement_pdf"],
        paths["supplement_png"],
    )

    main_ok = limits_contain_data(
        cases,
        ("nominal",),
        main_layout["y_limits"],
        interior,
    )
    supplement_ok = limits_contain_data(
        cases,
        STATES,
        supplement_layout["y_limits"],
        interior,
    )
    require(main_ok and supplement_ok, "computed y-limits clip plotted data")

    manifest = {
        "schema": "final-nominal-pmp-figure-v3",
        "layout_smoke": bool(args.layout_smoke),
        "interior": [args.interior_start, args.interior_end],
        "figure_contract": {
            "main_states": ["nominal"],
            "supplement_states": list(STATES),
            "rows": ["u", "sum_N", *DERIVATIVES],
            "columns": [label for _, label in CASE_SPECS],
            "control_and_population_horizon": [0.0, 10.0],
            "derivative_horizon": [
                args.interior_start,
                args.interior_end,
            ],
            "derivative_axes_symmetric_about_zero": True,
            "main_derivative_scale_policy": (
                "one global symmetric y-limit shared by all nine "
                "derivative panels"
            ),
            "supplement_derivative_scale_policy": (
                "one tight symmetric y-limit per derivative row, shared "
                "across cases"
            ),
            "main_curve_grid": (
                "801 base-grid coordinates plus all 800 interval midpoints"
            ),
            "supplement_curve_grid": (
                "all multiplier-32 coordinates in each plotted domain"
            ),
            "open_circle_semantics": (
                "representative markers from the 801-node base-grid support"
            ),
            "main_cross_semantics": (
                (
                    "representative markers from all 800 direct "
                    "base-interval midpoint queries"
                )
                if not args.hide_main_query_markers
                else "not displayed; midpoint values remain in every curve"
            ),
            "supplement_cross_semantics": (
                "strict multiplier-32 query outside multiplier-8 "
                "refinement grid"
            ),
            "main_markers_subsampled_for_readability": True,
            "main_query_markers_displayed": (
                not args.hide_main_query_markers
            ),
            "supplement_markers_subsampled_for_readability": True,
            "pdf_fonttype": 42,
        },
        "inputs": {
            case.case_id: {
                "path": str(case.path) if case.path is not None else None,
                "sha256": case.input_sha256,
                "sibling_checkpoint_sha256": (
                    case.sibling_summary.get("checkpoint_sha256")
                    if case.sibling_summary is not None
                    else None
                ),
                "mask_summary": mask_record(case, interior),
                "metrics": metrics_record(case, interior),
            }
            for case in cases
        },
        "main_layout": {
            **main_layout,
            "all_plotted_data_within_y_limits": main_ok,
        },
        "supplement_layout": {
            **supplement_layout,
            "all_plotted_data_within_y_limits": supplement_ok,
        },
        "outputs": {
            name: {
                "path": str(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in paths.items()
            if name != "manifest"
        },
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"outputs": manifest["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
