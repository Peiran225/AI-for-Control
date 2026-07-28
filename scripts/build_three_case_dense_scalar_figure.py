#!/usr/bin/env python3
"""Build publication figures for three dense-time scalar diagnostics.

Two vector figures are written from the same validated data:

* ``*_main`` is a compact 2-by-3 figure.  The first row shows the complete
  control horizon.  The second row overlays the three scalar Hamiltonian
  derivatives on the singular interior ``[1.5, 8)``.
* ``*_supplement`` is a full-horizon 4-by-3 diagnostic figure with one row
  for the control and one row for each scalar derivative.

Production input can be one combined NPZ with keys such as
``time_only__nominal__H_u`` or a per-case NPZ from
``evaluate_single_feedback_offgrid_scalar.py`` with keys such as
``nominal__H_u``.  The former may be declared once at config level; the
latter may be declared once per case.  The legacy two-file-per-case config
is also accepted.

``--layout-smoke`` loads and plots no numerical data.  It creates only the
empty axes, labels, legends, and explanatory text used to inspect layout.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


TARGET_WEIGHTS = {"alpha": 1.0, "beta": 40.0, "gamma": 8000.0}
REQUIRED_SERIES = ("t", "u", "H_u", "dH_u_dt", "d2H_u_dt2")
ALIASES = {
    "t": ("t", "time"),
    "u": ("u", "control", "action"),
    "H_u": ("H_u", "psi", "switching_function"),
    "dH_u_dt": ("dH_u_dt", "dot_psi", "H_u_dot"),
    "d2H_u_dt2": ("d2H_u_dt2", "ddot_psi", "H_u_ddot"),
}
STATE_IDS = ("nominal", "resistant_heavy")
STATE_LABELS = {
    "nominal": "Nominal",
    "resistant_heavy": "Resistant-heavy",
}
STATE_STYLES = {
    "nominal": "-",
    "resistant_heavy": "--",
}
DERIVATIVE_FIELDS = ("H_u", "dH_u_dt", "d2H_u_dt2")
DERIVATIVE_LABELS = {
    "H_u": r"$H_u$",
    "dH_u_dt": r"$\mathrm{d}H_u/\mathrm{d}t$",
    "d2H_u_dt2": r"$\mathrm{d}^2H_u/\mathrm{d}t^2$",
}
DERIVATIVE_COLORS = {
    "H_u": "#2468A2",
    "dH_u_dt": "#D05A32",
    "d2H_u_dt2": "#6A51A3",
}
COLORS = {
    "control": "#214A67",
    "grid": "#D8DEE5",
    "excluded": "#EEF2F6",
    "zero": "#59636E",
    "text": "#25313C",
    "muted": "#78838F",
    "frame": "#AAB4BF",
}


@dataclass(frozen=True)
class Trajectory:
    path: Path
    t: np.ndarray
    u: np.ndarray
    H_u: np.ndarray
    dH_u_dt: np.ndarray
    d2H_u_dt2: np.ndarray
    physical_scale_factor: float
    key_prefix: str


@dataclass(frozen=True)
class Case:
    case_id: str
    label: str
    nominal: Trajectory
    resistant_heavy: Trajectory

    def state(self, state_id: str) -> Trajectory:
        return getattr(self, state_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON configuration for three real cases.",
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        required=True,
        help=(
            "Output stem. Files are written as *_main.pdf/png, "
            "*_supplement.pdf/png, and *_manifest.json."
        ),
    )
    parser.add_argument(
        "--layout-smoke",
        action="store_true",
        help="Draw the empty layout only; load and plot no numerical data.",
    )
    parser.add_argument("--expected-points", type=int, default=12801)
    parser.add_argument("--zoom-start", type=float, default=1.5)
    parser.add_argument("--zoom-end", type=float, default=8.0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing figure files.",
    )
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "mathtext.fontset": "stixsans",
            "font.size": 7.0,
            "axes.titlesize": 8.4,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.4,
            "axes.linewidth": 0.62,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.42,
            "grid.alpha": 0.66,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "path.simplify": True,
            "path.simplify_threshold": 0.015,
        }
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    require(isinstance(value, dict), f"{path}: top-level JSON must be an object")
    return value


def resolve_alias(keys: Iterable[str], canonical: str, path: Path) -> str:
    key_set = set(keys)
    for alias in ALIASES[canonical]:
        if alias in key_set:
            return alias
    raise ValueError(
        f"{path}: missing {canonical}; accepted names are {ALIASES[canonical]}"
    )


def resolve_prefixed_alias(
    keys: Iterable[str],
    canonical: str,
    prefixes: Sequence[str],
    path: Path,
) -> tuple[str, str]:
    key_set = set(keys)
    for prefix in prefixes:
        for alias in ALIASES[canonical]:
            key = f"{prefix}__{alias}" if prefix else alias
            if key in key_set:
                return key, prefix
    prefix_text = ", ".join(repr(prefix) for prefix in prefixes)
    raise ValueError(
        f"{path}: missing {canonical} for prefixes [{prefix_text}]; "
        f"accepted suffixes are {ALIASES[canonical]}"
    )


def load_npz_series(
    path: Path,
    prefixes: Sequence[str],
) -> tuple[dict[str, np.ndarray], str]:
    with np.load(path, allow_pickle=False) as archive:
        time_key = resolve_alias(archive.files, "t", path)
        resolved: dict[str, str] = {"t": time_key}
        used_prefix: str | None = None
        for canonical in REQUIRED_SERIES[1:]:
            key, prefix = resolve_prefixed_alias(
                archive.files,
                canonical,
                prefixes,
                path,
            )
            if used_prefix is None:
                used_prefix = prefix
            require(
                prefix == used_prefix,
                f"{path}: mixed prefixes while loading one trajectory",
            )
            resolved[canonical] = key
        assert used_prefix is not None
        raw = {
            canonical: np.asarray(archive[key], dtype=np.float64)
            for canonical, key in resolved.items()
        }
    return raw, used_prefix


def load_csv_series(path: Path) -> tuple[dict[str, np.ndarray], str]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None, f"{path}: no CSV header")
        mapping = {
            canonical: resolve_alias(reader.fieldnames, canonical, path)
            for canonical in REQUIRED_SERIES
        }
        columns: dict[str, list[float]] = {
            canonical: [] for canonical in REQUIRED_SERIES
        }
        for row in reader:
            for canonical, source_name in mapping.items():
                columns[canonical].append(float(row[source_name]))
    return (
        {
            canonical: np.asarray(values, dtype=np.float64)
            for canonical, values in columns.items()
        },
        "",
    )


def load_raw_series(
    path: Path,
    prefixes: Sequence[str],
) -> tuple[dict[str, np.ndarray], str]:
    require(path.exists(), f"input does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return load_npz_series(path, prefixes)
    if suffix == ".csv":
        require(
            prefixes == ("",) or "" in prefixes,
            f"{path}: prefixed multi-trajectory CSV input is not supported",
        )
        return load_csv_series(path)
    raise ValueError(f"{path}: expected .npz or .csv")


def merge_spec(
    global_spec: Mapping[str, Any],
    case_spec: Mapping[str, Any],
    state_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for source in (global_spec, case_spec, state_spec or {}):
        for key in ("path", "input_scale", "source_weights"):
            if key in source:
                merged[key] = source[key]
    return merged


def physical_scale_factor(spec: Mapping[str, Any], path: Path) -> float:
    input_scale = spec.get("input_scale")
    require(
        input_scale in {"physical", "normalized"},
        f"{path}: input_scale must be physical or normalized",
    )
    source = spec.get("source_weights")
    require(isinstance(source, dict), f"{path}: source_weights must be explicit")
    for key in TARGET_WEIGHTS:
        require(key in source, f"{path}: source_weights missing {key}")
        require(
            float(source[key]) > 0.0,
            f"{path}: source weight {key} must be positive",
        )

    ratios = np.asarray(
        [TARGET_WEIGHTS[key] / float(source[key]) for key in TARGET_WEIGHTS],
        dtype=np.float64,
    )
    if input_scale == "physical":
        require(
            np.allclose(ratios, 1.0, rtol=0.0, atol=1e-12),
            f"{path}: physical input must declare weights {TARGET_WEIGHTS}",
        )
        return 1.0

    require(
        np.max(ratios) - np.min(ratios) <= 1e-10 * np.max(ratios),
        f"{path}: normalized weights are not one common positive scaling of "
        f"{TARGET_WEIGHTS}; ratios={ratios.tolist()}",
    )
    return float(np.mean(ratios))


def load_trajectory(
    spec: Mapping[str, Any],
    prefixes: Sequence[str],
    expected_points: int,
    expected_t: np.ndarray | None,
) -> Trajectory:
    require("path" in spec, "trajectory path is missing from config")
    path = Path(str(spec["path"])).expanduser().resolve()
    raw, used_prefix = load_raw_series(path, prefixes)
    factor = physical_scale_factor(spec, path)
    lengths = {name: len(raw[name]) for name in REQUIRED_SERIES}
    require(
        len(set(lengths.values())) == 1,
        f"{path}: unequal series lengths {lengths}",
    )
    require(
        lengths["t"] == expected_points,
        f"{path}: expected {expected_points} points, found {lengths['t']}",
    )
    for name, values in raw.items():
        require(values.ndim == 1, f"{path}: {name} must be one-dimensional")
        require(
            np.all(np.isfinite(values)),
            f"{path}: {name} contains non-finite values",
        )
    require(np.all(np.diff(raw["t"]) > 0.0), f"{path}: t must increase strictly")
    require(abs(raw["t"][0]) <= 1e-12, f"{path}: horizon must start at t=0")
    require(
        abs(raw["t"][-1] - 10.0) <= 1e-10,
        f"{path}: horizon must end at T=10",
    )
    if expected_t is not None:
        require(
            np.allclose(raw["t"], expected_t, rtol=0.0, atol=1e-12),
            f"{path}: time coordinates differ from the first trajectory",
        )
    require(
        np.min(raw["u"]) >= -1e-8 and np.max(raw["u"]) <= 3.0 + 1e-8,
        f"{path}: control leaves [0,3]",
    )
    return Trajectory(
        path=path,
        t=raw["t"],
        u=raw["u"],
        H_u=factor * raw["H_u"],
        dH_u_dt=factor * raw["dH_u_dt"],
        d2H_u_dt2=factor * raw["d2H_u_dt2"],
        physical_scale_factor=factor,
        key_prefix=used_prefix,
    )


def load_cases(config: Mapping[str, Any], expected_points: int) -> list[Case]:
    target = config.get("target_weights")
    require(isinstance(target, dict), "target_weights must be an object")
    require(
        all(
            key in target
            and np.isclose(
                float(target[key]),
                value,
                rtol=0.0,
                atol=1e-12,
            )
            for key, value in TARGET_WEIGHTS.items()
        ),
        "target_weights must be alpha=1, beta=40, gamma=8000",
    )
    case_specs = config.get("cases")
    require(
        isinstance(case_specs, list) and len(case_specs) == 3,
        "exactly three cases are required",
    )
    global_spec = {
        key: config[key]
        for key in ("path", "input_scale", "source_weights")
        if key in config
    }
    cases: list[Case] = []
    expected_t: np.ndarray | None = None
    for raw_case_spec in case_specs:
        require(isinstance(raw_case_spec, dict), "each case must be an object")
        case_id = str(raw_case_spec["id"])
        label = str(raw_case_spec["label"])
        trajectories: dict[str, Trajectory] = {}
        legacy = all(state_id in raw_case_spec for state_id in STATE_IDS)
        for state_id in STATE_IDS:
            if legacy:
                state_entry = raw_case_spec[state_id]
                require(
                    isinstance(state_entry, dict),
                    f"{case_id}.{state_id} must be an object",
                )
                merged = merge_spec(global_spec, raw_case_spec, state_entry)
                prefixes = ("",)
            else:
                merged = merge_spec(global_spec, raw_case_spec)
                explicit_prefix = raw_case_spec.get(f"{state_id}_prefix")
                prefix_candidates = []
                if explicit_prefix is not None:
                    prefix_candidates.append(str(explicit_prefix))
                prefix_candidates.extend(
                    (
                        f"{case_id}__{state_id}",
                        state_id,
                    )
                )
                prefixes = tuple(dict.fromkeys(prefix_candidates))
            trajectory = load_trajectory(
                merged,
                prefixes,
                expected_points,
                expected_t,
            )
            if expected_t is None:
                expected_t = trajectory.t
            trajectories[state_id] = trajectory
        cases.append(
            Case(
                case_id=case_id,
                label=label,
                nominal=trajectories["nominal"],
                resistant_heavy=trajectories["resistant_heavy"],
            )
        )
    require(len({case.case_id for case in cases}) == 3, "case ids must be unique")
    return cases


def padded_limits(values: np.ndarray, symmetric: bool = False) -> tuple[float, float]:
    require(values.size > 0, "cannot derive limits from an empty array")
    low = float(np.min(values))
    high = float(np.max(values))
    if symmetric:
        bound = max(abs(low), abs(high), 1e-12)
        return -1.08 * bound, 1.08 * bound
    span = high - low
    if span <= 1e-12:
        span = max(abs(low), 1.0) * 0.08
    return low - 0.06 * span, high + 0.06 * span


def all_values(
    cases: Sequence[Case],
    fields: Sequence[str],
    mask: np.ndarray | None = None,
) -> np.ndarray:
    arrays = []
    for case in cases:
        for state_id in STATE_IDS:
            trajectory = case.state(state_id)
            for field in fields:
                values = np.asarray(getattr(trajectory, field), dtype=np.float64)
                arrays.append(values if mask is None else values[mask])
    return np.concatenate(arrays)


def clean_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def shade_noninterior(
    axis: plt.Axes,
    zoom_start: float,
    zoom_end: float,
) -> None:
    axis.axvspan(0.0, zoom_start, color=COLORS["excluded"], zorder=-10)
    axis.axvspan(zoom_end, 10.0, color=COLORS["excluded"], zorder=-10)
    axis.axvline(zoom_start, color=COLORS["frame"], linewidth=0.55, linestyle=":")
    axis.axvline(zoom_end, color=COLORS["frame"], linewidth=0.55, linestyle=":")


def plot_control(axis: plt.Axes, case: Case) -> None:
    for state_id in STATE_IDS:
        trajectory = case.state(state_id)
        axis.plot(
            trajectory.t,
            trajectory.u,
            color=COLORS["control"],
            linestyle=STATE_STYLES[state_id],
            linewidth=1.05 if state_id == "nominal" else 0.96,
        )


def plot_derivative_overlay(
    axis: plt.Axes,
    case: Case,
    mask: np.ndarray,
) -> None:
    for field in DERIVATIVE_FIELDS:
        for state_id in STATE_IDS:
            trajectory = case.state(state_id)
            axis.plot(
                trajectory.t[mask],
                getattr(trajectory, field)[mask],
                color=DERIVATIVE_COLORS[field],
                linestyle=STATE_STYLES[state_id],
                linewidth=0.94 if state_id == "nominal" else 0.86,
            )


def add_main_legends(figure: plt.Figure) -> None:
    quantity_handles = [
        Line2D(
            [0],
            [0],
            color=DERIVATIVE_COLORS[field],
            linewidth=1.15,
            label=DERIVATIVE_LABELS[field],
        )
        for field in DERIVATIVE_FIELDS
    ]
    state_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle=STATE_STYLES[state_id],
            linewidth=1.05,
            label=STATE_LABELS[state_id],
        )
        for state_id in STATE_IDS
    ]
    quantity_legend = figure.legend(
        handles=quantity_handles,
        loc="upper center",
        bbox_to_anchor=(0.56, 0.995),
        ncol=3,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.0,
    )
    figure.add_artist(quantity_legend)
    figure.legend(
        handles=state_handles,
        loc="upper left",
        bbox_to_anchor=(0.075, 0.995),
        ncol=2,
        frameon=False,
        handlelength=2.0,
        columnspacing=0.9,
    )


def add_state_legend(figure: plt.Figure) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle=STATE_STYLES[state_id],
            linewidth=1.05,
            label=STATE_LABELS[state_id],
        )
        for state_id in STATE_IDS
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        handlelength=2.3,
        columnspacing=1.4,
    )


def case_labels(cases: Sequence[Case] | None) -> list[str]:
    if cases is not None:
        return [case.label for case in cases]
    return [
        "PMP/KKT-Time",
        "PMP/KKT-CF",
        "PMP/KKT-DER",
    ]


def smoke_annotation(axis: plt.Axes, text: str) -> None:
    axis.text(
        0.5,
        0.52,
        text,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=6.2,
        color=COLORS["muted"],
    )
    axis.set_yticks([])


def build_main_figure(
    cases: Sequence[Case] | None,
    zoom_start: float,
    zoom_end: float,
    layout_smoke: bool,
) -> plt.Figure:
    configure_style()
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(7.02, 3.32),
        sharex=False,
        sharey="row",
        gridspec_kw={
            "hspace": 0.34,
            "wspace": 0.18,
            "height_ratios": [0.86, 1.18],
        },
    )
    labels = case_labels(cases)
    if cases is not None:
        t = cases[0].nominal.t
        interior = (t >= zoom_start) & (t < zoom_end)
        require(np.count_nonzero(interior) > 2, "interior interval is empty")
        shared_derivative_limits = padded_limits(
            all_values(cases, DERIVATIVE_FIELDS, interior),
            symmetric=True,
        )
    else:
        interior = None
        shared_derivative_limits = None

    for col, label in enumerate(labels):
        control_axis = axes[0, col]
        derivative_axis = axes[1, col]
        control_axis.set_title(
            f"({chr(ord('a') + col)}) {label}",
            fontweight="semibold",
            pad=4.0,
        )
        clean_axis(control_axis)
        clean_axis(derivative_axis)
        control_axis.set_xlim(0.0, 10.0)
        control_axis.set_ylim(-0.08, 3.08)
        control_axis.set_xticks([0, 2, 4, 6, 8, 10])
        control_axis.set_xlabel(r"time $t$")
        derivative_axis.set_xlim(zoom_start, zoom_end)
        derivative_axis.set_xticks([2, 4, 6, 8])
        derivative_axis.set_xlabel(r"time $t$")
        derivative_axis.axhline(
            0.0,
            color=COLORS["zero"],
            linewidth=0.56,
            zorder=0,
        )
        if layout_smoke:
            smoke_annotation(control_axis, "full-horizon control layout")
            smoke_annotation(
                derivative_axis,
                "interior scalar-derivative overlay layout",
            )
        else:
            assert cases is not None
            assert interior is not None
            plot_control(control_axis, cases[col])
            plot_derivative_overlay(derivative_axis, cases[col], interior)
            derivative_axis.set_ylim(*shared_derivative_limits)
        if col == 0:
            control_axis.set_ylabel(r"control $u$")
            derivative_axis.set_ylabel(
                "Hamiltonian derivatives\n(physical scale)"
            )

    add_main_legends(figure)
    figure.text(
        0.995,
        0.006,
        r"Derivative panels show the singular interior $1.5\leq t<8$; "
        r"physical objective scale $(\alpha,\beta,\gamma)=(1,40,8000)$.",
        ha="right",
        va="bottom",
        fontsize=5.8,
        color=COLORS["text"],
    )
    figure.subplots_adjust(
        top=0.875,
        bottom=0.14,
        left=0.095,
        right=0.992,
    )
    return figure


def build_supplement_figure(
    cases: Sequence[Case] | None,
    zoom_start: float,
    zoom_end: float,
    layout_smoke: bool,
) -> plt.Figure:
    configure_style()
    figure, axes = plt.subplots(
        4,
        3,
        figsize=(7.02, 5.72),
        sharex="col",
        sharey="row",
        gridspec_kw={
            "hspace": 0.18,
            "wspace": 0.18,
            "height_ratios": [0.75, 1.0, 1.0, 1.0],
        },
    )
    labels = case_labels(cases)
    fields = ("u",) + DERIVATIVE_FIELDS
    row_labels = (
        r"control $u$",
        r"$H_u$ (physical)",
        r"$\mathrm{d}H_u/\mathrm{d}t$ (physical)",
        r"$\mathrm{d}^2H_u/\mathrm{d}t^2$ (physical)",
    )
    row_limits: dict[str, tuple[float, float]] = {}
    if cases is not None:
        row_limits = {
            field: padded_limits(
                all_values(cases, (field,)),
                symmetric=True,
            )
            for field in DERIVATIVE_FIELDS
        }

    for col, label in enumerate(labels):
        axes[0, col].set_title(
            f"({chr(ord('a') + col)}) {label}",
            fontweight="semibold",
            pad=4.0,
        )
        for row, field in enumerate(fields):
            axis = axes[row, col]
            clean_axis(axis)
            axis.set_xlim(0.0, 10.0)
            shade_noninterior(axis, zoom_start, zoom_end)
            if field == "u":
                axis.set_ylim(-0.08, 3.08)
            else:
                axis.axhline(
                    0.0,
                    color=COLORS["zero"],
                    linewidth=0.54,
                    zorder=0,
                )
            if layout_smoke:
                smoke_annotation(
                    axis,
                    "full-horizon control layout"
                    if field == "u"
                    else "full-horizon scalar diagnostic layout",
                )
            else:
                assert cases is not None
                case = cases[col]
                for state_id in STATE_IDS:
                    trajectory = case.state(state_id)
                    axis.plot(
                        trajectory.t,
                        getattr(trajectory, field),
                        color=(
                            COLORS["control"]
                            if field == "u"
                            else DERIVATIVE_COLORS[field]
                        ),
                        linestyle=STATE_STYLES[state_id],
                        linewidth=0.98 if state_id == "nominal" else 0.88,
                    )
                if field != "u":
                    axis.set_ylim(*row_limits[field])
            if col == 0:
                axis.set_ylabel(row_labels[row])
            if row == 3:
                axis.set_xlabel(r"time $t$")

    add_state_legend(figure)
    figure.text(
        0.995,
        0.006,
        r"Complete horizon; physical objective scale "
        r"$(\alpha,\beta,\gamma)=(1,40,8000)$. "
        r"Shading marks $t<1.5$ and $t\geq8$.",
        ha="right",
        va="bottom",
        fontsize=5.8,
        color=COLORS["text"],
    )
    figure.subplots_adjust(
        top=0.93,
        bottom=0.085,
        left=0.105,
        right=0.992,
    )
    return figure


def output_paths(prefix: Path) -> dict[str, Path]:
    return {
        "main_pdf": prefix.with_name(prefix.name + "_main").with_suffix(".pdf"),
        "main_png": prefix.with_name(prefix.name + "_main").with_suffix(".png"),
        "supplement_pdf": prefix.with_name(prefix.name + "_supplement").with_suffix(
            ".pdf"
        ),
        "supplement_png": prefix.with_name(prefix.name + "_supplement").with_suffix(
            ".png"
        ),
        "manifest": prefix.with_name(prefix.name + "_manifest.json"),
    }


def save_outputs(
    main_figure: plt.Figure,
    supplement_figure: plt.Figure,
    prefix: Path,
    overwrite: bool,
    manifest: dict[str, Any],
) -> dict[str, Path]:
    outputs = output_paths(prefix)
    if not overwrite:
        existing = [path for path in outputs.values() if path.exists()]
        require(
            not existing,
            f"refusing to overwrite existing files: {existing}",
        )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    main_figure.savefig(
        outputs["main_pdf"],
        bbox_inches="tight",
        pad_inches=0.025,
    )
    main_figure.savefig(
        outputs["main_png"],
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.025,
    )
    supplement_figure.savefig(
        outputs["supplement_pdf"],
        bbox_inches="tight",
        pad_inches=0.025,
    )
    supplement_figure.savefig(
        outputs["supplement_png"],
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.025,
    )
    plt.close(main_figure)
    plt.close(supplement_figure)
    manifest["outputs"] = {key: str(path.resolve()) for key, path in outputs.items()}
    with outputs["manifest"].open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return outputs


def build_manifest(
    args: argparse.Namespace,
    cases: Sequence[Case] | None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "mode": "layout_smoke" if args.layout_smoke else "production",
        "contains_numerical_data": not args.layout_smoke,
        "target_weights": TARGET_WEIGHTS,
        "expected_points": args.expected_points,
        "interior_interval": [args.zoom_start, args.zoom_end],
        "main_layout": {
            "shape": [2, 3],
            "row_1": "full-horizon control",
            "row_2": "three physical scalar derivatives over the interior",
            "derivative_y_limits": (
                "one shared symmetric limit from every case, state, and "
                "derivative; no clipping"
            ),
        },
        "supplement_layout": {
            "shape": [4, 3],
            "rows": [
                "full-horizon control",
                "full-horizon H_u",
                "full-horizon dH_u_dt",
                "full-horizon d2H_u_dt2",
            ],
            "y_limits": (
                "one shared symmetric limit per derivative row; no clipping"
            ),
            "insets": False,
        },
    }
    if args.config is not None:
        manifest["config"] = str(args.config.resolve())
    if cases is not None:
        manifest["cases"] = [
            {
                "id": case.case_id,
                "label": case.label,
                **{
                    state_id: {
                        "path": str(case.state(state_id).path),
                        "key_prefix": case.state(state_id).key_prefix,
                        "physical_scale_factor": (
                            case.state(state_id).physical_scale_factor
                        ),
                    }
                    for state_id in STATE_IDS
                },
            }
            for case in cases
        ]
    return manifest


def main() -> None:
    args = parse_args()
    require(args.zoom_start < args.zoom_end, "interior interval is empty")
    require(
        0.0 <= args.zoom_start < args.zoom_end <= 10.0,
        "interior interval must lie inside [0,10]",
    )
    if args.layout_smoke:
        require(args.config is None, "layout smoke must not load a data config")
        cases = None
    else:
        require(args.config is not None, "--config is required in production mode")
        cases = load_cases(read_json(args.config), args.expected_points)
    main_figure = build_main_figure(
        cases,
        args.zoom_start,
        args.zoom_end,
        args.layout_smoke,
    )
    supplement_figure = build_supplement_figure(
        cases,
        args.zoom_start,
        args.zoom_end,
        args.layout_smoke,
    )
    outputs = save_outputs(
        main_figure,
        supplement_figure,
        args.out_prefix,
        args.overwrite,
        build_manifest(args, cases),
    )
    for key in ("main_pdf", "main_png", "supplement_pdf", "supplement_png"):
        print(f"Wrote {outputs[key]}")
    print(f"Wrote {outputs['manifest']}")


if __name__ == "__main__":
    main()
