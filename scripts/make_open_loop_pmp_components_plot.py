"""Rebuild the final pointwise PMP/KKT diagnostics figure for the report."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper_runs" / "open_loop_ut_report"
UMAX = 3.0


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    data: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            data.setdefault(key, []).append(float(value))
    return {key: np.asarray(values, dtype=float) for key, values in data.items()}


def main() -> None:
    traj = read_csv(OUT_DIR / "trajectory_full.csv")
    t = traj["t"]
    u = traj["u"]
    u_sing = traj["u_sing"]
    psi = traj["psi"]
    q = traj["q"]

    singular_error = q * (u - u_sing) ** 2
    boundary_kkt_error = (1.0 - q) * (np.maximum(psi, 0.0) * u + np.maximum(-psi, 0.0) * (UMAX - u)) ** 2

    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "legend.fontsize": 12.5,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.0, 7.56),
        dpi=160,
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.12},
    )

    ax = axes[0]
    eps = 1e-8
    ax.plot(t, np.maximum(singular_error, eps), color="#e45756", lw=2.3, label="singular-condition error")
    ax.plot(t, np.maximum(boundary_kkt_error, eps), color="#72b7b2", lw=2.3, label="boundary KKT error")
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 1.8)
    ax.set_ylabel("pointwise loss")
    ax.set_title("Final pointwise PMP/KKT loss components")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left", frameon=True)

    ax2 = axes[1]
    ax2.plot(t, u, color="#1f77b4", lw=2.6, label="control $u(t)$")
    ax2.plot(t, psi, color="#2ca02c", lw=2.3, label=r"switching function $\psi(t)=H_u$")
    ax2.axhline(0.0, color="black", lw=1.0, alpha=0.75)
    ax2.set_ylim(-3.3, 3.1)
    ax2.set_xlabel("time")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="lower left", frameon=True)

    out = OUT_DIR / "pmp_condition_components_clean.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    pdf_asset_dir = ROOT / "reports" / "pdf_assets"
    if pdf_asset_dir.exists():
        shutil.copy2(out, pdf_asset_dir / out.name)
    print(out)


if __name__ == "__main__":
    main()
