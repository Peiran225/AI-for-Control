"""Rebuild the Transformer u(t), N(t), and PMP diagnostic trajectory figure."""

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
    mean_n = traj["mean_N"]
    total_n = traj["total_N"]

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
        3,
        1,
        figsize=(10.6, 9.2),
        dpi=170,
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.15, 0.95], "hspace": 0.12},
    )

    ax = axes[0]
    ax.plot(t, u, color="#1f77b4", lw=2.7, label="Transformer control $u(t)$")
    ax.plot(t, u_sing, color="#ff7f0e", lw=2.2, ls="--", label="clipped singular candidate")
    ax.axhline(UMAX, color="black", lw=1.0, ls=":", alpha=0.8)
    ax.set_ylim(-0.1, 3.15)
    ax.set_ylabel("$u(t)$")
    ax.set_title("Transformer $u(t)$ control and population trajectory")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center", frameon=True)

    ax = axes[1]
    ax.plot(t, mean_n, color="#d62728", lw=2.5, label="mean population")
    ax.plot(t, total_n, color="#9467bd", lw=2.5, label="total population")
    ax.fill_between(t, 0.0, mean_n, color="#d62728", alpha=0.08)
    ax.set_ylabel("$N(t)$")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=True)

    ax = axes[2]
    ax.plot(t, psi, color="#2ca02c", lw=2.5, label=r"switching function $\psi(t)$")
    ax.axhline(0.0, color="black", lw=1.0, alpha=0.75)
    ax.set_ylabel(r"$\psi(t)$")
    ax.set_xlabel("time")
    ax.grid(True, alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(t, q, color="#8c564b", lw=2.0, label="singular weight $q(t)$")
    ax2.set_ylabel("$q(t)$")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="lower left", frameon=True)

    out = OUT_DIR / "ut_nt_trajectory_clean.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    pdf_asset_dir = ROOT / "reports" / "pdf_assets"
    if pdf_asset_dir.exists():
        shutil.copy2(out, pdf_asset_dir / out.name)
    print(out)


if __name__ == "__main__":
    main()
