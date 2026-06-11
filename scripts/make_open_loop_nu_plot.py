"""Create the scalar N(u) phase plot for the Transformer u(t) report."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper_runs" / "open_loop_ut_report"


def read_trajectory(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    data: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            data.setdefault(key, []).append(float(value))
    return {key: np.asarray(values, dtype=float) for key, values in data.items()}


def total_population(traj: dict[str, np.ndarray]) -> np.ndarray:
    if "total_N" in traj:
        return traj["total_N"]
    n_keys = sorted((key for key in traj if key.startswith("N_")), key=lambda x: int(x.split("_")[1]))
    return np.column_stack([traj[key] for key in n_keys]).sum(axis=1)


def main() -> None:
    traj = read_trajectory(OUT_DIR / "trajectory_full.csv")
    u = traj["u"]
    N_sum = total_population(traj)

    fig, ax = plt.subplots(figsize=(8.4, 5.8), dpi=260)
    ax.plot(u, N_sum, color="#2563EB", linewidth=2.6)
    ax.scatter(u, N_sum, s=10, color="#2563EB", alpha=0.7, linewidths=0)

    ax.set_xlim(0.0, 3.05)
    ax.set_ylim(0.0, float(np.nanmax(N_sum)) * 1.06)
    ax.set_xlabel(r"control $u$", fontsize=14)
    ax.set_ylabel(r"total population $N=\sum_i N_i$", fontsize=14)
    ax.set_title(r"$N(u)$ phase plot", fontsize=17, pad=14)
    ax.text(
        0.5,
        1.01,
        r"Here $N$ is the sum of all 21 subpopulations.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#4B5563",
    )
    ax.grid(True, color="#E5E7EB", linewidth=0.8)
    fig.tight_layout()

    out_png = OUT_DIR / "nu_phase_plot_clean.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(out_png)


if __name__ == "__main__":
    main()
