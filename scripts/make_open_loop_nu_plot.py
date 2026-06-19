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
    ax.plot(u, N_sum, color="#2563EB", linewidth=3.0)
    ax.scatter([u[0]], [N_sum[0]], s=72, color="#111827", zorder=4)
    ax.scatter([u[-1]], [N_sum[-1]], s=76, color="#DC2626", marker="s", zorder=4)
    ax.annotate("initial", xy=(u[0], N_sum[0]), xytext=(-58, -8), textcoords="offset points", fontsize=11, color="#111827")
    ax.annotate("terminal", xy=(u[-1], N_sum[-1]), xytext=(-70, 10), textcoords="offset points", fontsize=11, color="#991B1B")

    u_pad = 0.08 * (float(np.nanmax(u)) - float(np.nanmin(u)))
    n_pad = 0.08 * (float(np.nanmax(N_sum)) - float(np.nanmin(N_sum)))
    ax.set_xlim(max(0.0, float(np.nanmin(u)) - u_pad), min(3.05, float(np.nanmax(u)) + u_pad))
    ax.set_ylim(max(0.0, float(np.nanmin(N_sum)) - n_pad), float(np.nanmax(N_sum)) + n_pad)
    ax.set_xlabel(r"control $u$", fontsize=14)
    ax.set_ylabel(r"total population $N=\sum_i N_i$", fontsize=14)
    ax.set_title(r"$N(u)$ phase plot", fontsize=17, pad=14)
    ax.grid(True, color="#E5E7EB", linewidth=0.8)
    fig.tight_layout()

    out_png = OUT_DIR / "nu_phase_plot_clean.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(out_png)


if __name__ == "__main__":
    main()
