"""Create the N(u) phase plot for the Transformer u(t) report."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


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


def main() -> None:
    traj = read_trajectory(OUT_DIR / "trajectory_full.csv")
    u = traj["u"]
    n_keys = sorted((key for key in traj if key.startswith("N_")), key=lambda x: int(x.split("_")[1]))
    states = np.column_stack([traj[key] for key in n_keys])
    m = states.shape[1]

    fig, ax = plt.subplots(figsize=(8.8, 6.2), dpi=240)
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=0, vmax=m - 1)

    segments = []
    colors = []
    for i in range(m):
        points = np.column_stack([u, states[:, i]])
        segments.extend(np.stack([points[:-1], points[1:]], axis=1))
        colors.extend([i] * (len(points) - 1))

    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=0.9, alpha=0.95)
    lc.set_array(np.asarray(colors, dtype=float))
    ax.add_collection(lc)

    for i in range(m):
        ax.scatter(u[::8], states[::8, i], s=5, color=cmap(norm(i)), linewidths=0, alpha=0.95)

    ax.set_xlim(0.0, 3.05)
    ax.set_ylim(0.0, float(np.nanmax(states)) * 1.06)
    ax.set_xlabel(r"control $u(t)$", fontsize=14)
    ax.set_ylabel(r"state $N_i(t)$", fontsize=14)
    ax.set_title(r"Transformer $u(t)$: $N(u)$ phase plot", fontsize=16, pad=16)
    ax.text(
        0.5,
        1.01,
        r"Each curve plots one state component against the applied control; color denotes the state index $i$, not time.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#4B5563",
    )
    ax.grid(True, color="#E5E7EB", linewidth=0.8)

    cbar = fig.colorbar(lc, ax=ax, pad=0.025, shrink=0.62)
    cbar.set_label(r"state index $i$", fontsize=11)
    cbar.set_ticks([0, m // 2, m - 1])
    cbar.set_ticklabels([r"$N_0$", rf"$N_{{{m//2}}}$", rf"$N_{{{m-1}}}$"])

    fig.tight_layout()
    out_png = OUT_DIR / "nu_phase_plot_clean.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(out_png)


if __name__ == "__main__":
    main()
