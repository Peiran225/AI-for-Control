"""Rebuild clear report figures for the Neural-PMP [3] comparison."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper_runs" / "neural_pmp_baseline_beta01"


def read_history(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def copy_to_assets(path: Path) -> None:
    asset_dir = ROOT / "reports" / "pdf_assets"
    if asset_dir.exists():
        shutil.copy2(path, asset_dir / path.name)


def selected_history(rows: list[dict[str, str]], run_id: int = 22) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = [r for r in rows if int(r["run_id"]) == run_id]
    pts.sort(key=lambda r: int(r["iter"]))
    return (
        np.asarray([int(r["iter"]) for r in pts], dtype=float),
        np.asarray([float(r["best_J"]) for r in pts]),
        np.asarray([float(r["legacy_pmp_gap"]) for r in pts]),
    )


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "legend.fontsize": 13,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
        }
    )

    sol = np.load(OUT_DIR / "best_neural_pmp_solution.npz")
    t = sol["t"]
    u = sol["u"]
    N = sol["N"]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.0, 6.6),
        dpi=180,
        sharex=True,
        gridspec_kw={"height_ratios": [0.85, 1.15], "hspace": 0.14},
    )
    ax = axes[0]
    ax.step(t, u, where="post", color="#4c78a8", lw=2.2, label=r"Neural-PMP $u(t)$")
    ax.set_ylabel(r"control $u(t)$")
    ax.set_ylim(-0.12, 3.12)
    ax.set_yticks([0, 1.5, 3.0])
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=True)
    ax.set_title(r"Neural-PMP [3] implementation: control and state trajectory")

    ax = axes[1]
    n_min = N.min(axis=1)
    n_max = N.max(axis=1)
    n_mean = N.mean(axis=1)
    ax.fill_between(t, n_min, n_max, color="#b8b8b8", alpha=0.35, label=r"range of $N_i(t)$")
    for j in range(N.shape[1]):
        ax.plot(t, N[:, j], color="#8a8a8a", alpha=0.22, lw=0.8)
    ax.plot(t, n_mean, color="#d62728", lw=2.6, label=r"mean population")
    ax.set_xlabel("time")
    ax.set_ylabel(r"state $N_i(t)$")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=True)

    out = OUT_DIR / "neural_pmp_ut_nt.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    copy_to_assets(out)

    rows = read_history(OUT_DIR / "neural_pmp_history.csv")
    iters, best_j, gap = selected_history(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.35), dpi=180)
    axes[0].plot(iters, best_j, color="#4c78a8", lw=2.6)
    axes[0].scatter([iters[np.argmin(best_j)]], [best_j.min()], color="#4c78a8", s=42, zorder=3)
    axes[0].set_title(r"Best objective $J$ during training")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel(r"best $J$")
    axes[0].grid(True, alpha=0.25)
    axes[0].text(0.98, 0.08, f"best J = {best_j.min():.2f}", ha="right", va="bottom", transform=axes[0].transAxes)

    axes[1].plot(iters, gap, color="#f58518", lw=2.6)
    axes[1].set_title("Training-time PMP/KKT diagnostic")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("PMP/KKT gap")
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", alpha=0.25)

    fig.suptitle("Neural-PMP [3] implementation: selected-run training trajectories", y=1.03, fontsize=19)
    fig.tight_layout()

    out = OUT_DIR / "neural_pmp_training_curve.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    copy_to_assets(out)

    print(OUT_DIR / "neural_pmp_ut_nt.png")
    print(OUT_DIR / "neural_pmp_training_curve.png")


if __name__ == "__main__":
    main()
