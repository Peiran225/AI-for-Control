"""Rebuild the open-loop u(t) training-loss figure used in the report."""

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


def read_history(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    data: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            data.setdefault(key, []).append(float(value))
    return {key: np.asarray(values, dtype=float) for key, values in data.items()}


def moving_average(y: np.ndarray, window: int = 25) -> np.ndarray:
    if window <= 1 or len(y) < window:
        return y
    kernel = np.ones(window, dtype=float) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(y, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def main() -> None:
    hist = read_history(OUT_DIR / "history_components.csv")
    epoch = hist["epoch"]

    total = hist["opt_gap"]
    singular = hist["singular_component"]
    nonsingular = hist["nonsingular_component"]
    objective = hist["objective"]
    psi_mean_abs = hist["psi_mean_abs"]
    q_mean = hist["q_mean"]

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
        figsize=(11.0, 7.975),
        dpi=160,
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.12},
    )

    ax = axes[0]
    eps = 1e-4
    ax.plot(epoch, np.maximum(total, eps), color="#1f77b4", alpha=0.22, lw=0.9)
    ax.plot(epoch, np.maximum(singular, eps), color="#2ca02c", alpha=0.20, lw=0.9)
    ax.plot(epoch, np.maximum(nonsingular, eps), color="#d62728", alpha=0.22, lw=0.9)
    ax.plot(epoch, np.maximum(moving_average(total), eps), color="#1f77b4", lw=2.4, label="PMP/KKT gap")
    ax.plot(epoch, np.maximum(moving_average(singular), eps), color="#2ca02c", lw=2.4, label="singular component")
    ax.plot(epoch, np.maximum(moving_average(nonsingular), eps), color="#d62728", lw=2.4, label="non-singular KKT component")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.6e2)
    ax.set_ylabel("loss")
    ax.set_title("Open-loop Transformer u(t): training loss trajectories")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", frameon=True)

    ax2 = axes[1]
    ax2.plot(epoch, objective, color="#4c78a8", lw=2.3, label="discrete objective J")
    ax2.set_ylabel("J")
    ax2.set_ylim(min(objective) - 0.8, max(objective) + 0.8)
    ax2.grid(True, alpha=0.25)
    ax2.set_xlabel("epoch")

    ax_diag = ax2.twinx()
    ax_diag.plot(epoch, moving_average(psi_mean_abs), color="#f58518", lw=2.3, label=r"mean $|\psi(t)|$ (switching residual)")
    ax_diag.plot(epoch, moving_average(q_mean), color="#54a24b", lw=2.3, label=r"mean $q(t)$ (singular weight)")
    ax_diag.set_ylabel("diagnostics")
    ax_diag.set_ylim(-0.15, max(4.7, float(np.nanmax(psi_mean_abs))) + 0.35)

    handles, labels = ax2.get_legend_handles_labels()
    handles2, labels2 = ax_diag.get_legend_handles_labels()
    ax2.legend(handles + handles2, labels + labels2, loc="upper right", frameon=True)

    out = OUT_DIR / "training_loss_trajectory_clean.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    pdf_asset_dir = ROOT / "reports" / "pdf_assets"
    if pdf_asset_dir.exists():
        shutil.copy2(out, pdf_asset_dir / out.name)
    print(out)


if __name__ == "__main__":
    main()
