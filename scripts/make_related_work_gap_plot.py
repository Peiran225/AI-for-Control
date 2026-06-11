"""Rebuild the related-work objective-gap close-up figure for the report."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper_runs" / "neural_pmp_baseline_beta01"


def main() -> None:
    # These are the rounded values used in the report table. The plot is only a
    # close-up visual of the same comparison, not a separate evaluation source.
    values = {
        "direct\ncost ref.": 386.47,
        "Transformer": 386.70,
        "Neural-PMP\nctrl-stage [3]": 387.02,
    }
    base = values["direct\ncost ref."]
    gaps = {name: value - base for name, value in values.items()}

    colors = ["#4c78a8", "#72b7b2", "#f58518"]
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
        }
    )
    fig, ax = plt.subplots(figsize=(6.8, 3.8), dpi=180)
    names = list(gaps)
    ys = [gaps[name] for name in names]
    bars = ax.bar(names, ys, color=colors, width=0.62)
    ax.axhline(0, color="#333333", lw=1)
    ax.set_ylabel(r"$\Delta J$")
    ax.set_title("Objective J gap relative to direct-cost reference")
    ax.grid(axis="y", alpha=0.22)
    ax.set_ylim(0, max(ys) * 1.35)
    for bar, y in zip(bars, ys):
        ax.text(bar.get_x() + bar.get_width() / 2, y + 0.015, f"{y:.3f}", ha="center", va="bottom", fontsize=13)
    fig.tight_layout()

    out = OUT_DIR / "neural_pmp_reference_gap_closeup.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    pdf_asset_dir = ROOT / "reports" / "pdf_assets"
    if pdf_asset_dir.exists():
        shutil.copy2(out, pdf_asset_dir / out.name)
    print(out)


if __name__ == "__main__":
    main()
