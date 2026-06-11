"""Create the N(u) phase plot for the open-loop u(t) report.

This uses Pillow instead of matplotlib so it runs in the lightweight local
runtime used by Codex.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


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


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def blend(c1: tuple[int, int, int], c2: tuple[int, int, int], a: float) -> tuple[int, int, int]:
    return tuple(int(round((1 - a) * x + a * y)) for x, y in zip(c1, c2))


def time_color(z: float) -> tuple[int, int, int]:
    # A compact viridis-like gradient: dark blue -> teal -> green -> yellow.
    stops = [
        (0.00, (68, 1, 84)),
        (0.35, (49, 104, 142)),
        (0.70, (53, 183, 121)),
        (1.00, (253, 231, 37)),
    ]
    z = min(1.0, max(0.0, z))
    for (z0, c0), (z1, c1) in zip(stops[:-1], stops[1:]):
        if z <= z1:
            return blend(c0, c1, (z - z0) / (z1 - z0))
    return stops[-1][1]


def draw_dashed_vertical(draw: ImageDraw.ImageDraw, x: float, y0: float, y1: float, color: tuple[int, int, int]) -> None:
    dash = 10
    gap = 7
    y = y0
    while y < y1:
        draw.line([(x, y), (x, min(y + dash, y1))], fill=color, width=3)
        y += dash + gap


def draw_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    u: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    title: str,
    ylabel: str,
    u_mean: float,
    u_sing_mean: float,
) -> None:
    left, top, right, bottom = box
    plot_left = left + 82
    plot_top = top + 48
    plot_right = right - 28
    plot_bottom = bottom - 72

    xmin, xmax = 0.0, 3.05
    ymin = max(0.0, float(np.nanmin(y)) * 0.92)
    ymax = float(np.nanmax(y)) * 1.04
    if ymax <= ymin:
        ymax = ymin + 1.0

    def sx(v: float) -> float:
        return plot_left + (v - xmin) / (xmax - xmin) * (plot_right - plot_left)

    def sy(v: float) -> float:
        return plot_bottom - (v - ymin) / (ymax - ymin) * (plot_bottom - plot_top)

    title_font = font(24, bold=True)
    label_font = font(18)
    small_font = font(15)
    tick_font = font(14)

    draw.text(((left + right) / 2, top + 10), title, fill=(20, 20, 20), anchor="ma", font=title_font)

    # Grid and ticks.
    for tick in np.linspace(0, 3, 7):
        x = sx(float(tick))
        draw.line([(x, plot_top), (x, plot_bottom)], fill=(225, 225, 225), width=1)
        draw.line([(x, plot_bottom), (x, plot_bottom + 6)], fill=(40, 40, 40), width=2)
        draw.text((x, plot_bottom + 10), f"{tick:.1f}", fill=(40, 40, 40), anchor="ma", font=tick_font)

    y_ticks = np.linspace(ymin, ymax, 5)
    for tick in y_ticks:
        yy = sy(float(tick))
        draw.line([(plot_left, yy), (plot_right, yy)], fill=(225, 225, 225), width=1)
        draw.line([(plot_left - 6, yy), (plot_left, yy)], fill=(40, 40, 40), width=2)
        draw.text((plot_left - 10, yy), f"{tick:.1f}", fill=(40, 40, 40), anchor="rm", font=tick_font)

    # Axes.
    draw.rectangle([plot_left, plot_top, plot_right, plot_bottom], outline=(25, 25, 25), width=2)
    draw.text(((plot_left + plot_right) / 2, bottom - 30), "control u(t)", fill=(20, 20, 20), anchor="ma", font=label_font)
    draw.text((left + 12, top + 56), ylabel, fill=(20, 20, 20), anchor="la", font=small_font)

    # Reference verticals.
    draw_dashed_vertical(draw, sx(u_mean), plot_top, plot_bottom, (205, 48, 48))
    draw_dashed_vertical(draw, sx(u_sing_mean), plot_top, plot_bottom, (45, 150, 80))

    # Trajectory line.
    points = [(sx(float(a)), sy(float(b))) for a, b in zip(u, y)]
    if len(points) > 1:
        draw.line(points, fill=(70, 70, 70), width=2)

    # Time-colored points.
    tmin, tmax = float(np.nanmin(t)), float(np.nanmax(t))
    denom = max(tmax - tmin, 1e-12)
    for a, b, tt in zip(u, y, t):
        color = time_color((float(tt) - tmin) / denom)
        x, yy = sx(float(a)), sy(float(b))
        draw.ellipse([x - 4, yy - 4, x + 4, yy + 4], fill=color, outline=None)

    # Legend.
    lx, ly = plot_right - 210, plot_top + 16
    draw.rectangle([lx - 12, ly - 10, lx + 198, ly + 48], fill=(255, 255, 255), outline=(210, 210, 210), width=1)
    draw_dashed_vertical(draw, lx + 10, ly - 2, ly + 16, (205, 48, 48))
    draw.text((lx + 26, ly - 2), "mean u", fill=(40, 40, 40), font=small_font)
    draw_dashed_vertical(draw, lx + 10, ly + 24, ly + 42, (45, 150, 80))
    draw.text((lx + 26, ly + 24), "mean u_sing", fill=(40, 40, 40), font=small_font)


def draw_colorbar(draw: ImageDraw.ImageDraw, x: int, y0: int, y1: int, tmin: float, tmax: float) -> None:
    for i, y in enumerate(range(y0, y1)):
        z = i / max(1, y1 - y0 - 1)
        draw.line([(x, y), (x + 18, y)], fill=time_color(z), width=1)
    draw.rectangle([x, y0, x + 18, y1], outline=(40, 40, 40), width=1)
    draw.text((x + 28, y0), f"t={tmin:.0f}", fill=(30, 30, 30), anchor="lm", font=font(14))
    draw.text((x + 28, y1), f"t={tmax:.0f}", fill=(30, 30, 30), anchor="lm", font=font(14))
    draw.text((x - 4, (y0 + y1) / 2), "time", fill=(30, 30, 30), anchor="rm", font=font(15))


def main() -> None:
    traj = read_trajectory(OUT_DIR / "trajectory_full.csv")
    t = traj["t"]
    u = traj["u"]
    mean_n = traj["mean_N"]
    total_n = traj["total_N"]
    u_sing = traj["u_sing"]

    image = Image.new("RGB", (1800, 720), "white")
    draw = ImageDraw.Draw(image)
    draw.text((900, 28), "Open-loop Transformer u(t): N(u) phase plots", fill=(10, 10, 10), anchor="ma", font=font(34, bold=True))

    draw_panel(
        draw,
        (35, 76, 870, 690),
        u,
        mean_n,
        t,
        "Mean population vs control",
        "mean N(t)",
        float(np.nanmean(u)),
        float(np.nanmean(u_sing)),
    )
    draw_panel(
        draw,
        (895, 76, 1730, 690),
        u,
        total_n,
        t,
        "Total population vs control",
        "sum_i N_i(t)",
        float(np.nanmean(u)),
        float(np.nanmean(u_sing)),
    )
    draw_colorbar(draw, 1748, 150, 610, float(np.nanmin(t)), float(np.nanmax(t)))

    out_png = OUT_DIR / "nu_phase_plot_clean.png"
    image.save(out_png)
    print(out_png)


if __name__ == "__main__":
    main()
