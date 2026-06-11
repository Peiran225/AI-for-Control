"""Create the N(u) phase plot for the Transformer u(t) report.

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


def component_color(i: int, m: int) -> tuple[int, int, int]:
    # Blue -> teal -> green palette by state component, not by time.
    return time_color(i / max(1, m - 1))


def draw_nu_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    u: np.ndarray,
    N: np.ndarray,
) -> None:
    left, top, right, bottom = box
    plot_left = left + 130
    plot_top = top + 130
    plot_right = right - 60
    plot_bottom = bottom - 120

    xmin, xmax = 0.0, 3.05
    ymin = max(0.0, float(np.nanmin(N)) * 0.92)
    ymax = float(np.nanmax(N)) * 1.04
    if ymax <= ymin:
        ymax = ymin + 1.0

    def sx(v: float) -> float:
        return plot_left + (v - xmin) / (xmax - xmin) * (plot_right - plot_left)

    def sy(v: float) -> float:
        return plot_bottom - (v - ymin) / (ymax - ymin) * (plot_bottom - plot_top)

    title_font = font(36, bold=True)
    label_font = font(26)
    small_font = font(18)
    tick_font = font(18)

    draw.text(
        ((left + right) / 2, top + 18),
        "Transformer u(t): N(u) phase plot",
        fill=(20, 20, 20),
        anchor="ma",
        font=title_font,
    )
    draw.text(
        ((left + right) / 2, top + 52),
        "Each curve uses the same time-grid samples: horizontal u(t_k), vertical N_i(t_k), i=0,...,20.",
        fill=(80, 80, 80),
        anchor="ma",
        font=small_font,
    )

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
    draw.text(((plot_left + plot_right) / 2, bottom - 42), "control u(t_k)", fill=(20, 20, 20), anchor="ma", font=label_font)
    draw.text((plot_left, plot_top - 34), "state N_i(t_k)", fill=(20, 20, 20), anchor="la", font=label_font)

    # Draw all state components against the same control samples.
    m = N.shape[1]
    for i in range(m):
        color = component_color(i, m)
        points = [(sx(float(a)), sy(float(b))) for a, b in zip(u, N[:, i])]
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
        for x, yy in points[:: max(1, len(points) // 24)]:
            draw.ellipse([x - 2.8, yy - 2.8, x + 2.8, yy + 2.8], fill=color, outline=None)


def main() -> None:
    traj = read_trajectory(OUT_DIR / "trajectory_full.csv")
    u = traj["u"]
    n_keys = sorted((key for key in traj if key.startswith("N_")), key=lambda x: int(x.split("_")[1]))
    N = np.column_stack([traj[key] for key in n_keys])

    image = Image.new("RGB", (1600, 1200), "white")
    draw = ImageDraw.Draw(image)
    draw_nu_panel(draw, (55, 50, 1545, 1140), u, N)

    out_png = OUT_DIR / "nu_phase_plot_clean.png"
    image.save(out_png)
    print(out_png)


if __name__ == "__main__":
    main()
