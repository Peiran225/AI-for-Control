"""Resolution-aware measurements for early and late control transitions."""

from __future__ import annotations

import numpy as np


def plateau_median(
    t: np.ndarray,
    u: np.ndarray,
    left: float,
    right: float,
) -> float:
    interval_midpoints = 0.5 * (t[:-1] + t[1:])
    values = u[:-1][
        (interval_midpoints >= left - 1.0e-12)
        & (interval_midpoints < right - 1.0e-12)
    ]
    if values.size < 2:
        raise ValueError("plateau window contains fewer than two intervals")
    return float(np.median(values))


def progress_crossing(
    t: np.ndarray,
    progress: np.ndarray,
    level: float,
    left: float,
    right: float,
) -> float:
    candidates: list[float] = []
    for index in range(t.size - 1):
        if t[index + 1] < left or t[index] > right:
            continue
        p0 = progress[index]
        p1 = progress[index + 1]
        if p0 <= level <= p1 and p1 > p0:
            candidates.append(
                float(
                    t[index]
                    + (level - p0)
                    / (p1 - p0)
                    * (t[index + 1] - t[index])
                )
            )
    if not candidates:
        raise ValueError(f"no directed crossing found for progress={level:g}")
    center = 0.5 * (left + right)
    return min(candidates, key=lambda value: abs(value - center))


def switch_metrics(
    t: np.ndarray,
    u: np.ndarray,
    side: str,
) -> dict[str, float | int]:
    """Measure the 10--90% width and largest node jump of one transition."""

    t = np.asarray(t, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    if t.ndim != 1 or u.ndim != 1 or t.shape != u.shape or t.size < 3:
        raise ValueError("t and u must be matching one-dimensional node arrays")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("t must be strictly increasing")

    differences = np.diff(u)
    if side == "early":
        candidates = np.flatnonzero(t[:-1] < 2.0)
        jump_index = int(candidates[np.argmin(differences[candidates])])
    elif side == "late":
        candidates = np.flatnonzero(t[:-1] >= 7.0)
        jump_index = int(candidates[np.argmax(differences[candidates])])
    else:
        raise ValueError(f"unknown switch side {side!r}")

    junction = float(t[jump_index + 1])
    pre = plateau_median(t, u, max(0.0, junction - 0.4), junction - 0.2)
    post = plateau_median(t, u, junction + 0.2, min(10.0, junction + 0.4))
    amplitude = abs(post - pre)
    if amplitude <= np.finfo(np.float64).eps:
        raise ValueError("local transition amplitude is numerically zero")
    progress = (u - pre) / (post - pre)
    search_left, search_right = junction - 0.35, junction + 0.35
    t10 = progress_crossing(t, progress, 0.1, search_left, search_right)
    t50 = progress_crossing(t, progress, 0.5, search_left, search_right)
    t90 = progress_crossing(t, progress, 0.9, search_left, search_right)
    local_indices = np.flatnonzero(
        (t >= search_left - 1.0e-12) & (t <= search_right + 1.0e-12)
    )
    node10 = next(int(index) for index in local_indices if progress[index] >= 0.1)
    node90 = next(
        int(index)
        for index in local_indices
        if index >= node10 and progress[index] >= 0.9
    )
    intermediate = np.count_nonzero(
        (t >= search_left - 1.0e-12)
        & (t <= search_right + 1.0e-12)
        & (progress > 0.1)
        & (progress < 0.9)
    )
    dt = float(np.median(np.diff(t)))
    jump = float(abs(differences[jump_index]))
    return {
        "junction": junction,
        "pre_plateau_median": pre,
        "post_plateau_median": post,
        "local_amplitude": amplitude,
        "t10_linear_node_interpolation": t10,
        "t50_linear_node_interpolation": t50,
        "t90_linear_node_interpolation": t90,
        "width_10_90_linear": t90 - t10,
        "width_10_90_in_grid_steps": (t90 - t10) / dt,
        "zoh_threshold_node_span": float(t[node90] - t[node10]),
        "zoh_threshold_node_span_in_grid_steps": float(
            (t[node90] - t[node10]) / dt
        ),
        "intermediate_control_node_count": int(intermediate),
        "largest_signed_jump_at_junction": float(differences[jump_index]),
        "largest_abs_jump_at_junction": jump,
        "largest_jump_fraction_of_local_amplitude": jump / amplitude,
        "largest_abs_slope_between_nodes": jump / dt,
    }
