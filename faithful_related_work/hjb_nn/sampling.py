"""Algorithm 4.1 sampling, split, and convergence-test utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import subprocess

import numpy as np


Array = np.ndarray


def git_checkout_provenance(path: Path) -> dict[str, object]:
    """Record base revision plus tracked/untracked worktree differences."""
    path = Path(path)

    def run(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(path), *arguments],
            text=True,
            stderr=subprocess.STDOUT,
        )

    revision = run("rev-parse", "HEAD").strip()
    status_lines = [line for line in run("status", "--short").splitlines() if line]
    name_status = [line for line in run("diff", "--name-status").splitlines() if line]
    numstat = [line for line in run("diff", "--numstat").splitlines() if line]
    patch = run("diff", "--no-ext-diff", "--binary")
    untracked = [line[3:] for line in status_lines if line.startswith("?? ")]
    return {
        "base_revision": revision,
        "worktree_clean": not status_lines,
        "tracked_dirty": bool(name_status),
        "untracked_present": bool(untracked),
        "status_porcelain": status_lines,
        "tracked_diff_name_status": name_status,
        "tracked_diff_numstat": numstat,
        "tracked_diff_sha256": hashlib.sha256(patch.encode()).hexdigest(),
        "tracked_diff_bytes": len(patch.encode()),
        "untracked_paths": untracked,
    }


def refuse_nonempty_output(path: Path) -> Path:
    """Create an output root, refusing any directory with prior artifacts."""
    path = Path(path)
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"output path exists and is not a directory: {path}")
        if any(path.iterdir()):
            raise FileExistsError(
                f"refusing non-empty output directory to prevent artifact mixing: {path}"
            )
    else:
        path.mkdir(parents=True)
    return path


def _state_keys(states: Array, decimals: int = 14) -> set[bytes]:
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2:
        raise ValueError("initial states must have shape (state_dim, trajectories)")
    rounded = np.round(states.T, decimals=decimals)
    return {np.ascontiguousarray(row).tobytes() for row in rounded}


def validate_trajectory_disjoint(train_initial: Array, validation_initial: Array) -> None:
    """Reject any initial trajectory shared by train and validation sets."""
    overlap = _state_keys(train_initial) & _state_keys(validation_initial)
    if overlap:
        raise ValueError(f"train/validation trajectory overlap: {len(overlap)} initial states")


def split_initial_state_box(
    lower: Array,
    upper: Array,
    train_count: int,
    validation_count: int,
    *,
    seed: int,
) -> tuple[Array, Array]:
    """Independently sample trajectory-level train/validation initial states."""
    lower = np.asarray(lower, dtype=np.float64).reshape(-1, 1)
    upper = np.asarray(upper, dtype=np.float64).reshape(-1, 1)
    if lower.shape != upper.shape or np.any(lower >= upper):
        raise ValueError("invalid initial-state box")
    if train_count <= 0 or validation_count <= 0:
        raise ValueError("both split sizes must be positive")
    train_rng = np.random.default_rng(seed)
    validation_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    train = train_rng.uniform(lower, upper, size=(lower.shape[0], train_count))
    validation = validation_rng.uniform(lower, upper, size=(lower.shape[0], validation_count))
    validate_trajectory_disjoint(train, validation)
    return train, validation


def select_largest_gradient_candidates(
    candidate_states: Array,
    predicted_gradients: Array,
    count: int = 1,
) -> tuple[Array, Array, Array]:
    """Select Algorithm 4.1 candidates with largest predicted ||V_x||_2."""
    candidate_states = np.asarray(candidate_states, dtype=np.float64)
    predicted_gradients = np.asarray(predicted_gradients, dtype=np.float64)
    if candidate_states.shape != predicted_gradients.shape or candidate_states.ndim != 2:
        raise ValueError("candidate states and gradients must share (state_dim, candidates)")
    if count <= 0 or count > candidate_states.shape[1]:
        raise ValueError("invalid number of selected candidates")
    norms = np.linalg.norm(predicted_gradients, axis=0)
    indices = np.argsort(norms, kind="stable")[-count:][::-1]
    return candidate_states[:, indices], indices, norms


@dataclass(frozen=True)
class ConvergenceDecision:
    converged: bool
    sample_gradient_l1: float
    gradient_standard_deviation_norm: float
    current_size: int
    next_size: int


def convergence_test_and_next_size(
    sample_gradient: Array,
    per_sample_gradients: Array,
    *,
    tolerance: float,
    current_size: int,
    growth_limit: float = 2.0,
) -> ConvergenceDecision:
    """Implement paper equations (4.8)--(4.9).

    ``per_sample_gradients`` contains one loss-gradient vector per sampled data
    point.  The numerator is the square root of the sum of component-wise
    sample variances; the denominator uses the L1 norm, matching the paper and
    released implementation.
    """
    sample_gradient = np.asarray(sample_gradient, dtype=np.float64).reshape(-1)
    per_sample_gradients = np.asarray(per_sample_gradients, dtype=np.float64)
    if per_sample_gradients.ndim != 2 or per_sample_gradients.shape[0] < 2:
        raise ValueError("at least two per-sample gradients are required")
    if per_sample_gradients.shape[1] != sample_gradient.size:
        raise ValueError("gradient dimensions do not match")
    if tolerance <= 0.0 or current_size <= 0 or growth_limit <= 1.0:
        raise ValueError("invalid convergence-test parameters")

    sample_count = per_sample_gradients.shape[0]
    sample_gradient_l1 = float(np.linalg.norm(sample_gradient, ord=1))
    component_variance = np.var(per_sample_gradients, axis=0, ddof=1, dtype=np.float64)
    standard_deviation_norm = float(np.sqrt(np.sum(component_variance)))
    threshold = tolerance * sample_gradient_l1 * np.sqrt(sample_count)
    converged = standard_deviation_norm <= threshold

    if converged:
        next_size = current_size
    elif sample_gradient_l1 == 0.0:
        next_size = int(np.ceil(growth_limit * current_size))
    else:
        estimated = int(np.ceil((standard_deviation_norm / (tolerance * sample_gradient_l1)) ** 2))
        if current_size >= estimated:
            estimated = int(np.ceil(current_size * estimated / sample_count))
        next_size = max(current_size + 1, min(int(np.ceil(growth_limit * current_size)), estimated))

    return ConvergenceDecision(
        converged=converged,
        sample_gradient_l1=sample_gradient_l1,
        gradient_standard_deviation_norm=standard_deviation_norm,
        current_size=current_size,
        next_size=next_size,
    )


def extract_initial_states(time: Array, states: Array, *, atol: float = 1.0e-12) -> Array:
    """Recover trajectory starts from author-format concatenated columns."""
    time = np.asarray(time, dtype=np.float64).reshape(-1)
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != time.size:
        raise ValueError("time and state columns do not align")
    start_indices = np.flatnonzero(np.isclose(time, 0.0, rtol=0.0, atol=atol))
    if start_indices.size == 0:
        raise ValueError("no t=0 trajectory columns found")
    return states[:, start_indices]
