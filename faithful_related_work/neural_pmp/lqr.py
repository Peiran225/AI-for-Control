"""Original LQR example from the Neural-PMP paper/repository."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn


LQR_STATE_LOW = -5.0
LQR_STATE_HIGH = 5.0
# Official repository dynamics-data sampling domain (Env.py:110--112).
LQR_OFFICIAL_SAMPLE_ACTION_LOW = -5.0
LQR_OFFICIAL_SAMPLE_ACTION_HIGH = 5.0
# Appendix-C text sensitivity domain; this is not the official-code default.
LQR_PAPER_TEXT_SAMPLE_ACTION_LOW = -100_000.0
LQR_PAPER_TEXT_SAMPLE_ACTION_HIGH = 100_000.0
# Controller feasibility bounds, distinct from dynamics-data sampling.
LQR_ACTION_LOW = -100_000.0
LQR_ACTION_HIGH = 100_000.0
LQR_HORIZON = 10
LQR_DYNAMICS_SAMPLES = 2_000
LQR_DYNAMICS_EPOCHS = 50_000
LQR_PMP_ITERATIONS = 3_000
LQR_PMP_LEARNING_RATE = 1e-3
LQR_REPRODUCTION_RUNS = 10


class OriginalLQRMap(nn.Module):
    def __init__(self, *, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.register_buffer("A", torch.eye(5, dtype=dtype))
        self.register_buffer(
            "B",
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0],
                ],
                dtype=dtype,
            ),
        )

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        return self.A @ state + self.B @ action


def original_lqr_components(*, dtype: torch.dtype = torch.float64):
    dynamics = OriginalLQRMap(dtype=dtype)
    q = torch.eye(5, dtype=dtype)
    r = torch.eye(3, dtype=dtype)
    qt = torch.diag(torch.tensor([5.0, 4.0, 2.0, 1.0, 3.0], dtype=dtype))
    initial_state = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0], dtype=dtype)

    def stage_cost(state: Tensor, action: Tensor, _index: int) -> Tensor:
        return state @ q @ state + action @ r @ action

    def terminal_cost(state: Tensor) -> Tensor:
        return state @ qt @ state

    return dynamics, initial_state, stage_cost, terminal_cost


def generate_lqr_dynamics_dataset(
    *,
    samples: int,
    seed: int,
    state_low: float = LQR_STATE_LOW,
    state_high: float = LQR_STATE_HIGH,
    action_sample_low: float = LQR_OFFICIAL_SAMPLE_ACTION_LOW,
    action_sample_high: float = LQR_OFFICIAL_SAMPLE_ACTION_HIGH,
) -> tuple[np.ndarray, np.ndarray]:
    """Offline LQR dynamics samples with explicit, separately named domains."""
    if samples <= 0 or state_low >= state_high or action_sample_low >= action_sample_high:
        raise ValueError("invalid LQR dataset domain")
    rng = np.random.default_rng(seed)
    states = rng.uniform(state_low, state_high, size=(samples, 5))
    actions = rng.uniform(action_sample_low, action_sample_high, size=(samples, 3))
    a = np.eye(5)
    b = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]],
        dtype=np.float64,
    )
    targets = states @ a.T + actions @ b.T
    inputs = np.concatenate((states, actions), axis=1)
    return inputs.astype(np.float32), targets.astype(np.float32)


def lqr_initial_control(kind: str, *, seed: int, dtype: torch.dtype = torch.float32) -> Tensor:
    if kind == "zero":
        values = np.zeros((LQR_HORIZON, 3), dtype=np.float64)
    elif kind == "random":
        rng = np.random.default_rng(seed)
        values = rng.uniform(LQR_ACTION_LOW, LQR_ACTION_HIGH, size=(LQR_HORIZON, 3))
    else:
        raise ValueError(f"unknown LQR control initialization: {kind}")
    return torch.as_tensor(values, dtype=dtype)


def riccati_control(horizon: int = 10) -> np.ndarray:
    """Closed-form finite-horizon LQR control for the original example."""
    a = np.eye(5)
    b = np.array(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]],
        dtype=np.float64,
    )
    q = np.eye(5)
    r = np.eye(3)
    p = np.diag([5.0, 4.0, 2.0, 1.0, 3.0])
    gains: list[np.ndarray] = []
    for _ in range(horizon):
        gain = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
        gains.append(gain)
        p = q + a.T @ p @ (a - b @ gain)
    gains.reverse()
    state = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
    controls = []
    for gain in gains:
        action = -gain @ state
        controls.append(action)
        state = a @ state + b @ action
    return np.asarray(controls)
