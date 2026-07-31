"""Tumor one-step environment for the faithful Neural-PMP reproduction."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TumorConfig:
    final_time: float = 10.0
    intervals: int = 200
    state_dim: int = 21
    action_lower: float = 0.0
    action_upper: float = 3.0
    beta: float = 0.1
    alpha: float = 1.0
    gamma: float = 20.0
    initial_state: float = 10.0
    suppression: float = 0.5

    @property
    def dt(self) -> float:
        return self.final_time / self.intervals

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def tumor_vectors(config: TumorConfig, *, dtype: np.dtype = np.float64) -> dict[str, np.ndarray]:
    grid = np.linspace(0.0, 1.0, config.state_dim, dtype=dtype)
    return {
        "r": np.asarray(2.0 / (1.0 + 3.0 * grid**4), dtype=dtype),
        "phi": np.asarray(1.0 / (1.0 + grid**2), dtype=dtype),
        "M": np.full(config.state_dim, config.suppression, dtype=dtype),
        "beta": np.full(config.state_dim, config.beta, dtype=dtype),
        "alpha": np.full(config.state_dim, config.alpha, dtype=dtype),
        "initial": np.full(config.state_dim, config.initial_state, dtype=dtype),
    }


def true_tumor_step_numpy(state: np.ndarray, action: np.ndarray | float, config: TumorConfig) -> np.ndarray:
    vectors = tumor_vectors(config, dtype=np.float64)
    x = np.asarray(state, dtype=np.float64).reshape(config.state_dim)
    u = float(np.asarray(action).reshape(-1)[0])
    growth_suppression = np.log1p(x.mean())
    rhs = (vectors["r"] - vectors["phi"] * u - vectors["M"] * growth_suppression) * x
    return x + config.dt * rhs


class ExactTumorEulerMap(nn.Module):
    """Known differentiable one-step map, useful for oracle and gradient tests."""

    def __init__(self, config: TumorConfig, *, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.config = config
        vectors = tumor_vectors(config)
        self.register_buffer("r", torch.as_tensor(vectors["r"], dtype=dtype))
        self.register_buffer("phi", torch.as_tensor(vectors["phi"], dtype=dtype))
        self.register_buffer("suppression", torch.as_tensor(vectors["M"], dtype=dtype))

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        u = action.reshape(-1)[0]
        g = torch.log1p(state.mean())
        rhs = (self.r - self.phi * u - self.suppression * g) * state
        return state + self.config.dt * rhs


def make_costs(config: TumorConfig, *, dtype: torch.dtype) -> tuple:
    vectors = tumor_vectors(config)
    beta = torch.as_tensor(vectors["beta"], dtype=dtype)
    alpha = torch.as_tensor(vectors["alpha"], dtype=dtype)

    def stage_cost(state: Tensor, action: Tensor, _index: int) -> Tensor:
        return config.dt * (torch.dot(beta.to(state.device), state) + config.gamma * action.reshape(-1)[0])

    def terminal_cost(state: Tensor) -> Tensor:
        return torch.dot(alpha.to(state.device), state)

    return stage_cost, terminal_cost


def generate_dynamics_dataset(
    config: TumorConfig,
    *,
    samples: int,
    seed: int,
    state_low: float = 0.1,
    state_high: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a deterministic offline (state, action, next-state) dataset."""
    if samples <= 0 or state_low <= 0.0 or state_low >= state_high:
        raise ValueError("invalid dynamics-dataset configuration")
    rng = np.random.default_rng(seed)
    states = rng.uniform(state_low, state_high, size=(samples, config.state_dim))
    actions = rng.uniform(config.action_lower, config.action_upper, size=(samples, 1))
    targets = np.stack([true_tumor_step_numpy(states[i], actions[i], config) for i in range(samples)])
    return np.concatenate((states, actions), axis=1).astype(np.float32), targets.astype(np.float32)


def validation_initial_states(
    config: TumorConfig,
    *,
    count: int,
    seed: int,
    relative_std: float = 0.02,
    dtype: torch.dtype = torch.float32,
) -> list[Tensor]:
    """Predeclared perturbations used only for learned-model control selection."""
    rng = np.random.default_rng(seed)
    base = np.full(config.state_dim, config.initial_state, dtype=np.float64)
    states = [base]
    for _ in range(max(0, count - 1)):
        perturbation = rng.normal(0.0, relative_std, size=config.state_dim)
        states.append(np.maximum(base * (1.0 + perturbation), 1e-8))
    return [torch.as_tensor(state, dtype=dtype) for state in states]


def initial_control(config: TumorConfig, kind: str, *, seed: int, dtype: torch.dtype) -> Tensor:
    n = config.intervals
    if kind == "zero":
        values = np.zeros(n)
    elif kind == "mid":
        values = np.full(n, 0.5 * config.action_upper)
    elif kind == "max":
        values = np.full(n, config.action_upper)
    elif kind == "front":
        values = np.linspace(config.action_upper, config.action_lower, n)
    elif kind == "back":
        values = np.linspace(config.action_lower, config.action_upper, n)
    elif kind == "random":
        rng = np.random.default_rng(seed)
        values = rng.uniform(config.action_lower, config.action_upper, size=n)
    else:
        raise ValueError(f"unknown control initialization: {kind}")
    return torch.as_tensor(values[:, None], dtype=dtype)


def true_discrete_objective(config: TumorConfig, control: np.ndarray) -> float:
    vectors = tumor_vectors(config)
    state = vectors["initial"].copy()
    value = 0.0
    flat_control = np.asarray(control, dtype=np.float64).reshape(config.intervals)
    for action in flat_control:
        value += config.dt * (float(vectors["beta"] @ state) + config.gamma * action)
        state = true_tumor_step_numpy(state, action, config)
    return float(value + vectors["alpha"] @ state)
