"""Core PI-DeepONet operators implementing paper Eq. (2.3)--(2.5)."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn

from .problems import ArgminResult, ControlProblem


@dataclass(frozen=True)
class TrainConfig:
    h: float
    viscosity_N: float
    outer_iterations: int = 3
    steps_per_outer: int = 100
    batch_size: int = 32
    terminal_batch_size: int = 32
    sensors: int = 64
    width: int = 64
    latent_dim: int = 64
    branch_depth: int = 2
    trunk_depth: int = 3
    learning_rate: float = 1.0e-3
    pde_weight: float = 1.0
    terminal_weight: float = 1.0
    weight_decay: float = 0.0
    # Algorithm 1 specifies Adam on the stated physics-informed loss and does
    # not specify gradient clipping.  Keep clipping disabled by default; a
    # non-None value is an explicitly labelled stabilization ablation.
    gradient_clip: float | None = None
    log_every: int = 10
    value_scale: float = 1.0
    branch_scale: float = 1.0
    tie_tolerance: float = 0.0
    dtype: str = "float64"
    device: str = "cpu"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def required_viscosity_constant(dynamics_sup_bound: float) -> float:
    """Theorem-1 lower bound ``N >= max(1, ||f||_inf / 2)``."""

    if not math.isfinite(dynamics_sup_bound) or dynamics_sup_bound < 0.0:
        raise ValueError("dynamics_sup_bound must be finite and nonnegative")
    return max(1.0, 0.5 * dynamics_sup_bound)


def validate_viscosity_constant(viscosity_N: float, dynamics_sup_bound: float) -> float:
    required = required_viscosity_constant(dynamics_sup_bound)
    if viscosity_N + 1.0e-14 < required:
        raise ValueError(
            "artificial-viscosity monotonicity condition violated: "
            f"N={viscosity_N:.12g} < max(1, ||f||_inf/2)={required:.12g} "
            f"for declared ||f||_inf bound {dynamics_sup_bound:.12g}"
        )
    return required


def _mlp(input_dim: int, output_dim: int, width: int, depth: int) -> nn.Sequential:
    if depth < 1:
        raise ValueError("network depth must be at least one")
    layers: list[nn.Module] = []
    current = input_dim
    for _ in range(depth):
        layers.extend([nn.Linear(current, width), nn.Tanh()])
        current = width
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class DeepONet(nn.Module):
    """Branch/trunk operator network from paper Eq. (2.6).

    Equation (2.6) is the unnormalized branch/trunk inner product.  The
    displayed paper architecture has neither a ``1/sqrt(p)`` factor nor a
    separately learned output bias.
    """

    def __init__(
        self,
        *,
        sensor_count: int,
        state_dim: int,
        width: int,
        latent_dim: int,
        branch_depth: int,
        trunk_depth: int,
        horizon: float,
        branch_scale: float = 1.0,
        value_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if branch_scale <= 0.0 or value_scale <= 0.0:
            raise ValueError("branch_scale and value_scale must be positive")
        self.horizon = float(horizon)
        self.branch_scale = float(branch_scale)
        self.value_scale = float(value_scale)
        self.branch = _mlp(sensor_count, latent_dim, width, branch_depth)
        self.trunk = _mlp(state_dim + 1, latent_dim, width, trunk_depth)

    def forward(self, branch_values: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if branch_values.ndim != 2 or x.ndim != 2 or t.ndim != 1:
            raise ValueError("expected branch=(batch,sensors), t=(batch,), x=(batch,state_dim)")
        trunk_input = torch.cat([(t / self.horizon).unsqueeze(-1), x], dim=-1)
        branch_latent = self.branch(branch_values / self.branch_scale)
        trunk_latent = self.trunk(trunk_input)
        return self.value_scale * (branch_latent * trunk_latent).sum(dim=-1)


def finite_difference_operators(
    model: nn.Module,
    branch_values: torch.Tensor,
    t: torch.Tensor,
    x: torch.Tensor,
    h: float,
    *,
    center_value: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Paper central ``nabla^h`` and ``Delta^h`` without spatial autograd.

    The network is evaluated at every ``x +/- h e_i``.  The returned tensors
    are the central-difference gradient and summed discrete Laplacian.
    """

    if h <= 0.0:
        raise ValueError("finite-difference step h must be positive")
    if x.requires_grad:
        raise ValueError("spatial coordinates must not require gradients; Eq. (2.3) uses finite differences")
    batch, dim = x.shape
    eye = torch.eye(dim, device=x.device, dtype=x.dtype)
    plus = x[:, None, :] + h * eye[None, :, :]
    minus = x[:, None, :] - h * eye[None, :, :]
    points = torch.cat([plus, minus], dim=1).reshape(batch * 2 * dim, dim)
    branch_repeated = branch_values[:, None, :].expand(batch, 2 * dim, branch_values.shape[1]).reshape(batch * 2 * dim, -1)
    time_repeated = t[:, None].expand(batch, 2 * dim).reshape(-1)
    shifted_values = model(branch_repeated, time_repeated, points).reshape(batch, 2 * dim)
    value_plus = shifted_values[:, :dim]
    value_minus = shifted_values[:, dim:]
    if center_value is None:
        center_value = model(branch_values, t, x)
    gradient = (value_plus - value_minus) / (2.0 * h)
    laplacian = ((value_plus - 2.0 * center_value[:, None] + value_minus) / (h * h)).sum(dim=-1)
    return gradient, laplacian


def terminal_branch_values(
    problem: ControlProblem,
    terminal_parameters: torch.Tensor,
    sensor_states: torch.Tensor,
) -> torch.Tensor:
    batch = terminal_parameters.numel()
    sensors = sensor_states.shape[0]
    repeated_states = sensor_states.unsqueeze(0).expand(batch, sensors, problem.state_dim).reshape(batch * sensors, problem.state_dim)
    repeated_parameters = terminal_parameters.reshape(-1, 1).expand(batch, sensors).reshape(-1)
    return problem.terminal_cost(repeated_states, repeated_parameters).reshape(batch, sensors)


class FrozenPolicy:
    """A fixed policy coefficient used during one policy-evaluation solve."""

    def __call__(self, branch: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> ArgminResult:
        raise NotImplementedError


class ConstantPolicy(FrozenPolicy):
    def __init__(self, control: np.ndarray) -> None:
        self.control = np.asarray(control, dtype=np.float64).reshape(1, -1)

    def __call__(self, branch: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> ArgminResult:
        del branch, t
        value = torch.as_tensor(self.control, device=x.device, dtype=x.dtype).expand(x.shape[0], -1)
        return ArgminResult(value, torch.zeros_like(value, dtype=torch.bool))


class ImprovedPolicy(FrozenPolicy):
    """Eq. (2.4), evaluated from a frozen previous value network."""

    def __init__(self, model: DeepONet, problem: ControlProblem, h: float, tie_tolerance: float = 0.0) -> None:
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.problem = problem
        self.h = h
        self.tie_tolerance = tie_tolerance

    def __call__(self, branch: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> ArgminResult:
        with torch.no_grad():
            gradient, _ = finite_difference_operators(self.model, branch, t, x.detach(), self.h)
            return self.problem.exact_hamiltonian_argmin(
                t,
                x,
                gradient,
                tie_tolerance=self.tie_tolerance,
            )


def physics_informed_loss(
    model: DeepONet,
    problem: ControlProblem,
    policy: FrozenPolicy,
    branch: torch.Tensor,
    terminal_parameters: torch.Tensor,
    t: torch.Tensor,
    x: torch.Tensor,
    terminal_x: torch.Tensor,
    sensor_states: torch.Tensor,
    config: TrainConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Algorithm-1 policy-evaluation loss for paper Eq. (2.3)."""

    t_for_derivative = t.detach().clone().requires_grad_(True)
    x_fixed = x.detach()
    value = model(branch, t_for_derivative, x_fixed)
    value_t = torch.autograd.grad(value.sum(), t_for_derivative, create_graph=True)[0]
    gradient_h, laplacian_h = finite_difference_operators(
        model,
        branch,
        t_for_derivative,
        x_fixed,
        config.h,
        center_value=value,
    )
    policy_result = policy(branch, t.detach(), x_fixed)
    dynamics = problem.dynamics(t.detach(), x_fixed, policy_result.control)
    running = problem.running_cost(t.detach(), x_fixed, policy_result.control)
    # Eq. (2.3): V_t + L + grad^h V.f = -N h Delta^h V.
    residual = value_t + running + (gradient_h * dynamics).sum(dim=-1) + config.viscosity_N * config.h * laplacian_h

    terminal_branch = terminal_branch_values(problem, terminal_parameters, sensor_states)
    terminal_t = torch.full((terminal_x.shape[0],), problem.T, device=terminal_x.device, dtype=terminal_x.dtype)
    terminal_value = model(terminal_branch, terminal_t, terminal_x)
    terminal_target = problem.terminal_cost(terminal_x, terminal_parameters)
    terminal_residual = terminal_value - terminal_target
    pde_mse = residual.square().mean()
    terminal_mse = terminal_residual.square().mean()
    total = config.pde_weight * pde_mse + config.terminal_weight * terminal_mse
    metrics = {
        "loss": float(total.detach().cpu()),
        "pde_mse": float(pde_mse.detach().cpu()),
        "terminal_mse": float(terminal_mse.detach().cpu()),
        "residual_abs_mean": float(residual.abs().mean().detach().cpu()),
        "control_mean": float(policy_result.control.mean().detach().cpu()),
        "nonunique_fraction": float(policy_result.nonunique.to(torch.float64).mean().detach().cpu()),
    }
    return total, metrics


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _dtype(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def sample_state_box(
    problem: ControlProblem,
    count: int,
    h: float,
    rng: np.random.Generator,
) -> np.ndarray:
    lower = problem.state_lower + h
    upper = problem.state_upper - h
    if np.any(lower >= upper):
        raise ValueError("finite-difference stencil does not fit inside declared state domain")
    return rng.uniform(lower, upper, size=(count, problem.state_dim))


def make_sensor_states(problem: ControlProblem, count: int, rng: np.random.Generator) -> np.ndarray:
    """Sample sensors only from the declared domain; no trajectory is read."""

    return rng.uniform(problem.state_lower, problem.state_upper, size=(count, problem.state_dim))


@dataclass
class TrainingResult:
    model: DeepONet
    sensor_states: np.ndarray
    history: list[dict[str, float | int]]
    checkpoints: list[Path]
    required_viscosity_N: float
    dynamics_sup_bound: float


def build_model(problem: ControlProblem, config: TrainConfig) -> DeepONet:
    return DeepONet(
        sensor_count=config.sensors,
        state_dim=problem.state_dim,
        width=config.width,
        latent_dim=config.latent_dim,
        branch_depth=config.branch_depth,
        trunk_depth=config.trunk_depth,
        horizon=problem.T,
        branch_scale=config.branch_scale,
        value_scale=config.value_scale,
    )


def train_policy_iteration(
    problem: ControlProblem,
    terminal_parameter_family: Sequence[float],
    config: TrainConfig,
    *,
    seed: int,
    output_dir: Path,
    initial_control: np.ndarray,
    progress: Callable[[dict[str, float | int]], None] | None = None,
) -> TrainingResult:
    """Run Algorithm 1 with a frozen policy at every outer iteration."""

    if config.outer_iterations < 1 or config.steps_per_outer < 1:
        raise ValueError("outer_iterations and steps_per_outer must be positive")
    if not terminal_parameter_family:
        raise ValueError("terminal_parameter_family must not be empty")
    dynamics_bound = problem.dynamics_sup_bound()
    required_N = validate_viscosity_constant(config.viscosity_N, dynamics_bound)
    set_deterministic_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device(config.device)
    dtype = _dtype(config.dtype)
    output_dir.mkdir(parents=True, exist_ok=True)

    sensors_np = make_sensor_states(problem, config.sensors, rng)
    sensors = torch.as_tensor(sensors_np, device=device, dtype=dtype)
    family = np.asarray(terminal_parameter_family, dtype=np.float64)
    model = build_model(problem, config).to(device=device, dtype=dtype)
    current_policy: FrozenPolicy = ConstantPolicy(initial_control)
    history: list[dict[str, float | int]] = []
    checkpoints: list[Path] = []

    for outer in range(config.outer_iterations):
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        for step in range(1, config.steps_per_outer + 1):
            # Algorithm 1, lines 5--7, explicitly evaluates the losses "with
            # all g^k in G".  Collocation locations remain stochastic, but
            # every terminal function is evaluated at every sampled location.
            t_collocation = rng.uniform(0.0, problem.T, size=config.batch_size)
            x_collocation = sample_state_box(problem, config.batch_size, config.h, rng)
            terminal_collocation = sample_state_box(problem, config.terminal_batch_size, config.h, rng)
            parameter_np = np.repeat(family, config.batch_size)
            terminal_parameter_np = np.repeat(family, config.terminal_batch_size)
            t_np = np.tile(t_collocation, family.size)
            x_np = np.tile(x_collocation, (family.size, 1))
            terminal_x_np = np.tile(terminal_collocation, (family.size, 1))
            parameter = torch.as_tensor(parameter_np, device=device, dtype=dtype)
            terminal_parameter = torch.as_tensor(terminal_parameter_np, device=device, dtype=dtype)
            t = torch.as_tensor(t_np, device=device, dtype=dtype)
            x = torch.as_tensor(x_np, device=device, dtype=dtype)
            terminal_x = torch.as_tensor(terminal_x_np, device=device, dtype=dtype)
            branch = terminal_branch_values(problem, parameter, sensors)
            loss, metrics = physics_informed_loss(
                model,
                problem,
                current_policy,
                branch,
                terminal_parameter,
                t,
                x,
                terminal_x,
                sensors,
                config,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.gradient_clip is not None:
                if config.gradient_clip <= 0.0:
                    raise ValueError("gradient_clip must be positive when enabled")
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            if step == 1 or step % config.log_every == 0 or step == config.steps_per_outer:
                row: dict[str, float | int] = {
                    "outer": outer,
                    "step": step,
                    **metrics,
                    "h": config.h,
                    "viscosity_N": config.viscosity_N,
                    "viscosity_Nh": config.viscosity_N * config.h,
                    "dynamics_sup_bound": dynamics_bound,
                    "required_viscosity_N": required_N,
                    "terminal_functions_per_step": int(family.size),
                }
                history.append(row)
                if progress is not None:
                    progress(row)

        checkpoint = output_dir / f"checkpoint_outer_{outer:03d}.pt"
        torch.save(
            {
                "method": "PI-DeepONet Eq.(2.3)-(2.5) Algorithm 1",
                "outer": outer,
                "seed": seed,
                "model_state_dict": model.state_dict(),
                "sensor_states": sensors_np,
                "terminal_parameter_family": family,
                "terminal_function_usage": "all functions at every Adam step (Algorithm 1 lines 5--7)",
                "problem": problem.metadata(),
                "train_config": config.to_dict(),
                "selection_rule": "final_predeclared_outer_iteration",
            },
            checkpoint,
        )
        checkpoints.append(checkpoint)
        # Critical for true policy iteration: u_{n+1} is computed from a frozen
        # copy of V_n and remains fixed while V_{n+1} is trained.
        current_policy = ImprovedPolicy(model, problem, config.h, config.tie_tolerance)

    return TrainingResult(
        model=model,
        sensor_states=sensors_np,
        history=history,
        checkpoints=checkpoints,
        required_viscosity_N=required_N,
        dynamics_sup_bound=dynamics_bound,
    )
