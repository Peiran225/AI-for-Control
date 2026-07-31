"""Deterministic offline dynamics learning used by Neural-PMP."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn


def seed_everything(seed: int) -> None:
    """Seed every RNG used here and request deterministic torch kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


class DynamicsMLP(nn.Module):
    """Paper/repository Type-B dynamics network: two ReLU hidden layers."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128, *, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim, dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim, dtype=dtype),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.network(value)


@dataclass(frozen=True)
class DynamicsTrainingConfig:
    epochs: int = 50_000
    learning_rate: float = 5e-3
    weight_decay: float = 1e-4
    validation_interval: int = 100
    validation_patience: int = 100
    minimum_improvement: float = 1e-10


@dataclass
class DynamicsTrainingResult:
    model: DynamicsMLP
    history: list[dict[str, float | int]]
    best_epoch: int
    selected_epoch: int
    checkpoint_policy: str
    stop_reason: str
    best_validation_mse: float


def array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(str(contiguous.shape).encode("utf-8"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def train_dynamics_model(
    *,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    validation_inputs: np.ndarray,
    validation_targets: np.ndarray,
    state_dim: int,
    action_dim: int,
    hidden_dim: int,
    seed: int,
    config: DynamicsTrainingConfig,
    dtype: torch.dtype = torch.float32,
    checkpoint_policy: str = "best_validation",
    device: torch.device | str = "cpu",
) -> DynamicsTrainingResult:
    """Full-batch Adam with either held-out selection or fixed-final return."""
    if checkpoint_policy not in {"best_validation", "fixed_final"}:
        raise ValueError("checkpoint_policy must be best_validation or fixed_final")
    seed_everything(seed)
    device = torch.device(device)
    model = DynamicsMLP(state_dim, action_dim, hidden_dim, dtype=dtype).to(device)
    train_x = torch.as_tensor(train_inputs, dtype=dtype, device=device)
    train_y = torch.as_tensor(train_targets, dtype=dtype, device=device)
    validation_x = torch.as_tensor(validation_inputs, dtype=dtype, device=device)
    validation_y = torch.as_tensor(validation_targets, dtype=dtype, device=device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: list[dict[str, float | int]] = []
    best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    best_epoch = 0
    best_validation = float("inf")
    stale = 0
    stop_reason = "max_epochs"

    for epoch in range(config.epochs + 1):
        should_evaluate = epoch == 0 or epoch % config.validation_interval == 0 or epoch == config.epochs
        if should_evaluate:
            model.eval()
            with torch.no_grad():
                validation_mse = float(torch.mean((model(validation_x) - validation_y) ** 2))
                training_mse = float(torch.mean((model(train_x) - train_y) ** 2))
            history.append(
                {
                    "epoch": epoch,
                    "training_mse": training_mse,
                    "validation_mse": validation_mse,
                }
            )
            if validation_mse < best_validation - config.minimum_improvement:
                best_validation = validation_mse
                best_epoch = epoch
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if checkpoint_policy == "best_validation" and stale >= config.validation_patience:
                stop_reason = "validation_patience"
                break
        if epoch == config.epochs:
            break

        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(train_x) - train_y) ** 2)
        loss.backward()
        optimizer.step()

    if checkpoint_policy == "best_validation":
        model.load_state_dict(best_state)
        selected_epoch = best_epoch
    else:
        # Official Neural-PMP flow uses the model after the fixed training
        # budget. Validation is logged as a diagnostic and never selects a
        # checkpoint in this mode.
        selected_epoch = config.epochs
        stop_reason = "fixed_budget_final"
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return DynamicsTrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        selected_epoch=selected_epoch,
        checkpoint_policy=checkpoint_policy,
        stop_reason=stop_reason,
        best_validation_mse=best_validation,
    )
