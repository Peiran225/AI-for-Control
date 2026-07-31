"""Problem definitions and exact Hamiltonian minimizers for PI-DeepONet."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np
import torch


@dataclass(frozen=True)
class ArgminResult:
    """A control attaining the pointwise Hamiltonian minimum.

    ``nonunique`` is true only where the Hamiltonian is exactly flat in the
    control (up to an explicitly requested numerical tolerance).
    """

    control: torch.Tensor
    nonunique: torch.Tensor


class ControlProblem(Protocol):
    name: str
    T: float
    state_dim: int
    control_dim: int
    state_lower: np.ndarray
    state_upper: np.ndarray
    control_lower: np.ndarray
    control_upper: np.ndarray

    def dynamics(self, t: torch.Tensor, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor: ...

    def running_cost(self, t: torch.Tensor, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor: ...

    def terminal_cost(self, x: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor: ...

    def exact_hamiltonian_argmin(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        gradient: torch.Tensor,
        *,
        tie_control: torch.Tensor | None = None,
        tie_tolerance: float = 0.0,
    ) -> ArgminResult: ...

    def dynamics_sup_bound(self) -> float: ...

    def metadata(self) -> dict[str, Any]: ...


def _tensor(array: np.ndarray, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(array, device=like.device, dtype=like.dtype)


@dataclass(frozen=True)
class PaperLQR5D:
    """The paper's Section 4.2.1 compact-control five-dimensional LQR."""

    T: float = 0.5
    name: str = "paper_lqr_5d_3control"

    @property
    def state_dim(self) -> int:
        return 5

    @property
    def control_dim(self) -> int:
        return 3

    @property
    def A(self) -> np.ndarray:
        return np.array(
            [
                [0.08, 0.01, 0.01, 0.15, 0.05],
                [0.16, 0.16, 0.15, 0.18, 0.16],
                [0.14, 0.17, 0.07, 0.08, 0.12],
                [0.12, 0.15, 0.11, 0.18, 0.02],
                [0.12, 0.08, 0.13, 0.10, 0.09],
            ],
            dtype=np.float64,
        )

    @property
    def B(self) -> np.ndarray:
        return np.array(
            [
                [0.00, 0.05, 0.06],
                [0.07, 0.01, 0.04],
                [0.02, 0.00, 0.10],
                [0.09, 0.08, 0.08],
                [0.01, 0.07, 0.05],
            ],
            dtype=np.float64,
        )

    @property
    def state_lower(self) -> np.ndarray:
        return np.full(self.state_dim, -1.0, dtype=np.float64)

    @property
    def state_upper(self) -> np.ndarray:
        return np.full(self.state_dim, 1.0, dtype=np.float64)

    @property
    def control_lower(self) -> np.ndarray:
        return np.full(self.control_dim, -1.0 / 3.0, dtype=np.float64)

    @property
    def control_upper(self) -> np.ndarray:
        return np.full(self.control_dim, 0.5, dtype=np.float64)

    def dynamics(self, t: torch.Tensor, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        del t
        return x @ _tensor(self.A, x).T + u @ _tensor(self.B, x).T

    def running_cost(self, t: torch.Tensor, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        del t
        return x.square().sum(dim=-1) + u.square().sum(dim=-1)

    def terminal_cost(self, x: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor:
        """Paper Section 4.2.1 terminal-function family and held-out inputs.

        Algorithm 1 accepts whole functions, while this runner uses a scalar
        identifier only as a compact way to construct them.  Identifiers
        ``1,2,3`` mean the three *training* functions

            g_k(x) = 0.3 + 0.1 k ||x||^2.

        Any other identifier ``c`` means the held-out function
        ``g(x)=c||x||^2`` (the paper reports ``c=0.57`` and ``0.45``).
        This distinction preserves the additive 0.3 term that would be lost
        by treating the training identifiers as quadratic coefficients.
        """

        identifier = parameter.reshape(-1)
        norm_squared = x.square().sum(dim=-1)
        rounded = torch.round(identifier)
        is_training_identifier = (
            torch.isclose(identifier, rounded, rtol=0.0, atol=1.0e-12)
            & (rounded >= 1.0)
            & (rounded <= 3.0)
        )
        training_value = 0.3 + 0.1 * rounded * norm_squared
        held_out_value = identifier * norm_squared
        return torch.where(is_training_identifier, training_value, held_out_value)

    def exact_hamiltonian_argmin(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        gradient: torch.Tensor,
        *,
        tie_control: torch.Tensor | None = None,
        tie_tolerance: float = 0.0,
    ) -> ArgminResult:
        """Exact minimizer of ``p.Bu + u.T u`` on the compact box."""

        del t, x, tie_control, tie_tolerance
        unconstrained = -0.5 * (gradient @ _tensor(self.B, gradient))
        lower = _tensor(self.control_lower, gradient)
        upper = _tensor(self.control_upper, gradient)
        return ArgminResult(torch.clamp(unconstrained, min=lower, max=upper), torch.zeros_like(unconstrained, dtype=torch.bool))

    def dynamics_sup_bound(self) -> float:
        x_abs = np.maximum(np.abs(self.state_lower), np.abs(self.state_upper))
        u_abs = np.maximum(np.abs(self.control_lower), np.abs(self.control_upper))
        component_bounds = np.abs(self.A) @ x_abs + np.abs(self.B) @ u_abs
        return float(component_bounds.max())

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "T": self.T,
            "state_dim": self.state_dim,
            "control_dim": self.control_dim,
            "A": self.A.tolist(),
            "B": self.B.tolist(),
            "Q": np.eye(self.state_dim).tolist(),
            "R": np.eye(self.control_dim).tolist(),
            "terminal_training_family": [
                "0.3 + 0.1 ||x||^2",
                "0.3 + 0.2 ||x||^2",
                "0.3 + 0.3 ||x||^2",
            ],
            "terminal_identifier_convention": (
                "identifiers 1,2,3 construct the paper training family; "
                "other values c construct held-out c||x||^2"
            ),
            "state_domain": [self.state_lower.tolist(), self.state_upper.tolist()],
            "control_domain": [self.control_lower.tolist(), self.control_upper.tolist()],
        }


@dataclass(frozen=True)
class TumorAdaptation:
    """The common tumor OCP in a bounded, dimensionless state coordinate.

    The network coordinate is ``x = N / state_scale``.  This keeps the
    finite-difference step ``h`` meaningful and makes the bounded-domain
    dynamics norm in Theorem 1 explicit.  Exported trajectories are converted
    back to the physical ``N`` coordinate before common evaluation.
    """

    T: float = 10.0
    state_dim: int = 21
    control_dim: int = 1
    umax: float = 3.0
    beta: float = 0.1
    alpha: float = 1.0
    gamma: float = 20.0
    initial_N: float = 10.0
    suppression: float = 0.5
    state_scale: float = 220.0
    normalized_state_lower: float = 0.0
    normalized_state_upper: float = 1.0
    name: str = "tumor_bounded_domain_adaptation"

    @property
    def grid(self) -> np.ndarray:
        return np.linspace(0.0, 1.0, self.state_dim, dtype=np.float64)

    @property
    def r(self) -> np.ndarray:
        return 2.0 / (1.0 + 3.0 * self.grid**4)

    @property
    def phi(self) -> np.ndarray:
        return 1.0 / (1.0 + self.grid**2)

    @property
    def M(self) -> np.ndarray:
        return np.full(self.state_dim, self.suppression, dtype=np.float64)

    @property
    def state_lower(self) -> np.ndarray:
        return np.full(self.state_dim, self.normalized_state_lower, dtype=np.float64)

    @property
    def state_upper(self) -> np.ndarray:
        return np.full(self.state_dim, self.normalized_state_upper, dtype=np.float64)

    @property
    def control_lower(self) -> np.ndarray:
        return np.zeros(1, dtype=np.float64)

    @property
    def control_upper(self) -> np.ndarray:
        return np.full(1, self.umax, dtype=np.float64)

    @property
    def initial_state(self) -> np.ndarray:
        return np.full(self.state_dim, self.initial_N / self.state_scale, dtype=np.float64)

    def physical_state(self, x: torch.Tensor) -> torch.Tensor:
        return self.state_scale * x

    def dynamics(self, t: torch.Tensor, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        del t
        physical = self.physical_state(x)
        G = torch.log1p(physical.mean(dim=-1, keepdim=True))
        rate = _tensor(self.r, x) - _tensor(self.phi, x) * u - _tensor(self.M, x) * G
        return rate * x

    def running_cost(self, t: torch.Tensor, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        del t
        return self.beta * self.physical_state(x).sum(dim=-1) + self.gamma * u.squeeze(-1)

    def terminal_cost(self, x: torch.Tensor, parameter: torch.Tensor) -> torch.Tensor:
        return parameter.reshape(-1) * self.alpha * self.physical_state(x).sum(dim=-1)

    def switching_coefficient(self, x: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
        # p_x . d(f_x)/du + gamma = gamma - sum_i p_{x_i} phi_i x_i.
        return self.gamma - (gradient * _tensor(self.phi, x) * x).sum(dim=-1, keepdim=True)

    def exact_hamiltonian_argmin(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        gradient: torch.Tensor,
        *,
        tie_control: torch.Tensor | None = None,
        tie_tolerance: float = 0.0,
    ) -> ArgminResult:
        """Exact box argmin for the affine-in-control tumor Hamiltonian.

        Positive coefficient selects ``u=0`` and negative coefficient selects
        ``u=umax``.  At an exactly flat Hamiltonian every feasible control is a
        minimizer; the default deterministic convention is the box midpoint.
        A positive ``tie_tolerance`` is optional and is recorded by runners.
        """

        del t
        coefficient = self.switching_coefficient(x, gradient)
        if tie_tolerance < 0.0:
            raise ValueError("tie_tolerance must be nonnegative")
        if tie_tolerance == 0.0:
            nonunique = coefficient == 0.0
        else:
            nonunique = coefficient.abs() <= tie_tolerance
        if tie_control is None:
            tie = torch.full_like(coefficient, 0.5 * self.umax)
        else:
            tie = torch.clamp(tie_control, 0.0, self.umax)
        control = torch.where(coefficient > tie_tolerance, torch.zeros_like(coefficient), torch.full_like(coefficient, self.umax))
        control = torch.where(nonunique, tie, control)
        return ArgminResult(control=control, nonunique=nonunique)

    def dynamics_sup_bound(self) -> float:
        # On the declared box, |x_i'| <= x_hi (r_i + phi_i umax + M_i Gmax).
        upper = float(self.normalized_state_upper)
        max_G = float(np.log1p(self.state_scale * upper))
        component = upper * (np.abs(self.r) + np.abs(self.phi) * self.umax + np.abs(self.M) * max_G)
        return float(component.max())

    def metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            {
                "state_coordinate": "x=N/state_scale",
                "state_domain": [self.state_lower.tolist(), self.state_upper.tolist()],
                "control_domain": [self.control_lower.tolist(), self.control_upper.tolist()],
                "r": self.r.tolist(),
                "phi": self.phi.tolist(),
                "M": self.M.tolist(),
            }
        )
        return data
