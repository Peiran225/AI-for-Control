"""Entropy-consistent tumor OCP for the Adaptive HJB-NN method.

The original HJB-NN examples assume a differentiable Hamiltonian minimizer.
For the common tumor problem the unregularized Hamiltonian is affine in the
bounded control.  We therefore use the exact minimizer of the explicitly
regularized running cost

    L_tau(N, u) = beta.N + gamma*u
                  + tau*U*[p log(p) + (1-p) log(1-p)], p=u/U.

Its first-order condition is ``psi + tau*log(p/(1-p)) = 0`` and hence
``u = U*sigmoid(-psi/tau)``.  The same entropy term is used in characteristic
value labels; it is not a numerical minimizer detached from the objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


Array = np.ndarray


def _stable_sigmoid(x: Array) -> Array:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0.0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exponential = np.exp(x[~positive])
    out[~positive] = exponential / (1.0 + exponential)
    return out


def _as_columns(values: Array, rows: int) -> Array:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape((-1, 1))
    if values.ndim != 2 or values.shape[0] != rows:
        raise ValueError(f"expected ({rows}, samples), got {values.shape}")
    return values


@dataclass(frozen=True)
class TumorParameters:
    state_dim: int = 21
    final_time: float = 10.0
    umax: float = 3.0
    beta: float = 0.1
    alpha: float = 1.0
    gamma: float = 20.0
    initial_state: float = 10.0
    suppression: float = 0.5
    initial_lower: float = 5.0
    initial_upper: float = 20.0


class TumorEntropyProblem:
    """Official-HJBnet-compatible entropy-regularized tumor problem."""

    def __init__(self, tau: float, parameters: TumorParameters | None = None):
        if not np.isfinite(tau) or tau <= 0.0:
            raise ValueError("tau must be finite and strictly positive")
        params = parameters or TumorParameters()
        if params.state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if params.initial_lower <= 0.0 or params.initial_lower >= params.initial_upper:
            raise ValueError("initial-state bounds must satisfy 0 < lower < upper")

        self.params = params
        self.tau = float(tau)
        self.N_states = params.state_dim
        self.N_controls = 1
        self.t1 = params.final_time
        self.umax = params.umax
        self.gamma = params.gamma
        self.r = 2.0 / (1.0 + 3.0 * np.linspace(0.0, 1.0, self.N_states) ** 4)
        self.phi = 1.0 / (1.0 + np.linspace(0.0, 1.0, self.N_states) ** 2)
        self.M = np.full(self.N_states, params.suppression, dtype=np.float64)
        self.beta = np.full(self.N_states, params.beta, dtype=np.float64)
        self.alpha = np.full(self.N_states, params.alpha, dtype=np.float64)
        self.X0_lb = np.full((self.N_states, 1), params.initial_lower, dtype=np.float64)
        self.X0_ub = np.full((self.N_states, 1), params.initial_upper, dtype=np.float64)

    @property
    def nominal_initial_state(self) -> Array:
        return np.full(self.N_states, self.params.initial_state, dtype=np.float64)

    def sample_initial_states(self, count: int, rng: np.random.Generator) -> Array:
        """Sample every state coordinate independently from the declared box."""
        if count <= 0:
            raise ValueError("count must be positive")
        return rng.uniform(self.X0_lb, self.X0_ub, size=(self.N_states, count))

    def sample_X0(self, count: int) -> Array:
        """Compatibility hook for the author runner; prefer explicit RNG above."""
        values = np.random.uniform(self.X0_lb, self.X0_ub, size=(self.N_states, count))
        return values[:, 0] if count == 1 else values

    def _positive_state(self, state: Array) -> Array:
        state = _as_columns(state, self.N_states)
        if not np.all(np.isfinite(state)):
            raise FloatingPointError("tumor states must be finite")
        # Collocation Newton iterates can temporarily leave the physical
        # positive orthant even when the converged characteristic is positive.
        # This domain extension is inactive on every accepted data point; the
        # BVP wrapper rejects any converged trajectory with N_i <= 0.
        return np.maximum(state, 1.0e-12)

    def crowding(self, state: Array) -> Array:
        state = self._positive_state(state)
        return np.log1p(np.mean(state, axis=0, keepdims=True))

    def switching_function(self, state: Array, costate: Array) -> Array:
        state = self._positive_state(state)
        costate = _as_columns(costate, self.N_states)
        return self.gamma - np.sum(self.phi[:, None] * state * costate, axis=0, keepdims=True)

    def control_from_switching(self, psi: Array) -> Array:
        return self.umax * _stable_sigmoid(-np.asarray(psi, dtype=np.float64) / self.tau)

    def control_first_order_residual(self, psi: Array, control: Array) -> Array:
        """Return dH_tau/du for an interior entropy-regularized control."""
        p = np.asarray(control, dtype=np.float64) / self.umax
        if np.any((p <= 0.0) | (p >= 1.0)):
            raise ValueError("entropy first-order residual requires 0 < u < umax")
        return np.asarray(psi, dtype=np.float64) + self.tau * np.log(p / (1.0 - p))

    def entropy_regularizer(self, control: Array) -> Array:
        """Evaluate tau*U*(p log p + (1-p) log(1-p)), with 0 log 0 = 0."""
        control = np.asarray(control, dtype=np.float64)
        if np.any(control < 0.0) or np.any(control > self.umax):
            raise ValueError("control lies outside [0, umax]")
        p = control / self.umax
        p_log_p = np.zeros_like(p)
        q_log_q = np.zeros_like(p)
        interior_p = p > 0.0
        interior_q = p < 1.0
        p_log_p[interior_p] = p[interior_p] * np.log(p[interior_p])
        q = 1.0 - p
        q_log_q[interior_q] = q[interior_q] * np.log(q[interior_q])
        return self.tau * self.umax * (p_log_p + q_log_q)

    def U_star(self, augmented_state: Array) -> Array:
        augmented_state = np.asarray(augmented_state, dtype=np.float64)
        if augmented_state.ndim == 1:
            augmented_state = augmented_state.reshape((-1, 1))
        if augmented_state.shape[0] < 2 * self.N_states:
            raise ValueError("augmented state must contain state and costate")
        state = augmented_state[: self.N_states]
        costate = augmented_state[self.N_states : 2 * self.N_states]
        return self.control_from_switching(self.switching_function(state, costate))

    def make_U_NN(self, costate, state):
        """TensorFlow graph of the same entropy-regularized minimizer."""
        import tensorflow.compat.v1 as tf

        phi = tf.constant(self.phi.reshape((-1, 1)), dtype=tf.float32)
        psi = self.gamma - tf.reduce_sum(phi * state * costate, axis=0, keepdims=True)
        return self.umax * tf.sigmoid(-psi / self.tau)

    def running_cost(self, state: Array, control: Array, *, regularized: bool = True) -> Array:
        state = self._positive_state(state)
        control = np.asarray(control, dtype=np.float64)
        if control.ndim == 1:
            control = control.reshape((1, -1))
        if control.shape != (1, state.shape[1]):
            raise ValueError(f"expected control shape {(1, state.shape[1])}, got {control.shape}")
        base = np.sum(self.beta[:, None] * state, axis=0, keepdims=True) + self.gamma * control
        return base + self.entropy_regularizer(control) if regularized else base

    def terminal_cost(self, state: Array):
        state = self._positive_state(state)
        value = np.sum(self.alpha[:, None] * state, axis=0, keepdims=True)
        return float(value[0, 0]) if value.shape[1] == 1 else value

    def compute_cost(self, time: Array, state: Array, control: Array, *, regularized: bool = True) -> Array:
        time = np.asarray(time, dtype=np.float64).reshape(-1)
        state = self._positive_state(state)
        control = np.asarray(control, dtype=np.float64).reshape((1, -1))
        if time.size != state.shape[1] or time.size != control.shape[1]:
            raise ValueError("time, state, and control samples must align")
        running = self.running_cost(state, control, regularized=regularized).reshape(-1)
        integral = np.trapz(running, time)
        terminal = float(np.sum(self.alpha * state[:, -1]))
        return np.array([[terminal + integral]], dtype=np.float64)

    def make_bc(self, initial_state: Array) -> Callable[[Array, Array], Array]:
        initial_state = np.asarray(initial_state, dtype=np.float64).reshape(self.N_states)

        def boundary(initial_augmented: Array, terminal_augmented: Array) -> Array:
            return np.concatenate(
                (
                    initial_augmented[: self.N_states] - initial_state,
                    terminal_augmented[self.N_states : 2 * self.N_states] - self.alpha,
                    terminal_augmented[2 * self.N_states : 2 * self.N_states + 1],
                )
            )

        return boundary

    def dynamics(self, time: float, state: Array, control_function) -> Array:
        state_vector = np.asarray(state, dtype=np.float64).reshape(self.N_states)
        if np.any(state_vector <= 0.0):
            raise FloatingPointError("tumor states must remain strictly positive")
        control = float(
            np.asarray(control_function([[time]], state_vector.reshape((-1, 1)))).reshape(-1)[0]
        )
        if control < 0.0 or control > self.umax:
            raise FloatingPointError("feedback returned an infeasible control")
        crowding = np.log1p(np.mean(state_vector))
        return (self.r - self.phi * control - self.M * crowding) * state_vector

    def aug_dynamics(self, time: Array, augmented_state: Array) -> Array:
        del time
        augmented_state = np.asarray(augmented_state, dtype=np.float64)
        state = self._positive_state(augmented_state[: self.N_states])
        costate = _as_columns(
            augmented_state[self.N_states : 2 * self.N_states], self.N_states
        )
        control = self.U_star(augmented_state)
        crowding = self.crowding(state)
        coefficient = (
            self.r[:, None]
            - self.phi[:, None] * control
            - self.M[:, None] * crowding
        )
        state_rhs = coefficient * state

        denominator = self.N_states + np.sum(state, axis=0, keepdims=True)
        coupling = np.sum(self.M[:, None] * costate * state, axis=0, keepdims=True)
        costate_rhs = -self.beta[:, None] - costate * coefficient + coupling / denominator
        value_rhs = -self.running_cost(state, control, regularized=True)
        return np.vstack((state_rhs, costate_rhs, value_rhs))

    def Hamiltonian(self, time: Array, augmented_state: Array) -> Array:
        del time
        augmented_state = np.asarray(augmented_state, dtype=np.float64)
        state = _as_columns(augmented_state[: self.N_states], self.N_states)
        costate = _as_columns(
            augmented_state[self.N_states : 2 * self.N_states], self.N_states
        )
        control = self.U_star(augmented_state)
        dynamics = self.aug_dynamics(np.array([0.0]), augmented_state)[: self.N_states]
        return self.running_cost(state, control, regularized=True) + np.sum(
            costate * dynamics, axis=0, keepdims=True
        )


def evaluate_zoh_cost_pair(
    problem: TumorEntropyProblem,
    time: Array,
    control: Array,
    *,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-12,
) -> dict[str, float]:
    """Integrate regularized and unregularized costs on identical ZOH segments."""
    from scipy.integrate import solve_ivp

    time = np.asarray(time, dtype=np.float64).reshape(-1)
    control = np.asarray(control, dtype=np.float64).reshape(-1)
    if time.size < 2 or not np.all(np.diff(time) > 0.0):
        raise ValueError("time must be a strictly increasing breakpoint grid")
    if not np.isclose(time[0], 0.0, atol=1.0e-12) or not np.isclose(
        time[-1], problem.t1, atol=1.0e-12
    ):
        raise ValueError(f"time grid must span [0,{problem.t1}]")
    if control.size == time.size:
        control = control[:-1]
    if control.size != time.size - 1:
        raise ValueError("ZOH control must have len(time)-1 interval values")
    if np.any(~np.isfinite(control)) or np.any((control < 0.0) | (control > problem.umax)):
        raise ValueError("ZOH control contains a nonfinite or infeasible value")

    # Last two states accumulate the declared entropy-regularized cost and the
    # common unregularized cost along exactly the same numerical trajectory.
    augmented = np.concatenate(
        (problem.nominal_initial_state, np.zeros(2, dtype=np.float64))
    )
    for index, action in enumerate(control):
        entropy = float(problem.entropy_regularizer(np.array([action]))[0])

        def rhs(_time, value):
            state = value[: problem.N_states]
            if np.any(state <= 0.0):
                raise FloatingPointError("ZOH realized trajectory left positive orthant")
            crowding = np.log1p(np.mean(state))
            state_rhs = (
                problem.r - problem.phi * action - problem.M * crowding
            ) * state
            unregularized = float(np.sum(problem.beta * state) + problem.gamma * action)
            return np.concatenate(
                (state_rhs, np.array([unregularized + entropy, unregularized]))
            )

        solution = solve_ivp(
            rhs,
            (float(time[index]), float(time[index + 1])),
            augmented,
            method="DOP853",
            rtol=rtol,
            atol=atol,
        )
        if not solution.success:
            raise RuntimeError(f"ZOH cost integration failed: {solution.message}")
        augmented = solution.y[:, -1]

    final_state = augmented[: problem.N_states]
    if np.any(final_state <= 0.0):
        raise RuntimeError("ZOH realized terminal state is nonpositive")
    terminal = float(np.sum(problem.alpha * final_state))
    regularized_running = float(augmented[-2])
    unregularized_running = float(augmented[-1])
    return {
        "terminal_cost": terminal,
        "regularized_running_cost": regularized_running,
        "regularized_J": terminal + regularized_running,
        "unregularized_running_cost": unregularized_running,
        "unregularized_J": terminal + unregularized_running,
        "entropy_integral": regularized_running - unregularized_running,
        "final_total_N": float(np.sum(final_state)),
    }
