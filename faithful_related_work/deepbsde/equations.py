"""Equations for a faithful Han--Jentzen--E DeepBSDE reproduction.

The faithful tumor equation is deliberately implemented outside
``external/DeepBSDE``.  Its tracked ``equation.py`` has a legacy TumorHJB
append, but the original upstream prefix is verified against Git and that
legacy class is never imported for the faithful tumor run.  The upstream
``solver.py`` is verified byte-for-byte against its Git ``HEAD`` before a full
HJB-LQ benchmark.

Let ``x = log(N)``.  The deterministic controlled dynamics become

    dx_i = [r_i - M_i G(exp(x)) - phi_i u] dt.

For viscosity ``sigma > 0`` we select the uncontrolled part as the reference
forward SDE,

    dX_i = [r_i - M_i G(exp(X))] dt + sigma dW_i.

The semilinear HJB equation is

    V_t + b0 . grad(V) + sigma^2/2 Delta(V)
        + beta . exp(x)
        + min_{0 <= u <= U} u [gamma - phi . grad(V)] = 0,
    V(T, x) = alpha . exp(x).

In the DeepBSDE convention ``Z = sigma * grad(V)``.  Consequently the BSDE
generator contains ``psi = gamma - phi . Z / sigma`` and the Hamiltonian
minimizer is exactly 0 for ``psi >= 0`` and U for ``psi < 0``.  No state
clipping or smoothed control is used.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf


def tumor_parameter_vectors(dim: int, *, beta: float, alpha: float, m_suppression: float) -> dict[str, np.ndarray]:
    """Return the parameter vectors used by the canonical tumor problem."""

    grid = np.linspace(0.0, 1.0, int(dim), dtype=np.float64)
    return {
        "grid": grid,
        "r": 2.0 / (1.0 + 3.0 * grid**4),
        "phi": 1.0 / (1.0 + grid**2),
        "M": np.full(int(dim), float(m_suppression), dtype=np.float64),
        "beta": np.full(int(dim), float(beta), dtype=np.float64),
        "alpha": np.full(int(dim), float(alpha), dtype=np.float64),
    }


def switching_function_numpy(
    grad_log_value: np.ndarray,
    phi: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Compute ``gamma - phi . grad_x(V)`` for ``x = log(N)``."""

    gradient = np.asarray(grad_log_value, dtype=np.float64)
    phi_vector = np.asarray(phi, dtype=np.float64)
    return float(gamma) - np.sum(gradient * phi_vector, axis=-1)


def hamiltonian_argmin_numpy(
    grad_log_value: np.ndarray,
    phi: np.ndarray,
    gamma: float,
    umax: float,
) -> np.ndarray:
    """Return the exact minimizer of the control-affine box Hamiltonian.

    At an exact tie (``psi == 0``), both endpoints and all interior controls
    minimize the Hamiltonian.  We use the deterministic convention ``u=0``.
    """

    psi = switching_function_numpy(grad_log_value, phi, gamma)
    return np.where(psi < 0.0, float(umax), 0.0)


def _stable_log_one_plus_mean_exp(x: np.ndarray) -> np.ndarray:
    """Compute ``log(1 + mean(exp(x)))`` without clipping ``x``."""

    x_array = np.asarray(x, dtype=np.float64)
    maximum = np.max(x_array, axis=-1, keepdims=True)
    log_mean_exp = maximum + np.log(np.mean(np.exp(x_array - maximum), axis=-1, keepdims=True))
    return np.logaddexp(0.0, log_mean_exp)


class LogStateTumorHJB:
    """Log-state semilinear parabolic HJB for the tumor OCP.

    The class implements the equation interface expected by the official
    DeepBSDE solver: ``sample``, ``f_tf`` and ``g_tf`` plus the standard shape
    attributes.  Sampling owns an explicit NumPy generator, so a run is fully
    determined by its seed and does not depend on unrelated global RNG calls.
    """

    uses_state_clipping = False

    def __init__(self, eqn_config: Any):
        self.dim = int(eqn_config.dim)
        self.total_time = float(eqn_config.total_time)
        self.num_time_interval = int(eqn_config.num_time_interval)
        if self.dim <= 0 or self.num_time_interval <= 0 or self.total_time <= 0.0:
            raise ValueError("dim, total_time and num_time_interval must be positive")
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.sigma = float(eqn_config.sigma)
        if self.sigma <= 0.0:
            raise ValueError("DeepBSDE requires sigma > 0; use a positive viscosity sweep")

        self.umax = float(getattr(eqn_config, "umax", 3.0))
        self.gamma = float(getattr(eqn_config, "gamma", 20.0))
        self.n0 = float(getattr(eqn_config, "n0", 10.0))
        if self.n0 <= 0.0:
            raise ValueError("log-state coordinates require n0 > 0")
        beta = float(getattr(eqn_config, "beta", 0.1))
        alpha = float(getattr(eqn_config, "alpha", 1.0))
        m_suppression = float(getattr(eqn_config, "m_suppression", 0.5))
        self.params = tumor_parameter_vectors(
            self.dim,
            beta=beta,
            alpha=alpha,
            m_suppression=m_suppression,
        )
        self.x_init = np.full(self.dim, np.log(self.n0), dtype=np.float64)
        self.seed = int(getattr(eqn_config, "seed", 0))
        self.rng = np.random.default_rng(self.seed)
        self.y_init = None
        self.sample_calls = 0

    def uncontrolled_log_drift_numpy(self, x: np.ndarray) -> np.ndarray:
        """Reference-SDE drift ``r - M G(exp(x))`` with no clipping."""

        x_array = np.asarray(x, dtype=np.float64)
        if x_array.shape[-1] != self.dim:
            raise ValueError(f"expected last dimension {self.dim}, got {x_array.shape}")
        growth_suppression = _stable_log_one_plus_mean_exp(x_array)
        return self.params["r"] - self.params["M"] * growth_suppression

    def controlled_log_drift_numpy(self, x: np.ndarray, control: float) -> np.ndarray:
        """Deterministic physical log-state drift used for policy rollout."""

        return self.uncontrolled_log_drift_numpy(x) - self.params["phi"] * float(control)

    def sample(self, num_sample: int) -> tuple[np.ndarray, np.ndarray]:
        """Euler-sample the reference log-state SDE without state clipping."""

        count = int(num_sample)
        if count <= 0:
            raise ValueError("num_sample must be positive")
        dw = self.rng.normal(size=(count, self.dim, self.num_time_interval)) * self.sqrt_delta_t
        x = np.empty((count, self.dim, self.num_time_interval + 1), dtype=np.float64)
        x[:, :, 0] = self.x_init
        for index in range(self.num_time_interval):
            current = x[:, :, index]
            drift = self.uncontrolled_log_drift_numpy(current)
            x[:, :, index + 1] = current + drift * self.delta_t + self.sigma * dw[:, :, index]
        if not np.all(np.isfinite(x)):
            raise FloatingPointError("non-finite log-state sample; no clipping was applied")
        self.sample_calls += 1
        return dw, x

    def f_tf(self, t: float, x: tf.Tensor, y: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
        """DeepBSDE generator for the semilinear log-state HJB."""

        del t, y
        dtype = z.dtype
        beta = tf.convert_to_tensor(self.params["beta"], dtype=dtype)
        phi = tf.convert_to_tensor(self.params["phi"], dtype=dtype)
        populations = tf.debugging.check_numerics(tf.exp(x), "exp(log-state) is non-finite")
        running_state = tf.reduce_sum(beta * populations, axis=1, keepdims=True)
        grad_log_value = z / tf.constant(self.sigma, dtype=dtype)
        psi = tf.constant(self.gamma, dtype=dtype) - tf.reduce_sum(phi * grad_log_value, axis=1, keepdims=True)
        control_hamiltonian = tf.constant(self.umax, dtype=dtype) * tf.minimum(psi, tf.zeros_like(psi))
        return running_state + control_hamiltonian

    def g_tf(self, t: float, x: tf.Tensor) -> tf.Tensor:
        """Terminal condition ``alpha . N(T) = alpha . exp(x(T))``."""

        del t
        dtype = x.dtype
        alpha = tf.convert_to_tensor(self.params["alpha"], dtype=dtype)
        populations = tf.debugging.check_numerics(tf.exp(x), "terminal exp(log-state) is non-finite")
        return tf.reduce_sum(alpha * populations, axis=1, keepdims=True)
