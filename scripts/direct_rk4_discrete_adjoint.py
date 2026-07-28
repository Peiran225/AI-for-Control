#!/usr/bin/env python3
"""Fast exact value/gradient for the common RK4--ZOH tumor objective.

The common evaluator represents the control by ``n`` interval values.  Each
value is held fixed over its interval, while the state and running objective
are advanced by four classical-RK4 microsteps.  Differentiating that objective
with PyTorch autograd builds a graph with 3,200 RK4 steps when ``n=800``.

This module implements the matching discrete adjoint explicitly in NumPy.  It
uses the rank-one structure of the tumor-dynamics state Jacobian, so neither a
21-by-21 Jacobian nor an autograd graph is formed.  The returned gradient is
the derivative of the *same discretized objective*, not a continuous-adjoint
approximation.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class TumorObjective:
    """Numerical data for the physical tumor-control objective."""

    T: float
    umax: float
    r: np.ndarray
    phi: np.ndarray
    M: np.ndarray
    beta: np.ndarray
    alpha: np.ndarray
    gamma: float

    @classmethod
    def canonical(
        cls,
        *,
        m: int = 21,
        T: float = 10.0,
        umax: float = 3.0,
        alpha: float = 1.0,
        beta: float = 40.0,
        gamma: float = 8000.0,
        suppression: float = 0.5,
    ) -> "TumorObjective":
        phenotype = np.linspace(0.0, 1.0, m, dtype=np.float64)
        return cls(
            T=float(T),
            umax=float(umax),
            r=2.0 / (1.0 + 3.0 * phenotype**4),
            phi=1.0 / (1.0 + phenotype**2),
            M=np.full(m, suppression, dtype=np.float64),
            beta=np.full(m, beta, dtype=np.float64),
            alpha=np.full(m, alpha, dtype=np.float64),
            gamma=float(gamma),
        )

    @classmethod
    def from_mapping(
        cls,
        params: Mapping[str, object],
        *,
        T: float,
        umax: float,
    ) -> "TumorObjective":
        def vector(name: str) -> np.ndarray:
            value = params[name]
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()  # type: ignore[union-attr]
            result = np.asarray(value, dtype=np.float64).reshape(-1)
            if not result.flags.c_contiguous:
                result = np.ascontiguousarray(result)
            return result

        gamma_value = params["gamma"]
        if hasattr(gamma_value, "detach"):
            gamma_value = gamma_value.detach().cpu().item()  # type: ignore[union-attr]
        result = cls(
            T=float(T),
            umax=float(umax),
            r=vector("r"),
            phi=vector("phi"),
            M=vector("M"),
            beta=vector("beta"),
            alpha=vector("alpha"),
            gamma=float(gamma_value),
        )
        result.validate()
        return result

    def validate(self) -> None:
        arrays = (self.r, self.phi, self.M, self.beta, self.alpha)
        if not arrays or arrays[0].ndim != 1 or arrays[0].size == 0:
            raise ValueError("parameter vectors must be nonempty and one-dimensional")
        if any(array.shape != arrays[0].shape for array in arrays):
            raise ValueError("all parameter vectors must have identical shapes")
        if any(array.dtype != np.float64 for array in arrays):
            raise ValueError("all parameter vectors must use float64")
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("parameter vectors contain non-finite values")
        if not np.isfinite(self.T) or self.T <= 0.0:
            raise ValueError("T must be positive and finite")
        if not np.isfinite(self.gamma):
            raise ValueError("gamma must be finite")


class RK4ZOHDiscreteAdjoint:
    """Reusable exact value-and-gradient evaluator.

    Work arrays are allocated once.  A single instance is therefore intended
    for one optimizer/thread at a time.
    """

    def __init__(
        self,
        problem: TumorObjective,
        intervals: int,
        *,
        substeps: int = 4,
    ) -> None:
        problem.validate()
        if intervals <= 0 or substeps <= 0:
            raise ValueError("intervals and substeps must be positive")
        self.problem = problem
        self.intervals = int(intervals)
        self.substeps = int(substeps)
        self.microsteps = self.intervals * self.substeps
        self.step = problem.T / self.microsteps
        self._stages = np.empty(
            (self.microsteps, 4, problem.r.size), dtype=np.float64
        )
        self._gradient = np.empty(self.intervals, dtype=np.float64)

    def _dynamics(self, state: np.ndarray, control: float) -> np.ndarray:
        problem = self.problem
        growth = (
            problem.r
            - problem.phi * control
            - problem.M * np.log1p(state.mean())
        )
        return growth * state

    def _jacobian_transpose_vector(
        self,
        state: np.ndarray,
        control: float,
        vector: np.ndarray,
    ) -> np.ndarray:
        """Return ``f_N(state,control).T @ vector`` in O(m)."""

        problem = self.problem
        drift = (
            problem.r
            - problem.phi * control
            - problem.M * np.log1p(state.mean())
        )
        coupling = np.dot(vector * problem.M, state) / (
            state.size + state.sum()
        )
        return drift * vector - coupling

    def value_and_gradient(
        self,
        control: np.ndarray,
        initial_state: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Return the exact discrete objective and reduced gradient."""

        control = np.asarray(control, dtype=np.float64)
        initial_state = np.asarray(initial_state, dtype=np.float64)
        if control.shape != (self.intervals,):
            raise ValueError(
                f"expected control shape {(self.intervals,)}, got {control.shape}"
            )
        if initial_state.shape != self.problem.r.shape:
            raise ValueError(
                f"expected state shape {self.problem.r.shape}, "
                f"got {initial_state.shape}"
            )
        if not np.all(np.isfinite(control)) or not np.all(np.isfinite(initial_state)):
            raise ValueError("control or initial state contains non-finite values")

        problem = self.problem
        h = self.step
        stages = self._stages
        state = initial_state.copy()
        running = 0.0

        micro_index = 0
        for interval_index in range(self.intervals):
            u = float(control[interval_index])
            for _ in range(self.substeps):
                x1 = state
                k1 = self._dynamics(x1, u)
                x2 = state + (0.5 * h) * k1
                k2 = self._dynamics(x2, u)
                x3 = state + (0.5 * h) * k2
                k3 = self._dynamics(x3, u)
                x4 = state + h * k3
                k4 = self._dynamics(x4, u)

                stages[micro_index, 0] = x1
                stages[micro_index, 1] = x2
                stages[micro_index, 2] = x3
                stages[micro_index, 3] = x4

                running += (h / 6.0) * (
                    np.dot(problem.beta, x1)
                    + 2.0 * np.dot(problem.beta, x2)
                    + 2.0 * np.dot(problem.beta, x3)
                    + np.dot(problem.beta, x4)
                    + 6.0 * problem.gamma * u
                )
                state = state + (h / 6.0) * (
                    k1 + 2.0 * k2 + 2.0 * k3 + k4
                )
                micro_index += 1

        value = float(running + np.dot(problem.alpha, state))

        gradient = self._gradient
        gradient.fill(0.0)
        costate_next = problem.alpha.copy()

        for micro_index in range(self.microsteps - 1, -1, -1):
            interval_index = micro_index // self.substeps
            u = float(control[interval_index])
            x1, x2, x3, x4 = stages[micro_index]

            a_k4 = (h / 6.0) * costate_next
            a_x4 = (h / 6.0) * problem.beta + self._jacobian_transpose_vector(
                x4, u, a_k4
            )
            a_k3 = (h / 3.0) * costate_next + h * a_x4
            a_x3 = (h / 3.0) * problem.beta + self._jacobian_transpose_vector(
                x3, u, a_k3
            )
            a_k2 = (h / 3.0) * costate_next + (0.5 * h) * a_x3
            a_x2 = (h / 3.0) * problem.beta + self._jacobian_transpose_vector(
                x2, u, a_k2
            )
            a_k1 = (h / 6.0) * costate_next + (0.5 * h) * a_x2

            interval_gradient = h * problem.gamma
            interval_gradient -= np.dot(problem.phi * x1, a_k1)
            interval_gradient -= np.dot(problem.phi * x2, a_k2)
            interval_gradient -= np.dot(problem.phi * x3, a_k3)
            interval_gradient -= np.dot(problem.phi * x4, a_k4)
            gradient[interval_index] += interval_gradient

            costate_next = (
                costate_next
                + (h / 6.0) * problem.beta
                + a_x4
                + a_x3
                + a_x2
                + self._jacobian_transpose_vector(x1, u, a_k1)
            )

        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError("non-finite objective or gradient")
        return value, gradient.copy()

    __call__ = value_and_gradient


def _load_npz_control(path: Path, intervals: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as source:
        control = np.asarray(source["u"], dtype=np.float64).reshape(-1)
    if control.size == intervals + 1:
        control = control[:-1]
    if control.size != intervals:
        raise ValueError(f"{path}: expected {intervals} controls, got {control.size}")
    return control


def benchmark(args: argparse.Namespace) -> None:
    """Compare against the authoritative float64 PyTorch autograd evaluator."""

    import torch

    from evaluate_feedback_section5 import rk4_zoh_open_loop
    from train_paper_pmp_kkt import ProblemConfig, build_params

    cfg = ProblemConfig(
        m=args.m,
        n=args.intervals,
        T=args.T,
        umax=args.umax,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        n0=args.N0,
        m_suppression=args.suppression,
    )
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    problem = TumorObjective.from_mapping(params, T=cfg.T, umax=cfg.umax)
    evaluator = RK4ZOHDiscreteAdjoint(
        problem, cfg.n, substeps=args.substeps
    )

    if args.control:
        control = _load_npz_control(Path(args.control), cfg.n)
    else:
        rng = np.random.default_rng(args.seed)
        control = rng.uniform(0.05, cfg.umax - 0.05, size=cfg.n)
    rng = np.random.default_rng(args.seed + 1)
    initial = cfg.n0 * (1.0 + args.radius * rng.uniform(-1.0, 1.0, cfg.m))

    torch_control = torch.tensor(control, dtype=torch.float64, requires_grad=True)
    torch_initial = torch.tensor(initial[None, :], dtype=torch.float64)

    # Warm both paths once.
    fast_value, fast_gradient = evaluator(control, initial)
    torch_value, _ = rk4_zoh_open_loop(
        torch_control, torch_initial, cfg, params, substeps=args.substeps
    )
    torch_gradient = torch.autograd.grad(torch_value.sum(), torch_control)[0]

    torch_times: list[float] = []
    fast_times: list[float] = []
    for _ in range(args.repeats):
        torch_control = torch.tensor(control, dtype=torch.float64, requires_grad=True)
        start = time.perf_counter()
        torch_value, _ = rk4_zoh_open_loop(
            torch_control, torch_initial, cfg, params, substeps=args.substeps
        )
        torch_gradient = torch.autograd.grad(torch_value.sum(), torch_control)[0]
        torch_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        fast_value, fast_gradient = evaluator(control, initial)
        fast_times.append(time.perf_counter() - start)

    reference_value = float(torch_value.detach())
    reference_gradient = torch_gradient.detach().numpy()
    value_absolute_error = abs(fast_value - reference_value)
    value_relative_error = value_absolute_error / max(abs(reference_value), 1.0)
    gradient_absolute_error = float(
        np.max(np.abs(fast_gradient - reference_gradient))
    )
    gradient_relative_l2_error = float(
        np.linalg.norm(fast_gradient - reference_gradient)
        / max(np.linalg.norm(reference_gradient), np.finfo(np.float64).tiny)
    )
    result = {
        "schema": "rk4-zoh-discrete-adjoint-benchmark-v1",
        "intervals": cfg.n,
        "substeps": args.substeps,
        "phenotypes": cfg.m,
        "objective": fast_value,
        "value_absolute_error": value_absolute_error,
        "value_relative_error": value_relative_error,
        "gradient_max_absolute_error": gradient_absolute_error,
        "gradient_relative_l2_error": gradient_relative_l2_error,
        "torch_autograd_seconds_median": float(np.median(torch_times)),
        "exact_adjoint_seconds_median": float(np.median(fast_times)),
        "speedup": float(np.median(torch_times) / np.median(fast_times)),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    if value_relative_error > args.rtol or gradient_relative_l2_error > args.rtol:
        raise SystemExit(
            "discrete-adjoint validation failed: "
            f"value rel={value_relative_error:.3e}, "
            f"gradient rel-L2={gradient_relative_l2_error:.3e}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--intervals", type=int, default=800)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--m", type=int, default=21)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--umax", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=40.0)
    parser.add_argument("--gamma", type=float, default=8000.0)
    parser.add_argument("--N0", type=float, default=10.0)
    parser.add_argument("--suppression", type=float, default=0.5)
    parser.add_argument("--radius", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rtol", type=float, default=1.0e-9)
    parser.add_argument("--output", type=Path)
    return parser


if __name__ == "__main__":
    benchmark(build_parser().parse_args())
