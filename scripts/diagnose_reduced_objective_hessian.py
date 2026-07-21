#!/usr/bin/env python3
"""Full reduced-objective gradient/Hessian diagnostics for ZOH controls.

The instantaneous tumor Hamiltonian is affine in ``u``, so its partial
derivative ``H_uu`` is zero when the state and costate are held fixed.  That
partial derivative is not the Hessian of the reduced objective.  This script
instead differentiates

    J_hat_h(u_0, ..., u_{n-1}) = J_h(N_h(u), u)

through the complete RK4 state recursion.  It reports the dense full Hessian,
box-constrained first-order residuals, and the Hessian restricted to the free
(critical, under strict complementarity) variables.

Independent checks are included:

* an exact discrete adjoint of the same RK4 one-step map;
* the continuous ZOH adjoint integral computed with segmented DOP853;
* central finite differences of both objective values and gradients.

The final network sample at t=T is deliberately omitted: for a ZOH control it
does not drive any interval and has zero measure in the physical objective.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from scipy.integrate import simpson

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_paper_pmp_kkt import ProblemConfig, build_params, dynamics  # noqa: E402
from tumor_problem import (  # noqa: E402
    NOMINAL_TUMOR_PROBLEM,
    evaluate_zoh_control,
)


DEFAULT_CASES = (
    (
        "original_seed4",
        ROOT / "paper_runs/first_layer_ut_benchmark/transformer_seeds/seed_4/solution.npz",
    ),
    (
        "smooth_w3_seed4",
        ROOT / "paper_runs/smoothness_weight_sweep/w3/seed_4/solution.npz",
    ),
    (
        "direct_n200",
        ROOT / "paper_runs/canonical_results/solutions/direct_time_mesh_n200.npz",
    ),
)


@dataclass(frozen=True)
class Case:
    label: str
    source: Path


def parse_case(text: str) -> Case:
    if "=" not in text:
        raise argparse.ArgumentTypeError("case must have the form LABEL=PATH")
    label, path = text.split("=", 1)
    label = label.strip()
    if not label or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in label):
        raise argparse.ArgumentTypeError("case label may contain only letters, digits, '_' and '-'")
    source = Path(path).expanduser()
    if not source.is_absolute():
        source = (ROOT / source).resolve()
    return Case(label=label, source=source)


def make_config(n: int, T: float) -> ProblemConfig:
    problem = NOMINAL_TUMOR_PROBLEM
    return ProblemConfig(
        T=T,
        n=n,
        m=problem.m,
        umax=problem.umax,
        beta=problem.beta,
        alpha=problem.alpha,
        gamma=problem.gamma,
        n0=problem.n0,
        m_suppression=problem.m_suppression,
    )


def rk4_extended_step(
    y: torch.Tensor,
    u: torch.Tensor,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Advance state and accumulated running cost by one ZOH interval."""

    dt = cfg.T / cfg.n
    N = y[:-1]
    accumulated = y[-1]
    beta = params["beta"]
    gamma = params["gamma"]

    k1_N = dynamics(N, u, params)
    k1_J = (beta * N).sum() + gamma * u

    N2 = torch.clamp(N + 0.5 * dt * k1_N, min=1e-10)
    k2_N = dynamics(N2, u, params)
    k2_J = (beta * N2).sum() + gamma * u

    N3 = torch.clamp(N + 0.5 * dt * k2_N, min=1e-10)
    k3_N = dynamics(N3, u, params)
    k3_J = (beta * N3).sum() + gamma * u

    N4 = torch.clamp(N + dt * k3_N, min=1e-10)
    k4_N = dynamics(N4, u, params)
    k4_J = (beta * N4).sum() + gamma * u

    N_next = torch.clamp(
        N + dt * (k1_N + 2.0 * k2_N + 2.0 * k3_N + k4_N) / 6.0,
        min=1e-10,
    )
    accumulated_next = accumulated + dt * (k1_J + 2.0 * k2_J + 2.0 * k3_J + k4_J) / 6.0
    return torch.cat((N_next, accumulated_next.reshape(1)))


def make_reduced_objective(
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    def reduced_objective(u: torch.Tensor) -> torch.Tensor:
        if u.numel() != cfg.n:
            raise ValueError(f"expected {cfg.n} interval controls, got {u.numel()}")
        y = torch.cat((params["N0"], torch.zeros(1, dtype=u.dtype, device=u.device)))
        for index in range(cfg.n):
            y = rk4_extended_step(y, u[index], cfg, params)
        return (params["alpha"] * y[:-1]).sum() + y[-1]

    return reduced_objective


def discrete_adjoint_gradient(
    u: torch.Tensor,
    cfg: ProblemConfig,
    params: dict[str, torch.Tensor],
) -> np.ndarray:
    """Differentiate the RK4 step map using a separate discrete adjoint."""

    states: list[torch.Tensor] = []
    y = torch.cat((params["N0"], torch.zeros(1, dtype=u.dtype, device=u.device)))
    with torch.no_grad():
        for index in range(cfg.n):
            states.append(y)
            y = rk4_extended_step(y, u[index], cfg, params)

    def one_step(state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        return rk4_extended_step(state, control, cfg, params)

    local_jacobian = torch.func.jacrev(one_step, argnums=(0, 1))
    adjoint = torch.cat((params["alpha"], torch.ones(1, dtype=u.dtype, device=u.device)))
    gradient = torch.empty_like(u)
    for index in range(cfg.n - 1, -1, -1):
        state_jacobian, control_jacobian = local_jacobian(states[index], u[index])
        gradient[index] = torch.dot(control_jacobian, adjoint)
        adjoint = state_jacobian.T @ adjoint
    return gradient.detach().cpu().numpy()


def continuous_adjoint_gradient(
    t: np.ndarray,
    u: np.ndarray,
    *,
    samples_per_interval: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Compute dJ/du_k = integral_{I_k} H_u dt with segmented DOP853."""

    n = len(u)
    result = evaluate_zoh_control(
        t,
        u,
        diagnostic_points=n * samples_per_interval + 1,
        rtol=1e-11,
        atol=1e-13,
    )
    sample_t = np.asarray(result["diagnostic_t"], dtype=np.float64)
    psi = np.asarray(result["diagnostic_psi"], dtype=np.float64)
    gradient = np.empty(n, dtype=np.float64)
    for index in range(n):
        left = index * samples_per_interval
        right = (index + 1) * samples_per_interval + 1
        gradient[index] = simpson(psi[left:right], x=sample_t[left:right])
    return gradient, result


def projected_kkt(
    u: np.ndarray,
    gradient: np.ndarray,
    lower: float,
    upper: float,
    bound_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lower_active = u <= lower + bound_tolerance
    upper_active = u >= upper - bound_tolerance
    free = ~(lower_active | upper_active)
    mapping = u - np.clip(u - gradient, lower, upper)
    signed_stationarity = gradient.copy()
    signed_stationarity[lower_active] = np.minimum(gradient[lower_active], 0.0)
    signed_stationarity[upper_active] = np.maximum(gradient[upper_active], 0.0)
    return mapping, signed_stationarity, free, lower_active | upper_active


def safe_relative(error: float, reference: float) -> float:
    return error / max(abs(reference), 1e-14)


def make_directions(
    u: np.ndarray,
    gradient: np.ndarray,
    t: np.ndarray,
    lower: float,
    upper: float,
) -> list[tuple[str, np.ndarray]]:
    directions: list[tuple[str, np.ndarray]] = []
    seen_indices: set[int] = set()

    def coordinate(label: str, index: int) -> None:
        if index in seen_indices:
            return
        direction = np.zeros_like(u)
        direction[index] = 1.0
        directions.append((label, direction))
        seen_indices.add(index)

    coordinate("coordinate_max_abs_gradient", int(np.argmax(np.abs(gradient))))
    coordinate("coordinate_t_8p75", int(np.argmin(np.abs(t[:-1] - 8.75))))
    coordinate("coordinate_t_8p80", int(np.argmin(np.abs(t[:-1] - 8.80))))

    rng = np.random.default_rng(20260714)
    random_direction = rng.normal(size=len(u))
    random_direction /= np.linalg.norm(random_direction)
    directions.append(("deterministic_random_unit", random_direction))

    projected_step = np.clip(u - gradient, lower, upper) - u
    if np.linalg.norm(projected_step) > 0.0:
        directions.append(("normalized_projected_descent", projected_step / np.linalg.norm(projected_step)))
    return directions


def finite_difference_checks(
    t: np.ndarray,
    u: np.ndarray,
    reduced_objective: Callable[[torch.Tensor], torch.Tensor],
    gradient: np.ndarray,
    hessian: np.ndarray,
    continuous_gradient: np.ndarray,
    directions: list[tuple[str, np.ndarray]],
    epsilon: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def rk4_value(values: np.ndarray) -> float:
        tensor = torch.tensor(values, dtype=torch.float64)
        return float(reduced_objective(tensor).detach().cpu())

    autodiff_gradient = torch.func.grad(reduced_objective)

    def rk4_gradient(values: np.ndarray) -> np.ndarray:
        tensor = torch.tensor(values, dtype=torch.float64)
        return autodiff_gradient(tensor).detach().cpu().numpy()

    def continuous_value(values: np.ndarray) -> float:
        return float(
            evaluate_zoh_control(
                t,
                values,
                include_diagnostics=False,
                rtol=1e-11,
                atol=1e-13,
            )["J"]
        )

    for label, direction in directions:
        plus = u + epsilon * direction
        minus = u - epsilon * direction
        if np.any(plus < 0.0) or np.any(plus > NOMINAL_TUMOR_PROBLEM.umax):
            continue
        if np.any(minus < 0.0) or np.any(minus > NOMINAL_TUMOR_PROBLEM.umax):
            continue

        rk4_fd = (rk4_value(plus) - rk4_value(minus)) / (2.0 * epsilon)
        rk4_exact = float(gradient @ direction)
        rk4_error = abs(rk4_fd - rk4_exact)
        rows.append(
            {
                "direction": label,
                "check": "RK4 objective directional derivative",
                "epsilon": epsilon,
                "reference": rk4_exact,
                "finite_difference": rk4_fd,
                "absolute_error": rk4_error,
                "relative_error": safe_relative(rk4_error, rk4_exact),
            }
        )

        hessian_exact = hessian @ direction
        hessian_fd = (rk4_gradient(plus) - rk4_gradient(minus)) / (2.0 * epsilon)
        hessian_error = float(np.linalg.norm(hessian_fd - hessian_exact))
        hessian_reference = float(np.linalg.norm(hessian_exact))
        rows.append(
            {
                "direction": label,
                "check": "RK4 Hessian-vector product",
                "epsilon": epsilon,
                "reference": hessian_reference,
                "finite_difference": float(np.linalg.norm(hessian_fd)),
                "absolute_error": hessian_error,
                "relative_error": safe_relative(hessian_error, hessian_reference),
            }
        )

        continuous_fd = (continuous_value(plus) - continuous_value(minus)) / (2.0 * epsilon)
        continuous_exact = float(continuous_gradient @ direction)
        continuous_error = abs(continuous_fd - continuous_exact)
        rows.append(
            {
                "direction": label,
                "check": "DOP853 continuous-adjoint directional derivative",
                "epsilon": epsilon,
                "reference": continuous_exact,
                "finite_difference": continuous_fd,
                "absolute_error": continuous_error,
                "relative_error": safe_relative(continuous_error, continuous_exact),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def analyze_case(
    case: Case,
    out_dir: Path,
    *,
    fd_epsilon: float,
    samples_per_interval: int,
    bound_tolerance: float,
    kkt_tolerance: float,
) -> dict[str, object]:
    data = np.load(case.source)
    t = np.asarray(data["t"], dtype=np.float64).reshape(-1)
    stored_u = np.asarray(data["u"], dtype=np.float64).reshape(-1)
    if len(stored_u) == len(t):
        u = stored_u[:-1].copy()
        omitted_terminal_sample = True
    elif len(stored_u) == len(t) - 1:
        u = stored_u.copy()
        omitted_terminal_sample = False
    else:
        raise ValueError(f"{case.source}: incompatible t/u sizes {len(t)} and {len(stored_u)}")
    if len(t) != len(u) + 1 or not np.allclose(np.diff(t), np.diff(t)[0], rtol=1e-12, atol=1e-12):
        raise ValueError(f"{case.source}: expected a uniform interval grid")
    if not np.isclose(t[0], 0.0) or not np.isclose(t[-1], NOMINAL_TUMOR_PROBLEM.T):
        raise ValueError(f"{case.source}: expected grid [0, {NOMINAL_TUMOR_PROBLEM.T}]")

    cfg = make_config(len(u), float(t[-1] - t[0]))
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    reduced_objective = make_reduced_objective(cfg, params)
    u_tensor = torch.tensor(u, dtype=torch.float64)

    gradient_tensor = torch.func.grad(reduced_objective)(u_tensor)
    hessian_tensor = torch.func.hessian(reduced_objective)(u_tensor)
    gradient = gradient_tensor.detach().cpu().numpy()
    hessian_raw = hessian_tensor.detach().cpu().numpy()
    hessian = 0.5 * (hessian_raw + hessian_raw.T)
    rk4_J = float(reduced_objective(u_tensor).detach().cpu())

    discrete_gradient = discrete_adjoint_gradient(u_tensor, cfg, params)
    continuous_gradient, continuous_result = continuous_adjoint_gradient(
        t,
        u,
        samples_per_interval=samples_per_interval,
    )
    continuous_J = float(continuous_result["J"])

    lower = 0.0
    upper = cfg.umax
    projected_mapping, signed_stationarity, free, active = projected_kkt(
        u,
        gradient,
        lower,
        upper,
        bound_tolerance,
    )
    lower_active = u <= lower + bound_tolerance
    upper_active = u >= upper - bound_tolerance
    strongly_active = (lower_active & (gradient > kkt_tolerance)) | (
        upper_active & (gradient < -kkt_tolerance)
    )
    weakly_active = active & ~strongly_active

    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    spectral_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    eigenvalue_tolerance = 1e-8 * spectral_scale
    if np.any(free):
        free_hessian = hessian[np.ix_(free, free)]
        free_eigenvalues, free_eigenvectors = np.linalg.eigh(free_hessian)
    else:
        free_hessian = np.empty((0, 0), dtype=np.float64)
        free_eigenvalues = np.empty(0, dtype=np.float64)
        free_eigenvectors = np.empty((0, 0), dtype=np.float64)

    directions = make_directions(u, gradient, t, lower, upper)
    finite_difference_rows = finite_difference_checks(
        t,
        u,
        reduced_objective,
        gradient,
        hessian,
        continuous_gradient,
        directions,
        fd_epsilon,
    )

    projected_trial_u = np.clip(u - gradient, lower, upper)
    projected_trial_J = float(
        reduced_objective(torch.tensor(projected_trial_u, dtype=torch.float64)).detach().cpu()
    )
    projected_trial_continuous_J = float(
        evaluate_zoh_control(t, projected_trial_u, include_diagnostics=False)["J"]
    )

    symmetry_error = float(np.linalg.norm(hessian_raw - hessian_raw.T))
    symmetry_relative = symmetry_error / max(float(np.linalg.norm(hessian_raw)), 1e-14)
    discrete_error = discrete_gradient - gradient
    continuous_error = continuous_gradient - gradient
    kkt_pass = float(np.max(np.abs(projected_mapping))) <= kkt_tolerance
    free_psd = bool(
        len(free_eigenvalues) == 0
        or float(np.min(free_eigenvalues)) >= -eigenvalue_tolerance
    )
    full_psd = bool(float(np.min(eigenvalues)) >= -eigenvalue_tolerance)
    second_order_applicable = bool(kkt_pass)
    critical_subspace_equals_free = bool(kkt_pass and np.all(~weakly_active))

    summary: dict[str, object] = {
        "label": case.label,
        "source": str(case.source),
        "n_intervals": cfg.n,
        "dt": cfg.T / cfg.n,
        "stored_control_samples": len(stored_u),
        "omitted_terminal_network_sample": omitted_terminal_sample,
        "rk4_reduced_objective": rk4_J,
        "continuous_DOP853_objective": continuous_J,
        "rk4_minus_continuous_objective": rk4_J - continuous_J,
        "gradient_l2": float(np.linalg.norm(gradient)),
        "gradient_linf": float(np.max(np.abs(gradient))),
        "gradient_min": float(np.min(gradient)),
        "gradient_max": float(np.max(gradient)),
        "projected_kkt_l2": float(np.linalg.norm(projected_mapping)),
        "projected_kkt_linf": float(np.max(np.abs(projected_mapping))),
        "signed_box_stationarity_linf": float(np.max(np.abs(signed_stationarity))),
        "kkt_tolerance": kkt_tolerance,
        "kkt_pass_at_tolerance": kkt_pass,
        "free_variables": int(np.sum(free)),
        "lower_active_variables": int(np.sum(lower_active)),
        "upper_active_variables": int(np.sum(upper_active)),
        "strongly_active_variables": int(np.sum(strongly_active)),
        "weakly_active_variables": int(np.sum(weakly_active)),
        "hessian_symmetry_relative_error": symmetry_relative,
        "hessian_min_eigenvalue_full": float(np.min(eigenvalues)),
        "hessian_max_eigenvalue_full": float(np.max(eigenvalues)),
        "hessian_negative_eigenvalues_full": int(np.sum(eigenvalues < -eigenvalue_tolerance)),
        "hessian_eigenvalue_tolerance": eigenvalue_tolerance,
        "hessian_full_psd_at_tolerance": full_psd,
        "hessian_min_eigenvalue_free": float(np.min(free_eigenvalues)) if len(free_eigenvalues) else None,
        "hessian_max_eigenvalue_free": float(np.max(free_eigenvalues)) if len(free_eigenvalues) else None,
        "hessian_negative_eigenvalues_free": int(np.sum(free_eigenvalues < -eigenvalue_tolerance)),
        "hessian_free_psd_at_tolerance": free_psd,
        "second_order_test_applicable_after_first_order": second_order_applicable,
        "critical_subspace_equals_free_subspace": critical_subspace_equals_free,
        "autodiff_vs_discrete_adjoint_linf": float(np.max(np.abs(discrete_error))),
        "autodiff_vs_discrete_adjoint_l2": float(np.linalg.norm(discrete_error)),
        "rk4_vs_continuous_adjoint_linf": float(np.max(np.abs(continuous_error))),
        "rk4_vs_continuous_adjoint_l2": float(np.linalg.norm(continuous_error)),
        "projected_gradient_trial_rk4_objective": projected_trial_J,
        "projected_gradient_trial_continuous_objective": projected_trial_continuous_J,
        "projected_gradient_trial_rk4_change": projected_trial_J - rk4_J,
        "projected_gradient_directional_derivative": float(gradient @ (projected_trial_u - u)),
        "instantaneous_partial_H_uu": 0.0,
        "interpretation": (
            "first-order box KKT fails; a second-order optimality claim is not applicable"
            if not kkt_pass
            else (
                "first-order box KKT passes numerically and the Hessian is PSD on the critical/free subspace"
                if free_psd and critical_subspace_equals_free
                else "first-order box KKT passes numerically; inspect weakly active critical directions"
            )
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=False)
    active_status = np.full(cfg.n, "free", dtype=object)
    active_status[lower_active] = "lower_active"
    active_status[upper_active] = "upper_active"
    gradient_rows = [
        {
            "k": index,
            "t_left": t[index],
            "t_right": t[index + 1],
            "u_k": u[index],
            "autodiff_full_gradient": gradient[index],
            "discrete_adjoint_gradient": discrete_gradient[index],
            "continuous_adjoint_gradient": continuous_gradient[index],
            "projected_kkt_mapping": projected_mapping[index],
            "signed_box_stationarity": signed_stationarity[index],
            "active_status": active_status[index],
        }
        for index in range(cfg.n)
    ]
    write_csv(out_dir / "gradient_by_interval.csv", gradient_rows)
    write_csv(out_dir / "finite_difference_checks.csv", finite_difference_rows)
    eigenvalue_rows = [
        {"index": index, "full_hessian_eigenvalue": value}
        for index, value in enumerate(eigenvalues)
    ]
    write_csv(out_dir / "hessian_eigenvalues.csv", eigenvalue_rows)
    free_eigenvalue_rows = [
        {"index": index, "free_hessian_eigenvalue": value}
        for index, value in enumerate(free_eigenvalues)
    ]
    write_csv(out_dir / "free_hessian_eigenvalues.csv", free_eigenvalue_rows)
    np.savez_compressed(
        out_dir / "full_derivatives.npz",
        t=t,
        interval_u=u,
        full_gradient=gradient,
        discrete_adjoint_gradient=discrete_gradient,
        continuous_adjoint_gradient=continuous_gradient,
        projected_kkt_mapping=projected_mapping,
        signed_box_stationarity=signed_stationarity,
        full_hessian=hessian,
        full_hessian_eigenvalues=eigenvalues,
        full_hessian_eigenvectors=eigenvectors,
        free_indices=np.flatnonzero(free),
        free_hessian=free_hessian,
        free_hessian_eigenvalues=free_eigenvalues,
        free_hessian_eigenvectors=free_eigenvectors,
    )
    (out_dir / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def format_scientific(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6e}"


def write_report(out_dir: Path, summaries: list[dict[str, object]]) -> None:
    interval_counts = sorted({int(summary["n_intervals"]) for summary in summaries})
    if len(interval_counts) == 1:
        interval_description = (
            f"The physical objective is discretized on {interval_counts[0]} "
            "zero-order-hold intervals"
        )
        matrix_description = (
            f"Each case directory contains the complete "
            f"{interval_counts[0]}x{interval_counts[0]} matrix"
        )
    else:
        joined_counts = ", ".join(str(count) for count in interval_counts)
        interval_description = (
            "Each physical objective is discretized on the number of "
            f"zero-order-hold intervals listed in the table ({joined_counts})"
        )
        matrix_description = (
            "Each case directory contains its complete n-by-n matrix, with n "
            "given by the interval count in the table"
        )
    lines = [
        "# Reduced-objective full gradient and Hessian diagnostics",
        "",
        interval_description + " with the same differentiable RK4 map used by the direct-objective reference. For `N_{k+1}=Phi_k(N_k,u_k)`, the script differentiates `J_hat_h(u)=J_h(N_h(u),u)` through every state update. Thus `g_i=dJ_hat_h/du_i` and `H_ij=d^2J_hat_h/(du_i du_j)` include the full dependence `N=N(u)`. The final stored network output at `t=T` is omitted because it drives no ZOH interval.",
        "",
        "The instantaneous partial derivative `partial^2 H / partial u^2` is zero because the Hamiltonian is affine in `u`; it is not the reduced Hessian reported here. The reduced Hessian is generally dense and nonzero.",
        "",
        "Writing `S=dN/du` for the sensitivity of the stacked state trajectory, the chain rule contains `J_uu + J_uN S + S^T J_Nu + S^T J_NN S + sum_a J_Na d^2N_a/du^2`; the last four terms are exactly what is lost if `N` is incorrectly held fixed. The separate discrete-adjoint check uses `p_n=(alpha,1)`, `p_k=A_k^T p_{k+1}`, and `g_k=b_k^T p_{k+1}` for `A_k=dPhi_k/dy_k` and `b_k=dPhi_k/du_k` on the state-plus-running-cost map.",
        "",
        "For box constraints, the first-order residual is the projected-gradient mapping `G(u)=u-P_[0,3](u-g)`. A Hessian test is an optimality condition only after this first-order residual is small. Under strict complementarity, active coordinates are fixed in the critical cone and the relevant Hessian is the free-variable principal block. When every coordinate is interior, the free and full spaces coincide.",
        "",
        "| case | n | J (DOP853) | ||G||_inf | KKT tolerance | KKT pass | lambda_min(full) | lambda_min(free) | free / active | conclusion |",
        "|---|---:|---:|---:|---:|:---:|---:|---:|---:|---|",
    ]
    for summary in summaries:
        lines.append(
            "| {label} | {n} | {J:.9f} | {kkt:.3e} | {tolerance:.1e} | {passed} | {full_min} | {free_min} | {free}/{active} | {conclusion} |".format(
                label=summary["label"],
                n=summary["n_intervals"],
                J=float(summary["continuous_DOP853_objective"]),
                kkt=float(summary["projected_kkt_linf"]),
                tolerance=float(summary["kkt_tolerance"]),
                passed="yes" if summary["kkt_pass_at_tolerance"] else "no",
                full_min=format_scientific(summary["hessian_min_eigenvalue_full"]),
                free_min=format_scientific(summary["hessian_min_eigenvalue_free"]),
                free=summary["free_variables"],
                active=int(summary["lower_active_variables"]) + int(summary["upper_active_variables"]),
                conclusion=summary["interpretation"],
            )
        )
    lines.extend(
        [
            "",
            "## Verification",
            "",
            matrix_description + " and three gradient calculations. The RK4 autodiff gradient is checked against a separately assembled discrete adjoint. A segmented DOP853 state/costate solve supplies the continuous ZOH adjoint integral, and central finite differences check directional gradients and Hessian-vector products.",
            "",
            "## Scope of the conclusion",
            "",
            "Passing these tests supports local optimality only for the stated finite-dimensional ZOH/RK4 transcription. It is not a proof of a global optimum of the continuous-time problem. Conversely, a failed projected-gradient test is already enough to rule out a discrete local optimum at an interior control vector, regardless of the Hessian.",
            "",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def make_comparison_figure(out_dir: Path, summaries: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    styles = {
        "refined_time_only": {
            "label": "refined time-only Transformer",
            "color": "#D97706",
            "linewidth": 1.55,
            "marker": None,
            "linestyle": "-.",
        },
        "smooth_w3_seed4": {
            "label": "PMP/KKT Transformer",
            "color": "#D97706",
            "linewidth": 1.55,
            "marker": None,
            "linestyle": "-.",
        },
        "direct_n200": {
            "label": r"direct time mesh ($n=200$)",
            "color": "#2A7F62",
            "linewidth": 2.0,
            "marker": "o",
            "linestyle": "-",
        },
    }

    available_transformers = {
        str(summary["label"])
        for summary in summaries
        if str(summary["label"]) in {"refined_time_only", "smooth_w3_seed4"}
    }
    if len(available_transformers) != 1:
        raise ValueError(
            "comparison figure requires exactly one Transformer case: "
            "refined_time_only or smooth_w3_seed4"
        )
    figure_labels = {available_transformers.pop(), "direct_n200"}
    figure_summaries = [summary for summary in summaries if str(summary["label"]) in figure_labels]
    available_labels = {str(summary["label"]) for summary in figure_summaries}
    if available_labels != figure_labels:
        missing = sorted(figure_labels - available_labels)
        raise ValueError(f"comparison figure is missing required case(s): {missing}")

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.35), dpi=220)
    left, right = axes
    direct_t: np.ndarray | None = None
    direct_mapping: np.ndarray | None = None

    for summary in figure_summaries:
        label = str(summary["label"])
        arrays = np.load(out_dir / label / "full_derivatives.npz")
        style = styles.get(
            label,
            {
                "label": label,
                "color": "#B4473A",
                "linewidth": 1.5,
                "marker": None,
                "linestyle": "-",
            },
        )
        time = arrays["t"][:-1]
        mapping = arrays["projected_kkt_mapping"]
        left.plot(
            time,
            mapping,
            color=style["color"],
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=10 if style["marker"] else None,
            markersize=3.3,
            markerfacecolor="white" if style["marker"] else None,
            markeredgewidth=0.9,
            label=style["label"],
            zorder=4 if label == "direct_n200" else 2,
        )
        if label == "direct_n200":
            direct_t = time
            direct_mapping = mapping

        eigenvalues = np.sort(arrays["free_hessian_eigenvalues"])
        normalized_rank = np.linspace(0.0, 1.0, len(eigenvalues))
        right.plot(
            normalized_rank,
            eigenvalues,
            color=style["color"],
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=max(len(eigenvalues) // 18, 1) if style["marker"] else None,
            markersize=3.1,
            markerfacecolor="white" if style["marker"] else None,
            markeredgewidth=0.9,
            label=f"{style['label']}  ($\\lambda_{{\\min}}={eigenvalues[0]:.2e}$)",
            zorder=4 if label == "direct_n200" else 2,
        )

    zero_color = "#111827"
    grid_color = "#E5E7EB"
    left.axhline(0.0, color=zero_color, linewidth=0.8, zorder=1)
    left.set_xlabel(r"interval start time $t_k$")
    left.set_ylabel(r"projected-gradient mapping $G_k(u)$")
    left.set_title("(a) Box-constrained first-order condition", loc="left", fontsize=10.5)
    left.grid(True, color=grid_color, linewidth=0.55)
    left.legend(frameon=False, fontsize=7.8, loc="lower left")
    left.margins(x=0.01)

    if direct_t is not None and direct_mapping is not None:
        inset = left.inset_axes([0.57, 0.56, 0.40, 0.35])
        inset.plot(
            direct_t,
            direct_mapping,
            color=styles["direct_n200"]["color"],
            linewidth=1.35,
            marker="o",
            markevery=10,
            markersize=2.4,
            markerfacecolor="white",
            markeredgewidth=0.7,
        )
        inset.axhline(0.0, color=zero_color, linewidth=0.6)
        limit = max(6.0e-5, 1.15 * float(np.max(np.abs(direct_mapping))))
        inset.set_ylim(-limit, limit)
        inset.set_xlim(float(direct_t[0]), float(direct_t[-1]))
        inset.set_title("direct projected gradient (zoom)", fontsize=7.2, pad=2)
        inset.tick_params(axis="both", labelsize=6.2, length=2)
        inset.grid(True, color=grid_color, linewidth=0.45)

    right.axhline(0.0, color=zero_color, linewidth=0.8, zorder=1)
    right.set_yscale("symlog", linthresh=2.0e-5, linscale=0.85)
    right.set_xlabel("normalized sorted-eigenvalue rank")
    right.set_ylabel("Hessian eigenvalue on the tested subspace")
    right.set_title(
        "(b) Reduced-Hessian spectrum",
        loc="left",
        fontsize=10.5,
    )
    right.grid(True, which="both", color=grid_color, linewidth=0.55)
    right.legend(frameon=False, fontsize=7.4, loc="upper left")
    right.margins(x=0.01)

    minimum_inset = right.inset_axes([0.58, 0.12, 0.38, 0.30])
    minimum_labels: list[str] = []
    minimum_values: list[float] = []
    minimum_colors: list[str] = []
    for summary in figure_summaries:
        label = str(summary["label"])
        minimum_labels.append(
            {
                "refined_time_only": "refined\nTransformer",
                "smooth_w3_seed4": "Transformer",
                "direct_n200": "direct",
            }.get(
                label, label
            )
        )
        minimum_values.append(float(summary["hessian_min_eigenvalue_free"]))
        minimum_colors.append(str(styles.get(label, {"color": "#B4473A"})["color"]))
    minimum_inset.axhline(0.0, color=zero_color, linewidth=0.6)
    minimum_inset.scatter(
        np.arange(len(minimum_values)),
        minimum_values,
        color=minimum_colors,
        s=23,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    minimum_inset.set_xticks(np.arange(len(minimum_values)), minimum_labels)
    minimum_inset.set_ylim(-6.2e-5, 6.2e-5)
    minimum_inset.set_title(r"minimum tested eigenvalue", fontsize=7.2, pad=2)
    minimum_inset.tick_params(axis="both", labelsize=6.2, length=2)
    minimum_inset.grid(True, axis="y", color=grid_color, linewidth=0.45)

    figure.suptitle("First- and second-order checks for the reduced objective", fontsize=11.5, y=1.005)
    figure.tight_layout()
    figure.savefig(out_dir / "reduced_gradient_hessian_comparison.png", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        type=parse_case,
        help="LABEL=PATH; may be repeated. Defaults to original, smooth-w3, and direct-n200.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "paper_runs/reduced_objective_full_hessian",
    )
    parser.add_argument("--fd-epsilon", type=float, default=1e-4)
    parser.add_argument("--samples-per-interval", type=int, default=40)
    parser.add_argument("--bound-tolerance", type=float, default=1e-8)
    parser.add_argument("--kkt-tolerance", type=float, default=1e-4)
    parser.add_argument("--force", action="store_true", help="replace only the requested diagnostic output directory")
    args = parser.parse_args()

    cases = args.case or [Case(label=label, source=path) for label, path in DEFAULT_CASES]
    for case in cases:
        if not case.source.exists():
            raise FileNotFoundError(case.source)

    out_dir = args.out_dir.expanduser().resolve()
    if out_dir.exists():
        if not args.force:
            raise FileExistsError(f"diagnostic output already exists: {out_dir}; use --force to replace it")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    summaries: list[dict[str, object]] = []
    for case in cases:
        print(f"analyzing {case.label}: {case.source}", flush=True)
        summaries.append(
            analyze_case(
                case,
                out_dir / case.label,
                fd_epsilon=args.fd_epsilon,
                samples_per_interval=args.samples_per_interval,
                bound_tolerance=args.bound_tolerance,
                kkt_tolerance=args.kkt_tolerance,
            )
        )

    comparison_fields = [
        "label",
        "source",
        "rk4_reduced_objective",
        "continuous_DOP853_objective",
        "gradient_l2",
        "gradient_linf",
        "projected_kkt_l2",
        "projected_kkt_linf",
        "kkt_pass_at_tolerance",
        "free_variables",
        "lower_active_variables",
        "upper_active_variables",
        "hessian_min_eigenvalue_full",
        "hessian_max_eigenvalue_full",
        "hessian_negative_eigenvalues_full",
        "hessian_min_eigenvalue_free",
        "hessian_max_eigenvalue_free",
        "hessian_negative_eigenvalues_free",
        "hessian_free_psd_at_tolerance",
        "second_order_test_applicable_after_first_order",
        "autodiff_vs_discrete_adjoint_linf",
        "rk4_vs_continuous_adjoint_linf",
        "projected_gradient_trial_continuous_objective",
        "interpretation",
    ]
    comparison_rows = [{field: summary.get(field) for field in comparison_fields} for summary in summaries]
    write_csv(out_dir / "comparison.csv", comparison_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(json_safe({"cases": summaries}), indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(out_dir, summaries)
    make_comparison_figure(out_dir, summaries)
    print(json.dumps(json_safe({"cases": summaries}), indent=2), flush=True)


if __name__ == "__main__":
    main()
