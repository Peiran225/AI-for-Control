#!/usr/bin/env python3
"""Build one native-n=800, three-case diagnostic bundle for teacher review.

This generator is intentionally separate from the existing report pipeline.
It evaluates the selected sharp time-only Transformer and two learned
state-feedback policies on one common nominal initial condition.  A feedback
checkpoint may be native to the requested mesh or deployed at a different
resolution; in either case, its stored nominal-reference trajectory is mapped
to the requested node grid before the closed-loop rollout.

For every realized physical control vector ``u`` the script reports

    F_h(u) = J_h(N_h(u), u),
    g      = d F_h / d u,
    R      = d^2 F_h / d u^2,

where the complete RK4 recursion ``N_h=N_h(u)`` is differentiated.  It also
computes the continuous instantaneous Hamiltonian H(t) on a segmented DOP853
state/costate trajectory.  H(t) and F_h(u) are kept as distinct quantities.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for search_path in (ROOT, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from boundary_control import BoundaryProjectedControl  # noqa: E402
from refine_time_only_singular_plateau import build_model  # noqa: E402
from train_teacher_free_resolution_curriculum import (  # noqa: E402
    FixedBoxProjection,
)
from diagnose_reduced_objective_hessian import (  # noqa: E402
    make_reduced_objective,
    projected_kkt,
)
from evaluate_feedback_section5 import (  # noqa: E402
    configured_state_mode,
    load_feedback_checkpoint,
)
from generate_three_case_full_derivative_results import (  # noqa: E402
    instantaneous_hamiltonian,
)
from generate_three_case_hamiltonian_results import (  # noqa: E402
    RealizedCase,
    assert_common_problem,
    evaluate_case,
    problem_from_config,
    solve_state_interval,
)
from train_paper_pmp_kkt import ProblemConfig, build_params  # noqa: E402


DEFAULT_TIME = (
    ROOT
    / "outputs/teacher_free_n800_strict_20260720/"
    "network_lbfgs_after_learn_tau/selected_checkpoint.pt"
)
DEFAULT_CF = (
    ROOT
    / "outputs/feedback_teacher_followup_20260719/"
    "cf_seed1/train/best_feedback_section5.pt"
)
DEFAULT_DER = (
    ROOT
    / "outputs/feedback_case2_alpha1_guardrail_20260720/"
    "w50_guard_smooth/train/best_feedback_section5_full_gradient.pt"
)
DEFAULT_OUT = ROOT / "outputs/teacher_three_case_time_only_transformer_results_20260720"


def resolve(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def cfg_with_n(cfg: ProblemConfig, n: int) -> ProblemConfig:
    values = asdict(cfg)
    values["n"] = int(n)
    return ProblemConfig(**values)


def interpolate_nominal_reference(model: torch.nn.Module, n: int) -> dict[str, Any]:
    stored = model.nominal_reference.detach().cpu().numpy()
    if stored.ndim != 2:
        raise ValueError("feedback checkpoint has no two-dimensional nominal reference")
    source_nodes = stored.shape[0]
    source_t = np.linspace(0.0, 1.0, source_nodes, dtype=np.float64)
    target_t = np.linspace(0.0, 1.0, n + 1, dtype=np.float64)
    interpolated = np.column_stack(
        [np.interp(target_t, source_t, stored[:, j]) for j in range(stored.shape[1])]
    )
    model.set_nominal_reference(torch.tensor(interpolated, dtype=torch.float64))
    return {
        "source_nominal_reference_nodes": source_nodes,
        "deployed_nominal_reference_nodes": n + 1,
        "nominal_reference_interpolation": "componentwise linear in normalized time",
    }


def load_time_case(path: Path, n: int) -> tuple[RealizedCase, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    source_cfg = ProblemConfig(**payload["problem"])
    wrapper = dict(payload.get("wrapper", {}))
    wrapper_class = str(wrapper.get("class", ""))
    if wrapper_class not in {"BoundaryProjectedControl", "FixedBoxProjection"}:
        raise ValueError(
            f"unsupported selected time-only wrapper: {wrapper_class!r}"
        )
    base = build_model(dict(payload["base_model_args"]), source_cfg).to(
        dtype=torch.float64
    )
    if wrapper_class == "BoundaryProjectedControl":
        model = BoundaryProjectedControl(
            base,
            umax=source_cfg.umax,
            scale_mode="fixed",
            initial_scale=float(wrapper["scale"]),
        ).to(dtype=torch.float64)
    else:
        model = FixedBoxProjection(
            base,
            source_cfg.umax,
            float(wrapper["scale"]),
            temperature=float(wrapper.get("temperature", 1.0)),
            learn_temperature=bool(wrapper.get("learn_temperature", False)),
        ).to(dtype=torch.float64)
    model.load_state_dict(payload["model_state"])
    if source_cfg.n != n:
        raise ValueError(f"time-only checkpoint has n={source_cfg.n}, expected n={n}")
    model.eval()
    grid = torch.linspace(0.0, 1.0, n + 1, dtype=torch.float64)
    with torch.inference_mode():
        controls_with_terminal = model(grid).detach().cpu().numpy().astype(np.float64)
    controls = controls_with_terminal[:n].copy()
    case = RealizedCase(
        case_id="time_only",
        label=r"Transformer $u(t)$",
        checkpoint=path,
        cfg=source_cfg,
        breakpoints=np.linspace(0.0, source_cfg.T, n + 1, dtype=np.float64),
        controls=controls,
    )
    teacher_free = bool(payload.get("teacher_free", False))
    training_description = (
        "PMP/KKT optimality-gap training followed by projected-gradient "
        "residual refinement"
        if teacher_free
        else (
            "baseline-guided sharpness training followed by projected "
            "full-gradient refinement"
        )
    )
    return case, {
        "source_training_n": int(source_cfg.n),
        "deployed_n": n,
        "deployment": "native checkpoint grid",
        "training_description": training_description,
        "checkpoint_teacher_free": teacher_free,
        "wrapper_class": wrapper_class,
        "objective_value_used_as_loss_or_selection": payload.get(
            "objective_value_used_as_loss_or_selection"
        ),
        "direct_or_manual_solution_used": payload.get(
            "direct_or_manual_solution_used"
        ),
        "switching_time_or_mask_used": payload.get("switching_time_or_mask_used"),
    }


def realize_feedback_n800(
    path: Path,
    expected_option: str,
    case_id: str,
    label: str,
    initial_state: np.ndarray,
    n: int,
    *,
    rtol: float,
    atol: float,
) -> tuple[RealizedCase, dict[str, Any]]:
    model, source_cfg, checkpoint_args = load_feedback_checkpoint(path)
    actual_option = str(getattr(checkpoint_args, "option", ""))
    if actual_option != expected_option:
        raise ValueError(f"{path}: option={actual_option!r}, expected {expected_option!r}")
    state_mode = configured_state_mode(checkpoint_args)
    if state_mode != "feedback":
        raise ValueError(f"{path}: state_mode={state_mode!r}, expected 'feedback'")
    cfg = cfg_with_n(source_cfg, n)
    problem = problem_from_config(cfg)
    reference_audit = interpolate_nominal_reference(model, n)
    model.eval()

    breakpoints = np.linspace(0.0, cfg.T, n + 1, dtype=np.float64)
    normalized_grid = torch.linspace(0.0, 1.0, n + 1, dtype=torch.float64)
    with torch.inference_mode():
        base_logits = model.time_logits(normalized_grid)[:n]

    state = np.asarray(initial_state, dtype=np.float64).copy()
    controls = np.empty(n, dtype=np.float64)
    for index in range(n):
        normalized_time = torch.tensor([index / n], dtype=torch.float64)
        state_tensor = torch.tensor(state, dtype=torch.float64).unsqueeze(0)
        with torch.inference_mode():
            action = model.interval_action(
                base_logits[index],
                normalized_time,
                state_tensor,
                state_mode="feedback",
            )
        control = float(action.item())
        if not -1.0e-10 <= control <= cfg.umax + 1.0e-10:
            raise RuntimeError(f"{case_id}: policy action outside [0,{cfg.umax}]")
        controls[index] = np.clip(control, 0.0, cfg.umax)
        solution = solve_state_interval(
            float(breakpoints[index]),
            float(breakpoints[index + 1]),
            state,
            controls[index],
            problem,
            rtol=rtol,
            atol=atol,
            dense_output=False,
        )
        state = np.asarray(solution.y[:, -1], dtype=np.float64)

    case = RealizedCase(
        case_id=case_id,
        label=label,
        checkpoint=path,
        cfg=cfg,
        breakpoints=breakpoints,
        controls=controls,
        feedback_model=model,
    )
    return case, {
        "source_training_n": int(source_cfg.n),
        "deployed_n": n,
        "deployment": (
            "same learned continuous time/state policy evaluated on an 801-node "
            "Transformer sequence and closed at 800 ZOH intervals"
        ),
        "state_mode": state_mode,
        "state_feature_mode": str(getattr(checkpoint_args, "state_feature_mode", "")),
        "center_state_correction": bool(
            getattr(checkpoint_args, "center_state_correction", False)
        ),
        **reference_audit,
    }


def evaluate_reduced_derivatives(
    case: RealizedCase,
    initial_state: np.ndarray,
    bound_tolerance: float,
    kkt_tolerance: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cfg = case.cfg
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    params["N0"] = torch.tensor(initial_state, dtype=torch.float64)
    reduced = make_reduced_objective(cfg, params)
    control = torch.tensor(case.controls, dtype=torch.float64)

    started = time.perf_counter()
    gradient_tensor = torch.func.grad(reduced)(control)
    gradient_seconds = time.perf_counter() - started
    started = time.perf_counter()
    hessian_raw_tensor = torch.func.hessian(reduced)(control)
    hessian_seconds = time.perf_counter() - started
    reduced_value = float(reduced(control).detach())
    gradient = gradient_tensor.detach().cpu().numpy()
    hessian_raw = hessian_raw_tensor.detach().cpu().numpy()
    hessian = 0.5 * (hessian_raw + hessian_raw.T)
    projected, signed, free, active = projected_kkt(
        case.controls, gradient, 0.0, cfg.umax, bound_tolerance
    )
    lower_active = case.controls <= bound_tolerance
    upper_active = case.controls >= cfg.umax - bound_tolerance
    strongly_active = (lower_active & (gradient > kkt_tolerance)) | (
        upper_active & (gradient < -kkt_tolerance)
    )
    weakly_active = active & ~strongly_active
    critical_subspace = free | weakly_active
    eigenvalues = np.linalg.eigvalsh(hessian)
    free_indices = np.flatnonzero(free)
    if free_indices.size:
        free_hessian = hessian[np.ix_(free_indices, free_indices)]
        free_eigenvalues = np.linalg.eigvalsh(free_hessian)
    else:
        free_eigenvalues = np.empty(0, dtype=np.float64)
    critical_subspace_indices = np.flatnonzero(critical_subspace)
    if critical_subspace_indices.size:
        critical_subspace_hessian = hessian[
            np.ix_(critical_subspace_indices, critical_subspace_indices)
        ]
        critical_subspace_eigenvalues = np.linalg.eigvalsh(
            critical_subspace_hessian
        )
    else:
        critical_subspace_eigenvalues = np.empty(0, dtype=np.float64)
    projected_linf = float(np.max(np.abs(projected)))
    summary = {
        "reduced_objective_F_h": reduced_value,
        "full_gradient_l2": float(np.linalg.norm(gradient)),
        "full_gradient_linf": float(np.max(np.abs(gradient))),
        "projected_gradient_l2": float(np.linalg.norm(projected)),
        "projected_gradient_linf": projected_linf,
        "signed_box_stationarity_linf": float(np.max(np.abs(signed))),
        "kkt_tolerance": kkt_tolerance,
        "first_order_pass": projected_linf <= kkt_tolerance,
        "free_variables": int(np.sum(free)),
        "active_variables": int(np.sum(active)),
        "strongly_active_variables": int(np.sum(strongly_active)),
        "weakly_active_variables": int(np.sum(weakly_active)),
        "tested_critical_subspace_variables": int(np.sum(critical_subspace)),
        "hessian_diagonal_min": float(np.diag(hessian).min()),
        "hessian_diagonal_max": float(np.diag(hessian).max()),
        "hessian_min_eigenvalue_full": float(eigenvalues.min()),
        "hessian_max_eigenvalue_full": float(eigenvalues.max()),
        "hessian_negative_eigenvalues_full": int(np.sum(eigenvalues < -1.0e-8)),
        "hessian_min_eigenvalue_free": (
            float(free_eigenvalues.min()) if free_eigenvalues.size else None
        ),
        "hessian_min_eigenvalue_tested_critical_subspace": (
            float(critical_subspace_eigenvalues.min())
            if critical_subspace_eigenvalues.size
            else None
        ),
        "hessian_max_eigenvalue_tested_critical_subspace": (
            float(critical_subspace_eigenvalues.max())
            if critical_subspace_eigenvalues.size
            else None
        ),
        "hessian_negative_eigenvalues_tested_critical_subspace": int(
            np.sum(critical_subspace_eigenvalues < -1.0e-8)
        ),
        "hessian_symmetry_relative_error": float(
            np.linalg.norm(hessian_raw - hessian_raw.T)
            / max(np.linalg.norm(hessian_raw), 1.0e-14)
        ),
        "gradient_wall_seconds": gradient_seconds,
        "dense_hessian_wall_seconds": hessian_seconds,
    }
    arrays = {
        "full_gradient": gradient,
        "projected_gradient": projected,
        "full_hessian": hessian,
        "hessian_diagonal": np.diag(hessian).copy(),
        "hessian_eigenvalues": eigenvalues,
        "free_indices": free_indices,
        "free_hessian_eigenvalues": free_eigenvalues,
        "weakly_active_indices": np.flatnonzero(weakly_active),
        "tested_critical_subspace_indices": critical_subspace_indices,
        "tested_critical_subspace_hessian_eigenvalues": (
            critical_subspace_eigenvalues
        ),
    }
    return summary, arrays


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fields = list(summaries[0].keys())
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time_checkpoint", type=Path, default=DEFAULT_TIME)
    parser.add_argument("--cf_checkpoint", type=Path, default=DEFAULT_CF)
    parser.add_argument("--der_checkpoint", type=Path, default=DEFAULT_DER)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n", type=int, default=800)
    parser.add_argument("--diagnostic_points", type=int, default=4001)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--bound_tolerance", type=float, default=1.0e-6)
    parser.add_argument("--kkt_tolerance", type=float, default=1.0e-4)
    args = parser.parse_args()

    time_checkpoint = resolve(args.time_checkpoint)
    cf_checkpoint = resolve(args.cf_checkpoint)
    der_checkpoint = resolve(args.der_checkpoint)
    out_dir = resolve(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    time_case, time_audit = load_time_case(time_checkpoint, args.n)
    problem = assert_common_problem([time_case])
    initial_state = np.full(problem.m, problem.n0, dtype=np.float64)
    cf_case, cf_audit = realize_feedback_n800(
        cf_checkpoint,
        "cf",
        "feedback_cf",
        r"Case 1 $u(N,t)$",
        initial_state,
        args.n,
        rtol=args.rtol,
        atol=args.atol,
    )
    der_case, der_audit = realize_feedback_n800(
        der_checkpoint,
        "der",
        "feedback_der",
        r"Case 2 $u(N,t)$",
        initial_state,
        args.n,
        rtol=args.rtol,
        atol=args.atol,
    )
    cases = [time_case, cf_case, der_case]
    problem = assert_common_problem(cases)
    shared_time = np.linspace(0.0, problem.T, args.diagnostic_points)
    deployment_audits = {
        "time_only": time_audit,
        "feedback_cf": cf_audit,
        "feedback_der": der_audit,
    }

    summaries: list[dict[str, Any]] = []
    for case in cases:
        print(f"[{case.case_id}] continuous trajectory and H(t)", flush=True)
        evaluated = evaluate_case(
            case,
            shared_time,
            initial_state,
            problem,
            rtol=args.rtol,
            atol=args.atol,
        )
        hamiltonian = instantaneous_hamiltonian(evaluated, problem)
        print(f"[{case.case_id}] full n={args.n} gradient and dense Hessian", flush=True)
        derivative_summary, derivative_arrays = evaluate_reduced_derivatives(
            case, initial_state, args.bound_tolerance, args.kkt_tolerance
        )
        case_dir = out_dir / case.case_id
        case_dir.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(
            case_dir / "teacher_facing_diagnostics.npz",
            breakpoints=case.breakpoints,
            interval_control=case.controls,
            continuous_time=evaluated.time,
            continuous_state=evaluated.state,
            continuous_costate=evaluated.costate,
            continuous_control=evaluated.control,
            continuous_hamiltonian=hamiltonian,
            total_population=evaluated.state.sum(axis=1),
            **derivative_arrays,
        )
        case_summary = {
            "case_id": case.case_id,
            "case_label": case.label.replace("$", ""),
            "checkpoint": str(case.checkpoint),
            "checkpoint_sha256": sha256(case.checkpoint),
            "n_intervals": args.n,
            "dt": float(problem.T / args.n),
            "initial_state": initial_state.tolist(),
            "hamiltonian_min": float(hamiltonian.min()),
            "hamiltonian_max": float(hamiltonian.max()),
            "hamiltonian_rms": float(np.sqrt(np.mean(hamiltonian**2))),
            "control_min": float(case.controls.min()),
            "control_max": float(case.controls.max()),
            "feedback_closure_error": evaluated.feedback_closure_error,
            **derivative_summary,
        }
        (case_dir / "summary.json").write_text(
            json.dumps(json_safe(case_summary), indent=2) + "\n", encoding="utf-8"
        )
        summaries.append(case_summary)
        print(
            f"[{case.case_id}] F={case_summary['reduced_objective_F_h']:.9f} "
            f"PGinf={case_summary['projected_gradient_linf']:.6g} "
            f"lambda_min={case_summary['hessian_min_eigenvalue_full']:.6g}",
            flush=True,
        )

    write_summary_csv(out_dir / "three_case_summary.csv", summaries)
    metadata = {
        "definition": {
            "instantaneous_hamiltonian": "H(N(t), lambda(t), u(t)) on a segmented DOP853 trajectory",
            "reduced_functional": "F_h(u)=J_h(N_h(u),u) on the n=800 RK4/ZOH transcription",
            "full_first_derivative": "g=nabla_u F_h; includes the complete N_h=N_h(u) recursion",
            "full_second_derivative": "R=nabla_u^2 F_h; dense 800-by-800 derivative of g",
        },
        "important_distinction": (
            "H(t) is an instantaneous trajectory diagnostic near 32; F_h(u) is "
            "the scalar reduced objective near 385. They are not the same quantity."
        ),
        "common_problem": {
            "n_intervals": args.n,
            "dt": problem.T / args.n,
            "initial_state": initial_state.tolist(),
            "diagnostic_points": args.diagnostic_points,
            "rtol": args.rtol,
            "atol": args.atol,
        },
        "feedback_resolution_note": (
            "Both feedback checkpoints are native to the requested n=800 mesh; "
            "their stored nominal references are copied to the common 801-node grid."
            if all(
                deployment_audits[name]["source_training_n"] == args.n
                for name in ("feedback_cf", "feedback_der")
            )
            else (
                "At least one feedback checkpoint was trained at a different "
                "resolution. Its learned time/state map is deployed on the common "
                "n=800 mesh, and its stored nominal reference is interpolated to "
                "801 nodes; this is a deployment audit, not additional training."
            )
        ),
        "deployments": deployment_audits,
        "cases": summaries,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(json_safe(metadata), indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
