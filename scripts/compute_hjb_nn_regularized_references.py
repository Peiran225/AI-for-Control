#!/usr/bin/env python3
"""Direct ZOH references for the entropy-regularized HJB-NN tumor objectives."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
import scipy.io
from scipy.optimize import minimize
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from faithful_related_work.hjb_nn.problem import (  # noqa: E402
    TumorEntropyProblem,
    evaluate_zoh_cost_pair,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_objective(problem: TumorEntropyProblem, intervals: int):
    dtype = torch.float64
    dt = problem.t1 / intervals
    growth = torch.tensor(problem.r, dtype=dtype)
    sensitivity = torch.tensor(problem.phi, dtype=dtype)
    suppression = torch.tensor(problem.M, dtype=dtype)
    beta = torch.tensor(problem.beta, dtype=dtype)
    initial = torch.tensor(problem.nominal_initial_state, dtype=dtype)

    def objective_and_gradient(control_numpy: np.ndarray):
        control = torch.tensor(control_numpy, dtype=dtype, requires_grad=True)
        state = initial
        running = torch.zeros((), dtype=dtype)

        def state_rate(value, action):
            crowding = torch.log1p(torch.mean(value))
            return (growth - sensitivity * action - suppression * crowding) * value

        def running_rate(value, action):
            p = action / problem.umax
            entropy = problem.tau * problem.umax * (
                p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)
            )
            return torch.dot(beta, value) + problem.gamma * action + entropy

        for action in control:
            k1 = state_rate(state, action)
            l1 = running_rate(state, action)
            state2 = state + 0.5 * dt * k1
            k2 = state_rate(state2, action)
            l2 = running_rate(state2, action)
            state3 = state + 0.5 * dt * k2
            k3 = state_rate(state3, action)
            l3 = running_rate(state3, action)
            state4 = state + dt * k3
            k4 = state_rate(state4, action)
            l4 = running_rate(state4, action)
            state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            running = running + (dt / 6.0) * (l1 + 2.0 * l2 + 2.0 * l3 + l4)
        objective = running + torch.sum(state)
        objective.backward()
        return float(objective.detach()), control.grad.detach().numpy().copy()

    return objective_and_gradient


def load_initial_controls(intervals: int) -> dict[str, np.ndarray]:
    time = np.linspace(0.0, 10.0, intervals + 1)
    legacy_path = REPO_ROOT / "paper_runs/related_work_corrected/hjb_nn_tumor_nominal.mat"
    legacy = scipy.io.loadmat(legacy_path)
    legacy_control = np.interp(
        time[:-1],
        np.asarray(legacy["t"], dtype=np.float64).reshape(-1),
        np.asarray(legacy["U_NN"], dtype=np.float64).reshape(-1),
    )
    direct_path = (
        REPO_ROOT / "paper_runs/canonical_results/solutions/direct_time_mesh_n200.npz"
    )
    with np.load(direct_path, allow_pickle=False) as direct:
        direct_control = np.asarray(direct["u"], dtype=np.float64).reshape(-1)
    if direct_control.size == intervals + 1:
        direct_control = direct_control[:-1]
    if direct_control.size != intervals:
        raise ValueError("unregularized direct control does not match the reference grid")
    return {"legacy_hjb": legacy_control, "unregularized_direct": direct_control}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="paper_runs/faithful_related_work/hjb_nn_regularized_direct_references",
    )
    parser.add_argument("--taus", default="10,5,2")
    parser.add_argument("--intervals", type=int, default=200)
    args = parser.parse_args(argv)

    output_dir = Path(args.out_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    taus = [float(value) for value in args.taus.split(",") if value.strip()]
    if not taus or args.intervals <= 0:
        raise ValueError("positive intervals and at least one tau are required")

    initial_controls = load_initial_controls(args.intervals)
    time = np.linspace(0.0, 10.0, args.intervals + 1)
    previous_best = None
    rows = []
    selected_controls = []
    all_candidates = []
    for tau in taus:
        problem = TumorEntropyProblem(tau)
        objective_and_gradient = make_objective(problem, args.intervals)
        starts = {
            "legacy_hjb": initial_controls["legacy_hjb"],
            "unregularized_direct": initial_controls["unregularized_direct"],
            "midpoint": np.full(args.intervals, 1.5, dtype=np.float64),
        }
        if previous_best is not None:
            starts["previous_tau_best"] = previous_best
            starts.pop("midpoint")

        candidates = []
        for start_name, start in starts.items():
            result = minimize(
                objective_and_gradient,
                np.asarray(start, dtype=np.float64),
                method="L-BFGS-B",
                jac=True,
                bounds=[(1.0e-8, problem.umax - 1.0e-8)] * args.intervals,
                options={
                    "maxiter": 1000,
                    "maxfun": 4000,
                    "ftol": 1.0e-12,
                    "gtol": 1.0e-8,
                    "maxls": 50,
                },
            )
            exact = evaluate_zoh_cost_pair(problem, time, result.x)
            record = {
                "tau": tau,
                "start": start_name,
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "iterations": int(result.nit),
                "function_evaluations": int(result.nfev),
                "gradient_evaluations": int(result.njev),
                "rk4_regularized_J": float(result.fun),
                **{key: float(value) for key, value in exact.items()},
                "u_min": float(np.min(result.x)),
                "u_max": float(np.max(result.x)),
                "control": np.asarray(result.x, dtype=np.float64),
            }
            candidates.append(record)
            all_candidates.append(record)
        selected = min(candidates, key=lambda item: item["regularized_J"])
        previous_best = np.array(selected["control"], copy=True)
        selected_controls.append(previous_best)
        rows.append(
            {
                key: value
                for key, value in selected.items()
                if key != "control"
            }
        )

    np.savez_compressed(
        output_dir / "selected_controls.npz",
        t=time,
        taus=np.asarray(taus, dtype=np.float64),
        u=np.vstack(selected_controls),
        control_semantics=np.asarray("left-endpoint ZOH"),
    )
    with (output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    candidate_json = []
    for record in all_candidates:
        candidate_json.append({key: value for key, value in record.items() if key != "control"})
    (output_dir / "candidates.json").write_text(
        json.dumps(candidate_json, indent=2, sort_keys=True)
    )

    sources = [
        Path(__file__),
        REPO_ROOT / "faithful_related_work/hjb_nn/problem.py",
        REPO_ROOT / "paper_runs/related_work_corrected/hjb_nn_tumor_nominal.mat",
        REPO_ROOT / "paper_runs/canonical_results/solutions/direct_time_mesh_n200.npz",
    ]
    artifacts = {
        path.name: sha256(path)
        for path in output_dir.iterdir()
        if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "experiment_label": "direct numerical references for entropy-regularized HJB-NN tumor adaptations",
        "not_a_learned_method": True,
        "taus": taus,
        "intervals": args.intervals,
        "control_semantics": "left-endpoint ZOH",
        "discretized_optimizer": {
            "state_and_cost_integrator": "float64 RK4 with four-stage running-cost quadrature",
            "gradient": "PyTorch float64 reverse-mode differentiation",
            "method": "L-BFGS-B",
            "maxiter": 1000,
            "maxfun": 4000,
            "ftol": 1.0e-12,
            "gtol": 1.0e-8,
            "maxls": 50,
            "bounds": [1.0e-8, 3.0 - 1.0e-8],
        },
        "independent_realization": "DOP853 cost-state integration, rtol=1e-10, atol=1e-12",
        "selection_rule": "lowest independently realized regularized J across every retained declared start",
        "all_candidates": candidate_json,
        "selected": rows,
        "source_sha256": {str(path.relative_to(REPO_ROOT)): sha256(path) for path in sources},
        "artifact_sha256": artifacts,
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
        },
        "command": sys.argv,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)
    )
    print(json.dumps({"selected": rows}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
