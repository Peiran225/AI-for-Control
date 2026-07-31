#!/usr/bin/env python3
"""Run the DeepBSDE official solver path on the tumor HJB adaptation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
DEEP = ROOT / "external" / "DeepBSDE"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(DEEP))

import equation as eqn  # noqa: E402
from solver import BSDESolver  # noqa: E402

from run_transformer_objective_ablation import eval_common_grid, eval_rk4  # noqa: E402
from tumor_problem import NOMINAL_TUMOR_PROBLEM  # noqa: E402


class DictToObject:
    def __init__(self, dictionary):
        self._dict = dictionary
        for key, value in dictionary.items():
            setattr(self, key, value)

    def to_dict(self):
        return self._dict


class Config:
    def __init__(self, config_dict):
        self.eqn_config = DictToObject(config_dict["eqn_config"])
        self.net_config = DictToObject(config_dict["net_config"])
        self._original_dict = config_dict

    def to_dict(self):
        return self._original_dict


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def z_at_time(model, bsde, x_np: np.ndarray, time_index: int) -> np.ndarray:
    x = tf.convert_to_tensor(x_np.reshape(1, -1), dtype=tf.float64)
    one = tf.ones((1, 1), dtype=tf.float64)
    if time_index == 0:
        z = tf.matmul(one, model.z_init)
    else:
        idx = min(max(time_index - 1, 0), len(model.subnet) - 1)
        z = model.subnet[idx](x, training=False) / bsde.dim
    return np.asarray(z.numpy(), dtype=np.float64).reshape(-1)


def tumor_policy_from_model(model, bsde, x_np: np.ndarray, time_index: int, mode: str, temp: float) -> float:
    z = z_at_time(model, bsde, x_np, time_index)
    grad_v = z / max(float(bsde.sigma), 1e-12)
    x_pos = np.maximum(x_np, 1e-10)
    psi = float(bsde.gamma - np.sum(bsde.phi_np * x_pos * grad_v))
    if mode == "hard":
        return float(bsde.umax if psi < 0.0 else 0.0)
    return float(bsde.umax / (1.0 + np.exp(np.clip(psi / max(temp, 1e-6), -60.0, 60.0))))


def deterministic_rollout_control(model, bsde, n_eval: int, mode: str, temp: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0.0, bsde.total_time, n_eval + 1)
    dt = bsde.total_time / n_eval
    N = np.empty((n_eval + 1, bsde.dim), dtype=np.float64)
    u = np.empty(n_eval + 1, dtype=np.float64)
    N[0] = bsde.x_init

    def dyn(x: np.ndarray, u_val: float) -> np.ndarray:
        x_pos = np.maximum(x, 1e-12)
        G = np.log1p(x_pos.mean())
        return (bsde.r_np - bsde.phi_np * u_val - bsde.M_np * G) * x_pos

    for k in range(n_eval):
        idx = min(int(k * bsde.num_time_interval / n_eval), bsde.num_time_interval - 1)
        uk = tumor_policy_from_model(model, bsde, N[k], idx, mode=mode, temp=temp)
        u[k] = uk
        x = N[k]
        k1 = dyn(x, uk)
        k2 = dyn(np.maximum(x + 0.5 * dt * k1, 1e-12), uk)
        k3 = dyn(np.maximum(x + 0.5 * dt * k2, 1e-12), uk)
        k4 = dyn(np.maximum(x + dt * k3, 1e-12), uk)
        N[k + 1] = np.maximum(x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0, 1e-12)
    u[-1] = tumor_policy_from_model(model, bsde, N[-1], bsde.num_time_interval - 1, mode=mode, temp=temp)
    return t, N, u


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default=str(DEEP / "configs" / "tumor_hjb_d21.json"))
    parser.add_argument("--out_dir", default=str(ROOT / "paper_runs" / "official_related_work_adaptations"))
    parser.add_argument("--exp_name", default="deepbsde_tumor_official")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--policy_mode", choices=["hard", "smooth"], default="smooth")
    parser.add_argument("--policy_temp", type=float, default=0.5)
    parser.add_argument("--n_eval", type=int, default=400)
    parser.add_argument("--num_iterations", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--valid_size", type=int, default=None)
    parser.add_argument("--logging_frequency", type=int, default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.keras.backend.clear_session()

    with open(args.config_path) as f:
        config_dict = json.load(f)
    eqn_config = config_dict["eqn_config"]
    expected = {
        "total_time": NOMINAL_TUMOR_PROBLEM.T,
        "dim": NOMINAL_TUMOR_PROBLEM.m,
        "umax": NOMINAL_TUMOR_PROBLEM.umax,
        "beta": NOMINAL_TUMOR_PROBLEM.beta,
        "alpha": NOMINAL_TUMOR_PROBLEM.alpha,
        "gamma": NOMINAL_TUMOR_PROBLEM.gamma,
        "n0": NOMINAL_TUMOR_PROBLEM.n0,
        "m_suppression": NOMINAL_TUMOR_PROBLEM.m_suppression,
    }
    mismatches = {key: (eqn_config.get(key), value) for key, value in expected.items() if eqn_config.get(key) != value}
    if mismatches:
        raise ValueError(f"DeepBSDE tumor configuration does not match the canonical problem: {mismatches}")
    for key in ("num_iterations", "batch_size", "valid_size", "logging_frequency"):
        value = getattr(args, key)
        if value is not None:
            config_dict["net_config"][key] = value
    config = Config(config_dict)
    tf.keras.backend.set_floatx(config.net_config.dtype)
    bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)
    solver = BSDESolver(config, bsde)
    history = solver.train()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / args.exp_name
    np.savetxt(
        str(prefix) + "_training_history.csv",
        history,
        fmt=["%d", "%.8e", "%.8e", "%d"],
        delimiter=",",
        header="step,loss_function,target_value,elapsed_time",
        comments="",
    )

    t, N, u = deterministic_rollout_control(
        solver.model,
        bsde,
        n_eval=args.n_eval,
        mode=args.policy_mode,
        temp=args.policy_temp,
    )
    ref = eval_rk4(u)
    common = eval_common_grid(u)
    native = float(solver.model.y_init.numpy()[0])
    np.savez(
        str(prefix) + ".npz",
        t=t,
        N=N,
        u=u,
        native_estimate=native,
        training_history=history,
        policy_mode=args.policy_mode,
        policy_temp=args.policy_temp,
        sigma=float(bsde.sigma),
        m_suppression=float(bsde.M_np[0]),
    )
    rows = [
        {
            "method": f"Vanishing-viscosity DeepBSDE [2] adaptation (sigma={bsde.sigma:g})",
            "native_estimate": native,
            "realized_J": ref["J_ref_25600"],
            "J_gap_to_direct": "",
            "relative_gap_percent": "",
            "pmp_kkt_gap": common["common_gap"],
            "u_min": float(u.min()),
            "u_max": float(u.max()),
            "u_mean": float(u.mean()),
            "final_mean_N": ref["final_mean_N_ref"],
            "source": str((prefix.with_suffix(".npz")).relative_to(ROOT)),
            "note": (
                f"official DeepBSDE solver with tumor HJB generator; sigma={bsde.sigma:g}; "
                f"M={bsde.M_np[0]:g}; {args.policy_mode} value-gradient policy"
            ),
        }
    ]
    fields = list(rows[0].keys())
    write_csv(prefix.with_name(prefix.name + "_summary.csv"), rows, fields)
    print(prefix.with_suffix(".npz"))
    print(rows[0])


if __name__ == "__main__":
    main()
