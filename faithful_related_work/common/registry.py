"""Declarative smoke registry; adding a method does not change orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class SmokeSpec:
    method_id: str
    method_name: str
    claim_label: str
    source_paths: tuple[str, ...]
    seeds: tuple[int, ...]
    budget: dict[str, object]
    command: Callable[[str, Path], Sequence[str]]
    upstream_kind: str
    upstream_revision: str | None
    selection_rule: str = "fixed predeclared smoke budget; no cross-candidate objective selection"
    selection_metric_scope: str = "fixed-budget"
    output_manifest: str | None = "manifest.json"
    stdout_is_json: bool = False
    availability_probe: Callable[[str], Sequence[str]] | None = None


def _deepbsde(python: str, output: Path) -> Sequence[str]:
    return (
        python,
        "-m",
        "faithful_related_work.deepbsde.runner",
        "hjb-lq-smoke",
        "--output-dir",
        str(output),
    )


def _neural_pmp(python: str, output: Path) -> Sequence[str]:
    return (
        python,
        "-m",
        "faithful_related_work.neural_pmp.run_lqr_smoke",
        "--out-dir",
        str(output),
    )


def _pi_deeponet(python: str, output: Path) -> Sequence[str]:
    return (
        python,
        "faithful_related_work/pi_deeponet/run_lqr_smoke.py",
        "--output-dir",
        str(output),
    )


def _lyznet(python: str, output: Path) -> Sequence[str]:
    return (
        python,
        "faithful_related_work/lyznet/run_original_pinn_pi.py",
        "--smoke",
        "--out-dir",
        str(output),
    )


def _lyznet_probe(python: str) -> Sequence[str]:
    return (
        python,
        "faithful_related_work/lyznet/run_original_pinn_pi.py",
        "--preflight",
    )


def _hjb_nn(python: str, output: Path) -> Sequence[str]:
    runner = REPO_ROOT / "faithful_related_work/hjb_nn/runner.py"
    if not runner.is_file():
        # The orchestrator turns this sentinel into an explicit unavailable entry.
        return ()
    return (
        python,
        "-m",
        "faithful_related_work.hjb_nn.runner",
        "validate-satellite",
        "--out-dir",
        str(output),
    )


SMOKE_SPECS: tuple[SmokeSpec, ...] = (
    SmokeSpec(
        method_id="deepbsde",
        method_name="DeepBSDE",
        # This command intentionally performs only two optimizer updates.  The
        # method runner marks it as a wiring smoke, so the normalized claim must
        # not promote it to a numerical reproduction.
        claim_label="non-comparable adaptation",
        source_paths=(
            "faithful_related_work/deepbsde/runner.py",
            "external/DeepBSDE/solver.py",
            "external/DeepBSDE/equation.py",
        ),
        seeds=(0,),
        budget={"optimizer_updates": 2, "time_intervals": 4, "batch_size": 8},
        command=_deepbsde,
        upstream_kind="author repository checkout",
        upstream_revision="e76fed80b1995daf4a0dd6d3e2cf64a0931dabda",
    ),
    SmokeSpec(
        method_id="neural_pmp",
        method_name="Neural-PMP",
        claim_label="original-method reproduction",
        source_paths=(
            "faithful_related_work/neural_pmp/run_lqr_smoke.py",
            "faithful_related_work/neural_pmp/run_lqr_reproduction.py",
            "faithful_related_work/neural_pmp/core.py",
        ),
        seeds=(0, 1),
        budget={
            "dynamics_train_samples": 128,
            "dynamics_validation_samples": 64,
            "dynamics_epochs": 25,
            "control_iterations": 20,
            "horizon_steps": 10,
        },
        command=_neural_pmp,
        upstream_kind="paper specification",
        upstream_revision="arXiv:2212.14566",
        selection_rule=(
            "fixed final dynamics checkpoint and fixed final control at the "
            "predeclared budget; first predeclared start; canonical seed 0 fixed"
        ),
        selection_metric_scope="fixed-budget",
    ),
    SmokeSpec(
        method_id="pi_deeponet",
        method_name="PI-DeepONet",
        claim_label="original-method reproduction",
        source_paths=(
            "faithful_related_work/pi_deeponet/run_lqr_smoke.py",
            "faithful_related_work/pi_deeponet/core.py",
            "faithful_related_work/pi_deeponet/experiment.py",
        ),
        seeds=(0,),
        budget={"outer_iterations": 3, "steps_per_outer": 20, "batch_size": 16},
        command=_pi_deeponet,
        upstream_kind="paper specification",
        upstream_revision="arXiv:2406.10920",
    ),
    SmokeSpec(
        method_id="lyznet",
        method_name="LyZNet PINN-PI",
        claim_label="original-method reproduction",
        source_paths=(
            "faithful_related_work/lyznet/run_original_pinn_pi.py",
            "external/lyznet/examples/icml24-pinn-pi/pendulum/pendulum.py",
        ),
        seeds=(123,),
        budget={"policy_iterations": 1, "epochs": 1, "collocation_points": 64},
        command=_lyznet,
        upstream_kind="author repository checkout",
        upstream_revision="21d8e81a4a21aa990532275836944d9384089f09",
        availability_probe=_lyznet_probe,
    ),
    SmokeSpec(
        method_id="hjb_nn",
        method_name="Adaptive HJB-NN",
        claim_label="original-method reproduction",
        source_paths=(
            "faithful_related_work/hjb_nn/runner.py",
            "faithful_related_work/hjb_nn/satellite.py",
            "faithful_related_work/hjb_nn/problem.py",
            "faithful_related_work/hjb_nn/sampling.py",
            "external/HJB_NN/utilities/neural_networks.py",
            "external/HJB_NN/utilities/optimize.py",
        ),
        seeds=(),
        budget={"mode": "released checkpoint validation", "dataset": "data_test.mat"},
        command=_hjb_nn,
        upstream_kind="paper specification and author repository checkout",
        upstream_revision="11cfa0e721f02bf36d0cdf03b16e08cf6a06e49c",
    ),
)


def smoke_specs(method_ids: Sequence[str] | None = None) -> tuple[SmokeSpec, ...]:
    if method_ids is None:
        return SMOKE_SPECS
    requested = set(method_ids)
    known = {spec.method_id for spec in SMOKE_SPECS}
    unknown = requested - known
    if unknown:
        raise KeyError(f"unknown method ids: {', '.join(sorted(unknown))}")
    return tuple(spec for spec in SMOKE_SPECS if spec.method_id in requested)
