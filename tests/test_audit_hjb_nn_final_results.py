from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_hjb_nn_final_results import (
    AuditProtocol,
    EXPECTED_SEEDS,
    EXPECTED_TAUS,
    NominalTumorProblem,
    audit_aggregate,
    entropy_density,
    integrate_zoh_independent,
    sha256_file,
)


SYNTHETIC_PROTOCOL = AuditProtocol(
    taus=EXPECTED_TAUS,
    seeds=EXPECTED_SEEDS,
    training={"width": 8, "max_rounds": 1},
    network={"layers": [22, 8, 1]},
    optimizer_budget={"method": "L-BFGS-B", "maxiter_per_round": 1},
    bvp={"tolerance": 1.0e-3, "max_nodes": 2_000},
    evaluation_intervals=2,
    adaptive_max_failures=2,
    train_trajectories=2,
    validation_trajectories=2,
    completed_rounds_min=1,
    completed_rounds_max=1,
    state_box_lower=5.0,
    state_box_upper=20.0,
    sampling_declaration="independent uniform coordinates over [5,20]^21",
    executed_source_keys=("tumor_problem.py",),
)


def _write_synthetic_aggregate(root: Path) -> Path:
    training = dict(SYNTHETIC_PROTOCOL.training)
    bvp = dict(SYNTHETIC_PROTOCOL.bvp)
    optimizer = dict(SYNTHETIC_PROTOCOL.optimizer_budget)
    source_payload = b"synthetic executed source\n"
    source_hash = hashlib.sha256(source_payload).hexdigest()
    source_signature = {
        "repo_revision": "synthetic-repo",
        "upstream_HJB_NN_base_revision": "synthetic-upstream",
        "upstream_git_diff_sha256": "0" * 64,
        "file_sha256": {"tumor_problem.py": source_hash},
        "python": "test-python",
        "platform": "test-platform",
        "dependency_versions": {"numpy": "test", "scipy": "test", "tensorflow": "test"},
    }
    signature_hash = hashlib.sha256(
        json.dumps(source_signature, sort_keys=True).encode("utf-8")
    ).hexdigest()
    time = np.linspace(0.0, 10.0, SYNTHETIC_PROTOCOL.evaluation_intervals + 1)
    control = np.full(SYNTHETIC_PROTOCOL.evaluation_intervals, 1.5)
    integrations = {
        tau: integrate_zoh_independent(time, control, tau) for tau in EXPECTED_TAUS
    }
    runs: list[dict[str, object]] = []

    for tau in EXPECTED_TAUS:
        for seed in EXPECTED_SEEDS:
            run_dir = root / f"tau_{tau:g}" / f"seed_{seed}"
            snapshot = run_dir / "executed_source_snapshot" / "tumor_problem.py"
            snapshot.parent.mkdir(parents=True)
            snapshot.write_bytes(source_payload)
            integrated = integrations[tau]
            train_initial = np.random.default_rng(seed).uniform(
                SYNTHETIC_PROTOCOL.state_box_lower,
                SYNTHETIC_PROTOCOL.state_box_upper,
                size=(21, SYNTHETIC_PROTOCOL.train_trajectories),
            )
            validation_initial = np.random.default_rng(
                np.random.SeedSequence([seed, 1])
            ).uniform(
                SYNTHETIC_PROTOCOL.state_box_lower,
                SYNTHETIC_PROTOCOL.state_box_upper,
                size=(21, SYNTHETIC_PROTOCOL.validation_trajectories),
            )
            train_path = run_dir / "dataset_train_initial.npz"
            validation_path = run_dir / "dataset_validation.npz"
            final_path = run_dir / "dataset_train_final.npz"
            np.savez_compressed(
                train_path,
                t=np.zeros((1, SYNTHETIC_PROTOCOL.train_trajectories)),
                X=train_initial,
                trajectory_id=np.arange(SYNTHETIC_PROTOCOL.train_trajectories)[None, :],
                initial_states=train_initial,
            )
            np.savez_compressed(
                validation_path,
                t=np.zeros((1, SYNTHETIC_PROTOCOL.validation_trajectories)),
                X=validation_initial,
                trajectory_id=(1_000_000 + np.arange(SYNTHETIC_PROTOCOL.validation_trajectories))[None, :],
                initial_states=validation_initial,
            )
            np.savez_compressed(
                final_path,
                t=np.zeros((1, SYNTHETIC_PROTOCOL.train_trajectories)),
                X=train_initial,
                trajectory_id=np.arange(SYNTHETIC_PROTOCOL.train_trajectories)[None, :],
            )
            bvp_record = {
                "success": True,
                "target_tau": tau,
                "max_rms_residual": 1.0e-5,
                "boundary_residual_max_abs": 0.0,
            }
            bvp_history_path = run_dir / "bvp_initial_history.json"
            bvp_history_path.write_text(
                json.dumps(
                    {
                        "train": [dict(bvp_record) for _ in range(SYNTHETIC_PROTOCOL.train_trajectories)],
                        "validation": [dict(bvp_record) for _ in range(SYNTHETIC_PROTOCOL.validation_trajectories)],
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            history_path = run_dir / "history.json"
            history_path.write_text(
                json.dumps(
                    {
                        "round_iters": [1],
                        "convergence_tests": [{}],
                        "optimizer_rounds": [{}],
                        "adaptive_events": [],
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            control_path = run_dir / "canonical_control.npz"
            np.savez_compressed(
                control_path,
                t=time,
                u=control,
                tau=np.array(tau),
                feedback_rollout_N=integrated.breakpoint_states,
                J_regularized_native_value_prediction=np.array(integrated.regularized_J),
                J_regularized_realized=np.array(integrated.regularized_J),
                regularized_terminal_cost=np.array(integrated.terminal_cost),
                regularized_running_cost=np.array(integrated.regularized_running_cost),
                regularized_entropy_integral=np.array(integrated.entropy_integral),
                J_unregularized_realized=np.array(integrated.unregularized_J),
                control_semantics=np.array("left-endpoint ZOH export of learned feedback"),
            )
            metrics = {
                "regularized_native_value_prediction": integrated.regularized_J,
                "regularized_realized_J": integrated.regularized_J,
                "unregularized_realized_J": integrated.unregularized_J,
            }
            artifact_hashes = {
                "canonical_control.npz": sha256_file(control_path),
                "dataset_train_initial.npz": sha256_file(train_path),
                "dataset_validation.npz": sha256_file(validation_path),
                "dataset_train_final.npz": sha256_file(final_path),
                "bvp_initial_history.json": sha256_file(bvp_history_path),
                "history.json": sha256_file(history_path),
                "executed_source_snapshot/tumor_problem.py": sha256_file(snapshot),
            }
            worker = {
                "tau": tau,
                "seed": seed,
                "regularized_training_objective": True,
                "unregularized_common_objective_used_for_training_or_selection": False,
                "all_candidates_retained": True,
                "initial_state_sampling": SYNTHETIC_PROTOCOL.sampling_declaration,
                "trajectory_disjoint_validation": True,
                "train_trajectories_initial": SYNTHETIC_PROTOCOL.train_trajectories,
                "validation_trajectories": SYNTHETIC_PROTOCOL.validation_trajectories,
                "initial_training_points": SYNTHETIC_PROTOCOL.train_trajectories,
                "validation_points": SYNTHETIC_PROTOCOL.validation_trajectories,
                "final_training_points": SYNTHETIC_PROTOCOL.train_trajectories,
                "completed_rounds": 1,
                "training": training,
                "network": dict(SYNTHETIC_PROTOCOL.network),
                "bvp": bvp,
                "optimizer_budget": optimizer,
                "evaluation_intervals": SYNTHETIC_PROTOCOL.evaluation_intervals,
                "adaptive_events": 0,
                "adaptive_successes": 0,
                "adaptive_failures": 0,
                "adaptive_max_failures": SYNTHETIC_PROTOCOL.adaptive_max_failures,
                "metrics": metrics,
                "artifact_sha256": artifact_hashes,
                "provenance": source_signature,
                "executed_source_snapshot": {
                    "tumor_problem.py": "executed_source_snapshot/tumor_problem.py"
                },
            }
            worker_path = run_dir / "manifest.json"
            worker_path.write_text(json.dumps(worker, sort_keys=True), encoding="utf-8")
            runs.append(
                {
                    "tau": tau,
                    "seed": seed,
                    "run_dir": str(run_dir.relative_to(root)),
                    "worker_manifest_sha256": sha256_file(worker_path),
                    **metrics,
                }
            )

    root.mkdir(parents=True, exist_ok=True)
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(runs[0]))
        writer.writeheader()
        writer.writerows(runs)
    aggregate = {
        "status": "completed",
        "worker_source_and_dependencies_uniform": True,
        "worker_source_signature_sha256": signature_hash,
        "taus": list(EXPECTED_TAUS),
        "seeds": list(EXPECTED_SEEDS),
        "training": training,
        "bvp": bvp,
        "optimizer_budget": optimizer,
        "evaluation_intervals": SYNTHETIC_PROTOCOL.evaluation_intervals,
        "adaptive_max_failures": SYNTHETIC_PROTOCOL.adaptive_max_failures,
        "runs": runs,
        "provenance": source_signature,
    }
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(aggregate, sort_keys=True), encoding="utf-8")
    return manifest


def _load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.array(loaded[key], copy=True) for key in loaded.files}


def _seal_worker_change(manifest: Path, relative_artifact: str) -> None:
    root = manifest.parent
    worker_path = root / "tau_10" / "seed_0" / "manifest.json"
    worker = json.loads(worker_path.read_text(encoding="utf-8"))
    worker["artifact_sha256"][relative_artifact] = sha256_file(worker_path.parent / relative_artifact)
    worker_path.write_text(json.dumps(worker, sort_keys=True), encoding="utf-8")
    _seal_worker_manifest_reference(manifest)


def _seal_worker_manifest_reference(manifest: Path) -> None:
    root = manifest.parent
    worker_path = root / "tau_10" / "seed_0" / "manifest.json"
    aggregate = json.loads(manifest.read_text(encoding="utf-8"))
    run = next(item for item in aggregate["runs"] if item["tau"] == 10.0 and item["seed"] == 0)
    run["worker_manifest_sha256"] = sha256_file(worker_path)
    manifest.write_text(json.dumps(aggregate, sort_keys=True), encoding="utf-8")


def _rewrite_npz_and_seal(
    manifest: Path,
    relative_artifact: str,
    mutate,
) -> None:
    artifact = manifest.parent / "tau_10" / "seed_0" / relative_artifact
    fields = _load_npz_dict(artifact)
    mutate(fields)
    np.savez_compressed(artifact, **fields)
    _seal_worker_change(manifest, relative_artifact)


def test_independent_integrator_tracks_both_costs_and_exact_zoh_entropy() -> None:
    time = np.array([0.0, 4.0, 10.0])
    control = np.array([0.0, 1.5])
    tau = 5.0
    problem = NominalTumorProblem()
    result = integrate_zoh_independent(time, control, tau)
    expected_entropy = (
        4.0 * entropy_density(0.0, tau, problem)
        + 6.0 * entropy_density(1.5, tau, problem)
    )
    assert result.entropy_integral == pytest.approx(expected_entropy, abs=1.0e-13)
    assert result.regularized_J - result.unregularized_J == pytest.approx(
        expected_entropy, abs=2.0e-9
    )
    assert result.breakpoint_states.shape == (21, 3)
    assert np.all(result.breakpoint_states > 0.0)


def test_audit_accepts_complete_hash_verified_3x3_grid(tmp_path: Path) -> None:
    manifest = _write_synthetic_aggregate(tmp_path / "aggregate")
    output = tmp_path / "audit"
    result = audit_aggregate(manifest, output, expected_protocol=SYNTHETIC_PROTOCOL)
    assert result["status"] == "passed"
    assert len(result["runs"]) == 9
    assert result["integrity_checks"]["worker_manifest_reference_hashes_verified"] == 9
    assert result["integrity_checks"]["canonical_control_hashes_verified"] == 9
    assert result["integrity_checks"]["initial_and_adaptive_bvp_metadata_verified"] is True
    assert result["max_discrepancies"]["canonical_unregularized_J_abs"] == pytest.approx(0.0)
    assert all(row["completed_rounds"] == 1 for row in result["runs"])
    assert all(row["max_accepted_bvp_rms_residual"] <= 1.0e-3 for row in result["runs"])
    assert (output / "audit_manifest.json").is_file()
    assert (output / "summary.csv").is_file()


def test_audit_fails_closed_on_tampered_canonical_artifact(tmp_path: Path) -> None:
    manifest = _write_synthetic_aggregate(tmp_path / "aggregate")
    canonical = tmp_path / "aggregate" / "tau_10" / "seed_0" / "canonical_control.npz"
    canonical.write_bytes(canonical.read_bytes() + b"tampered")
    output = tmp_path / "audit"
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        audit_aggregate(manifest, output, expected_protocol=SYNTHETIC_PROTOCOL)
    assert not output.exists()


def test_audit_fails_closed_on_protocol_budget_drift(tmp_path: Path) -> None:
    manifest = _write_synthetic_aggregate(tmp_path / "aggregate")
    aggregate = json.loads(manifest.read_text(encoding="utf-8"))
    aggregate["training"]["width"] = 9
    manifest.write_text(json.dumps(aggregate, sort_keys=True), encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"aggregate\.training\.width mismatch"):
        audit_aggregate(
            manifest,
            tmp_path / "audit",
            expected_protocol=SYNTHETIC_PROTOCOL,
        )


def test_audit_uses_final_v3_protocol_by_default(tmp_path: Path) -> None:
    manifest = _write_synthetic_aggregate(tmp_path / "aggregate")
    with pytest.raises(RuntimeError, match=r"aggregate\.training"):
        audit_aggregate(manifest, tmp_path / "audit")


@pytest.mark.parametrize(
    "replacement",
    [
        np.array([0.0, 4.9, 10.0]),
        np.array([0.0, 10.0, 5.0]),
    ],
    ids=["nonuniform", "not-strictly-increasing"],
)
def test_audit_fails_closed_on_illegal_or_nonuniform_breakpoints(
    tmp_path: Path,
    replacement: np.ndarray,
) -> None:
    manifest = _write_synthetic_aggregate(tmp_path / "aggregate")

    def mutate(fields: dict[str, np.ndarray]) -> None:
        fields["t"] = replacement

    _rewrite_npz_and_seal(manifest, "canonical_control.npz", mutate)
    with pytest.raises(RuntimeError, match="exact uniform linspace"):
        audit_aggregate(
            manifest,
            tmp_path / "audit",
            expected_protocol=SYNTHETIC_PROTOCOL,
        )


@pytest.mark.parametrize(
    "missing_field",
    ["J_regularized_native_value_prediction", "feedback_rollout_N"],
)
def test_audit_fails_closed_on_missing_native_or_feedback_states(
    tmp_path: Path,
    missing_field: str,
) -> None:
    manifest = _write_synthetic_aggregate(tmp_path / "aggregate")

    def mutate(fields: dict[str, np.ndarray]) -> None:
        del fields[missing_field]

    _rewrite_npz_and_seal(manifest, "canonical_control.npz", mutate)
    with pytest.raises(RuntimeError, match=missing_field):
        audit_aggregate(
            manifest,
            tmp_path / "audit",
            expected_protocol=SYNTHETIC_PROTOCOL,
        )


def test_audit_fails_closed_on_missing_dataset_initial_states(tmp_path: Path) -> None:
    manifest = _write_synthetic_aggregate(tmp_path / "aggregate")

    def mutate(fields: dict[str, np.ndarray]) -> None:
        del fields["initial_states"]

    _rewrite_npz_and_seal(manifest, "dataset_train_initial.npz", mutate)
    with pytest.raises(RuntimeError, match="missing initial_states"):
        audit_aggregate(
            manifest,
            tmp_path / "audit",
            expected_protocol=SYNTHETIC_PROTOCOL,
        )


def test_audit_fails_closed_on_noninteger_seed(tmp_path: Path) -> None:
    manifest = _write_synthetic_aggregate(tmp_path / "aggregate")
    aggregate = json.loads(manifest.read_text(encoding="utf-8"))
    aggregate["seeds"][0] = 0.0
    manifest.write_text(json.dumps(aggregate, sort_keys=True), encoding="utf-8")
    with pytest.raises(RuntimeError, match="strict integer"):
        audit_aggregate(
            manifest,
            tmp_path / "audit",
            expected_protocol=SYNTHETIC_PROTOCOL,
        )


def test_audit_fails_closed_on_forged_coordinate_uniform_sample(tmp_path: Path) -> None:
    manifest = _write_synthetic_aggregate(tmp_path / "aggregate")

    def mutate(fields: dict[str, np.ndarray]) -> None:
        fields["initial_states"][0, 0] = 12.3456789
        fields["X"][0, 0] = 12.3456789

    _rewrite_npz_and_seal(manifest, "dataset_train_initial.npz", mutate)
    with pytest.raises(RuntimeError, match="seed-reconstructable"):
        audit_aggregate(
            manifest,
            tmp_path / "audit",
            expected_protocol=SYNTHETIC_PROTOCOL,
        )


def test_audit_fails_closed_on_unreliable_initial_bvp(tmp_path: Path) -> None:
    manifest = _write_synthetic_aggregate(tmp_path / "aggregate")
    history_path = manifest.parent / "tau_10" / "seed_0" / "bvp_initial_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    history["train"][0]["max_rms_residual"] = 1.1e-3
    history_path.write_text(json.dumps(history, sort_keys=True), encoding="utf-8")
    _seal_worker_change(manifest, "bvp_initial_history.json")
    with pytest.raises(RuntimeError, match="exceeds"):
        audit_aggregate(
            manifest,
            tmp_path / "audit",
            expected_protocol=SYNTHETIC_PROTOCOL,
        )


def test_audit_fails_closed_on_adaptive_event_count_mismatch(tmp_path: Path) -> None:
    manifest = _write_synthetic_aggregate(tmp_path / "aggregate")
    history_path = manifest.parent / "tau_10" / "seed_0" / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    history["adaptive_events"] = [{"success": False}]
    history_path.write_text(json.dumps(history, sort_keys=True), encoding="utf-8")
    _seal_worker_change(manifest, "history.json")
    with pytest.raises(RuntimeError, match="adaptive event count"):
        audit_aggregate(
            manifest,
            tmp_path / "audit",
            expected_protocol=SYNTHETIC_PROTOCOL,
        )


def test_audit_fails_closed_on_successful_adaptive_bvp_above_tolerance(
    tmp_path: Path,
) -> None:
    manifest = _write_synthetic_aggregate(tmp_path / "aggregate")
    run_dir = manifest.parent / "tau_10" / "seed_0"
    history_path = run_dir / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    history["adaptive_events"] = [
        {
            "success": True,
            "bvp": {
                "success": True,
                "target_tau": 10.0,
                "max_rms_residual": 1.1e-3,
                "boundary_residual_max_abs": 0.0,
            },
        }
    ]
    history_path.write_text(json.dumps(history, sort_keys=True), encoding="utf-8")
    _seal_worker_change(manifest, "history.json")
    worker_path = run_dir / "manifest.json"
    worker = json.loads(worker_path.read_text(encoding="utf-8"))
    worker["adaptive_events"] = 1
    worker["adaptive_successes"] = 1
    worker_path.write_text(json.dumps(worker, sort_keys=True), encoding="utf-8")
    _seal_worker_manifest_reference(manifest)
    with pytest.raises(RuntimeError, match="adaptive event.*exceeds"):
        audit_aggregate(
            manifest,
            tmp_path / "audit",
            expected_protocol=SYNTHETIC_PROTOCOL,
        )
