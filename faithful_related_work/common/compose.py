"""Normalize existing method manifests without inventing missing provenance."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .manifest import sha256_file, validate_root_manifest
from .registry import REPO_ROOT
from .smoke import prepare_empty_output_directory


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"method manifest is not a JSON object: {path}")
    return value


def _inside(path: Path, root: Path, what: str) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as exc:
        raise ValueError(f"{what} lies outside declared root {root}: {path}") from exc


def _file_record(
    path: Path,
    root: Path,
    role: str,
    *,
    expected: str | None = None,
    hash_origin: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if expected is not None and actual != expected:
        raise ValueError(f"recorded SHA-256 mismatch for {path}: {expected} != {actual}")
    return {
        "path": _inside(path, root, role),
        "sha256": actual,
        "bytes": path.stat().st_size,
        "role": role,
        "hash_origin": hash_origin,
    }


def _import_time_sources(relative_paths: Sequence[str], source_root: Path) -> list[dict[str, Any]]:
    records = []
    for relative in relative_paths:
        path = source_root / relative
        records.append(
            _file_record(
                path,
                source_root,
                "runner-source",
                hash_origin="import-time snapshot; method manifest did not record this hash",
            )
        )
    return records


def _raw_manifest_artifact(path: Path, artifact_root: Path) -> dict[str, Any]:
    return _file_record(
        path,
        artifact_root,
        "method-manifest",
        hash_origin="import-time normalization",
    )


def _selection(rule: str, scope: str, *, candidates_retained: bool = True) -> dict[str, Any]:
    return {
        "rule": rule,
        "metric_scope": scope,
        "uses_realized_objective": False,
        "candidates_retained": candidates_retained,
    }


def _entry(
    *,
    run_id: str,
    method_id: str,
    method_name: str,
    claim_label: str,
    seeds: Sequence[int],
    budget: Mapping[str, Any],
    selection: Mapping[str, Any],
    upstream: Mapping[str, Any],
    sources: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    native: Mapping[str, Any] | None = None,
    realized: Mapping[str, Any] | None = None,
    canonical: Mapping[str, Any] | None = None,
    seed_policy: str | None = None,
) -> dict[str, Any]:
    value = {
        "run_id": run_id,
        "method_id": method_id,
        "method_name": method_name,
        "upstream": dict(upstream),
        "claim_label": claim_label,
        "status": "completed",
        "seeds": [int(seed) for seed in seeds],
        "seed_policy": seed_policy or ("explicit RNG seeds" if seeds else "deterministic/no RNG"),
        "budget": {"kind": "method-declared", "parameters": dict(budget)},
        "selection": dict(selection),
        "tumor_comparable": claim_label == "tumor adaptation",
        "metrics": {"native": native, "realized": realized},
        "sources": sources,
        "artifacts": artifacts,
    }
    if canonical is not None:
        value["canonical_control"] = dict(canonical)
    return value


def _recorded_artifact_mapping(
    mapping: Mapping[str, Any], manifest_dir: Path, artifact_root: Path
) -> list[dict[str, Any]]:
    records = []
    for relative, value in sorted(mapping.items()):
        digest = value.get("sha256") if isinstance(value, Mapping) else value
        if not isinstance(digest, str):
            raise ValueError(f"artifact {relative!r} has no recorded SHA-256")
        records.append(
            _file_record(
                manifest_dir / relative,
                artifact_root,
                "method-output",
                expected=digest,
                hash_origin="run-time method manifest",
            )
        )
    return records


def _deepbsde(path: Path, raw: Mapping[str, Any], artifact_root: Path) -> list[dict[str, Any]]:
    # Both the tumor sweep and the five-seed original HJB-LQ reproduction emit
    # a root manifest whose ``runs`` entries point at complete child manifests.
    # Normalize the children instead of trying to treat the aggregate (which
    # intentionally has no single seed) as one run.
    children = raw.get("runs")
    if isinstance(children, list) and children and all(
        isinstance(child, Mapping) and "manifest" in child for child in children
    ):
        entries = []
        for child in children:
            child_path = path.parent / child["manifest"]
            expected = child.get("manifest_sha256")
            if not isinstance(expected, str):
                raise ValueError(f"DeepBSDE root manifest lacks a child manifest hash: {path}")
            if sha256_file(child_path) != expected:
                raise ValueError(f"DeepBSDE child manifest hash mismatch: {child_path}")
            entries.extend(_deepbsde(child_path, _load(child_path), artifact_root))
        if not entries:
            raise ValueError("DeepBSDE root manifest contains no child runs")
        return entries

    sources = []
    for relative, value in sorted(raw.get("official_sources", {}).items()):
        digest = value.get("sha256") if isinstance(value, Mapping) else None
        sources.append(
            _file_record(
                REPO_ROOT / relative,
                REPO_ROOT,
                "upstream-source",
                expected=digest,
                hash_origin="run-time method manifest",
            )
        )
    artifacts = _recorded_artifact_mapping(raw.get("artifacts", {}), path.parent, artifact_root)
    artifacts.append(_raw_manifest_artifact(path, artifact_root))
    tumor = "tumor" in str(raw.get("method", "")).lower()
    seed = raw.get("seed")
    if seed is None:
        raise ValueError(f"DeepBSDE manifest does not declare a seed: {path}")
    if tumor:
        budget_source = raw.get("config", {}).get("net_config", {})
    else:
        # The per-seed benchmark manifest records the immutable solver config as
        # a hashed artifact.  Import the small set of compute-defining fields so
        # the normalized entry identifies the actual fixed budget rather than
        # merely the number of log rows.
        solver_config_path = path.parent / "solver_config.json"
        solver_config = _load(solver_config_path) if solver_config_path.is_file() else {}
        net_config = solver_config.get("net_config", {})
        equation_config = solver_config.get("eqn_config", {})
        budget_source = {
            "mode": raw.get("mode"),
            "optimizer_updates": net_config.get("num_iterations"),
            "batch_size": net_config.get("batch_size"),
            "time_intervals": equation_config.get("num_time_interval"),
            "state_dimension": equation_config.get("dim"),
            "history_rows": raw.get("history_rows"),
        }
    native = None
    realized = None
    canonical = None
    if tumor:
        metrics = raw.get("realized_metrics")
        if not isinstance(metrics, Mapping) or "J" not in metrics:
            raise ValueError(f"DeepBSDE tumor manifest lacks realized metrics: {path}")
        realized = {
            "name": "J",
            "value": float(metrics["J"]),
            "evaluator": "tumor_problem.evaluate_zoh_control",
        }
        control = next((item for item in artifacts if item["path"].endswith("canonical_control.npz")), None)
        if control is None:
            raise ValueError(f"DeepBSDE tumor manifest lacks canonical control hash: {path}")
        canonical = {**control, "control_semantics": "breakpoint-aligned ZOH"}
    elif "final_terminal_matching_loss" in raw:
        diagnostics = {}
        if "final_y_init" in raw:
            diagnostics["final_y_init"] = float(raw["final_y_init"])
        reference = raw.get("reference")
        if isinstance(reference, Mapping) and "value" in reference:
            diagnostics["reference_y_init"] = float(reference["value"])
        if "final_absolute_error" in raw:
            diagnostics["y0_absolute_error"] = float(raw["final_absolute_error"])
        if "final_relative_error" in raw:
            diagnostics["y0_relative_error"] = float(raw["final_relative_error"])
        native = {
            "name": "terminal_matching_loss",
            "value": float(raw["final_terminal_matching_loss"]),
            "diagnostics": diagnostics,
        }
    sigma = raw.get("sigma")
    wiring_smoke = not tumor and (
        raw.get("mode") == "smoke"
        or bool(raw.get("not_a_numerical_reproduction"))
        or "smoke" in str(raw.get("method", "")).lower()
    )
    suffix = (
        f"sigma-{sigma}-seed-{seed}"
        if tumor
        else f"original-{'smoke-' if wiring_smoke else ''}seed-{seed}"
    )
    return [
        _entry(
            run_id=f"deepbsde-{suffix}",
            method_id="deepbsde",
            method_name="DeepBSDE" + (" wiring smoke" if wiring_smoke else ""),
            claim_label=(
                "tumor adaptation"
                if tumor
                else ("non-comparable adaptation" if wiring_smoke else "original-method reproduction")
            ),
            seeds=[int(seed)],
            budget={key: value for key, value in budget_source.items() if value is not None},
            selection=_selection(
                "every predeclared run retained"
                if tumor
                else (
                    "fixed predeclared wiring-smoke budget"
                    if wiring_smoke
                    else "fixed predeclared official full budget; all seeds retained"
                ),
                "none" if tumor else "fixed-budget",
            ),
            upstream={"kind": "author repository checkout", "revision": "e76fed80b1995daf4a0dd6d3e2cf64a0931dabda"},
            sources=sources,
            artifacts=artifacts,
            native=native,
            realized=realized,
            canonical=canonical,
        )
    ]


def _neural_pmp(path: Path, raw: Mapping[str, Any], artifact_root: Path) -> list[dict[str, Any]]:
    if "Appendix C LQR" in str(raw.get("paper", "")):
        if raw.get("cross_seed_performance_selection") is not False:
            raise ValueError("Neural-PMP LQR manifest permits cross-seed performance selection")
        summary_path = path.parent / "run_config_and_summary.json"
        summary = _load(summary_path)
        args = summary.get("arguments")
        selection = summary.get("selection")
        aggregate = summary.get("aggregate")
        if not all(isinstance(item, Mapping) for item in (args, selection, aggregate)):
            raise ValueError("Neural-PMP LQR summary lacks arguments, selection, or aggregate")
        if selection.get("true_objective_available_to_selection") is not False:
            raise ValueError("Neural-PMP LQR selection saw the true objective")
        if selection.get("cross_seed_performance_selection") is not False:
            raise ValueError("Neural-PMP LQR selected a seed by performance")
        if selection.get("uses_validation_for_selection") is not False:
            raise ValueError("Neural-PMP LQR used validation to select a fixed-budget run")
        if raw.get("dynamics_checkpoint_policy") != "fixed_final":
            raise ValueError("Neural-PMP LQR did not retain the official fixed-final dynamics checkpoint")
        if raw.get("control_return_policy") != "fixed_final":
            raise ValueError("Neural-PMP LQR did not retain the official fixed-final control")
        artifacts = _recorded_artifact_mapping(raw.get("artifacts", {}), path.parent, artifact_root)
        artifacts.append(_raw_manifest_artifact(path, artifact_root))
        sources = _import_time_sources(
            (
                "faithful_related_work/neural_pmp/core.py",
                "faithful_related_work/neural_pmp/dynamics.py",
                "faithful_related_work/neural_pmp/run_lqr_reproduction.py",
            ),
            REPO_ROOT,
        )
        seeds = [int(item) for item in str(args["seeds"]).split(",") if item.strip()]
        return [
            _entry(
                run_id="neural-pmp-original-lqr-" + "-".join(map(str, seeds)),
                method_id="neural_pmp",
                method_name="Neural-PMP",
                claim_label="original-method reproduction",
                seeds=seeds,
                budget=dict(args),
                selection=_selection(
                    str(selection["within_seed_rule"])
                    + "; fixed-final dynamics/control"
                    + f"; canonical seed {selection['canonical_seed']} fixed",
                    "fixed-budget",
                ),
                upstream={"kind": "paper specification", "revision": "arXiv:2212.14566"},
                sources=sources,
                artifacts=artifacts,
                native={
                    "name": "mean_original_LQR_objective_post_selection",
                    "value": float(aggregate["mean_true_objective_post_selection"]),
                },
            )
        ]
    if raw.get("selection_frozen_before_post_selection_diagnostics") is not True:
        raise ValueError("Neural-PMP manifest does not freeze selection before realized diagnostics")
    summary_path = path.parent / "run_config_and_summary.json"
    summary = _load(summary_path)
    args = summary.get("runner_arguments")
    selected = summary.get("selected_run")
    selection = summary.get("selection")
    metrics = summary.get("selected_realized_metrics")
    if not all(isinstance(item, Mapping) for item in (args, selected, selection, metrics)):
        raise ValueError("Neural-PMP run_config_and_summary.json is incomplete")
    if selection.get("true_or_realized_objective_available_to_rule") is not False:
        raise ValueError("Neural-PMP selection declaration permits post-hoc realized-J selection")
    artifacts = _recorded_artifact_mapping(raw.get("artifacts", {}), path.parent, artifact_root)
    artifacts.append(_raw_manifest_artifact(path, artifact_root))
    sources = _import_time_sources(
        (
            "faithful_related_work/neural_pmp/core.py",
            "faithful_related_work/neural_pmp/dynamics.py",
            "faithful_related_work/neural_pmp/run_tumor.py",
        ),
        REPO_ROOT,
    )
    control = next((item for item in artifacts if item["path"].endswith("canonical_control.npz")), None)
    if control is None:
        raise ValueError("Neural-PMP manifest lacks canonical_control.npz")
    seeds = [int(item) for item in str(args["seeds"]).split(",") if item.strip()]
    return [
        _entry(
            run_id="neural-pmp-tumor-" + "-".join(map(str, seeds)),
            method_id="neural_pmp",
            method_name="Neural-PMP",
            claim_label="tumor adaptation",
            seeds=seeds,
            budget=dict(args),
            selection=_selection(str(selection["rule"]), "native-validation"),
            upstream={"kind": "paper specification", "revision": "arXiv:2212.14566"},
            sources=sources,
            artifacts=artifacts,
            native={"name": str(selected["selection_metric"]), "value": float(selected["selection_value"])},
            realized={"name": "J", "value": float(metrics["J"]), "evaluator": "tumor_problem.evaluate_zoh_control"},
            canonical={**control, "control_semantics": "breakpoint-aligned ZOH"},
        )
    ]


def _pi_deeponet(path: Path, raw: Mapping[str, Any], artifact_root: Path) -> list[dict[str, Any]]:
    experiment = raw.get("experiment")
    if not isinstance(experiment, Mapping):
        raise ValueError("PI-DeepONet manifest lacks experiment declaration")
    if experiment.get("selection_uses_nominal_realized_objective") is not False:
        raise ValueError("PI-DeepONet does not declare realized-J-free selection")
    sources = _import_time_sources(
        (
            "faithful_related_work/pi_deeponet/core.py",
            "faithful_related_work/pi_deeponet/experiment.py",
            "faithful_related_work/pi_deeponet/problems.py",
        ),
        REPO_ROOT,
    )
    tumor = raw.get("experiment_label") == "tumor adaptation"
    entries = []
    for seed_record in raw.get("seeds", []):
        seed = int(seed_record["seed"])
        seed_dir = path.parent / f"seed_{seed}"
        selected_path = seed_dir / seed_record["selected_solution"]
        selected = _file_record(
            selected_path,
            artifact_root,
            "canonical-control" if tumor else "selected-paper-benchmark-solution",
            expected=seed_record["selected_solution_sha256"],
            hash_origin="run-time method manifest",
        )
        artifacts = [selected, _raw_manifest_artifact(path, artifact_root)]
        history = seed_dir / seed_record["history"]
        artifacts.append(
            _file_record(
                history,
                artifact_root,
                "training-history",
                expected=seed_record["history_sha256"],
                hash_origin="run-time method manifest",
            )
        )
        for candidate in seed_record.get("candidates", []):
            for key, hash_key, role in (
                ("checkpoint", "checkpoint_sha256", "checkpoint-candidate"),
                ("solution", "solution_sha256", "control-candidate"),
            ):
                artifacts.append(
                    _file_record(
                        seed_dir / candidate[key],
                        artifact_root,
                        role,
                        expected=candidate[hash_key],
                        hash_origin="run-time method manifest",
                    )
                )
        with np.load(selected_path, allow_pickle=False) as data:
            if tumor:
                realized = {
                    "name": "J",
                    "value": float(data["J_realized"]),
                    "evaluator": "tumor_problem.evaluate_zoh_control",
                }
                native = None
            else:
                native = {"name": "J_native_left_rule", "value": float(data["J_native_left_rule"])}
                realized = None
        entries.append(
            _entry(
                run_id=f"pi-deeponet-{'tumor' if tumor else 'original'}-seed-{seed}",
                method_id="pi_deeponet",
                method_name="PI-DeepONet",
                claim_label="tumor adaptation" if tumor else "original-method reproduction",
                seeds=[seed],
                budget=raw.get("train_config", {}),
                selection=_selection(str(experiment["selection_rule"]), "fixed-budget"),
                upstream={"kind": "paper specification", "revision": str(raw.get("citation", "arXiv:2406.10920"))},
                sources=sources,
                artifacts=artifacts,
                native=native,
                realized=realized,
                canonical={**selected, "control_semantics": "breakpoint-aligned ZOH"} if tumor else None,
            )
        )
    if not entries:
        raise ValueError("PI-DeepONet manifest contains no seed records")
    return entries


def _lyznet(path: Path, raw: Mapping[str, Any], artifact_root: Path) -> list[dict[str, Any]]:
    if raw.get("status") != "completed":
        raise ValueError("only completed LyZNet manifests can be normalized")
    runner_path = REPO_ROOT / "faithful_related_work/lyznet/run_original_pinn_pi.py"
    recorded_runner = raw.get("runner_sha256")
    if not isinstance(recorded_runner, str):
        raise ValueError("LyZNet manifest does not record the runner SHA-256")
    if sha256_file(runner_path) != recorded_runner:
        raise ValueError(
            "LyZNet artifact was produced by a different runner revision; "
            "rerun the real-dReal container before normalization"
        )
    sources = []
    for value in raw.get("source_sha256", {}).values():
        sources.append(
            _file_record(
                REPO_ROOT / value["path"],
                REPO_ROOT,
                "upstream-source",
                expected=value["sha256"],
                hash_origin="run-time method manifest",
            )
        )
    sources.append(
        _file_record(
            runner_path,
            REPO_ROOT,
            "runner-source",
            expected=recorded_runner,
            hash_origin="run-time method manifest",
        )
    )
    artifacts = []
    for value in raw.get("artifacts", []):
        artifacts.append(
            _file_record(
                path.parent / value["path"],
                artifact_root,
                "method-output",
                expected=value["sha256"],
                hash_origin="run-time method manifest",
            )
        )
    artifacts.append(_raw_manifest_artifact(path, artifact_root))
    return [
        _entry(
            run_id=f"lyznet-original-seed-{raw['seed']}",
            method_id="lyznet",
            method_name="LyZNet PINN-PI",
            claim_label="original-method reproduction",
            seeds=[int(raw["seed"])],
            budget=raw["effective_budget"],
            selection=_selection("fixed author/smoke budget", "fixed-budget"),
            upstream={"kind": "author repository checkout", "revision": str(raw["upstream_revision"])},
            sources=sources,
            artifacts=artifacts,
        )
    ]


def _hjb_nn(path: Path, raw: Mapping[str, Any], artifact_root: Path) -> list[dict[str, Any]]:
    # Tumor sweep roots point to complete per-tau/per-seed child manifests.
    if "runs" in raw and "taus" in raw:
        entries = []
        for child in raw["runs"]:
            child_path = path.parent / child["run_dir"] / "manifest.json"
            entries.extend(_hjb_nn(child_path, _load(child_path), artifact_root))
        return entries
    provenance = raw.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("HJB-NN manifest lacks provenance")
    upstream_revision = provenance.get(
        "upstream_HJB_NN_base_revision", provenance.get("upstream_HJB_NN_revision")
    )
    if not isinstance(upstream_revision, str) or not upstream_revision:
        raise ValueError("HJB-NN manifest lacks the upstream base revision")
    compatibility_port = bool(
        provenance.get("tensorflow2_compatibility_port", raw.get("tensorflow2_compatibility_port"))
    )
    upstream = {
        "kind": (
            "author repository base plus tracked TensorFlow-2 compatibility port"
            if compatibility_port
            else "author repository checkout"
        ),
        "revision": upstream_revision,
    }
    sources = []
    for relative, digest in provenance.get("file_sha256", {}).items():
        sources.append(
            _file_record(
                REPO_ROOT / relative,
                REPO_ROOT,
                "runner-or-upstream-source",
                expected=digest,
                hash_origin="run-time method manifest",
            )
        )
    artifacts = [_raw_manifest_artifact(path, artifact_root)]
    tumor = "tumor adaptation" in str(raw.get("experiment_label", ""))
    if tumor:
        artifacts.extend(_recorded_artifact_mapping(raw.get("artifact_sha256", {}), path.parent, artifact_root))
        control = next((item for item in artifacts if item["path"].endswith("canonical_control.npz")), None)
        metrics = raw.get("metrics")
        if control is None or not isinstance(metrics, Mapping):
            raise ValueError("HJB-NN tumor manifest lacks canonical control or metrics")
        seed = int(raw["seed"])
        return [
            _entry(
                run_id=f"hjb-nn-tumor-tau-{raw['tau']}-seed-{seed}",
                method_id="hjb_nn",
                method_name="Adaptive HJB-NN",
                claim_label="tumor adaptation",
                seeds=[seed],
                budget={
                    **raw.get("training", {}),
                    **{
                        f"bvp_{key}": value
                        for key, value in raw.get("bvp", {}).items()
                    },
                    **{
                        f"optimizer_{key}": value
                        for key, value in raw.get("optimizer_budget", {}).items()
                    },
                    **{
                        key: raw[key]
                        for key in (
                            "initial_training_points",
                            "validation_points",
                            "final_training_points",
                            "completed_rounds",
                        )
                        if key in raw
                    },
                },
                selection=_selection("all predeclared tau/seed runs retained", "none"),
                upstream=upstream,
                sources=sources,
                artifacts=artifacts,
                native={"name": "regularized_native_value_prediction", "value": float(metrics["regularized_native_value_prediction"])},
                realized={"name": "J", "value": float(metrics["unregularized_realized_J"]), "evaluator": "tumor_problem.evaluate_zoh_control"},
                canonical={**control, "control_semantics": "breakpoint-aligned ZOH"},
            )
        ]
    if "closed-loop" in str(raw.get("benchmark", "")):
        artifacts.extend(
            _recorded_artifact_mapping(
                raw.get("artifact_sha256", {}), path.parent, artifact_root
            )
        )
        native_metrics = raw.get("author_script_reproduction")
        realized_metrics = raw.get("independent_realization")
        if not isinstance(native_metrics, Mapping) or not isinstance(
            realized_metrics, Mapping
        ):
            raise ValueError("HJB-NN satellite closed-loop manifest lacks cost metrics")
        seed = int(raw["seed"])
        return [
            _entry(
                run_id=f"hjb-nn-original-satellite-closed-loop-seed-{seed}",
                method_id="hjb_nn",
                method_name="Adaptive HJB-NN",
                claim_label="original-method reproduction",
                seeds=[seed],
                budget=dict(raw.get("protocol", {})),
                selection=_selection(
                    "predeclared RNG seed; released checkpoint; no objective-based selection",
                    "fixed-budget",
                ),
                upstream=upstream,
                sources=sources,
                artifacts=artifacts,
                native={
                    "name": "high_accuracy_noisy_ZOH_NN_J",
                    "value": float(realized_metrics["NN_J"]),
                    "diagnostics": {
                        "legacy_author_script_NN_cost": float(
                            native_metrics["NN_cost"]
                        ),
                        "evaluator": "per-interval DOP853 cost-state integration",
                    },
                },
            )
        ]
    # From-scratch satellite retraining is a separate original-benchmark claim
    # from released-checkpoint validation.  It must retain its declared seed,
    # adaptive stopping budget and every hashed training artifact.
    if "retraining" in str(raw.get("benchmark", "")):
        artifacts.extend(
            _recorded_artifact_mapping(
                raw.get("artifact_sha256", {}), path.parent, artifact_root
            )
        )
        metrics = raw.get("final_validation")
        if not isinstance(metrics, Mapping):
            raise ValueError("HJB-NN satellite retraining manifest lacks final validation metrics")
        seed = int(raw["seed"])
        return [
            _entry(
                run_id=f"hjb-nn-original-satellite-retrain-seed-{seed}",
                method_id="hjb_nn",
                method_name="Adaptive HJB-NN",
                claim_label="original-method reproduction",
                seeds=[seed],
                budget={
                    "initial_training_samples": int(raw["initial_training_samples"]),
                    "final_training_samples": int(raw["final_training_samples"]),
                    "completed_rounds": int(raw["completed_rounds"]),
                    "max_rounds": int(raw["max_rounds"]),
                    "min_rounds": int(raw["min_rounds"]),
                    "maxiter_per_round": int(raw["maxiter_per_round"]),
                    "maxfun_per_round": int(raw["maxfun_per_round"]),
                },
                selection=_selection(
                    "predeclared Algorithm 4.1 convergence rule; final completed round retained",
                    "native-training",
                ),
                upstream=upstream,
                sources=sources,
                artifacts=artifacts,
                native={
                    "name": "validation_value_RMAE",
                    "value": float(metrics["value_RMAE"]),
                },
            )
        ]
    # Released-checkpoint validation is deterministic and exports no tumor control.
    metrics = raw.get("recomputed_metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("HJB-NN original validation manifest lacks recomputed metrics")
    checkpoint = REPO_ROOT / raw["checkpoint"]
    sources.append(
        _file_record(
            checkpoint,
            REPO_ROOT,
            "released-checkpoint",
            expected=raw["checkpoint_sha256"],
            hash_origin="run-time method manifest",
        )
    )
    return [
        _entry(
            run_id="hjb-nn-original-satellite-validation",
            method_id="hjb_nn",
            method_name="Adaptive HJB-NN",
            claim_label="original-method reproduction",
            seeds=[],
            seed_policy="deterministic/no RNG",
            budget={
                "train_trajectories": raw["train_trajectories"],
                "validation_trajectories": raw["validation_trajectories"],
                "validation_points_t0": raw["validation_points_t0"],
            },
            selection=_selection("released checkpoint evaluated on fixed held-out author data", "fixed-budget"),
            upstream=upstream,
            sources=sources,
            artifacts=artifacts,
            native={"name": "validation_value_RMAE", "value": float(metrics["value_RMAE"])},
        )
    ]


ADAPTERS = {
    "deepbsde": _deepbsde,
    "neural_pmp": _neural_pmp,
    "pi_deeponet": _pi_deeponet,
    "lyznet": _lyznet,
    "hjb_nn": _hjb_nn,
}


def compose_root_manifest(
    inputs: Sequence[tuple[str, Path]],
    *,
    artifact_root: Path,
    source_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    artifact_root = artifact_root.resolve()
    if source_root.resolve() != REPO_ROOT.resolve():
        raise ValueError("current adapters resolve audited sources from the repository root")
    runs = []
    for method_id, path in inputs:
        if method_id not in ADAPTERS:
            raise KeyError(f"no normalized importer for {method_id!r}")
        path = path.resolve()
        _inside(path, artifact_root, "method manifest")
        runs.extend(ADAPTERS[method_id](path, _load(path), artifact_root))
    manifest = {
        "schema_version": 1,
        "manifest_type": "faithful-related-work-root",
        "created_at_utc": _utc_now(),
        "protocol": "faithful_related_work/FIDELITY_CONTRACT.md",
        "runs": runs,
    }
    validate_root_manifest(
        manifest,
        artifact_root=artifact_root,
        source_root=source_root,
        verify_hashes=True,
    )
    return manifest


def _input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("input must be METHOD_ID=/path/to/manifest.json")
    method, path = value.split("=", 1)
    if not method or not path:
        raise argparse.ArgumentTypeError("input must be METHOD_ID=/path/to/manifest.json")
    return method, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=_input, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = compose_root_manifest(
        args.input,
        artifact_root=args.artifact_root,
        source_root=args.source_root,
    )
    prepare_empty_output_directory(args.output_dir)
    (args.output_dir / "root_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
