#!/usr/bin/env python3
"""Patch timing provenance in retained strict-benchmark results without recomputation.

The default mode only validates and reports planned changes.  Pass ``--write``
to update benchmark_results.json, construction_times.csv, and
deployment_times.csv atomically.  One backup of each original file is retained.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "outputs/fair_efficiency_benchmark_20260720/strict_unified_benchmark"
DEFAULT_MANIFEST = ROOT / "outputs/fair_efficiency_benchmark_20260720/matched_n800_training_timing.json"

DIRECT_J = "Direct-J Transformer"
PMP = "PMP/KKT Transformer"
MLP = "PMP/KKT MLP"
REFINED = "Transformer with projected-gradient refinement"
NEURAL = "Neural-PMP exact-dynamics PMP-gradient schedule"
DIRECT = "Direct time mesh n=800"
MANIFEST_NAMES = {
    DIRECT_J: "Direct-J Transformer",
    PMP: "PMP/KKT Transformer",
    MLP: "PMP/KKT TimeMLP",
    NEURAL: "Known-dynamics Neural-PMP",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_artifact(text: str) -> Path:
    path = Path(text).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def bool_key(key: str) -> bool:
    return (
        key == "success"
        or key.startswith("passes_")
        or key.endswith("_eligible")
        or key.endswith("_used")
    )


def restore_json_booleans(value: Any, key: str = "") -> Any:
    if isinstance(value, dict):
        return {name: restore_json_booleans(item, name) for name, item in value.items()}
    if isinstance(value, list):
        return [restore_json_booleans(item, key) for item in value]
    if bool_key(key) and type(value) is int and value in (0, 1):
        return bool(value)
    return value


def read_manifest(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("runs"), list):
        raise ValueError(f"invalid timing manifest: {path}")
    matched = payload.get("matched_configuration", {})
    expected = {
        "grid_intervals": 800,
        "grid_nodes": 801,
        "dtype": "float64",
        "device": "cpu",
    }
    for key, expected_value in expected.items():
        if matched.get(key) != expected_value:
            raise ValueError(f"manifest {key} mismatch")
    by_name = {str(row.get("method")): row for row in payload["runs"]}
    return payload, by_name


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def backup_once(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".pre_timing_metadata.bak")
    if not backup.exists():
        shutil.copy2(path, backup)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    results_dir = args.results_dir.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    results_path = results_dir / "benchmark_results.json"
    construction_path = results_dir / "construction_times.csv"
    deployment_path = results_dir / "deployment_times.csv"
    for path in (results_path, construction_path, deployment_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    results = restore_json_booleans(json.loads(results_path.read_text(encoding="utf-8")))
    manifest, manifest_by_name = read_manifest(manifest_path)
    manifest_hash = sha256(manifest_path)
    construction = results["construction"]
    deployment = results["deployment"]
    deployment_by_method = {row["method"]: row for row in deployment}

    artifacts: dict[str, Path] = {}
    for method in (DIRECT_J, PMP, MLP, NEURAL):
        deployment_row = deployment_by_method.get(method)
        if deployment_row is not None:
            artifacts[method] = resolve_artifact(str(deployment_row["artifact"]))

    manifest_entries: dict[str, dict[str, Any]] = {}
    for method, artifact in artifacts.items():
        manifest_name = MANIFEST_NAMES[method]
        entry = manifest_by_name.get(manifest_name)
        if not isinstance(entry, dict):
            raise ValueError(f"manifest entry missing for {method}")
        manifest_artifact = resolve_artifact(str(entry.get("checkpoint", "")))
        if artifact != manifest_artifact:
            raise ValueError(
                f"artifact mismatch for {method}: results={artifact}, manifest={manifest_artifact}"
            )
        manifest_entries[method] = entry

    for row in construction:
        method = row["method"]
        if method in manifest_entries:
            entry = manifest_entries[method]
            field = "training_loop_wall_seconds" if method == DIRECT_J else "process_wall_seconds"
            observed = row.get("seconds")
            expected_seconds = entry.get(field)
            if observed is not None and (
                expected_seconds is None
                or not math.isclose(
                    float(observed), float(expected_seconds), rel_tol=0.0, abs_tol=1.0e-9
                )
            ):
                raise ValueError(
                    f"timing mismatch for {method}: results={observed}, manifest={expected_seconds}"
                )
            row["timing_scope"] = entry["timing_scope"]
            row["source"] = (
                f"artifact {artifacts[method]}; timing manifest {manifest_path} "
                f"(sha256 {manifest_hash})"
            )
            row["training_timing_manifest"] = str(manifest_path)
            row["training_timing_manifest_sha256"] = manifest_hash
        elif method == DIRECT:
            if "complete fresh" in str(row.get("scope", "")):
                row["timing_scope"] = (
                    "Complete in-process n=200 -> n=400 -> n=800 optimizer chain including "
                    "interpolation; excludes artifact serialization, plotting, state/derivative "
                    "quality audits, and CSV/JSON output."
                )
            else:
                row["timing_scope"] = (
                    "L-BFGS-B optimizer wall time for this stage only; excludes interpolation, "
                    "artifact serialization, plotting, state/derivative quality audits, and CSV/JSON output."
                )
            row["training_timing_manifest"] = None
            row["training_timing_manifest_sha256"] = None
        elif method == REFINED:
            row["timing_scope"] = (
                "Sum of seven recorded continuation-stage training wall times only; the "
                "original base-checkpoint construction time is unavailable."
            )
            row["training_timing_manifest"] = None
            row["training_timing_manifest_sha256"] = None

    for row in deployment:
        method = row["method"]
        if method in (DIRECT_J, PMP, MLP, REFINED):
            row["timing_scope"] = (
                "Loaded-control preparation and joint deployment are timed separately; the "
                "joint block excludes checkpoint reload, artifact writes, plotting, DOP853 "
                "evaluation, and gradient/Hessian audits. Hot-process reload is reported separately."
            )
        elif method.startswith("Cached Direct time mesh"):
            row["timing_scope"] = (
                "Joint deployment includes the in-memory schedule copy and state rollout; it "
                "excludes NPZ reload, artifact writes, plotting, DOP853 evaluation, and "
                "gradient/Hessian audits. Hot-process NPZ reload is reported separately."
            )
        elif method == NEURAL:
            row["timing_scope"] = (
                "Joint deployment includes the in-memory schedule copy and state rollout; it "
                "excludes NPZ reload, artifact writes, plotting, DOP853 evaluation, and "
                "gradient/Hessian audits. No network inference or reload is applicable."
            )

    protocol = results["metadata"]["protocol"]
    protocol.pop("timing_excludes", None)
    protocol["timing_scope_policy"] = (
        "Use each construction/deployment row's timing_scope field. Direct solver, network "
        "training-loop, full-process training, and deployment timings have different explicitly "
        "recorded boundaries."
    )
    results["metadata"]["training_timing_manifest"] = {
        "path": str(manifest_path),
        "sha256": manifest_hash,
        "benchmark_date": manifest.get("benchmark_date"),
        "hardware": manifest.get("hardware"),
        "matched_configuration": manifest.get("matched_configuration"),
        "validated_methods": sorted(artifacts),
    }

    if not args.write:
        print(
            f"validated {len(artifacts)} manifest-bound methods; "
            f"would update {results_dir} (pass --write to apply)"
        )
        return

    for path in (results_path, construction_path, deployment_path):
        backup_once(path)
    temporary_json = results_path.with_suffix(results_path.suffix + ".tmp")
    temporary_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    temporary_json.replace(results_path)
    write_csv(construction_path, construction)
    write_csv(deployment_path, deployment)
    print(f"updated timing metadata in {results_dir}; numerical results were not recomputed")


if __name__ == "__main__":
    main()
