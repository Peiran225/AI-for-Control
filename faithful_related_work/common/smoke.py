"""Run every declared original-benchmark smoke and emit one root manifest."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .manifest import sha256_file, validate_root_manifest
from .registry import REPO_ROOT, SmokeSpec, smoke_specs


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def prepare_empty_output_directory(path: Path) -> None:
    """Create ``path`` while refusing to reuse any non-empty directory."""

    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _write_capture(directory: Path, completed: subprocess.CompletedProcess[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (directory / "stderr.txt").write_text(completed.stderr, encoding="utf-8")


def _load_json(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _json_stdout(text: str) -> Mapping[str, Any] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, Mapping) else None


def _native_metric(spec: SmokeSpec, raw: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    if spec.method_id == "deepbsde" and "final_terminal_matching_loss" in raw:
        return {"name": "terminal_matching_loss", "value": float(raw["final_terminal_matching_loss"])}
    if spec.method_id == "neural_pmp":
        aggregate = raw.get("aggregate")
        if isinstance(aggregate, Mapping) and "mean_true_objective_post_selection" in aggregate:
            return {
                "name": "mean_original_LQR_objective_post_selection",
                "value": float(aggregate["mean_true_objective_post_selection"]),
            }
    if spec.method_id == "hjb_nn":
        recomputed = raw.get("recomputed_metrics")
        if isinstance(recomputed, Mapping) and "value_RMAE" in recomputed:
            return {"name": "validation_value_RMAE", "value": float(recomputed["value_RMAE"])}
    summary = raw.get("summary")
    if isinstance(summary, Mapping) and "metric" in summary and "mean" in summary:
        return {"name": str(summary["metric"]), "value": float(summary["mean"])}
    return None


def _source_records(spec: SmokeSpec, command: Sequence[str]) -> list[dict[str, Any]]:
    paths = list(spec.source_paths)
    for token in command:
        if token.endswith(".py") and (REPO_ROOT / token).is_file() and token not in paths:
            paths.append(token)
    records = []
    for relative in paths:
        path = REPO_ROOT / relative
        if path.is_file():
            records.append(
                {
                    "path": relative,
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "role": "runner-or-upstream-source",
                }
            )
    return records


def _artifact_records(root: Path, method_dir: Path) -> list[dict[str, Any]]:
    result = []
    if not method_dir.is_dir():
        return result
    for path in sorted(method_dir.rglob("*")):
        if path.is_file():
            result.append(
                {
                    "path": str(path.relative_to(root)),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "role": "smoke-output",
                }
            )
    return result


def _base_entry(spec: SmokeSpec) -> dict[str, Any]:
    return {
        "run_id": f"{spec.method_id}-original-smoke",
        "method_id": spec.method_id,
        "method_name": spec.method_name,
        "upstream": {
            "kind": spec.upstream_kind,
            "revision": spec.upstream_revision,
        },
        "claim_label": spec.claim_label,
        "status": "planned",
        "seeds": list(spec.seeds),
        "seed_policy": "explicit RNG seeds" if spec.seeds else "deterministic/no RNG",
        "budget": {"kind": "smoke", "parameters": dict(spec.budget)},
        "selection": {
            "rule": spec.selection_rule,
            "metric_scope": spec.selection_metric_scope,
            "uses_realized_objective": False,
            "candidates_retained": True,
        },
        "tumor_comparable": False,
        "metrics": {"native": None, "realized": None},
        "sources": [],
        "artifacts": [],
    }


def run_one_smoke(spec: SmokeSpec, output_root: Path, python: str) -> dict[str, Any]:
    method_dir = output_root / spec.method_id
    command = tuple(spec.command(python, method_dir))
    entry = _base_entry(spec)
    entry["command"] = list(command)
    entry["sources"] = _source_records(spec, command)

    if not command:
        method_dir.mkdir(parents=True, exist_ok=True)
        entry.update(
            {
                "claim_label": "explicitly unavailable",
                "status": "unavailable",
                "reason": "no declared smoke runner was discovered",
            }
        )
        entry["artifacts"] = _artifact_records(output_root, method_dir)
        return entry

    preflight: Mapping[str, Any] | None = None
    if spec.availability_probe is not None:
        probe = _run(spec.availability_probe(python))
        preflight = _json_stdout(probe.stdout)
        if probe.returncode != 0 or preflight is None or not bool(preflight.get("local_ready", False)):
            method_dir.mkdir(parents=True, exist_ok=True)
            (method_dir / "preflight.json").write_text(
                json.dumps(preflight or {"stdout": probe.stdout, "stderr": probe.stderr}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            entry.update(
                {
                    "claim_label": "explicitly unavailable",
                    "status": "unavailable",
                    "reason": "real dReal binding is unavailable locally; no stub or fallback was used",
                }
            )
            entry["artifacts"] = _artifact_records(output_root, method_dir)
            return entry

    completed = _run(command)
    _write_capture(method_dir, completed)
    if preflight is not None:
        (method_dir / "preflight.json").write_text(
            json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    raw = (
        _json_stdout(completed.stdout)
        if spec.stdout_is_json
        else _load_json(method_dir / str(spec.output_manifest))
    )
    if spec.method_id == "neural_pmp" and completed.returncode == 0:
        summary = _load_json(method_dir / "run_config_and_summary.json")
        if summary is not None:
            raw = summary
    if completed.returncode == 0:
        entry["status"] = "completed"
        entry["metrics"]["native"] = _native_metric(spec, raw)
    else:
        entry["status"] = "failed"
        entry["reason"] = f"smoke command exited with status {completed.returncode}"
    entry["artifacts"] = _artifact_records(output_root, method_dir)
    return entry


def run_smokes(
    output_dir: Path,
    *,
    method_ids: Sequence[str] | None = None,
    python: str = sys.executable,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    prepare_empty_output_directory(output_dir)
    runs = [run_one_smoke(spec, output_dir, python) for spec in smoke_specs(method_ids)]
    manifest = {
        "schema_version": 1,
        "manifest_type": "faithful-related-work-root",
        "created_at_utc": _utc_now(),
        "protocol": "faithful_related_work/FIDELITY_CONTRACT.md",
        "runs": runs,
    }
    validate_root_manifest(
        manifest,
        artifact_root=output_dir,
        source_root=REPO_ROOT,
        verify_hashes=True,
    )
    (output_dir / "root_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--methods",
        default="all",
        help="comma-separated method ids, or all (deepbsde,neural_pmp,pi_deeponet,lyznet,hjb_nn)",
    )
    parser.add_argument("--python", default=sys.executable)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    method_ids = None if args.methods == "all" else [item.strip() for item in args.methods.split(",") if item.strip()]
    manifest = run_smokes(args.output_dir, method_ids=method_ids, python=args.python)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
