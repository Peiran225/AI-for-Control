"""Recompute and rank canonical tumor controls under the common evaluator."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from tumor_problem import NOMINAL_TUMOR_PROBLEM, evaluate_zoh_control, serializable_metrics

from .manifest import validate_root_manifest
from .registry import REPO_ROOT
from .smoke import prepare_empty_output_directory


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_control(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as artifact:
        if "t" not in artifact or "u" not in artifact:
            raise ValueError(f"canonical control must contain t and u arrays: {path}")
        t = np.asarray(artifact["t"], dtype=np.float64)
        u = np.asarray(artifact["u"], dtype=np.float64)
    if t.ndim != 1 or u.ndim != 1:
        raise ValueError(f"canonical tumor control must have one-dimensional t and u: {path}")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(u)):
        raise ValueError(f"canonical tumor control contains non-finite values: {path}")
    return t, u


def recompute_tumor_summary(
    manifest: Mapping[str, Any],
    *,
    artifact_root: Path,
    source_root: Path | None = None,
    verify_hashes: bool = True,
    stored_tolerance: float = 1.0e-8,
) -> dict[str, Any]:
    """Evaluate only completed entries explicitly labelled ``tumor adaptation``."""

    artifact_root = Path(artifact_root).resolve()
    if source_root is None:
        source_root = REPO_ROOT
    validate_root_manifest(
        manifest,
        artifact_root=artifact_root,
        source_root=source_root,
        verify_hashes=verify_hashes,
    )
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for run in manifest["runs"]:
        if run["status"] != "completed" or run["claim_label"] != "tumor adaptation":
            excluded.append(
                {
                    "run_id": str(run["run_id"]),
                    "reason": "not a completed tumor adaptation",
                }
            )
            continue
        canonical = run["canonical_control"]
        control_path = artifact_root / canonical["path"]
        t, u = _load_control(control_path)
        result = evaluate_zoh_control(
            t,
            u,
            NOMINAL_TUMOR_PROBLEM,
            include_diagnostics=False,
        )
        metrics = serializable_metrics(result)
        stored = run["metrics"]["realized"]
        if stored is not None and not np.isclose(
            float(stored["value"]), float(result["J"]), rtol=stored_tolerance, atol=stored_tolerance
        ):
            raise ValueError(
                f"{run['run_id']}: stored realized J={stored['value']} disagrees with "
                f"independent recomputation {result['J']}"
            )
        native = run["metrics"]["native"]
        rows.append(
            {
                "run_id": str(run["run_id"]),
                "method_id": str(run["method_id"]),
                "method_name": str(run["method_name"]),
                "seeds": list(run["seeds"]),
                "native_metric_name": None if native is None else native["name"],
                "native_metric_value": None if native is None else float(native["value"]),
                "realized_J": float(result["J"]),
                "running_cost": float(result["running_cost"]),
                "terminal_cost": float(result["terminal_cost"]),
                "control_path": str(canonical["path"]),
                "canonical_metrics": metrics,
            }
        )
    rows.sort(key=lambda row: (row["realized_J"], row["run_id"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return {
        "schema_version": 1,
        "summary_type": "canonical-tumor-realized-J",
        "created_at_utc": _utc_now(),
        "evaluator": "tumor_problem.evaluate_zoh_control",
        "control_semantics": "breakpoint-aligned ZOH",
        "included_claim_label": "tumor adaptation",
        "rows": rows,
        "excluded": excluded,
    }


def write_summary(summary: Mapping[str, Any], output_dir: Path) -> None:
    output_dir = Path(output_dir)
    prepare_empty_output_directory(output_dir)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = [
            "rank",
            "run_id",
            "method_id",
            "method_name",
            "seeds",
            "native_metric_name",
            "native_metric_value",
            "realized_J",
            "running_cost",
            "terminal_cost",
            "control_path",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summary["rows"]:
            writer.writerow({field: row[field] for field in fields})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-hash-verification", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_root = args.artifact_root.resolve() if args.artifact_root else manifest_path.parent
    summary = recompute_tumor_summary(
        manifest,
        artifact_root=artifact_root,
        source_root=REPO_ROOT if args.source_root is None else args.source_root.resolve(),
        verify_hashes=not args.skip_hash_verification,
    )
    write_summary(summary, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
