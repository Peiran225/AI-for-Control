"""Strict schema and provenance checks for cross-method run manifests."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1
MANIFEST_TYPE = "faithful-related-work-root"
CLAIM_LABELS = frozenset(
    {
        "original-method reproduction",
        "tumor adaptation",
        "non-comparable adaptation",
        "explicitly unavailable",
    }
)
RUN_STATUSES = frozenset({"completed", "failed", "unavailable", "planned"})
METRIC_SCOPES = frozenset({"none", "native-training", "native-validation", "fixed-budget"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ManifestValidationError(ValueError):
    """Raised when a normalized manifest violates the comparison protocol."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fail(location: str, message: str) -> None:
    raise ManifestValidationError(f"{location}: {message}")


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(location, "must be an object")
    return value


def _nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(location, "must be a non-empty string")
    return value


def _finite_number(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(location, "must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        _fail(location, "must be a finite number")
    return result


def _relative_path(value: Any, location: str) -> Path:
    text = _nonempty_string(value, location)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        _fail(location, "must be a relative path without '..'")
    return path


def _validate_hash_record(
    record: Any,
    location: str,
    *,
    root: Path | None,
    verify_hashes: bool,
) -> None:
    item = _mapping(record, location)
    path = _relative_path(item.get("path"), f"{location}.path")
    digest = item.get("sha256")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        _fail(f"{location}.sha256", "must be a lowercase 64-character SHA-256")
    if "role" in item:
        _nonempty_string(item["role"], f"{location}.role")
    if "bytes" in item and (not isinstance(item["bytes"], int) or item["bytes"] < 0):
        _fail(f"{location}.bytes", "must be a non-negative integer")
    if verify_hashes:
        if root is None:
            _fail(location, "hash verification requires a filesystem root")
        resolved = root / path
        if not resolved.is_file():
            _fail(f"{location}.path", f"file does not exist under hash root: {path}")
        actual = sha256_file(resolved)
        if actual != digest:
            _fail(f"{location}.sha256", f"hash mismatch for {path}: {actual}")
        if "bytes" in item and resolved.stat().st_size != item["bytes"]:
            _fail(f"{location}.bytes", f"size mismatch for {path}")


def _validate_metric(metric: Any, location: str, *, realized: bool) -> None:
    if metric is None:
        return
    item = _mapping(metric, location)
    name = _nonempty_string(item.get("name"), f"{location}.name")
    _finite_number(item.get("value"), f"{location}.value")
    if realized:
        if name != "J":
            _fail(f"{location}.name", "canonical realized metric must be named 'J'")
        _nonempty_string(item.get("evaluator"), f"{location}.evaluator")
    elif "evaluator" in item:
        _fail(location, "native metrics must not claim the canonical evaluator")


def _validate_budget(value: Any, location: str, *, required: bool) -> None:
    budget = _mapping(value, location)
    kind = _nonempty_string(budget.get("kind"), f"{location}.kind")
    parameters = _mapping(budget.get("parameters"), f"{location}.parameters")
    if required and not parameters:
        _fail(f"{location}.parameters", f"completed {kind!r} run needs an explicit budget")
    for name, amount in parameters.items():
        _nonempty_string(name, f"{location}.parameters key")
        if amount is not None and not isinstance(amount, (bool, int, float, str, list)):
            _fail(f"{location}.parameters.{name}", "must be null, scalar text/number, or a list")


def _validate_selection(value: Any, location: str, *, status: str) -> None:
    selection = _mapping(value, location)
    _nonempty_string(selection.get("rule"), f"{location}.rule")
    scope = selection.get("metric_scope")
    if scope not in METRIC_SCOPES:
        _fail(f"{location}.metric_scope", f"must be one of {sorted(METRIC_SCOPES)}")
    uses_realized = selection.get("uses_realized_objective")
    if not isinstance(uses_realized, bool):
        _fail(f"{location}.uses_realized_objective", "must be boolean")
    if uses_realized:
        _fail(
            f"{location}.uses_realized_objective",
            "post-hoc selection on nominal realized J is forbidden",
        )
    retained = selection.get("candidates_retained")
    if not isinstance(retained, bool):
        _fail(f"{location}.candidates_retained", "must be boolean")
    if status == "completed" and not retained:
        _fail(
            f"{location}.candidates_retained",
            "completed runs must retain every predeclared candidate",
        )


def validate_run_entry(
    value: Any,
    location: str,
    *,
    artifact_root: Path | None = None,
    source_root: Path | None = None,
    verify_hashes: bool = False,
) -> None:
    run = _mapping(value, location)
    for field in ("run_id", "method_id", "method_name"):
        _nonempty_string(run.get(field), f"{location}.{field}")

    upstream = _mapping(run.get("upstream"), f"{location}.upstream")
    _nonempty_string(upstream.get("kind"), f"{location}.upstream.kind")
    revision = upstream.get("revision")
    if revision is None:
        _nonempty_string(
            upstream.get("revision_unavailable_reason"),
            f"{location}.upstream.revision_unavailable_reason",
        )
    else:
        _nonempty_string(revision, f"{location}.upstream.revision")

    claim = run.get("claim_label")
    if claim not in CLAIM_LABELS:
        _fail(f"{location}.claim_label", f"must be one of {sorted(CLAIM_LABELS)}")
    status = run.get("status")
    if status not in RUN_STATUSES:
        _fail(f"{location}.status", f"must be one of {sorted(RUN_STATUSES)}")
    if status == "unavailable" and claim != "explicitly unavailable":
        _fail(f"{location}.claim_label", "unavailable runs must be explicitly labelled")
    if claim == "explicitly unavailable" and status != "unavailable":
        _fail(f"{location}.status", "explicitly unavailable label requires unavailable status")

    seeds = run.get("seeds")
    if not isinstance(seeds, list) or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        _fail(f"{location}.seeds", "must be a list of integer RNG seeds")
    if len(seeds) != len(set(seeds)):
        _fail(f"{location}.seeds", "must not contain duplicate seeds")
    seed_policy = run.get("seed_policy")
    if seed_policy not in {"explicit RNG seeds", "deterministic/no RNG"}:
        _fail(
            f"{location}.seed_policy",
            "must declare either explicit RNG seeds or a deterministic no-RNG run",
        )
    if seed_policy == "explicit RNG seeds" and status == "completed" and not seeds:
        _fail(f"{location}.seeds", "completed stochastic runs require at least one declared seed")
    if seed_policy == "deterministic/no RNG" and seeds:
        _fail(f"{location}.seeds", "deterministic no-RNG runs must not invent a seed")

    _validate_budget(run.get("budget"), f"{location}.budget", required=status == "completed")
    _validate_selection(run.get("selection"), f"{location}.selection", status=status)

    tumor_comparable = run.get("tumor_comparable")
    if not isinstance(tumor_comparable, bool):
        _fail(f"{location}.tumor_comparable", "must be boolean")
    if tumor_comparable != (claim == "tumor adaptation"):
        _fail(
            f"{location}.tumor_comparable",
            "only an explicitly labelled tumor adaptation enters the tumor ranking",
        )

    metrics = _mapping(run.get("metrics"), f"{location}.metrics")
    if set(metrics) != {"native", "realized"}:
        _fail(
            f"{location}.metrics",
            "must contain exactly separate 'native' and 'realized' fields",
        )
    _validate_metric(metrics["native"], f"{location}.metrics.native", realized=False)
    _validate_metric(metrics["realized"], f"{location}.metrics.realized", realized=True)
    if claim != "tumor adaptation" and metrics["realized"] is not None:
        _fail(
            f"{location}.metrics.realized",
            "non-tumor benchmarks cannot enter the common realized-J comparison",
        )

    sources = run.get("sources")
    artifacts = run.get("artifacts")
    if not isinstance(sources, list) or (status == "completed" and not sources):
        _fail(f"{location}.sources", "completed runs need a non-empty source hash list")
    if not isinstance(artifacts, list):
        _fail(f"{location}.artifacts", "must be a list")
    for index, source in enumerate(sources):
        _validate_hash_record(
            source,
            f"{location}.sources[{index}]",
            root=source_root,
            verify_hashes=verify_hashes,
        )
    for index, artifact in enumerate(artifacts):
        _validate_hash_record(
            artifact,
            f"{location}.artifacts[{index}]",
            root=artifact_root,
            verify_hashes=verify_hashes,
        )

    canonical = run.get("canonical_control")
    if status == "completed" and claim == "tumor adaptation":
        if canonical is None:
            _fail(f"{location}.canonical_control", "completed tumor adaptation requires canonical t,u NPZ")
        _validate_hash_record(
            canonical,
            f"{location}.canonical_control",
            root=artifact_root,
            verify_hashes=verify_hashes,
        )
        canonical_item = _mapping(canonical, f"{location}.canonical_control")
        if canonical_item.get("control_semantics") != "breakpoint-aligned ZOH":
            _fail(
                f"{location}.canonical_control.control_semantics",
                "must be 'breakpoint-aligned ZOH'",
            )
    elif canonical is not None:
        _fail(f"{location}.canonical_control", "only a completed tumor adaptation may export this field")

    if status in {"failed", "unavailable"}:
        _nonempty_string(run.get("reason"), f"{location}.reason")


def validate_root_manifest(
    value: Any,
    *,
    artifact_root: Path | None = None,
    source_root: Path | None = None,
    verify_hashes: bool = False,
) -> None:
    """Validate a normalized root manifest and all selection invariants.

    ``artifact_root`` resolves run-output paths; ``source_root`` resolves
    repository source paths.  Hash verification is opt-in so manifests remain
    portable, but hash syntax and path safety are always checked.
    """

    manifest = _mapping(value, "manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        _fail("manifest.schema_version", f"must equal {SCHEMA_VERSION}")
    if manifest.get("manifest_type") != MANIFEST_TYPE:
        _fail("manifest.manifest_type", f"must equal {MANIFEST_TYPE!r}")
    _nonempty_string(manifest.get("created_at_utc"), "manifest.created_at_utc")
    _nonempty_string(manifest.get("protocol"), "manifest.protocol")
    runs = manifest.get("runs")
    if not isinstance(runs, list):
        _fail("manifest.runs", "must be a list")
    run_ids: set[str] = set()
    for index, run in enumerate(runs):
        validate_run_entry(
            run,
            f"manifest.runs[{index}]",
            artifact_root=artifact_root,
            source_root=source_root,
            verify_hashes=verify_hashes,
        )
        run_id = str(run["run_id"])
        if run_id in run_ids:
            _fail(f"manifest.runs[{index}].run_id", "must be unique")
        run_ids.add(run_id)
