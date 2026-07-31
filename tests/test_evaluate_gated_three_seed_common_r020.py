from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
TESTS = ROOT / "tests"
for search_path in (ROOT, SCRIPTS, TESTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from evaluate_gated_three_seed_common_r020 import (  # noqa: E402
    COMMON_COUNT,
    COMMON_EVALUATION_SEED,
    COMMON_RADIUS,
    KNOWN_FORMAL_OR_RESERVED_SEEDS,
    MANIFEST_SCHEMA,
    common_initial_states,
    load_manifest,
    parser,
    run,
    sha256_file,
)
from test_evaluate_feedback_gated_probe_branch_blind import (  # noqa: E402
    _write_tiny_artifact,
)


def _write_manifest(
    path: Path,
    candidate: Path,
    locked: Path,
) -> Path:
    lanes = []
    for index, lane in enumerate(("base4", "base2", "base3")):
        candidate_copy = path.parent / f"candidate_{index}.pt"
        locked_copy = path.parent / f"locked_{index}.pt"
        shutil.copyfile(candidate, candidate_copy)
        shutil.copyfile(locked, locked_copy)
        lanes.append(
            {
                "lane": lane,
                "candidate": {
                    "path": candidate_copy.name,
                    "sha256": sha256_file(candidate_copy),
                },
                "locked_cf": {
                    "path": locked_copy.name,
                    "sha256": sha256_file(locked_copy),
                },
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema": MANIFEST_SCHEMA,
                "lanes": lanes,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_common_initial_states_are_fixed_shared_r020() -> None:
    cfg = SimpleNamespace(m=21, n0=10.0)
    first = common_initial_states(cfg)
    second = common_initial_states(cfg)
    assert COMMON_EVALUATION_SEED not in KNOWN_FORMAL_OR_RESERVED_SEEDS
    assert first.shape == (COMMON_COUNT, cfg.m)
    assert first.dtype == torch.float64
    assert first.equal(second)
    assert float(first.min()) >= cfg.n0 * (1.0 - COMMON_RADIUS)
    assert float(first.max()) <= cfg.n0 * (1.0 + COMMON_RADIUS)


def test_manifest_requires_exact_hashes_and_three_unique_lanes(
    tmp_path: Path,
) -> None:
    artifact, locked = _write_tiny_artifact(tmp_path)
    manifest = _write_manifest(tmp_path / "manifest.json", artifact, locked)
    lanes, manifest_hash = load_manifest(manifest)
    assert [lane.lane for lane in lanes] == ["base4", "base2", "base3"]
    assert manifest_hash == sha256_file(manifest)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["lanes"][0]["candidate"]["sha256"] = "not-a-hash"
    bad_manifest = tmp_path / "bad_manifest.json"
    bad_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        load_manifest(bad_manifest)


def test_cli_does_not_expose_seed_count_or_radius(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parser().parse_args(
            [
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--out-dir",
                str(tmp_path / "out"),
                "--seed",
                "1",
            ]
        )


def test_run_rejects_a_well_formed_but_wrong_candidate_hash(
    tmp_path: Path,
) -> None:
    artifact, locked = _write_tiny_artifact(tmp_path)
    manifest = _write_manifest(tmp_path / "manifest.json", artifact, locked)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["lanes"][0]["candidate"]["sha256"] = "0" * 64
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args = parser().parse_args(
        [
            "--manifest",
            str(manifest),
            "--out-dir",
            str(tmp_path / "failed_output"),
            "--device",
            "cpu",
            "--threads",
            "1",
        ]
    )
    with pytest.raises(RuntimeError, match="candidate SHA256 mismatch"):
        run(args)


def test_cpu_common_three_lane_smoke_is_exclusive_and_does_not_touch_ledger(
    tmp_path: Path,
) -> None:
    artifact, locked = _write_tiny_artifact(tmp_path)
    manifest = _write_manifest(tmp_path / "manifest.json", artifact, locked)
    output = tmp_path / "common_eval"
    ledger = (
        ROOT
        / "outputs"
        / "formal_blind_consumption"
        / "one_shot_blind_seed20261701.jsonl"
    )
    ledger_before = sha256_file(ledger) if ledger.is_file() else None
    args = parser().parse_args(
        [
            "--manifest",
            str(manifest),
            "--out-dir",
            str(output),
            "--device",
            "cpu",
            "--threads",
            "1",
        ]
    )
    summary = run(args)
    assert summary["evaluation"]["seed"] == COMMON_EVALUATION_SEED
    assert summary["evaluation"]["count"] == COMMON_COUNT
    assert summary["evaluation"]["formal_seed_or_ledger_used"] is False
    assert len(summary["lanes"]) == 3
    assert all(
        all(value == 0.0 for value in lane["protected_identity"].values())
        for lane in summary["lanes"]
    )
    initial = np.load(output / "initial_states.npy", allow_pickle=False)
    assert initial.shape == (COMMON_COUNT, summary["problem"]["m"])
    rows = (output / "per_sample.csv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(rows) == 1 + 3 * COMMON_COUNT
    provenance = json.loads(
        (output / "provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["formal_seed_or_ledger_used"] is False
    for filename in ("initial_states.npy", "per_sample.csv", "summary.json"):
        assert (
            provenance["output_file_sha256"][filename]
            == sha256_file(output / filename)
        )
    ledger_after = sha256_file(ledger) if ledger.is_file() else None
    assert ledger_after == ledger_before
    with pytest.raises(FileExistsError):
        run(args)
