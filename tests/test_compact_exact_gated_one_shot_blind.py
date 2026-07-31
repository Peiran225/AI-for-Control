from __future__ import annotations

import json
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.evaluate_compact_exact_gated_one_shot_blind import (
    BOOTSTRAP_REPEATS,
    BOOTSTRAP_SEED,
    FORMAL_COUNT,
    FORMAL_RADIUS,
    FORMAL_SEED,
    PROTOCOL,
    ConsumptionLedger,
    assert_exact_protected_identity,
    build_parser,
    canonical_continuous_objective,
    paired_bootstrap_mean,
    per_sample_records,
    protected_identity_metrics,
    run,
    tensor_commitment,
    validate_formal_artifact_metadata,
    write_json_exclusive,
)
from train_paper_pmp_kkt import ProblemConfig, build_params


class _ConstantGate(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(
        self,
        normalized_time: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        return torch.full_like(normalized_time, self.value)


class _ToyContinuousPolicy(torch.nn.Module):
    def __init__(
        self,
        *,
        gate: _ConstantGate | None = None,
        residual: float = 0.0,
    ) -> None:
        super().__init__()
        self.gate = gate
        self.residual = residual

    def time_logits(self, normalized_time: torch.Tensor) -> torch.Tensor:
        return 0.2 + 0.1 * normalized_time

    def interval_action(
        self,
        base_logit: torch.Tensor,
        normalized_time: torch.Tensor,
        state: torch.Tensor,
        *,
        state_mode: str,
    ) -> torch.Tensor:
        action = base_logit.expand(state.shape[0])
        if self.gate is not None and state_mode == "feedback":
            action = action + self.residual * self.gate(
                normalized_time,
                state,
            )
        return action


def valid_metadata() -> dict[str, object]:
    return {
        "format": "exact_gated_compact_residual_adapter_v1",
        "protected_radii": [0.0, 0.10, 0.20],
        "source_checkpoint": "/frozen/source.pt",
        "source_checkpoint_sha256": "a" * 64,
        "protected_identity": {
            "gate_max_abs": 0.0,
            "control_max_abs": 0.0,
            "state_max_abs": 0.0,
            "objective_max_abs": 0.0,
        },
        "seeds": {
            "teacher_train": 11,
            "teacher_validation": 12,
            "teacher_optimizer": 13,
            "cleanup_train": 21,
            "cleanup_validation": 22,
            "cleanup_optimizer": 23,
            "development_audit": 31,
            "reserved_final_blind": FORMAL_SEED,
            "reserved_final_blind_used": False,
        },
    }


def test_formal_protocol_is_hard_locked_and_not_cli_adjustable() -> None:
    assert PROTOCOL == "compact_exact_gated_one_shot_blind_v1"
    assert FORMAL_SEED == 20261701
    assert FORMAL_COUNT == 128
    assert FORMAL_RADIUS == 0.20
    assert BOOTSTRAP_REPEATS == 20_000
    assert BOOTSTRAP_SEED == 20261702

    parser = build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--seed" not in option_strings
    assert "--count" not in option_strings
    assert "--radius" not in option_strings
    assert "--bootstrap-seed" not in option_strings
    assert "--bootstrap-repeats" not in option_strings
    assert "--consumption-ledger" not in option_strings


def test_reserved_states_are_materialized_only_after_ledger_reservation() -> None:
    source = inspect.getsource(run)
    reserve = source.index("ConsumptionLedger.reserve")
    materialize = source.index("_formal_initial_states")
    physical_j = source.index("canonical_continuous_rollout")
    assert source.count("_formal_initial_states") == 1
    assert reserve < materialize < physical_j


def test_artifact_metadata_requires_unused_reserved_seed() -> None:
    metadata = valid_metadata()
    validated = validate_formal_artifact_metadata(metadata)
    assert validated["reserved_final_blind_seed"] == FORMAL_SEED
    assert validated["reserved_final_blind_used_before_evaluation"] is False

    metadata["seeds"]["reserved_final_blind_used"] = True  # type: ignore[index]
    with pytest.raises(RuntimeError, match="already used"):
        validate_formal_artifact_metadata(metadata)


def test_artifact_metadata_rejects_reserved_seed_reuse_elsewhere() -> None:
    metadata = valid_metadata()
    metadata["seeds"]["development_audit"] = FORMAL_SEED  # type: ignore[index]
    with pytest.raises(RuntimeError, match="also appears"):
        validate_formal_artifact_metadata(metadata)


def test_artifact_metadata_requires_exact_radii_and_zero_identity() -> None:
    metadata = valid_metadata()
    metadata["protected_radii"] = [0.0, 0.1]
    with pytest.raises(RuntimeError, match="protected radii"):
        validate_formal_artifact_metadata(metadata)

    metadata = valid_metadata()
    metadata["protected_identity"]["state_max_abs"] = 1.0e-16  # type: ignore[index]
    with pytest.raises(RuntimeError, match="protected identity"):
        validate_formal_artifact_metadata(metadata)


def test_tensor_commitment_is_shape_dtype_and_value_sensitive() -> None:
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    commitment = tensor_commitment(tensor)
    assert commitment["protocol"] == (
        "sha256_shape_float64_le_c_contiguous_bytes_v1"
    )
    assert commitment["shape"] == [2, 2]
    assert commitment["dtype"] == "float64-le"
    assert len(commitment["sha256"]) == 64
    assert commitment == tensor_commitment(tensor.clone())
    assert commitment["sha256"] != tensor_commitment(tensor.reshape(1, 4))[
        "sha256"
    ]
    changed = tensor.clone()
    changed[0, 0] = np.nextafter(1.0, 2.0)
    assert commitment["sha256"] != tensor_commitment(changed)["sha256"]
    with pytest.raises(ValueError, match="float64"):
        tensor_commitment(tensor.float())


def test_canonical_objective_uses_all_four_rk4_stages() -> None:
    trace = {
        "states": torch.tensor(
            [[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]],
            dtype=torch.float64,
        ),
        "controls": torch.tensor(
            [[[2.0, 4.0, 6.0, 8.0]]], dtype=torch.float64
        ),
        "terminal_state": torch.tensor(
            [[9.0, 10.0]], dtype=torch.float64
        ),
    }
    cfg = SimpleNamespace(T=2.0, n=1)
    params = {
        "beta": torch.tensor([1.0, 10.0], dtype=torch.float64),
        "gamma": torch.tensor(0.5, dtype=torch.float64),
        "alpha": torch.tensor([2.0, 3.0], dtype=torch.float64),
    }
    stage_running = torch.tensor(
        [1.0 + 20.0 + 1.0, 3.0 + 40.0 + 2.0,
         5.0 + 60.0 + 3.0, 7.0 + 80.0 + 4.0],
        dtype=torch.float64,
    )
    expected_running = (
        2.0
        * (
            stage_running[0]
            + 2.0 * stage_running[1]
            + 2.0 * stage_running[2]
            + stage_running[3]
        )
        / 6.0
    )
    expected_terminal = 9.0 * 2.0 + 10.0 * 3.0
    objective = canonical_continuous_objective(trace, cfg, params)
    assert torch.equal(
        objective,
        torch.tensor(
            [expected_running + expected_terminal],
            dtype=torch.float64,
        ),
    )


def test_paired_bootstrap_is_deterministic_and_validates_input() -> None:
    values = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float64)
    first = paired_bootstrap_mean(values, seed=91, repeats=200)
    second = paired_bootstrap_mean(values, seed=91, repeats=200)
    assert first == second
    assert first["lower_95"] <= float(values.mean()) <= first["upper_95"]

    with pytest.raises(ValueError, match="positive"):
        paired_bootstrap_mean(values, seed=91, repeats=0)
    with pytest.raises(ValueError, match="nonempty vector"):
        paired_bootstrap_mean(torch.empty(0), seed=91, repeats=10)
    with pytest.raises(ValueError, match="finite"):
        paired_bootstrap_mean(
            torch.tensor([1.0, float("nan")]), seed=91, repeats=10
        )


def test_per_sample_records_preserve_pairing_and_delta_signs() -> None:
    candidate = torch.tensor([3.0, 7.0], dtype=torch.float64)
    source = torch.tensor([5.0, 6.0], dtype=torch.float64)
    frozen = torch.tensor([4.0, 9.0], dtype=torch.float64)
    rows = per_sample_records(candidate, source, frozen)
    assert rows == [
        {
            "index": 0,
            "candidate_physical_J": 3.0,
            "source_feedback_physical_J": 5.0,
            "frozen_time_physical_J": 4.0,
            "delta_frozen_time_minus_candidate": 1.0,
            "delta_source_feedback_minus_candidate": 2.0,
            "delta_source_feedback_minus_frozen_time": 1.0,
        },
        {
            "index": 1,
            "candidate_physical_J": 7.0,
            "source_feedback_physical_J": 6.0,
            "frozen_time_physical_J": 9.0,
            "delta_frozen_time_minus_candidate": 2.0,
            "delta_source_feedback_minus_candidate": -1.0,
            "delta_source_feedback_minus_frozen_time": -3.0,
        },
    ]


def test_exact_identity_rejects_any_nonzero_metric() -> None:
    identity = {
        "source_stage_gate_max_abs": 0.0,
        "candidate_stage_gate_max_abs": 0.0,
        "source_node_gate_max_abs": 0.0,
        "candidate_node_gate_max_abs": 0.0,
        "node_time_logit_max_abs": 0.0,
        "stage_control_max_abs": 0.0,
        "stage_state_max_abs": 0.0,
        "terminal_state_max_abs": 0.0,
        "node_state_max_abs": 0.0,
        "objective_max_abs": 0.0,
    }
    assert_exact_protected_identity(identity)
    identity["stage_control_max_abs"] = torch.finfo(torch.float64).tiny
    with pytest.raises(RuntimeError, match="not bitwise exact"):
        assert_exact_protected_identity(identity)


def test_continuous_identity_checks_actual_four_stage_queries() -> None:
    cfg = ProblemConfig(
        T=1.0,
        n=3,
        m=2,
        umax=3.0,
        beta=0.1,
        alpha=0.0025,
        gamma=20.0,
        n0=10.0,
        m_suppression=0.5,
    )
    params = build_params(cfg, torch.device("cpu"), torch.float64)
    initial = torch.tensor(
        [[9.0, 11.0], [8.0, 12.0]],
        dtype=torch.float64,
    )
    source = _ToyContinuousPolicy()
    gate = _ConstantGate(0.0)
    candidate = _ToyContinuousPolicy(gate=gate, residual=1.0)
    identity = protected_identity_metrics(
        source,
        candidate,
        initial,
        cfg,
        params,
    )
    assert_exact_protected_identity(identity)
    assert set(identity) == {
        "source_stage_gate_max_abs",
        "candidate_stage_gate_max_abs",
        "source_node_gate_max_abs",
        "candidate_node_gate_max_abs",
        "node_time_logit_max_abs",
        "stage_control_max_abs",
        "stage_state_max_abs",
        "terminal_state_max_abs",
        "node_state_max_abs",
        "objective_max_abs",
    }

    gate.value = torch.finfo(torch.float64).eps
    broken = protected_identity_metrics(
        source,
        candidate,
        initial,
        cfg,
        params,
    )
    with pytest.raises(RuntimeError, match="not bitwise exact"):
        assert_exact_protected_identity(broken)


def test_ledger_is_exclusive_and_records_terminal_event(
    tmp_path: Path,
) -> None:
    path = tmp_path / "consumption.jsonl"
    ledger = ConsumptionLedger.reserve(
        path,
        {
            "event": "started",
            "protocol": "synthetic_nonreserved_test",
            "seed": 17,
        },
    )
    with pytest.raises(FileExistsError):
        ConsumptionLedger.reserve(
            path,
            {
                "event": "started",
                "protocol": "synthetic_nonreserved_test",
                "seed": 17,
            },
        )
    ledger.append({"event": "completed", "output_sha256": "b" * 64})
    ledger.close_read_only()

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["event"] for row in rows] == ["started", "completed"]


def test_result_writer_refuses_existing_output(tmp_path: Path) -> None:
    path = tmp_path / "blind.json"
    write_json_exclusive(path, {"first": True})
    with pytest.raises(FileExistsError):
        write_json_exclusive(path, {"second": True})
    assert json.loads(path.read_text(encoding="utf-8")) == {"first": True}


def test_result_writer_refuses_a_broken_symlink(tmp_path: Path) -> None:
    path = tmp_path / "blind.json"
    path.symlink_to(tmp_path / "missing-target.json")
    with pytest.raises(FileExistsError):
        write_json_exclusive(path, {"must_not_write": True})
    assert path.is_symlink()
