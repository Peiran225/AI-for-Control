from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from refine_feedback_gated_residual_adapter import (  # noqa: E402
    GatedResidualFeedback,
    ProtectedTrajectoryGate,
    load_initial_head,
)
from evaluate_gated_residual_adapter_blind import (  # noqa: E402
    paired_bootstrap_mean,
)
from train_feedback_section5 import NestedFeedbackTransformer  # noqa: E402


def tiny_source() -> NestedFeedbackTransformer:
    model = NestedFeedbackTransformer(
        m=3,
        umax=3.0,
        state_scale=15.0,
        state_hidden=(128,),
        d_model=8,
        heads=2,
        layers=1,
        init_u=1.5,
        state_feature_mode="relative_nominal",
        center_state_correction=True,
    ).double()
    reference = torch.full((5, 3), 10.0, dtype=torch.float64)
    model.set_nominal_reference(reference)
    return model.eval()


def test_gate_is_exactly_zero_on_protected_nodes_and_nonzero_elsewhere() -> None:
    protected = torch.stack(
        (
            torch.full((5, 3), 10.0, dtype=torch.float64),
            torch.linspace(9.0, 11.0, 5, dtype=torch.float64)
            .reshape(5, 1)
            .expand(5, 3),
        )
    )
    gate = ProtectedTrajectoryGate(protected, distance_scale=0.02)
    time = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    value = gate(time, protected[0])
    assert torch.equal(value, torch.zeros_like(value))
    random = protected[0] + torch.tensor(
        [1.0, -0.5, 0.25], dtype=torch.float64
    )
    assert torch.all(gate(time, random) > 0.0)


def test_adapter_has_129_parameters_and_preserves_protected_actions() -> None:
    source = tiny_source()
    protected = torch.full((2, 5, 3), 10.0, dtype=torch.float64)
    gate = ProtectedTrajectoryGate(protected, distance_scale=0.02)
    model = GatedResidualFeedback(source, gate)
    assert model.adapter_parameter_count == 129
    with torch.no_grad():
        model.residual_head.weight.fill_(0.7)
        model.residual_head.bias.fill_(0.2)
    time = torch.linspace(0.0, 0.75, 4, dtype=torch.float64)
    state = protected[0, :4]
    base = source.time_logits(
        torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    )[:4]
    source_action = source.interval_action(
        base, time, state, state_mode="feedback"
    )
    adapted_action = model.interval_action(
        base, time, state, state_mode="feedback"
    )
    assert torch.equal(source_action, adapted_action)
    assert all(
        not parameter.requires_grad
        for parameter in model.source.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in model.residual_head.parameters()
    )


def test_gate_interpolation_vanishes_between_nodes() -> None:
    first = torch.tensor(
        [[1.0], [2.0], [4.0]], dtype=torch.float64
    )
    second = torch.tensor(
        [[4.0], [3.0], [2.0]], dtype=torch.float64
    )
    gate = ProtectedTrajectoryGate(
        torch.stack((first, second)), distance_scale=0.1
    )
    time = torch.tensor([0.25, 0.75], dtype=torch.float64)
    interpolated = torch.tensor([[1.5], [3.0]], dtype=torch.float64)
    assert torch.equal(
        gate(time, interpolated), torch.zeros(2, dtype=torch.float64)
    )


def test_quintic_gate_has_zero_first_and_second_join_derivatives() -> None:
    protected = torch.ones((1, 2, 1), dtype=torch.float64)
    gate = ProtectedTrajectoryGate(
        protected,
        distance_scale=0.2,
        zero_tube_squared=0.01,
    )
    for squared_distance in (0.01, 0.01 + 0.2**2):
        state = torch.tensor(
            [[1.0 + squared_distance**0.5]],
            dtype=torch.float64,
            requires_grad=True,
        )
        value = gate(torch.tensor([0.0], dtype=torch.float64), state)
        first = torch.autograd.grad(
            value.sum(), state, create_graph=True
        )[0]
        second = torch.autograd.grad(first.sum(), state)[0]
        assert abs(float(first.detach())) < 1.0e-10
        assert abs(float(second.detach())) < 1.0e-9


def test_initial_head_is_validated_against_source_and_split(
    tmp_path: Path,
) -> None:
    source = tiny_source()
    gate = ProtectedTrajectoryGate(
        torch.full((2, 5, 3), 10.0, dtype=torch.float64),
        distance_scale=0.02,
    )
    model = GatedResidualFeedback(source, gate)
    with torch.no_grad():
        target_weight = torch.arange(
            128, dtype=torch.float64
        ).reshape(1, 128) / 128
        target_bias = torch.tensor([0.25], dtype=torch.float64)
    artifact = {
        "format": "teacher_solver_probe_v1",
        "solver": "lbfgs_zero",
        "residual_head_state": {
            "weight": target_weight,
            "bias": target_bias,
        },
        "source_model_state": source.state_dict(),
        "problem": {"m": 3},
        "direct_teacher_indices": {
            "train": list(range(24)),
            "validation": list(range(24, 32)),
        },
        "physical_objective_used": False,
        "physical_objective_used_for_ranking": False,
    }
    path = tmp_path / "head.pt"
    torch.save(artifact, path)
    records = [{"index": index} for index in range(32)]
    metadata = load_initial_head(
        path,
        model=model,
        source_payload={
            "model_state": source.state_dict(),
            "problem": {"m": 3},
        },
        train_records=records[:24],
        validation_records=records[24:],
    )
    assert metadata["solver"] == "lbfgs_zero"
    assert torch.equal(model.residual_head.weight, target_weight)
    assert torch.equal(model.residual_head.bias, target_bias)


def test_paired_bootstrap_constant_has_exact_interval() -> None:
    result = paired_bootstrap_mean(
        torch.full((8,), 1.25, dtype=torch.float64),
        seed=1,
        repeats=100,
    )
    assert result["lower_95"] == 1.25
    assert result["upper_95"] == 1.25
