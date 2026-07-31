from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from train_feedback_section5 import (  # noqa: E402
    FIXED_ANCHOR_NAMES,
    NestedFeedbackTransformer,
    anchor_gated_candidate_is_better,
    anchor_loss_limits,
    build_parser,
    sample_fixed_anchor_random_initial_states,
    train,
)
from train_paper_pmp_kkt import ProblemConfig  # noqa: E402


def make_cfg() -> ProblemConfig:
    return ProblemConfig(
        T=2.0,
        n=8,
        m=21,
        umax=3.0,
        beta=0.1,
        alpha=1.0,
        gamma=20.0,
        n0=10.0,
        m_suppression=0.5,
    )


def test_fixed_anchor_protocol_is_opt_in() -> None:
    args = build_parser().parse_args(["--option", "der"])
    assert args.fixed_anchor_random_protocol is False
    assert args.fixed_anchor_max_relative_loss_increase == 0.01
    assert args.fixed_anchor_max_absolute_loss_increase == 1.0e-12


def test_fixed_anchor_random_sampler_prefixes_all_three_anchors() -> None:
    cfg = make_cfg()

    def sample(seed: int) -> torch.Tensor:
        return sample_fixed_anchor_random_initial_states(
            5,
            cfg,
            torch.device("cpu"),
            torch.float64,
            generator=torch.Generator().manual_seed(seed),
        )

    first = sample(17)
    second = sample(17)
    third = sample(18)
    assert first.shape == (len(FIXED_ANCHOR_NAMES) + 5, cfg.m)
    assert torch.equal(first, second)
    assert not torch.equal(
        first[len(FIXED_ANCHOR_NAMES) :],
        third[len(FIXED_ANCHOR_NAMES) :],
    )

    direction = torch.linspace(-1.0, 1.0, cfg.m, dtype=torch.float64)
    torch.testing.assert_close(
        first[0],
        torch.full((cfg.m,), cfg.n0, dtype=torch.float64),
    )
    torch.testing.assert_close(first[1], cfg.n0 * (1.0 + 0.10 * direction))
    torch.testing.assert_close(first[2], cfg.n0 * (1.0 + 0.20 * direction))
    random = first[len(FIXED_ANCHOR_NAMES) :]
    assert torch.all(random >= 0.8 * cfg.n0)
    assert torch.all(random <= 1.2 * cfg.n0)


def test_anchor_gate_rejects_r010_regression_despite_better_random_loss() -> None:
    baseline = {
        "nominal": 1.0,
        "structured_r0p10": 2.0,
        "structured_r0p20": 3.0,
    }
    limits = anchor_loss_limits(
        baseline,
        max_relative_increase=0.01,
        max_absolute_increase=1.0e-12,
    )
    regressed_r010 = {
        "nominal": 0.9,
        "structured_r0p10": 2.03,
        "structured_r0p20": 2.8,
    }
    assert not anchor_gated_candidate_is_better(
        random_validation_loss=0.5,
        anchor_losses=regressed_r010,
        anchor_limits=limits,
        best_random_validation_loss=1.0,
    )

    eligible = {
        "nominal": 0.9,
        "structured_r0p10": limits["structured_r0p10"],
        "structured_r0p20": 2.8,
    }
    assert anchor_gated_candidate_is_better(
        random_validation_loss=0.5,
        anchor_losses=eligible,
        anchor_limits=limits,
        best_random_validation_loss=1.0,
    )
    assert not anchor_gated_candidate_is_better(
        random_validation_loss=1.1,
        anchor_losses=eligible,
        anchor_limits=limits,
        best_random_validation_loss=1.0,
    )


def test_epoch_zero_checkpoint_records_fixed_anchor_selection_protocol(
    tmp_path: Path,
) -> None:
    cfg = make_cfg()
    source = NestedFeedbackTransformer(
        cfg.m,
        cfg.umax,
        15.0,
        (8,),
        8,
        2,
        1,
        1.5,
    ).double()
    time_checkpoint = tmp_path / "time.pt"
    torch.save(
        {
            "model_state": source.time_branch.state_dict(),
            "args": {
                "d_model": 8,
                "heads": 2,
                "layers": 1,
                "init_u": 1.5,
            },
            "problem": cfg.__dict__,
        },
        time_checkpoint,
    )
    output = tmp_path / "run"
    args = build_parser().parse_args(
        [
            "--option",
            "der",
            "--time_checkpoint",
            str(time_checkpoint),
            "--out_dir",
            str(output),
            "--n",
            str(cfg.n),
            "--T",
            str(cfg.T),
            "--d_model",
            "8",
            "--heads",
            "2",
            "--layers",
            "1",
            "--state_hidden",
            "8",
            "--epochs",
            "0",
            "--batch_size",
            "2",
            "--validation_size",
            "2",
            "--test_size",
            "2",
            "--test_radii",
            "0.20",
            "--eval_every",
            "1",
            "--device",
            "cpu",
            "--float64",
            "--fixed_anchor_random_protocol",
        ]
    )
    train(args)

    expected_metrics = {
        "best_feedback_section5.pt": (
            "random_validation_loss_subject_to_independent_anchor_gates"
        ),
        "best_feedback_section5_full_gradient.pt": (
            "random_validation_full_gradient_loss_subject_to_"
            "independent_anchor_gates"
        ),
        "best_feedback_section5_full_gradient_linf.pt": (
            "random_validation_full_gradient_residual_linf_subject_to_"
            "independent_anchor_gates"
        ),
    }
    for filename, expected_metric in expected_metrics.items():
        checkpoint = torch.load(
            output / filename,
            map_location="cpu",
            weights_only=False,
        )
        assert checkpoint["selection_metric"] == expected_metric
        protocol = checkpoint["selection_protocol"]
        assert protocol["anchor_names"] == list(FIXED_ANCHOR_NAMES)
        assert protocol["anchor_radii"] == [0.0, 0.1, 0.2]
        assert protocol["random_radius"] == 0.2
        assert protocol["training_random_count"] == 2
        assert protocol["validation_random_count"] == 2
        assert protocol["physical_objective_used_for_training"] is False
        assert (
            protocol["physical_objective_used_for_checkpoint_selection"]
            is False
        )
        for name in FIXED_ANCHOR_NAMES:
            assert protocol["selected_anchor_losses"][name] <= (
                protocol["anchor_loss_limits"][name]
            )
