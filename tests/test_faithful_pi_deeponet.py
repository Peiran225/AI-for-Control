import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from faithful_related_work.pi_deeponet.core import (
    DeepONet,
    ImprovedPolicy,
    TrainConfig,
    build_model,
    finite_difference_operators,
    required_viscosity_constant,
    train_policy_iteration,
    validate_viscosity_constant,
)
from faithful_related_work.pi_deeponet.experiment import ExperimentConfig, run_experiment
from faithful_related_work.pi_deeponet.problems import PaperLQR5D, TumorAdaptation


class QuadraticValue(nn.Module):
    def forward(self, branch: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        del branch
        return t.square() + x.square().sum(dim=-1)


def tiny_config(problem, *, h: float, sensors: int = 4) -> TrainConfig:
    required = required_viscosity_constant(problem.dynamics_sup_bound())
    return TrainConfig(
        h=h,
        viscosity_N=required,
        outer_iterations=1,
        steps_per_outer=1,
        batch_size=2,
        terminal_batch_size=2,
        sensors=sensors,
        width=6,
        latent_dim=6,
        branch_depth=1,
        trunk_depth=1,
        log_every=1,
        value_scale=1.0 if isinstance(problem, PaperLQR5D) else 500.0,
        branch_scale=4.0 if isinstance(problem, PaperLQR5D) else 5000.0,
        dtype="float64",
        device="cpu",
    )


class FaithfulPIDeepONetTests(unittest.TestCase):
    def test_deeponet_uses_literal_unnormalized_inner_product(self) -> None:
        model = DeepONet(
            sensor_count=2,
            state_dim=1,
            width=2,
            latent_dim=2,
            branch_depth=1,
            trunk_depth=1,
            horizon=2.0,
        ).to(dtype=torch.float64)
        model.branch = nn.Identity()
        model.trunk = nn.Identity()
        branch = torch.tensor([[2.0, 3.0]], dtype=torch.float64)
        t = torch.tensor([1.0], dtype=torch.float64)
        x = torch.tensor([[4.0]], dtype=torch.float64)
        # Eq. (2.6): 2*(t/T) + 3*x = 13, with no sqrt(p) factor or output bias.
        torch.testing.assert_close(model(branch, t, x), torch.tensor([13.0], dtype=torch.float64))

    def test_central_finite_difference_gradient_and_laplacian(self) -> None:
        model = QuadraticValue()
        branch = torch.zeros((2, 3), dtype=torch.float64)
        t = torch.tensor([0.2, 0.7], dtype=torch.float64)
        x = torch.tensor([[0.1, -0.4, 0.7], [-0.8, 0.2, 0.3]], dtype=torch.float64)
        gradient, laplacian = finite_difference_operators(model, branch, t, x, 0.01)
        torch.testing.assert_close(gradient, 2.0 * x, rtol=0.0, atol=2.0e-12)
        torch.testing.assert_close(laplacian, torch.full((2,), 6.0, dtype=torch.float64), rtol=0.0, atol=2.0e-11)
        with self.assertRaisesRegex(ValueError, "must not require gradients"):
            finite_difference_operators(model, branch, t, x.clone().requires_grad_(True), 0.01)

    def test_viscosity_monotonicity_condition_is_enforced(self) -> None:
        self.assertEqual(required_viscosity_constant(0.5), 1.0)
        self.assertEqual(required_viscosity_constant(8.0), 4.0)
        self.assertEqual(validate_viscosity_constant(4.0, 8.0), 4.0)
        with self.assertRaisesRegex(ValueError, "monotonicity condition violated"):
            validate_viscosity_constant(3.99, 8.0)

    def test_tumor_exact_affine_argmin_and_nonunique_tie(self) -> None:
        problem = TumorAdaptation()
        t = torch.zeros(1, dtype=torch.float64)
        x = torch.full((1, problem.state_dim), 0.1, dtype=torch.float64)

        positive = problem.exact_hamiltonian_argmin(t, x, torch.zeros_like(x))
        self.assertEqual(float(positive.control.item()), 0.0)
        self.assertFalse(bool(positive.nonunique.item()))

        negative = problem.exact_hamiltonian_argmin(t, x, torch.full_like(x, 100.0))
        self.assertEqual(float(negative.control.item()), problem.umax)
        self.assertFalse(bool(negative.nonunique.item()))

        tie_gradient = torch.zeros_like(x)
        tie_gradient[0, 0] = problem.gamma / (problem.phi[0] * float(x[0, 0]))
        tie = problem.exact_hamiltonian_argmin(t, x, tie_gradient, tie_tolerance=1.0e-12)
        self.assertTrue(bool(tie.nonunique.item()))
        self.assertEqual(float(tie.control.item()), 0.5 * problem.umax)

        for gradient, result in ((torch.zeros_like(x), positive), (torch.full_like(x, 100.0), negative)):
            coefficient = problem.switching_coefficient(x, gradient).item()
            candidates = np.array([coefficient * 0.0, coefficient * problem.umax])
            chosen = coefficient * float(result.control.item())
            self.assertAlmostEqual(chosen, float(candidates.min()), places=12)

    def test_lqr_exact_quadratic_argmin(self) -> None:
        problem = PaperLQR5D()
        t = torch.zeros(3, dtype=torch.float64)
        x = torch.zeros((3, problem.state_dim), dtype=torch.float64)
        gradient = torch.tensor(
            [[0.1, -0.2, 0.3, -0.4, 0.5], [20.0] * 5, [-20.0] * 5],
            dtype=torch.float64,
        )
        result = problem.exact_hamiltonian_argmin(t, x, gradient)
        expected = -0.5 * gradient.numpy() @ problem.B
        expected = np.clip(expected, problem.control_lower, problem.control_upper)
        np.testing.assert_allclose(result.control.numpy(), expected, rtol=0.0, atol=1.0e-14)

    def test_lqr_terminal_functions_match_paper_family_and_holdout(self) -> None:
        problem = PaperLQR5D()
        x = torch.tensor(
            [[1.0, 2.0, 0.0, 0.0, 0.0], [0.0, 1.0, 2.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        # Both rows have ||x||^2=5.  Identifiers 1 and 3 construct the
        # training functions 0.3+0.1*k||x||^2, not 0.4/0.6 times ||x||^2.
        training = problem.terminal_cost(x, torch.tensor([1.0, 3.0], dtype=torch.float64))
        torch.testing.assert_close(training, torch.tensor([0.8, 1.8], dtype=torch.float64))
        held_out = problem.terminal_cost(
            x[:1], torch.tensor([0.57], dtype=torch.float64)
        )
        torch.testing.assert_close(held_out, torch.tensor([2.85], dtype=torch.float64))

    def test_gradient_clipping_is_not_part_of_default_algorithm(self) -> None:
        self.assertIsNone(tiny_config(PaperLQR5D(), h=0.005).gradient_clip)

    def test_improved_policy_is_a_deep_frozen_snapshot(self) -> None:
        problem = PaperLQR5D()
        config = tiny_config(problem, h=0.005)
        live_model = build_model(problem, config).to(dtype=torch.float64)
        policy = ImprovedPolicy(live_model, problem, config.h)
        live_parameters = list(live_model.parameters())
        frozen_parameters = list(policy.model.parameters())
        self.assertEqual(len(live_parameters), len(frozen_parameters))
        for live, frozen in zip(live_parameters, frozen_parameters):
            self.assertNotEqual(live.data_ptr(), frozen.data_ptr())
            torch.testing.assert_close(live, frozen, rtol=0.0, atol=0.0)
            self.assertFalse(frozen.requires_grad)
        with torch.no_grad():
            live_parameters[0].add_(1.0)
        self.assertFalse(torch.equal(live_parameters[0], frozen_parameters[0]))

    def test_training_is_deterministic_and_saves_every_outer(self) -> None:
        problem = PaperLQR5D()
        config = tiny_config(problem, h=0.005)
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            result_a = train_policy_iteration(
                problem,
                (1.0, 2.0, 3.0),
                config,
                seed=17,
                output_dir=Path(first),
                initial_control=np.zeros(problem.control_dim),
            )
            result_b = train_policy_iteration(
                problem,
                (1.0, 2.0, 3.0),
                config,
                seed=17,
                output_dir=Path(second),
                initial_control=np.zeros(problem.control_dim),
            )
            np.testing.assert_array_equal(result_a.sensor_states, result_b.sensor_states)
            self.assertEqual(result_a.history, result_b.history)
            for key, value in result_a.model.state_dict().items():
                torch.testing.assert_close(value, result_b.model.state_dict()[key], rtol=0.0, atol=0.0)
            self.assertEqual(len(result_a.checkpoints), config.outer_iterations)
            self.assertTrue(all(path.exists() for path in result_a.checkpoints))
            self.assertEqual(result_a.history[0]["terminal_functions_per_step"], 3)

    def test_small_budget_lqr_and_tumor_artifact_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lqr = PaperLQR5D()
            lqr_output = root / "lqr"
            run_experiment(
                lqr,
                tiny_config(lqr, h=0.005),
                ExperimentConfig(
                    seeds=(0,),
                    terminal_parameter_family=(1.0, 2.0, 3.0),
                    target_terminal_parameter=0.57,
                    initial_control=(0.0, 0.0, 0.0),
                    initial_state=(0.3, -0.3, 0.2, -0.2, 0.1),
                    evaluation_intervals=4,
                    output_dir=lqr_output,
                ),
                command=("unit-test", "lqr"),
            )
            lqr_solution = np.load(lqr_output / "seed_0" / "solution.npz")
            self.assertEqual(lqr_solution["t"].shape, (5,))
            self.assertEqual(lqr_solution["u"].shape, (4, 3))

            tumor = TumorAdaptation()
            tumor_output = root / "tumor"
            run_experiment(
                tumor,
                tiny_config(tumor, h=0.02),
                ExperimentConfig(
                    seeds=(0,),
                    terminal_parameter_family=(0.8, 1.0, 1.2),
                    target_terminal_parameter=1.0,
                    initial_control=(0.0,),
                    initial_state=tuple(tumor.initial_state.tolist()),
                    evaluation_intervals=20,
                    output_dir=tumor_output,
                ),
                command=("unit-test", "tumor"),
            )
            tumor_solution = np.load(tumor_output / "seed_0" / "solution.npz")
            self.assertEqual(tumor_solution["t"].shape, (21,))
            self.assertEqual(tumor_solution["u"].shape, (20,))
            self.assertTrue(np.isfinite(float(tumor_solution["J_realized"])))
            with (tumor_output / "manifest.json").open() as stream:
                manifest = json.load(stream)
            self.assertEqual(manifest["experiment_label"], "tumor adaptation")
            self.assertFalse(manifest["fidelity_invariants"]["direct_reference_used"])
            self.assertEqual(len(manifest["seeds"]), 1)

    def test_package_never_reads_a_direct_reference(self) -> None:
        package = Path(__file__).resolve().parents[1] / "faithful_related_work" / "pi_deeponet"
        source = "\n".join(path.read_text() for path in package.glob("*.py"))
        self.assertNotIn("direct_openloop", source)
        self.assertNotIn("direct_solution", source)
        self.assertNotIn("paper_runs", source)


if __name__ == "__main__":
    unittest.main()
