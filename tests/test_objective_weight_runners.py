import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_deep_refined_new_objective_setting as deep_setting_runner
from scripts import run_deep_time_only_refinement as deep_time_runner
from scripts import run_new_objective_weight_setting as weight_runner


class ObjectiveWeightRunnerTests(unittest.TestCase):
    def test_alpha_five_problem_and_training_normalization(self) -> None:
        physical = weight_runner.physical_problem(400, 4000, alpha=5)
        normalized, scale = weight_runner.normalized_problem(
            400, 4000, alpha=5
        )

        self.assertEqual(
            (physical["alpha"], physical["beta"], physical["gamma"]),
            (5.0, 400.0, 4000.0),
        )
        self.assertEqual(scale, 200.0)
        self.assertEqual(
            (normalized["alpha"], normalized["beta"], normalized["gamma"]),
            (0.025, 2.0, 20.0),
        )

    def test_weight_runner_tag_and_cli_parse_alpha(self) -> None:
        root = ROOT / "outputs" / "not-run"
        argv = [
            "run_new_objective_weight_setting.py",
            "--alpha",
            "5",
            "--beta",
            "400",
            "--gamma",
            "4000",
            "--root",
            str(root),
        ]
        with patch.object(sys, "argv", argv), patch.object(
            weight_runner, "build"
        ) as mocked_build:
            weight_runner.main()

        args = mocked_build.call_args.args[0]
        self.assertEqual((args.alpha, args.beta, args.gamma), (5.0, 400.0, 4000.0))
        self.assertEqual(args.root, root)
        tag = (
            f"a{weight_runner.scalar_tag(args.alpha)}_"
            f"b{weight_runner.scalar_tag(args.beta)}_"
            f"g{weight_runner.scalar_tag(args.gamma)}"
        )
        self.assertEqual(tag, "a5_b400_g4000")

    def test_deep_setting_runner_parses_alpha_and_diagnostics_dirname(self) -> None:
        root = ROOT / "outputs" / "not-run"
        argv = [
            "run_deep_refined_new_objective_setting.py",
            "--alpha",
            "5",
            "--beta",
            "400",
            "--gamma",
            "4000",
            "--root",
            str(root),
            "--diagnostics-dirname",
            "two_state_diagnostics_a5",
        ]
        with patch.object(sys, "argv", argv), patch.object(
            deep_setting_runner, "build"
        ) as mocked_build:
            deep_setting_runner.main()

        args = mocked_build.call_args.args[0]
        self.assertEqual((args.alpha, args.beta, args.gamma), (5.0, 400.0, 4000.0))
        self.assertEqual(args.root, root)
        self.assertEqual(args.diagnostics_dirname, "two_state_diagnostics_a5")

    def test_deep_setting_runner_rejects_nested_diagnostics_dirname(self) -> None:
        argv = [
            "run_deep_refined_new_objective_setting.py",
            "--alpha",
            "5",
            "--beta",
            "400",
            "--gamma",
            "4000",
            "--diagnostics-dirname",
            "nested/diagnostics",
        ]
        with patch.object(sys, "argv", argv), patch.object(
            deep_setting_runner, "build"
        ) as mocked_build:
            with self.assertRaisesRegex(
                ValueError, "diagnostics dirname must be one plain directory name"
            ):
                deep_setting_runner.main()
        mocked_build.assert_not_called()

    def test_deep_time_runner_parses_setting_directory_without_running(self) -> None:
        setting = ROOT / "outputs" / "not-run" / "a5_b400_g4000"
        argv = [
            "run_deep_time_only_refinement.py",
            "--setting-dir",
            str(setting),
        ]
        with patch.object(sys, "argv", argv), patch.object(
            deep_time_runner, "build"
        ) as mocked_build:
            deep_time_runner.main()

        args = mocked_build.call_args.args[0]
        self.assertEqual(args.setting_dir, setting)

    def test_runner_sources_compile(self) -> None:
        for relative_path in (
            "scripts/run_new_objective_weight_setting.py",
            "scripts/run_deep_time_only_refinement.py",
            "scripts/run_deep_refined_new_objective_setting.py",
        ):
            path = ROOT / relative_path
            compile(path.read_text(encoding="utf-8"), str(path), "exec")


if __name__ == "__main__":
    unittest.main()
