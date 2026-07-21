import unittest

import numpy as np

from tumor_problem import NOMINAL_TUMOR_PROBLEM, evaluate_zoh_control, singular_control_numpy


class TumorProblemTests(unittest.TestCase):
    def test_canonical_problem_parameters(self) -> None:
        problem = NOMINAL_TUMOR_PROBLEM
        self.assertEqual(problem.m, 21)
        self.assertEqual(problem.m_suppression, 0.5)
        self.assertEqual(problem.beta, 0.1)
        self.assertEqual(problem.gamma, 20.0)

    def test_constant_control_is_grid_independent(self) -> None:
        coarse_t = np.linspace(0.0, 10.0, 11)
        fine_t = np.linspace(0.0, 10.0, 101)
        coarse = evaluate_zoh_control(coarse_t, np.full_like(coarse_t, 1.5), include_diagnostics=False)
        fine = evaluate_zoh_control(fine_t, np.full_like(fine_t, 1.5), include_diagnostics=False)
        self.assertAlmostEqual(coarse["J"], fine["J"], places=7)

    def test_singular_control_is_finite_and_admissible_at_nominal_state(self) -> None:
        state = np.full(NOMINAL_TUMOR_PROBLEM.m, NOMINAL_TUMOR_PROBLEM.n0)
        control = float(singular_control_numpy(state, NOMINAL_TUMOR_PROBLEM))
        self.assertTrue(np.isfinite(control))
        self.assertGreaterEqual(control, 0.0)
        self.assertLessEqual(control, NOMINAL_TUMOR_PROBLEM.umax)


if __name__ == "__main__":
    unittest.main()
