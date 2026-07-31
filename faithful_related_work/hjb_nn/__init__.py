"""Faithful [7] Adaptive HJB-NN reproductions and tumor adaptations.

The original-paper satellite path delegates to an audited TensorFlow-2
compatibility port of the author code in ``external/HJB_NN``. The tumor path preserves its characteristic-BVP,
value/costate supervision, full-batch L-BFGS, and Algorithm 4.1 adaptive-data
mechanisms while making the bounded-control smoothing mathematically explicit.
"""

from .problem import TumorEntropyProblem, evaluate_zoh_cost_pair
from .sampling import (
    convergence_test_and_next_size,
    git_checkout_provenance,
    refuse_nonempty_output,
    select_largest_gradient_candidates,
    split_initial_state_box,
    validate_trajectory_disjoint,
)

__all__ = [
    "TumorEntropyProblem",
    "evaluate_zoh_cost_pair",
    "convergence_test_and_next_size",
    "git_checkout_provenance",
    "refuse_nonempty_output",
    "select_largest_gradient_candidates",
    "split_initial_state_box",
    "validate_trajectory_disjoint",
]
