"""Faithful implementation of Neural-PMP (arXiv:2212.14566).

The package intentionally does not import or modify the historical external
checkout.  Its central invariant is the paper's update order:

    raw dH/du -> gradient step -> projection onto the action set.
"""

from .core import (
    NeuralPMPResult,
    discrete_adjoint_gradient,
    full_autodiff_gradient,
    project_action,
    solve_neural_pmp,
)

__all__ = [
    "NeuralPMPResult",
    "discrete_adjoint_gradient",
    "full_autodiff_gradient",
    "project_action",
    "solve_neural_pmp",
]
