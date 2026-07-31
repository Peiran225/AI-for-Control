"""Faithful Eq. (2.3)--(2.5) PI-DeepONet implementation.

This package intentionally does not import or read any direct-control result.
It contains an original-paper LQR smoke and a separately labelled tumor
adaptation.
"""

from .core import (
    DeepONet,
    TrainConfig,
    finite_difference_operators,
    required_viscosity_constant,
    validate_viscosity_constant,
)
from .problems import PaperLQR5D, TumorAdaptation

__all__ = [
    "DeepONet",
    "PaperLQR5D",
    "TrainConfig",
    "TumorAdaptation",
    "finite_difference_operators",
    "required_viscosity_constant",
    "validate_viscosity_constant",
]
