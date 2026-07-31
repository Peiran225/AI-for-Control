"""Shared audit protocol for faithful related-work reproductions.

The method packages intentionally remain independent.  This package only
normalizes their provenance, selection declarations, and exported controls;
it does not alter a method's optimizer or numerical mechanism.
"""

from .manifest import (
    CLAIM_LABELS,
    ManifestValidationError,
    sha256_file,
    validate_root_manifest,
)

__all__ = [
    "CLAIM_LABELS",
    "ManifestValidationError",
    "sha256_file",
    "validate_root_manifest",
]
