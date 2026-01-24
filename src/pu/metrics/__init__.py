"""
Metrics module for comparing neural network representations.

This module provides a unified API for computing various representational
similarity metrics between embedding matrices.
"""

from pu.metrics._base import (
    validate_inputs,
    center,
    normalize_rows,
    gram_matrix,
    center_gram,
    rbf_kernel,
)

__all__ = [
    # Base utilities
    "validate_inputs",
    "center",
    "normalize_rows",
    "gram_matrix",
    "center_gram",
    "rbf_kernel",
]
