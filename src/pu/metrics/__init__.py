"""
Metrics module for comparing neural network representations.

This module provides a unified API for computing various representational
similarity metrics between embedding matrices.

Usage:
    >>> from pu import metrics
    >>> Z1, Z2 = np.random.randn(100, 64), np.random.randn(100, 64)
    >>> metrics.cka(Z1, Z2)
    0.85
    >>> metrics.compare(Z1, Z2, metrics=["cka", "mknn", "procrustes"])
    {"cka": 0.85, "mknn": 0.72, "procrustes": 0.15}
"""

# Kernel-based metrics
from pu.metrics.kernel import cka, mmd

# Geometric metrics
from pu.metrics.geometric import procrustes, cosine_similarity, frechet

# CCA-based metrics
from pu.metrics.cca import svcca, pwcca

# Spectral metrics
from pu.metrics.spectral import tucker_congruence, eigenspectrum, riemannian

# Information-theoretic metrics
from pu.metrics.information import kl_divergence, js_divergence, mutual_information

# Neighbor-based metrics
from pu.metrics.neighbors import mknn, jaccard, rsa

# Regression-based metrics
from pu.metrics.regression import linear_r2, bidirectional_linear_r2

# I/O and batch comparison
from pu.metrics.io import (
    list_metrics,
    compare,
    compare_from_parquet,
    load_embeddings_from_parquet,
    get_available_sizes,
    METRICS_REGISTRY,
)

# Base utilities (for advanced users)
from pu.metrics._base import (
    validate_inputs,
    center,
    normalize_rows,
    gram_matrix,
    center_gram,
    rbf_kernel,
)

__all__ = [
    # Kernel-based
    "cka",
    "mmd",
    # Geometric
    "procrustes",
    "cosine_similarity",
    "frechet",
    # CCA
    "svcca",
    "pwcca",
    # Spectral
    "tucker_congruence",
    "eigenspectrum",
    "riemannian",
    # Information-theoretic
    "kl_divergence",
    "js_divergence",
    "mutual_information",
    # Neighbor-based
    "mknn",
    "jaccard",
    "rsa",
    # Regression
    "linear_r2",
    "bidirectional_linear_r2",
    # I/O
    "list_metrics",
    "compare",
    "compare_from_parquet",
    "load_embeddings_from_parquet",
    "get_available_sizes",
    "METRICS_REGISTRY",
    # Base utilities
    "validate_inputs",
    "center",
    "normalize_rows",
    "gram_matrix",
    "center_gram",
    "rbf_kernel",
]
