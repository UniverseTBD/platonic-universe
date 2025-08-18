"""
Utility functions for Platonic Universe.

This module provides common utilities for k-nearest neighbor analysis,
metrics computation, and other helper functions.
"""

from .knn_metrics import (
    compute_mknn_score,
    compute_mknn_prh,
    mutual_knn_overlap,
    KNNAnalyzer,
)

from .metrics import (
    cosine_similarity_matrix,
    euclidean_similarity_matrix, 
    normalize_embeddings,
    compute_alignment_score,
)

from .helpers import (
    set_seed,
    clear_gpu_memory,
    get_device_info,
    safe_divide,
    ensure_numpy,
    validate_embeddings,
)

__all__ = [
    "compute_mknn_score",
    "compute_mknn_prh", 
    "mutual_knn_overlap",
    "KNNAnalyzer",
    "cosine_similarity_matrix",
    "euclidean_similarity_matrix",
    "normalize_embeddings", 
    "compute_alignment_score",
    "set_seed",
    "clear_gpu_memory",
    "get_device_info", 
    "safe_divide",
    "ensure_numpy",
    "validate_embeddings",
]