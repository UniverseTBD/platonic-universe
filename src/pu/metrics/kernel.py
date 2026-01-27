"""
Kernel-based similarity metrics: CKA and MMD.
"""

import numpy as np
from numpy.typing import NDArray

from pu.metrics._base import (
    validate_inputs,
    center,
    gram_matrix,
    center_gram,
    rbf_kernel,
)


def cka(
    Z1: NDArray[np.floating],
    Z2: NDArray[np.floating],
    kernel: str = "linear",
    gamma: float | None = None,
) -> float:
    """
    Centered Kernel Alignment (CKA) between two embedding matrices.

    CKA measures similarity between representations by comparing their
    centered Gram matrices. It's invariant to orthogonal transformations
    and isotropic scaling.

    Args:
        Z1: (n_samples, d1) embedding matrix
        Z2: (n_samples, d2) embedding matrix
        kernel: Kernel type - "linear" or "rbf"
        gamma: RBF bandwidth (only used if kernel="rbf").
               If None, uses median heuristic.

    Returns:
        float in [0, 1] where 1 = perfect alignment, 0 = no alignment

    Reference:
        Kornblith et al. (2019) "Similarity of Neural Network Representations
        Revisited" (ICML)
    """
    Z1, Z2 = validate_inputs(Z1, Z2)

    # Compute kernel matrices
    if kernel == "linear":
        K1 = gram_matrix(Z1)
        K2 = gram_matrix(Z2)
    elif kernel == "rbf":
        K1 = rbf_kernel(Z1, gamma=gamma)
        K2 = rbf_kernel(Z2, gamma=gamma)
    else:
        raise ValueError(f"Unknown kernel: {kernel}. Use 'linear' or 'rbf'")

    # Center the kernel matrices
    K1 = center_gram(K1)
    K2 = center_gram(K2)

    # CKA = HSIC(K1, K2) / sqrt(HSIC(K1, K1) * HSIC(K2, K2))
    # HSIC can be computed as trace(K1 @ K2) for centered kernels
    numerator = np.trace(K1 @ K2)
    denominator = np.sqrt(np.trace(K1 @ K1) * np.trace(K2 @ K2))

    if denominator < 1e-12:
        return 0.0

    return float(numerator / denominator)


def mmd(
    Z1: NDArray[np.floating],
    Z2: NDArray[np.floating],
    kernel: str = "rbf",
    gamma: float | None = None,
) -> float:
    """
    Maximum Mean Discrepancy (MMD) between two embedding matrices.

    MMD measures the distance between distributions in a reproducing kernel
    Hilbert space. Lower values indicate more similar distributions.

    Args:
        Z1: (n_samples, d1) embedding matrix
        Z2: (n_samples, d2) embedding matrix
        kernel: Kernel type - "linear", "rbf", or "polynomial"
        gamma: RBF bandwidth (only used if kernel="rbf").
               If None, uses median heuristic.

    Returns:
        float >= 0 where 0 = identical distributions

    Note:
        This computes the biased MMD estimator, which is consistent but
        has O(n^2) complexity.
    """
    Z1, Z2 = validate_inputs(Z1, Z2, require_same_dim=True)

    n = Z1.shape[0]

    # Compute kernel matrices
    if kernel == "linear":
        K11 = gram_matrix(Z1)
        K22 = gram_matrix(Z2)
        K12 = Z1 @ Z2.T
    elif kernel == "rbf":
        # Compute gamma using pooled data for symmetry
        if gamma is None:
            Z_all = np.vstack([Z1, Z2])
            sq_dists_all = (
                np.sum(Z_all**2, axis=1, keepdims=True)
                + np.sum(Z_all**2, axis=1)
                - 2 * Z_all @ Z_all.T
            )
            median_dist = np.median(np.sqrt(sq_dists_all[sq_dists_all > 0]))
            if median_dist < 1e-12:
                median_dist = 1.0
            gamma = 1.0 / (2 * median_dist**2)

        K11 = rbf_kernel(Z1, gamma=gamma)
        K22 = rbf_kernel(Z2, gamma=gamma)
        # Cross-kernel between Z1 and Z2
        sq_dists = (
            np.sum(Z1**2, axis=1, keepdims=True)
            + np.sum(Z2**2, axis=1)
            - 2 * Z1 @ Z2.T
        )
        sq_dists = np.maximum(sq_dists, 0)
        K12 = np.exp(-gamma * sq_dists)
    elif kernel == "polynomial":
        # Polynomial kernel: (1 + x.T @ y)^2
        K11 = (1 + gram_matrix(Z1)) ** 2
        K22 = (1 + gram_matrix(Z2)) ** 2
        K12 = (1 + Z1 @ Z2.T) ** 2
    else:
        raise ValueError(
            f"Unknown kernel: {kernel}. Use 'linear', 'rbf', or 'polynomial'"
        )

    # Biased MMD^2 = E[k(X,X')] + E[k(Y,Y')] - 2*E[k(X,Y)]
    mmd_sq = K11.sum() / (n * n) + K22.sum() / (n * n) - 2 * K12.sum() / (n * n)

    # Return MMD (take sqrt, ensuring non-negative)
    return float(np.sqrt(max(mmd_sq, 0.0)))


def compute_cka_mmap(file1: str, file2: str, n_rows: int, n_cols: int) -> float:
    """
    Compute CKA using memory-mapped kernel matrices via C++ extension.
    
    For large-scale computation where kernel matrices don't fit in memory.
    
    Args:
        file1: Path to binary file containing first kernel matrix
        file2: Path to binary file containing second kernel matrix  
        n_rows: Number of rows in kernel matrices
        n_cols: Number of columns in kernel matrices
        
    Returns:
        CKA score in [0, 1]
    """
    try:
        import pu_cka
        return pu_cka.compute_cka(file1, file2, n_rows, n_cols)
    except ImportError as e:
        raise ImportError(
            "C++ CKA extension (pu_cka) not found. Build with cmake or install the package."
        ) from e
