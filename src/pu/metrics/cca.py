"""
Canonical Correlation Analysis based metrics: SVCCA and PWCCA.
"""

import numpy as np
from numpy.typing import NDArray

from pu.metrics._base import validate_inputs, center


def _svd_reduce(
    Z: NDArray[np.floating], threshold: float = 0.99
) -> NDArray[np.floating]:
    """
    Reduce dimensionality using SVD, keeping components that explain
    the given threshold of variance.

    Args:
        Z: (n_samples, d) matrix
        threshold: Fraction of variance to retain (0 to 1)

    Returns:
        (n_samples, k) reduced matrix where k is chosen to explain threshold variance
    """
    # Center the matrix
    Z = center(Z)

    # Compute SVD
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)

    # Compute variance explained
    var_explained = S**2
    var_ratio = var_explained / var_explained.sum()
    cumsum = np.cumsum(var_ratio)

    # Find number of components to keep
    k = np.searchsorted(cumsum, threshold) + 1
    k = min(k, len(S))

    # Return reduced representation
    return U[:, :k] * S[:k]


def _cca_correlations(
    Z1: NDArray[np.floating], Z2: NDArray[np.floating]
) -> NDArray[np.floating]:
    """
    Compute CCA correlations between two matrices.

    Args:
        Z1: (n_samples, d1) matrix
        Z2: (n_samples, d2) matrix

    Returns:
        Array of canonical correlations (length = min(d1, d2))
    """
    n = Z1.shape[0]
    d1, d2 = Z1.shape[1], Z2.shape[1]

    # Center the matrices
    Z1 = center(Z1)
    Z2 = center(Z2)

    # Compute covariance matrices
    S11 = Z1.T @ Z1 / (n - 1)
    S22 = Z2.T @ Z2 / (n - 1)
    S12 = Z1.T @ Z2 / (n - 1)

    # Regularize for numerical stability
    eps = 1e-8
    S11 += eps * np.eye(d1)
    S22 += eps * np.eye(d2)

    # Compute inverse square roots
    U1, s1, _ = np.linalg.svd(S11)
    U2, s2, _ = np.linalg.svd(S22)

    s1_inv_sqrt = 1.0 / np.sqrt(s1)
    s2_inv_sqrt = 1.0 / np.sqrt(s2)

    S11_inv_sqrt = U1 @ np.diag(s1_inv_sqrt) @ U1.T
    S22_inv_sqrt = U2 @ np.diag(s2_inv_sqrt) @ U2.T

    # Compute canonical correlation matrix
    M = S11_inv_sqrt @ S12 @ S22_inv_sqrt

    # SVD to get canonical correlations
    _, correlations, _ = np.linalg.svd(M)

    # Clamp to [0, 1] for numerical stability
    correlations = np.clip(correlations, 0, 1)

    return correlations


def svcca(
    Z1: NDArray[np.floating],
    Z2: NDArray[np.floating],
    threshold: float = 0.99,
) -> float:
    """
    Singular Vector Canonical Correlation Analysis (SVCCA).

    Reduces both representations using SVD to remove noise dimensions,
    then computes CCA and returns the mean canonical correlation.

    Args:
        Z1: (n_samples, d1) embedding matrix
        Z2: (n_samples, d2) embedding matrix
        threshold: Fraction of variance to retain in SVD step (0 to 1)

    Returns:
        float in [0, 1] where 1 = perfect alignment

    Reference:
        Raghu et al. (2017) "SVCCA: Singular Vector Canonical Correlation
        Analysis for Deep Learning Dynamics and Interpretability"
    """
    Z1, Z2 = validate_inputs(Z1, Z2)

    # SVD reduction
    Z1_reduced = _svd_reduce(Z1, threshold=threshold)
    Z2_reduced = _svd_reduce(Z2, threshold=threshold)

    # Handle edge cases
    if Z1_reduced.shape[1] == 0 or Z2_reduced.shape[1] == 0:
        return 0.0

    # Compute CCA correlations
    correlations = _cca_correlations(Z1_reduced, Z2_reduced)

    if len(correlations) == 0:
        return 0.0

    return float(np.mean(correlations))


def pwcca(
    Z1: NDArray[np.floating],
    Z2: NDArray[np.floating],
    threshold: float = 0.99,
) -> float:
    """
    Projection Weighted Canonical Correlation Analysis (PWCCA).

    Similar to SVCCA but weights each canonical correlation by how much
    variance it explains in the original representations, making it more
    robust to noise.

    Args:
        Z1: (n_samples, d1) embedding matrix
        Z2: (n_samples, d2) embedding matrix
        threshold: Fraction of variance to retain in SVD step (0 to 1)

    Returns:
        float in [0, 1] where 1 = perfect alignment

    Reference:
        Morcos et al. (2018) "Insights on representational similarity in
        neural networks with canonical correlation"
    """
    Z1, Z2 = validate_inputs(Z1, Z2)

    # Center the matrices
    Z1_centered = center(Z1)
    Z2_centered = center(Z2)

    # SVD reduction
    Z1_reduced = _svd_reduce(Z1, threshold=threshold)
    Z2_reduced = _svd_reduce(Z2, threshold=threshold)

    # Handle edge cases
    if Z1_reduced.shape[1] == 0 or Z2_reduced.shape[1] == 0:
        return 0.0

    n = Z1.shape[0]
    d1_reduced = Z1_reduced.shape[1]

    # Compute covariance matrices for CCA
    S11 = Z1_reduced.T @ Z1_reduced / (n - 1)
    S22 = Z2_reduced.T @ Z2_reduced / (n - 1)
    S12 = Z1_reduced.T @ Z2_reduced / (n - 1)

    # Regularize
    eps = 1e-8
    S11 += eps * np.eye(d1_reduced)
    S22 += eps * np.eye(Z2_reduced.shape[1])

    # Compute inverse square roots
    U1, s1, _ = np.linalg.svd(S11)
    U2, s2, _ = np.linalg.svd(S22)

    s1_inv_sqrt = 1.0 / np.sqrt(s1)
    s2_inv_sqrt = 1.0 / np.sqrt(s2)

    S11_inv_sqrt = U1 @ np.diag(s1_inv_sqrt) @ U1.T
    S22_inv_sqrt = U2 @ np.diag(s2_inv_sqrt) @ U2.T

    # Compute canonical correlation matrix
    M = S11_inv_sqrt @ S12 @ S22_inv_sqrt

    # SVD to get canonical correlations and directions
    U, correlations, Vt = np.linalg.svd(M)
    correlations = np.clip(correlations, 0, 1)

    if len(correlations) == 0:
        return 0.0

    # Compute canonical directions in original reduced space
    A = S11_inv_sqrt @ U  # Canonical directions for Z1_reduced

    # Compute projection weights (variance explained by each direction)
    # Project Z1_reduced onto canonical directions
    projections = Z1_reduced @ A

    # Compute variance of each projection
    weights = np.var(projections, axis=0)

    # Normalize weights
    total_weight = weights.sum()
    if total_weight < 1e-12:
        return float(np.mean(correlations))

    weights = weights / total_weight

    # Weight the correlations
    # Use only as many weights as correlations
    n_components = min(len(correlations), len(weights))
    weighted_corr = np.sum(correlations[:n_components] * weights[:n_components])

    return float(weighted_corr)
