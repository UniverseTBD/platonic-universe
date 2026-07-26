"""
Geometric similarity metrics: Procrustes, Cosine Similarity, Frechet Distance.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import orthogonal_procrustes, sqrtm

from pu.metrics._base import validate_inputs, center, normalize_rows


def procrustes(
    Z1: NDArray[np.floating],
    Z2: NDArray[np.floating],
) -> float:
    """
    Procrustes distance between two embedding matrices.

    Finds the optimal orthogonal transformation to align Z1 to Z2,
    then returns the Frobenius norm of the residual (normalized).

    Args:
        Z1: (n_samples, d) embedding matrix
        Z2: (n_samples, d) embedding matrix (must have same dimensions as Z1)

    Returns:
        float >= 0 where 0 = perfect alignment after orthogonal transformation

    Note:
        Both matrices are centered and Frobenius-normalized before alignment.
        Lower values indicate more similar representations.
    """
    Z1, Z2 = validate_inputs(Z1, Z2, require_same_dim=True)

    # Center the matrices
    Z1 = center(Z1)
    Z2 = center(Z2)

    # Normalize by Frobenius norm
    norm1 = np.linalg.norm(Z1, "fro")
    norm2 = np.linalg.norm(Z2, "fro")

    if norm1 < 1e-12 or norm2 < 1e-12:
        return 0.0

    Z1 = Z1 / norm1
    Z2 = Z2 / norm2

    # Find optimal orthogonal transformation R such that Z1 @ R ≈ Z2
    R, _ = orthogonal_procrustes(Z1, Z2)

    # Compute residual
    residual = Z1 @ R - Z2
    distance = np.linalg.norm(residual, "fro")

    return float(distance)


def cosine_similarity(
    Z1: NDArray[np.floating],
    Z2: NDArray[np.floating],
) -> float:
    """
    Mean pairwise cosine similarity between corresponding rows.

    Args:
        Z1: (n_samples, d) embedding matrix
        Z2: (n_samples, d) embedding matrix (must have same dimensions as Z1)

    Returns:
        float in [-1, 1] where 1 = perfectly aligned directions

    Note:
        Computes cosine similarity for each sample pair (row i of Z1 vs row i of Z2)
        and returns the mean. This measures how well the embedding directions align.
    """
    Z1, Z2 = validate_inputs(Z1, Z2, require_same_dim=True)

    # Normalize rows to unit length
    Z1_norm = normalize_rows(Z1)
    Z2_norm = normalize_rows(Z2)

    # Compute cosine similarity for each pair
    cos_sim = np.sum(Z1_norm * Z2_norm, axis=1)

    return float(np.mean(cos_sim))


def frechet(
    Z1: NDArray[np.floating],
    Z2: NDArray[np.floating],
) -> float:
    """
    Fréchet distance (Wasserstein-2) between Gaussian approximations.

    Models each embedding set as a multivariate Gaussian and computes
    the 2-Wasserstein distance between them. This is the same metric
    used in FID (Fréchet Inception Distance).

    Args:
        Z1: (n_samples, d1) embedding matrix
        Z2: (n_samples, d2) embedding matrix

    Returns:
        float >= 0 where 0 = identical distributions

    Note:
        When dimensions differ, the smaller matrix is zero-padded.
        The distance is computed as:
        ||μ1 - μ2||² + Tr(Σ1 + Σ2 - 2(Σ1^{1/2} Σ2 Σ1^{1/2})^{1/2})
    """
    Z1, Z2 = validate_inputs(Z1, Z2)

    # Pad to same dimension if needed
    d1, d2 = Z1.shape[1], Z2.shape[1]
    if d1 < d2:
        Z1 = np.pad(Z1, ((0, 0), (0, d2 - d1)))
    elif d2 < d1:
        Z2 = np.pad(Z2, ((0, 0), (0, d1 - d2)))

    # Compute means
    mu1 = np.mean(Z1, axis=0)
    mu2 = np.mean(Z2, axis=0)

    # Compute covariances
    # Use n-1 normalization for unbiased estimator
    sigma1 = np.cov(Z1, rowvar=False)
    sigma2 = np.cov(Z2, rowvar=False)

    # Handle 1D case
    if sigma1.ndim == 0:
        sigma1 = np.array([[sigma1]])
    if sigma2.ndim == 0:
        sigma2 = np.array([[sigma2]])

    # Mean difference term
    diff = mu1 - mu2
    mean_term = np.dot(diff, diff)

    # Covariance term: Tr(Σ1 + Σ2 - 2(Σ1^{1/2} Σ2 Σ1^{1/2})^{1/2})
    # Use matrix square root
    sqrt_sigma1 = sqrtm(sigma1)

    # Handle complex results from sqrtm (numerical issues)
    if np.iscomplexobj(sqrt_sigma1):
        sqrt_sigma1 = sqrt_sigma1.real

    product = sqrt_sigma1 @ sigma2 @ sqrt_sigma1
    sqrt_product = sqrtm(product)

    if np.iscomplexobj(sqrt_product):
        sqrt_product = sqrt_product.real

    cov_term = np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(sqrt_product)

    # Ensure non-negative (numerical stability)
    distance_sq = mean_term + cov_term
    distance_sq = max(distance_sq, 0.0)

    return float(np.sqrt(distance_sq))
