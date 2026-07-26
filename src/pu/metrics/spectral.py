"""
Spectral similarity metrics: Tucker Congruence, Eigenspectrum Distance, Riemannian Distance.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import logm

from pu.metrics._base import validate_inputs, center


def tucker_congruence(
    Z1: NDArray[np.floating],
    Z2: NDArray[np.floating],
) -> float:
    """
    Tucker's Congruence Coefficient (factor congruence coefficient).

    Measures the similarity between two sets of factor loadings or embeddings
    by computing the mean absolute cosine similarity between corresponding
    columns.

    Args:
        Z1: (n_samples, d) embedding matrix
        Z2: (n_samples, d) embedding matrix (must have same dimensions as Z1)

    Returns:
        float in [0, 1] where 1 = perfect congruence

    Note:
        This metric compares column-wise structure. Values > 0.95 are typically
        considered "excellent" congruence, > 0.85 "good", > 0.65 "fair".

    Reference:
        Tucker (1951) "A method for synthesis of factor analysis studies"
    """
    Z1, Z2 = validate_inputs(Z1, Z2, require_same_dim=True)

    n_cols = Z1.shape[1]

    # Compute congruence coefficient for each column pair
    congruences = []
    for i in range(n_cols):
        col1 = Z1[:, i]
        col2 = Z2[:, i]

        # Tucker's coefficient: (x.T @ y) / sqrt((x.T @ x)(y.T @ y))
        numerator = np.dot(col1, col2)
        denominator = np.sqrt(np.dot(col1, col1) * np.dot(col2, col2))

        if denominator < 1e-12:
            congruences.append(0.0)
        else:
            # Take absolute value since sign is arbitrary for factors
            congruences.append(abs(numerator / denominator))

    return float(np.mean(congruences))


def eigenspectrum(
    Z1: NDArray[np.floating],
    Z2: NDArray[np.floating],
    normalize: bool = True,
) -> float:
    """
    Eigenspectrum distance between covariance matrices.

    Computes the L2 distance between the sorted eigenvalues of the
    covariance matrices of the two embeddings.

    Args:
        Z1: (n_samples, d1) embedding matrix
        Z2: (n_samples, d2) embedding matrix
        normalize: If True, normalize eigenvalues to sum to 1

    Returns:
        float >= 0 where 0 = identical eigenspectra

    Note:
        When dimensions differ, the smaller eigenspectrum is zero-padded.
        This metric captures differences in the variance structure of
        the representations.
    """
    Z1, Z2 = validate_inputs(Z1, Z2)

    # Center the matrices
    Z1 = center(Z1)
    Z2 = center(Z2)

    # Compute covariance matrices
    n = Z1.shape[0]
    cov1 = Z1.T @ Z1 / (n - 1)
    cov2 = Z2.T @ Z2 / (n - 1)

    # Compute eigenvalues (sorted in descending order)
    eig1 = np.sort(np.linalg.eigvalsh(cov1))[::-1]
    eig2 = np.sort(np.linalg.eigvalsh(cov2))[::-1]

    # Ensure non-negative (numerical stability)
    eig1 = np.maximum(eig1, 0)
    eig2 = np.maximum(eig2, 0)

    # Pad to same length
    max_len = max(len(eig1), len(eig2))
    eig1 = np.pad(eig1, (0, max_len - len(eig1)))
    eig2 = np.pad(eig2, (0, max_len - len(eig2)))

    if normalize:
        # Normalize to sum to 1 (compare relative variance distribution)
        sum1 = eig1.sum()
        sum2 = eig2.sum()
        if sum1 > 1e-12:
            eig1 = eig1 / sum1
        if sum2 > 1e-12:
            eig2 = eig2 / sum2

    # L2 distance
    distance = np.linalg.norm(eig1 - eig2)

    return float(distance)


def riemannian(
    Z1: NDArray[np.floating],
    Z2: NDArray[np.floating],
    metric: str = "log_euclidean",
) -> float:
    """
    Riemannian distance between covariance matrices.

    Computes the geodesic distance on the manifold of symmetric positive
    definite (SPD) matrices.

    Args:
        Z1: (n_samples, d) embedding matrix
        Z2: (n_samples, d) embedding matrix (must have same dimensions as Z1)
        metric: Distance metric on SPD manifold
            - "log_euclidean": Log-Euclidean distance (default)
            - "affine_invariant": Affine-invariant Riemannian metric

    Returns:
        float >= 0 where 0 = identical covariance structures

    Note:
        Both matrices must have the same dimensions. The covariance matrices
        are regularized slightly for numerical stability.

    Reference:
        Arsigny et al. (2007) "Geometric means in a novel vector space
        structure on symmetric positive-definite matrices"
    """
    Z1, Z2 = validate_inputs(Z1, Z2, require_same_dim=True)

    # Center the matrices
    Z1 = center(Z1)
    Z2 = center(Z2)

    # Compute covariance matrices
    n = Z1.shape[0]
    d = Z1.shape[1]
    cov1 = Z1.T @ Z1 / (n - 1)
    cov2 = Z2.T @ Z2 / (n - 1)

    # Regularize for numerical stability (ensure SPD)
    eps = 1e-6
    cov1 += eps * np.eye(d)
    cov2 += eps * np.eye(d)

    if metric == "log_euclidean":
        # Log-Euclidean distance: ||log(Σ1) - log(Σ2)||_F
        log_cov1 = logm(cov1)
        log_cov2 = logm(cov2)

        # Handle complex results (numerical issues)
        if np.iscomplexobj(log_cov1):
            log_cov1 = log_cov1.real
        if np.iscomplexobj(log_cov2):
            log_cov2 = log_cov2.real

        distance = np.linalg.norm(log_cov1 - log_cov2, "fro")

    elif metric == "affine_invariant":
        # Affine-invariant distance: ||log(Σ1^{-1/2} Σ2 Σ1^{-1/2})||_F
        # Compute Σ1^{-1/2}
        eigvals, eigvecs = np.linalg.eigh(cov1)
        eigvals = np.maximum(eigvals, eps)
        cov1_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        # Compute Σ1^{-1/2} Σ2 Σ1^{-1/2}
        product = cov1_inv_sqrt @ cov2 @ cov1_inv_sqrt

        # Compute log of product
        log_product = logm(product)

        if np.iscomplexobj(log_product):
            log_product = log_product.real

        distance = np.linalg.norm(log_product, "fro")

    else:
        raise ValueError(
            f"Unknown metric: {metric}. Use 'log_euclidean' or 'affine_invariant'"
        )

    return float(distance)
