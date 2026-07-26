"""
Shared utilities for metrics computation.
"""

import numpy as np
from numpy.typing import NDArray


def validate_inputs(
    Z1: NDArray[np.floating],
    Z2: NDArray[np.floating],
    require_same_dim: bool = False,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Validate and prepare input matrices for metric computation.

    Args:
        Z1: (n_samples, d1) embedding matrix
        Z2: (n_samples, d2) embedding matrix
        require_same_dim: If True, require d1 == d2

    Returns:
        Tuple of validated arrays as float64

    Raises:
        ValueError: If inputs are invalid
    """
    Z1 = np.asarray(Z1, dtype=np.float64)
    Z2 = np.asarray(Z2, dtype=np.float64)

    if Z1.ndim != 2 or Z2.ndim != 2:
        raise ValueError("Inputs must be 2D arrays")

    if Z1.shape[0] != Z2.shape[0]:
        raise ValueError(
            f"Number of samples must match: {Z1.shape[0]} != {Z2.shape[0]}"
        )

    if Z1.shape[0] == 0:
        raise ValueError("Inputs must have at least one sample")

    if require_same_dim and Z1.shape[1] != Z2.shape[1]:
        raise ValueError(
            f"Dimensions must match: {Z1.shape[1]} != {Z2.shape[1]}"
        )

    return Z1, Z2


def center(Z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Mean-center columns of a matrix.

    Args:
        Z: (n_samples, d) matrix

    Returns:
        Column-centered matrix
    """
    return Z - Z.mean(axis=0, keepdims=True)


def normalize_rows(Z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    L2-normalize rows of a matrix.

    Args:
        Z: (n_samples, d) matrix

    Returns:
        Row-normalized matrix (each row has unit L2 norm)
    """
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return Z / norms


def gram_matrix(Z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Compute the Gram matrix (linear kernel).

    Args:
        Z: (n_samples, d) matrix

    Returns:
        (n_samples, n_samples) Gram matrix Z @ Z.T
    """
    return Z @ Z.T


def center_gram(K: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Center a Gram matrix (HSIC centering).

    Args:
        K: (n, n) Gram matrix

    Returns:
        Centered Gram matrix H @ K @ H where H = I - 1/n
    """
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def rbf_kernel(
    Z: NDArray[np.floating], gamma: float | None = None
) -> NDArray[np.floating]:
    """
    Compute RBF (Gaussian) kernel matrix.

    Args:
        Z: (n_samples, d) matrix
        gamma: RBF bandwidth parameter. If None, uses 1/(2 * median distance^2)

    Returns:
        (n_samples, n_samples) RBF kernel matrix
    """
    # Compute pairwise squared distances
    sq_dists = (
        np.sum(Z**2, axis=1, keepdims=True)
        + np.sum(Z**2, axis=1)
        - 2 * Z @ Z.T
    )
    sq_dists = np.maximum(sq_dists, 0)  # Numerical stability

    if gamma is None:
        # Use median heuristic
        median_dist = np.median(np.sqrt(sq_dists[sq_dists > 0]))
        if median_dist < 1e-12:
            median_dist = 1.0
        gamma = 1.0 / (2 * median_dist**2)

    return np.exp(-gamma * sq_dists)
