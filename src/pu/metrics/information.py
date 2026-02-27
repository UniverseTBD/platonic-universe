"""
Information-theoretic metrics: KL Divergence, JS Divergence, Mutual Information.
"""

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma

from pu.metrics._base import validate_inputs, center


def _gaussian_params(
    Z: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Estimate Gaussian parameters (mean, covariance) from samples.

    Args:
        Z: (n_samples, d) matrix

    Returns:
        Tuple of (mean, covariance) with regularization for stability
    """
    mean = np.mean(Z, axis=0)
    cov = np.cov(Z, rowvar=False)

    # Handle 1D case
    if cov.ndim == 0:
        cov = np.array([[cov]])

    # Regularize for numerical stability
    eps = 1e-6
    cov += eps * np.eye(cov.shape[0])

    return mean, cov


def kl_divergence(
    Z1: NDArray[np.floating],
    Z2: NDArray[np.floating],
) -> float:
    """
    KL divergence from Z1 to Z2 using Gaussian approximation.

    Approximates both distributions as multivariate Gaussians and computes
    the KL divergence D_KL(P || Q) where P ~ N(μ1, Σ1) and Q ~ N(μ2, Σ2).

    Args:
        Z1: (n_samples, d) embedding matrix (source distribution P)
        Z2: (n_samples, d) embedding matrix (target distribution Q)

    Returns:
        float >= 0 where 0 = identical distributions

    Note:
        KL divergence is asymmetric: KL(P||Q) ≠ KL(Q||P).
        Both matrices must have the same dimensions.

    Formula:
        D_KL(P || Q) = 0.5 * (tr(Σ2^{-1}Σ1) + (μ2-μ1)^T Σ2^{-1} (μ2-μ1)
                       - d + log(|Σ2|/|Σ1|))
    """
    Z1, Z2 = validate_inputs(Z1, Z2, require_same_dim=True)

    d = Z1.shape[1]

    # Estimate Gaussian parameters
    mu1, sigma1 = _gaussian_params(Z1)
    mu2, sigma2 = _gaussian_params(Z2)

    # Compute KL divergence for multivariate Gaussians
    sigma2_inv = np.linalg.inv(sigma2)
    diff = mu2 - mu1

    # tr(Σ2^{-1}Σ1)
    trace_term = np.trace(sigma2_inv @ sigma1)

    # (μ2-μ1)^T Σ2^{-1} (μ2-μ1)
    quad_term = diff @ sigma2_inv @ diff

    # log(|Σ2|/|Σ1|) = log|Σ2| - log|Σ1|
    _, logdet1 = np.linalg.slogdet(sigma1)
    _, logdet2 = np.linalg.slogdet(sigma2)
    logdet_term = logdet2 - logdet1

    kl = 0.5 * (trace_term + quad_term - d + logdet_term)

    # Ensure non-negative (numerical stability)
    return float(max(kl, 0.0))


def js_divergence(
    Z1: NDArray[np.floating],
    Z2: NDArray[np.floating],
) -> float:
    """
    Jensen-Shannon divergence using Gaussian approximation.

    A symmetric and bounded version of KL divergence:
    JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q).

    Args:
        Z1: (n_samples, d) embedding matrix
        Z2: (n_samples, d) embedding matrix

    Returns:
        float in [0, log(2)] where 0 = identical distributions

    Note:
        For Gaussian approximations, M is approximated as a Gaussian with:
        μ_M = 0.5 * (μ1 + μ2)
        Σ_M = 0.5 * (Σ1 + Σ2) + 0.25 * (μ1 - μ2)(μ1 - μ2)^T
    """
    Z1, Z2 = validate_inputs(Z1, Z2, require_same_dim=True)

    d = Z1.shape[1]

    # Estimate Gaussian parameters
    mu1, sigma1 = _gaussian_params(Z1)
    mu2, sigma2 = _gaussian_params(Z2)

    # Compute mixture distribution M parameters
    # For a mixture of two Gaussians, the mean is the average of means
    # The covariance is 0.5*(Σ1 + Σ2) + 0.25*(μ1 - μ2)(μ1 - μ2)^T
    mu_m = 0.5 * (mu1 + mu2)
    diff = mu1 - mu2
    sigma_m = 0.5 * (sigma1 + sigma2) + 0.25 * np.outer(diff, diff)
    # Note: sigma_m is already regularized since sigma1, sigma2 are regularized

    # Compute KL(P || M)
    sigma_m_inv = np.linalg.inv(sigma_m)
    diff1 = mu_m - mu1
    trace1 = np.trace(sigma_m_inv @ sigma1)
    quad1 = diff1 @ sigma_m_inv @ diff1
    _, logdet1 = np.linalg.slogdet(sigma1)
    _, logdet_m = np.linalg.slogdet(sigma_m)
    kl1 = 0.5 * (trace1 + quad1 - d + logdet_m - logdet1)

    # Compute KL(Q || M)
    diff2 = mu_m - mu2
    trace2 = np.trace(sigma_m_inv @ sigma2)
    quad2 = diff2 @ sigma_m_inv @ diff2
    _, logdet2 = np.linalg.slogdet(sigma2)
    kl2 = 0.5 * (trace2 + quad2 - d + logdet_m - logdet2)

    js = 0.5 * (max(kl1, 0.0) + max(kl2, 0.0))

    return float(js)


def mutual_information(
    Z1: NDArray[np.floating],
    Z2: NDArray[np.floating],
    k: int = 3,
) -> float:
    """
    Mutual information estimated using k-NN method (Kraskov estimator).

    Estimates I(X; Y) = H(X) + H(Y) - H(X, Y) using the k-nearest neighbor
    method which doesn't require binning or density estimation.

    Args:
        Z1: (n_samples, d1) embedding matrix (variable X)
        Z2: (n_samples, d2) embedding matrix (variable Y)
        k: Number of nearest neighbors to use

    Returns:
        float >= 0 (in nats, not bits)

    Note:
        This uses the Kraskov-Stögbauer-Grassberger (KSG) estimator.
        Higher k gives more stable but potentially biased estimates.

    Reference:
        Kraskov et al. (2004) "Estimating mutual information"
    """
    Z1, Z2 = validate_inputs(Z1, Z2)

    n = Z1.shape[0]
    d1, d2 = Z1.shape[1], Z2.shape[1]

    if k >= n:
        k = max(1, n - 1)

    # Concatenate for joint space
    Z_joint = np.hstack([Z1, Z2])

    # Find k-th nearest neighbor distances in joint space (Chebyshev/max norm)
    nn_joint = NearestNeighbors(n_neighbors=k + 1, metric="chebyshev")
    nn_joint.fit(Z_joint)
    distances, _ = nn_joint.kneighbors(Z_joint)
    eps = distances[:, k]  # k-th neighbor distance (excluding self)

    # Count neighbors within eps in marginal spaces
    nn1 = NearestNeighbors(metric="chebyshev")
    nn1.fit(Z1)

    nn2 = NearestNeighbors(metric="chebyshev")
    nn2.fit(Z2)

    # For each point, count neighbors within eps distance in each marginal
    nx = np.zeros(n)
    ny = np.zeros(n)

    for i in range(n):
        # Use radius slightly larger than eps to handle numerical issues
        eps_i = eps[i] + 1e-10

        # Count neighbors in X space
        neighbors_x = nn1.radius_neighbors([Z1[i]], radius=eps_i, return_distance=False)
        nx[i] = len(neighbors_x[0]) - 1  # Subtract 1 for self

        # Count neighbors in Y space
        neighbors_y = nn2.radius_neighbors([Z2[i]], radius=eps_i, return_distance=False)
        ny[i] = len(neighbors_y[0]) - 1  # Subtract 1 for self

    # KSG estimator: I(X;Y) ≈ ψ(k) - <ψ(nx+1) + ψ(ny+1)> + ψ(N)
    # where ψ is the digamma function
    mi = digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(n)

    # Ensure non-negative
    return float(max(mi, 0.0))
