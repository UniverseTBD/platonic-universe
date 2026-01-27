"""
Neighbor-based similarity metrics: MKNN, Jaccard, RSA.
"""

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, pearsonr

from pu.metrics._base import validate_inputs


def mknn(
    Z1: NDArray[np.floating],
    Z2: NDArray[np.floating],
    k: int = 10,
) -> float:
    """
    Mutual k-Nearest Neighbors overlap.

    Measures the overlap between k-nearest neighbor sets in two
    embedding spaces. For each sample, finds its k nearest neighbors
    in both spaces and computes the intersection.

    Args:
        Z1: (n_samples, d1) embedding matrix
        Z2: (n_samples, d2) embedding matrix
        k: Number of nearest neighbors

    Returns:
        float in [0, 1] where 1 = identical neighbor sets

    Note:
        Uses cosine distance for neighbor computation.
    """
    Z1, Z2 = validate_inputs(Z1, Z2)

    n = Z1.shape[0]
    if k >= n:
        k = max(1, n - 1)

    nn1 = (
        NearestNeighbors(n_neighbors=k, metric="cosine")
        .fit(Z1)
        .kneighbors(return_distance=False)
    )
    nn2 = (
        NearestNeighbors(n_neighbors=k, metric="cosine")
        .fit(Z2)
        .kneighbors(return_distance=False)
    )

    overlap = [len(set(a).intersection(b)) for a, b in zip(nn1, nn2)]

    return float(np.mean(overlap) / k)


def jaccard(
    Z1: NDArray[np.floating],
    Z2: NDArray[np.floating],
    k: int = 10,
) -> float:
    """
    Jaccard index of k-nearest neighbor sets.

    More strict than MKNN - measures the ratio of intersection to union
    of neighbor sets.

    Args:
        Z1: (n_samples, d1) embedding matrix
        Z2: (n_samples, d2) embedding matrix
        k: Number of nearest neighbors

    Returns:
        float in [0, 1] where 1 = identical neighbor sets

    Note:
        Uses cosine distance for neighbor computation.
    """
    Z1, Z2 = validate_inputs(Z1, Z2)

    n = Z1.shape[0]
    if k >= n:
        k = max(1, n - 1)

    nn1 = (
        NearestNeighbors(n_neighbors=k, metric="cosine")
        .fit(Z1)
        .kneighbors(return_distance=False)
    )
    nn2 = (
        NearestNeighbors(n_neighbors=k, metric="cosine")
        .fit(Z2)
        .kneighbors(return_distance=False)
    )

    jaccard_scores = [
        len(set(a).intersection(b)) / len(set(a).union(b)) for a, b in zip(nn1, nn2)
    ]

    return float(np.mean(jaccard_scores))


def rsa(
    Z1: NDArray[np.floating],
    Z2: NDArray[np.floating],
    method: str = "spearman",
    metric: str = "cosine",
) -> float:
    """
    Representational Similarity Analysis (RSA).

    Computes pairwise distances between samples in each embedding space,
    then correlates these distance matrices. This measures whether the
    two embedding spaces preserve similar relational structure.

    Args:
        Z1: (n_samples, d1) embedding matrix
        Z2: (n_samples, d2) embedding matrix
        method: Correlation method - "spearman" (rank, more robust) or "pearson"
        metric: Distance metric - "cosine", "euclidean", "correlation", etc.

    Returns:
        float in [-1, 1] where 1 = perfect agreement, -1 = perfect disagreement

    Reference:
        Kriegeskorte et al. (2008) "Representational similarity analysis"
    """
    Z1, Z2 = validate_inputs(Z1, Z2)

    # Compute pairwise distances (condensed upper triangle)
    dist1 = pdist(Z1, metric=metric)
    dist2 = pdist(Z2, metric=metric)

    # Compute correlation
    if method == "spearman":
        corr, _ = spearmanr(dist1, dist2)
    elif method == "pearson":
        corr, _ = pearsonr(dist1, dist2)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'spearman' or 'pearson'")

    return float(corr)
