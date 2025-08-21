import logging
import numpy as np
from sklearn.neighbors import NearestNeighbors

def mknn_score(Z1: np.ndarray, Z2: np.ndarray, k: int = 10, batch: int = None, repeats: int = 3, seed: int = 42) -> float:
    """
    Calculates the mean k-nearest neighbor (mknn) overlap between two embedding sets
    using the robust PRH Eq.11 implementation with cosine similarity via L2-normalization.

    Args:
        Z1 (np.ndarray): The first set of embeddings.
        Z2 (np.ndarray): The second set of embeddings.
        k (int): The number of neighbors to consider.
        batch (int, optional): Batch size for random sampling. If None, uses full dataset.
        repeats (int): Number of repetitions for batched computation (default: 3).
        seed (int): Random seed for reproducibility (default: 42).

    Returns:
        float: The mean k-nearest neighbor overlap score, normalized by k.
    """
    logging.info(f"Computing MKNN score with k={k}, batch={batch}, repeats={repeats}...")
    
    assert Z1.shape[0] == Z2.shape[0], "Row-aligned arrays required."
    N = Z1.shape[0]
    if N < 2:
        logging.warning("Too few samples for MKNN computation.")
        return 0.0

    def _norm(X):
        """L2-normalize embeddings for cosine similarity via Euclidean distance."""
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    # Normalize embeddings for cosine similarity
    A = _norm(Z1)
    B = _norm(Z2)
    rng = np.random.default_rng(seed)

    def _compute_overlap(idxs):
        """Compute MKNN overlap for given indices."""
        X1, X2 = A[idxs], B[idxs]
        b = len(idxs)
        kk = min(k, b - 1)
        if kk <= 0:
            return 0.0
        
        # Find k nearest neighbors (sklearn automatically excludes self)
        nn1 = NearestNeighbors(n_neighbors=kk, metric="euclidean").fit(X1)
        n1 = nn1.kneighbors(return_distance=False)
        
        nn2 = NearestNeighbors(n_neighbors=kk, metric="euclidean").fit(X2)
        n2 = nn2.kneighbors(return_distance=False)
        
        # Calculate overlap for each point
        overlaps = [len(set(n1[i]) & set(n2[i])) / kk for i in range(b)]
        return float(np.mean(overlaps))

    # Use full dataset if no batching specified
    if batch is None or batch >= N:
        score = _compute_overlap(np.arange(N))
        logging.info(f"MKNN computation complete (full dataset). Score: {score:.4f}")
        return score
    
    # Batched computation with multiple repeats for robustness
    vals = []
    for _ in range(repeats):
        idxs = rng.choice(N, size=batch, replace=False)
        vals.append(_compute_overlap(idxs))
    
    score = float(np.mean(vals))
    logging.info(f"MKNN computation complete (batched: {batch}x{repeats}). Score: {score:.4f}")
    return score


def compute_mknn_simple(Z1: np.ndarray, Z2: np.ndarray, k: int = 10) -> float:
    """
    Calculate mutual k nearest neighbours (simple implementation from script).
    No normalization, no projection, no padding. Separate kNN graphs in each space.
    
    Args:
        Z1 (np.ndarray): The first set of embeddings.
        Z2 (np.ndarray): The second set of embeddings.
        k (int): The number of neighbors to consider.
        
    Returns:
        float: The mean k-nearest neighbor overlap score, normalized by k.
    """
    assert len(Z1) == len(Z2)
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
    return np.mean(overlap) / k