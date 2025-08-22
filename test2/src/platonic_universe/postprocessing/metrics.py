import logging
import numpy as np
import heapq
import gc
from typing import List, Tuple
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

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


class TopKTracker:
    """Efficient top-k tracking using heaps for memory-efficient kNN."""
    
    def __init__(self, n_samples: int, k: int, chunk_size: int):
        self.k = k
        self.n_samples = n_samples
        self.chunk_size = chunk_size
        # Each sample gets a max-heap of (negative_distance, chunk_idx, within_chunk_idx)
        self.heaps = [[] for _ in range(n_samples)]
        
    def update_neighbors(self, sample_indices: np.ndarray, distances: np.ndarray, target_chunk_idx: int, target_chunk_start: int):
        """Update top-k neighbors for samples based on new distance calculations.
        
        Args:
            sample_indices: Indices of samples being processed (within current chunk)
            distances: Distance matrix (len(sample_indices) x chunk_size)
            target_chunk_idx: Index of the target chunk for global reference
            target_chunk_start: Starting global index of the target chunk
        """
        for i, sample_idx in enumerate(sample_indices):
            # Get distances for this sample to all targets in current chunk
            sample_distances = distances[i]
            
            # Add all valid neighbors (excluding self if same chunk)
            for j, dist in enumerate(sample_distances):
                # Calculate global target index
                global_target_idx = target_chunk_start + j
                
                # Skip self-connections
                if sample_idx == global_target_idx:
                    continue
                    
                heap = self.heaps[sample_idx]
                
                if len(heap) < self.k:
                    # Haven't found k neighbors yet, add this one
                    heapq.heappush(heap, (-dist, global_target_idx))
                elif len(heap) > 0 and -heap[0][0] > dist:  # Current worst is farther than this one
                    # Replace worst neighbor with this better one
                    heapq.heapreplace(heap, (-dist, global_target_idx))
                    
    def get_neighbor_indices(self, sample_idx: int) -> List[int]:
        """Get the global indices of top-k neighbors for a sample."""
        heap = self.heaps[sample_idx]
        neighbors = []
        for neg_dist, global_idx in heap:
            neighbors.append(global_idx)
        return neighbors


def _normalize_chunk(X: np.ndarray) -> np.ndarray:
    """L2-normalize a chunk of embeddings for cosine similarity."""
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def _compute_chunk_distances(chunk1: np.ndarray, chunk2: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Compute distances between two chunks efficiently."""
    if metric == "euclidean":
        return euclidean_distances(chunk1, chunk2)
    elif metric == "cosine":
        return cosine_distances(chunk1, chunk2)
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def mknn_score_memory_efficient(
    Z1: np.ndarray, 
    Z2: np.ndarray, 
    k: int = 10, 
    chunk_size: int = None,
    batch: int = None, 
    repeats: int = 3, 
    seed: int = 42
) -> float:
    """
    Memory-efficient MKNN computation for large datasets (200GB+).
    Uses chunked processing to maintain O(N*k + chunk_size²) memory complexity.
    
    Args:
        Z1, Z2: Embedding arrays (must be row-aligned)
        k: Number of neighbors
        chunk_size: Samples per chunk (default: auto-detect based on memory)
        batch: If specified, sample this many points for estimation
        repeats: Number of repeats for batched estimation
        seed: Random seed
        
    Returns:
        float: MKNN overlap score
    """
    logging.info(f"Computing memory-efficient MKNN score with k={k}, chunk_size={chunk_size}...")
    
    assert Z1.shape[0] == Z2.shape[0], "Row-aligned arrays required."
    N = Z1.shape[0]
    
    if N < 2:
        logging.warning("Too few samples for MKNN computation.")
        return 0.0
    
    # Auto-detect chunk size based on available memory and embedding dimension
    if chunk_size is None:
        # Estimate memory usage: chunk_size² * 8 bytes + overhead
        # Target ~1GB chunks for safety
        target_memory = 1024 * 1024 * 1024  # 1GB
        embedding_size = Z1.shape[1] * 8  # 8 bytes per float64
        # Distance matrix is chunk_size² * 8 bytes
        chunk_size = int(np.sqrt(target_memory // 8))
        chunk_size = min(chunk_size, N // 4, 50000)  # Reasonable limits
        chunk_size = max(chunk_size, k + 1, 100)  # Minimum viable chunk
    
    logging.info(f"Using chunk_size={chunk_size} for {N} samples")
    
    # Handle small datasets with original algorithm
    if N <= chunk_size * 2:
        logging.info("Dataset small enough for standard algorithm")
        return mknn_score(Z1, Z2, k, batch, repeats, seed)
    
    rng = np.random.default_rng(seed)
    
    def _compute_chunked_mknn(sample_indices: np.ndarray) -> float:
        """Compute MKNN for specified sample indices using chunked processing."""
        n_samples = len(sample_indices)
        
        # Initialize top-k trackers for both spaces
        tracker1 = TopKTracker(n_samples, k, chunk_size)
        tracker2 = TopKTracker(n_samples, k, chunk_size)
        
        # Process in chunks
        n_chunks = (N + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, N)
            
            # Extract and normalize chunks
            chunk1 = _normalize_chunk(Z1[start_idx:end_idx])
            chunk2 = _normalize_chunk(Z2[start_idx:end_idx])
            
            # For each sample in our analysis set, compute distances to this chunk
            for i, sample_idx in enumerate(sample_indices):
                if sample_idx < len(Z1):  # Valid sample
                    # Get normalized sample embeddings
                    sample1 = _normalize_chunk(Z1[sample_idx:sample_idx+1])
                    sample2 = _normalize_chunk(Z2[sample_idx:sample_idx+1])
                    
                    # Compute distances to chunk
                    dist1 = _compute_chunk_distances(sample1, chunk1, "euclidean")
                    dist2 = _compute_chunk_distances(sample2, chunk2, "euclidean")
                    
                    # Update top-k trackers
                    tracker1.update_neighbors([i], dist1, chunk_idx, start_idx)
                    tracker2.update_neighbors([i], dist2, chunk_idx, start_idx)
            
            # Clear memory
            del chunk1, chunk2
            gc.collect()
        
        # Compute overlaps
        overlaps = []
        for i in range(n_samples):
            neighbors1 = set(tracker1.get_neighbor_indices(i)[:k])
            neighbors2 = set(tracker2.get_neighbor_indices(i)[:k])
            overlap = len(neighbors1 & neighbors2) / k
            overlaps.append(overlap)
        
        return float(np.mean(overlaps))
    
    # Use sampling if requested
    if batch is None or batch >= N:
        sample_indices = np.arange(N)
        score = _compute_chunked_mknn(sample_indices)
        logging.info(f"Memory-efficient MKNN complete (full dataset). Score: {score:.4f}")
        return score
    else:
        # Batched computation
        scores = []
        for _ in range(repeats):
            sample_indices = rng.choice(N, size=batch, replace=False)
            score = _compute_chunked_mknn(sample_indices)
            scores.append(score)
        
        final_score = float(np.mean(scores))
        logging.info(f"Memory-efficient MKNN complete (batched: {batch}x{repeats}). Score: {final_score:.4f}")
        return final_score


def compute_mknn_simple_memory_efficient(
    Z1: np.ndarray, 
    Z2: np.ndarray, 
    k: int = 10,
    chunk_size: int = None
) -> float:
    """
    Memory-efficient simple MKNN computation for large datasets.
    Uses chunked processing with cosine distance metric.
    
    Args:
        Z1, Z2: Embedding arrays  
        k: Number of neighbors
        chunk_size: Samples per chunk (auto-detect if None)
        
    Returns:
        float: MKNN overlap score
    """
    logging.info(f"Computing memory-efficient simple MKNN with k={k}, chunk_size={chunk_size}...")
    
    assert len(Z1) == len(Z2), "Arrays must have same length"
    N = len(Z1)
    
    if N < 2:
        return 0.0
    
    # Auto-detect chunk size
    if chunk_size is None:
        target_memory = 1024 * 1024 * 1024  # 1GB
        chunk_size = int(np.sqrt(target_memory // 8))
        chunk_size = min(chunk_size, N // 4, 50000)
        chunk_size = max(chunk_size, k + 1, 100)
    
    logging.info(f"Using chunk_size={chunk_size} for {N} samples")
    
    # Handle small datasets
    if N <= chunk_size * 2:
        logging.info("Dataset small enough for standard algorithm")
        return compute_mknn_simple(Z1, Z2, k)
    
    # Initialize top-k trackers
    tracker1 = TopKTracker(N, k, chunk_size)
    tracker2 = TopKTracker(N, k, chunk_size)
    
    # Process chunks
    n_chunks = (N + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size  
        end_idx = min(start_idx + chunk_size, N)
        chunk_indices = np.arange(start_idx, end_idx)
        
        # Extract chunks
        chunk1 = Z1[start_idx:end_idx]
        chunk2 = Z2[start_idx:end_idx]
        
        # Compute distances from each sample to this chunk
        for sample_idx in range(N):
            # Sample embeddings
            sample1 = Z1[sample_idx:sample_idx+1]
            sample2 = Z2[sample_idx:sample_idx+1]
            
            # Compute cosine distances to chunk  
            dist1 = cosine_distances(sample1, chunk1).flatten()
            dist2 = cosine_distances(sample2, chunk2).flatten()
            
            # Update trackers
            tracker1.update_neighbors([sample_idx], dist1.reshape(1, -1), chunk_idx, start_idx)
            tracker2.update_neighbors([sample_idx], dist2.reshape(1, -1), chunk_idx, start_idx)
        
        # Memory cleanup
        del chunk1, chunk2
        gc.collect()
        
        if chunk_idx % 10 == 0:
            logging.info(f"Processed chunk {chunk_idx+1}/{n_chunks}")
    
    # Compute final overlaps
    overlaps = []
    for i in range(N):
        neighbors1 = set(tracker1.get_neighbor_indices(i)[:k])
        neighbors2 = set(tracker2.get_neighbor_indices(i)[:k]) 
        overlap = len(neighbors1 & neighbors2)
        overlaps.append(overlap)
    
    score = np.mean(overlaps) / k
    logging.info(f"Memory-efficient simple MKNN complete. Score: {score:.4f}")
    return score


def mknn_score_auto(Z1: np.ndarray, Z2: np.ndarray, k: int = 10, **kwargs) -> float:
    """
    Automatically choose between standard and memory-efficient MKNN based on data size.
    
    Args:
        Z1, Z2: Embedding arrays
        k: Number of neighbors  
        **kwargs: Additional arguments passed to chosen implementation
        
    Returns:
        float: MKNN score
    """
    N = Z1.shape[0]
    embedding_dim = Z1.shape[1]
    
    # Estimate memory usage: 2 * N * embedding_dim * 8 bytes for arrays
    # Plus distance matrix of N² * 8 bytes  
    estimated_memory_gb = (2 * N * embedding_dim * 8 + N * N * 8) / (1024**3)
    
    # Use memory-efficient version for datasets > 2GB estimated memory
    if estimated_memory_gb > 2.0:
        logging.info(f"Large dataset detected ({estimated_memory_gb:.1f}GB estimated), using memory-efficient MKNN")
        return mknn_score_memory_efficient(Z1, Z2, k, **kwargs)
    else:
        logging.info(f"Small dataset detected ({estimated_memory_gb:.1f}GB estimated), using standard MKNN") 
        return mknn_score(Z1, Z2, k, **kwargs)


def compute_mknn_simple_auto(Z1: np.ndarray, Z2: np.ndarray, k: int = 10, **kwargs) -> float:
    """
    Automatically choose between standard and memory-efficient simple MKNN based on data size.
    """
    N = len(Z1)
    embedding_dim = Z1.shape[1] if hasattr(Z1, 'shape') else len(Z1[0])
    estimated_memory_gb = (2 * N * embedding_dim * 8 + N * N * 8) / (1024**3)
    
    if estimated_memory_gb > 2.0:
        logging.info(f"Large dataset detected ({estimated_memory_gb:.1f}GB estimated), using memory-efficient simple MKNN")
        return compute_mknn_simple_memory_efficient(Z1, Z2, k, **kwargs)
    else:
        logging.info(f"Small dataset detected ({estimated_memory_gb:.1f}GB estimated), using standard simple MKNN")
        return compute_mknn_simple(Z1, Z2, k)