"""
k-Nearest Neighbor metrics for cross-modal analysis.
"""

from typing import Optional, Union, List, Tuple
import numpy as np
from sklearn.neighbors import NearestNeighbors


def normalize_embeddings(embeddings: np.ndarray, method: str = "l2") -> np.ndarray:
    """
    Normalize embeddings.
    
    Args:
        embeddings: Input embeddings of shape (n_samples, n_features)
        method: Normalization method ('l2', 'l1', or 'none')
        
    Returns:
        Normalized embeddings
    """
    if method == "l2":
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-8)
    elif method == "l1":
        norms = np.sum(np.abs(embeddings), axis=1, keepdims=True)
        return embeddings / (norms + 1e-8)
    elif method == "none":
        return embeddings
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_mknn_prh(embeddings_a: np.ndarray, 
                    embeddings_b: np.ndarray,
                    k: int = 10,
                    metric: str = "cosine",
                    normalize: bool = True) -> float:
    """
    Compute mutual k-NN score according to PRH Equation 11.
    
    This implements the exact PRH formulation:
    - Find k-nearest neighbors in A for each point in A (excluding self)
    - Find k-nearest neighbors in B for each point in B (excluding self)  
    - Score = mean over all points of |N_A(i) ∩ N_B(i)| / k
    
    Args:
        embeddings_a: First set of embeddings, shape (n, d_a)
        embeddings_b: Second set of embeddings, shape (n, d_b)
        k: Number of neighbors
        metric: Distance metric ('cosine', 'euclidean', 'manhattan')
        normalize: Whether to L2-normalize embeddings (recommended for cosine)
        
    Returns:
        Mutual k-NN overlap score between 0 and 1
    """
    assert embeddings_a.shape[0] == embeddings_b.shape[0], "Row-aligned inputs required"
    n = embeddings_a.shape[0]
    
    if n < 2:
        return 0.0
    
    k = min(k, n - 1)  # Can't have more neighbors than available
    if k <= 0:
        return 0.0
    
    # Normalize if requested (typically for cosine similarity)
    if normalize:
        embeddings_a = normalize_embeddings(embeddings_a, "l2")
        embeddings_b = normalize_embeddings(embeddings_b, "l2")
    
    # Build k-NN models
    nn_a = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(embeddings_a)
    nn_b = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(embeddings_b)
    
    # Find neighbors (excluding self, hence [:, 1:])
    indices_a = nn_a.kneighbors(embeddings_a, return_distance=False)[:, 1:]
    indices_b = nn_b.kneighbors(embeddings_b, return_distance=False)[:, 1:]
    
    # Compute overlap for each point
    overlaps = []
    for i in range(n):
        neighbors_a = set(indices_a[i])
        neighbors_b = set(indices_b[i])
        overlap_size = len(neighbors_a.intersection(neighbors_b))
        overlaps.append(overlap_size / k)
    
    return float(np.mean(overlaps))


def compute_mknn_score(embeddings_a: np.ndarray,
                      embeddings_b: np.ndarray, 
                      k: int = 10,
                      batch_size: Optional[int] = None,
                      repeats: int = 1,
                      random_seed: int = 42) -> float:
    """
    Compute mutual k-NN score with optional batching and averaging.
    
    Args:
        embeddings_a: First set of embeddings
        embeddings_b: Second set of embeddings
        k: Number of neighbors
        batch_size: If specified, randomly sample this many points for multiple runs
        repeats: Number of random samples to average (only used if batch_size is set)
        random_seed: Random seed for reproducibility
        
    Returns:
        Average mutual k-NN score
    """
    if batch_size is None or batch_size >= len(embeddings_a):
        # Use all data
        return compute_mknn_prh(embeddings_a, embeddings_b, k)
    
    # Batched evaluation with averaging
    rng = np.random.default_rng(random_seed)
    scores = []
    
    for _ in range(repeats):
        # Random sample of indices
        indices = rng.choice(len(embeddings_a), size=batch_size, replace=False)
        
        # Compute score on subset
        score = compute_mknn_prh(
            embeddings_a[indices], 
            embeddings_b[indices], 
            k
        )
        scores.append(score)
    
    return float(np.mean(scores))


def mutual_knn_overlap(embeddings_a: np.ndarray,
                      embeddings_b: np.ndarray,
                      k_values: List[int] = [5, 10, 20, 50],
                      **kwargs) -> dict:
    """
    Compute mutual k-NN overlap for multiple k values.
    
    Args:
        embeddings_a: First set of embeddings
        embeddings_b: Second set of embeddings
        k_values: List of k values to evaluate
        **kwargs: Additional arguments passed to compute_mknn_score
        
    Returns:
        Dictionary mapping k values to scores
    """
    results = {}
    
    for k in k_values:
        if k < len(embeddings_a):
            score = compute_mknn_score(embeddings_a, embeddings_b, k, **kwargs)
            results[k] = score
        else:
            results[k] = 0.0  # Not enough samples
    
    return results


class KNNAnalyzer:
    """Analyzer for k-nearest neighbor metrics."""
    
    def __init__(self, 
                 metric: str = "cosine",
                 normalize: bool = True,
                 random_seed: int = 42):
        """
        Initialize KNN analyzer.
        
        Args:
            metric: Distance metric for k-NN
            normalize: Whether to normalize embeddings
            random_seed: Random seed for reproducibility
        """
        self.metric = metric
        self.normalize = normalize
        self.random_seed = random_seed
        
    def compute_mutual_knn(self, 
                          embeddings_a: np.ndarray,
                          embeddings_b: np.ndarray,
                          k: int = 10,
                          **kwargs) -> float:
        """
        Compute mutual k-NN score.
        
        Args:
            embeddings_a: First embedding set
            embeddings_b: Second embedding set  
            k: Number of neighbors
            **kwargs: Additional arguments
            
        Returns:
            Mutual k-NN score
        """
        return compute_mknn_prh(
            embeddings_a, 
            embeddings_b, 
            k, 
            metric=self.metric,
            normalize=self.normalize
        )
    
    def compute_cross_modal_alignment(self,
                                    image_embeddings: np.ndarray,
                                    spectrum_embeddings: np.ndarray,
                                    k_values: List[int] = [5, 10, 20, 50],
                                    batch_size: Optional[int] = None,
                                    repeats: int = 3) -> dict:
        """
        Compute cross-modal alignment between images and spectra.
        
        Args:
            image_embeddings: Image embeddings
            spectrum_embeddings: Spectrum embeddings
            k_values: List of k values to evaluate
            batch_size: Batch size for evaluation
            repeats: Number of repeats for batched evaluation
            
        Returns:
            Dictionary with alignment scores
        """
        results = {
            "k_values": k_values,
            "scores": {},
            "metrics": {
                "metric": self.metric,
                "normalize": self.normalize,
                "batch_size": batch_size,
                "repeats": repeats,
                "n_samples": len(image_embeddings)
            }
        }
        
        for k in k_values:
            score = compute_mknn_score(
                image_embeddings,
                spectrum_embeddings,
                k=k,
                batch_size=batch_size,
                repeats=repeats,
                random_seed=self.random_seed
            )
            results["scores"][k] = score
        
        # Add summary statistics
        scores = list(results["scores"].values())
        results["summary"] = {
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "max_score": np.max(scores),
            "min_score": np.min(scores),
            "best_k": k_values[np.argmax(scores)]
        }
        
        return results
    
    def compare_models(self,
                      embeddings_dict: dict,
                      reference_key: str,
                      k: int = 10,
                      **kwargs) -> dict:
        """
        Compare multiple models against a reference.
        
        Args:
            embeddings_dict: Dictionary mapping model names to embeddings
            reference_key: Key for reference embeddings
            k: Number of neighbors
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with comparison results
        """
        if reference_key not in embeddings_dict:
            raise ValueError(f"Reference key '{reference_key}' not found in embeddings")
        
        reference_embeddings = embeddings_dict[reference_key]
        results = {"reference": reference_key, "k": k, "scores": {}}
        
        for model_name, embeddings in embeddings_dict.items():
            if model_name != reference_key:
                score = self.compute_mutual_knn(
                    reference_embeddings,
                    embeddings, 
                    k=k,
                    **kwargs
                )
                results["scores"][model_name] = score
        
        # Rank models by score
        sorted_scores = sorted(results["scores"].items(), key=lambda x: x[1], reverse=True)
        results["ranking"] = [{"model": name, "score": score} for name, score in sorted_scores]
        
        return results


def analyze_embedding_alignment(embedding_sets: dict,
                              k_values: List[int] = [5, 10, 20, 50],
                              pairwise: bool = True) -> dict:
    """
    Analyze alignment between multiple embedding sets.
    
    Args:
        embedding_sets: Dictionary mapping names to embedding arrays
        k_values: List of k values to evaluate
        pairwise: Whether to compute all pairwise comparisons
        
    Returns:
        Dictionary with alignment analysis results
    """
    analyzer = KNNAnalyzer()
    results = {
        "embedding_sets": list(embedding_sets.keys()),
        "k_values": k_values,
        "pairwise_scores": {} if pairwise else None,
        "summary": {}
    }
    
    if pairwise:
        # Compute all pairwise alignments
        set_names = list(embedding_sets.keys())
        for i, name_a in enumerate(set_names):
            for j, name_b in enumerate(set_names):
                if i < j:  # Avoid duplicates
                    pair_key = f"{name_a}_vs_{name_b}"
                    pair_scores = {}
                    
                    for k in k_values:
                        score = analyzer.compute_mutual_knn(
                            embedding_sets[name_a],
                            embedding_sets[name_b],
                            k=k
                        )
                        pair_scores[k] = score
                    
                    results["pairwise_scores"][pair_key] = pair_scores
    
    # Compute summary statistics
    if pairwise and results["pairwise_scores"]:
        all_scores = []
        for pair_scores in results["pairwise_scores"].values():
            all_scores.extend(pair_scores.values())
        
        results["summary"] = {
            "mean_alignment": np.mean(all_scores),
            "std_alignment": np.std(all_scores),
            "max_alignment": np.max(all_scores),
            "min_alignment": np.min(all_scores)
        }
    
    return results