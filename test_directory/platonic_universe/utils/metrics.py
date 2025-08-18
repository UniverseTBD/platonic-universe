"""
Additional metrics and similarity functions.
"""

from typing import Optional, Tuple, Union
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def normalize_embeddings(embeddings: np.ndarray, method: str = "l2") -> np.ndarray:
    """
    Normalize embeddings using specified method.
    
    Args:
        embeddings: Input embeddings of shape (n_samples, n_features)
        method: Normalization method ('l2', 'l1', 'standard', or 'none')
        
    Returns:
        Normalized embeddings
    """
    if method == "l2":
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-8)
    elif method == "l1":
        norms = np.sum(np.abs(embeddings), axis=1, keepdims=True) 
        return embeddings / (norms + 1e-8)
    elif method == "standard":
        # Z-score normalization
        mean = np.mean(embeddings, axis=0, keepdims=True)
        std = np.std(embeddings, axis=0, keepdims=True)
        return (embeddings - mean) / (std + 1e-8)
    elif method == "none":
        return embeddings
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def cosine_similarity_matrix(embeddings_a: np.ndarray, 
                           embeddings_b: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute cosine similarity matrix between embeddings.
    
    Args:
        embeddings_a: First set of embeddings
        embeddings_b: Second set of embeddings (if None, use embeddings_a)
        
    Returns:
        Cosine similarity matrix
    """
    if embeddings_b is None:
        embeddings_b = embeddings_a
    
    return cosine_similarity(embeddings_a, embeddings_b)


def euclidean_similarity_matrix(embeddings_a: np.ndarray,
                              embeddings_b: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute Euclidean distance matrix between embeddings.
    
    Args:
        embeddings_a: First set of embeddings
        embeddings_b: Second set of embeddings (if None, use embeddings_a)
        
    Returns:
        Euclidean distance matrix
    """
    if embeddings_b is None:
        embeddings_b = embeddings_a
    
    return euclidean_distances(embeddings_a, embeddings_b)


def compute_alignment_score(embeddings_a: np.ndarray,
                          embeddings_b: np.ndarray,
                          method: str = "cosine_mean") -> float:
    """
    Compute alignment score between two embedding sets.
    
    Args:
        embeddings_a: First embedding set
        embeddings_b: Second embedding set
        method: Alignment method ('cosine_mean', 'cosine_max', 'euclidean_mean')
        
    Returns:
        Alignment score
    """
    assert embeddings_a.shape[0] == embeddings_b.shape[0], "Must have same number of samples"
    
    if method == "cosine_mean":
        # Mean cosine similarity between corresponding pairs
        similarities = []
        for i in range(len(embeddings_a)):
            sim = cosine_similarity([embeddings_a[i]], [embeddings_b[i]])[0, 0]
            similarities.append(sim)
        return float(np.mean(similarities))
    
    elif method == "cosine_max":
        # Maximum cosine similarity for each point in A to any point in B
        sim_matrix = cosine_similarity(embeddings_a, embeddings_b)
        return float(np.mean(np.max(sim_matrix, axis=1)))
    
    elif method == "euclidean_mean":
        # Mean Euclidean distance between corresponding pairs (lower is better)
        distances = []
        for i in range(len(embeddings_a)):
            dist = np.linalg.norm(embeddings_a[i] - embeddings_b[i])
            distances.append(dist)
        return float(np.mean(distances))
    
    else:
        raise ValueError(f"Unknown alignment method: {method}")


def project_to_common_space(embeddings_a: np.ndarray,
                          embeddings_b: np.ndarray,
                          method: str = "pca",
                          target_dim: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project embeddings to common dimensionality space.
    
    Args:
        embeddings_a: First embedding set
        embeddings_b: Second embedding set  
        method: Projection method ('pca', 'truncate', 'pad')
        target_dim: Target dimensionality (if None, use minimum of input dims)
        
    Returns:
        Tuple of projected embeddings
    """
    dim_a, dim_b = embeddings_a.shape[1], embeddings_b.shape[1]
    
    if target_dim is None:
        target_dim = min(dim_a, dim_b)
    
    if method == "pca":
        # Apply PCA to each set independently
        proj_a = embeddings_a
        if dim_a > target_dim:
            pca_a = PCA(n_components=target_dim, random_state=42)
            proj_a = pca_a.fit_transform(embeddings_a)
        
        proj_b = embeddings_b
        if dim_b > target_dim:
            pca_b = PCA(n_components=target_dim, random_state=42)
            proj_b = pca_b.fit_transform(embeddings_b)
        
        return proj_a, proj_b
    
    elif method == "truncate":
        # Simply truncate to target dimensions
        proj_a = embeddings_a[:, :target_dim]
        proj_b = embeddings_b[:, :target_dim]
        return proj_a, proj_b
    
    elif method == "pad":
        # Pad smaller embedding to target dimensions
        proj_a = embeddings_a
        if dim_a < target_dim:
            padding = np.zeros((len(embeddings_a), target_dim - dim_a))
            proj_a = np.concatenate([embeddings_a, padding], axis=1)
        elif dim_a > target_dim:
            proj_a = embeddings_a[:, :target_dim]
        
        proj_b = embeddings_b
        if dim_b < target_dim:
            padding = np.zeros((len(embeddings_b), target_dim - dim_b))
            proj_b = np.concatenate([embeddings_b, padding], axis=1)
        elif dim_b > target_dim:
            proj_b = embeddings_b[:, :target_dim]
        
        return proj_a, proj_b
    
    else:
        raise ValueError(f"Unknown projection method: {method}")


def compute_embedding_quality_metrics(embeddings: np.ndarray) -> dict:
    """
    Compute quality metrics for embeddings.
    
    Args:
        embeddings: Input embeddings
        
    Returns:
        Dictionary with quality metrics
    """
    # Clean embeddings
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=1.0, neginf=-1.0)
    
    metrics = {}
    
    # Basic statistics
    metrics["shape"] = embeddings.shape
    metrics["mean_norm"] = float(np.mean(np.linalg.norm(embeddings, axis=1)))
    metrics["std_norm"] = float(np.std(np.linalg.norm(embeddings, axis=1)))
    
    # Sparsity
    metrics["sparsity"] = float(np.mean(embeddings == 0))
    
    # Numerical stability
    metrics["has_nan"] = bool(np.any(np.isnan(embeddings)))
    metrics["has_inf"] = bool(np.any(np.isinf(embeddings)))
    metrics["finite_ratio"] = float(np.mean(np.isfinite(embeddings)))
    
    # Distribution properties
    metrics["mean_activation"] = float(np.mean(embeddings))
    metrics["std_activation"] = float(np.std(embeddings))
    metrics["min_activation"] = float(np.min(embeddings))
    metrics["max_activation"] = float(np.max(embeddings))
    
    # Dimensionality analysis
    try:
        # Effective rank (numerical rank)
        U, s, Vt = np.linalg.svd(embeddings, full_matrices=False)
        normalized_s = s / s[0] if s[0] > 0 else s
        effective_rank = np.sum(normalized_s > 0.01)  # Threshold for "significant" singular values
        metrics["effective_rank"] = int(effective_rank)
        metrics["rank_ratio"] = float(effective_rank / min(embeddings.shape))
    except Exception:
        metrics["effective_rank"] = None
        metrics["rank_ratio"] = None
    
    # Pairwise similarity statistics
    try:
        if len(embeddings) > 1 and len(embeddings) <= 5000:  # Avoid memory issues
            # Sample if too large
            if len(embeddings) > 1000:
                indices = np.random.choice(len(embeddings), 1000, replace=False)
                sample_embeddings = embeddings[indices]
            else:
                sample_embeddings = embeddings
            
            # Cosine similarities
            cos_sim = cosine_similarity(sample_embeddings)
            # Remove diagonal (self-similarities)
            cos_sim_off_diag = cos_sim[~np.eye(cos_sim.shape[0], dtype=bool)]
            
            metrics["mean_cosine_similarity"] = float(np.mean(cos_sim_off_diag))
            metrics["std_cosine_similarity"] = float(np.std(cos_sim_off_diag))
            
    except Exception:
        metrics["mean_cosine_similarity"] = None
        metrics["std_cosine_similarity"] = None
    
    return metrics


def compare_embedding_quality(embeddings_dict: dict) -> dict:
    """
    Compare quality metrics across multiple embedding sets.
    
    Args:
        embeddings_dict: Dictionary mapping names to embedding arrays
        
    Returns:
        Dictionary with comparison results
    """
    results = {"individual_metrics": {}, "comparison": {}}
    
    # Compute metrics for each embedding set
    for name, embeddings in embeddings_dict.items():
        results["individual_metrics"][name] = compute_embedding_quality_metrics(embeddings)
    
    # Compare across sets
    if len(embeddings_dict) > 1:
        metric_names = list(results["individual_metrics"].values())[0].keys()
        
        for metric_name in metric_names:
            values = []
            valid_names = []
            
            for name in embeddings_dict.keys():
                value = results["individual_metrics"][name].get(metric_name)
                if value is not None and not (isinstance(value, float) and np.isnan(value)):
                    values.append(value)
                    valid_names.append(name)
            
            if len(values) > 1:
                if isinstance(values[0], (int, float)):
                    # Numerical comparison
                    best_idx = np.argmax(values) if "sparsity" not in metric_name else np.argmin(values)
                    results["comparison"][metric_name] = {
                        "values": dict(zip(valid_names, values)),
                        "best": valid_names[best_idx],
                        "best_value": values[best_idx],
                        "range": [min(values), max(values)]
                    }
    
    return results