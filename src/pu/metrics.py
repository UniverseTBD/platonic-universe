import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, pearsonr, wasserstein_distance
from typing import Any, Dict, List, Tuple, Optional
import polars as pl


def mknn(Z1: np.ndarray, Z2: np.ndarray, k: int = 10) -> float:
    """
    Calculate mutual k nearest neighbours overlap.
    
    For each sample, finds its k nearest neighbors in both embedding spaces
    and computes the average overlap ratio.
    
    Args:
        Z1: (n_samples, d1) array of embeddings
        Z2: (n_samples, d2) array of embeddings
        k: Number of nearest neighbors
        
    Returns:
        float in [0, 1] where 1 = perfect overlap, 0 = no overlap
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

    return float(np.mean(overlap) / k)


def jaccard_index(Z1: np.ndarray, Z2: np.ndarray, k: int = 10) -> float:
    """
    Calculate Jaccard index of k nearest neighbours.

    Args:
        Z1: (n_samples, d1) array of embeddings
        Z2: (n_samples, d2) array of embeddings
        k: Number of nearest neighbors
        
    Returns:
        float in [0, 1] where 1 = identical neighbors, 0 = no overlap
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

    jaccard = [
        len(set(a).intersection(b)) / len(set(a).union(b)) for a, b in zip(nn1, nn2)
    ]

    return float(np.mean(jaccard))


def linear_cka(Z1: np.ndarray, Z2: np.ndarray) -> float:
    """
    Linear Centered Kernel Alignment (CKA) between two embedding matrices.
    
    CKA measures similarity between representations by comparing their
    centered Gram matrices. It's invariant to orthogonal transformations
    and isotropic scaling.
    
    Args:
        Z1: (n_samples, d1) array of embeddings
        Z2: (n_samples, d2) array of embeddings
    
    Returns:
        float in [0, 1] where 1 = perfect alignment, 0 = no alignment
    
    Reference:
        Kornblith et al. (2019) "Similarity of Neural Network Representations Revisited"
    """
    assert len(Z1) == len(Z2)
    
    # Compute Gram matrices (kernel matrices)
    K1 = Z1 @ Z1.T  # (n, n)
    K2 = Z2 @ Z2.T  # (n, n)
    
    # Center the Gram matrices
    n = len(Z1)
    H = np.eye(n) - np.ones((n, n)) / n
    K1 = H @ K1 @ H
    K2 = H @ K2 @ H
    
    # Correct Linear CKA formula: trace(K1 @ K2) / sqrt(trace(K1 @ K1) * trace(K2 @ K2))
    # This is equivalent to ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F) for centered matrices
    numerator = np.trace(K1 @ K2)
    denominator = np.sqrt(np.trace(K1 @ K1) * np.trace(K2 @ K2))
    
    if denominator < 1e-12:
        return 0.0
    
    return float(numerator / denominator)


def rsm_correlation(
    Z1: np.ndarray, 
    Z2: np.ndarray, 
    method: str = 'spearman', 
    metric: str = 'cosine'
) -> float:
    """
    Representational Similarity Matrix (RSM) correlation.
    
    Computes pairwise distances between samples in each embedding space,
    then correlates these distance matrices.
    
    Args:
        Z1: (n_samples, d1) array of embeddings
        Z2: (n_samples, d2) array of embeddings
        method: 'spearman' (rank correlation) or 'pearson'
        metric: distance metric - 'cosine', 'euclidean', 'correlation', etc.
    
    Returns:
        float correlation coefficient in [-1, 1]
    
    Reference:
        Kriegeskorte et al. (2008) "Representational similarity analysis"
    """
    assert len(Z1) == len(Z2)
    
    # Compute pairwise distances (upper triangular, excluding diagonal)
    # pdist returns condensed distance matrix (1D array of upper triangle)
    dist1 = pdist(Z1, metric=metric)
    dist2 = pdist(Z2, metric=metric)
    
    # Compute correlation between the two distance vectors
    if method == 'spearman':
        corr, _ = spearmanr(dist1, dist2)
    elif method == 'pearson':
        corr, _ = pearsonr(dist1, dist2)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'spearman' or 'pearson'")
    
    return float(corr)


def wass_distance(
    Z1: np.ndarray,
    Z2: np.ndarray,
    k: int = 5,
    params: Optional[Dict[str, np.ndarray]] = None
) -> Dict[str, float]:
    """
    Compute median Wasserstein distance of physical parameters in k-NN neighborhoods.
    
    For each sample, finds k nearest neighbors in both embedding spaces,
    then computes the Wasserstein distance between the parameter distributions
    of these neighbor sets. Returns the median distance across all samples.
    
    This measures how much the local neighborhood structure differs between
    two representations in terms of physical properties.
    
    Args:
        Z1: (n_samples, d1) embeddings from first representation
        Z2: (n_samples, d2) embeddings from second representation
        k: Number of nearest neighbors
        params: Dict mapping parameter names to (n_samples,) arrays
        
    Returns:
        Dict mapping parameter names to median Wasserstein distances
        
    Example:
        >>> params = {'redshift': z_values, 'stellar_mass': mass_values}
        >>> distances = wass_distance(emb1, emb2, k=10, params=params)
        >>> print(distances['redshift'])  # How much redshift distributions differ
    """
    assert len(Z1) == len(Z2)
    
    if params is None:
        return {}

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

    n = Z1.shape[0]
    w_ds = {param: np.zeros(n) for param in params.keys()}
        
    for i in range(n):
        idxs1 = nn1[i, :]
        idxs2 = nn2[i, :]
        for param in params.keys():
            w_ds[param][i] = wasserstein_distance(params[param][idxs1], params[param][idxs2])

    return {param: float(np.median(w_ds[param])) for param in params.keys()}


def _get_available_sizes(parquet_file: str) -> List[str]:
    """Get all available sizes for a given parquet file."""
    df = pl.read_parquet(parquet_file)
    columns = df.columns
    
    if not columns:
        return []
    
    # Extract model from first column
    example_col = columns[0]
    parts = example_col.split("_")
    if len(parts) < 3:
        raise ValueError("Column names must be in the format <model>_<size>_<mode>")
    
    model = parts[0]
    
    # Find all unique sizes
    sizes = set()
    for col in columns:
        if col.startswith(f"{model}_"):
            parts = col.split("_")
            if len(parts) >= 3:
                sizes.add(parts[1])
    
    return sorted(list(sizes))


def _load_pair_from_parquet(
    parquet_file: str, 
    size: str = None
) -> Tuple[np.ndarray, np.ndarray, Tuple[str, str], Tuple[str, str]]:
    """
    Load two embedding columns from a parquet file.

    Args:
        parquet_file: Path to parquet file
        size: Optional size to use. If None, uses the size from the first column.
    
    Returns:
        (arr1, arr2, (model, size), (mode1, mode2))
    """
    # Load the parquet file
    df = pl.read_parquet(parquet_file)
    columns = df.columns

    # Extract model, size, and modes from column names
    example_col = columns[0]
    parts = example_col.split("_")
    if len(parts) < 3:
        raise ValueError("Column names must be in the format <model>_<size>_<mode>")

    model = parts[0]
    
    # Use provided size or default to first column's size
    if size is None:
        size = parts[1]
    else:
        # Verify the size exists
        available_sizes = _get_available_sizes(parquet_file)
        if size not in available_sizes:
            raise ValueError(f"Size '{size}' not found. Available sizes: {available_sizes}")
    
    modes = [col.split(f"{model}_{size}_")[1] for col in columns if col.startswith(f"{model}_{size}_")]
    if len(modes) != 2:
        raise ValueError(f"Expected exactly two modes for size '{size}'. Found: {modes}")

    mode1, mode2 = modes

    # Convert the embeddings to numpy arrays
    embs1 = df[f"{model}_{size}_{mode1}"].to_numpy()
    embs2 = df[f"{model}_{size}_{mode2}"].to_numpy()

    # Ensure we pass a sequence of arrays to np.vstack (convert ndarray-of-objects to a list)
    arr1 = np.vstack(list(embs1))
    arr2 = np.vstack(list(embs2))

    return arr1, arr2, (model, size), (mode1, mode2)


def run_comparisons(
    parquet_file: str, 
    metrics: List[str], 
    k: int = 10, 
    size: str = None
) -> Dict[str, Any]:
    """
    Compute one or more metrics between embedding columns in a parquet file.

    Args:
        parquet_file: Path to parquet file
        metrics: List of metrics to compute
        k: K value for k-NN based metrics
        size: Optional size to use. If None, uses first. If "all", processes all sizes.
    
    Supported metrics: 'mknn', 'jaccard', 'cka', 'rsm', 'all'.
    """
    if size == "all":
        available_sizes = _get_available_sizes(parquet_file)
        results = []
        for s in available_sizes:
            result = run_comparisons(parquet_file, metrics, k, size=s)
            results.append(result)
        return results

    # Load the embeddings from the parquet file
    arr1, arr2, (model, size), (mode1, mode2) = _load_pair_from_parquet(parquet_file, size=size)

    # If the user wants to run all metrics, set the metrics to the default list
    if len(metrics) == 1 and metrics[0].lower() == "all":
        metrics = ["mknn", "jaccard", "cka", "rsm"] #TODO: Add more metrics

    # Create a dictionary to store the results
    results = {
        "model": model,
        "size": size,
        "modes": [mode1, mode2],
    }

    for metric in metrics:
        if metric == "mknn":
            results[f"mknn_k{k}"] = mknn(arr1, arr2, k=k)
        elif metric == "jaccard":
            results[f"jaccard_k{k}"] = jaccard_index(arr1, arr2, k=k)
        elif metric == "cka":
            results["cka"] = linear_cka(arr1, arr2)
        elif metric == "rsm":
            results["rsm"] = rsm_correlation(arr1, arr2)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return results


def run_mknn_comparison(parquet_file: str) -> Dict[str, Any]:
    """
    Load embeddings from a Parquet file and compute the mknn metric.

    Args:
        parquet_file: Path to the Parquet file containing embeddings.
        
    Returns:
        Dictionary containing the mknn score and embeddings.
    """
    arr1, arr2, (model, size), (mode1, mode2) = _load_pair_from_parquet(parquet_file)
    mknn_score = mknn(arr1, arr2, k=10)
    return {"mknn_score": mknn_score, "embeddings": {mode1: arr1, mode2: arr2}}


def compute_cka_mmap(file1: str, file2: str, n: int, m: int) -> float:
    """
    Compute CKA between two memory-mapped matrices stored in files.

    Args:
        file1: Path to the first memory-mapped file.
        file2: Path to the second memory-mapped file.
        n: Number of samples (rows).
        m: Number of features (columns).

    Returns:
        The CKA similarity score between the two matrices.
    """
    from pu_cka import compute_cka
    return compute_cka(str(file1), str(file2), int(n), int(m))
