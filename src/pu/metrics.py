import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, pearsonr
from scipy.linalg import orthogonal_procrustes
from typing import Any, Dict, List, Tuple
import polars as pl


def mknn(Z1, Z2, k=10):
    """
    Calculate mutual k nearest neighbours
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


def jaccard_index(Z1, Z2, k=10):
    """
    Calculate Jaccard index of k nearest neighbours

    Gives a value between 0 and 1, where 1 means the k nearest neighbours are identical.
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

def linear_cka(Z1, Z2):
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
        Kornblith et al. (2019) "Similarity of Neural Network Representations 
        Revisited" (ICML)
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
    
    cka_value = numerator / denominator
    return float(cka_value)

def rsm_correlation(Z1, Z2, method='spearman', metric='cosine'):
    """
    Representational Similarity Matrix (RSM) correlation.
    
    Computes pairwise distances between samples in each embedding space,
    then correlates these distance matrices. This measures whether the
    two embedding spaces preserve similar relational structure.
    
    Args:
        Z1: (n_samples, d1) array of embeddings
        Z2: (n_samples, d2) array of embeddings
        method: 'spearman' (rank correlation, more robust) or 'pearson'
        metric: distance metric - 'cosine', 'euclidean', 'correlation', etc.
    
    Returns:
        float correlation coefficient in [-1, 1]
        1 = perfect agreement, -1 = perfect disagreement, 0 = no correlation
    
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


def _load_pair_from_parquet(parquet_file: str, size: str = None) -> Tuple[np.ndarray, np.ndarray, Tuple[str, str], Tuple[str, str]]:
    """Helper to load two embedding columns from a parquet file.

    Args:
        parquet_file: Path to parquet file
        size: Optional size to use. If None, uses the size from the first column.
    
    Returns (arr1, arr2, (model,size), (mode1, mode2))
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


def run_comparisons(parquet_file: str, metrics: List[str], k: int = 10, size: str = None) -> Dict[str, Any]:
    """Compute one or more metrics between the two embedding columns in a parquet file.

    Args:
        parquet_file: Path to parquet file
        metrics: List of metrics to compute
        k: K value for k-NN based metrics
        size: Optional size to use. If None, uses the size from the first column.
              If "all", processes all available sizes and returns a list of results.
    
    Supported metrics: 'mknn', 'jaccard', 'cka', 'rsm', 'all'.
    """

    # If processing all sizes
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

    # Run the metrics
    for metric in metrics:
        try:
            if metric == "mknn":
                metric_name = "mknn_k{k}"
                results[metric_name] = mknn(arr1, arr2, k=k)
            elif metric == "jaccard":
                metric_name = "jaccard_k{k}"
                results[metric] = jaccard_index(arr1, arr2, k=k)
            elif metric == "cka":
                metric_name = "cka"
                results[metric_name] = linear_cka(arr1, arr2)
            elif metric == "rsm":
                metric_name = "rsm"
                results[metric_name] = rsm_correlation(arr1, arr2)
            else:
                raise ValueError(f"Unknown metric: {metric}")
        except ValueError as e:
            raise

    return results

def run_mknn_comparison(parquet_file: str) -> Dict[str, Any]:
    """
    Load embeddings from a Parquet file and compute the mknn metric between
    two sets of embeddings.

    Assumes the Parquet file has columns named in the format:
    - <model>_<size>_<mode1>
    - <model>_<size>_<mode2>

    where <mode1> and <mode2> are the two datasets to compare (e.g., 'hsc' and 'jwst').

    Args:
        parquet_file (str): Path to the Parquet file containing embeddings.
    Returns:
        Dict[str, Any]: Dictionary containing the mknn score and raw embeddings.
    """

    # Load the embeddings from the parquet file
    arr1, arr2, (model, size), (mode1, mode2) = _load_pair_from_parquet(parquet_file)

    mknn_score = mknn(arr1, arr2, k=10) # Default k=10

    return {"mknn_score": mknn_score, "embeddings": {mode1: embs1, mode2: embs2}}

def compute_cka_mmap(file1: str, file2: str, n: int, m: int) -> float:
    """
    Compute CKA between two memory-mapped matrices stored in files.

    Args:
        file1 (str): Path to the first memory-mapped file.
        file2 (str): Path to the second memory-mapped file.
        n (int): Number of samples (rows).
        m (int): Number of features (columns).

    Returns:
        float: The CKA similarity score between the two matrices.
    """
    from pu_cka import compute_cka

    cka_score = compute_cka(str(file1), str(file2), int(n), int(m))
    return cka_score