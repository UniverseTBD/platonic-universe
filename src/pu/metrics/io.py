"""
I/O utilities and batch comparison functions.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Any
import polars as pl

from pu.metrics.kernel import cka, mmd
from pu.metrics.geometric import procrustes, cosine_similarity, frechet
from pu.metrics.cca import svcca, pwcca
from pu.metrics.spectral import tucker_congruence, eigenspectrum, riemannian
from pu.metrics.information import kl_divergence, js_divergence, mutual_information
from pu.metrics.neighbors import mknn, jaccard, rsa
from pu.metrics.regression import linear_r2


# Registry of all available metrics
METRICS_REGISTRY = {
    # Kernel-based
    "cka": cka,
    "mmd": mmd,
    # Geometric
    "procrustes": procrustes,
    "cosine_similarity": cosine_similarity,
    "frechet": frechet,
    # CCA
    "svcca": svcca,
    "pwcca": pwcca,
    # Spectral
    "tucker_congruence": tucker_congruence,
    "eigenspectrum": eigenspectrum,
    "riemannian": riemannian,
    # Information-theoretic
    "kl_divergence": kl_divergence,
    "js_divergence": js_divergence,
    "mutual_information": mutual_information,
    # Neighbor-based
    "mknn": mknn,
    "jaccard": jaccard,
    "rsa": rsa,
    # Regression
    "linear_r2": linear_r2,
}


def list_metrics() -> list[str]:
    """
    List all available metric names.

    Returns:
        List of metric names that can be used with compare()
    """
    return list(METRICS_REGISTRY.keys())


def compare(
    Z1: NDArray[np.floating],
    Z2: NDArray[np.floating],
    metrics: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """
    Compute multiple metrics between two embedding matrices.

    Args:
        Z1: (n_samples, d1) embedding matrix
        Z2: (n_samples, d2) embedding matrix
        metrics: List of metric names to compute. If None or ["all"],
                 computes all available metrics.
        **kwargs: Additional keyword arguments passed to specific metrics.
                  Use metric_name__param format, e.g., mknn__k=10.

    Returns:
        Dictionary mapping metric names to their computed values.

    Example:
        >>> results = compare(Z1, Z2, metrics=["cka", "mknn"], mknn__k=5)
        >>> print(results)
        {"cka": 0.85, "mknn": 0.72}
    """
    if metrics is None or (len(metrics) == 1 and metrics[0].lower() == "all"):
        metrics = list(METRICS_REGISTRY.keys())

    results = {}

    for metric_name in metrics:
        if metric_name not in METRICS_REGISTRY:
            raise ValueError(
                f"Unknown metric: {metric_name}. "
                f"Available: {list(METRICS_REGISTRY.keys())}"
            )

        metric_fn = METRICS_REGISTRY[metric_name]

        # Extract metric-specific kwargs (format: metric__param)
        metric_kwargs = {}
        prefix = f"{metric_name}__"
        for key, value in kwargs.items():
            if key.startswith(prefix):
                param_name = key[len(prefix) :]
                metric_kwargs[param_name] = value

        try:
            results[metric_name] = metric_fn(Z1, Z2, **metric_kwargs)
        except Exception as e:
            # Some metrics may fail (e.g., dimension mismatch)
            results[metric_name] = None
            # Optionally log the error
            # print(f"Warning: {metric_name} failed: {e}")

    return results


def get_available_sizes(parquet_file: str) -> list[str]:
    """
    Get all available model sizes from a parquet file.

    Args:
        parquet_file: Path to parquet file

    Returns:
        Sorted list of available size strings
    """
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


def load_embeddings_from_parquet(
    parquet_file: str, size: str | None = None
) -> tuple[NDArray[np.floating], NDArray[np.floating], dict[str, str]]:
    """
    Load two embedding columns from a parquet file.

    Args:
        parquet_file: Path to parquet file
        size: Optional size to use. If None, uses the size from the first column.

    Returns:
        Tuple of (embeddings1, embeddings2, metadata) where metadata contains
        model, size, mode1, mode2.

    Raises:
        ValueError: If file format is invalid or size not found.
    """
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
        available_sizes = get_available_sizes(parquet_file)
        if size not in available_sizes:
            raise ValueError(f"Size '{size}' not found. Available sizes: {available_sizes}")

    # Find modes for this model/size
    modes = [
        col.split(f"{model}_{size}_")[1]
        for col in columns
        if col.startswith(f"{model}_{size}_")
    ]
    if len(modes) != 2:
        raise ValueError(f"Expected exactly two modes for size '{size}'. Found: {modes}")

    mode1, mode2 = modes

    # Convert the embeddings to numpy arrays
    embs1 = df[f"{model}_{size}_{mode1}"].to_numpy()
    embs2 = df[f"{model}_{size}_{mode2}"].to_numpy()

    # Ensure we pass a sequence of arrays to np.vstack
    arr1 = np.vstack(list(embs1))
    arr2 = np.vstack(list(embs2))

    metadata = {
        "model": model,
        "size": size,
        "mode1": mode1,
        "mode2": mode2,
    }

    return arr1, arr2, metadata


def compare_from_parquet(
    parquet_file: str,
    metrics: list[str] | None = None,
    size: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Load embeddings from a parquet file and compute metrics.

    Args:
        parquet_file: Path to parquet file
        metrics: List of metric names to compute
        size: Optional size to use. If None, uses first available.
              If "all", processes all available sizes.
        **kwargs: Additional keyword arguments passed to metrics.

    Returns:
        Dictionary with metadata and computed metrics.
        If size="all", returns a list of such dictionaries.
    """
    if size == "all":
        available_sizes = get_available_sizes(parquet_file)
        results = []
        for s in available_sizes:
            result = compare_from_parquet(parquet_file, metrics, size=s, **kwargs)
            results.append(result)
        return results

    # Load embeddings
    Z1, Z2, metadata = load_embeddings_from_parquet(parquet_file, size=size)

    # Compute metrics
    metric_results = compare(Z1, Z2, metrics=metrics, **kwargs)

    return {
        **metadata,
        "metrics": metric_results,
    }
