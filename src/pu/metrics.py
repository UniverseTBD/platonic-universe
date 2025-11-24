import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Any, Dict
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

    return np.mean(overlap) / k


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

    return np.mean(jaccard)


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

    df = pl.read_parquet(parquet_file)
    columns = df.columns

    # Extract model, size, and modes from column names
    example_col = columns[0]
    parts = example_col.split("_")
    if len(parts) < 3:
        raise ValueError("Column names must be in the format <model>_<size>_<mode>")

    model = parts[0]
    size = parts[1]
    modes = [col.split(f"{model}_{size}_")[1] for col in columns if col.startswith(f"{model}_{size}_")]
    
    if len(modes) != 2:
        raise ValueError("Expected exactly two modes to compare in the Parquet file.")

    mode1, mode2 = modes

    embs1 = df[f"{model}_{size}_{mode1}"].to_numpy()
    embs2 = df[f"{model}_{size}_{mode2}"].to_numpy()

    # Ensure we pass a sequence of arrays to np.vstack (convert ndarray-of-objects to a list)
    arr1 = np.vstack(list(embs1))
    arr2 = np.vstack(list(embs2))
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