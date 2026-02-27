"""
Physics-informed validation metrics.

Test whether embeddings encode physically meaningful galaxy properties
from the Smith42/galaxies dataset. These metrics complement the
representational similarity metrics (CKA, MKNN, etc.) by checking that
convergent representations actually capture real astrophysics.

The key idea: if foundation models converge toward a shared representation
of reality (the Platonic Representation Hypothesis), then embeddings should
predict physical galaxy properties — and larger models should do so better.

Requires: Smith42/galaxies (revision v2.0) which bundles metadata directly.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Any

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Canonical physical properties to probe, grouped by science domain.
# Each entry maps a short key to the column name in Smith42/galaxies v2.0.
# ---------------------------------------------------------------------------

PROPERTY_GROUPS: dict[str, dict[str, str]] = {
    "morphology": {
        "smooth_fraction": "smooth-or-featured_smooth_fraction",
        "featured_fraction": "smooth-or-featured_featured-or-disk_fraction",
        "spiral_arms": "has-spiral-arms_yes_fraction",
        "bar_strong": "bar_strong_fraction",
        "bulge_dominant": "bulge-size_dominant_fraction",
        "edge_on": "disk-edge-on_yes_fraction",
        "merging": "merging_merger_fraction",
    },
    "photometry": {
        "mag_r": "mag_r",
        "mag_g": "mag_g",
        "u_minus_r": "u_minus_r",
    },
    "structure": {
        "sersic_n": "sersic_n",
        "petro_th50": "petro_th50",
        "petro_th90": "petro_th90",
        "elpetro_ba": "elpetro_ba",
    },
    "physical": {
        "stellar_mass": "elpetro_mass_log",
        "redshift": "redshift",
    },
    "star_formation": {
        "sfr": "total_sfr_median",
        "ssfr": "total_ssfr_median",
    },
}

# Flat lookup: short_key -> column_name
ALL_PROPERTIES: dict[str, str] = {}
for _group in PROPERTY_GROUPS.values():
    ALL_PROPERTIES.update(_group)

# A sensible default subset for quick runs
DEFAULT_PROPERTIES = [
    "stellar_mass",
    "u_minus_r",
    "redshift",
    "sersic_n",
    "smooth_fraction",
    "spiral_arms",
    "sfr",
]


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------


def linear_probe(
    Z: NDArray[np.floating],
    y: NDArray[np.floating],
    alpha: float = 1.0,
    cv: int = 5,
) -> float:
    """
    Linear probe: cross-validated R² for predicting property *y* from embeddings *Z*.

    A high R² means the embedding space linearly encodes this physical property.

    Args:
        Z: (n_samples, d) embedding matrix
        y: (n_samples,) target physical property
        alpha: Ridge regularisation strength
        cv: Number of cross-validation folds

    Returns:
        Mean cross-validated R² (can be negative if worse than predicting the mean)
    """
    Z, y = _clean_inputs(Z, y)
    if len(Z) < cv:
        return float("nan")

    scaler = StandardScaler()
    Z_scaled = scaler.fit_transform(Z)

    model = Ridge(alpha=alpha)
    scores = cross_val_score(model, Z_scaled, y, cv=cv, scoring="r2")
    return float(np.mean(scores))


def nonlinear_probe(
    Z: NDArray[np.floating],
    y: NDArray[np.floating],
    cv: int = 5,
    n_estimators: int = 100,
    max_depth: int = 4,
) -> float:
    """
    Non-linear probe using gradient boosting.

    Captures non-linear relationships between embeddings and physical properties.
    Comparing linear_probe vs nonlinear_probe reveals how much non-linear
    structure the embedding encodes.

    Args:
        Z: (n_samples, d) embedding matrix
        y: (n_samples,) target physical property
        cv: Number of cross-validation folds
        n_estimators: Number of boosting rounds
        max_depth: Max tree depth

    Returns:
        Mean cross-validated R²
    """
    Z, y = _clean_inputs(Z, y)
    if len(Z) < cv:
        return float("nan")

    scaler = StandardScaler()
    Z_scaled = scaler.fit_transform(Z)

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        subsample=0.8,
        random_state=42,
    )
    scores = cross_val_score(model, Z_scaled, y, cv=cv, scoring="r2")
    return float(np.mean(scores))


def neighbor_property_consistency(
    Z: NDArray[np.floating],
    y: NDArray[np.floating],
    k: int = 10,
) -> float:
    """
    Neighbour property consistency ratio.

    For each sample, find its k nearest neighbours in embedding space and
    measure the standard deviation of *y* among those neighbours.  Return
    the ratio of mean neighbour-std to global std.

    A ratio < 1 means neighbours in embedding space are more similar in
    this physical property than random pairs — i.e. the embedding captures
    information about *y*.

    Args:
        Z: (n_samples, d) embedding matrix
        y: (n_samples,) target physical property
        k: Number of nearest neighbours

    Returns:
        Consistency ratio (lower is better, < 1 means physics is encoded)
    """
    Z, y = _clean_inputs(Z, y)
    n = len(Z)
    if n < k + 1:
        return float("nan")

    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k, metric="cosine").fit(Z)
    indices = nn.kneighbors(return_distance=False)

    neighbor_stds = np.array([np.std(y[idx]) for idx in indices])
    global_std = np.std(y)

    if global_std < 1e-12:
        return float("nan")

    return float(np.mean(neighbor_stds) / global_std)


def embedding_property_correlation(
    Z: NDArray[np.floating],
    y: NDArray[np.floating],
    max_samples: int = 5000,
) -> float:
    """
    Spearman correlation between pairwise embedding distances and pairwise
    property differences.

    Tests whether the *geometry* of the embedding space reflects physical
    property space: galaxies that are nearby in embedding space should have
    similar physical properties.

    Args:
        Z: (n_samples, d) embedding matrix
        y: (n_samples,) target physical property
        max_samples: Subsample to this many points (pdist is O(n²))

    Returns:
        Spearman rho in [-1, 1].  Positive = distance correlates with
        property difference (expected for well-structured embeddings).
    """
    Z, y = _clean_inputs(Z, y)

    # Subsample for computational tractability
    if len(Z) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(Z), max_samples, replace=False)
        Z, y = Z[idx], y[idx]

    emb_dists = pdist(Z, metric="cosine")
    prop_dists = pdist(y.reshape(-1, 1), metric="euclidean")

    corr, _ = spearmanr(emb_dists, prop_dists)
    return float(corr)


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


def run_physics_tests(
    Z: NDArray[np.floating],
    properties: dict[str, NDArray[np.floating]],
    property_keys: list[str] | None = None,
    k: int = 10,
    cv: int = 5,
) -> dict[str, dict[str, float]]:
    """
    Run a suite of physics tests for one embedding matrix.

    Args:
        Z: (n_samples, d) embedding matrix
        properties: dict mapping short property keys to (n_samples,) arrays
        property_keys: which properties to test (default: DEFAULT_PROPERTIES
                       intersected with available keys)
        k: k for neighbour consistency
        cv: folds for linear probe

    Returns:
        Nested dict: {property_key: {metric_name: value, ...}, ...}

    Example:
        >>> results = run_physics_tests(Z, {"stellar_mass": mass_arr, "redshift": z_arr})
        >>> results["stellar_mass"]["linear_probe_r2"]
        0.72
    """
    if property_keys is None:
        property_keys = [p for p in DEFAULT_PROPERTIES if p in properties]

    results: dict[str, dict[str, float]] = {}

    for key in property_keys:
        if key not in properties:
            continue

        y = properties[key]

        results[key] = {
            "linear_probe_r2": linear_probe(Z, y, cv=cv),
            "neighbor_consistency": neighbor_property_consistency(Z, y, k=k),
            "distance_correlation": embedding_property_correlation(Z, y),
        }

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_inputs(
    Z: NDArray[np.floating],
    y: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Remove NaN/Inf rows and ensure matching lengths."""
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()

    if Z.shape[0] != y.shape[0]:
        raise ValueError(
            f"Z has {Z.shape[0]} samples but y has {y.shape[0]}"
        )

    # Drop rows where y is NaN or Inf, or any Z feature is NaN/Inf
    valid = np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    Z = Z[valid]
    y = y[valid]

    if len(Z) == 0:
        raise ValueError("No valid samples after removing NaN/Inf")

    return Z, y
