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

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, KFold
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
        "disk_fraction": "smooth-or-featured_featured-or-disk_fraction",
        "artifact": "smooth-or-featured_artifact_fraction",
        "edge_on": "disk-edge-on_yes_fraction",
        "tight_spiral": "spiral-winding_tight_fraction",
    },
    "photometry": {
        "mag_r_desi": "mag_r_desi",
        "mag_g_desi": "mag_g_desi",
    },
    "physical": {
        "photo_z": "photo_z",
        "spec_z": "spec_z",
        "stellar_mass": "mass_med_photoz",
    },
    "star_formation": {
        "sfr": "total_sfr_avg",
        "ssfr": "total_ssfr_avg",
    },
}

# Flat lookup: short_key -> column_name
ALL_PROPERTIES: dict[str, str] = {}
for _group in PROPERTY_GROUPS.values():
    ALL_PROPERTIES.update(_group)

# A sensible default subset for quick runs
DEFAULT_PROPERTIES = [
        "smooth_fraction",
        "disk_fraction",
        "artifact",
        "edge_on",
        "tight_spiral",
        "mag_r_desi",
        "mag_g_desi",
        "photo_z",
        "spec_z",
        "stellar_mass",
        "sfr",
]


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------


def linear_probe(
    Z: NDArray[np.floating],
    y: NDArray[np.floating],
    cv: int = 5,
    pca_components: int | None = None,
) -> float:
    """
    Linear probe: cross-validated R² for predicting property *y* from embeddings *Z*.

    A high R² means the embedding space linearly encodes this physical property.

    Args:
        Z: (n_samples, d) embedding matrix
        y: (n_samples,) target physical property
        cv: Number of cross-validation folds
        pca_components: If set, reduce embeddings to this many PCA components
                        before regression (fit on train fold only to avoid leakage)

    Returns:
        Mean cross-validated R² (can be negative if worse than predicting the mean)
    """
    Z, y = _clean_inputs(Z, y)
    if len(Z) < cv:
        return float("nan")
    # L2-normalise so that models with different embedding scales are
    # compared fairly — without this, large models with big norms
    # destabilise the unconstrained linear regression.
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    Z = Z / norms

    if pca_components is not None:
        n_components = min(pca_components, Z.shape[1], Z.shape[0])
        kf = KFold(n_splits=cv)
        scores = []
        for train_idx, test_idx in kf.split(Z):
            pca = PCA(n_components=n_components)
            Z_train = pca.fit_transform(Z[train_idx])
            Z_test = pca.transform(Z[test_idx])
            model = LinearRegression()
            model.fit(Z_train, y[train_idx])
            scores.append(r2_score(y[test_idx], model.predict(Z_test)))
    else:
        model = LinearRegression()
        scores = cross_val_score(model, Z, y, cv=cv, scoring="r2").tolist()

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "folds": [float(s) for s in scores],
    }


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
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "folds": scores.tolist(),
    }


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


def neighbor_set_overlap(
    Z: NDArray[np.floating],
    y: NDArray[np.floating],
    k: int = 10,
) -> float:
    """
    Neighbor set overlap between embedding space and a single physical property.

    For each galaxy, find its k nearest neighbours in embedding space
    (cosine distance) and its k nearest neighbours in property space
    (absolute difference on the scalar property).  Return the mean
    fraction of overlap.

    This is analogous to MKNN but with one side being a physical
    property rather than a second model's embeddings.  Unlike
    ``neighbor_property_consistency`` (which measures neighbourhood
    *spread*), this directly measures whether the embedding retrieves
    the *same* neighbours that the property would — a true retrieval
    metric.

    Args:
        Z: (n_samples, d) embedding matrix
        y: (n_samples,) target physical property
        k: Number of nearest neighbours

    Returns:
        Overlap fraction in [0, 1].  Higher is better.
        1 = embedding and property spaces agree perfectly on neighbourhoods.
    """
    Z, y = _clean_inputs(Z, y)
    n = len(Z)
    if n < k + 1:
        return float("nan")

    k = min(k, n - 1)

    # Neighbours in embedding space (cosine distance)
    nn_emb = NearestNeighbors(n_neighbors=k, metric="cosine").fit(Z)
    idx_emb = nn_emb.kneighbors(return_distance=False)

    # Neighbours in property space (absolute difference on scalar)
    y_col = y.reshape(-1, 1)
    nn_prop = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(y_col)
    idx_prop = nn_prop.kneighbors(return_distance=False)

    # Mean overlap fraction
    overlaps = [
        len(set(e).intersection(p)) / k
        for e, p in zip(idx_emb, idx_prop)
    ]

    return float(np.mean(overlaps))


def joint_neighbor_set_overlap(
    Z: NDArray[np.floating],
    properties: dict[str, NDArray[np.floating]],
    property_keys: list[str] | None = None,
    k: int = 10,
) -> dict[str, float]:
    """
    Neighbor set overlap between embedding space and *joint* physical
    property space.

    Concatenates all requested physical properties into a single
    standardised feature vector per galaxy, computes k nearest neighbours
    in that joint property space, and measures the overlap with k nearest
    neighbours in embedding space.

    This is the headline retrieval metric: it asks whether the embedding's
    neighbourhood structure reflects *overall* physical similarity, not
    just one property at a time.

    Galaxies with NaN/Inf in *any* included property are dropped so that
    all properties share the same sample set.

    Args:
        Z: (n_samples, d) embedding matrix
        properties: dict mapping short property keys to (n_samples,) arrays
        property_keys: which properties to include in the joint space
                       (default: all keys in ``properties``)
        k: Number of nearest neighbours

    Returns:
        Dict with:
            - ``"joint_overlap"``: the headline overlap fraction in [0, 1]
            - ``"n_properties"``: how many properties were used
            - ``"n_samples"``: how many galaxies survived NaN filtering
            - ``"properties_used"``: list of property keys included
    """
    if property_keys is None:
        property_keys = list(properties.keys())

    # --- Build the joint property matrix, dropping NaN rows across all ---
    Z = np.asarray(Z, dtype=np.float64)
    n = Z.shape[0]

    # Stack available property columns
    available_keys: list[str] = []
    cols: list[NDArray[np.floating]] = []
    for key in property_keys:
        if key not in properties:
            continue
        arr = np.asarray(properties[key], dtype=np.float64).ravel()
        if len(arr) != n:
            continue
        available_keys.append(key)
        cols.append(arr)

    if len(cols) < 2:
        # Need at least 2 properties for a meaningful joint space
        return {
            "joint_overlap": float("nan"),
            "n_properties": len(cols),
            "n_samples": 0,
            "properties_used": available_keys,
        }

    Y = np.column_stack(cols)  # (n, n_properties)

    # Drop rows where any property or any embedding dim is NaN/Inf
    valid = (
        np.all(np.isfinite(Y), axis=1)
        & np.all(np.isfinite(Z), axis=1)
    )
    Z_clean = Z[valid]
    Y_clean = Y[valid]
    n_clean = len(Z_clean)

    if n_clean < k + 1:
        return {
            "joint_overlap": float("nan"),
            "n_properties": len(available_keys),
            "n_samples": n_clean,
            "properties_used": available_keys,
        }

    k_use = min(k, n_clean - 1)

    # Standardise each property column to zero mean, unit variance
    # so that no single property dominates the distance calculation
    scaler = StandardScaler()
    Y_scaled = scaler.fit_transform(Y_clean)

    # Neighbours in embedding space (cosine distance)
    nn_emb = NearestNeighbors(n_neighbors=k_use, metric="cosine").fit(Z_clean)
    idx_emb = nn_emb.kneighbors(return_distance=False)

    # Neighbours in joint property space (Euclidean on standardised properties)
    nn_prop = NearestNeighbors(n_neighbors=k_use, metric="euclidean").fit(Y_scaled)
    idx_prop = nn_prop.kneighbors(return_distance=False)

    # Mean overlap fraction
    overlaps = [
        len(set(e).intersection(p)) / k_use
        for e, p in zip(idx_emb, idx_prop)
    ]

    return {
        "joint_overlap": float(np.mean(overlaps)),
        "n_properties": len(available_keys),
        "n_samples": n_clean,
        "properties_used": available_keys,
    }



# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


def run_physics_tests(
    Z: NDArray[np.floating],
    properties: dict[str, NDArray[np.floating]],
    property_keys: list[str] | None = None,
    k: int = 10,
    cv: int = 5,
    pca_components: int | None = None,
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
    # Collect per-fold R² arrays for SE propagation
    fold_arrays: dict[str, list[float]] = {}

    for key in property_keys:
        if key not in properties:
            continue

        y = properties[key]
        lp = linear_probe(Z, y, cv=cv, pca_components=pca_components)

        # linear_probe returns a dict on success, float("nan") on too-few samples
        if isinstance(lp, dict):
            lp_r2 = lp["mean"]
            lp_std = lp["std"]
            lp_folds = lp["folds"]
        else:
            lp_r2 = float("nan")
            lp_std = float("nan")
            lp_folds = []

        results[key] = {
            "linear_probe_r2": lp_r2,
            "linear_probe_r2_std": lp_std,
            "linear_probe_r2_folds": lp_folds,
        }
        fold_arrays[key] = lp_folds

    # Summary: mean R² across all tested properties with propagated SE.
    #
    # Each property i has F cross-validation fold scores.  Its mean R²_i
    # has standard error  se_i = std_i / sqrt(F).
    # The grand mean  R̄² = (1/K) Σ R²_i  (K properties) has variance
    #   Var(R̄²) = (1/K²) Σ se_i²  = (1/K²) Σ (std_i² / F)
    # assuming independence across properties (different targets, same
    # embedding matrix — fold splits are shared but targets are independent
    # so the R² estimators are effectively uncorrelated).
    r2_values = {k: v["linear_probe_r2"] for k, v in results.items()}
    r2_means = np.array([v for v in r2_values.values()])
    K = np.sum(np.isfinite(r2_means))

    if K > 0 and len(fold_arrays) > 0:
        # Propagate CV fold variances → SE of the grand mean
        sum_var = 0.0
        for folds in fold_arrays.values():
            arr = np.array(folds)
            if len(arr) > 1 and np.all(np.isfinite(arr)):
                sum_var += np.var(arr, ddof=1) / len(arr)  # se_i² = s²/F
        r2_se = float(np.sqrt(sum_var) / K)
    else:
        r2_se = float("nan")

    results["_summary"] = {
        "r2_per_property": r2_values,
        "r2_mean": float(np.nanmean(r2_means)),
        "r2_se": r2_se,
        "r2_std": float(np.nanstd(r2_means)),
        "n_properties": int(K),
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
    # Drop outliers
    lower, upper = np.nanpercentile(y, [1, 99])
    inliers = (y >= lower) & (y <= upper)
    mask = valid & inliers
    Z = Z[mask]
    y = y[mask]

    if len(Z) == 0:
        raise ValueError("No valid samples after removing NaN/Inf")

    return Z, y
