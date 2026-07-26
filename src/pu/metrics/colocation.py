"""
Triple co-location primitives (Experiment 2).

Three "special" relative depths are defined by three *independent*
measurements and the hypothesis is that they coincide:

  * ``L_conv``  : depth where cross-model neighbourhood agreement peaks.
  * ``L*``      : depth where the manifold emerges (Experiment 1;
                  proxied here by the minimum of the per-layer intrinsic
                  dimension when no external value is supplied).
  * ``L_phys``  : depth where alignment with the physical catalogue peaks.

The atom underneath every measurement is the *neighbourhood overlap*: for
each row, the fraction of its ``k`` nearest neighbours in space A that are
also among its ``k`` nearest neighbours in space B, averaged over rows.
This is exactly ``dadapy.Data(A).return_data_overlap(coordinates=B, k=k)``.

DADApy itself is an *optional* backend here. Its compiled Cython extension
does not build against NumPy >= 2 (it references the removed
``_PyArray_Descr.subarray`` C field), and this repository pins
``numpy>=2.3``, so the pure-NumPy path below is the default. When DADApy is
importable (e.g. in a ``numpy<2`` environment) ``overlap(..., backend="dadapy")``
routes through it instead; ``tests/test_colocation.py`` asserts the two
paths agree.

Design notes that matter for correctness:

  * **Row order is meaning.** Overlap only makes sense when row ``i`` is the
    same physical object in both matrices. The permutation null deliberately
    breaks that correspondence to measure chance agreement.
  * **Build the base graph once, reuse it across the row.** For a full
    cross-layer matrix we build each layer-of-A neighbour graph a single
    time and score it against every layer-of-B (and against every permuted
    null), mirroring DADApy's "build ``Data`` once, call
    ``return_data_overlap`` many times" idiom.
  * **Embeddings vs. catalogue are normalised differently.** Embedding rows
    are L2-normalised (so Euclidean kNN == cosine kNN, matching the MKNN
    metric used elsewhere in the repo). The physical/nuisance catalogue is
    already column-standardised (``P_std``/``Q_std``), so its rows are used
    as-is under a plain Euclidean graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def l2_normalize_rows(X: NDArray[np.floating], eps: float = 1e-12) -> NDArray[np.float64]:
    """Scale every row to unit L2 norm (zero rows are left untouched)."""
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    return X / norms


# ---------------------------------------------------------------------------
# Neighbour graphs and overlap
# ---------------------------------------------------------------------------


def knn_indices(
    X: NDArray[np.floating],
    k: int = 10,
    normalize_rows: bool = True,
) -> NDArray[np.int64]:
    """Return the ``(n, k)`` matrix of nearest-neighbour row indices.

    Neighbours exclude the point itself. On L2-normalised rows the Euclidean
    ordering is identical to the cosine ordering, so ``normalize_rows=True``
    reproduces cosine kNN while keeping a single distance backend.

    Args:
        X: ``(n, d)`` coordinates.
        k: Number of neighbours.
        normalize_rows: L2-normalise rows first (use for embeddings; leave
            ``False`` for an already column-standardised catalogue).
    """
    X = l2_normalize_rows(X) if normalize_rows else np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(X)
    return nn.kneighbors(return_distance=False)


def _shared_counts(
    idx_a: NDArray[np.integer],
    idx_b: NDArray[np.integer],
) -> NDArray[np.int64]:
    """Per-row count of neighbours shared between two ``(n, k)`` graphs."""
    # (n, k, 1) == (n, 1, k) -> for each A-neighbour, is it among the B-neighbours.
    eq = idx_a[:, :, None] == idx_b[:, None, :]
    return eq.any(axis=2).sum(axis=1)


def _overlap_from_indices(
    idx_a: NDArray[np.integer],
    idx_b: NDArray[np.integer],
) -> float:
    """Mean fraction of shared neighbours between two ``(n, k)`` graphs."""
    k = idx_a.shape[1]
    return float(_shared_counts(idx_a, idx_b).mean() / k)


def _permute_graph(idx_b: NDArray[np.integer], perm: NDArray[np.integer]) -> NDArray[np.int64]:
    """Neighbour graph of ``B[perm]`` in the permuted index space.

    kNN is permutation-equivariant, so we never rebuild the graph: if object
    ``perm[i]`` sits at new row ``i``, its neighbours (old indices
    ``idx_b[perm[i]]``) map to new indices ``inv[idx_b[perm[i]]]``.
    """
    perm = np.asarray(perm)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv[idx_b[perm]]


def overlap(
    A: NDArray[np.floating],
    B: NDArray[np.floating],
    k: int = 10,
    normalize_a: bool = True,
    normalize_b: bool = True,
    backend: str = "auto",
) -> float:
    """Neighbourhood overlap between coordinate sets ``A`` and ``B``.

    Equivalent to ``dadapy.Data(A).return_data_overlap(coordinates=B, k=k)``.

    Args:
        A, B: ``(n, d_a)`` / ``(n, d_b)`` row-aligned coordinates.
        k: Neighbours per point.
        normalize_a, normalize_b: L2-normalise rows before building the graph.
        backend: ``"numpy"`` (default logic), ``"dadapy"`` (route through the
            optional DADApy backend), or ``"auto"`` (DADApy if importable).
    """
    if backend in ("auto", "dadapy"):
        try:  # pragma: no cover - exercised only where dadapy builds
            from dadapy import Data

            An = l2_normalize_rows(A) if normalize_a else np.asarray(A, dtype=np.float64)
            Bn = l2_normalize_rows(B) if normalize_b else np.asarray(B, dtype=np.float64)
            d = Data(An, verbose=False)
            return float(d.return_data_overlap(coordinates=Bn, k=k))
        except Exception:
            if backend == "dadapy":
                raise
    idx_a = knn_indices(A, k=k, normalize_rows=normalize_a)
    idx_b = knn_indices(B, k=k, normalize_rows=normalize_b)
    return _overlap_from_indices(idx_a, idx_b)


# ---------------------------------------------------------------------------
# Permutation calibration
# ---------------------------------------------------------------------------


@dataclass
class CalibratedScore:
    """Result of permutation-calibrating a single overlap cell."""

    calibrated: float  # z = (raw - null_mean) / null_std
    raw: float
    null_mean: float
    null_std: float
    nulls: NDArray[np.float64] = field(repr=False, default_factory=lambda: np.array([]))


def calibrate_overlap_graphs(
    idx_a: NDArray[np.integer],
    idx_b: NDArray[np.integer],
    n_perm: int = 10,
    eps: float = 1e-9,
) -> CalibratedScore:
    """Permutation-calibrate the overlap of two precomputed neighbour graphs.

    The null shuffles the row correspondence of ``B`` (seeds ``0 .. n_perm-1``)
    and recomputes the overlap; the calibrated score is the z-statistic
    ``(raw - null_mean) / null_std``. This removes the depth/width inflation
    that makes a wider or deeper representation look more aligned for free.
    """
    raw = _overlap_from_indices(idx_a, idx_b)
    n = idx_a.shape[0]
    nulls = np.empty(n_perm, dtype=np.float64)
    for s in range(n_perm):
        perm = np.random.default_rng(s).permutation(n)
        nulls[s] = _overlap_from_indices(idx_a, _permute_graph(idx_b, perm))
    mu = float(nulls.mean())
    sd = float(nulls.std())
    z = (raw - mu) / (sd if sd > eps else eps)
    return CalibratedScore(calibrated=z, raw=raw, null_mean=mu, null_std=sd, nulls=nulls)


# ---------------------------------------------------------------------------
# Step 1: cross-layer calibrated matrix and L_conv
# ---------------------------------------------------------------------------


@dataclass
class CrossLayerResult:
    raw: NDArray[np.float64]        # M[i, j]
    calibrated: NDArray[np.float64]  # M_cal[i, j]
    peak_layer_a: int
    peak_layer_b: int


def cross_layer_matrix(
    layers_a: Sequence[NDArray[np.floating]],
    layers_b: Sequence[NDArray[np.floating]],
    k: int = 10,
    n_perm: int = 10,
    normalize_rows: bool = True,
) -> CrossLayerResult:
    """Full calibrated cross-layer overlap matrix for one model pair (Step 1).

    ``M[i, j]`` is the raw overlap between layer ``i`` of model A and layer
    ``j`` of model B; ``M_cal`` is its permutation-calibrated z-score. Each
    A-layer graph and each B-layer graph is built exactly once and reused
    across the whole row/column, matching the spec's "build ``Data`` once per
    layer of A, call ``return_data_overlap`` once per layer of B".

    The peak cell of ``M_cal`` yields a peak layer for each of the two models.
    """
    graphs_b = [knn_indices(B, k=k, normalize_rows=normalize_rows) for B in layers_b]
    na, nb = len(layers_a), len(layers_b)
    raw = np.zeros((na, nb))
    cal = np.zeros((na, nb))
    for i, A in enumerate(layers_a):
        idx_a = knn_indices(A, k=k, normalize_rows=normalize_rows)  # built once, reused across the row
        for j, idx_b in enumerate(graphs_b):
            cs = calibrate_overlap_graphs(idx_a, idx_b, n_perm=n_perm)
            raw[i, j] = cs.raw
            cal[i, j] = cs.calibrated
    peak_i, peak_j = np.unravel_index(int(np.argmax(cal)), cal.shape)
    return CrossLayerResult(raw=raw, calibrated=cal, peak_layer_a=int(peak_i), peak_layer_b=int(peak_j))


def l_conv_from_pairs(
    peak_layers: dict[str, list[int]],
    n_layers: dict[str, int],
) -> dict[str, dict[str, float]]:
    """Median peak layer per model across all pairs it appears in.

    Args:
        peak_layers: ``{model: [peak_layer_index_in_pair_1, ...]}`` with one
            entry per pair the model participates in (as A or B).
        n_layers: ``{model: L}`` layer count per model.

    Returns:
        ``{model: {"layer": median_index, "relative": median/(L-1),
                   "peaks": [...]}}``.
    """
    out: dict[str, dict[str, float]] = {}
    for model, peaks in peak_layers.items():
        med = float(np.median(peaks))
        L = n_layers[model]
        out[model] = {
            "layer": med,
            "relative": med / (L - 1) if L > 1 else 0.0,
            "peaks": list(peaks),
        }
    return out


# ---------------------------------------------------------------------------
# Step 2: physics overlap curve, L_phys, and supporting probes
# ---------------------------------------------------------------------------


def physics_overlap_curve(
    layers: Sequence[NDArray[np.floating]],
    catalog_std: NDArray[np.floating],
    k: int = 10,
    n_perm: int = 10,
    normalize_emb: bool = True,
) -> dict[str, NDArray[np.float64]]:
    """Calibrated per-layer overlap between embeddings and a catalogue (Step 2).

    ``catalog_std`` is the already column-standardised catalogue (``P_std`` for
    physics, ``Q_std`` for the nuisance control). The null permutes the
    catalogue's rows. ``L_phys`` is ``argmax`` of the calibrated curve.

    Returns a dict of per-layer arrays: ``calibrated``, ``raw``,
    ``null_mean``, ``null_std``.
    """
    cat = np.asarray(catalog_std, dtype=np.float64)
    idx_cat = knn_indices(cat, k=k, normalize_rows=False)
    nL = len(layers)
    calibrated = np.zeros(nL)
    raw = np.zeros(nL)
    null_mean = np.zeros(nL)
    null_std = np.zeros(nL)
    for li, X in enumerate(layers):
        idx_x = knn_indices(X, k=k, normalize_rows=normalize_emb)
        cs = calibrate_overlap_graphs(idx_x, idx_cat, n_perm=n_perm)
        calibrated[li] = cs.calibrated
        raw[li] = cs.raw
        null_mean[li] = cs.null_mean
        null_std[li] = cs.null_std
    return {"calibrated": calibrated, "raw": raw, "null_mean": null_mean, "null_std": null_std}


def probe_r2_curve(
    layers: Sequence[NDArray[np.floating]],
    catalog: NDArray[np.floating],
    normalize_emb: bool = True,
    alphas: Sequence[float] = (0.1, 1.0, 10.0, 100.0),
) -> NDArray[np.float64]:
    """Per-layer held-out probe R^2 averaged over catalogue columns (supporting).

    ``RidgeCV`` inside ``cross_val_score`` with ``KFold(5, shuffle=True,
    random_state=0)``, one target column at a time. Rows with a non-finite
    target are dropped per column. This is a supporting curve, not the headline.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, cross_val_score

    cat = np.asarray(catalog, dtype=np.float64)
    if cat.ndim == 1:
        cat = cat[:, None]
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    out = np.zeros(len(layers))
    for li, X in enumerate(layers):
        Xn = l2_normalize_rows(X) if normalize_emb else np.asarray(X, dtype=np.float64)
        col_scores = []
        for c in range(cat.shape[1]):
            y = cat[:, c]
            mask = np.isfinite(y)
            if mask.sum() < 10:
                continue
            scores = cross_val_score(
                RidgeCV(alphas=list(alphas)), Xn[mask], y[mask], cv=kf, scoring="r2"
            )
            col_scores.append(float(np.mean(scores)))
        out[li] = float(np.mean(col_scores)) if col_scores else np.nan
    return out


def cca_curve(
    layers: Sequence[NDArray[np.floating]],
    catalog: NDArray[np.floating],
    normalize_emb: bool = True,
    train_frac: float = 0.7,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Per-layer mean held-out canonical correlation (supporting curve).

    ``CCA(n_components=p)`` fit on a ``train_frac`` split and scored on the
    held-out remainder by correlating each paired transformed component with
    ``numpy.corrcoef``. Averaged over the ``p`` components.
    """
    from sklearn.cross_decomposition import CCA

    cat = np.asarray(catalog, dtype=np.float64)
    if cat.ndim == 1:
        cat = cat[:, None]
    # Keep rows finite across all catalogue columns so CCA sees a clean block.
    mask = np.all(np.isfinite(cat), axis=1)
    cat = cat[mask]
    p = cat.shape[1]
    n = cat.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr = int(round(train_frac * n))
    tr, te = perm[:n_tr], perm[n_tr:]
    out = np.zeros(len(layers))
    for li, X in enumerate(layers):
        Xn = l2_normalize_rows(X) if normalize_emb else np.asarray(X, dtype=np.float64)
        Xn = Xn[mask]
        n_comp = min(p, Xn.shape[1])
        try:
            cca = CCA(n_components=n_comp, max_iter=1000)
            cca.fit(Xn[tr], cat[tr])
            Xc, Yc = cca.transform(Xn[te], cat[te])
            corrs = [
                abs(np.corrcoef(Xc[:, i], Yc[:, i])[0, 1])
                for i in range(n_comp)
                if np.std(Xc[:, i]) > 0 and np.std(Yc[:, i]) > 0
            ]
            out[li] = float(np.mean(corrs)) if corrs else np.nan
        except Exception:
            out[li] = np.nan
    return out


# ---------------------------------------------------------------------------
# L*: intrinsic dimension (proxy for Experiment 1's manifold layer)
# ---------------------------------------------------------------------------


def twonn_dimension(
    X: NDArray[np.floating],
    normalize_rows: bool = True,
    discard_fraction: float = 0.1,
) -> float:
    """Global intrinsic dimension via TwoNN (Facco et al. 2017).

    For each point ``mu = r2 / r1`` (ratio of 2nd to 1st neighbour distance).
    The MLE slope of ``log(mu)`` against the empirical ``-log(1 - F)`` gives
    the intrinsic dimension. The largest ``discard_fraction`` of ``mu`` values
    is dropped to reduce the influence of density outliers.
    """
    Xn = l2_normalize_rows(X) if normalize_rows else np.asarray(X, dtype=np.float64)
    nn = NearestNeighbors(n_neighbors=3, metric="euclidean").fit(Xn)
    dists, _ = nn.kneighbors(Xn)  # column 0 is the point itself (distance 0)
    r1 = dists[:, 1]
    r2 = dists[:, 2]
    good = r1 > 0
    mu = np.sort(r2[good] / r1[good])
    m = len(mu)
    if m < 3:
        return float("nan")
    # Empirical CDF over the *full* sample, then discard the largest-mu tail
    # (density outliers) for the linear fit, keeping the original F values.
    Femp = np.arange(1, m + 1) / m
    keep = int(m * (1 - discard_fraction))
    x = np.log(mu[:keep])
    y = -np.log(1.0 - Femp[:keep] + 1e-12)
    # Slope through the origin: d = sum(x*y) / sum(x*x).
    denom = float(np.sum(x * x))
    return float(np.sum(x * y) / denom) if denom > 0 else float("nan")


def intrinsic_dimension_curve(
    layers: Sequence[NDArray[np.floating]],
    normalize_rows: bool = True,
) -> NDArray[np.float64]:
    """Per-layer TwoNN intrinsic dimension."""
    return np.array([twonn_dimension(X, normalize_rows=normalize_rows) for X in layers])


def manifold_layer(id_curve: NDArray[np.floating]) -> int:
    """L*: the layer where the manifold emerges = argmin of the ID curve.

    A proxy for Experiment 1's output; if Experiment 1 supplies an explicit
    ``L*`` prefer that and skip this.
    """
    id_curve = np.asarray(id_curve, dtype=np.float64)
    return int(np.nanargmin(id_curve))


# ---------------------------------------------------------------------------
# Step 3: depth assembly, bootstrap, relative depth
# ---------------------------------------------------------------------------


def relative_depth(layer_index: float, n_layers: int) -> float:
    """Convert an absolute layer index to relative depth ``L/(L-1)``."""
    return float(layer_index) / (n_layers - 1) if n_layers > 1 else 0.0


def iqr(values: Sequence[float]) -> float:
    """Interquartile range (75th minus 25th percentile)."""
    v = np.asarray(values, dtype=np.float64)
    return float(np.percentile(v, 75) - np.percentile(v, 25))


@dataclass
class BootstrapDepth:
    point: float                    # relative depth on the full sample
    median: float                   # median of half-sample estimates
    q25: float
    q75: float
    iqr: float
    estimates: list[float] = field(default_factory=list)


def bootstrap_depth(
    estimator: Callable[[NDArray[np.integer]], float],
    n_rows: int,
    point_estimate: float | None = None,
    seeds: Sequence[int] = range(10),
    frac: float = 0.5,
) -> BootstrapDepth:
    """Half-sample bootstrap of a relative-depth estimator (Step 3).

    ``estimator(row_indices) -> relative_depth`` is recomputed on 10 random
    half-samples (seeds ``0 .. 9``); the uncertainty is the IQR of the 10
    estimates and the ``[q25, q75]`` interval is used for the co-location test.
    """
    ests = []
    size = max(2, int(round(frac * n_rows)))
    for s in seeds:
        idx = np.random.default_rng(s).choice(n_rows, size=size, replace=False)
        ests.append(float(estimator(idx)))
    arr = np.asarray(ests, dtype=np.float64)
    point = float(point_estimate) if point_estimate is not None else float(np.median(arr))
    return BootstrapDepth(
        point=point,
        median=float(np.median(arr)),
        q25=float(np.percentile(arr, 25)),
        q75=float(np.percentile(arr, 75)),
        iqr=float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        estimates=ests,
    )


# ---------------------------------------------------------------------------
# Step 4: coincidence test
# ---------------------------------------------------------------------------


def intervals_mutually_overlap(intervals: Sequence[tuple[float, float]]) -> bool:
    """True iff every pair of ``[low, high]`` intervals intersects.

    For a set of intervals this holds iff ``max(lows) <= min(highs)``.
    """
    lows = [lo for lo, _ in intervals]
    highs = [hi for _, hi in intervals]
    return max(lows) <= min(highs)


@dataclass
class ColocationVerdict:
    passed: bool
    gaps: dict[str, float]       # pairwise |Δ relative depth|
    max_gap: float
    mean_gap: float
    iqrs_overlap: bool
    depths: dict[str, float]     # point relative depths


def colocation_test(
    depths: dict[str, float],
    intervals: dict[str, tuple[float, float]],
    gap_threshold: float = 0.10,
) -> ColocationVerdict:
    """Declare co-location for one model (Step 4).

    Co-location holds iff all three pairwise gaps in relative depth are
    ``<= gap_threshold`` **and** the three bootstrap ``[q25, q75]`` intervals
    mutually overlap.

    Args:
        depths: ``{"conv": .., "star": .., "phys": ..}`` point relative depths.
        intervals: same keys → ``(q25, q75)`` from the bootstrap.
    """
    keys = ("conv", "star", "phys")
    pairs = {
        "conv-star": abs(depths["conv"] - depths["star"]),
        "conv-phys": abs(depths["conv"] - depths["phys"]),
        "star-phys": abs(depths["star"] - depths["phys"]),
    }
    max_gap = max(pairs.values())
    mean_gap = float(np.mean(list(pairs.values())))
    ov = intervals_mutually_overlap([intervals[k] for k in keys])
    passed = (max_gap <= gap_threshold) and ov
    return ColocationVerdict(
        passed=passed,
        gaps=pairs,
        max_gap=float(max_gap),
        mean_gap=mean_gap,
        iqrs_overlap=bool(ov),
        depths={k: float(depths[k]) for k in keys},
    )


def summarize_colocation(verdicts: dict[str, ColocationVerdict]) -> dict[str, float]:
    """Fraction of models passing and the mean pairwise gap across models."""
    if not verdicts:
        return {"fraction_passing": float("nan"), "mean_gap": float("nan"), "n_models": 0}
    n = len(verdicts)
    frac = sum(v.passed for v in verdicts.values()) / n
    mean_gap = float(np.mean([v.mean_gap for v in verdicts.values()]))
    return {"fraction_passing": float(frac), "mean_gap": mean_gap, "n_models": n}
