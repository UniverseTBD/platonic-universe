"""
Confound residualization: remove variance explained by physical confounds.
"""

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import KFold


def confound_basis(
    C: NDArray[np.floating],
    degree: int = 3,
    standardize: bool = True,
    include_bias: bool = True,
) -> NDArray[np.floating]:
    """
    Polynomial design matrix from confounds *C*.

    Each confound is optionally standardised and expanded to powers 1..degree;
    a constant column is prepended when *include_bias*.

    Args:
        C: (n_samples, n_confounds) or (n_samples,) confound matrix (finite).
        degree: Highest polynomial power per confound (1 = linear).
        standardize: z-score each confound before taking powers.
        include_bias: Prepend a constant column.

    Returns:
        (n_samples, m) design matrix, m = include_bias + n_confounds * degree.
    """
    C = np.asarray(C, dtype=np.float64)
    if C.ndim == 1:
        C = C[:, None]
    n, p = C.shape
    cols = [np.ones((n, 1))] if include_bias else []
    for j in range(p):
        x = C[:, j]
        if standardize:
            sd = np.std(x)
            x = (x - np.mean(x)) / (sd if sd > 1e-12 else 1.0)
        cols += [(x**d)[:, None] for d in range(1, degree + 1)]
    return np.concatenate(cols, axis=1)


def residualize(
    Z: NDArray[np.floating],
    C: NDArray[np.floating],
    degree: int = 3,
    standardize: bool = True,
) -> NDArray[np.floating]:
    """
    Remove from *Z* the variance explained by the confounds *C*.

    Returns *Z* minus its least-squares projection onto the polynomial confound
    basis, so each embedding dimension is orthogonal to (nonlinear functions of)
    the confounds.

    Args:
        Z: (n_samples, d) embedding matrix.
        C: (n_samples, n_confounds) confound matrix (finite).
        degree: Polynomial degree of the confound basis.
        standardize: z-score confounds before expansion.

    Returns:
        (n_samples, d) residualized embeddings (float32).
    """
    Z = np.asarray(Z, dtype=np.float64)
    B = confound_basis(C, degree=degree, standardize=standardize)
    coef, *_ = np.linalg.lstsq(B, Z, rcond=None)
    return (Z - B @ coef).astype(np.float32)


def residualize_crossfit(
    Z: NDArray[np.floating],
    C: NDArray[np.floating],
    degree: int = 3,
    standardize: bool = True,
    n_splits: int = 5,
    seed: int = 0,
) -> NDArray[np.floating]:
    """
    Cross-fitted (out-of-fold) :func:`residualize`.

    Fits the projection on the train folds and applies it to the held-out fold,
    so no row's residual depends on itself through the fit.

    Args:
        Z: (n_samples, d) embedding matrix.
        C: (n_samples, n_confounds) confound matrix (finite).
        degree: Polynomial degree of the confound basis.
        standardize: z-score confounds before expansion.
        n_splits: Number of folds.
        seed: Fold-shuffling seed.

    Returns:
        (n_samples, d) residualized embeddings (float32).
    """
    Z = np.asarray(Z, dtype=np.float64)
    B = confound_basis(C, degree=degree, standardize=standardize)
    out = np.empty_like(Z)
    for tr, te in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(Z):
        coef, *_ = np.linalg.lstsq(B[tr], Z[tr], rcond=None)
        out[te] = Z[te] - B[te] @ coef
    return out.astype(np.float32)
