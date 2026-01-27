"""
Regression-based similarity metrics: Linear R².
"""

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from pu.metrics._base import validate_inputs


def linear_r2(
    Z1: NDArray[np.floating],
    Z2: NDArray[np.floating],
    alpha: float = 1.0,
    cv: int | None = None,
) -> float:
    """
    Linear R² score: variance explained by linear mapping from Z1 to Z2.

    Fits a ridge regression from Z1 to Z2 and returns the R² score,
    measuring how much of Z2's variance can be explained by a linear
    transformation of Z1.

    Args:
        Z1: (n_samples, d1) source embedding matrix
        Z2: (n_samples, d2) target embedding matrix
        alpha: Ridge regularization strength (higher = more regularization)
        cv: Number of cross-validation folds. If None, uses train=test.

    Returns:
        float in (-inf, 1] where 1 = perfect linear mapping
        Negative values indicate the model performs worse than predicting the mean.

    Note:
        Uses Ridge regression for numerical stability, especially when
        d1 > n_samples or features are correlated.
    """
    Z1, Z2 = validate_inputs(Z1, Z2)

    model = Ridge(alpha=alpha)

    if cv is not None:
        # Cross-validated R²
        scores = cross_val_score(model, Z1, Z2, cv=cv, scoring="r2")
        return float(np.mean(scores))
    else:
        # Train on all data and evaluate on same data
        model.fit(Z1, Z2)
        return float(model.score(Z1, Z2))


def bidirectional_linear_r2(
    Z1: NDArray[np.floating],
    Z2: NDArray[np.floating],
    alpha: float = 1.0,
    cv: int | None = None,
) -> float:
    """
    Bidirectional linear R²: average of Z1→Z2 and Z2→Z1 mappings.

    A symmetric version of linear_r2 that measures how well each
    representation can predict the other.

    Args:
        Z1: (n_samples, d1) embedding matrix
        Z2: (n_samples, d2) embedding matrix
        alpha: Ridge regularization strength
        cv: Number of cross-validation folds

    Returns:
        float in (-inf, 1] where 1 = perfect bidirectional mapping
    """
    r2_forward = linear_r2(Z1, Z2, alpha=alpha, cv=cv)
    r2_backward = linear_r2(Z2, Z1, alpha=alpha, cv=cv)
    return float((r2_forward + r2_backward) / 2)
