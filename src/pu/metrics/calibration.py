"""
Calibrated similarity using the calibrated_similarity package.

Wraps existing metrics with permutation-based calibration to produce
calibrated scores, p-values, and significance thresholds.

Requires: pip install calibrated_similarity
"""

import numpy as np
from numpy.typing import NDArray
from typing import Callable

import torch

from pu.metrics._base import validate_inputs


def calibrate(
    Z1: NDArray[np.floating],
    Z2: NDArray[np.floating],
    similarity_fn: Callable,
    n_permutations: int = 1000,
    seed: int | None = None,
) -> dict[str, float]:
    """
    Calibrate a similarity metric using permutation testing.

    Args:
        Z1: (n_samples, d1) embedding matrix
        Z2: (n_samples, d2) embedding matrix
        similarity_fn: Any callable (Z1, Z2) -> float (numpy inputs).
        n_permutations: Number of permutations for null distribution.
        seed: Random seed for reproducibility.

    Returns:
        Dict with calibrated_score, p_value, and threshold.

    Example:
        >>> from pu.metrics import cka, calibrate
        >>> calibrate(Z1, Z2, cka)
    """
    try:
        from calibrated_similarity import calibrate as cs_calibrate
    except ImportError as e:
        raise ImportError(
            "calibrated_similarity is not installed. "
            "Install with: pip install calibrated_similarity"
        ) from e

    Z1, Z2 = validate_inputs(Z1, Z2)

    def torch_fn(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        score = similarity_fn(
            X.detach().cpu().numpy().astype(np.float64),
            Y.detach().cpu().numpy().astype(np.float64),
        )
        return torch.tensor(float(score))

    X = torch.from_numpy(np.asarray(Z1, dtype=np.float32))
    Y = torch.from_numpy(np.asarray(Z2, dtype=np.float32))

    kwargs = {"K": n_permutations}
    if seed is not None:
        kwargs["seed"] = seed

    calibrated_score, p_value, threshold = cs_calibrate(X, Y, torch_fn, **kwargs)

    return {
        "calibrated_score": float(calibrated_score),
        "p_value": float(p_value),
        "threshold": float(threshold),
    }
