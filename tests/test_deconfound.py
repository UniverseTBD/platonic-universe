"""Tests for pu.metrics.deconfound.residualize."""

import numpy as np

from pu.metrics.deconfound import confound_basis, residualize


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return 0.0 if d < 1e-12 else float((a @ b) / d)


def test_residual_orthogonal_to_linear_confound():
    rng = np.random.default_rng(0)
    n, d = 800, 16
    c = rng.normal(size=n)
    # embedding = linear-in-c signal + noise
    Z = np.outer(c, rng.normal(size=d)) + 0.1 * rng.normal(size=(n, d))
    Zr = residualize(Z, c[:, None], degree=1)
    # every residual column is uncorrelated with the confound
    for j in range(d):
        assert abs(_corr(Zr[:, j], c)) < 1e-6
    # and mean-zero (bias removed)
    assert np.allclose(Zr.mean(axis=0), 0, atol=1e-5)


def test_residual_removes_nonlinear_confound_with_degree3():
    rng = np.random.default_rng(1)
    n, d = 1500, 8
    c = rng.uniform(-2, 2, size=n)
    # embedding carries a quadratic + cubic function of the confound
    signal = (0.7 * c**2 - 0.3 * c**3)[:, None] * rng.normal(size=(1, d))
    Z = signal + 0.05 * rng.normal(size=(n, d))

    # linear residualization leaves nonlinear dependence behind ...
    Zr1 = residualize(Z, c[:, None], degree=1)
    r1 = np.mean([abs(_corr(Zr1[:, j], c**2)) for j in range(d)])
    # ... degree-3 removes it
    Zr3 = residualize(Z, c[:, None], degree=3)
    r3 = np.mean([abs(_corr(Zr3[:, j], c**2)) for j in range(d)])

    assert r3 < 1e-6
    assert r3 < r1


def test_probe_r2_collapses_after_residualization():
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score

    rng = np.random.default_rng(2)
    n, d = 2000, 32
    c = rng.normal(size=n)
    Z = np.outer(c, rng.normal(size=d)) + 0.2 * rng.normal(size=(n, d))

    r2_before = np.mean(cross_val_score(Ridge(), Z, c, cv=5, scoring="r2"))
    Zr = residualize(Z, c[:, None], degree=3)
    r2_after = np.mean(cross_val_score(Ridge(), Zr, c, cv=5, scoring="r2"))

    assert r2_before > 0.5
    assert r2_after < 0.05


def test_basis_shape():
    rng = np.random.default_rng(3)
    C = rng.normal(size=(50, 3))
    B = confound_basis(C, degree=3, include_bias=True)
    assert B.shape == (50, 1 + 3 * 3)
