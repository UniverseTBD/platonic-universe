"""Tests for metrics.spectral module (Tucker, Eigenspectrum, Riemannian)."""

import numpy as np
import pytest

from pu.metrics.spectral import tucker_congruence, eigenspectrum, riemannian


class TestTuckerCongruence:
    def test_identical_embeddings_perfect_congruence(self):
        Z = np.random.randn(100, 64)
        score = tucker_congruence(Z, Z)
        np.testing.assert_allclose(score, 1.0, atol=1e-10)

    def test_scaled_embeddings_perfect_congruence(self):
        Z1 = np.random.randn(100, 64)
        Z2 = Z1 * 3.0
        score = tucker_congruence(Z1, Z2)
        np.testing.assert_allclose(score, 1.0, atol=1e-10)

    def test_opposite_sign_perfect_congruence(self):
        # Tucker coefficient uses absolute value, so sign flip should give 1.0
        Z1 = np.random.randn(100, 64)
        Z2 = -Z1
        score = tucker_congruence(Z1, Z2)
        np.testing.assert_allclose(score, 1.0, atol=1e-10)

    def test_random_embeddings_in_range(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 64)
        Z2 = np.random.randn(100, 64)
        score = tucker_congruence(Z1, Z2)
        assert 0 <= score <= 1

    def test_requires_same_dimension(self):
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 64)
        with pytest.raises(ValueError, match="Dimensions must match"):
            tucker_congruence(Z1, Z2)

    def test_symmetric(self):
        np.random.seed(42)
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 32)
        score1 = tucker_congruence(Z1, Z2)
        score2 = tucker_congruence(Z2, Z1)
        np.testing.assert_allclose(score1, score2, atol=1e-10)

    def test_small_matrix(self):
        Z1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        Z2 = np.array([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]])  # Scaled version
        score = tucker_congruence(Z1, Z2)
        np.testing.assert_allclose(score, 1.0, atol=1e-10)


class TestEigenspectrum:
    def test_identical_embeddings_zero_distance(self):
        Z = np.random.randn(100, 64)
        dist = eigenspectrum(Z, Z)
        np.testing.assert_allclose(dist, 0.0, atol=1e-10)

    def test_scaled_embeddings_different_spectrum(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 64)
        Z2 = Z1 * 2.0
        # Without normalization, scaling changes eigenvalues
        dist_unnorm = eigenspectrum(Z1, Z2, normalize=False)
        assert dist_unnorm > 0
        # With normalization, relative spectrum is the same
        dist_norm = eigenspectrum(Z1, Z2, normalize=True)
        np.testing.assert_allclose(dist_norm, 0.0, atol=1e-10)

    def test_different_dimensions(self):
        Z1 = np.random.randn(100, 32)
        Z2 = np.random.randn(100, 64)
        dist = eigenspectrum(Z1, Z2)
        assert dist >= 0

    def test_random_embeddings_positive_distance(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 64)
        Z2 = np.random.randn(100, 64)
        dist = eigenspectrum(Z1, Z2)
        assert dist > 0

    def test_symmetric(self):
        np.random.seed(42)
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 32)
        dist1 = eigenspectrum(Z1, Z2)
        dist2 = eigenspectrum(Z2, Z1)
        np.testing.assert_allclose(dist1, dist2, atol=1e-10)

    def test_normalize_parameter(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 64)
        Z2 = np.random.randn(100, 64)
        dist_norm = eigenspectrum(Z1, Z2, normalize=True)
        dist_unnorm = eigenspectrum(Z1, Z2, normalize=False)
        # Both should be non-negative
        assert dist_norm >= 0
        assert dist_unnorm >= 0


class TestRiemannian:
    def test_identical_embeddings_zero_distance(self):
        Z = np.random.randn(100, 64)
        dist = riemannian(Z, Z)
        np.testing.assert_allclose(dist, 0.0, atol=1e-4)

    def test_log_euclidean_metric(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 32)
        Z2 = np.random.randn(100, 32)
        dist = riemannian(Z1, Z2, metric="log_euclidean")
        assert dist >= 0

    def test_affine_invariant_metric(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 32)
        Z2 = np.random.randn(100, 32)
        dist = riemannian(Z1, Z2, metric="affine_invariant")
        assert dist >= 0

    def test_requires_same_dimension(self):
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 64)
        with pytest.raises(ValueError, match="Dimensions must match"):
            riemannian(Z1, Z2)

    def test_unknown_metric_raises(self):
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 32)
        with pytest.raises(ValueError, match="Unknown metric"):
            riemannian(Z1, Z2, metric="unknown")

    def test_symmetric_log_euclidean(self):
        np.random.seed(42)
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 32)
        dist1 = riemannian(Z1, Z2, metric="log_euclidean")
        dist2 = riemannian(Z2, Z1, metric="log_euclidean")
        np.testing.assert_allclose(dist1, dist2, atol=1e-6)

    def test_symmetric_affine_invariant(self):
        np.random.seed(42)
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 32)
        dist1 = riemannian(Z1, Z2, metric="affine_invariant")
        dist2 = riemannian(Z2, Z1, metric="affine_invariant")
        np.testing.assert_allclose(dist1, dist2, atol=1e-6)

    def test_scaled_embeddings_positive_distance(self):
        Z1 = np.random.randn(100, 32)
        Z2 = Z1 * 2.0  # Different covariance
        dist = riemannian(Z1, Z2)
        assert dist > 0
