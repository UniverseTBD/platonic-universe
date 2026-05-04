"""Tests for metrics.geometric module (Procrustes, Cosine, Frechet)."""

import numpy as np
import pytest

from pu.metrics.geometric import procrustes, cosine_similarity, frechet


class TestProcrustes:
    def test_identical_embeddings_zero_distance(self):
        Z = np.random.randn(100, 64)
        dist = procrustes(Z, Z)
        np.testing.assert_allclose(dist, 0.0, atol=1e-10)

    def test_orthogonal_transformation_zero_distance(self):
        Z1 = np.random.randn(100, 64)
        Q, _ = np.linalg.qr(np.random.randn(64, 64))
        Z2 = Z1 @ Q
        dist = procrustes(Z1, Z2)
        np.testing.assert_allclose(dist, 0.0, atol=1e-6)

    def test_scaling_invariant(self):
        Z1 = np.random.randn(100, 64)
        Z2 = Z1 * 5.0
        dist = procrustes(Z1, Z2)
        np.testing.assert_allclose(dist, 0.0, atol=1e-10)

    def test_different_embeddings_positive_distance(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 64)
        Z2 = np.random.randn(100, 64)
        dist = procrustes(Z1, Z2)
        assert dist > 0

    def test_symmetric(self):
        np.random.seed(42)
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 32)
        dist1 = procrustes(Z1, Z2)
        dist2 = procrustes(Z2, Z1)
        np.testing.assert_allclose(dist1, dist2, atol=1e-10)

    def test_requires_same_dimension(self):
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 64)
        with pytest.raises(ValueError, match="Dimensions must match"):
            procrustes(Z1, Z2)

    def test_small_matrix(self):
        Z1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        Z2 = np.array([[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]])
        dist = procrustes(Z1, Z2)
        assert dist >= 0


class TestCosineSimilarity:
    def test_identical_embeddings_perfect_similarity(self):
        Z = np.random.randn(100, 64)
        sim = cosine_similarity(Z, Z)
        np.testing.assert_allclose(sim, 1.0, atol=1e-10)

    def test_opposite_embeddings_negative_similarity(self):
        Z = np.random.randn(100, 64)
        sim = cosine_similarity(Z, -Z)
        np.testing.assert_allclose(sim, -1.0, atol=1e-10)

    def test_orthogonal_embeddings_zero_similarity(self):
        # Create orthogonal vectors
        Z1 = np.zeros((10, 2))
        Z1[:, 0] = 1.0
        Z2 = np.zeros((10, 2))
        Z2[:, 1] = 1.0
        sim = cosine_similarity(Z1, Z2)
        np.testing.assert_allclose(sim, 0.0, atol=1e-10)

    def test_scaling_invariant(self):
        Z1 = np.random.randn(100, 64)
        Z2 = Z1 * 5.0
        sim = cosine_similarity(Z1, Z2)
        np.testing.assert_allclose(sim, 1.0, atol=1e-10)

    def test_random_embeddings_range(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 64)
        Z2 = np.random.randn(100, 64)
        sim = cosine_similarity(Z1, Z2)
        assert -1 <= sim <= 1

    def test_requires_same_dimension(self):
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 64)
        with pytest.raises(ValueError, match="Dimensions must match"):
            cosine_similarity(Z1, Z2)

    def test_symmetric(self):
        np.random.seed(42)
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 32)
        sim1 = cosine_similarity(Z1, Z2)
        sim2 = cosine_similarity(Z2, Z1)
        np.testing.assert_allclose(sim1, sim2, atol=1e-10)


class TestFrechet:
    def test_identical_embeddings_zero_distance(self):
        Z = np.random.randn(100, 64)
        dist = frechet(Z, Z)
        np.testing.assert_allclose(dist, 0.0, atol=1e-6)

    def test_shifted_distribution_positive_distance(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 64)
        Z2 = Z1 + 5.0  # Shift the mean
        dist = frechet(Z1, Z2)
        assert dist > 0

    def test_larger_shift_larger_distance(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 64)
        Z2_small = Z1 + 1.0
        Z2_large = Z1 + 5.0
        dist_small = frechet(Z1, Z2_small)
        dist_large = frechet(Z1, Z2_large)
        assert dist_large > dist_small

    def test_different_dimensions(self):
        Z1 = np.random.randn(100, 32)
        Z2 = np.random.randn(100, 64)
        dist = frechet(Z1, Z2)
        assert dist >= 0

    def test_symmetric(self):
        np.random.seed(42)
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 32)
        dist1 = frechet(Z1, Z2)
        dist2 = frechet(Z2, Z1)
        np.testing.assert_allclose(dist1, dist2, atol=1e-6)

    def test_1d_embeddings(self):
        Z1 = np.random.randn(100, 1)
        Z2 = np.random.randn(100, 1) + 2.0
        dist = frechet(Z1, Z2)
        assert dist >= 0

    def test_scaled_covariance(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 32)
        Z2 = Z1 * 2.0  # Different covariance
        dist = frechet(Z1, Z2)
        assert dist > 0
