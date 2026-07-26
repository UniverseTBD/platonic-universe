"""Tests for metrics.neighbors module (MKNN, Jaccard, RSA)."""

import numpy as np
import pytest

from pu.metrics.neighbors import mknn, jaccard, rsa


class TestMKNN:
    def test_identical_embeddings_perfect_overlap(self):
        Z = np.random.randn(100, 64)
        score = mknn(Z, Z)
        np.testing.assert_allclose(score, 1.0, atol=1e-10)

    def test_random_embeddings_in_range(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 64)
        Z2 = np.random.randn(100, 64)
        score = mknn(Z1, Z2)
        assert 0 <= score <= 1

    def test_k_parameter(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 64)
        Z2 = np.random.randn(100, 64)
        score_k5 = mknn(Z1, Z2, k=5)
        score_k20 = mknn(Z1, Z2, k=20)
        # Both should be valid
        assert 0 <= score_k5 <= 1
        assert 0 <= score_k20 <= 1

    def test_different_dimensions(self):
        Z1 = np.random.randn(100, 32)
        Z2 = np.random.randn(100, 64)
        score = mknn(Z1, Z2)
        assert 0 <= score <= 1

    def test_symmetric(self):
        np.random.seed(42)
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 32)
        score1 = mknn(Z1, Z2)
        score2 = mknn(Z2, Z1)
        np.testing.assert_allclose(score1, score2, atol=1e-10)

    def test_similar_embeddings_high_overlap(self):
        Z1 = np.random.randn(100, 64)
        Z2 = Z1 + 0.01 * np.random.randn(100, 64)  # Small perturbation
        score = mknn(Z1, Z2)
        assert score > 0.8

    def test_small_k_large_sample(self):
        Z = np.random.randn(200, 32)
        score = mknn(Z, Z, k=3)
        np.testing.assert_allclose(score, 1.0, atol=1e-10)


class TestJaccard:
    def test_identical_embeddings_perfect_score(self):
        Z = np.random.randn(100, 64)
        score = jaccard(Z, Z)
        np.testing.assert_allclose(score, 1.0, atol=1e-10)

    def test_random_embeddings_in_range(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 64)
        Z2 = np.random.randn(100, 64)
        score = jaccard(Z1, Z2)
        assert 0 <= score <= 1

    def test_jaccard_le_mknn(self):
        # Jaccard should always be <= MKNN since it's stricter
        np.random.seed(42)
        Z1 = np.random.randn(100, 64)
        Z2 = np.random.randn(100, 64)
        mknn_score = mknn(Z1, Z2)
        jaccard_score = jaccard(Z1, Z2)
        assert jaccard_score <= mknn_score + 1e-10

    def test_k_parameter(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 64)
        Z2 = np.random.randn(100, 64)
        score_k5 = jaccard(Z1, Z2, k=5)
        score_k20 = jaccard(Z1, Z2, k=20)
        assert 0 <= score_k5 <= 1
        assert 0 <= score_k20 <= 1

    def test_symmetric(self):
        np.random.seed(42)
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 32)
        score1 = jaccard(Z1, Z2)
        score2 = jaccard(Z2, Z1)
        np.testing.assert_allclose(score1, score2, atol=1e-10)

    def test_different_dimensions(self):
        Z1 = np.random.randn(100, 32)
        Z2 = np.random.randn(100, 64)
        score = jaccard(Z1, Z2)
        assert 0 <= score <= 1


class TestRSA:
    def test_identical_embeddings_perfect_correlation(self):
        Z = np.random.randn(100, 64)
        score = rsa(Z, Z)
        np.testing.assert_allclose(score, 1.0, atol=1e-10)

    def test_random_embeddings_in_range(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 64)
        Z2 = np.random.randn(100, 64)
        score = rsa(Z1, Z2)
        assert -1 <= score <= 1

    def test_spearman_method(self):
        np.random.seed(42)
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 32)
        score = rsa(Z1, Z2, method="spearman")
        assert -1 <= score <= 1

    def test_pearson_method(self):
        np.random.seed(42)
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 32)
        score = rsa(Z1, Z2, method="pearson")
        assert -1 <= score <= 1

    def test_unknown_method_raises(self):
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 32)
        with pytest.raises(ValueError, match="Unknown method"):
            rsa(Z1, Z2, method="unknown")

    def test_different_metrics(self):
        np.random.seed(42)
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 32)
        score_cosine = rsa(Z1, Z2, metric="cosine")
        score_euclidean = rsa(Z1, Z2, metric="euclidean")
        assert -1 <= score_cosine <= 1
        assert -1 <= score_euclidean <= 1

    def test_symmetric(self):
        np.random.seed(42)
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 32)
        score1 = rsa(Z1, Z2)
        score2 = rsa(Z2, Z1)
        np.testing.assert_allclose(score1, score2, atol=1e-10)

    def test_different_dimensions(self):
        Z1 = np.random.randn(100, 32)
        Z2 = np.random.randn(100, 64)
        score = rsa(Z1, Z2)
        assert -1 <= score <= 1

    def test_linear_transformation_high_correlation(self):
        Z1 = np.random.randn(100, 64)
        A = np.random.randn(64, 64)
        Z2 = Z1 @ A
        score = rsa(Z1, Z2)
        # Linear transformations should mostly preserve distances
        assert score > 0.5
