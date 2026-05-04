"""Tests for metrics.cca module (SVCCA and PWCCA)."""

import numpy as np
import pytest

from pu.metrics.cca import svcca, pwcca


class TestSVCCA:
    def test_identical_embeddings_perfect_correlation(self):
        Z = np.random.randn(100, 64)
        score = svcca(Z, Z)
        np.testing.assert_allclose(score, 1.0, atol=1e-6)

    def test_linear_transformation_high_correlation(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 64)
        # Linear transformation should preserve most correlation
        A = np.random.randn(64, 64)
        Z2 = Z1 @ A
        score = svcca(Z1, Z2)
        assert score > 0.8

    def test_uncorrelated_embeddings_lower_than_correlated(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 64)
        Z2 = np.random.randn(100, 64)
        score_random = svcca(Z1, Z2)
        # Linear transformation should have higher score than random
        A = np.random.randn(64, 64)
        score_linear = svcca(Z1, Z1 @ A)
        assert score_linear > score_random

    def test_different_dimensions(self):
        Z1 = np.random.randn(100, 32)
        Z2 = np.random.randn(100, 64)
        score = svcca(Z1, Z2)
        assert 0 <= score <= 1

    def test_threshold_parameter(self):
        Z1 = np.random.randn(100, 64)
        Z2 = np.random.randn(100, 64)
        # Different thresholds should give different results
        score_low = svcca(Z1, Z2, threshold=0.5)
        score_high = svcca(Z1, Z2, threshold=0.99)
        # Both should be valid
        assert 0 <= score_low <= 1
        assert 0 <= score_high <= 1

    def test_small_matrix(self):
        Z1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        Z2 = np.array([[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]])
        score = svcca(Z1, Z2)
        assert 0 <= score <= 1

    def test_symmetric(self):
        np.random.seed(42)
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 64)
        score1 = svcca(Z1, Z2)
        score2 = svcca(Z2, Z1)
        # SVCCA should be approximately symmetric
        np.testing.assert_allclose(score1, score2, atol=0.1)

    def test_orthogonal_transformation(self):
        Z1 = np.random.randn(100, 64)
        Q, _ = np.linalg.qr(np.random.randn(64, 64))
        Z2 = Z1 @ Q
        score = svcca(Z1, Z2)
        # Orthogonal transformation should preserve CCA correlations
        np.testing.assert_allclose(score, 1.0, atol=1e-4)


class TestPWCCA:
    def test_identical_embeddings_perfect_correlation(self):
        Z = np.random.randn(100, 64)
        score = pwcca(Z, Z)
        np.testing.assert_allclose(score, 1.0, atol=1e-6)

    def test_linear_transformation_high_correlation(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 64)
        A = np.random.randn(64, 64)
        Z2 = Z1 @ A
        score = pwcca(Z1, Z2)
        # Linear transformation should give high correlation
        assert score > 0.7

    def test_uncorrelated_embeddings_lower_than_correlated(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 64)
        Z2 = np.random.randn(100, 64)
        score_random = pwcca(Z1, Z2)
        # Linear transformation should have higher score than random
        A = np.random.randn(64, 64)
        score_linear = pwcca(Z1, Z1 @ A)
        assert score_linear > score_random

    def test_different_dimensions(self):
        Z1 = np.random.randn(100, 32)
        Z2 = np.random.randn(100, 64)
        score = pwcca(Z1, Z2)
        assert 0 <= score <= 1

    def test_threshold_parameter(self):
        Z1 = np.random.randn(100, 64)
        Z2 = np.random.randn(100, 64)
        score_low = pwcca(Z1, Z2, threshold=0.5)
        score_high = pwcca(Z1, Z2, threshold=0.99)
        assert 0 <= score_low <= 1
        assert 0 <= score_high <= 1

    def test_small_matrix(self):
        Z1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        Z2 = np.array([[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]])
        score = pwcca(Z1, Z2)
        assert 0 <= score <= 1

    def test_pwcca_vs_svcca(self):
        # PWCCA and SVCCA should give similar but not identical results
        np.random.seed(42)
        Z1 = np.random.randn(100, 64)
        Z2 = np.random.randn(100, 64)
        svcca_score = svcca(Z1, Z2)
        pwcca_score = pwcca(Z1, Z2)
        # Both should be in valid range
        assert 0 <= svcca_score <= 1
        assert 0 <= pwcca_score <= 1

    def test_orthogonal_transformation(self):
        Z1 = np.random.randn(100, 64)
        Q, _ = np.linalg.qr(np.random.randn(64, 64))
        Z2 = Z1 @ Q
        score = pwcca(Z1, Z2)
        np.testing.assert_allclose(score, 1.0, atol=1e-4)
