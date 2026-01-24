"""Tests for metrics.regression module (Linear R²)."""

import numpy as np
import pytest

from pu.metrics.regression import linear_r2, bidirectional_linear_r2


class TestLinearR2:
    def test_identical_embeddings_perfect_score(self):
        Z = np.random.randn(100, 64)
        score = linear_r2(Z, Z)
        # Ridge regularization prevents exact 1.0
        np.testing.assert_allclose(score, 1.0, atol=1e-3)

    def test_linear_transformation_perfect_score(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 64)
        A = np.random.randn(64, 64)
        Z2 = Z1 @ A
        score = linear_r2(Z1, Z2)
        # Ridge regularization prevents exact 1.0
        np.testing.assert_allclose(score, 1.0, atol=1e-3)

    def test_random_embeddings_lower_than_linear(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 64)
        Z2 = np.random.randn(100, 64)
        score_random = linear_r2(Z1, Z2)
        # Linear transformation should have higher R² than random
        A = np.random.randn(64, 64)
        score_linear = linear_r2(Z1, Z1 @ A)
        assert score_linear > score_random

    def test_different_dimensions(self):
        Z1 = np.random.randn(100, 32)
        Z2 = np.random.randn(100, 64)
        score = linear_r2(Z1, Z2)
        assert score <= 1.0

    def test_alpha_parameter(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 64)
        Z2 = np.random.randn(100, 64)
        score_low = linear_r2(Z1, Z2, alpha=0.01)
        score_high = linear_r2(Z1, Z2, alpha=100.0)
        # Both should be valid
        assert score_low <= 1.0
        assert score_high <= 1.0

    def test_cross_validation(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 32)
        Z2 = np.random.randn(100, 32)
        score_cv = linear_r2(Z1, Z2, cv=5)
        score_no_cv = linear_r2(Z1, Z2, cv=None)
        # CV score should typically be lower (more honest)
        assert score_cv <= score_no_cv + 0.5

    def test_noisy_linear_relation(self):
        np.random.seed(42)
        Z1 = np.random.randn(200, 64)
        A = np.random.randn(64, 32)
        Z2 = Z1 @ A + 0.1 * np.random.randn(200, 32)
        score = linear_r2(Z1, Z2)
        # Should be high but not perfect due to noise
        assert score > 0.8

    def test_asymmetric(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 32)
        Z2 = np.random.randn(100, 64)
        score_forward = linear_r2(Z1, Z2)
        score_backward = linear_r2(Z2, Z1)
        # Different dimensions mean different R² scores
        assert score_forward != score_backward


class TestBidirectionalLinearR2:
    def test_identical_embeddings_perfect_score(self):
        Z = np.random.randn(100, 64)
        score = bidirectional_linear_r2(Z, Z)
        # Ridge regularization prevents exact 1.0
        np.testing.assert_allclose(score, 1.0, atol=1e-3)

    def test_symmetric(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 32)
        Z2 = np.random.randn(100, 64)
        score1 = bidirectional_linear_r2(Z1, Z2)
        score2 = bidirectional_linear_r2(Z2, Z1)
        np.testing.assert_allclose(score1, score2, atol=1e-10)

    def test_random_embeddings(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 64)
        Z2 = np.random.randn(100, 64)
        score = bidirectional_linear_r2(Z1, Z2)
        assert score <= 1.0

    def test_linear_transformation(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 64)
        A = np.random.randn(64, 64)
        Z2 = Z1 @ A
        score = bidirectional_linear_r2(Z1, Z2)
        # Should be very high for invertible linear transformation
        assert score > 0.9

    def test_alpha_parameter(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 32)
        Z2 = np.random.randn(100, 32)
        score = bidirectional_linear_r2(Z1, Z2, alpha=10.0)
        assert score <= 1.0

    def test_cross_validation(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 32)
        Z2 = np.random.randn(100, 32)
        score = bidirectional_linear_r2(Z1, Z2, cv=3)
        assert score <= 1.0
