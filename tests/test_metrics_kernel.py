"""Tests for metrics.kernel module (CKA and MMD)."""

import numpy as np
import pytest

from pu.metrics.kernel import cka, mmd


class TestCKA:
    def test_identical_embeddings_perfect_alignment(self):
        Z = np.random.randn(100, 64)
        score = cka(Z, Z)
        np.testing.assert_allclose(score, 1.0, atol=1e-10)

    def test_orthogonal_transformation_invariant(self):
        Z1 = np.random.randn(100, 64)
        # Apply random orthogonal transformation
        Q, _ = np.linalg.qr(np.random.randn(64, 64))
        Z2 = Z1 @ Q
        score = cka(Z1, Z2)
        np.testing.assert_allclose(score, 1.0, atol=1e-6)

    def test_isotropic_scaling_invariant(self):
        Z1 = np.random.randn(100, 64)
        Z2 = Z1 * 5.0  # Scale by constant
        score = cka(Z1, Z2)
        np.testing.assert_allclose(score, 1.0, atol=1e-10)

    def test_uncorrelated_embeddings_low_score(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 64)
        Z2 = np.random.randn(100, 64)
        score = cka(Z1, Z2)
        # Should be relatively low for uncorrelated random matrices
        # (but not necessarily near 0 for small sample sizes)
        assert score < 0.5

    def test_different_dimensions(self):
        Z1 = np.random.randn(100, 64)
        Z2 = np.random.randn(100, 128)
        score = cka(Z1, Z2)
        assert 0 <= score <= 1

    def test_linear_kernel(self):
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 32)
        score = cka(Z1, Z2, kernel="linear")
        assert 0 <= score <= 1

    def test_rbf_kernel(self):
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 32)
        score = cka(Z1, Z2, kernel="rbf")
        assert 0 <= score <= 1

    def test_rbf_kernel_custom_gamma(self):
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 32)
        score = cka(Z1, Z2, kernel="rbf", gamma=0.5)
        assert 0 <= score <= 1

    def test_unknown_kernel_raises(self):
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 32)
        with pytest.raises(ValueError, match="Unknown kernel"):
            cka(Z1, Z2, kernel="unknown")

    def test_symmetric(self):
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 64)
        score1 = cka(Z1, Z2)
        score2 = cka(Z2, Z1)
        np.testing.assert_allclose(score1, score2, atol=1e-10)

    def test_small_matrix(self):
        Z1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        Z2 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        score = cka(Z1, Z2)
        assert 0 <= score <= 1


class TestMMD:
    def test_identical_embeddings_zero_distance(self):
        Z = np.random.randn(100, 64)
        score = mmd(Z, Z)
        np.testing.assert_allclose(score, 0.0, atol=1e-10)

    def test_different_embeddings_positive_distance(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 64)
        Z2 = np.random.randn(100, 64) + 5.0  # Shifted distribution
        score = mmd(Z1, Z2)
        assert score > 0

    def test_linear_kernel(self):
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 32)
        score = mmd(Z1, Z2, kernel="linear")
        assert score >= 0

    def test_rbf_kernel(self):
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 32)
        score = mmd(Z1, Z2, kernel="rbf")
        assert score >= 0

    def test_polynomial_kernel(self):
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 32)
        score = mmd(Z1, Z2, kernel="polynomial")
        assert score >= 0

    def test_rbf_custom_gamma(self):
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 32)
        score = mmd(Z1, Z2, kernel="rbf", gamma=1.0)
        assert score >= 0

    def test_unknown_kernel_raises(self):
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 32)
        with pytest.raises(ValueError, match="Unknown kernel"):
            mmd(Z1, Z2, kernel="unknown")

    def test_requires_same_dimension(self):
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 64)
        with pytest.raises(ValueError, match="Dimensions must match"):
            mmd(Z1, Z2)

    def test_symmetric(self):
        np.random.seed(123)
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 32)
        score1 = mmd(Z1, Z2)
        score2 = mmd(Z2, Z1)
        np.testing.assert_allclose(score1, score2, atol=1e-10)

    def test_larger_shift_larger_mmd(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 32)
        Z2_small_shift = Z1 + 0.5
        Z2_large_shift = Z1 + 2.0
        mmd_small = mmd(Z1, Z2_small_shift)
        mmd_large = mmd(Z1, Z2_large_shift)
        assert mmd_large > mmd_small
