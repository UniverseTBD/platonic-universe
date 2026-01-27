"""Tests for metrics._base module."""

import numpy as np
import pytest

from pu.metrics._base import (
    validate_inputs,
    center,
    normalize_rows,
    gram_matrix,
    center_gram,
    rbf_kernel,
)


class TestValidateInputs:
    def test_valid_inputs(self):
        Z1 = np.random.randn(100, 64)
        Z2 = np.random.randn(100, 32)
        out1, out2 = validate_inputs(Z1, Z2)
        assert out1.shape == (100, 64)
        assert out2.shape == (100, 32)
        assert out1.dtype == np.float64
        assert out2.dtype == np.float64

    def test_converts_to_float64(self):
        Z1 = np.random.randn(10, 5).astype(np.float32)
        Z2 = np.random.randn(10, 5).astype(np.float32)
        out1, out2 = validate_inputs(Z1, Z2)
        assert out1.dtype == np.float64
        assert out2.dtype == np.float64

    def test_mismatched_samples_raises(self):
        Z1 = np.random.randn(100, 64)
        Z2 = np.random.randn(50, 64)
        with pytest.raises(ValueError, match="Number of samples must match"):
            validate_inputs(Z1, Z2)

    def test_require_same_dim(self):
        Z1 = np.random.randn(100, 64)
        Z2 = np.random.randn(100, 32)
        with pytest.raises(ValueError, match="Dimensions must match"):
            validate_inputs(Z1, Z2, require_same_dim=True)

    def test_require_same_dim_passes(self):
        Z1 = np.random.randn(100, 64)
        Z2 = np.random.randn(100, 64)
        out1, out2 = validate_inputs(Z1, Z2, require_same_dim=True)
        assert out1.shape == out2.shape

    def test_1d_array_raises(self):
        Z1 = np.random.randn(100)
        Z2 = np.random.randn(100)
        with pytest.raises(ValueError, match="must be 2D"):
            validate_inputs(Z1, Z2)

    def test_empty_array_raises(self):
        Z1 = np.random.randn(0, 64)
        Z2 = np.random.randn(0, 64)
        with pytest.raises(ValueError, match="at least one sample"):
            validate_inputs(Z1, Z2)


class TestCenter:
    def test_center_subtracts_mean(self):
        Z = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        centered = center(Z)
        # Check that column means are zero
        np.testing.assert_allclose(centered.mean(axis=0), [0.0, 0.0], atol=1e-10)

    def test_center_preserves_shape(self):
        Z = np.random.randn(50, 10)
        centered = center(Z)
        assert centered.shape == Z.shape


class TestNormalizeRows:
    def test_rows_have_unit_norm(self):
        Z = np.random.randn(50, 10)
        normalized = normalize_rows(Z)
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_allclose(norms, np.ones(50), atol=1e-10)

    def test_handles_zero_rows(self):
        Z = np.array([[0.0, 0.0], [1.0, 0.0]])
        normalized = normalize_rows(Z)
        # Zero row should remain zero (divided by 1.0 fallback)
        np.testing.assert_allclose(normalized[0], [0.0, 0.0])
        np.testing.assert_allclose(normalized[1], [1.0, 0.0])


class TestGramMatrix:
    def test_gram_matrix_shape(self):
        Z = np.random.randn(50, 10)
        K = gram_matrix(Z)
        assert K.shape == (50, 50)

    def test_gram_matrix_symmetric(self):
        Z = np.random.randn(50, 10)
        K = gram_matrix(Z)
        np.testing.assert_allclose(K, K.T)

    def test_gram_matrix_values(self):
        Z = np.array([[1.0, 0.0], [0.0, 1.0]])
        K = gram_matrix(Z)
        expected = np.array([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_allclose(K, expected)


class TestCenterGram:
    def test_centered_gram_row_col_sums_zero(self):
        Z = np.random.randn(50, 10)
        K = gram_matrix(Z)
        Kc = center_gram(K)
        # Row and column sums should be approximately zero
        np.testing.assert_allclose(Kc.sum(axis=0), np.zeros(50), atol=1e-10)
        np.testing.assert_allclose(Kc.sum(axis=1), np.zeros(50), atol=1e-10)

    def test_centered_gram_symmetric(self):
        Z = np.random.randn(50, 10)
        K = gram_matrix(Z)
        Kc = center_gram(K)
        np.testing.assert_allclose(Kc, Kc.T)


class TestRbfKernel:
    def test_rbf_kernel_shape(self):
        Z = np.random.randn(50, 10)
        K = rbf_kernel(Z)
        assert K.shape == (50, 50)

    def test_rbf_kernel_symmetric(self):
        Z = np.random.randn(50, 10)
        K = rbf_kernel(Z)
        np.testing.assert_allclose(K, K.T)

    def test_rbf_kernel_diagonal_ones(self):
        Z = np.random.randn(50, 10)
        K = rbf_kernel(Z)
        # Diagonal should be 1 (distance to self is 0, exp(0) = 1)
        np.testing.assert_allclose(np.diag(K), np.ones(50), atol=1e-10)

    def test_rbf_kernel_values_in_range(self):
        Z = np.random.randn(50, 10)
        K = rbf_kernel(Z)
        assert np.all(K >= 0)
        assert np.all(K <= 1)

    def test_rbf_kernel_custom_gamma(self):
        Z = np.random.randn(50, 10)
        K1 = rbf_kernel(Z, gamma=0.1)
        K2 = rbf_kernel(Z, gamma=1.0)
        # Higher gamma = faster decay = lower off-diagonal values
        assert K1.mean() > K2.mean()
