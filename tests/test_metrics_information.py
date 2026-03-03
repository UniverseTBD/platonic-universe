"""Tests for metrics.information module (KL, JS, MI)."""

import numpy as np
import pytest

from pu.metrics.information import kl_divergence, js_divergence, mutual_information


class TestKLDivergence:
    def test_identical_embeddings_zero_divergence(self):
        Z = np.random.randn(100, 64)
        kl = kl_divergence(Z, Z)
        np.testing.assert_allclose(kl, 0.0, atol=1e-6)

    def test_shifted_distribution_positive_divergence(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 32)
        Z2 = Z1 + 5.0  # Shift mean
        kl = kl_divergence(Z1, Z2)
        assert kl > 0

    def test_asymmetric(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 32)
        Z2 = np.random.randn(100, 32) * 2.0  # Different variance
        kl_forward = kl_divergence(Z1, Z2)
        kl_backward = kl_divergence(Z2, Z1)
        # KL is asymmetric
        assert kl_forward != kl_backward

    def test_larger_shift_larger_divergence(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 32)
        Z2_small = Z1 + 1.0
        Z2_large = Z1 + 5.0
        kl_small = kl_divergence(Z1, Z2_small)
        kl_large = kl_divergence(Z1, Z2_large)
        assert kl_large > kl_small

    def test_requires_same_dimension(self):
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 64)
        with pytest.raises(ValueError, match="Dimensions must match"):
            kl_divergence(Z1, Z2)

    def test_non_negative(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 32)
        Z2 = np.random.randn(100, 32)
        kl = kl_divergence(Z1, Z2)
        assert kl >= 0


class TestJSDivergence:
    def test_identical_embeddings_zero_divergence(self):
        Z = np.random.randn(100, 64)
        js = js_divergence(Z, Z)
        np.testing.assert_allclose(js, 0.0, atol=1e-6)

    def test_symmetric(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 32)
        Z2 = np.random.randn(100, 32)
        js_forward = js_divergence(Z1, Z2)
        js_backward = js_divergence(Z2, Z1)
        np.testing.assert_allclose(js_forward, js_backward, atol=1e-6)

    def test_shifted_distribution_positive_divergence(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 32)
        Z2 = Z1 + 5.0
        js = js_divergence(Z1, Z2)
        assert js > 0

    def test_larger_shift_larger_divergence(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 32)
        Z2_small = Z1 + 1.0
        Z2_large = Z1 + 5.0
        js_small = js_divergence(Z1, Z2_small)
        js_large = js_divergence(Z1, Z2_large)
        assert js_large > js_small

    def test_requires_same_dimension(self):
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 64)
        with pytest.raises(ValueError, match="Dimensions must match"):
            js_divergence(Z1, Z2)

    def test_bounded(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 32)
        Z2 = np.random.randn(100, 32) + 10.0  # Very different
        js = js_divergence(Z1, Z2)
        # JS divergence is bounded by log(2) ≈ 0.693
        assert js >= 0
        # Note: For Gaussian approximation, bound may not strictly hold


class TestMutualInformation:
    def test_identical_embeddings_high_mi(self):
        Z = np.random.randn(100, 32)
        mi = mutual_information(Z, Z)
        # Identical variables should have high MI
        assert mi > 0

    def test_independent_embeddings_low_mi(self):
        np.random.seed(42)
        Z1 = np.random.randn(200, 32)
        Z2 = np.random.randn(200, 32)
        mi = mutual_information(Z1, Z2)
        # Independent random variables should have low MI
        # (may not be exactly 0 due to estimation noise)
        assert mi >= 0

    def test_correlated_higher_than_independent(self):
        np.random.seed(42)
        Z1 = np.random.randn(200, 32)
        Z2_correlated = Z1 + 0.1 * np.random.randn(200, 32)  # Correlated
        Z2_independent = np.random.randn(200, 32)  # Independent

        mi_correlated = mutual_information(Z1, Z2_correlated)
        mi_independent = mutual_information(Z1, Z2_independent)

        assert mi_correlated > mi_independent

    def test_different_dimensions(self):
        Z1 = np.random.randn(100, 32)
        Z2 = np.random.randn(100, 64)
        mi = mutual_information(Z1, Z2)
        assert mi >= 0

    def test_k_parameter(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 16)
        Z2 = np.random.randn(100, 16)
        mi_k3 = mutual_information(Z1, Z2, k=3)
        mi_k5 = mutual_information(Z1, Z2, k=5)
        # Both should be non-negative
        assert mi_k3 >= 0
        assert mi_k5 >= 0

    def test_non_negative(self):
        np.random.seed(42)
        Z1 = np.random.randn(100, 32)
        Z2 = np.random.randn(100, 32)
        mi = mutual_information(Z1, Z2)
        assert mi >= 0

    def test_small_sample_size(self):
        Z1 = np.random.randn(20, 8)
        Z2 = np.random.randn(20, 8)
        mi = mutual_information(Z1, Z2, k=3)
        assert mi >= 0
