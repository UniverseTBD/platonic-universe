"""Tests for metrics.io module (compare, list_metrics, etc.)."""

import numpy as np
import pytest

from pu.metrics.io import list_metrics, compare, METRICS_REGISTRY


class TestListMetrics:
    def test_returns_list(self):
        metrics = list_metrics()
        assert isinstance(metrics, list)
        assert len(metrics) > 0

    def test_contains_expected_metrics(self):
        metrics = list_metrics()
        expected = [
            "cka", "mmd", "procrustes", "cosine_similarity", "frechet",
            "svcca", "pwcca", "tucker_congruence", "eigenspectrum", "riemannian",
            "kl_divergence", "js_divergence", "mutual_information",
            "mknn", "jaccard", "rsa", "linear_r2",
        ]
        for m in expected:
            assert m in metrics


class TestCompare:
    def test_single_metric(self):
        np.random.seed(42)
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 32)
        results = compare(Z1, Z2, metrics=["cka"])
        assert "cka" in results
        assert isinstance(results["cka"], float)

    def test_multiple_metrics(self):
        np.random.seed(42)
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 32)
        results = compare(Z1, Z2, metrics=["cka", "mknn", "procrustes"])
        assert "cka" in results
        assert "mknn" in results
        assert "procrustes" in results

    def test_all_metrics(self):
        np.random.seed(42)
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 32)
        results = compare(Z1, Z2, metrics=["all"])
        # Should have all metrics
        for metric in list_metrics():
            assert metric in results

    def test_none_metrics_means_all(self):
        np.random.seed(42)
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 32)
        results = compare(Z1, Z2, metrics=None)
        for metric in list_metrics():
            assert metric in results

    def test_metric_specific_kwargs(self):
        np.random.seed(42)
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 32)
        results = compare(Z1, Z2, metrics=["mknn"], mknn__k=5)
        assert "mknn" in results
        assert isinstance(results["mknn"], float)

    def test_unknown_metric_raises(self):
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 32)
        with pytest.raises(ValueError, match="Unknown metric"):
            compare(Z1, Z2, metrics=["nonexistent_metric"])

    def test_handles_dimension_mismatch_gracefully(self):
        # Some metrics require same dimensions
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 64)
        # This should not raise, but some metrics will return None
        results = compare(Z1, Z2, metrics=["cka", "procrustes"])
        assert "cka" in results
        # CKA works with different dimensions
        assert results["cka"] is not None
        # Procrustes requires same dimensions, should be None
        assert results["procrustes"] is None

    def test_identical_embeddings(self):
        Z = np.random.randn(50, 32)
        results = compare(Z, Z, metrics=["cka", "mknn", "cosine_similarity"])
        # Identical embeddings should give perfect scores
        np.testing.assert_allclose(results["cka"], 1.0, atol=1e-6)
        np.testing.assert_allclose(results["mknn"], 1.0, atol=1e-6)
        np.testing.assert_allclose(results["cosine_similarity"], 1.0, atol=1e-6)


class TestMetricsRegistry:
    def test_all_metrics_callable(self):
        for name, fn in METRICS_REGISTRY.items():
            assert callable(fn), f"{name} is not callable"

    def test_all_metrics_accept_two_arrays(self):
        np.random.seed(42)
        Z1 = np.random.randn(50, 32)
        Z2 = np.random.randn(50, 32)
        for name, fn in METRICS_REGISTRY.items():
            try:
                result = fn(Z1, Z2)
                assert isinstance(result, float), f"{name} did not return float"
            except Exception as e:
                pytest.fail(f"{name} failed with: {e}")
