"""Tests for the ``light-curve`` feature-extractor adapter.

Unlike TabPFN / MOMENT (gated or pinned), ``light-curve`` installs from PyPI,
needs no weights and runs on CPU, so the live path is exercised directly here
whenever the package is importable. The stub-independent tests cover the pure
input-handling logic (channel splitting, batching, sort-by-time) and always run.
"""
import importlib.util

import numpy as np
import pytest
import torch

from pu.models import get_adapter
from pu.models.lightcurve import (
    FEATURE_SETS,
    LIGHTCURVE_AVAILABLE,
    LightCurveAdapter,
    PreprocessLightCurve,
)

_HAS_LC = importlib.util.find_spec("light_curve") is not None


# --------------------------------------------------------------------------- #
# Registration + static contract (no light-curve needed)                      #
# --------------------------------------------------------------------------- #

def test_registry_resolves_lightcurve():
    assert get_adapter("lightcurve") is LightCurveAdapter


def test_lightcurve_is_not_layerwise():
    ad = LightCurveAdapter.__new__(LightCurveAdapter)
    assert ad.supports_layerwise() is False


def test_feature_sets_are_nested_and_growing():
    assert set(FEATURE_SETS) == {"cheap", "default", "full"}
    assert len(FEATURE_SETS["cheap"]) < len(FEATURE_SETS["default"]) < len(FEATURE_SETS["full"])
    # default extends cheap; full extends default (prefix property).
    assert FEATURE_SETS["default"][: len(FEATURE_SETS["cheap"])] == FEATURE_SETS["cheap"]
    assert FEATURE_SETS["full"][: len(FEATURE_SETS["default"])] == FEATURE_SETS["default"]


# --------------------------------------------------------------------------- #
# Channel-splitting logic (pure numpy, always runs)                           #
# --------------------------------------------------------------------------- #

def test_split_channels_one_channel_synthesizes_time_and_errors():
    m = np.array([2.0, 1.0, 3.0])
    t, mag, sigma = LightCurveAdapter._split_channels(m[np.newaxis, :])
    assert np.array_equal(t, [0, 1, 2])          # arange time
    assert np.array_equal(mag, [2.0, 1.0, 3.0])  # unchanged (already time-sorted)
    assert np.array_equal(sigma, [1.0, 1.0, 1.0])  # unit errors


def test_split_channels_sorts_by_time():
    series = np.array([[3.0, 1.0, 2.0],   # times out of order
                       [30.0, 10.0, 20.0]])  # magnitudes
    t, mag, sigma = LightCurveAdapter._split_channels(series)
    assert np.array_equal(t, [1.0, 2.0, 3.0])
    assert np.array_equal(mag, [10.0, 20.0, 30.0])  # carried along with time


def test_split_channels_three_channels_uses_supplied_sigma():
    series = np.array([[0.0, 1.0], [15.0, 16.0], [0.1, 0.2]])  # (time, mag, sigma)
    t, mag, sigma = LightCurveAdapter._split_channels(series)
    assert np.array_equal(sigma, [0.1, 0.2])


def test_batch_to_triples_promotes_2d_and_validates_rank():
    ad = LightCurveAdapter.__new__(LightCurveAdapter)
    triples = ad._batch_to_triples(np.random.RandomState(0).randn(4, 20))  # (B, T)
    assert len(triples) == 4 and len(triples[0]) == 3
    with pytest.raises(ValueError):
        ad._batch_to_triples(np.zeros((2, 3, 4, 5)))  # rank-4 rejected


def test_preprocess_reads_series_key_and_promotes_1d():
    pre = PreprocessLightCurve(modes=["lc"], series_key="lightcurve")
    out = pre({"lightcurve": [15.0, 15.1, 14.9]})
    assert out["lc"].shape == (1, 3)  # (T,) -> (1, T)


# --------------------------------------------------------------------------- #
# Live path (runs whenever light-curve is importable — no download, CPU-only) #
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not _HAS_LC, reason="pip install light-curve to run the live path")
def test_available_flag_matches_import():
    assert LIGHTCURVE_AVAILABLE is True


@pytest.mark.skipif(not _HAS_LC, reason="pip install light-curve to run the live path")
@pytest.mark.parametrize("preset,dim", [("cheap", 10), ("default", 23), ("full", 42)])
def test_embed_shape_matches_preset(preset, dim):
    ad = get_adapter("lightcurve")("light-curve", preset, alias="lightcurve")
    ad.load()
    assert len(ad.feature_names) == dim
    rng = np.random.RandomState(0)
    # (B, 3, T): time, magnitude, sigma
    B, T = 5, 120
    t = np.sort(rng.uniform(0, 100, (B, T)), axis=1)
    m = 15 + 0.3 * np.sin(t / 5.0) + rng.normal(0, 0.05, (B, T))
    s = np.full((B, T), 0.05)
    batch = {"lc": np.stack([t, m, s], axis=1)}
    emb = ad.embed_for_mode(batch, "lc")
    assert emb.shape == (B, dim)
    assert emb.dtype == torch.float32
    assert torch.isfinite(emb).all()


@pytest.mark.skipif(not _HAS_LC, reason="pip install light-curve to run the live path")
def test_embed_is_deterministic():
    ad = get_adapter("lightcurve")("light-curve", "default", alias="lightcurve")
    ad.load()
    rng = np.random.RandomState(1)
    batch = {"lc": rng.randn(4, 1, 64).astype(np.float32)}  # magnitude-only
    a = ad.embed_for_mode(batch, "lc")
    b = ad.embed_for_mode(batch, "lc")
    assert torch.equal(a, b)  # pure function of the input: bit-reproducible


@pytest.mark.skipif(not _HAS_LC, reason="pip install light-curve to run the live path")
def test_embed_handles_torch_and_unsorted_time():
    ad = get_adapter("lightcurve")("light-curve", "cheap", alias="lightcurve")
    ad.load()
    # times deliberately unsorted; adapter must sort before extraction.
    t = torch.tensor([[5.0, 1.0, 3.0, 2.0, 4.0]])
    m = torch.tensor([[15.5, 15.1, 15.3, 15.2, 15.4]])
    s = torch.full((1, 5), 0.05)
    emb = ad.embed_for_mode({"lc": torch.stack([t, m, s], dim=1)}, "lc")
    assert emb.shape == (1, 10) and torch.isfinite(emb).all()


@pytest.mark.skipif(not _HAS_LC, reason="pip install light-curve to run the live path")
def test_explicit_feature_list_overrides_preset():
    import light_curve as lc

    ad = get_adapter("lightcurve")(
        "light-curve", "default", alias="lightcurve",
        features=[lc.Amplitude(), lc.Mean(), lc.StandardDeviation()],
    )
    ad.load()
    assert ad.feature_names == ["amplitude", "mean", "standard_deviation"]
    emb = ad.embed_for_mode({"lc": np.random.RandomState(0).randn(3, 1, 40)}, "lc")
    assert emb.shape == (3, 3)
