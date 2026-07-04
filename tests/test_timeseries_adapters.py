"""Tests for the time-series foundation-model adapters.

Two layers of coverage:

1. **Stub tests** (always run, CPU-only, no network): verify the *API contract*
   the adapters expose to the layer-wise MKNN pipeline. They bypass model
   downloads with tiny ``nn.Module`` stubs, exactly like
   ``test_hf_clip_and_llava_ov.py``.

2. **Live smoke test** (opt-in): only runs when ``chronos-forecasting`` is
   installed *and* ``PU_RUN_SLOW=1``; downloads ``chronos-t5-tiny`` and checks
   the real forward path + determinism.
"""
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from pu.metrics.neighbors import mknn
from pu.models import get_adapter
from pu.models.base import set_deterministic
from pu.models.timeseries import (
    CHRONOS_AVAILABLE,
    ChronosAdapter,
    MomentAdapter,
    PreprocessTimeSeries,
    fit_length,
)

# ── registry ─────────────────────────────────────────────────────────────

def test_registry_resolves_ts_aliases():
    assert get_adapter("chronos") is ChronosAdapter
    assert get_adapter("moment") is MomentAdapter


# ── fit_length ───────────────────────────────────────────────────────────

def test_fit_length_pads_short_series_and_masks_padding():
    x = torch.ones(2, 1, 10)
    out, mask = fit_length(x, target=16)
    assert out.shape == (2, 1, 16)
    assert mask.shape == (2, 16)
    assert mask[:, :10].sum() == 2 * 10        # real observations flagged 1
    assert mask[:, 10:].sum() == 0             # padding flagged 0
    assert torch.equal(out[:, :, 10:], torch.zeros(2, 1, 6))


def test_fit_length_truncates_long_series():
    x = torch.arange(20, dtype=torch.float32).reshape(1, 1, 20)
    out, mask = fit_length(x, target=8)
    assert out.shape == (1, 1, 8)
    assert mask.sum() == 8
    assert torch.equal(out[0, 0], torch.arange(8, dtype=torch.float32))


def test_fit_length_rejects_non_3d():
    with pytest.raises(ValueError, match="B, C, T"):
        fit_length(torch.ones(4, 10), target=16)


# ── preprocessor ─────────────────────────────────────────────────────────

def test_preprocess_promotes_univariate_to_channel():
    pre = PreprocessTimeSeries(modes=["lc"], series_key="flux")
    out = pre({"flux": [1.0, 2.0, 3.0]})
    assert out["lc"].shape == (1, 3)           # (C, T)
    assert out["lc"].dtype == np.float32


def test_preprocess_falls_back_to_mode_key():
    pre = PreprocessTimeSeries(modes=["lc"])   # default series_key='lightcurve'
    out = pre({"lc": np.zeros((2, 5), dtype=np.float32)})
    assert out["lc"].shape == (2, 5)           # already (C, T), left as-is


# ── stub models exercising the layer-wise contract ───────────────────────

class _StubEncoder(nn.Module):
    """Encoder with a ``block`` ModuleList so 'blocks' granularity finds layers."""

    def __init__(self, d: int = 8, n_blocks: int = 3):
        super().__init__()
        self.block = nn.ModuleList([nn.Linear(d, d) for _ in range(n_blocks)])

    def forward(self, x):  # x: (B, S, d)
        for blk in self.block:
            x = blk(x)
        return x


class _StubChronosPipeline:
    """Minimal stand-in for BaseChronosPipeline: embed() runs the encoder."""

    def __init__(self, encoder: _StubEncoder, d: int = 8):
        self._enc = encoder
        self.d = d

    def embed(self, context):                  # context: (B, T)
        B, T = context.shape
        x = context[..., None].repeat(1, 1, self.d).float()  # (B, T, d)
        x = torch.cat([x, x[:, :1]], dim=1)                  # + EOS token
        return self._enc(x), None              # (B, T+1, d)


def _make_chronos_stub() -> ChronosAdapter:
    ad = ChronosAdapter.__new__(ChronosAdapter)
    ad.alias = "chronos"
    ad.device = "cpu"
    enc = _StubEncoder(d=8, n_blocks=3)
    ad.model = enc
    ad.pipeline = _StubChronosPipeline(enc, d=8)
    return ad


class _StubMomentModel(nn.Module):
    def __init__(self, d: int = 8, n_blocks: int = 3):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.block = nn.ModuleList([nn.Linear(d, d) for _ in range(n_blocks)])
        self.d = d

    def forward(self, x_enc=None, input_mask=None):
        h = x_enc.mean(dim=2).mean(dim=1, keepdim=True).repeat(1, self.d)  # (B, d)
        for blk in self.encoder.block:
            h = blk(h)
        return SimpleNamespace(embeddings=h)


def _make_moment_stub() -> MomentAdapter:
    ad = MomentAdapter.__new__(MomentAdapter)
    ad.alias = "moment"
    ad.device = "cpu"
    ad.model = _StubMomentModel(d=8, n_blocks=3)
    return ad


@pytest.mark.parametrize("make_stub", [_make_chronos_stub, _make_moment_stub])
def test_embed_for_mode_returns_batch_by_dim(make_stub):
    ad = make_stub()
    batch = {"lc": torch.randn(5, 1, 32)}
    emb = ad.embed_for_mode(batch, "lc")
    assert emb.dim() == 2 and emb.shape[0] == 5   # (B, D)
    assert emb.dtype == torch.float32


@pytest.mark.parametrize("make_stub", [_make_chronos_stub, _make_moment_stub])
def test_layerwise_returns_ordered_bd_tensors(make_stub):
    ad = make_stub()
    batch = {"lc": torch.randn(5, 1, 32)}
    layers = ad.embed_all_layers_for_mode(batch, "lc", granularity="blocks")
    assert len(layers) >= 3                        # at least the 3 stub blocks
    for name, z in layers.items():
        assert z.dim() == 2 and z.shape[0] == 5, name
    # get_layer_names order is a superset ending in the model's final key
    assert ad.supports_layerwise()


@pytest.mark.parametrize("make_stub", [_make_chronos_stub, _make_moment_stub])
def test_layerwise_is_deterministic(make_stub):
    """Same input twice -> identical activations (drift == 0, self-MKNN == 1)."""
    set_deterministic(0)
    ad = make_stub()
    batch = {"lc": torch.randn(8, 1, 32)}
    a = ad.embed_all_layers_for_mode(batch, "lc")
    b = ad.embed_all_layers_for_mode(batch, "lc")
    assert set(a) == set(b)
    for name in a:
        za, zb = a[name].numpy(), b[name].numpy()
        assert np.max(np.abs(za - zb)) == 0.0, name
        assert mknn(za, zb, k=3) == 1.0, name


# ── MKNN gotchas the contract warns about ────────────────────────────────

def test_mknn_is_row_order_sensitive():
    """Row i of both matrices must be the same series; shuffling one breaks it."""
    rng = np.random.default_rng(0)
    Z1 = rng.standard_normal((60, 8))
    Z2 = Z1 + 0.01 * rng.standard_normal((60, 8))
    aligned = mknn(Z1, Z2, k=5)
    shuffled = mknn(Z1, Z2[rng.permutation(60)], k=5)
    assert aligned > 0.8
    assert aligned > shuffled


def test_mknn_repeatable_within_run():
    rng = np.random.default_rng(1)
    Z1 = rng.standard_normal((80, 16))
    Z2 = rng.standard_normal((80, 16))
    assert mknn(Z1, Z2, k=10) == mknn(Z1, Z2, k=10)


# ── opt-in live smoke test ───────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.skipif(
    not (CHRONOS_AVAILABLE and os.environ.get("PU_RUN_SLOW") == "1"),
    reason="set PU_RUN_SLOW=1 and install chronos-forecasting to run",
)
def test_chronos_live_smoke():
    set_deterministic(0)
    ad = get_adapter("chronos")("amazon/chronos-t5-tiny", "tiny", alias="chronos")
    ad.load()
    batch = {"lc": torch.randn(8, 1, 64)}
    emb = ad.embed_for_mode(batch, "lc")
    assert emb.shape[0] == 8 and emb.dim() == 2
    layers1 = ad.embed_all_layers_for_mode(batch, "lc")
    layers2 = ad.embed_all_layers_for_mode(batch, "lc")
    for name in layers1:
        assert np.max(np.abs(layers1[name].numpy() - layers2[name].numpy())) == 0.0
