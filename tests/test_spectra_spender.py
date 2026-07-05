"""Tests for the Spender spectrum-autoencoder adapter.

Stub-based (CPU, no download): verify the spectra adapter honours the same
layer-wise contract as the other unimodal adapters. Weights load via torch.hub
from an external host, so the live path is an opt-in smoke test.
"""
import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from pu.metrics.neighbors import mknn
from pu.models import get_adapter
from pu.models.base import set_deterministic
from pu.models.spender import (
    SPENDER_AVAILABLE,
    PreprocessSpender,
    SpenderAdapter,
)


def test_registry_resolves_spender():
    assert get_adapter("spender") is SpenderAdapter


def test_preprocess_reads_spectrum_flux():
    pre = PreprocessSpender(modes=["sdss"])
    out = pre({"spectrum": {"flux": [1.0, 2.0, 3.0, 4.0]}})
    assert out["sdss"].shape == (4,)
    assert out["sdss"].dtype == np.float32


def test_preprocess_skips_missing_mode():
    pre = PreprocessSpender(modes=["sdss"])
    assert pre({"image": [[0, 0]]}) == {}


class _StubSpenderEncoder(nn.Module):
    def __init__(self, L: int = 64, latent: int = 6, n_blocks: int = 3):
        super().__init__()
        dims = [L] + [latent] * n_blocks
        self.block = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(n_blocks)]
        )

    def forward(self, x):  # (B, L) -> (B, latent)
        for blk in self.block:
            x = blk(x)
        return x


class _StubSpenderModel(nn.Module):
    def __init__(self, L: int = 64, latent: int = 6):
        super().__init__()
        self.encoder = _StubSpenderEncoder(L, latent)

    def encode(self, spec):
        return self.encoder(spec)


def _make_spender_stub(L: int = 64, latent: int = 6) -> SpenderAdapter:
    ad = SpenderAdapter.__new__(SpenderAdapter)
    ad.alias = "spender"
    ad.device = "cpu"
    ad.n_wave = L
    ad.instrument = None
    ad.model = _StubSpenderModel(L, latent)
    return ad


def test_embed_for_mode_returns_latent():
    ad = _make_spender_stub(L=64, latent=6)
    emb = ad.embed_for_mode({"sdss": torch.randn(5, 64)}, "sdss")
    assert emb.shape == (5, 6)
    assert emb.dtype == torch.float32


def test_embed_for_mode_squeezes_trailing_channel():
    ad = _make_spender_stub(L=64, latent=6)
    emb = ad.embed_for_mode({"sdss": torch.randn(5, 64, 1)}, "sdss")  # (B, L, 1)
    assert emb.shape == (5, 6)


def test_layerwise_returns_ordered_bd_tensors_and_is_deterministic():
    set_deterministic(0)
    ad = _make_spender_stub(L=64, latent=6)
    batch = {"sdss": torch.randn(6, 64)}
    a = ad.embed_all_layers_for_mode(batch, "sdss")
    b = ad.embed_all_layers_for_mode(batch, "sdss")
    assert len(a) >= 3 and "latent" in a
    for name in a:
        assert a[name].dim() == 2 and a[name].shape[0] == 6, name
        assert np.max(np.abs(a[name].numpy() - b[name].numpy())) == 0.0, name
        assert mknn(a[name].numpy(), b[name].numpy(), k=3) == 1.0, name


@pytest.mark.slow
@pytest.mark.skipif(
    not (SPENDER_AVAILABLE and os.environ.get("PU_RUN_SLOW") == "1"),
    reason="set PU_RUN_SLOW=1 and install spender to run",
)
def test_spender_live_smoke():
    set_deterministic(0)
    ad = get_adapter("spender")("sdss_II", "sdss", alias="spender")
    ad.load()
    spec = torch.randn(4, ad.n_wave)
    emb = ad.embed_for_mode({"sdss": spec}, "sdss")
    assert emb.shape[0] == 4 and emb.dim() == 2
