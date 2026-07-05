"""Tests for the Moirai adapter.

Stub tests (CPU, no network) cover the patchifying + encoder-hook wiring against
a stand-in MoiraiModule. The live smoke test (opt-in) needs uni2ts and downloads
real Salesforce/moirai weights.
"""
import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from pu.metrics.neighbors import mknn
from pu.models import get_adapter
from pu.models.base import set_deterministic
from pu.models.moirai import MOIRAI_AVAILABLE, MoiraiAdapter


def test_registry_resolves_moirai():
    assert get_adapter("moirai") is MoiraiAdapter


class _StubEncoder(nn.Module):
    """Encoder that consumes the packed target and returns (B, n_patch, d)."""

    def __init__(self, mps: int = 128, d: int = 16, n_layers: int = 3):
        super().__init__()
        self.in_proj = nn.Linear(mps, d)
        self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(n_layers)])

    def forward(self, target, *rest):
        h = self.in_proj(target)  # (B, n_patch, d)
        for layer in self.layers:
            h = layer(h)
        return h


class _StubMoiraiModule(nn.Module):
    def __init__(self, mps: int = 128, d: int = 16):
        super().__init__()
        self.patch_sizes = [8, 16, 32, 64, 128]
        self.encoder = _StubEncoder(mps=mps, d=d)

    def forward(self, target, observed_mask, sample_id, time_id,
                variate_id, prediction_mask, patch_size):
        # Mimics MoiraiModule: encoder runs inside forward; return value unused
        # by the adapter (it hooks the encoder output instead).
        return self.encoder(target)


def _make_moirai_stub(mps: int = 128, d: int = 16) -> MoiraiAdapter:
    ad = MoiraiAdapter.__new__(MoiraiAdapter)
    ad.alias = "moirai"
    ad.device = "cpu"
    ad.patch_size = 16
    ad.max_patch_size = mps
    ad.model = _StubMoiraiModule(mps=mps, d=d)
    return ad


def test_pack_produces_moirai_packed_shapes():
    ad = _make_moirai_stub(mps=128)
    target, observed, sample_id, time_id, variate_id, pred_mask, patch_size = ad._pack(
        torch.randn(3, 1, 128)
    )
    n_patch = 128 // 16
    assert target.shape == (3, n_patch, 128)          # (B, n_patch, max_patch_size)
    assert observed.dtype == torch.bool
    assert observed[..., :16].all() and not observed[..., 16:].any()
    assert time_id.tolist()[0] == list(range(n_patch))
    assert torch.equal(patch_size, torch.full((3, n_patch), 16))
    assert not pred_mask.any()                        # context-only (no forecast)


def test_pack_rejects_series_shorter_than_patch():
    ad = _make_moirai_stub()
    with pytest.raises(ValueError, match="shorter than patch size"):
        ad._pack(torch.randn(2, 1, 8))                # T=8 < patch_size=16


def test_embed_for_mode_returns_bd():
    ad = _make_moirai_stub(d=16)
    emb = ad.embed_for_mode({"lc": torch.randn(4, 1, 128)}, "lc")
    assert emb.shape == (4, 16)
    assert emb.dtype == torch.float32


def test_layerwise_is_deterministic_and_bd():
    set_deterministic(0)
    ad = _make_moirai_stub(d=16)
    batch = {"lc": torch.randn(5, 1, 128)}
    a = ad.embed_all_layers_for_mode(batch, "lc")
    b = ad.embed_all_layers_for_mode(batch, "lc")
    assert "encoder_out" in a and len(a) >= 3
    for name in a:
        assert a[name].dim() == 2 and a[name].shape[0] == 5, name
        assert np.max(np.abs(a[name].numpy() - b[name].numpy())) == 0.0, name
        assert mknn(a[name].numpy(), b[name].numpy(), k=2) == 1.0, name


@pytest.mark.slow
@pytest.mark.skipif(
    not (MOIRAI_AVAILABLE and os.environ.get("PU_RUN_SLOW") == "1"),
    reason="set PU_RUN_SLOW=1 and install uni2ts to run",
)
def test_moirai_live_smoke():
    set_deterministic(0)
    ad = get_adapter("moirai")("Salesforce/moirai-1.0-R-small", "small", alias="moirai")
    ad.load()
    emb = ad.embed_for_mode({"lc": torch.randn(4, 1, 128)}, "lc")
    assert emb.shape[0] == 4 and emb.dim() == 2
