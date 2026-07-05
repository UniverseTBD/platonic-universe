"""Tests for the PatchTST / PatchTSMixer adapters.

Stub tests (CPU, no network) verify the channel/time fitting, 4D pooling, and
layer-wise contract. A live smoke test (opt-in) loads the real Granite
checkpoints, which download from HuggingFace.
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
from pu.models.patchts import PatchTSAdapter


def test_registry_resolves_patchts_aliases():
    assert get_adapter("patchtst") is PatchTSAdapter
    assert get_adapter("patchtsmixer") is PatchTSAdapter


class _StubPatchModel(nn.Module):
    def __init__(self, ctx: int = 32, ch: int = 3, d: int = 8, patches: int = 4, n_blocks: int = 3):
        super().__init__()
        self.config = SimpleNamespace(context_length=ctx, num_input_channels=ch)
        self.encoder = nn.Module()
        self.encoder.block = nn.ModuleList([nn.Linear(d, d) for _ in range(n_blocks)])
        self.d, self.patches = d, patches

    def forward(self, past_values=None):
        # past_values: (B, ctx, ch) -> last_hidden_state (B, ch, patches, d)
        base = past_values.mean(dim=1)                      # (B, ch)
        h = base[..., None, None].repeat(1, 1, self.patches, self.d)
        for blk in self.encoder.block:
            h = blk(h)
        return SimpleNamespace(last_hidden_state=h)


def _make_patch_stub(alias: str = "patchtst", ctx: int = 32, ch: int = 3, d: int = 8) -> PatchTSAdapter:
    ad = PatchTSAdapter.__new__(PatchTSAdapter)
    ad.alias = alias
    ad.device = "cpu"
    ad.context_length = ctx
    ad.num_channels = ch
    ad.model = _StubPatchModel(ctx=ctx, ch=ch, d=d)
    return ad


def test_prepare_tiles_univariate_to_model_channels_and_fits_time():
    ad = _make_patch_stub(ctx=32, ch=3)
    x = ad._prepare(torch.randn(5, 1, 20))     # univariate, short
    assert x.shape == (5, 32, 3)               # (B, context_length, num_channels), channels-last


def test_prepare_truncates_extra_channels_and_long_time():
    ad = _make_patch_stub(ctx=16, ch=2)
    x = ad._prepare(torch.randn(4, 5, 40))     # 5 channels, long
    assert x.shape == (4, 16, 2)


def test_pool_reduces_4d_hidden_state_to_bd():
    ad = _make_patch_stub(d=8)
    assert ad._pool(torch.randn(6, 3, 4, 8)).shape == (6, 8)   # (B,C,P,d)->(B,d)
    assert ad._pool(torch.randn(6, 4, 8)).shape == (6, 8)      # (B,P,d)->(B,d)


def test_embed_for_mode_returns_bd():
    ad = _make_patch_stub(d=8)
    emb = ad.embed_for_mode({"lc": torch.randn(5, 1, 24)}, "lc")
    assert emb.shape == (5, 8)


def test_layerwise_is_deterministic_and_bd():
    set_deterministic(0)
    ad = _make_patch_stub(d=8)
    batch = {"lc": torch.randn(6, 1, 24)}
    a = ad.embed_all_layers_for_mode(batch, "lc")
    b = ad.embed_all_layers_for_mode(batch, "lc")
    assert "last_hidden_state" in a and len(a) >= 3
    for name in a:
        assert a[name].dim() == 2 and a[name].shape[0] == 6, name
        assert np.max(np.abs(a[name].numpy() - b[name].numpy())) == 0.0, name
        assert mknn(a[name].numpy(), b[name].numpy(), k=3) == 1.0, name


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("PU_RUN_SLOW") != "1",
    reason="set PU_RUN_SLOW=1 to download the Granite PatchTS checkpoints",
)
@pytest.mark.parametrize("alias,repo", [
    ("patchtst", "ibm-granite/granite-timeseries-patchtst"),
    ("patchtsmixer", "ibm-granite/granite-timeseries-patchtsmixer"),
])
def test_patchts_live_smoke(alias, repo):
    set_deterministic(0)
    ad = get_adapter(alias)(repo, "granite", alias=alias)
    ad.load()
    batch = {"lc": torch.randn(8, 1, 96)}
    emb = ad.embed_for_mode(batch, "lc")
    assert emb.shape[0] == 8 and emb.dim() == 2
