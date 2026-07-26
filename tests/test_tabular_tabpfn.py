"""Tests for the TabPFN tabular adapter.

Stub-based (CPU, no download): TabPFN weights are gated on HuggingFace, so the
live path is an opt-in smoke test. These verify the embedding plumbing:
numpy coercion, default-label synthesis, ensemble averaging, and shape.
"""
import os

import numpy as np
import pytest
import torch

from pu.models import get_adapter
from pu.models.tabular import (
    TABPFN_AVAILABLE,
    PreprocessTabular,
    TabPFNAdapter,
)


def test_registry_resolves_tabpfn():
    assert get_adapter("tabpfn") is TabPFNAdapter


def test_tabpfn_is_not_layerwise():
    ad = TabPFNAdapter.__new__(TabPFNAdapter)
    assert ad.supports_layerwise() is False


def test_default_labels_are_two_class_parity():
    ad = TabPFNAdapter.__new__(TabPFNAdapter)
    y = ad._default_labels(5)
    assert list(y) == [0, 1, 0, 1, 0]
    assert set(np.unique(y)) == {0, 1}


def test_as_numpy_handles_torch_and_lists():
    ad = TabPFNAdapter.__new__(TabPFNAdapter)
    assert ad._as_numpy(torch.ones(2, 3)).shape == (2, 3)
    assert ad._as_numpy([[1.0, 2.0]]).dtype == np.float32


def test_preprocess_reads_features_and_optional_label():
    pre = PreprocessTabular(modes=["tab"], feature_key="features", label_key="label")
    out = pre({"features": [1.0, 2.0, 3.0], "label": 1})
    assert out["tab"].shape == (3,)
    assert out["tab_y"] == 1


class _StubTabPFN:
    """Stand-in that mimics TabPFNClassifier.fit / get_embeddings deterministically."""

    def __init__(self, d: int = 8):
        self.d = d
        self._fitted = False

    def fit(self, X, y):
        assert len(np.unique(y)) >= 2, "TabPFN needs >= 2 classes"
        self._fitted = True
        return self

    def get_embeddings(self, X):
        assert self._fitted, "must fit before embedding"
        col0 = X[:, :1] if X.shape[1] >= 1 else np.zeros((len(X), 1), dtype=np.float32)
        emb = np.tile(col0, (1, self.d)).astype(np.float32)  # (B, d)
        return np.stack([emb, emb + 1.0], axis=0)            # (n_estimators=2, B, d)


def _make_tabpfn_stub(d: int = 8) -> TabPFNAdapter:
    ad = TabPFNAdapter.__new__(TabPFNAdapter)
    ad.alias = "tabpfn"
    ad.device = "cpu"
    ad.n_estimators = 2
    ad.model = _StubTabPFN(d)
    return ad


def test_embed_for_mode_averages_ensemble_to_bd():
    ad = _make_tabpfn_stub(d=8)
    X = torch.randn(6, 5)
    emb = ad.embed_for_mode({"tab": X}, "tab")
    assert emb.shape == (6, 8)          # ensemble averaged away
    assert emb.dtype == torch.float32


def test_embed_for_mode_synthesizes_labels_when_absent():
    ad = _make_tabpfn_stub(d=4)
    # No "tab_y" key -> adapter must build a valid 2-class support set (no crash).
    emb = ad.embed_for_mode({"tab": np.random.randn(7, 3).astype(np.float32)}, "tab")
    assert emb.shape == (7, 4)


def test_embed_for_mode_promotes_1d_features():
    ad = _make_tabpfn_stub(d=4)
    emb = ad.embed_for_mode({"tab": torch.randn(5)}, "tab")  # (B,) -> (B, 1)
    assert emb.shape == (5, 4)


@pytest.mark.slow
@pytest.mark.skipif(
    not (TABPFN_AVAILABLE and os.environ.get("PU_RUN_SLOW") == "1"),
    reason="set PU_RUN_SLOW=1, install tabpfn, and accept the gated weights terms",
)
def test_tabpfn_live_smoke():
    ad = get_adapter("tabpfn")("Prior-Labs/TabPFN-v2-clf", "default", alias="tabpfn")
    ad.load()
    X = np.random.RandomState(0).randn(32, 6).astype(np.float32)
    emb = ad.embed_for_mode({"tab": X, "tab_y": (np.arange(32) % 2)}, "tab")
    assert emb.shape[0] == 32 and emb.dim() == 2
