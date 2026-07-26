"""TabPFN tabular-foundation-model adapter (tabular modality).

Adds TabPFN v2 (Prior Labs, https://github.com/PriorLabs/TabPFN) so the Platonic
Representation Hypothesis can be probed on tabular catalogues (for example the
photometric / morphological columns that ship with ``Smith42/galaxies``). TabPFN
is an in-context transformer: it conditions on a fitted ``(X, y)`` support set
and produces a per-row embedding from its internal representation.

Contract
--------
``batch[mode]`` is a float tensor / array ``(B, n_features)`` of catalogue rows.
Optional per-row labels live at ``batch[f"{mode}_y"]``; when absent a
deterministic 2-class parity label is used so the in-context model has a valid
support set (downstream should pass a real target column for a meaningful
comparison). ``embed_for_mode`` returns ``(B, embedding_dim)``.

Layer-wise extraction is intentionally unsupported: TabPFN's alternating
row/column attention has no clean per-layer, per-row residual stream to pool, so
it exposes a single embedding (see PriorLabs/TabPFN#243).

Weights are gated on HuggingFace (``Prior-Labs/tabpfn_*``): accept the terms once
and authenticate (``hf auth login`` or ``HF_TOKEN``). Install with
``pip install ".[tabular]"``.
"""

import importlib.util
from typing import Any, Dict, Iterable

import numpy as np
import torch

from pu.models.base import ModelAdapter, resolve_device
from pu.models.registry import register_adapter

TABPFN_AVAILABLE = importlib.util.find_spec("tabpfn") is not None


class PreprocessTabular:
    """Pass catalogue feature rows (and optional labels) through unchanged.

    Expects each example to expose the feature vector under ``feature_key`` (or
    the mode name) and, optionally, a scalar target under ``label_key``.
    """

    def __init__(self, modes: Iterable[str], feature_key: str = "features",
                 label_key: str = "label"):
        self.modes = list(modes)
        self.feature_key = feature_key
        self.label_key = label_key

    def __call__(self, idx: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for mode in self.modes:
            feats = idx.get(self.feature_key, idx.get(mode))
            if feats is None:
                continue
            result[mode] = np.asarray(feats, dtype=np.float32)
            if self.label_key in idx and idx[self.label_key] is not None:
                result[f"{mode}_y"] = np.asarray(idx[self.label_key])
        return result


class TabPFNAdapter(ModelAdapter):
    """Adapter for TabPFN v2 tabular embeddings.

    ``model_name`` is informational (the weights are resolved by the ``tabpfn``
    package); ``size`` selects the estimator count via the alias map.
    """

    def __init__(self, model_name: str, size: str, alias: str = None):
        super().__init__(model_name, size, alias)
        self.model = None
        self.device = None
        self.n_estimators = 4

    def load(self, compile_model: bool = False) -> None:
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as exc:
            raise ImportError(
                "tabpfn is not installed. Install the extra with:\n"
                '  pip install ".[tabular]"   (or: pip install tabpfn)\n'
                "Weights are gated: accept the terms at "
                "https://huggingface.co/Prior-Labs/TabPFN-v2-clf and run `hf auth login`."
            ) from exc

        self.device = resolve_device()
        self.model = TabPFNClassifier(device=self.device, n_estimators=self.n_estimators)

    def get_preprocessor(self, modes, resize: bool = False, resize_mode: str = "fill"):
        return PreprocessTabular(modes)

    @staticmethod
    def _as_numpy(x) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float32)

    @staticmethod
    def _default_labels(n: int) -> np.ndarray:
        """Deterministic 2-class parity labels: a valid support set with no data leak."""
        return (np.arange(n) % 2).astype(int)

    def embed_for_mode(self, batch: Dict[str, Any], mode: str):
        X = self._as_numpy(batch[mode])
        if X.ndim == 1:
            X = X[:, None]
        y = batch.get(f"{mode}_y")
        y = self._as_numpy(y).astype(int).ravel() if y is not None else self._default_labels(len(X))

        self.model.fit(X, y)
        # get_embeddings -> (n_estimators, n_samples, embedding_dim); average the ensemble.
        emb = np.asarray(self.model.get_embeddings(X))
        if emb.ndim == 3:
            emb = emb.mean(axis=0)
        return torch.as_tensor(emb, dtype=torch.float32)  # (B, D)

    # TabPFN exposes a single embedding, not a per-layer residual stream.
    def supports_layerwise(self) -> bool:
        return False


# Canonical alias -> (sizes, estimator counts) map for downstream discovery.
TABULAR_MODEL_MAP = {
    "tabpfn": (["default"], ["Prior-Labs/TabPFN-v2-clf"]),
}

register_adapter("tabpfn", TabPFNAdapter)
