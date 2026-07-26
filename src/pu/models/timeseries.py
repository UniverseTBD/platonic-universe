"""Time-series foundation-model adapters.

Brings general-purpose time-series foundation models into platonic-universe so
the Platonic Representation Hypothesis can be probed on the time-series
modality (for example astronomical light curves). This module deliberately
defines the *API contract* that downstream light-curve data plugs into.

Input contract
--------------
Each adapter's ``embed_for_mode(batch, mode)`` expects ``batch[mode]`` to be a
float tensor shaped ``(B, n_channels, seq_len)`` (channels-first). Univariate
series use ``n_channels == 1``. Series shorter or longer than a model's native
context window are right-padded / truncated by the adapter, and an input mask
is built for models that accept one.

Output contract
---------------
``embed_for_mode`` returns a ``(B, D)`` float tensor: one embedding per series.
``embed_all_layers_for_mode`` returns an *ordered* dict mapping layer name to a
``(B, D)`` tensor, one entry per hooked block, so layer-wise representational
alignment (MKNN/CKA/...) is computed exactly as for the vision models.

Determinism
-----------
Models are loaded in ``.eval()`` and run under ``torch.no_grad()``. Call
``pu.models.base.set_deterministic`` before extraction. Seeding alone is not
enough: MKNN is row-order sensitive and near-tied cosine distances can flip
neighbour sets across BLAS backends. See ``scripts/timeseries_layerwise_mknn.py``.

Dependencies live behind the ``timeseries`` extra and are imported lazily inside
``load`` so ``import pu.models`` never requires them:

    pip install ".[timeseries]"          # chronos-forecasting
    # MOMENT pins transformers==4.33.3 and must live in its own environment;
    # see MomentAdapter for details.
"""

import importlib.util
from typing import Any, Dict, Iterable

import numpy as np
import torch

from pu.models.base import ModelAdapter, resolve_device
from pu.models.registry import register_adapter


def _has(pkg: str) -> bool:
    """True if ``pkg`` is importable, without importing it (keeps import cheap)."""
    try:
        return importlib.util.find_spec(pkg) is not None
    except (ImportError, ValueError):
        return False


CHRONOS_AVAILABLE = _has("chronos")
MOMENT_AVAILABLE = _has("momentfm")


# --------------------------------------------------------------------------- #
# Shared helpers                                                              #
# --------------------------------------------------------------------------- #

def fit_length(x: torch.Tensor, target: int):
    """Right-pad (with zeros) or truncate a ``(B, C, T)`` batch to length ``target``.

    Returns ``(x_fitted, mask)`` where ``mask`` is ``(B, target)`` with 1 for
    real observations and 0 for padding, matching the ``input_mask`` convention
    used by patch-based time-series models.
    """
    if x.dim() != 3:
        raise ValueError(f"expected (B, C, T) series, got shape {tuple(x.shape)}")
    B, C, T = x.shape
    if T > target:
        x = x[..., :target]
        mask = torch.ones(B, target, device=x.device, dtype=x.dtype)
    elif T < target:
        x = torch.nn.functional.pad(x, (0, target - T))
        mask = torch.zeros(B, target, device=x.device, dtype=x.dtype)
        mask[:, :T] = 1
    else:
        mask = torch.ones(B, target, device=x.device, dtype=x.dtype)
    return x, mask


class PreprocessTimeSeries:
    """Turn a HuggingFace example into the ``(B, C, T)`` contract tensor.

    Reads the raw series from ``series_key`` (falling back to the mode name) and
    returns ``{mode: float32 (C, T)}`` for each requested mode. Univariate
    series are promoted to a single channel. Downstream datasets can subclass
    this to pull flux/time columns from their own schema.
    """

    def __init__(self, modes: Iterable[str], series_key: str = "lightcurve"):
        self.modes = list(modes)
        self.series_key = series_key

    def __call__(self, idx: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for mode in self.modes:
            raw = idx.get(self.series_key, idx.get(mode))
            if raw is None:
                continue
            series = np.asarray(raw, dtype=np.float32)
            if series.ndim == 1:
                series = series[np.newaxis, :]  # (T,) -> (1, T)
            result[mode] = series
        return result


# --------------------------------------------------------------------------- #
# Chronos (Amazon)                                                            #
# --------------------------------------------------------------------------- #

class ChronosAdapter(ModelAdapter):
    """Adapter for Amazon Chronos time-series foundation models.

    Handles both families that ``chronos-forecasting`` ships:

    - **Chronos-T5** (e.g. ``amazon/chronos-t5-*``): tokenises a scaled series
      and runs a T5 encoder-decoder; ``embed`` returns ``(B, ctx_len + 1, d)``.
    - **Chronos-Bolt** (e.g. ``amazon/chronos-bolt-*``): a patch-based direct
      multistep model; ``embed`` returns ``(B, n_patches, d)``.

    Either way we mean-pool the encoder sequence to one ``(B, d_model)`` vector.
    Chronos is univariate, so a multivariate ``(B, C, T)`` batch is reduced to
    its first channel. Install with ``pip install ".[timeseries]"``.
    """

    def __init__(self, model_name: str, size: str, alias: str = None):
        super().__init__(model_name, size, alias)
        self.pipeline = None
        self.model = None  # underlying encoder stack (hookable subtree)
        self.device = None

    def load(self, compile_model: bool = False) -> None:
        try:
            from chronos import BaseChronosPipeline
        except ImportError as exc:
            raise ImportError(
                "chronos-forecasting is not installed. Install the extra with:\n"
                '  pip install ".[timeseries]"   (or: pip install chronos-forecasting)'
            ) from exc

        self.device = resolve_device()
        # BaseChronosPipeline auto-selects ChronosPipeline (T5) or
        # ChronosBoltPipeline (Bolt) from the checkpoint config.
        self.pipeline = BaseChronosPipeline.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=torch.float32,
        )
        self.model = self._find_encoder()
        self.model.eval()
        if compile_model:
            self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)

    def _find_encoder(self):
        """Locate the encoder stack across Chronos-T5 and Chronos-Bolt layouts."""
        inner = self.pipeline.model  # ChronosModel / ChronosBoltModelForForecasting
        if hasattr(inner, "model") and hasattr(inner.model, "encoder"):
            return inner.model.encoder  # Chronos-T5
        if hasattr(inner, "encoder"):
            return inner.encoder         # Chronos-Bolt
        raise AttributeError(
            f"Could not locate an encoder on {type(inner).__name__}; "
            "Chronos internals may have changed."
        )

    def get_preprocessor(self, modes, resize: bool = False, resize_mode: str = "fill"):
        return PreprocessTimeSeries(modes)

    def _get_hookable_model(self):
        # Hook the encoder stack: that is what pipeline.embed() runs.
        return self.model

    @staticmethod
    def _to_context(series: torch.Tensor) -> torch.Tensor:
        """Reduce a ``(B, C, T)`` (or ``(B, T)``) batch to Chronos context ``(B, T)``."""
        if series.dim() == 3:
            series = series[:, 0, :]  # univariate: first channel
        return series.float().cpu()

    def embed_for_mode(self, batch: Dict[str, Any], mode: str):
        context = self._to_context(batch[mode])
        with torch.no_grad():
            emb, _ = self.pipeline.embed(context)  # (B, T+1, d)
        return emb.mean(dim=1).float().detach()  # (B, d)

    def supports_layerwise(self) -> bool:
        return True

    def get_layer_names(self, granularity: str = "blocks") -> list:
        names = super().get_layer_names(granularity=granularity)
        names.append("encoder_embeddings")
        return names

    def embed_all_layers_for_mode(self, batch, mode, granularity: str = "blocks"):
        context = self._to_context(batch[mode])
        captured = {}

        def forward_fn():
            emb, _ = self.pipeline.embed(context)
            captured["encoder_embeddings"] = emb

        results = self._capture_module_outputs(
            forward_fn,
            model=self._get_hookable_model(),
            pool_fn=self._generic_pool,
            granularity=granularity,
        )
        if "encoder_embeddings" in captured:
            results["encoder_embeddings"] = self._generic_pool(
                captured["encoder_embeddings"]
            ).float().detach()
        return results


# --------------------------------------------------------------------------- #
# MOMENT (AutonLab)                                                           #
# --------------------------------------------------------------------------- #

class MomentAdapter(ModelAdapter):
    """Adapter for MOMENT-1 (AutonLab), a masked time-series transformer.

    MOMENT patches a fixed-length (512) series and encodes it with a T5-style
    stack; the ``"embedding"`` task head returns a per-series representation in
    ``output.embeddings``. Multivariate series are supported natively as
    ``(B, n_channels, 512)``.

    Environment note
    ----------------
    ``momentfm`` pins ``transformers==4.33.3``, which is incompatible with this
    repo's modern ``transformers``. Run MOMENT in a dedicated environment and
    extract embeddings to parquet/npy, then compare offline with ``pu compare``
    or ``pu.metrics.mknn`` (the same save-then-compare flow the repo already
    uses for pre-computed embeddings). The adapter code is transformers-version
    agnostic; only the ``momentfm`` package carries the pin.
    """

    CONTEXT_LENGTH = 512

    def __init__(self, model_name: str, size: str, alias: str = None):
        super().__init__(model_name, size, alias)
        self.model = None
        self.device = None

    def load(self, compile_model: bool = False) -> None:
        try:
            from momentfm import MOMENTPipeline
        except ImportError as exc:
            raise ImportError(
                "momentfm is not installed. MOMENT pins transformers==4.33.3, so "
                "install it in a dedicated environment:\n"
                "  pip install momentfm\n"
                "See MomentAdapter docstring for the save-then-compare workflow."
            ) from exc

        self.device = resolve_device()
        self.model = MOMENTPipeline.from_pretrained(
            self.model_name, model_kwargs={"task_name": "embedding"}
        )
        self.model.init()
        self.model = self.model.to(self.device).float().eval()
        if compile_model:
            self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)

    def get_preprocessor(self, modes, resize: bool = False, resize_mode: str = "fill"):
        return PreprocessTimeSeries(modes)

    def _prepare(self, series: torch.Tensor):
        x = series.to(self.device).float()
        return fit_length(x, self.CONTEXT_LENGTH)

    def embed_for_mode(self, batch: Dict[str, Any], mode: str):
        x, mask = self._prepare(batch[mode])
        with torch.no_grad():
            out = self.model(x_enc=x, input_mask=mask)
        return self._generic_pool(out.embeddings).float().detach()  # (B, D)

    def supports_layerwise(self) -> bool:
        return True

    def get_layer_names(self, granularity: str = "blocks") -> list:
        names = super().get_layer_names(granularity=granularity)
        names.append("embeddings")
        return names

    def embed_all_layers_for_mode(self, batch, mode, granularity: str = "blocks"):
        x, mask = self._prepare(batch[mode])
        captured = {}

        def forward_fn():
            out = self.model(x_enc=x, input_mask=mask)
            captured["embeddings"] = out.embeddings

        results = self._capture_module_outputs(
            forward_fn,
            model=self.model,
            pool_fn=self._generic_pool,
            granularity=granularity,
        )
        if "embeddings" in captured:
            results["embeddings"] = self._generic_pool(captured["embeddings"]).float().detach()
        return results


# --------------------------------------------------------------------------- #
# Canonical alias -> (sizes, HF repos) map for downstream discovery.          #
# Mirrors the model_map in experiments.py so the time-series modality has a    #
# single source of truth for which checkpoints are supported.                 #
# --------------------------------------------------------------------------- #

TS_MODEL_MAP = {
    "chronos": (
        ["tiny", "mini", "small", "base", "large",
         "bolt-tiny", "bolt-mini", "bolt-small", "bolt-base"],
        [f"amazon/chronos-t5-{s}" for s in ["tiny", "mini", "small", "base", "large"]]
        + [f"amazon/chronos-bolt-{s}" for s in ["tiny", "mini", "small", "base"]],
    ),
    "moment": (
        ["small", "base", "large"],
        [f"AutonLab/MOMENT-1-{s}" for s in ["small", "base", "large"]],
    ),
}


register_adapter("chronos", ChronosAdapter)
register_adapter("moment", MomentAdapter)
