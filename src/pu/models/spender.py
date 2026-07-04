"""Spender spectrum-autoencoder adapter (spectra modality).

Adds Spender (Melchior et al., https://github.com/pmelchior/spender) as a second
galaxy-spectrum foundation model alongside the existing SpecFormer. Spender is a
convolutional-attention autoencoder that compresses an observed-frame spectrum
into a low-dimensional, redshift-invariant latent; that latent is the embedding
we compare across models.

Contract (same shape rules as the other unimodal adapters)
----------------------------------------------------------
``batch[mode]`` is a float tensor ``(B, n_wave)`` of flux sampled on the
instrument's observed-wavelength grid (``adapter.n_wave`` after ``load``).
``embed_for_mode`` returns ``(B, latent_dim)``; ``embed_all_layers_for_mode``
returns one pooled ``(B, D)`` tensor per encoder block.

Weights load through ``spender.hub.load`` (torch.hub), not HuggingFace, so the
checkpoint host must be reachable. Install with ``pip install ".[spectra]"``.
"""

import importlib.util
from typing import Any, Dict, Iterable

import numpy as np
import torch

from pu.models.base import ModelAdapter, resolve_device
from pu.models.registry import register_adapter

SPENDER_AVAILABLE = importlib.util.find_spec("spender") is not None


class PreprocessSpender:
    """Extract a flux array per spectral mode into the ``(n_wave,)`` contract.

    Reads ``idx["spectrum"]["flux"]`` (the schema used by the DESI/SDSS
    crossmatched datasets) or a bare ``idx[mode]`` fallback. Image modes are
    skipped because Spender is spectral-only.
    """

    def __init__(self, modes: Iterable[str]):
        self.modes = list(modes)

    def __call__(self, idx: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for mode in self.modes:
            if mode in ("desi", "sdss") and isinstance(idx.get("spectrum"), dict):
                flux = np.asarray(idx["spectrum"]["flux"], dtype=np.float32)
            elif mode in idx:
                flux = np.asarray(idx[mode], dtype=np.float32)
            else:
                continue
            result[mode] = flux
        return result


class SpenderAdapter(ModelAdapter):
    """Adapter for the Spender galaxy-spectrum autoencoder.

    ``model_name`` is a ``spender.hub`` tag (e.g. ``"sdss_II"``), not a
    HuggingFace repo. After ``load`` the expected input length is available as
    ``self.n_wave``.
    """

    def __init__(self, model_name: str, size: str, alias: str = None):
        super().__init__(model_name, size, alias)
        self.model = None
        self.instrument = None
        self.device = None
        self.n_wave = None

    def load(self, compile_model: bool = False) -> None:
        try:
            import spender
        except ImportError as exc:
            raise ImportError(
                "spender is not installed. Install the extra with:\n"
                '  pip install ".[spectra]"   (or: pip install spender)'
            ) from exc

        self.device = resolve_device()
        self.instrument, self.model = spender.hub.load(
            self.model_name, map_location=self.device
        )
        self.model = self.model.to(self.device).eval()
        wave = getattr(self.instrument, "wave_obs", None)
        self.n_wave = int(len(wave)) if wave is not None else None
        if compile_model:
            self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)

    def get_preprocessor(self, modes, resize: bool = False, resize_mode: str = "fill"):
        return PreprocessSpender(modes)

    def _get_hookable_model(self):
        return self.model.encoder

    def _prepare(self, spectra: torch.Tensor) -> torch.Tensor:
        x = spectra.to(self.device).float()
        if x.dim() == 3 and x.shape[-1] == 1:
            x = x[..., 0]          # (B, L, 1) -> (B, L)
        return x

    def embed_for_mode(self, batch: Dict[str, Any], mode: str):
        spectra = self._prepare(batch[mode])
        with torch.no_grad():
            latent = self.model.encode(spectra)
        return latent.float().detach()  # (B, latent_dim)

    def supports_layerwise(self) -> bool:
        return True

    def get_layer_names(self, granularity: str = "blocks") -> list:
        names = super().get_layer_names(granularity=granularity)
        names.append("latent")
        return names

    def embed_all_layers_for_mode(self, batch, mode, granularity: str = "blocks"):
        spectra = self._prepare(batch[mode])
        captured = {}

        def forward_fn():
            captured["latent"] = self.model.encode(spectra)

        results = self._capture_module_outputs(
            forward_fn,
            model=self._get_hookable_model(),
            pool_fn=self._generic_pool,
            granularity=granularity,
        )
        if "latent" in captured:
            results["latent"] = self._generic_pool(captured["latent"]).float().detach()
        return results


# Canonical alias -> (sizes, hub tags) map for downstream discovery.
SPECTRA_MODEL_MAP = {
    "spender": (["sdss"], ["sdss_II"]),
}

register_adapter("spender", SpenderAdapter)
