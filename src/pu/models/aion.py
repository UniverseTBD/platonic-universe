from typing import Any, Dict, Iterable

import numpy as np
import torch

from pu.models.base import ModelAdapter
from pu.models.registry import register_adapter

try:
    from aion import AION
    AION_AVAILABLE = True
except ImportError:
    AION_AVAILABLE = False


class PreprocessAION:
    """Preprocessor that tokenizes raw flux data through AION's codec
    pipeline instead of the lossy flux_to_pil path."""

    # Map dataset mode names to AION modality classes and token keys
    _MODE_MAP = {
        "hsc": ("HSCImage", "tok_image_hsc"),
        "legacy": ("LegacySurveyImage", "tok_image"),
        "jwst": ("Image", "tok_image"),
    }

    def __init__(self, modes, model):
        self.modes = modes
        self.model = model
        from aion.codecs.manager import CodecManager
        self.codec_mgr = CodecManager(device="cpu")

    def __call__(self, idx):
        import aion.modalities as mod

        result = {}
        for mode in self.modes:
            if mode in ("desi", "sdss"):
                continue
            if mode not in self._MODE_MAP:
                continue
            cls_name, token_key = self._MODE_MAP[mode]
            modality_cls = getattr(mod, cls_name)

            flux = np.asarray(idx[f"{mode}_image"]["flux"], dtype=np.float32)
            flux_t = torch.from_numpy(flux)
            if flux_t.ndim == 2:
                flux_t = flux_t.unsqueeze(0)  # (H, W) -> (1, H, W)
            elif flux_t.ndim == 3 and flux_t.shape[-1] in (3, 5):
                flux_t = flux_t.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            flux_t = flux_t.unsqueeze(0)  # add batch dim
            # Use actual survey band names that AION's codec expects
            _BAND_NAMES = {
                "hsc": ["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"],
                "legacy": ["DES-G", "DES-R", "DES-I", "DES-Z"],
            }
            band_list = _BAND_NAMES.get(mode, [f"band_{i}" for i in range(flux_t.shape[1])])
            bands = band_list[:flux_t.shape[1]]
            modality = modality_cls(flux=flux_t, bands=bands)
            tokens = self.codec_mgr.encode(modality)
            result[mode] = tokens[token_key].squeeze(0)
        return result


class AIONAdapter(ModelAdapter):
    """
    Adapter for AION (AstronomIcal Omni-modal Network) by Polymathic AI.

    AION natively supports HSC, Legacy Survey, SDSS, and DESI data formats,
    so we use its own modality-aware preprocessing rather than flux_to_pil.

    Install with: pip install polymathic-aion
    """

    def __init__(self, model_name: str, size: str, alias: str = None):
        super().__init__(model_name, size, alias)
        if not AION_AVAILABLE:
            raise ImportError(
                "AION is not installed. Please install it with:\n"
                "  pip install polymathic-aion"
            )
        self.model = None

    def load(self, compile_model: bool = False) -> None:
        self.model = AION.from_pretrained(self.model_name)
        self.model.to("cuda").eval()

        if compile_model:
            self.model = torch.compile(
                self.model, mode="reduce-overhead", fullgraph=False
            )

    def get_preprocessor(
        self, modes: Iterable[str], resize: bool = False, resize_mode: str = "fill"
    ):
        return PreprocessAION(modes, self.model)

    def embed_for_mode(self, batch: Dict[str, Any], mode: str):
        tokens = batch[mode].to("cuda")

        # Map mode names to AION token keys
        token_key_map = {
            "galaxies": "tok_image_hsc",
            "hsc": "tok_image_hsc",
            "legacy": "tok_image",
            "jwst": "tok_image",
        }
        token_key = token_key_map.get(mode, f"tok_{mode}")

        with torch.no_grad():
            outputs = self.model.encode({token_key: tokens})

            # Pool spatial dims if present (B, C, H, W) -> (B, C)
            if outputs.dim() == 4:
                outputs = outputs.mean(dim=(2, 3))
            # Pool sequence dim if present (B, S, D) -> (B, D)
            elif outputs.dim() == 3:
                outputs = outputs.mean(dim=1)

        return outputs.float().detach()


if AION_AVAILABLE:
    register_adapter("aion", AIONAdapter)
