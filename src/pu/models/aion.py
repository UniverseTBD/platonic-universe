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
    """Preprocessor that passes raw flux data through AION's native
    modality-aware transforms instead of the lossy flux_to_pil path."""

    def __init__(self, modes, model):
        self.modes = modes
        self.model = model

    def __call__(self, idx):
        result = {}
        for mode in self.modes:
            if mode in ("desi", "sdss"):
                continue
            flux = np.asarray(idx[f"{mode}_image"]["flux"], dtype=np.float32)
            preprocessed = self.model.preprocess(flux, modality=mode)
            result[mode] = preprocessed
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
        inputs = batch[mode].to("cuda")

        with torch.no_grad():
            outputs = self.model(inputs)

            # Pool spatial dims if present (B, C, H, W) -> (B, C)
            if outputs.dim() == 4:
                outputs = outputs.mean(dim=(2, 3))
            # Pool sequence dim if present (B, S, D) -> (B, D)
            elif outputs.dim() == 3:
                outputs = outputs.mean(dim=1)

        return outputs.float().detach()


if AION_AVAILABLE:
    register_adapter("aion", AIONAdapter)
