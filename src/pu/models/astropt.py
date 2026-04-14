from typing import Any, Dict, Iterable

import torch
from astropt.model_utils import load_astropt

from pu.models.base import ModelAdapter
from pu.models.registry import register_adapter
from pu.preprocess import PreprocessAstropt


class AstroptAdapter(ModelAdapter):
    """Adapter for astroPT models."""

    def __init__(self, model_name: str, size: str, alias: str = None):
        super().__init__(model_name, size, alias)
        self.model = None

    def load(self, compile_model: bool = False) -> None:
        self.model = load_astropt(self.model_name, path=f"astropt/{self.size}").to("cuda")
        self.model.eval()
        if compile_model:
            self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)

    def get_preprocessor(self, modes: Iterable[str], resize: bool = False, resize_mode: str = "fill"):
        return PreprocessAstropt(self.model.modality_registry, modes, resize=resize, resize_mode=resize_mode)

    def embed_for_mode(self, batch: Dict[str, Any], mode: str):
        inputs = {
            "images": batch[f"{mode}_images"].to("cuda"),
            "images_positions": batch[f"{mode}_positions"].to("cuda"),
        }
        with torch.no_grad():
            return self.model.generate_embeddings(inputs)["images"].detach()

    def supports_layerwise(self) -> bool:
        return True

    def embed_all_layers_for_mode(
        self,
        batch: Dict[str, Any],
        mode: str,
    ) -> Dict[str, torch.Tensor]:
        inputs = {
            "images": batch[f"{mode}_images"].to("cuda"),
            "images_positions": batch[f"{mode}_positions"].to("cuda"),
        }

        def forward_fn():
            self.model.generate_embeddings(inputs)

        return self._capture_all_leaf_outputs(forward_fn)


register_adapter("astropt", AstroptAdapter)
