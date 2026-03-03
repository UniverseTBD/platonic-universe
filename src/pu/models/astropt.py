import torch
from typing import Any, Dict, Iterable
from pu.models.base import ModelAdapter
from pu.preprocess import PreprocessAstropt
from pu.models.registry import register_adapter
from astropt.model_utils import load_astropt

class AstroptAdapter(ModelAdapter):
    """
    Adapter for astroPT models. Wraps `load_astropt` and uses `PreprocessAstropt`
    for preprocessing and the model's `generate_embeddings` for embedding.
    """

    def __init__(self, model_name: str, size: str, alias: str = None):
        super().__init__(model_name, size, alias)
        self.model = None

    def load(self, compile_model: bool = False) -> None:
        # follow previous code: model is loaded with a path containing the size
        self.model = load_astropt(self.model_name, path=f"astropt/{self.size}").to("cuda")
        self.model.eval()

        if compile_model:
            self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)

    def get_preprocessor(self, modes: Iterable[str], resize: bool = False, resize_mode: str = "fill"):
        # PreprocessAstropt needs the modality_registry from the loaded model
        return PreprocessAstropt(self.model.modality_registry, modes, resize=resize, resize_mode=resize_mode)

    def embed_for_mode(self, batch: Dict[str, Any], mode: str):
        # Expects batch to contain f"{mode}_images" and f"{mode}_positions" as tensors
        inputs = {
            "images": batch[f"{mode}_images"].to("cuda"),
            "images_positions": batch[f"{mode}_positions"].to("cuda"),
        }
        with torch.no_grad():
            outputs = self.model.generate_embeddings(inputs)["images"].detach()
        return outputs

# Register adapter
register_adapter("astropt", AstroptAdapter)
