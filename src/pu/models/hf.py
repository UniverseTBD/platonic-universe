import torch
from transformers import AutoModel, AutoImageProcessor, AutoVideoProcessor, HieraModel
from typing import Any, Dict, Iterable
from pu.models.base import ModelAdapter
from pu.preprocess import PreprocessHF
from pu.models.registry import register_adapter

class HFAdapter(ModelAdapter):
    """
    Adapter for HuggingFace vision models using AutoModel + AutoImageProcessor.
    The adapter uses the 'alias' passed at construction to decide pooling:
      - 'vit' -> CLS excluded mean over tokens (last_hidden_state[:,1:].mean)
      - 'dino' -> CLS token (last_hidden_state[:,0])
      - 'convnext' -> spatial mean over HxW (last_hidden_state.mean(dim=(2,3)))
      - 'ijepa' -> mean over token dim (last_hidden_state.mean(dim=1))
      - 'vjepa' -> mean over token dim (last_hidden_state.mean(dim=1))
      - 'vit-mae' -> CLS excluded mean over tokens (last_hidden_state[:,1:].mean)
    """

    def __init__(self, model_name: str, size: str, alias: str = None):
        super().__init__(model_name, size, alias)
        self.processor = None
        self.model = None

    def load(self) -> None:
        if self.alias == "vjepa":
            self.processor = AutoVideoProcessor.from_pretrained(self.model_name)
        else:
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        
        if self.alias == "hiera":
            self.model = HieraModel.from_pretrained(self.model_name).to("cuda").eval()
        else:
            self.model = AutoModel.from_pretrained(self.model_name).to("cuda").eval()


    def get_preprocessor(self, modes: Iterable[str]):
        # Return a callable compatible with datasets.Dataset.map
        return PreprocessHF(modes, self.processor, resize=False)

    def embed_for_mode(self, batch: Dict[str, Any], mode: str):
        # batch is a dict produced by the DataLoader; HF preprocess stores tensors under f"{mode}"
        inputs = batch[f"{mode}"].to("cuda")
        with torch.no_grad():
            outputs = self.model(inputs).last_hidden_state
            if self.alias == "vit" or self.alias == "vit-mae":
                emb = outputs[:, 1:].mean(dim=1).detach()
            elif self.alias == "convnext":
                emb = outputs.mean(dim=(2, 3)).detach()
            elif self.alias == "dino":
                emb = outputs[:, 0].detach()
            elif self.alias == "dinov3":
                emb = outputs[:, 0, :].detach()
            elif self.alias == "ijepa":
                emb = outputs.mean(dim=1).detach()
            elif self.alias == "vjepa":
                emb = outputs.mean(dim=1).detach()
            elif self.alias == "hiera":
                #  Hiera output is (B, 49, C). 
                # We pool over the sequence dimension (dim=1).
                emb = outputs.mean(dim=1).detach()
            else:
                # Default fallback: mean over token dim excluding CLS if present
                emb = outputs.mean(dim=1).detach()
        return emb

# Register this adapter for the HF-style aliases used by the repo
for alias in ("vit", "dino","dinov3", "convnext", "ijepa", "vjepa", "vit-mae","hiera"):
    register_adapter(alias, HFAdapter)
