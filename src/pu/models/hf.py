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

    Supports:
      - torch.compile: Pass compile_model=True to load() for optimized inference
      - AMP: Call enable_amp(True) for float16 mixed precision inference
    """

    def __init__(self, model_name: str, size: str, alias: str = None):
        super().__init__(model_name, size, alias)
        self.processor = None
        self.model = None

    def load(self, compile_model: bool = False) -> None:
        if self.alias == "vjepa":
            self.processor = AutoVideoProcessor.from_pretrained(self.model_name)
        else:
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)

        if self.alias == "hiera":
            self.model = HieraModel.from_pretrained(self.model_name).to("cuda").eval()
        else:
            self.model = AutoModel.from_pretrained(self.model_name).to("cuda").eval()

        # Apply torch.compile for optimized inference
        if compile_model:
            self.model = torch.compile(
                self.model,
                mode="reduce-overhead",
                fullgraph=False,  # Allow graph breaks for complex HF models
            )


    def get_preprocessor(self, modes: Iterable[str]):
        # Return a callable compatible with datasets.Dataset.map
        return PreprocessHF(modes, self.processor, resize=False)

    def embed_for_mode(self, batch: Dict[str, Any], mode: str):
        # batch is a dict produced by the DataLoader; HF preprocess stores tensors under f"{mode}"
        inputs = batch[f"{mode}"].to("cuda")
        with torch.no_grad():
            # Use AMP if enabled for faster inference with lower memory
            with torch.amp.autocast("cuda", enabled=self._use_amp, dtype=torch.float16):
                outputs = self.model(inputs).last_hidden_state
                if self.alias == "vit" or self.alias == "vit-mae":
                    emb = outputs[:, 1:].mean(dim=1)
                elif self.alias == "convnext":
                    emb = outputs.mean(dim=(2, 3))
                elif self.alias == "dino":
                    emb = outputs[:, 0]
                elif self.alias == "dinov3":
                    emb = outputs[:, 0, :]
                elif self.alias == "ijepa":
                    emb = outputs.mean(dim=1)
                elif self.alias == "vjepa":
                    emb = outputs.mean(dim=1)
                elif self.alias == "hiera":
                    #  Hiera output is (B, 49, C).
                    # We pool over the sequence dimension (dim=1).
                    emb = outputs.mean(dim=1)
                else:
                    # Default fallback: mean over token dim excluding CLS if present
                    emb = outputs.mean(dim=1)

            # Always return float32 for downstream metric computation
            emb = emb.float().detach()
        return emb

# Register this adapter for the HF-style aliases used by the repo
for alias in ("vit", "dino","dinov3", "convnext", "ijepa", "vjepa", "vit-mae","hiera"):
    register_adapter(alias, HFAdapter)
