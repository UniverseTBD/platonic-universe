import torch
from transformers import (AutoModel, AutoImageProcessor, AutoVideoProcessor, HieraModel,AutoProcessor, AutoModelForImageTextToText,)
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

class VLMAdapter(HFAdapter):
    """
    subclass of HFAdapter for vision-language models that need:
      - AutoProcessor  instead of AutoImageProcessor
      - AutoModelForImageTextToText  instead of AutoModel
      - pixel_values passed explicitly (with dtype cast) alongside a text prompt
      - last_hidden_state mean-pooled over the full sequence

    PaliGemma2 (all sizes) and LLaVA-1.5 and LLaVA-OneVision.

    Difference between families:
      - the image placeholder moved into the prompt  (see _PROMPTS)
      - bfloat16 dtype (these large models are always loaded in bf16)

    PreprocessHF, embed_for_mode signature, registration
    pattern — is identical to HFAdapter.
    """

    # Minimal prompt that causes the processor to insert the right number
    # of image token slots. Empty string works for PaliGemma (processor
    # handles token injection implicitly); LLaVA variants need "<image>".
    _PROMPTS = {
        "paligemma": " ",
        "llava_15":  "USER: <image>\n ASSISTANT:",
        "llava_ov":  "<image>",
    }

    def load(self, compile_model: bool = False) -> None:
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        ).eval()
        if compile_model:
            self.model = torch.compile(
                self.model, mode="reduce-overhead", fullgraph=False
            )

    def get_preprocessor(self, modes: Iterable[str]):
        # PreprocessHF works with AutoProcessor just as well as AutoImageProcessor
        return PreprocessHF(modes, self.processor, resize=False)

    def embed_for_mode(self, batch: Dict[str, Any], mode: str):
        pv = batch[f"{mode}"].to("cuda")

        # Cast pixel_values to match model weights dtype (bf16) to avoid
        # the "Input type / weight type mismatch" RuntimeError
        model_dtype = next(self.model.parameters()).dtype
        pv = pv.to(dtype=model_dtype)

        # Build a minimal tokenised prompt so the model can place image tokens.
        # We pass dummy images of the correct size; actual visual content comes
        # from the pv tensor we pass explicitly to model.forward() below.
        import numpy as np
        from PIL import Image
        B      = pv.shape[0]
        prompt = self._PROMPTS.get(self.alias, " ")
        size   = pv.shape[-1]   # use the actual spatial size (224 or 336)
        dummy  = [
            Image.fromarray(np.ones((size, size, 3), dtype=np.uint8) * 128)
            for _ in range(B)
        ]
        enc = self.processor(
            images=dummy, text=[prompt] * B,
            return_tensors="pt", padding=True,
        )
        input_ids = enc["input_ids"].to("cuda")
        attn_mask = enc["attention_mask"].to("cuda")

        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                pixel_values=pv,
                attention_mask=attn_mask,
                return_dict=True,
            )
            # last_hidden_state is the LLM sequence output; mean-pool over tokens
            emb = out.last_hidden_state          # (B, seq_len, D)
            m   = attn_mask.float().unsqueeze(-1)
            emb = (emb * m).sum(1) / m.sum(1).clamp_min(1.0)
            emb = emb.float().detach()
        return emb

# Register this adapter for the HF-style aliases used by the repo
for alias in ("vit", "dino","dinov3", "convnext", "ijepa", "vjepa", "vit-mae","hiera"):
    register_adapter(alias, HFAdapter)

# VLM aliases — PaliGemma2 sizes
for alias in ("paligemma", "paligemma_3b", "paligemma_10b", "paligemma_28b"):
    register_adapter(alias, VLMAdapter)

# VLM aliases — LLaVA variants
for alias in ("llava_15", "llava_15_7b", "llava_15_13b", "llava_ov", "llava_ov_7b"):
    register_adapter(alias, VLMAdapter)
