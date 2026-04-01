import torch
from transformers import (AutoModel, AutoImageProcessor, AutoVideoProcessor, HieraModel,AutoProcessor, AutoModelForImageTextToText,)
from functools import partial
from transformers import AutoModel, AutoImageProcessor, AutoVideoProcessor, HieraModel, CLIPProcessor, CLIPModel
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
      - 'clip' -> 'image features': final, projected visual embs that have been
            aligned with text (get_image_features(), shape [batch, embedding_dim])

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
        elif self.alias == "clip":
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
        else:
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)

        if self.alias == "hiera":
            self.model = HieraModel.from_pretrained(self.model_name).to("cuda").eval()
        elif self.alias == "clip":
            self.model = CLIPModel.from_pretrained(self.model_name).to("cuda").eval()
        else:
            self.model = AutoModel.from_pretrained(self.model_name).to("cuda").eval()


        # Apply torch.compile for optimized inference
        if compile_model:
            self.model = torch.compile(
                self.model,
                mode="reduce-overhead",
                fullgraph=False,  # Allow graph breaks for complex HF models
            )


    def get_preprocessor(self, modes: Iterable[str], resize: bool = False, resize_mode: str = "fill"):
        return PreprocessHF(modes, self.processor, alias=self.alias, resize=resize, resize_mode=resize_mode)

    def embed_for_mode(self, batch: Dict[str, Any], mode: str):
        # batch is a dict produced by the DataLoader; HF preprocess stores tensors under f"{mode}"
        inputs = batch[f"{mode}"].to("cuda")
        with torch.no_grad():
            # Use AMP if enabled for faster inference with lower memory
            with torch.amp.autocast("cuda", enabled=self._use_amp, dtype=torch.float16):
                if self.alias == "clip":
                    outputs = self.model.get_image_features(pixel_values=inputs)
                    return outputs.float().detach()
                outputs = self.model(inputs).last_hidden_state
                if self.alias == "vit" or self.alias == "vit-mae":
                    emb = outputs[:, 1:].mean(dim=1)
                elif self.alias == "convnext":
                    emb = outputs.mean(dim=(2, 3))
                elif self.alias == "clip":
                    emb = outputs.mean(dim=1)
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
        "paligemma":    "<image> ",
        "paligemma_3b": "<image> ",
        "paligemma_10b":"<image> ",
        "paligemma_28b":"<image> ",
        "llava_15":     "USER: <image>\n ASSISTANT:",
        "llava_ov":     "<image>",
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

    def get_preprocessor(self, modes: Iterable[str], resize: bool = True, resize_mode: str = "match"):
        # PreprocessHF works with AutoProcessor just as well as AutoImageProcessor
        return PreprocessHF(modes, self.processor, alias=self.alias, resize=resize, resize_mode=resize_mode)

    def _forward_vlm(self, batch: Dict[str, Any], mode: str):
        """Shared VLM forward pass. Returns (hidden_states, attention_mask).

        hidden_states is a tuple of (N_layers+1) tensors, each (B, seq_len, D).
        The first element is the embedding layer; the rest are transformer layers.
        """
        import warnings
        warnings.filterwarnings("ignore", message=".*PaliGemma.*")
        warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
        warnings.filterwarnings("ignore", message=".*text prefix.*")
        warnings.filterwarnings("ignore", message=".*special image tokens.*")
        device = next(self.model.parameters()).device
        pv = batch[f"{mode}"].to(device)

        model_dtype = next(self.model.parameters()).dtype
        pv = pv.to(dtype=model_dtype)

        from PIL import Image
        B      = pv.shape[0]
        prompt = self._PROMPTS.get(self.alias, " ")
        pv_cpu = pv.cpu().float()
        pv_cpu = (pv_cpu - pv_cpu.min()) / (pv_cpu.max() - pv_cpu.min() + 1e-8)
        pv_cpu = (pv_cpu * 255).byte()
        pil_images = [
            Image.fromarray(pv_cpu[i].permute(1, 2, 0).numpy())
            for i in range(B)
        ]
        enc = self.processor(
            images=pil_images, text=[prompt] * B,
            return_tensors="pt", padding=True,
        )
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)
        pv = enc["pixel_values"].to(device, dtype=model_dtype)

        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                pixel_values=pv,
                attention_mask=attn_mask,
                return_dict=True,
                output_hidden_states=True,
            )

        if hasattr(out, "hidden_states") and out.hidden_states is not None:
            return out.hidden_states, attn_mask
        elif hasattr(out, "image_hidden_states") and out.image_hidden_states is not None:
            hs = out.image_hidden_states.mean(dim=1, keepdim=True)
            fallback_mask = torch.ones(hs.shape[:2], device=hs.device)
            return (hs,), fallback_mask
        else:
            raise AttributeError(
                f"Cannot extract embeddings from {type(out).__name__}. "
                f"Available keys: {[k for k,v in out.items() if v is not None]}"
            )

    @staticmethod
    def _masked_mean_pool(hs, attn_mask):
        """Attention-masked mean pooling over the sequence dimension."""
        m = attn_mask.float().unsqueeze(-1)
        return ((hs * m).sum(1) / m.sum(1).clamp_min(1.0)).float().detach()

    def embed_for_mode(self, batch: Dict[str, Any], mode: str):
        hidden_states, attn_mask = self._forward_vlm(batch, mode)
        # hidden_states[-1] is the final LLM layer output (B, seq_len, D)
        return self._masked_mean_pool(hidden_states[-1], attn_mask)

    def embed_all_layers_for_mode(self, batch: Dict[str, Any], mode: str):
        """Return mean-pooled embeddings from every hidden-state layer.

        Returns a list of (B, D) tensors, one per layer (including the
        initial embedding layer at index 0).
        """
        hidden_states, attn_mask = self._forward_vlm(batch, mode)
        return [self._masked_mean_pool(hs, attn_mask) for hs in hidden_states]

# Register this adapter for the HF-style aliases used by the repo
for alias in (
        "vit", "dino", "dinov3", "convnext", "ijepa", "vjepa", "vit-mae", "hiera", "clip"
        ):
    register_adapter(alias, HFAdapter)

# VLM aliases — PaliGemma2 sizes
for alias in ("paligemma", "paligemma_3b", "paligemma_10b", "paligemma_28b"):
    register_adapter(alias, VLMAdapter)

# VLM aliases — LLaVA variants
for alias in ("llava_15", "llava_15_7b", "llava_15_13b", "llava_ov", "llava_ov_7b"):
    register_adapter(alias, VLMAdapter)
