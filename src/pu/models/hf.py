from typing import Any, Dict, Iterable

import torch
import torch.nn as nn
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoVideoProcessor,
    CLIPModel,
    CLIPProcessor,
    HieraModel,
)

from pu.models.base import ModelAdapter
from pu.models.registry import register_adapter
from pu.preprocess import PreprocessHF

_CLIP_FAMILY = {"clip"}


class HFAdapter(ModelAdapter):
    """
    Adapter for HuggingFace vision models using AutoModel + AutoImageProcessor.
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

        if compile_model:
            self.model = torch.compile(
                self.model, mode="reduce-overhead", fullgraph=False,
            )

    def _get_hookable_model(self) -> nn.Module:
        if self.alias in _CLIP_FAMILY:
            return self.model.vision_model
        return self.model

    def get_preprocessor(self, modes: Iterable[str], resize: bool = False, resize_mode: str = "fill"):
        return PreprocessHF(modes, self.processor, alias=self.alias, resize=resize, resize_mode=resize_mode)

    def embed_for_mode(self, batch: Dict[str, Any], mode: str):
        inputs = batch[f"{mode}"].to("cuda")
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=self._use_amp, dtype=torch.float16):
                if self.alias == "clip":
                    outputs = self.model.get_image_features(pixel_values=inputs)
                    return outputs.float().detach()
                outputs = self.model(inputs).last_hidden_state
                if self.alias in ("vit", "vit-mae"):
                    emb = outputs[:, 1:].mean(dim=1)
                elif self.alias == "convnext":
                    emb = outputs.mean(dim=(2, 3))
                elif self.alias == "dino":
                    emb = outputs[:, 0]
                elif self.alias == "dinov3":
                    emb = outputs[:, 0, :]
                elif self.alias in ("ijepa", "vjepa", "hiera"):
                    emb = outputs.mean(dim=1)
                else:
                    emb = outputs.mean(dim=1)
            emb = emb.float().detach()
        return emb

    def supports_layerwise(self) -> bool:
        return True

    def _model_pool(self, t: torch.Tensor) -> torch.Tensor:
        """Pool using the same strategy as embed_for_mode."""
        if t.dim() == 4:
            return t.mean(dim=(2, 3))
        elif t.dim() == 3:
            if self.alias in ("vit", "vit-mae"):
                return t[:, 1:].mean(dim=1)
            elif self.alias in ("dino", "dinov3"):
                return t[:, 0]
            else:
                return t.mean(dim=1)
        elif t.dim() == 2:
            return t
        else:
            return t.reshape(t.shape[0], -1)

    def embed_all_layers_for_mode(
        self,
        batch: Dict[str, Any],
        mode: str,
    ) -> Dict[str, torch.Tensor]:
        inputs = batch[f"{mode}"].to("cuda")
        hookable = self._get_hookable_model()
        model_output = {}

        def forward_fn():
            with torch.amp.autocast("cuda", enabled=self._use_amp, dtype=torch.float16):
                out = hookable(inputs)
            # Capture last_hidden_state — this includes post-residual states
            # that no leaf module produces (residual add is in block forward, not a module)
            if hasattr(out, 'last_hidden_state') and out.last_hidden_state is not None:
                model_output['last_hidden_state'] = out.last_hidden_state

        results = self._capture_all_leaf_outputs(forward_fn, model=hookable, pool_fn=self._model_pool)

        # Add last_hidden_state as an explicit entry — guarantees exact match with embed_for_mode
        if 'last_hidden_state' in model_output:
            lhs = model_output['last_hidden_state']
            results["last_hidden_state"] = self._model_pool(lhs).float().detach()

        # For CLIP, also capture the visual projection (lives on CLIPModel, not vision_model)
        if self.alias in _CLIP_FAMILY and hasattr(self.model, 'visual_projection'):
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=self._use_amp, dtype=torch.float16):
                    projected = self.model.get_image_features(pixel_values=inputs)
            results["visual_projection"] = projected.float().detach()

        return results


class VLMAdapter(HFAdapter):
    """
    Adapter for vision-language models (PaliGemma2, LLaVA-1.5, LLaVA-OneVision).
    """

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

    def _get_hookable_model(self) -> nn.Module:
        return self.model

    def get_preprocessor(self, modes: Iterable[str], resize: bool = True, resize_mode: str = "match"):
        return PreprocessHF(modes, self.processor, alias=self.alias, resize=resize, resize_mode=resize_mode)

    def _prepare_vlm_inputs(self, batch, mode):
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
        B = pv.shape[0]
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
        pv_enc = enc["pixel_values"].to(device, dtype=model_dtype)
        return input_ids, pv_enc, attn_mask

    @staticmethod
    def _masked_mean_pool(hs, attn_mask):
        m = attn_mask.float().unsqueeze(-1)
        return ((hs * m).sum(1) / m.sum(1).clamp_min(1.0)).float().detach()

    def embed_for_mode(self, batch: Dict[str, Any], mode: str):
        input_ids, pv, attn_mask = self._prepare_vlm_inputs(batch, mode)
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids, pixel_values=pv,
                attention_mask=attn_mask, return_dict=True,
                output_hidden_states=True,
            )
        if hasattr(out, "hidden_states") and out.hidden_states is not None:
            return self._masked_mean_pool(out.hidden_states[-1], attn_mask)
        elif hasattr(out, "image_hidden_states") and out.image_hidden_states is not None:
            return out.image_hidden_states.mean(dim=1).float().detach()
        else:
            raise AttributeError(
                f"Cannot extract embeddings from {type(out).__name__}."
            )

    def embed_all_layers_for_mode(
        self,
        batch: Dict[str, Any],
        mode: str,
    ) -> Dict[str, torch.Tensor]:
        input_ids, pv, attn_mask = self._prepare_vlm_inputs(batch, mode)
        seq_len = attn_mask.shape[1]

        def pool_fn(t):
            # Use masked pooling for tensors matching the LM sequence length
            if t.dim() == 3 and t.shape[1] == seq_len:
                return self._masked_mean_pool(t, attn_mask)
            return self._generic_pool(t)

        def forward_fn():
            self.model(
                input_ids=input_ids, pixel_values=pv,
                attention_mask=attn_mask, return_dict=True,
            )

        return self._capture_all_leaf_outputs(forward_fn, pool_fn=pool_fn)


# Register adapters
for alias in ("vit", "dino", "dinov3", "convnext", "ijepa", "vjepa", "vit-mae", "hiera", "clip"):
    register_adapter(alias, HFAdapter)

for alias in ("paligemma", "paligemma_3b", "paligemma_10b", "paligemma_28b"):
    register_adapter(alias, VLMAdapter)

for alias in ("llava_15", "llava_15_7b", "llava_15_13b", "llava_ov", "llava_ov_7b"):
    register_adapter(alias, VLMAdapter)
