import torch
from typing import Any, Dict, Iterable

from pu.models.base import ModelAdapter
from pu.models.registry import register_adapter
from pu.preprocess import PreprocessHF

from transformers import AutoProcessor, AutoModelForImageTextToText


class LLaVAAdapter(ModelAdapter):
    """
    Adapter for LLaVA-1.5 models via AutoModelForImageTextToText.

    checkpoints
    
    llava-1.5-7b  : llava-hf/llava-1.5-7b-hf   (CLIP ViT-L/14 + Vicuna-7b,  32 LLM blocks)
    llava-1.5-13b : llava-hf/llava-1.5-13b-hf  (CLIP ViT-L/14 + Vicuna-13b, 40 LLM blocks)
    llava-ov-7b   : llava-hf/llava-onevision-qwen2-7b-ov-hf
                    (SigLIP2 ViT, Qwen2-7B, 28 LLM blocks)

    
  
    LLaVA requires "<image>" in the prompt string so the processor inserts
    the correct number of image token embeddings into the sequence.
    Without it the model raises "number of image tokens is 0".

    For the LLM component we therefore always run the full model forward
    (never call the vision tower alone) so that image token placement is
    handled correctly by the model's own merge logic.

    Batch keys consumed
   
    {mode}             : (B, C, H, W) pixel_values tensor  [from PreprocessHF]
    {mode}_component   : str  "vision" | "llm"   (default "vision")
    {mode}_layer       : int  block index, -1 = last layer (default -1)
    """

    _MODEL_IDS = {
        "7b":    "llava-hf/llava-1.5-7b-hf",
        "13b":   "llava-hf/llava-1.5-13b-hf",
        "ov_7b": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    }

    # Minimal prompts that contain the image placeholder each variant expects
    _PROMPTS = {
        "7b":    "USER: <image>\n ASSISTANT:",
        "13b":   "USER: <image>\n ASSISTANT:",
        "ov_7b": "<image>",
    }

    def __init__(self, model_name: str, size: str, alias: str = None):
        super().__init__(model_name, size, alias)
        self.model     = None
        self.processor = None

    
    # ModelAdapter interface
  

    def load(self, compile_model: bool = False) -> None:
        model_id = self._MODEL_IDS.get(self.size, self.model_name)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        if compile_model:
            self.model = torch.compile(
                self.model, mode="reduce-overhead", fullgraph=False
            )

    def get_preprocessor(self, modes: Iterable[str]):
        """
        Reuses PreprocessHF from pu.preprocess — same as all other HF vision
        models in this repo. Returns {mode: pixel_values_tensor}.
        """
        return PreprocessHF(modes=modes, autoproc=self.processor)

    def embed_for_mode(self, batch: Dict[str, Any], mode: str) -> torch.Tensor:
        """
        Parameters
        
        batch[mode]           : (B, C, H, W) float tensor — pixel values from PreprocessHF
        batch[mode_component] : "vision" or "llm"  (optional, default "vision")
        batch[mode_layer]     : int block index, -1 = last (optional, default -1)

        Returns
        
        (B, D) float32 tensor — mean-pooled layer embedding.
        """
        pv = batch[mode].to("cuda")
        model_dtype = next(self.model.parameters()).dtype
        pv = pv.to(dtype=model_dtype)

        component = batch.get(f"{mode}_component", "vision")
        layer_idx = batch.get(f"{mode}_layer", -1)

        if component == "vision":
            blocks = self._get_vision_blocks()
        else:
            blocks = self._get_llm_blocks()

        n_blocks = len(blocks)
        if layer_idx < 0:
            layer_idx = n_blocks + layer_idx

        return self._extract_layer(pv, component, blocks, layer_idx).float()

    
    # Internal helpers
  

    def _get_vision_blocks(self):
        """
        LLaVA-1.5 and OV both put the vision encoder at:
        model.vision_tower.vision_model.encoder.layers
        (CLIP ViT-L/14 for 1.5; SigLIP2 for OV)
        """
        candidates = [
            "model.vision_tower.vision_model.encoder.layers",
            "vision_tower.vision_model.encoder.layers",
            "model.vision_tower.encoder.layers",
        ]
        return self._resolve_path(candidates, "vision blocks")

    def _get_llm_blocks(self):
        """
        LLaVA-1.5 : model.language_model.model.layers  (Vicuna/LLaMA)
        LLaVA-OV  : model.language_model.model.layers  (Qwen2)
        """
        candidates = [
            "model.language_model.model.layers",
            "language_model.model.layers",
            "model.model.layers",
        ]
        return self._resolve_path(candidates, "LLM blocks")

    def _resolve_path(self, candidates, label):
        for path in candidates:
            cur = self.model
            try:
                for attr in path.split("."):
                    cur = getattr(cur, attr)
                return cur
            except AttributeError:
                continue
        raise RuntimeError(
            f"LLaVAAdapter (size='{self.size}'): cannot locate {label}. "
            "Run print(self.model) and add the correct path to _resolve_path."
        )

    @staticmethod
    def _pool(hidden: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """Mean-pool over the sequence (token) dimension."""
        if hidden.dim() == 2:
            return hidden
        if attn_mask is None:
            return hidden.mean(dim=1)
        m = attn_mask.float().unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(1) / m.sum(1).clamp_min(1.0)

    def _build_text_encoding(self, pixel_values: torch.Tensor) -> dict:
        """
        Build tokenised input with the image placeholder injected.
        We pass dummy PIL images alongside the prompt so the processor
        correctly counts image tokens — but the actual visual content
        comes from the pixel_values tensor we pass to model.forward().
        """
        import numpy as np
        from PIL import Image
        B = pixel_values.shape[0]
        prompt = self._PROMPTS.get(self.size, "<image>")
        dummy_imgs = [
            Image.fromarray(np.ones((336, 336, 3), dtype=np.uint8) * 128)
            for _ in range(B)
        ]
        enc = self.processor(
            images=dummy_imgs,
            text=[prompt] * B,
            return_tensors="pt",
            padding=True,
        )
        return {
            "input_ids":      enc["input_ids"].to("cuda"),
            "attention_mask": enc["attention_mask"].to("cuda"),
        }

    def _extract_layer(
        self,
        pixel_values: torch.Tensor,
        component: str,
        blocks,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Vision component: call the vision tower directly — safe for LLaVA
        because the CLIP/SigLIP2 encoder does not depend on token placement.

        LLM component: must use full model forward with "<image>" in the
        prompt so the image-to-text projector merges visual tokens into the
        LLM sequence before the hook fires.
        """
        captured = {}

        def _hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if isinstance(h, torch.Tensor):
                captured["h"] = h.detach()

        handle = blocks[layer_idx].register_forward_hook(_hook)
        try:
            with torch.inference_mode():
                if component == "vision":
                    vt = self._resolve_path(
                        ["model.vision_tower", "vision_tower"], "vision tower"
                    )
                    vt(pixel_values=pixel_values, return_dict=True)
                    return self._pool(captured["h"])

                else:  # llm
                    enc = self._build_text_encoding(pixel_values)
                    self.model(
                        input_ids=enc["input_ids"],
                        pixel_values=pixel_values,
                        attention_mask=enc["attention_mask"],
                        return_dict=True,
                    )
                    return self._pool(captured["h"], enc["attention_mask"])
        finally:
            handle.remove()



for _size, _key in [
    ("7b",    "llava_15_7b"),
    ("13b",   "llava_15_13b"),
    ("ov_7b", "llava_ov_7b"),
]:
    register_adapter(
        _key,
        lambda mn, sz=_size, al=None: LLaVAAdapter(mn, size=sz, alias=al),
    )

register_adapter("llava", LLaVAAdapter)
