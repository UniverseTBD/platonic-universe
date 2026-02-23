import torch
from typing import Any, Dict, Iterable

from pu.models.base import ModelAdapter
from pu.models.registry import register_adapter
from pu.preprocess import PreprocessHF
from transformers import AutoProcessor, AutoModelForImageTextToText


class PaliGemmaAdapter(ModelAdapter):
    """
    Adapter for PaliGemma2 models (google/paligemma2-{3b,10b,28b}-pt-224).

    Vision tower:   SigLIP ViT-SO400M — 27 transformer blocks
    Language model: Gemma2            — 18 / 46 / 64 blocks (3b / 10b / 28b)

    embed_for_mode returns mean-pooled hidden states captured via a forward
    hook on the requested component ("vision" or "llm") at the given layer.

    Batch keys consumed
    {mode}             : (B, C, H, W) pixel_values tensor  [from PreprocessHF]
    {mode}_component   : str  "vision" | "llm"   (default "vision")
    {mode}_layer       : int  block index, -1 = last layer (default -1)
    """

    _MODEL_IDS = {
        "3b":  "google/paligemma2-3b-pt-224",
        "10b": "google/paligemma2-10b-pt-224",
        "28b": "google/paligemma2-28b-pt-224",
    }

    def __init__(self, model_name: str, size: str, alias: str = None):
        super().__init__(model_name, size, alias)
        self.model     = None
        self.processor = None

    

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
        Reuses PreprocessHF from pu.preprocess.
        flux_to_pil is called internally; AutoProcessor runs on the result,
        returning {mode: pixel_values_tensor} — exactly what embed_for_mode expects.
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
        """SigLIP ViT-SO400M transformer blocks inside PaliGemma2."""
        candidates = [
            "model.vision_tower.vision_model.encoder.layers",
            "vision_tower.vision_model.encoder.layers",
        ]
        return self._resolve_path(candidates, "vision blocks")

    def _get_llm_blocks(self):
        """Gemma2 transformer blocks inside PaliGemma2."""
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
            f"PaliGemmaAdapter: cannot locate {label}. "
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

    def _extract_layer(
        self,
        pixel_values: torch.Tensor,
        component: str,
        blocks,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Hook one transformer block, run a forward pass, capture + pool output.

        Vision component: call SigLIP vision tower directly with pixel_values.
        PaliGemma2's vision tower accepts pixel_values standalone — no text
        tokens required — so no prompt injection is needed here.

        LLM component: run the full model forward with a minimal text prompt
        so image features are projected into the LLM sequence before the hook.
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
                    # Need a proper text encoding so image tokens are inserted.
                    # Use a blank prompt — processor injects the image token slots.
                    import numpy as np
                    from PIL import Image
                    B = pixel_values.shape[0]
                    dummy_imgs = [
                        Image.fromarray(
                            np.ones((224, 224, 3), dtype=np.uint8) * 128
                        )
                        for _ in range(B)
                    ]
                    enc = self.processor(
                        images=dummy_imgs,
                        text=[" "] * B,
                        return_tensors="pt",
                        padding=True,
                    )
                    input_ids = enc["input_ids"].to("cuda")
                    attn_mask = enc["attention_mask"].to("cuda")
                    self.model(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        attention_mask=attn_mask,
                        return_dict=True,
                    )
                    return self._pool(captured["h"], attn_mask)
        finally:
            handle.remove()

for _size in ("3b", "10b", "28b"):
    register_adapter(
        f"paligemma_{_size}",
        lambda mn, sz=_size, al=None: PaliGemmaAdapter(mn, size=sz, alias=al),
    )

register_adapter("paligemma", PaliGemmaAdapter)
