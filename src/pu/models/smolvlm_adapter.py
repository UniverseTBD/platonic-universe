import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from typing import Any, Dict, Iterable
from pu.models.base import ModelAdapter
from pu.preprocess import PreprocessHF
from pu.models.registry import register_adapter

class SmolVLMAdapter(ModelAdapter):
    """
    Adapter for SmolVLM vision-language models.
    
    The adapter extracts penultimate layer representations from either:
      - Vision tower only ('smolvlm-vision' alias)
      - Full multimodal model ('smolvlm' alias)
    
    Pooling strategy: Mean pooling over sequence dimension
    
    Supports:
      - torch.compile: Pass compile_model=True to load() for optimized inference
      - AMP: Call enable_amp(True) for float16 mixed precision inference
    """
    
    def __init__(self, model_name: str, size: str, alias: str = None):
        super().__init__(model_name, size, alias)
        self.processor = None
        self.model = None
        self.vision_tower = None
        self.captured = {}
        
    def load(self, compile_model: bool = False) -> None:
        """Load SmolVLM model and processor."""
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to("cuda").eval()
        
        # Get vision tower for vision-only mode
        self.vision_tower = self._get_vision_tower()
        
        # Apply torch.compile for optimized inference
        if compile_model:
            self.model = torch.compile(
                self.model,
                mode="reduce-overhead",
                fullgraph=False,
            )
    
    def _get_vision_tower(self):
        """Locate and return the vision encoder module."""
        for path in [
            "vision_model", "model.vision_model", "model.vision_encoder",
            "vision_encoder", "model.vision_tower", "vision_tower"
        ]:
            parts = path.split(".")
            obj = self.model
            for part in parts:
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                return obj
        raise RuntimeError("Could not locate vision tower in SmolVLM model")
    
    def _find_penultimate_layer(self, root):
        """Find the penultimate transformer layer."""
        best = None
        best_len = 0
        
        for _, mod in root.named_modules():
            if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 4:
                if len(mod) > best_len:
                    best = mod
                    best_len = len(mod)
        
        if best is None:
            raise RuntimeError("Failed to locate transformer blocks")
        
        # Return penultimate layer (second to last)
        return best[-2]
    
    def get_preprocessor(self, modes: Iterable[str]):
        """Return a callable compatible with datasets.Dataset.map"""
        # Use the image processor component for preprocessing
        image_processor = getattr(self.processor, "image_processor", None) or self.processor
        return PreprocessHF(modes, image_processor, resize=False)
    
    def embed_for_mode(self, batch: Dict[str, Any], mode: str):
        """
        Extract embeddings from penultimate layer.
        
        Uses vision tower for 'smolvlm-vision' alias, full model for 'smolvlm' alias.
        """
        # batch is a dict produced by the DataLoader
        inputs = batch[f"{mode}"].to("cuda")
        
        # Determine which encoder to use based on alias
        use_vision_only = self.alias == "smolvlm-vision"
        
        if use_vision_only:
            # Vision tower only
            target_module = self.vision_tower
            penult_layer = self._find_penultimate_layer(target_module)
        else:
            # Full multimodal model - exclude vision tower modules when finding LLM layers
            vision_ids = {id(x) for x in self.vision_tower.modules()}
            
            # Find LLM transformer blocks
            best = None
            best_len = 0
            for _, mod in self.model.named_modules():
                if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 4:
                    if len(mod) > 0 and id(mod[0]) not in vision_ids:
                        if len(mod) > best_len:
                            best = mod
                            best_len = len(mod)
            
            if best is None:
                raise RuntimeError("Failed to locate LLM transformer blocks")
            
            penult_layer = best[-2]
        
        # Register hook to capture penultimate layer output
        def hook_fn(module, input, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            self.captured["hidden"] = output
        
        handle = penult_layer.register_forward_hook(hook_fn)
        
        try:
            with torch.no_grad():
                # Use AMP if enabled
                with torch.amp.autocast("cuda", enabled=self._use_amp, dtype=torch.float16):
                    self.captured["hidden"] = None
                    
                    if use_vision_only:
                        # Forward through vision tower only
                        _ = target_module(pixel_values=inputs, return_dict=True)
                    else:
                        # For multimodal, need to create proper inputs with text prompts
                        # Create minimal prompt for each image
                        batch_size = inputs.shape[0]
                        
                        if hasattr(self.processor, "apply_chat_template"):
                            messages = [
                                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": " "}]}]
                                for _ in range(batch_size)
                            ]
                            prompts = [
                                self.processor.apply_chat_template(m, add_generation_prompt=False, tokenize=False)
                                for m in messages
                            ]
                        else:
                            prompts = [" "] * batch_size
                        
                        # Note: This is a simplified approach. For proper batching with varying text,
                        # you may need to handle tokenization separately
                        text_inputs = self.processor.tokenizer(
                            prompts,
                            return_tensors="pt",
                            padding=True
                        ).to("cuda")
                        
                        _ = self.model(
                            pixel_values=inputs,
                            input_ids=text_inputs["input_ids"],
                            attention_mask=text_inputs.get("attention_mask"),
                            return_dict=True
                        )
                    
                    # Get captured hidden states
                    hidden = self.captured["hidden"]
                    if hidden is None:
                        raise RuntimeError("Hook did not capture penultimate layer output")
                    
                    # Apply mean pooling
                    if hidden.dim() == 4:
                        # (B, C, H, W) -> mean over spatial dims
                        emb = hidden.mean(dim=(2, 3))
                    elif hidden.dim() == 3:
                        # (B, T, C) -> mean over token dim
                        emb = hidden.mean(dim=1)
                    else:
                        # (B, C) -> already pooled
                        emb = hidden
                    
                    # Always return float32 for downstream metric computation
                    emb = emb.float().detach()
        
        finally:
            handle.remove()
        
        return emb


# Register SmolVLM adapters
# Vision-only mode: uses vision tower penultimate layer
register_adapter("smolvlm-vision", SmolVLMAdapter)

# Multimodal mode: uses full VLM penultimate layer
register_adapter("smolvlm", SmolVLMAdapter)
