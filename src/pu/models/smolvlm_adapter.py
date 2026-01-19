import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from typing import Any, Dict, Iterable
from pu.models.base import ModelAdapter
from pu.preprocess import PreprocessHF
from pu.models.registry import register_adapter
try:
    from transformers import AutoProcessor, AutoModelForImageTextToText
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class SmolVLMAdapter(ModelAdapter):
    """
    The adapter extracts penultimate layer representations using hooks.
    Uses 'alias' to decide which representation to extract:
      - 'smolvlm-vision' -> Vision tower penultimate layer, mean pooled over tokens
      - 'smolvlm' -> Full VLM penultimate layer, mean pooled over tokens
    """
    
    def __init__(self, model_name: str, size: str, alias: str = None):
        super().__init__(model_name, size, alias)
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required. Install with:\n"
                "  pip install transformers"
            )
        self.processor = None
        self.model = None
        self.vision_tower = None
        self.penult_hook = None
        self.captured_hidden = None
        
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
        
        # Locate vision tower
        self.vision_tower = self._get_vision_tower()
        
        # Setup hook on penultimate layer
        self._setup_penultimate_hook()
        
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
    
    def _find_penultimate_layer(self, root, exclude_ids=None):
        """Find the penultimate transformer layer (second to last)."""
        exclude_ids = exclude_ids or set()
        best = None
        best_len = 0
        
        for _, mod in root.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) >= 4:
                if len(mod) > 0 and id(mod[0]) in exclude_ids:
                    continue
                if len(mod) > best_len:
                    best = mod
                    best_len = len(mod)
        
        if best is None:
            raise RuntimeError("Failed to locate transformer blocks")
        
        return best[-2]  # Penultimate layer
    
    def _setup_penultimate_hook(self):
        if self.alias == "smolvlm-vision":
            # Hook vision tower penultimate layer
            target = self._find_penultimate_layer(self.vision_tower)
        else:
            # Hook full VLM penultimate layer (excluding vision tower modules)
            vision_ids = {id(x) for x in self.vision_tower.modules()}
            target = self._find_penultimate_layer(self.model, exclude_ids=vision_ids)
        
        def hook_fn(module, input, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            self.captured_hidden = output
        
        self.penult_hook = target.register_forward_hook(hook_fn)
    
    def get_preprocessor(self, modes: Iterable[str]):
        # Use the image processor component
        image_processor = getattr(self.processor, "image_processor", None) or self.processor
        return PreprocessHF(modes, image_processor, resize=False)
    
    def embed_for_mode(self, batch: Dict[str, Any], mode: str):
        """
        Extract embeddings from penultimate layer.
        
        For 'smolvlm-vision': uses vision tower only
        For 'smolvlm': uses full VLM (requires text prompts)
        """
        # batch is a dict produced by the DataLoader
        inputs = batch[f"{mode}"].to("cuda")
        
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=self._use_amp, dtype=torch.float16):
                self.captured_hidden = None
                
                if self.alias == "smolvlm-vision":
                    # Vision tower only - simpler and faster
                    _ = self.vision_tower(pixel_values=inputs, return_dict=True)
                else:
                    # Full VLM - need text prompts
                    batch_size = inputs.shape[0]
                    
                    # Create minimal text prompts
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
                    
                    # Tokenize text
                    text_inputs = self.processor.tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding=True
                    ).to("cuda")
                    
                    # Forward through full model
                    _ = self.model(
                        pixel_values=inputs,
                        input_ids=text_inputs["input_ids"],
                        attention_mask=text_inputs.get("attention_mask"),
                        return_dict=True
                    )
                
                # Get captured hidden states from hook
                hidden = self.captured_hidden
                if hidden is None:
                    raise RuntimeError("Hook did not capture penultimate layer output")
                
                # Apply pooling based on tensor shape
                if hidden.dim() == 4:
                    # (B, C, H, W)  spatial mean
                    emb = hidden.mean(dim=(2, 3))
                elif hidden.dim() == 3:
                    # (B, T, C) mean over tokens
                    emb = hidden.mean(dim=1)
                else:
                    # (B, C) already pooled
                    emb = hidden
                
            # Always return float32 for downstream metric computation
            emb = emb.float().detach()
        
        return emb
    
    def __del__(self):
        """Clean up hook on deletion."""
        if self.penult_hook is not None:
            self.penult_hook.remove()
if TRANSFORMERS_AVAILABLE:
    register_adapter("smolvlm-vision", SmolVLMAdapter)
    register_adapter("smolvlm", SmolVLMAdapter)
