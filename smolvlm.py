

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from PIL import Image

try:
    from transformers import AutoProcessor, AutoModelForImageTextToText
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers")


class SmolVLMEncoder:
    
    
    # Model configurations matching HuggingFace hub
    MODEL_CONFIGS = {
        "small": "HuggingFaceTB/SmolVLM-256M-Instruct",
        "base": "HuggingFaceTB/SmolVLM-500M-Instruct",
        "large": "HuggingFaceTB/SmolVLM-Instruct",
        "giant": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    }
    
    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize SmolVLM encoder.
        
        Args:
            model_size: One of 'small', 'base', 'large', 'giant'
            device: Device to load model on (default: cuda if available)
            cache_dir: Cache directory for model weights
            dtype: Model dtype (default: bfloat16)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required. Install with: pip install transformers")
        
        if model_size not in self.MODEL_CONFIGS:
            raise ValueError(
                f"model_size must be one of {list(self.MODEL_CONFIGS.keys())}, "
                f"got {model_size}"
            )
        
        self.model_size = model_size
        self.model_id = self.MODEL_CONFIGS[model_size]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        
        print(f"Loading SmolVLM {model_size} ({self.model_id})...")
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            cache_dir=cache_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Setup hooks for penultimate layer extraction
        self._setup_hooks()
        
    def _setup_hooks(self):
        """Setup forward hooks to capture penultimate layer activations."""
        self.captured = {"vision_penult": None, "llm_penult": None}
        self.hooks = []
        
    def _get_vision_tower(self):
        """Get vision encoder module."""
        model = self.model
        for path in [
            "vision_model",
            "model.vision_model",
            "model.vision_encoder",
            "vision_encoder",
            "model.vision_tower",
            "vision_tower"
        ]:
            parts = path.split(".")
            obj = model
            for part in parts:
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                return obj
        raise RuntimeError("Could not locate vision tower in SmolVLM model")
    
    def _find_longest_modulelist(self, root: nn.Module, exclude_ids=None):
        """Find the longest ModuleList (transformer blocks)."""
        best = None
        best_len = 0
        exclude_ids = exclude_ids or set()
        
        for _, mod in root.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) >= 4:
                if len(mod) > 0 and id(mod[0]) in exclude_ids:
                    continue
                if len(mod) > best_len:
                    best = mod
                    best_len = len(mod)
        
        if best is None:
            raise RuntimeError(
                "Failed "
            )
        return best
    
    def _pool_mean(self, hidden_states, attention_mask=None):
        """Apply mean pooling to hidden states."""
        if hidden_states.dim() == 2:
            pooled = hidden_states
        else:
            if hidden_states.dim() == 4:
                hidden_states = hidden_states.mean(dim=1)
            
            if attention_mask is None:
                pooled = hidden_states.mean(dim=1)
            else:
                mask = attention_mask.float().unsqueeze(-1).type_as(hidden_states)
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        
        pooled = pooled.float().detach().cpu().numpy()
        return np.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
    
    @torch.inference_mode()
    def encode_vision(
        self,
        images: List[Image.Image],
        batch_size: int = 32,
        return_type: str = "vision_penult",
    ) -> np.ndarray:
       
        vision_tower = self._get_vision_tower()
        blocks = self._find_longest_modulelist(vision_tower)
        target_layer = blocks[-2]  # Penultimate layer
        
        def hook_fn(module, input, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            self.captured["vision_penult"] = output
        
        handle = target_layer.register_forward_hook(hook_fn)
        
        embeddings = []
        
        try:
            for i in range(0, len(images), batch_size):
                batch_imgs = images[i:i+batch_size]
                
                # Preprocess images
                image_processor = getattr(self.processor, "image_processor", None) or self.processor
                inputs = image_processor(images=batch_imgs, return_tensors="pt")
                pixel_values = inputs.get("pixel_values")
                
                if pixel_values is None:
                    raise RuntimeError("Image processor did not return pixel_values")
                
                # Handle different tensor shapes
                if pixel_values.ndim == 5 and pixel_values.shape[1] == 1:
                    pixel_values = pixel_values.squeeze(1)
                if pixel_values.ndim == 5:
                    b, n, c, h, w = pixel_values.shape
                    pixel_values = pixel_values.reshape(b * n, c, h, w)
                
                pixel_values = pixel_values.to(self.device, dtype=self.dtype)
                
                # Forward pass through vision tower
                self.captured["vision_penult"] = None
                _ = vision_tower(pixel_values=pixel_values, return_dict=True)
                
                hidden = self.captured["vision_penult"]
                if hidden is None:
                    raise RuntimeError("Hook did not capture penultimate activations")
                
                # Pool and collect
                pooled = self._pool_mean(hidden)
                embeddings.append(pooled)
                
                del pixel_values, hidden
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        finally:
            handle.remove()
        
        return np.concatenate(embeddings, axis=0)
    
    @torch.inference_mode()
    def encode_multimodal(
        self,
        images: List[Image.Image],
        batch_size: int = 4,
        return_type: str = "llm_penult",
    ) -> np.ndarray:
        """
        Encode images using full VLM (vision + language) penultimate layer.
        
        Args:
            images: List of PIL Images
            batch_size: Batch size for processing (smaller due to full model)
            return_type: 'llm_penult' for full VLM encoding
            
        Returns:
            numpy array of shape (n_images, embedding_dim)
        """
        vision_tower = self._get_vision_tower()
        vision_ids = {id(x) for x in vision_tower.modules()}
        
        # Find LLM blocks (excluding vision modules)
        blocks = self._find_longest_modulelist(self.model, exclude_ids=vision_ids)
        target_layer = blocks[-2]  # Penultimate layer
        
        def hook_fn(module, input, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            self.captured["llm_penult"] = output
        
        handle = target_layer.register_forward_hook(hook_fn)
        
        embeddings = []
        
        try:
            for i in range(0, len(images), batch_size):
                batch_imgs = images[i:i+batch_size]
                
                # Create prompts for VLM
                if hasattr(self.processor, "apply_chat_template"):
                    messages = [
                        [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": " "}]}]
                        for _ in batch_imgs
                    ]
                    prompts = [
                        self.processor.apply_chat_template(m, add_generation_prompt=False, tokenize=False)
                        for m in messages
                    ]
                else:
                    prompts = [" "] * len(batch_imgs)
                
                # Wrap images for processor
                images_wrapped = [[img] for img in batch_imgs]
                
                # Process inputs
                inputs = self.processor(
                    text=prompts,
                    images=images_wrapped,
                    return_tensors="pt",
                    padding=True
                )
                
                # Move to device
                inputs = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                }
                
                if "pixel_values" in inputs and torch.is_floating_point(inputs["pixel_values"]):
                    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.dtype)
                
                # Forward pass through full model
                self.captured["llm_penult"] = None
                _ = self.model(**inputs, return_dict=True)
                
                hidden = self.captured["llm_penult"]
                if hidden is None:
                    raise RuntimeError("Hook did not capture penultimate activations")
                
                # Pool with attention mask
                attn_mask = inputs.get("attention_mask", None)
                pooled = self._pool_mean(hidden, attention_mask=attn_mask)
                embeddings.append(pooled)
                
                del inputs, hidden
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        finally:
            handle.remove()
        
        return np.concatenate(embeddings, axis=0)
    
    def __call__(
        self,
        images: List[Image.Image],
        batch_size: int = 32,
        mode: str = "vision",
    ) -> np.ndarray:
        """
        Encode images with SmolVLM.
        
        Args:
            images: List of PIL Images
            batch_size: Batch size for processing
            mode: 'vision' for vision-only, 'multimodal' for full VLM
            
        Returns:
            numpy array of embeddings
        """
        if mode == "vision":
            return self.encode_vision(images, batch_size=batch_size)
        elif mode == "multimodal":
            return self.encode_multimodal(images, batch_size=batch_size)
        else:
            raise ValueError(f"mode must be 'vision' or 'multimodal', got {mode}")


def get_smolvlm_model(model_size: str = "base", **kwargs):
    """
    Factory function to create SmolVLM encoder.
    
    Compatible with Platonic Universe's model loading interface.
    
    Args:
        model_size: One of 'small', 'base', 'large', 'giant'
        **kwargs: Additional arguments passed to SmolVLMEncoder
        
    Returns:
        SmolVLMEncoder instance
    """
    return SmolVLMEncoder(model_size=model_size, **kwargs)


# Convenience aliases for different model sizes
def smolvlm_small(**kwargs):
    """SmolVLM Small (256M parameters)"""
    return get_smolvlm_model("small", **kwargs)


def smolvlm_base(**kwargs):
    """SmolVLM Base (500M parameters)"""
    return get_smolvlm_model("base", **kwargs)


def smolvlm_large(**kwargs):
    """SmolVLM Large (2B parameters)"""
    return get_smolvlm_model("large", **kwargs)


def smolvlm_giant(**kwargs):
    """SmolVLM Giant (2.2B parameters)"""
    return get_smolvlm_model("giant", **kwargs)


if __name__ == "__main__":
    # Example usage
    from PIL import Image
    import requests
    from io import BytesIO
    
    print("Testing SmolVLM encoder...")
    
    # Load a test image
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    
    # Test vision encoding
    encoder = smolvlm_small()
    embeddings_vision = encoder([img], mode="vision")
    print(f"Vision embeddings shape: {embeddings_vision.shape}")
    
    # Test multimodal encoding
    embeddings_mm = encoder([img], mode="multimodal", batch_size=1)
    print(f"Multimodal embeddings shape: {embeddings_mm.shape}")
    
    print("SmolVLM encoder test completed successfully!")