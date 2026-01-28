import torch
from typing import Any, Dict, Iterable
from pu.models.base import ModelAdapter
from pu.models.registry import register_adapter

try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class SmolVLMAdapter(ModelAdapter):
    """
    Adapter for SmolVLM (Vision Language Model) from HuggingFace.
    Extracts visual embeddings using the model's vision encoder.
    
    SmolVLM is a small, efficient multimodal model that processes both images and text.
    For this adapter, we extract only the visual features from the vision encoder.
    """

    def __init__(self, model_name: str, size: str, alias: str = None):
        super().__init__(model_name, size, alias)
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is not installed. Please install it with: pip install transformers")
        self.processor = None
        self.model = None

    def load(self, compile_model: bool = False) -> None:
        # Auto-detect device (GPU if available, else CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self.model.eval()

        # Apply torch.compile for optimized inference
        if compile_model:
            self.model = torch.compile(
                self.model,
                mode="reduce-overhead",
                fullgraph=False,
            )

    def get_preprocessor(self, modes: Iterable[str]):
        """
        Return a callable that preprocesses images for SmolVLM.
        SmolVLM uses an image processor that handles resizing and normalization.
        """
        def preprocess(example):
            """Preprocess function for SmolVLM that handles multiple modes."""
            result = {}
            for mode in modes:
                image_key = f"{mode}_image"
                if image_key in example:
                    # Get the image
                    image_data = example[image_key]
                    
                    # Handle different image formats
                    if isinstance(image_data, dict):
                        # If it's a dict with 'flux', extract the image
                        if "flux" in image_data:
                            import numpy as np
                            from PIL import Image
                            flux_data = image_data["flux"]
                            # Convert flux to PIL Image (assuming it's already in proper format)
                            if isinstance(flux_data, list):
                                flux_data = np.array(flux_data)
                            # Normalize and convert to uint8
                            if flux_data.dtype != np.uint8:
                                # Basic normalization - you may need to adjust this
                                flux_min, flux_max = np.nanpercentile(flux_data, [5, 99])
                                if flux_max - flux_min > 0:
                                    flux_data = ((flux_data - flux_min) / (flux_max - flux_min) * 255).astype(np.uint8)
                                else:
                                    flux_data = np.zeros_like(flux_data, dtype=np.uint8)
                            
                            # Handle multi-channel images (take first 3 channels or convert grayscale)
                            if len(flux_data.shape) == 3 and flux_data.shape[0] > 3:
                                # Take RGB channels (assuming channels-first format)
                                flux_data = flux_data[:3]
                            elif len(flux_data.shape) == 2:
                                # Convert grayscale to RGB
                                flux_data = np.stack([flux_data] * 3, axis=0)
                            
                            # Convert to channels-last for PIL
                            if len(flux_data.shape) == 3:
                                flux_data = np.transpose(flux_data, (1, 2, 0))
                            
                            image = Image.fromarray(flux_data)
                        else:
                            from PIL import Image
                            image = Image.fromarray(image_data)
                    else:
                        image = image_data
                    
                    # Process the image with SmolVLM's image processor
                    processed = self.processor.image_processor(
                        images=image,
                        return_tensors="pt"
                    )
                    
                    # Store processed tensors
                    result[f"{mode}_pixel_values"] = processed["pixel_values"].squeeze(0)
                    if "pixel_attention_mask" in processed:
                        result[f"{mode}_pixel_attention_mask"] = processed["pixel_attention_mask"].squeeze(0)
            
            return result
        
        return preprocess

    def embed_for_mode(self, batch: Dict[str, Any], mode: str):
        """
        Extract visual embeddings from SmolVLM's vision encoder.
        Uses the model's get_image_features() method to get pure visual features.
        """
        # Get pixel values and attention mask if available
        pixel_values = batch[f"{mode}_pixel_values"].to(self.device)
        pixel_attention_mask = batch.get(f"{mode}_pixel_attention_mask", None)
        if pixel_attention_mask is not None:
            pixel_attention_mask = pixel_attention_mask.to(self.device)
        
        with torch.no_grad():
            # Use AMP if enabled
            device_type = "cuda" if self.device == "cuda" else "cpu"
            with torch.amp.autocast(device_type, enabled=self._use_amp, dtype=torch.bfloat16):
                # Extract image features using the model's vision encoder
                image_outputs = self.model.get_image_features(
                    pixel_values=pixel_values,
                    pixel_attention_mask=pixel_attention_mask
                )
                
                # Get the last hidden state and pool it
                # SmolVLM returns BaseModelOutputWithPooling
                if hasattr(image_outputs, "last_hidden_state"):
                    hidden_states = image_outputs.last_hidden_state
                    # Pool over the sequence dimension (mean pooling)
                    emb = hidden_states.mean(dim=1)
                elif hasattr(image_outputs, "pooler_output"):
                    # Use pooler output if available
                    emb = image_outputs.pooler_output
                else:
                    # Fallback: treat output as tensor
                    if len(image_outputs.shape) == 3:  # (batch, seq, dim)
                        emb = image_outputs.mean(dim=1)
                    else:
                        emb = image_outputs
            
            # Always return float32 for downstream metric computation
            emb = emb.float().detach()
        
        return emb


# Register SmolVLM adapter
if TRANSFORMERS_AVAILABLE:
    register_adapter("smolvlm", SmolVLMAdapter)
