from typing import Any, Dict, Iterable, List

import torch

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
    
    Layer-by-layer extraction:
        SmolVLM's vision encoder (Idefics3VisionTransformer) consists of:
        - Patch embeddings layer (layer index 0)
        - N transformer encoder layers (indices 1 to N)
        - Post-layer normalization
        
        Use `embed_all_layers_for_mode()` to extract embeddings from all layers.
    """

    def __init__(self, model_name: str, size: str, alias: str = None):
        super().__init__(model_name, size, alias)
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is not installed. Please install it with: pip install transformers")
        self.processor = None
        self.model = None
        self._hooks = []
        self._layer_outputs = {}
        self.device = "cpu"  # Set in load()

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

    def supports_layerwise(self) -> bool:
        """SmolVLM supports layer-by-layer extraction via forward hooks."""
        return True

    def get_num_layers(self) -> int:
        """
        Return the total number of extractable layers in the vision encoder.
        
        For SmolVLM, this is:
        - 1 embedding layer (layer 0)
        - N encoder layers (layers 1 to N)
        
        Returns:
            int: Total number of layers (embeddings + encoder layers)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        vision_model = self.model.model.vision_model
        num_encoder_layers = len(vision_model.encoder.layers)
        return 1 + num_encoder_layers  # embeddings + encoder layers

    def get_layer_info(self) -> Dict[int, str]:
        """
        Return a mapping of layer indices to layer names.
        
        Returns:
            Dict[int, str]: Mapping of layer index to descriptive name
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        vision_model = self.model.model.vision_model
        num_encoder_layers = len(vision_model.encoder.layers)
        
        layer_info = {0: "embeddings"}
        for i in range(num_encoder_layers):
            layer_info[i + 1] = f"encoder.layer.{i}"
        
        return layer_info

    def _register_hooks(self) -> None:
        """
        Register forward hooks on all vision encoder layers to capture intermediate activations.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Clear any existing hooks
        self._remove_hooks()
        self._layer_outputs = {}
        
        vision_model = self.model.model.vision_model
        
        # Create hook factory
        def create_hook(layer_idx):
            def hook(module, input, output):
                # Handle both tuple and tensor outputs
                if isinstance(output, tuple):
                    self._layer_outputs[layer_idx] = output[0].detach()
                else:
                    self._layer_outputs[layer_idx] = output.detach()
            return hook
        
        # Register hook on embeddings layer (layer 0)
        h = vision_model.embeddings.register_forward_hook(create_hook(0))
        self._hooks.append(h)
        
        # Register hooks on encoder layers (layers 1 to N)
        for idx, layer in enumerate(vision_model.encoder.layers):
            h = layer.register_forward_hook(create_hook(idx + 1))
            self._hooks.append(h)

    def _remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._layer_outputs = {}

    def embed_all_layers_for_mode(
        self, 
        batch: Dict[str, Any], 
        mode: str,
        pool_method: str = "mean"
    ) -> Dict[int, torch.Tensor]:
        """
        Extract embeddings from ALL layers of the vision encoder.
        
        This method uses forward hooks to capture intermediate activations
        from each layer of the vision transformer.
        
        Args:
            batch: Dict from DataLoader containing preprocessed images
            mode: Dataset mode (e.g., 'hsc', 'jwst')
            pool_method: How to pool spatial dimensions:
                - "mean": Mean pooling over sequence dimension
                - "cls": Take first token (not applicable for SmolVLM)
                - "max": Max pooling over sequence dimension
        
        Returns:
            Dict[int, torch.Tensor]: Mapping of layer index to pooled embeddings
                - Key 0: Embeddings layer output
                - Keys 1 to N: Encoder layer outputs
                - Each tensor has shape (batch_size, hidden_dim)
        """
        # Register hooks before forward pass
        self._register_hooks()
        
        try:
            # Get pixel values and attention mask
            pixel_values = batch[f"{mode}_pixel_values"].to(self.device)
            pixel_attention_mask = batch.get(f"{mode}_pixel_attention_mask", None)
            if pixel_attention_mask is not None:
                pixel_attention_mask = pixel_attention_mask.to(self.device)
            
            with torch.no_grad():
                device_type = "cuda" if self.device == "cuda" else "cpu"
                with torch.amp.autocast(device_type, enabled=self._use_amp, dtype=torch.bfloat16):
                    # Forward pass - hooks will capture layer outputs
                    _ = self.model.get_image_features(
                        pixel_values=pixel_values,
                        pixel_attention_mask=pixel_attention_mask
                    )
            
            # Process captured layer outputs
            # SmolVLM processes images in patches: shape (num_patches, seq_len, hidden_dim)
            # We pool to get (hidden_dim,) per image
            # For proper batch handling, we treat num_patches as the batch dimension
            # since each image is split into patches
            
            layer_embeddings = {}
            for layer_idx, hidden_states in self._layer_outputs.items():
                # hidden_states shape: (num_patches, seq_len, hidden_dim)
                # Pool over seq_len first to get (num_patches, hidden_dim)
                # Then aggregate patches to get final embedding
                
                if pool_method == "mean":
                    # Mean pool over sequence, then over patches
                    emb = hidden_states.mean(dim=1)  # (num_patches, hidden_dim)
                    emb = emb.mean(dim=0)  # (hidden_dim,)
                elif pool_method == "max":
                    emb = hidden_states.amax(dim=1)  # (num_patches, hidden_dim)
                    emb = emb.amax(dim=0)  # (hidden_dim,)
                elif pool_method == "first_patch":
                    # Use first patch (original image) only
                    emb = hidden_states[0].mean(dim=0)  # (hidden_dim,)
                else:
                    # Default to mean
                    emb = hidden_states.mean(dim=(0, 1))
                
                # Ensure consistent shape: (1, hidden_dim) for batch dimension
                if emb.dim() == 1:
                    emb = emb.unsqueeze(0)
                
                # Convert to float32 for downstream computation
                layer_embeddings[layer_idx] = emb.float().detach()
            
            return layer_embeddings
            
        finally:
            # Always remove hooks to avoid memory leaks
            self._remove_hooks()

    def embed_all_layers_batch(
        self, 
        batch: Dict[str, Any], 
        mode: str,
        pool_method: str = "mean"
    ) -> List[torch.Tensor]:
        """
        Extract embeddings from ALL layers as a list (sorted by layer index).
        
        This is a convenience wrapper around embed_all_layers_for_mode that
        returns embeddings as a sorted list instead of a dict.
        
        Args:
            batch: Dict from DataLoader containing preprocessed images
            mode: Dataset mode (e.g., 'hsc', 'jwst')
            pool_method: How to pool spatial dimensions
        
        Returns:
            List[torch.Tensor]: List of embeddings, one per layer, sorted by layer index
        """
        layer_embeddings = self.embed_all_layers_for_mode(batch, mode, pool_method)
        return [layer_embeddings[i] for i in sorted(layer_embeddings.keys())]


# Register SmolVLM adapter
if TRANSFORMERS_AVAILABLE:
    register_adapter("smolvlm", SmolVLMAdapter)
