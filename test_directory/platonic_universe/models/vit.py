"""
Vision Transformer (ViT) model implementation for Platonic Universe.
"""

from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import tqdm

from .base import BaseVisionModel, ModelLoader

try:
    from transformers import ViTModel, ViTImageProcessor, AutoImageProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class ViTModel(BaseVisionModel):
    """Vision Transformer model implementation using HuggingFace transformers."""
    
    def __init__(self, model_name: str = "vit", device: Optional[str] = None):
        super().__init__(model_name, device)
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for ViT models")
        
    def load_model(self, model_id: str, use_safetensors: bool = True, **kwargs) -> None:
        """
        Load ViT model and processor from HuggingFace.
        
        Args:
            model_id: HuggingFace model identifier (e.g., 'google/vit-base-patch16-224')
            use_safetensors: Whether to use safetensors format
            **kwargs: Additional arguments for model loading
        """
        try:
            # Load processor
            try:
                self.processor = ViTImageProcessor.from_pretrained(model_id, **kwargs)
            except Exception:
                # Fallback to AutoImageProcessor
                self.processor = AutoImageProcessor.from_pretrained(model_id, **kwargs)
            
            # Load model
            self.model = ViTModel.from_pretrained(
                model_id,
                use_safetensors=use_safetensors,
                **kwargs
            )
            
            # Move to device and set eval mode
            self.model = self.model.to(self.device).eval()
            
            # Enable DataParallel if multiple GPUs available
            if torch.cuda.device_count() > 1 and self.device == "cuda":
                self.model = nn.DataParallel(self.model)
            
            self._is_loaded = True
            
        except Exception as e:
            raise RuntimeError(f"Failed to load ViT model '{model_id}': {e}")
    
    def preprocess_images(self, images: List[Image.Image], **kwargs) -> torch.Tensor:
        """
        Preprocess images using ViT processor.
        
        Args:
            images: List of PIL Images
            **kwargs: Additional preprocessing arguments
            
        Returns:
            Preprocessed tensor
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not self.processor:
            raise RuntimeError("Processor not initialized.")
        
        # Process images
        inputs = self.processor(images=images, return_tensors="pt", **kwargs)
        return inputs["pixel_values"]
    
    @torch.no_grad()
    def extract_features(self, images: List[Image.Image], 
                        batch_size: int = 32,
                        feature_mode: str = "cls",
                        last_n_layers: int = 1,
                        **kwargs) -> np.ndarray:
        """
        Extract features from images using ViT.
        
        Args:
            images: List of PIL Images
            batch_size: Batch size for processing
            feature_mode: Feature extraction mode ('cls', 'patch_mean', 'attn_pool')
            last_n_layers: Number of last layers to average (for multi-layer features)
            **kwargs: Additional arguments
            
        Returns:
            Feature embeddings of shape (n_images, feature_dim)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        features = []
        need_attention = (feature_mode == "attn_pool")
        
        for i in tqdm.trange(0, len(images), batch_size, desc=f"ViT-{feature_mode}"):
            batch_images = images[i:i + batch_size]
            
            try:
                # Preprocess batch
                pixel_batch = self.preprocess_images(batch_images)
                pixel_batch = pixel_batch.to(self.device)
                
                # Forward pass with appropriate outputs
                with torch.inference_mode():
                    if need_attention or last_n_layers > 1:
                        outputs = self.model(
                            pixel_batch,
                            output_hidden_states=True,
                            output_attentions=need_attention,
                            return_dict=True
                        )
                    else:
                        outputs = self.model(pixel_batch, return_dict=True)
                    
                    # Extract features based on mode
                    if feature_mode == "cls":
                        if last_n_layers > 1 and hasattr(outputs, "hidden_states"):
                            # Average CLS tokens from last N layers
                            hidden_states = outputs.hidden_states[-last_n_layers:]
                            cls_stack = torch.stack([h[:, 0, :] for h in hidden_states], dim=0)
                            batch_features = cls_stack.mean(dim=0)
                        else:
                            # Use CLS token from last layer
                            batch_features = outputs.last_hidden_state[:, 0, :]
                    
                    elif feature_mode == "patch_mean":
                        if last_n_layers > 1 and hasattr(outputs, "hidden_states"):
                            # Average patch tokens from last N layers
                            hidden_states = outputs.hidden_states[-last_n_layers:]
                            patch_stack = torch.stack([h[:, 1:, :].mean(dim=1) for h in hidden_states], dim=0)
                            batch_features = patch_stack.mean(dim=0)
                        else:
                            # Use patch tokens from last layer
                            batch_features = outputs.last_hidden_state[:, 1:, :].mean(dim=1)
                    
                    elif feature_mode == "attn_pool":
                        # Attention-weighted pooling using CLS attention to patches
                        if hasattr(outputs, "attentions") and outputs.attentions:
                            # Get attention from last layer, average over heads
                            att_last = outputs.attentions[-1].mean(dim=2)  # (B, T, T)
                            cls2patch = att_last[:, 0, 1:]  # (B, P) CLS to patch weights
                            cls2patch = cls2patch / (cls2patch.sum(dim=1, keepdim=True) + 1e-8)
                            
                            # Weighted sum of patch tokens
                            if last_n_layers > 1 and hasattr(outputs, "hidden_states"):
                                hidden_states = outputs.hidden_states[-last_n_layers:]
                                pooled_features = []
                                for h in hidden_states:
                                    patches = h[:, 1:, :]  # (B, P, D)
                                    weights = cls2patch.unsqueeze(-1)  # (B, P, 1)
                                    pooled_features.append((patches * weights).sum(dim=1))
                                batch_features = torch.stack(pooled_features, dim=0).mean(dim=0)
                            else:
                                patches = outputs.last_hidden_state[:, 1:, :]
                                weights = cls2patch.unsqueeze(-1)
                                batch_features = (patches * weights).sum(dim=1)
                        else:
                            # Fallback to patch mean if attention not available
                            batch_features = outputs.last_hidden_state[:, 1:, :].mean(dim=1)
                    
                    else:
                        raise ValueError(f"Unknown feature_mode: {feature_mode}")
                    
                    batch_features = batch_features.float().cpu().numpy()
                
                features.append(batch_features)
                
                # Cleanup
                del pixel_batch, outputs, batch_features
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Warning: Failed to process batch {i}: {e}")
                continue
        
        if not features:
            raise RuntimeError("No features could be extracted from any images.")
        
        final_features = np.concatenate(features, axis=0)
        
        # Clean NaN/inf values
        final_features = np.nan_to_num(final_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return final_features
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded ViT model."""
        info = super().get_model_info()
        info.update({
            "processor_available": self.processor is not None,
            "transformers_available": TRANSFORMERS_AVAILABLE,
        })
        return info


# Register ViT model
if TRANSFORMERS_AVAILABLE:
    ModelLoader.register_model("vit", ViTModel)