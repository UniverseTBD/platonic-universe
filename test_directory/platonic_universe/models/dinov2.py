"""
DinoV2 model implementation for Platonic Universe.
"""

from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import timm
import tqdm

from .base import BaseVisionModel, ModelLoader


class DinoV2Model(BaseVisionModel):
    """DinoV2 vision model implementation."""
    
    def __init__(self, model_name: str = "dinov2", device: Optional[str] = None):
        super().__init__(model_name, device)
        self.transform = None
        
    def load_model(self, model_id: str, num_classes: int = 0, **kwargs) -> None:
        """
        Load DinoV2 model using timm.
        
        Args:
            model_id: Model identifier (e.g., 'vit_base_patch14_dinov2.lvd142m')
            num_classes: Number of output classes (0 for feature extraction)
            **kwargs: Additional arguments for timm.create_model
        """
        try:
            # Create model
            self.model = timm.create_model(
                model_id, 
                pretrained=True, 
                num_classes=num_classes,
                **kwargs
            )
            
            # Move to device and set eval mode
            self.model = self.model.to(self.device).eval()
            
            # Enable DataParallel if multiple GPUs available
            if torch.cuda.device_count() > 1 and self.device == "cuda":
                self.model = nn.DataParallel(self.model)
            
            # Get data config and create transform
            cfg = timm.data.resolve_data_config({}, model=self.model)
            
            # Get actual model for patch_embed inspection
            actual_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            
            # Set image size based on patch_embed if available
            if hasattr(actual_model, "patch_embed") and hasattr(actual_model.patch_embed, "img_size"):
                img_size = actual_model.patch_embed.img_size
                if isinstance(img_size, (tuple, list)):
                    img_size = img_size[0]
                cfg.update(dict(input_size=(3, int(img_size), int(img_size))))
            
            # DinoV2 specific settings
            cfg.update(dict(crop_pct=1.0, interpolation="bicubic"))
            
            # Create transform
            self.transform = timm.data.create_transform(**cfg, is_training=False)
            
            self._is_loaded = True
            
        except Exception as e:
            raise RuntimeError(f"Failed to load DinoV2 model '{model_id}': {e}")
    
    def preprocess_images(self, images: List[Image.Image], **kwargs) -> torch.Tensor:
        """
        Preprocess images using DinoV2 transforms.
        
        Args:
            images: List of PIL Images
            **kwargs: Additional preprocessing arguments
            
        Returns:
            Preprocessed tensor of shape (batch_size, 3, H, W)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not self.transform:
            raise RuntimeError("Transform not initialized.")
        
        processed = []
        for img in images:
            try:
                tensor = self.transform(img)
                processed.append(tensor)
            except Exception:
                # Skip invalid images
                continue
        
        if not processed:
            raise ValueError("No valid images could be processed.")
        
        return torch.stack(processed)
    
    def _extract_dinov2_features(self, pixel_batch: torch.Tensor, 
                                mode: str = "patch_mean") -> torch.Tensor:
        """
        Extract features from DinoV2 model.
        
        Args:
            pixel_batch: Preprocessed image tensor
            mode: Feature extraction mode ('patch_mean' or 'cls')
            
        Returns:
            Feature tensor
        """
        actual_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        
        if hasattr(actual_model, "forward_features"):
            out = actual_model.forward_features(pixel_batch)
            
            if isinstance(out, dict):
                if mode == "patch_mean":
                    # Try to get patch tokens
                    if "x_norm_patchtokens" in out and out["x_norm_patchtokens"] is not None:
                        return out["x_norm_patchtokens"].mean(dim=1)
                    elif "x_norm" in out and isinstance(out["x_norm"], torch.Tensor) and out["x_norm"].ndim == 3:
                        return out["x_norm"].mean(dim=1)
                    # Fallback to CLS
                    elif "x_norm_clstoken" in out and out["x_norm_clstoken"] is not None:
                        return out["x_norm_clstoken"]
                elif mode == "cls":
                    # Try to get CLS token
                    if "x_norm_clstoken" in out and out["x_norm_clstoken"] is not None:
                        return out["x_norm_clstoken"]
                    # Fallback to patch mean
                    elif "x_norm_patchtokens" in out and out["x_norm_patchtokens"] is not None:
                        return out["x_norm_patchtokens"].mean(dim=1)
            elif isinstance(out, torch.Tensor):
                # Handle sequence format [CLS|patches]
                if out.ndim == 3 and out.shape[1] > 1:
                    if mode == "cls":
                        return out[:, 0, :]  # CLS token
                    else:
                        return out[:, 1:, :].mean(dim=1)  # Patch mean
                return out
        
        # Fallback to standard forward pass
        return self.model(pixel_batch)
    
    @torch.no_grad()
    def extract_features(self, images: List[Image.Image], 
                        batch_size: int = 32, 
                        feature_mode: str = "patch_mean",
                        **kwargs) -> np.ndarray:
        """
        Extract features from images using DinoV2.
        
        Args:
            images: List of PIL Images
            batch_size: Batch size for processing
            feature_mode: Feature extraction mode ('patch_mean' or 'cls')
            **kwargs: Additional arguments
            
        Returns:
            Feature embeddings of shape (n_images, feature_dim)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        features = []
        
        for i in tqdm.trange(0, len(images), batch_size, desc=f"DinoV2-{feature_mode}"):
            batch_images = images[i:i + batch_size]
            
            try:
                # Preprocess batch
                pixel_batch = self.preprocess_images(batch_images)
                pixel_batch = pixel_batch.to(self.device)
                
                # Extract features
                with torch.inference_mode():
                    batch_features = self._extract_dinov2_features(pixel_batch, feature_mode)
                    batch_features = batch_features.float().cpu().numpy()
                
                features.append(batch_features)
                
                # Cleanup
                del pixel_batch, batch_features
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
        """Get information about the loaded DinoV2 model."""
        info = super().get_model_info()
        info.update({
            "transform_available": self.transform is not None,
        })
        return info


# Register DinoV2 model
ModelLoader.register_model("dinov2", DinoV2Model)