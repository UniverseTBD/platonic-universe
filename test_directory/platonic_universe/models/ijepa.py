"""
I-JEPA model implementation for Platonic Universe.
"""

from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import tqdm

from .base import BaseVisionModel, ModelLoader

try:
    from transformers import AutoModel, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class IJEPAModel(BaseVisionModel):
    """I-JEPA (Image-based Joint Embedding Predictive Architecture) model implementation."""
    
    def __init__(self, model_name: str = "ijepa", device: Optional[str] = None):
        super().__init__(model_name, device)
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for I-JEPA models")
        
    def load_model(self, model_id: str, **kwargs) -> None:
        """
        Load I-JEPA model and processor from HuggingFace.
        
        Args:
            model_id: HuggingFace model identifier (e.g., 'facebook/ijepa_vith14_1k')
            **kwargs: Additional arguments for model loading
        """
        try:
            # Load processor
            self.processor = AutoProcessor.from_pretrained(model_id, **kwargs)
            
            # Load model
            self.model = AutoModel.from_pretrained(model_id, **kwargs)
            
            # Move to device and set eval mode
            self.model = self.model.to(self.device).eval()
            
            # Enable DataParallel if multiple GPUs available
            if torch.cuda.device_count() > 1 and self.device == "cuda":
                self.model = nn.DataParallel(self.model)
            
            self._is_loaded = True
            
        except Exception as e:
            raise RuntimeError(f"Failed to load I-JEPA model '{model_id}': {e}")
    
    def preprocess_images(self, images: List[Image.Image], **kwargs) -> torch.Tensor:
        """
        Preprocess images using I-JEPA processor.
        
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
                        batch_size: int = 16,
                        feature_mode: str = "mean_pool",
                        **kwargs) -> np.ndarray:
        """
        Extract features from images using I-JEPA.
        
        Args:
            images: List of PIL Images
            batch_size: Batch size for processing (I-JEPA models are large, use smaller batches)
            feature_mode: Feature extraction mode ('mean_pool', 'cls', 'last_token')
            **kwargs: Additional arguments
            
        Returns:
            Feature embeddings of shape (n_images, feature_dim)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        features = []
        
        for i in tqdm.trange(0, len(images), batch_size, desc=f"I-JEPA-{feature_mode}"):
            batch_images = images[i:i + batch_size]
            
            try:
                # Preprocess batch
                pixel_batch = self.preprocess_images(batch_images)
                pixel_batch = pixel_batch.to(self.device)
                
                # Forward pass
                with torch.inference_mode():
                    outputs = self.model(pixel_values=pixel_batch)
                    
                    # Extract features based on mode
                    if hasattr(outputs, "last_hidden_state"):
                        hidden_states = outputs.last_hidden_state
                        
                        if feature_mode == "mean_pool":
                            # Mean pool across sequence dimension
                            batch_features = hidden_states.mean(dim=1)
                        elif feature_mode == "cls":
                            # Use first token (CLS token)
                            batch_features = hidden_states[:, 0, :]
                        elif feature_mode == "last_token":
                            # Use last token
                            batch_features = hidden_states[:, -1, :]
                        else:
                            # Default to mean pooling
                            batch_features = hidden_states.mean(dim=1)
                    else:
                        # Fallback: use outputs directly if it's a tensor
                        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                            batch_features = outputs.pooler_output
                        elif hasattr(outputs, "last_hidden_state"):
                            batch_features = outputs.last_hidden_state.mean(dim=1)
                        else:
                            # Last resort: try to extract from outputs
                            if isinstance(outputs, torch.Tensor):
                                batch_features = outputs
                            else:
                                raise RuntimeError("Could not extract features from I-JEPA outputs")
                    
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
        """Get information about the loaded I-JEPA model."""
        info = super().get_model_info()
        info.update({
            "processor_available": self.processor is not None,
            "transformers_available": TRANSFORMERS_AVAILABLE,
        })
        return info


# Register I-JEPA model
if TRANSFORMERS_AVAILABLE:
    ModelLoader.register_model("ijepa", IJEPAModel)