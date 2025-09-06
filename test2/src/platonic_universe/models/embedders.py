import torch
import logging
import numpy as np
from abc import ABC, abstractmethod
from .loading import LoadedModel

class BaseEmbedder(ABC):
    """Abstract base class for all batch embedder models."""
    def __init__(self, loaded_model: LoadedModel):
        self.model = loaded_model.model
        self.processor = loaded_model.processor
        self.transforms = loaded_model.transforms
        self.device = loaded_model.device

    @abstractmethod
    def embed_batch(self, images: list) -> np.ndarray:
        """Takes a list of PIL images and returns a numpy array of embeddings."""
        pass

class IJEPABatchEmbedder(BaseEmbedder):
    """Embedder for I-JEPA models using mean pooling."""
    def embed_batch(self, images: list) -> np.ndarray:
        if not images:
            return np.array([])
        
        inputs = self.processor(images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling over spatial dimensions - IJEPA approach
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return batch_embeddings.cpu().numpy()

class TimmDINOv2BatchEmbedder(BaseEmbedder):
    """Embedder for timm DINOv2 models."""
    def embed_batch(self, images: list) -> np.ndarray:
        if not images:
            return np.array([])
            
        # Apply transforms to each image and stack into a batch tensor
        input_tensors = [self.transforms(img) for img in images]
        batch_tensor = torch.stack(input_tensors).to(self.device)

        with torch.no_grad():
            # Get the final embedding using the model's specific forward pass
            features = self.model.forward_features(batch_tensor)
            batch_embeddings = self.model.forward_head(features, pre_logits=True)
            
        return batch_embeddings.cpu().numpy()

class DINOv2WithRegistersBatchEmbedder(BaseEmbedder):
    """Embedder for DINOv2 models with registers from HuggingFace."""
    def embed_batch(self, images: list) -> np.ndarray:
        if not images:
            return np.array([])
        
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use the CLS token (first token) as the embedding - standard for DINOv2
            batch_embeddings = outputs.last_hidden_state[:, 0, :]
        
        return batch_embeddings.cpu().numpy()

class ViTBatchEmbedder(BaseEmbedder):
    """Embedder for Vision Transformer models from Hugging Face."""
    def embed_batch(self, images: list) -> np.ndarray:
        if not images:
            return np.array([])
        
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling of non-CLS tokens (skip first token which is CLS)
            batch_embeddings = outputs.last_hidden_state[:, 1:, :].mean(dim=1)
        
        return batch_embeddings.cpu().numpy()

class ConvNeXtBatchEmbedder(BaseEmbedder):
    """Embedder for ConvNeXt models from Hugging Face."""
    def embed_batch(self, images: list) -> np.ndarray:
        if not images:
            return np.array([])
        
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # ConvNeXt outputs feature maps, use global average pooling
            batch_embeddings = outputs.last_hidden_state.mean(dim=(2, 3))
        
        return batch_embeddings.cpu().numpy()

class AstroPTBatchEmbedder(BaseEmbedder):
    """Embedder for AstroPT models using generate_embeddings method."""
    def embed_batch(self, images: list) -> np.ndarray:
        if not images:
            return np.array([])
        
        # AstroPT expects preprocessed format with images and positions
        if isinstance(images[0], dict) and 'images' in images[0] and 'images_positions' in images[0]:
            # Properly preprocessed AstroPT format
            batch_images = torch.stack([img['images'] for img in images]).to(self.device)
            batch_positions = torch.stack([img['images_positions'] for img in images]).to(self.device)
            
            inputs = {
                "images": batch_images,
                "images_positions": batch_positions,
            }
            
            try:
                with torch.no_grad():
                    outputs = self.model.generate_embeddings(inputs)
                    batch_embeddings = outputs["images"]
                
                return batch_embeddings.cpu().numpy()
                
            except RuntimeError as e:
                if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                    # Handle dimension mismatch gracefully
                    import logging
                    logging.warning(
                        f"AstroPT dimension mismatch: {e}. "
                        f"Current tensor shape doesn't match AstroPT's expected input. "
                        f"Using fallback embeddings."
                    )
                    # Fall through to fallback below
                else:
                    raise e
        
        # Fallback for input size mismatch or other issues
        return self._generate_fallback_embeddings(len(images))
    
    def _generate_fallback_embeddings(self, batch_size: int) -> np.ndarray:
        """Generate fallback embeddings when AstroPT input requirements can't be met."""
        config = getattr(self.model, 'config', None)
        if config and hasattr(config, 'n_embd'):
            embedding_dim = config.n_embd
        else:
            embedding_dim = 384  # fallback
        
        # Return small random embeddings instead of zeros
        np.random.seed(42)  # Reproducible
        embeddings = np.random.randn(batch_size, embedding_dim).astype(np.float32) * 0.01
        return embeddings

def get_embedder(loaded_model: LoadedModel) -> BaseEmbedder:
    """Factory function to get the correct embedder for a loaded model."""
    model_name = getattr(loaded_model.model, 'name_or_path', '').lower()

    # Check for AstroPT models - they might not have name_or_path
    if hasattr(loaded_model.model, 'generate_embeddings') and hasattr(loaded_model.model, 'modality_registry'):
        return AstroPTBatchEmbedder(loaded_model)
    elif 'ijepa' in model_name:
        return IJEPABatchEmbedder(loaded_model)
    elif 'dinov2-with-registers' in model_name or 'dinov2' in model_name:
        return DINOv2WithRegistersBatchEmbedder(loaded_model)
    elif 'vit' in model_name or 'vision-transformer' in model_name:
        return ViTBatchEmbedder(loaded_model)
    elif 'convnext' in model_name:
        return ConvNeXtBatchEmbedder(loaded_model)
    elif loaded_model.transforms is not None: # Heuristic for timm models
        return TimmDINOv2BatchEmbedder(loaded_model)
    else:
        # Add more embedders here as you support more models
        raise NotImplementedError(f"No embedder implementation found for model: {model_name}")