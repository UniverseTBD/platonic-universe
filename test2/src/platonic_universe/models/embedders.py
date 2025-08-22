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

def get_embedder(loaded_model: LoadedModel) -> BaseEmbedder:
    """Factory function to get the correct embedder for a loaded model."""
    model_name = getattr(loaded_model.model, 'name_or_path', '').lower()

    if 'ijepa' in model_name:
        return IJEPABatchEmbedder(loaded_model)
    elif 'dinov2-with-registers' in model_name or 'dinov2' in model_name:
        return DINOv2WithRegistersBatchEmbedder(loaded_model)
    elif 'vit' in model_name or 'vision-transformer' in model_name:
        return ViTBatchEmbedder(loaded_model)
    elif loaded_model.transforms is not None: # Heuristic for timm models
        return TimmDINOv2BatchEmbedder(loaded_model)
    else:
        # Add more embedders here as you support more models
        raise NotImplementedError(f"No embedder implementation found for model: {model_name}")