"""
Base classes for vision models in Platonic Universe.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import torch
import torch.nn as nn
import numpy as np
from PIL import Image


class BaseVisionModel(ABC):
    """Abstract base class for all vision models."""
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        Initialize the base vision model.
        
        Args:
            model_name: Name/identifier of the model
            device: Device to load model on ('cuda', 'cpu', etc.)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self._is_loaded = False
    
    @abstractmethod
    def load_model(self, model_id: str, **kwargs) -> None:
        """
        Load the model and processor.
        
        Args:
            model_id: HuggingFace model identifier or path
            **kwargs: Additional model-specific arguments
        """
        pass
    
    @abstractmethod
    def preprocess_images(self, images: List[Image.Image], **kwargs) -> torch.Tensor:
        """
        Preprocess images for the model.
        
        Args:
            images: List of PIL Images
            **kwargs: Additional preprocessing arguments
            
        Returns:
            Preprocessed tensor ready for model input
        """
        pass
    
    @abstractmethod
    def extract_features(self, images: List[Image.Image], 
                        batch_size: int = 32, **kwargs) -> np.ndarray:
        """
        Extract features from images.
        
        Args:
            images: List of PIL Images
            batch_size: Batch size for processing
            **kwargs: Additional extraction arguments
            
        Returns:
            Feature embeddings as numpy array of shape (n_images, feature_dim)
        """
        pass
    
    def __call__(self, images: List[Image.Image], **kwargs) -> np.ndarray:
        """Convenience method for feature extraction."""
        return self.extract_features(images, **kwargs)
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self._is_loaded
    
    def to(self, device: str) -> 'BaseVisionModel':
        """Move model to specified device."""
        self.device = device
        if self.model is not None:
            self.model = self.model.to(device)
        return self
    
    def eval(self) -> 'BaseVisionModel':
        """Set model to evaluation mode."""
        if self.model is not None:
            self.model.eval()
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "model_type": self.__class__.__name__,
        }


class ModelLoader:
    """Factory class for loading different types of vision models."""
    
    _model_registry = {}
    
    @classmethod
    def register_model(cls, model_type: str, model_class: type):
        """
        Register a new model type.
        
        Args:
            model_type: String identifier for the model type
            model_class: Class that implements BaseVisionModel
        """
        cls._model_registry[model_type.lower()] = model_class
    
    @classmethod
    def load_model(cls, model_type: str, model_id: str, 
                   device: Optional[str] = None, **kwargs) -> BaseVisionModel:
        """
        Load a model of the specified type.
        
        Args:
            model_type: Type of model to load ('dinov2', 'vit', 'ijepa')
            model_id: HuggingFace model identifier
            device: Device to load on
            **kwargs: Additional arguments passed to model constructor
            
        Returns:
            Loaded vision model instance
            
        Raises:
            ValueError: If model_type is not registered
        """
        model_type = model_type.lower()
        if model_type not in cls._model_registry:
            available = ", ".join(cls._model_registry.keys())
            raise ValueError(f"Unknown model type '{model_type}'. Available: {available}")
        
        model_class = cls._model_registry[model_type]
        model = model_class(model_type, device=device)
        model.load_model(model_id, **kwargs)
        return model
    
    @classmethod
    def available_models(cls) -> List[str]:
        """Get list of available model types."""
        return list(cls._model_registry.keys())


class BaseSpectralModel(ABC):
    """Abstract base class for spectral/1D data models."""
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        Initialize the base spectral model.
        
        Args:
            model_name: Name/identifier of the model
            device: Device to load model on
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._is_loaded = False
    
    @abstractmethod
    def load_model(self, model_path: str, **kwargs) -> None:
        """Load the spectral model."""
        pass
    
    @abstractmethod
    def preprocess_spectra(self, spectra: List[np.ndarray], **kwargs) -> torch.Tensor:
        """Preprocess spectral data for the model."""
        pass
    
    @abstractmethod
    def extract_features(self, spectra: List[np.ndarray], 
                        batch_size: int = 32, **kwargs) -> np.ndarray:
        """Extract features from spectra."""
        pass
    
    def __call__(self, spectra: List[np.ndarray], **kwargs) -> np.ndarray:
        """Convenience method for feature extraction."""
        return self.extract_features(spectra, **kwargs)
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self._is_loaded