from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional

class ModelAdapter(ABC):
    """
    Minimal adapter interface for models used by experiments.
    Implementations should handle model loading, provide a preprocessor callable,
    and convert a batch from the DataLoader into embeddings.
    
    For layer-by-layer analysis, adapters can optionally implement:
    - supports_layerwise(): Returns True if layer extraction is supported
    - get_num_layers(): Returns number of extractable layers
    - get_layer_info(): Returns layer index to name mapping
    - embed_all_layers_for_mode(): Extract embeddings from ALL layers
    """

    def __init__(self, model_name: str, size: str, alias: str = None):
        self.model_name = model_name
        self.size = size
        self.alias = alias
        self._use_amp = False

    @abstractmethod
    def load(self, compile_model: bool = False) -> None:
        """Load model and any required resources (to cuda if needed).

        Args:
            compile_model: If True, wrap model with torch.compile for optimization.
        """
        raise NotImplementedError

    def enable_amp(self, enabled: bool = True) -> None:
        """Enable or disable automatic mixed precision (AMP) for inference.

        Args:
            enabled: If True, use float16 for inference via torch.amp.autocast.
        """
        self._use_amp = enabled

    @abstractmethod
    def get_preprocessor(self, modes: Iterable[str]):
        """
        Return a callable that can be passed to `datasets.Dataset.map`.
        Should accept a single example dict and return a dict of tensors/arrays.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_for_mode(self, batch: Dict[str, Any], mode: str):
        """
        Given a batch from the DataLoader and the mode name (e.g., 'hsc' or 'jwst'),
        return a torch.Tensor (or array-like) with embeddings for that batch.
        """
        raise NotImplementedError

    # --- Optional layer-wise extraction methods ---
    
    def supports_layerwise(self) -> bool:
        """
        Check if this adapter supports layer-by-layer extraction.
        
        Returns:
            True if embed_all_layers_for_mode is implemented, False otherwise.
        """
        return False
    
    def get_num_layers(self) -> int:
        """
        Return the number of extractable layers. Must be implemented if
        supports_layerwise() returns True.
        
        Returns:
            Number of layers that can be extracted.
        """
        raise NotImplementedError("Layer-wise extraction not supported for this adapter")
    
    def get_layer_info(self) -> Dict[int, str]:
        """
        Return a mapping of layer indices to descriptive names.
        
        Returns:
            Dict mapping layer index (0-based) to layer name string.
        """
        raise NotImplementedError("Layer-wise extraction not supported for this adapter")
    
    def embed_all_layers_for_mode(
        self, 
        batch: Dict[str, Any], 
        mode: str,
        pool_method: str = "mean"
    ) -> Dict[int, Any]:
        """
        Extract embeddings from ALL layers of the model.
        
        Args:
            batch: Dict from DataLoader containing preprocessed inputs
            mode: Dataset mode (e.g., 'hsc', 'jwst')
            pool_method: How to pool spatial/sequence dimensions
        
        Returns:
            Dict mapping layer index to embeddings tensor.
        """
        raise NotImplementedError("Layer-wise extraction not supported for this adapter")
