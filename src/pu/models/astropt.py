import torch
from typing import Any, Dict, Iterable, List
from pu.models.base import ModelAdapter
from pu.preprocess import PreprocessAstropt
from pu.models.registry import register_adapter
from astropt.model_utils import load_astropt

class AstroptAdapter(ModelAdapter):
    """
    Adapter for astroPT models. Wraps `load_astropt` and uses `PreprocessAstropt`
    for preprocessing and the model's `generate_embeddings` for embedding.
    
    Supports layer-by-layer extraction via forward hooks on transformer encoder layers.
    """

    def __init__(self, model_name: str, size: str, alias: str = None):
        super().__init__(model_name, size, alias)
        self.model = None
        self.device = "cpu"  # Set in load()
        self._hooks = []
        self._layer_outputs = {}
        self._num_layers = 0  # Set after loading

    def load(self, compile_model: bool = False, force_cpu: bool = False, **kwargs) -> None:
        # Auto-detect device: CUDA > MPS > CPU
        if force_cpu:
            self.device = "cpu"
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        # follow previous code: model is loaded with a path containing the size
        self.model = load_astropt(self.model_name, path=f"astropt/{self.size}").to(self.device)
        self.model.eval()

        # Count layers for layerwise extraction
        self._num_layers = self._count_layers()

        # Apply torch.compile if requested
        if compile_model:
            self.model = torch.compile(
                self.model,
                mode="reduce-overhead",
                fullgraph=False,
            )

    def get_preprocessor(self, modes: Iterable[str]):
        # PreprocessAstropt needs the modality_registry from the loaded model
        return PreprocessAstropt(self.model.modality_registry, modes, resize=False)

    def embed_for_mode(self, batch: Dict[str, Any], mode: str):
        # Expects batch to contain f"{mode}_images" and f"{mode}_positions" as tensors
        inputs = {
            "images": batch[f"{mode}_images"].to(self.device),
            "images_positions": batch[f"{mode}_positions"].to(self.device),
        }
        with torch.no_grad():
            outputs = self.model.generate_embeddings(inputs)["images"].detach()
        return outputs

    # --- Layer-wise extraction support ---

    def _count_layers(self) -> int:
        """Count extractable layers in the astroPT transformer.

        AstroPT is a GPT-style architecture with:
        - Embeddings (wte + wpe) followed by dropout (transformer.drop)
        - N transformer blocks (transformer.h)
        - Final layer norm (transformer.ln_f)

        We count: 1 (embeddings/drop) + len(transformer.h) blocks.
        """
        if self.model is None:
            return 0

        # AstroPT uses model.transformer.h for its GPT-style blocks
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return 1 + len(self.model.transformer.h)

        return 0

    def supports_layerwise(self) -> bool:
        """AstroPT supports layer-by-layer extraction if transformer layers are accessible."""
        return self._num_layers > 0

    def get_num_layers(self) -> int:
        """
        Return the total number of extractable layers.
        
        Returns:
            int: Total number of layers (embeddings + encoder layers)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._num_layers

    def get_layer_info(self) -> Dict[int, str]:
        """
        Return a mapping of layer indices to layer names.
        
        Returns:
            Dict[int, str]: Mapping of layer index to descriptive name
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        layer_info = {0: "astropt.embeddings"}
        
        # Encoder layers
        for i in range(1, self._num_layers):
            layer_info[i] = f"astropt.encoder.{i-1}"
        
        return layer_info

    def _register_hooks(self) -> None:
        """
        Register forward hooks on all layers to capture intermediate activations.

        AstroPT architecture:
          model.transformer.drop  -> layer 0 (after embeddings + positional encoding)
          model.transformer.h[0]  -> layer 1
          model.transformer.h[1]  -> layer 2
          ...
          model.transformer.h[N-1] -> layer N
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Clear any existing hooks
        self._remove_hooks()
        self._layer_outputs = {}

        def create_hook(layer_idx):
            def hook(module, input, output):
                # Handle both tuple and tensor outputs
                if isinstance(output, tuple):
                    self._layer_outputs[layer_idx] = output[0].detach()
                else:
                    self._layer_outputs[layer_idx] = output.detach()
            return hook

        # Hook on dropout layer (captures output after embeddings + position encoding)
        if hasattr(self.model.transformer, 'drop'):
            h = self.model.transformer.drop.register_forward_hook(create_hook(0))
            self._hooks.append(h)

        # Hook on each transformer block (model.transformer.h)
        for idx, block in enumerate(self.model.transformer.h):
            h = block.register_forward_hook(create_hook(idx + 1))
            self._hooks.append(h)

    def _remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._layer_outputs = {}

    def _pool_hidden_state(self, hidden_state: torch.Tensor, pool_method: str = "mean") -> torch.Tensor:
        """
        Pool hidden state to (batch_size, hidden_dim).
        
        Args:
            hidden_state: Layer output tensor
            pool_method: Pooling method ('mean', 'max', 'first')
        
        Returns:
            Pooled tensor of shape (batch_size, hidden_dim)
        """
        if hidden_state.dim() == 4:
            # (B, C, H, W) -> spatial pooling
            if pool_method == "mean":
                return hidden_state.mean(dim=(2, 3))
            elif pool_method == "max":
                return hidden_state.amax(dim=(2, 3))
            else:
                return hidden_state.mean(dim=(2, 3))
        elif hidden_state.dim() == 3:
            # (B, seq_len, hidden_dim) -> sequence pooling
            if pool_method == "mean":
                return hidden_state.mean(dim=1)
            elif pool_method == "max":
                return hidden_state.amax(dim=1)
            elif pool_method == "first":
                return hidden_state[:, 0]
            else:
                return hidden_state.mean(dim=1)
        elif hidden_state.dim() == 2:
            # Already pooled (B, hidden_dim)
            return hidden_state
        else:
            # Fallback: flatten everything except batch dim
            return hidden_state.reshape(hidden_state.shape[0], -1)

    def embed_all_layers_for_mode(
        self, 
        batch: Dict[str, Any], 
        mode: str,
        pool_method: str = "mean"
    ) -> Dict[int, torch.Tensor]:
        """
        Extract embeddings from ALL layers of astroPT's transformer.
        
        This method uses forward hooks to capture intermediate activations
        from each layer of the transformer encoder.
        
        Args:
            batch: Dict from DataLoader containing preprocessed inputs
            mode: Dataset mode (e.g., 'hsc', 'jwst')
            pool_method: How to pool spatial/sequence dimensions:
                - "mean": Mean pooling over sequence dimension
                - "max": Max pooling over sequence dimension
                - "first": Take first token
        
        Returns:
            Dict[int, torch.Tensor]: Mapping of layer index to pooled embeddings
                - Each tensor has shape (batch_size, hidden_dim)
        """
        # Register hooks before forward pass
        self._register_hooks()
        
        try:
            inputs = {
                "images": batch[f"{mode}_images"].to(self.device),
                "images_positions": batch[f"{mode}_positions"].to(self.device),
            }
            
            with torch.no_grad():
                # Forward pass to capture all layer outputs
                _ = self.model.generate_embeddings(inputs)
            
            # Process captured layer outputs
            layer_embeddings = {}
            for layer_idx, hidden_states in self._layer_outputs.items():
                emb = self._pool_hidden_state(hidden_states, pool_method)
                
                # Ensure consistent shape: (batch_size, hidden_dim)
                if emb.dim() == 1:
                    emb = emb.unsqueeze(0)
                
                # Convert to float32 for downstream computation
                layer_embeddings[layer_idx] = emb.float().detach()
            
            return layer_embeddings

        finally:
            # Always remove hooks to avoid memory leaks
            self._remove_hooks()

# Register adapter
register_adapter("astropt", AstroptAdapter)
