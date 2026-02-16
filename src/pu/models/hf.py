import torch
from transformers import AutoModel, AutoImageProcessor
try:
    from transformers import AutoVideoProcessor
except ImportError:
    AutoVideoProcessor = None  # Not available in older transformers
try:
    from transformers import HieraModel
except ImportError:
    HieraModel = None  # Not available in older transformers
from typing import Any, Dict, Iterable, List
from pu.models.base import ModelAdapter
from pu.preprocess import PreprocessHF
from pu.models.registry import register_adapter

# Model families grouped by architecture for layer extraction
_VIT_FAMILY = {"vit", "dino", "dinov3", "ijepa", "vjepa", "vit-mae", "hiera"}
_CONVNEXT_FAMILY = {"convnext"}


class HFAdapter(ModelAdapter):
    """
    Adapter for HuggingFace vision models using AutoModel + AutoImageProcessor.
    The adapter uses the 'alias' passed at construction to decide pooling:
      - 'vit' -> CLS excluded mean over tokens (last_hidden_state[:,1:].mean)
      - 'dino' -> CLS token (last_hidden_state[:,0])
      - 'convnext' -> spatial mean over HxW (last_hidden_state.mean(dim=(2,3)))
      - 'ijepa' -> mean over token dim (last_hidden_state.mean(dim=1))
      - 'vjepa' -> mean over token dim (last_hidden_state.mean(dim=1))
      - 'vit-mae' -> CLS excluded mean over tokens (last_hidden_state[:,1:].mean)

    Supports:
      - torch.compile: Pass compile_model=True to load() for optimized inference
      - AMP: Call enable_amp(True) for float16 mixed precision inference
      - Layer-by-layer extraction: Uses output_hidden_states for ViT-family,
        hooks for ConvNeXt/Hiera
    """

    def __init__(self, model_name: str, size: str, alias: str = None):
        super().__init__(model_name, size, alias)
        self.processor = None
        self.model = None
        self._hooks = []
        self._layer_outputs = {}
        self.device = "cpu"  # Set in load()
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

        if self.alias == "vjepa":
            if AutoVideoProcessor is None:
                raise ImportError("AutoVideoProcessor requires transformers>=4.40. Please upgrade transformers or use a different model.")
            self.processor = AutoVideoProcessor.from_pretrained(self.model_name)
        else:
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)

        if self.alias == "hiera":
            if HieraModel is None:
                raise ImportError("HieraModel requires transformers>=4.40. Please upgrade transformers or use a different model.")
            self.model = HieraModel.from_pretrained(self.model_name).to(self.device).eval()
        else:
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()

        # Count extractable layers
        self._num_layers = self._count_layers()

        # Apply torch.compile for optimized inference
        if compile_model:
            self.model = torch.compile(
                self.model,
                mode="reduce-overhead",
                fullgraph=False,  # Allow graph breaks for complex HF models
            )

    def _count_layers(self) -> int:
        """Count the number of extractable layers based on model architecture."""
        if self.alias in _CONVNEXT_FAMILY:
            # ConvNeXt: embeddings + stages (each stage has multiple layers)
            # We extract one output per stage
            return 1 + len(self.model.encoder.stages)
        elif self.alias == "hiera":
            # Hiera: embeddings + stages
            if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'stages'):
                return 1 + len(self.model.encoder.stages)
            # Fallback: try to get from config
            return 1 + getattr(self.model.config, 'num_hidden_layers', 0)
        else:
            # ViT-family: embeddings + N encoder layers
            # output_hidden_states returns (embeddings, layer_1, ..., layer_N)
            if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
                return 1 + len(self.model.encoder.layer)
            # Fallback for models with different encoder naming
            return 1 + getattr(self.model.config, 'num_hidden_layers', 0)

    def get_preprocessor(self, modes: Iterable[str]):
        # Return a callable compatible with datasets.Dataset.map
        return PreprocessHF(modes, self.processor, resize=False)

    def embed_for_mode(self, batch: Dict[str, Any], mode: str):
        # batch is a dict produced by the DataLoader; HF preprocess stores tensors under f"{mode}"
        inputs = batch[f"{mode}"].to(self.device)
        with torch.no_grad():
            # Use AMP if enabled for faster inference with lower memory
            device_type = "cuda" if self.device == "cuda" else "cpu"
            with torch.amp.autocast(device_type, enabled=self._use_amp, dtype=torch.float16):
                outputs = self.model(inputs).last_hidden_state
                if self.alias == "vit" or self.alias == "vit-mae":
                    emb = outputs[:, 1:].mean(dim=1)
                elif self.alias == "convnext":
                    emb = outputs.mean(dim=(2, 3))
                elif self.alias == "dino":
                    emb = outputs[:, 0]
                elif self.alias == "dinov3":
                    emb = outputs[:, 0, :]
                elif self.alias == "ijepa":
                    emb = outputs.mean(dim=1)
                elif self.alias == "vjepa":
                    emb = outputs.mean(dim=1)
                elif self.alias == "hiera":
                    #  Hiera output is (B, 49, C).
                    # We pool over the sequence dimension (dim=1).
                    emb = outputs.mean(dim=1)
                else:
                    # Default fallback: mean over token dim excluding CLS if present
                    emb = outputs.mean(dim=1)

            # Always return float32 for downstream metric computation
            emb = emb.float().detach()
        return emb

    # --- Layer-wise extraction support ---

    def supports_layerwise(self) -> bool:
        """HFAdapter supports layer-by-layer extraction for all model families."""
        return True

    def get_num_layers(self) -> int:
        """Return the number of extractable layers."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._num_layers

    def get_layer_info(self) -> Dict[int, str]:
        """Return a mapping of layer indices to descriptive names."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        layer_info = {0: f"{self.alias}.embeddings"}

        if self.alias in _CONVNEXT_FAMILY:
            for i in range(self._num_layers - 1):
                layer_info[i + 1] = f"{self.alias}.stage.{i}"
        else:
            for i in range(self._num_layers - 1):
                layer_info[i + 1] = f"{self.alias}.encoder.{i}"

        return layer_info

    def _pool_hidden_state(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Pool a hidden state tensor to (batch_size, hidden_dim).

        Handles different tensor shapes across architectures:
        - ViT-family: (B, seq_len, hidden_dim) -> mean pool over seq_len
        - ConvNeXt: (B, C, H, W) -> mean pool over spatial dims
        """
        if hidden_state.dim() == 4:
            # ConvNeXt spatial features: (B, C, H, W)
            return hidden_state.mean(dim=(2, 3))
        elif hidden_state.dim() == 3:
            # ViT-family sequence: (B, seq_len, hidden_dim)
            return hidden_state.mean(dim=1)
        elif hidden_state.dim() == 2:
            # Already pooled: (B, hidden_dim)
            return hidden_state
        else:
            # Fallback: flatten everything except batch dim
            return hidden_state.reshape(hidden_state.shape[0], -1)

    def _register_hooks(self) -> None:
        """Register forward hooks for ConvNeXt/Hiera architectures."""
        self._remove_hooks()
        self._layer_outputs = {}

        def create_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    self._layer_outputs[layer_idx] = output[0].detach()
                else:
                    self._layer_outputs[layer_idx] = output.detach()
            return hook

        if self.alias in _CONVNEXT_FAMILY:
            # ConvNeXt: hook embeddings + each stage
            h = self.model.embeddings.register_forward_hook(create_hook(0))
            self._hooks.append(h)
            for idx, stage in enumerate(self.model.encoder.stages):
                h = stage.register_forward_hook(create_hook(idx + 1))
                self._hooks.append(h)
        elif self.alias == "hiera":
            if hasattr(self.model, 'embeddings'):
                h = self.model.embeddings.register_forward_hook(create_hook(0))
                self._hooks.append(h)
            if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'stages'):
                for idx, stage in enumerate(self.model.encoder.stages):
                    h = stage.register_forward_hook(create_hook(idx + 1))
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
        Extract embeddings from ALL layers of the model.

        For ViT-family models, uses output_hidden_states=True.
        For ConvNeXt/Hiera, uses forward hooks on each stage.

        Args:
            batch: Dict from DataLoader containing preprocessed inputs
            mode: Dataset mode (e.g., 'hsc', 'jwst')
            pool_method: How to pool spatial/sequence dimensions:
                - "mean": Mean pooling (default)
                - "max": Max pooling
                - "cls": CLS token (ViT-family only, falls back to mean for ConvNeXt)

        Returns:
            Dict[int, torch.Tensor]: Mapping of layer index to embeddings (batch_size, hidden_dim)
        """
        inputs = batch[f"{mode}"].to(self.device)

        layer_embeddings = {}

        with torch.no_grad():
            device_type = "cuda" if self.device == "cuda" else "cpu"
            use_autocast = self._use_amp and self.device == "cuda"

            with torch.amp.autocast(device_type, enabled=use_autocast, dtype=torch.float16):
                if self.alias in _CONVNEXT_FAMILY or self.alias == "hiera":
                    # Hook-based extraction for ConvNeXt/Hiera
                    self._register_hooks()
                    try:
                        _ = self.model(inputs)
                        for layer_idx, hidden_state in self._layer_outputs.items():
                            emb = self._pool_hidden_state(hidden_state)
                            layer_embeddings[layer_idx] = emb.float().detach()
                    finally:
                        self._remove_hooks()
                else:
                    # ViT-family: use output_hidden_states
                    outputs = self.model(inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states  # tuple of (B, seq_len, dim)

                    for layer_idx, hidden_state in enumerate(hidden_states):
                        if pool_method == "cls":
                            # CLS token (index 0)
                            emb = hidden_state[:, 0]
                        elif pool_method == "max":
                            emb = hidden_state.amax(dim=1)
                        else:
                            # Mean pooling (default)
                            emb = self._pool_hidden_state(hidden_state)

                        layer_embeddings[layer_idx] = emb.float().detach()

        return layer_embeddings


# Register this adapter for the HF-style aliases used by the repo
for alias in ("vit", "dino","dinov3", "convnext", "ijepa", "vjepa", "vit-mae","hiera"):
    register_adapter(alias, HFAdapter)
