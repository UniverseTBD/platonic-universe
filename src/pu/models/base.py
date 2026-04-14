from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch
import torch.nn as nn


class ModelAdapter(ABC):
    """
    Minimal adapter interface for models used by experiments.
    Implementations should handle model loading, provide a preprocessor callable,
    and convert a batch from the DataLoader into embeddings.
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
    def get_preprocessor(self, modes: Iterable[str], resize: bool = False, resize_mode: str = "fill"):
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

    # --- Generic layer-wise extraction ---

    def supports_layerwise(self) -> bool:
        return False

    def _get_hookable_model(self) -> nn.Module:
        """Return the model subtree whose leaf modules get hooked.

        Override in subclasses if only part of the model should be hooked
        (e.g., CLIP's vision_model, SAM2's image_encoder).
        """
        return self.model

    @staticmethod
    def _generic_pool(t: torch.Tensor) -> torch.Tensor:
        """Pool any tensor to (batch, features)."""
        if t.dim() == 4:
            return t.mean(dim=(2, 3))
        elif t.dim() == 3:
            return t.mean(dim=1)
        elif t.dim() == 2:
            return t
        elif t.dim() == 1:
            return t.unsqueeze(0)
        else:
            return t.reshape(t.shape[0], -1)

    def _capture_all_leaf_outputs(
        self,
        forward_fn: Callable,
        model: Optional[nn.Module] = None,
        pool_fn: Optional[Callable] = None,
    ) -> Dict[str, torch.Tensor]:
        """Hook every leaf module, run a forward pass, return pooled outputs.

        Args:
            forward_fn: Callable that triggers the model forward pass.
            model: Module subtree to hook (default: self._get_hookable_model()).
            pool_fn: Custom pooling function (default: _generic_pool).

        Returns:
            Dict mapping module name to pooled (batch_size, dim) tensor.
        """
        target = model or self._get_hookable_model()
        pool = pool_fn or self._generic_pool
        results = {}
        hooks = []

        # Collect leaf module names in named_modules() order (deterministic DFS)
        leaf_names = []
        for name, mod in target.named_modules():
            if name and len(list(mod.children())) == 0:
                leaf_names.append(name)

        def _make_hook(name):
            def hook(module, input, output):
                t = output[0] if isinstance(output, tuple) else output
                if not isinstance(t, torch.Tensor) or t.dim() < 2:
                    return
                results[name] = pool(t).float().detach()
            return hook

        try:
            for name, mod in target.named_modules():
                if name and len(list(mod.children())) == 0:
                    h = mod.register_forward_hook(_make_hook(name))
                    hooks.append(h)

            with torch.no_grad():
                forward_fn()
        finally:
            for h in hooks:
                h.remove()

        # Reorder results to match named_modules() DFS order
        ordered = {}
        for name in leaf_names:
            if name in results:
                ordered[name] = results[name]
        return ordered

    def get_layer_names(self) -> List[str]:
        """Return ordered list of all hookable leaf module names."""
        target = self._get_hookable_model()
        return [
            name for name, mod in target.named_modules()
            if name and len(list(mod.children())) == 0
        ]

    def get_num_layers(self) -> int:
        return len(self.get_layer_names())

    def get_layer_info(self) -> Dict[str, str]:
        target = self._get_hookable_model()
        info = {}
        for name, mod in target.named_modules():
            if name and len(list(mod.children())) == 0:
                info[name] = mod.__class__.__name__
        return info

    def embed_all_layers_for_mode(
        self,
        batch: Dict[str, Any],
        mode: str,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Override in subclass")
