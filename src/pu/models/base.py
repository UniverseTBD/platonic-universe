from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import random
import torch
import torch.nn as nn

# Extraction granularity modes
EXTRACT_BLOCKS = "blocks"        # Top-level blocks only (encoder.layer.N) — default, matches upstream PRH
EXTRACT_RESIDUAL = "residual"    # All non-leaf modules (residual stream at every sub-block)
EXTRACT_LEAVES = "leaves"        # Leaf modules only (Linear, GELU, LayerNorm, etc.)
EXTRACT_ALL = "all"              # Everything: leaves + all parent modules


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        """Return the model subtree whose modules get hooked.

        Override in subclasses if only part of the model should be hooked
        (e.g., CLIP's vision_model).
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

    @staticmethod
    def _is_leaf(mod: nn.Module) -> bool:
        return len(list(mod.children())) == 0

    @staticmethod
    def _is_block(name: str, mod: nn.Module, target: nn.Module) -> bool:
        """Check if a module is a top-level block (direct child of a ModuleList/Sequential,
        or a top-level named child like 'embeddings', 'pooler', 'layernorm').

        This identifies the modules whose outputs form the residual stream at
        the coarsest granularity — matching output_hidden_states behavior.
        """
        parts = name.split(".")
        # Depth 1: top-level children (embeddings, encoder, pooler, layernorm)
        if len(parts) == 1:
            return True
        # Depth 2+: check if parent is a ModuleList or Sequential
        # e.g., encoder.layer.0 where encoder.layer is a ModuleList
        parent_name = ".".join(parts[:-1])
        for pname, pmod in target.named_modules():
            if pname == parent_name:
                return isinstance(pmod, (nn.ModuleList, nn.Sequential))
        return False

    def _should_hook(self, name: str, mod: nn.Module, target: nn.Module, granularity: str) -> bool:
        """Decide whether to hook this module based on granularity mode."""
        if granularity == EXTRACT_BLOCKS:
            return self._is_block(name, mod, target)
        elif granularity == EXTRACT_RESIDUAL:
            return not self._is_leaf(mod)
        elif granularity == EXTRACT_LEAVES:
            return self._is_leaf(mod)
        elif granularity == EXTRACT_ALL:
            return True
        else:
            raise ValueError(f"Unknown granularity: {granularity}. "
                             f"Use: {EXTRACT_BLOCKS}, {EXTRACT_RESIDUAL}, {EXTRACT_LEAVES}, {EXTRACT_ALL}")

    def _capture_module_outputs(
        self,
        forward_fn: Callable,
        model: Optional[nn.Module] = None,
        pool_fn: Optional[Callable] = None,
        granularity: str = EXTRACT_BLOCKS,
    ) -> Dict[str, torch.Tensor]:
        """Hook modules at the specified granularity, run a forward pass, return pooled outputs.

        Args:
            forward_fn: Callable that triggers the model forward pass.
            model: Module subtree to hook (default: self._get_hookable_model()).
            pool_fn: Custom pooling function (default: _generic_pool).
            granularity: Extraction granularity level:
                - "blocks": Top-level blocks only (encoder.layer.N). Default.
                    Matches output_hidden_states / upstream PRH. ~14 points for ViT-base.
                - "residual": All non-leaf modules (full residual stream).
                    ~76 points for ViT-base.
                - "leaves": Leaf modules only (Linear, GELU, etc.).
                    ~137 points for ViT-base.
                - "all": Everything (leaves + all parent modules).
                    ~213 points for ViT-base.

        Returns:
            Dict mapping module name to pooled (batch_size, dim) tensor.
        """
        target = model or self._get_hookable_model()
        pool = pool_fn or self._generic_pool
        results = {}
        hooks = []

        hook_names = []
        for name, mod in target.named_modules():
            if name and self._should_hook(name, mod, target, granularity):
                hook_names.append(name)

        def _make_hook(name):
            def hook(module, input, output):
                t = output[0] if isinstance(output, tuple) else output
                if not isinstance(t, torch.Tensor) or t.dim() < 2:
                    return
                results[name] = pool(t).float().detach()
            return hook

        try:
            for name, mod in target.named_modules():
                if name and self._should_hook(name, mod, target, granularity):
                    h = mod.register_forward_hook(_make_hook(name))
                    hooks.append(h)

            with torch.no_grad():
                forward_fn()
        finally:
            for h in hooks:
                h.remove()

        ordered = {}
        for name in hook_names:
            if name in results:
                ordered[name] = results[name]
        return ordered

    # Keep old name as alias for backwards compatibility
    def _capture_all_leaf_outputs(self, forward_fn, model=None, pool_fn=None):
        return self._capture_module_outputs(
            forward_fn, model=model, pool_fn=pool_fn, granularity=EXTRACT_ALL
        )

    def get_layer_names(self, granularity: str = EXTRACT_BLOCKS) -> List[str]:
        """Return ordered list of hookable module names at the given granularity."""
        target = self._get_hookable_model()
        return [
            name for name, mod in target.named_modules()
            if name and self._should_hook(name, mod, target, granularity)
        ]

    def get_num_layers(self, granularity: str = EXTRACT_BLOCKS) -> int:
        return len(self.get_layer_names(granularity=granularity))

    def get_layer_info(self, granularity: str = EXTRACT_BLOCKS) -> Dict[str, str]:
        target = self._get_hookable_model()
        info = {}
        for name, mod in target.named_modules():
            if name and self._should_hook(name, mod, target, granularity):
                info[name] = mod.__class__.__name__
        return info

    def embed_all_layers_for_mode(
        self,
        batch: Dict[str, Any],
        mode: str,
        granularity: str = EXTRACT_BLOCKS,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("Override in subclass")
