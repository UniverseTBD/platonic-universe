import torch
from typing import Any, Dict, Iterable, List
from pu.models.base import ModelAdapter
from pu.preprocess import PreprocessSAM2
from pu.models.registry import register_adapter

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False


class SAM2Adapter(ModelAdapter):
    """
    Adapter for SAM2 (Segment Anything Model 2) models.
    Uses SAM2's image encoder to extract image embeddings.

    Supports layer-by-layer extraction via forward hooks on the Hiera backbone
    trunk stages.

    Note: SAM2 must be installed separately. Install with:
        pip install git+https://github.com/facebookresearch/sam2.git
    """

    def __init__(self, model_name: str, size: str, alias: str = None):
        super().__init__(model_name, size, alias)
        if not SAM2_AVAILABLE:
            raise ImportError(
                "SAM2 is not installed. Please install it with:\n"
                "  pip install git+https://github.com/facebookresearch/sam2.git\n"
                "or:\n"
                "  cd /path/to/sam2 && SAM2_BUILD_CUDA=0 pip install -e ."
            )
        self.model = None
        self.predictor = None
        self.device = "cpu"  # Set in load()
        self._hooks = []
        self._layer_outputs = {}
        self._num_layers = 0  # Set after loading

    def load(self, compile_model: bool = False, force_cpu: bool = False, **kwargs) -> None:
        # Auto-detect device
        if force_cpu:
            self.device = "cpu"
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.predictor = SAM2ImagePredictor.from_pretrained(self.model_name)
        self.model = self.predictor.model
        self.model.to(self.device).eval()

        # Count layers from the Hiera backbone trunk
        self._num_layers = self._count_layers()

        if compile_model:
            self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)

    def _count_layers(self) -> int:
        """Count extractable layers in the Hiera backbone.

        SAM2's image encoder uses a Hiera trunk with multiple stages.
        We count: patch_embed (1) + all trunk blocks.
        """
        trunk = self.model.image_encoder.trunk
        # Hiera trunk has .blocks which is a flat list of all transformer blocks
        if hasattr(trunk, 'blocks'):
            return 1 + len(trunk.blocks)
        # Fallback: try stages
        if hasattr(trunk, 'stages'):
            total = 1
            for stage in trunk.stages:
                total += len(stage) if hasattr(stage, '__len__') else 1
            return total
        return 1

    def get_preprocessor(self, modes: Iterable[str]):
        # Return a callable compatible with datasets.Dataset.map
        return PreprocessSAM2(modes, self.predictor._transforms, resize=False)

    def embed_for_mode(self, batch: Dict[str, Any], mode: str):
        """
        Given a batch from the DataLoader and the mode name,
        return embeddings for that batch using SAM2's image encoder.
        """
        # batch contains preprocessed images under f"{mode}" key
        inputs = batch[f"{mode}"].to(self.device)

        with torch.no_grad():
	    # Case 1: user passed a list of numpy arrays (predictor expects that)
            if isinstance(inputs, list):
                # let the high-level predictor handle batching and transforms consistency
                # predictor.set_image_batch expects a List[np.ndarray]
                self.predictor.set_image_batch(inputs)
                emb = self.predictor.get_image_embedding()
                # get_image_embedding returns a list-like structure for batch case in predictor:
                # In predictor.set_image_batch it stores features as {"image_embed": feats[-1], ...}
                # and feats[-1] has shape (B, C, H_emb, W_emb)
                pooled = emb.mean(dim=(2, 3))
                return pooled

            # Case 2: inputs is a tensor (Bx3xHxW)
            if isinstance(inputs, torch.Tensor):
                img_batch = inputs.to(self.device)
                # forward through the model to get backbone outputs
                backbone_out = self.model.forward_image(img_batch)
                _, vision_feats, _, feat_sizes = self.model._prepare_backbone_features(
                    backbone_out
                )

                # Add no_mem_embed, which predictor does when directly_add_no_mem_embed is True
                if getattr(self.model, "directly_add_no_mem_embed", False):
                    vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

                batch_size = img_batch.shape[0]
                # same spatial sizes used in SAM2ImagePredictor
                bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
                feats = [
                    feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
                    for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
                ][::-1]
                # feats[-1] is the image embedding; feats[:-1] are high_res_feats
                image_embed = feats[-1].detach()
                pooled = image_embed.amax(dim=(2, 3))  # (B, C)
                return pooled

            raise TypeError(
                "Unsupported input type for SAM2Adapter.embed_for_mode: "
                f"{type(inputs)}. Expected torch.Tensor (Bx3xHxW) or List[np.ndarray]."
            )

    # --- Layer-wise extraction support ---

    def supports_layerwise(self) -> bool:
        """SAM2 supports layer-by-layer extraction via forward hooks on Hiera blocks."""
        return True

    def get_num_layers(self) -> int:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._num_layers

    def get_layer_info(self) -> Dict[int, str]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        layer_info = {0: "sam2.patch_embed"}
        trunk = self.model.image_encoder.trunk
        if hasattr(trunk, 'blocks'):
            for i in range(len(trunk.blocks)):
                layer_info[i + 1] = f"sam2.block.{i}"
        return layer_info

    def _register_hooks(self) -> None:
        """Register forward hooks on patch embedding and each Hiera block."""
        self._remove_hooks()
        self._layer_outputs = {}

        def create_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    self._layer_outputs[layer_idx] = output[0].detach()
                else:
                    self._layer_outputs[layer_idx] = output.detach()
            return hook

        trunk = self.model.image_encoder.trunk

        # Hook on patch embedding
        h = trunk.patch_embed.register_forward_hook(create_hook(0))
        self._hooks.append(h)

        # Hook on each block
        if hasattr(trunk, 'blocks'):
            for idx, block in enumerate(trunk.blocks):
                h = block.register_forward_hook(create_hook(idx + 1))
                self._hooks.append(h)

    def _remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self._layer_outputs = {}

    def _pool_hidden_state(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Pool hidden state to (batch_size, hidden_dim)."""
        if hidden_state.dim() == 4:
            # (B, C, H, W) -> spatial mean
            return hidden_state.mean(dim=(2, 3))
        elif hidden_state.dim() == 3:
            # (B, seq_len, dim) -> mean over seq
            return hidden_state.mean(dim=1)
        elif hidden_state.dim() == 2:
            return hidden_state
        else:
            return hidden_state.reshape(hidden_state.shape[0], -1)

    def embed_all_layers_for_mode(
        self,
        batch: Dict[str, Any],
        mode: str,
        pool_method: str = "mean"
    ) -> Dict[int, torch.Tensor]:
        """
        Extract embeddings from ALL layers of SAM2's Hiera backbone.

        Args:
            batch: Dict from DataLoader containing preprocessed inputs
            mode: Dataset mode (e.g., 'hsc', 'jwst')
            pool_method: How to pool spatial/sequence dimensions ("mean", "max")

        Returns:
            Dict[int, torch.Tensor]: Mapping of layer index to embeddings (batch_size, hidden_dim)
        """
        inputs = batch[f"{mode}"].to(self.device)
        layer_embeddings = {}

        self._register_hooks()
        try:
            with torch.no_grad():
                # Forward through the image encoder trunk
                _ = self.model.forward_image(inputs)

                for layer_idx, hidden_state in self._layer_outputs.items():
                    if pool_method == "max":
                        if hidden_state.dim() == 4:
                            emb = hidden_state.amax(dim=(2, 3))
                        elif hidden_state.dim() == 3:
                            emb = hidden_state.amax(dim=1)
                        else:
                            emb = hidden_state
                    else:
                        emb = self._pool_hidden_state(hidden_state)

                    layer_embeddings[layer_idx] = emb.float().detach()
        finally:
            self._remove_hooks()

        return layer_embeddings


# Register the adapter only if SAM2 is available
if SAM2_AVAILABLE:
    register_adapter("sam2", SAM2Adapter)
