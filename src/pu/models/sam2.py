from typing import Any, Dict, Iterable

import torch
import torch.nn as nn

from pu.models.base import ModelAdapter
from pu.models.registry import register_adapter
from pu.preprocess import PreprocessSAM2

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

    Note: SAM2 must be installed separately. Install with:
        pip install git+https://github.com/facebookresearch/sam2.git
    """

    def __init__(self, model_name: str, size: str, alias: str = None):
        super().__init__(model_name, size, alias)
        if not SAM2_AVAILABLE:
            raise ImportError(
                "SAM2 is not installed. Please install it with:\n"
                "  pip install git+https://github.com/facebookresearch/sam2.git"
            )
        self.model = None
        self.predictor = None

    def load(self, compile_model: bool = False) -> None:
        self.predictor = SAM2ImagePredictor.from_pretrained(self.model_name)
        self.model = self.predictor.model
        self.model.to("cuda").eval()
        if compile_model:
            self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)

    def _get_hookable_model(self) -> nn.Module:
        return self.model.image_encoder

    def get_preprocessor(self, modes: Iterable[str], resize: bool = False, resize_mode: str = "fill"):
        return PreprocessSAM2(modes, self.predictor._transforms, resize=resize, resize_mode=resize_mode)

    def embed_for_mode(self, batch: Dict[str, Any], mode: str):
        """
        Given a batch from the DataLoader and the mode name,
        return embeddings for that batch using SAM2's image encoder.
        """
        inputs = batch[f"{mode}"].to("cuda")
        with torch.no_grad():
            if isinstance(inputs, list):
                self.predictor.set_image_batch(inputs)
                emb = self.predictor.get_image_embedding()
                return emb.mean(dim=(2, 3))

            if isinstance(inputs, torch.Tensor):
                backbone_out = self.model.forward_image(inputs)
                _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
                if getattr(self.model, "directly_add_no_mem_embed", False):
                    vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
                B = inputs.shape[0]
                bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
                feats = [
                    feat.permute(1, 2, 0).view(B, -1, *feat_size)
                    for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
                ][::-1]
                return feats[-1].detach().amax(dim=(2, 3))

            raise TypeError(f"Unsupported input type: {type(inputs)}")

    def supports_layerwise(self) -> bool:
        return True

    def get_layer_names(self) -> list:
        names = super().get_layer_names()
        names.append("embed_for_mode_output")
        return names

    def embed_all_layers_for_mode(
        self,
        batch: Dict[str, Any],
        mode: str,
    ) -> Dict[str, torch.Tensor]:
        inputs = batch[f"{mode}"].to("cuda")
        model_output = {}

        def forward_fn():
            backbone_out = self.model.forward_image(inputs)
            # Reproduce embed_for_mode's output exactly
            _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
            if getattr(self.model, "directly_add_no_mem_embed", False):
                vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
            B = inputs.shape[0]
            bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
            feats = [
                feat.permute(1, 2, 0).view(B, -1, *feat_size)
                for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
            ][::-1]
            model_output["emb"] = feats[-1].detach().amax(dim=(2, 3))

        results = self._capture_all_leaf_outputs(forward_fn)
        if "emb" in model_output:
            results["embed_for_mode_output"] = model_output["emb"].float()
        return results


if SAM2_AVAILABLE:
    register_adapter("sam2", SAM2Adapter)
