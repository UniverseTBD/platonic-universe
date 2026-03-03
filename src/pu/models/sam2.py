import torch
from typing import Any, Dict, Iterable
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

    def load(self, compile_model: bool = False) -> None:
        # Build SAM2 model using the config and checkpoint
        # model_name is expected to be a Hugging Face model ID like "facebook/sam2.1-hiera-large"
        # We'll use the from_pretrained method to load from HuggingFace
        self.predictor = SAM2ImagePredictor.from_pretrained(self.model_name)
        self.model = self.predictor.model
        self.model.to("cuda").eval()

        if compile_model:
            self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)

    def get_preprocessor(self, modes: Iterable[str], resize: bool = False, resize_mode: str = "fill"):
        # Return a callable compatible with datasets.Dataset.map
        return PreprocessSAM2(modes, self.predictor._transforms, resize=resize, resize_mode=resize_mode)

    def embed_for_mode(self, batch: Dict[str, Any], mode: str):
        """
        Given a batch from the DataLoader and the mode name,
        return embeddings for that batch using SAM2's image encoder.
        """
        # batch contains preprocessed images under f"{mode}" key
        inputs = batch[f"{mode}"].to("cuda")

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
                img_batch = inputs.to("cuda")
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


# Register the adapter only if SAM2 is available
if SAM2_AVAILABLE:
    register_adapter("sam2", SAM2Adapter)
