import torch
import numpy as np
from typing import Any, Dict, Iterable
from pu.models.base import ModelAdapter
from pu.models.registry import register_adapter


class PreprocessSpecformer:
    """Preprocessor that extracts galaxy spectra for SpecFormer processing.

    Converts raw spectrum flux arrays from HuggingFace dataset format into
    tensors suitable for the SpecFormer model.
    """

    def __init__(self, modes):
        self.modes = modes

    def __call__(self, idx):
        result = {}
        for mode in self.modes:
            if mode in ("desi", "sdss"):
                spectra = np.array(idx["spectrum"]["flux"], dtype=np.float32)[
                    ..., np.newaxis
                ]
                result["spectra"] = spectra
            # SpecFormer is spectral-only; image modes are skipped
        return result


class SpecformerAdapter(ModelAdapter):
    """Adapter for the AstroCLIP SpecFormer model for galaxy spectra.

    SpecFormer is a 1D transformer that encodes galaxy spectra via masked
    reconstruction pre-training.  It produces per-token embeddings which are
    mean-pooled (excluding the statistics token at position 0) to yield a
    fixed-size representation per galaxy.

    This adapter loads the pre-trained checkpoint from HuggingFace
    (``polymathic-ai/specformer``) and wraps the model for use within the
    platonic-universe experiment pipeline.
    """

    def __init__(self, model_name: str, size: str, alias: str = None):
        super().__init__(model_name, size, alias)
        self.model = None

    def load(self, compile_model: bool = False) -> None:
        from huggingface_hub import hf_hub_download
        from pu.models.specformer_arch import SpecFormer

        checkpoint_path = hf_hub_download(
            repo_id=self.model_name, filename="specformer.ckpt"
        )
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cuda")
        self.model = SpecFormer(**checkpoint["hyper_parameters"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model = self.model.to("cuda").eval()

        if compile_model:
            self.model = torch.compile(
                self.model, mode="reduce-overhead", fullgraph=False
            )

    def get_preprocessor(
        self, modes: Iterable[str], resize: bool = False, resize_mode: str = "fill"
    ):
        return PreprocessSpecformer(modes)

    def embed_for_mode(self, batch: Dict[str, Any], mode: str):
        spectra = batch["spectra"].to("cuda")
        with torch.no_grad():
            with torch.amp.autocast(
                "cuda", enabled=self._use_amp, dtype=torch.float16
            ):
                output = self.model(spectra)
                # Mean-pool over sequence tokens, excluding the statistics
                # token at position 0 (which encodes mean/std of the input)
                embedding = output["embedding"][:, 1:, :].mean(dim=1)
        return embedding.float().detach()

    def embed_layerwise(self, batch: Dict[str, Any], mode: str):
        """Extract per-layer embeddings, mean-pooled excluding stats token."""
        spectra = batch["spectra"].to("cuda")
        with torch.no_grad():
            output = self.model.forward_layerwise(spectra)
            return [
                emb[:, 1:, :].mean(dim=1).float().cpu()
                for emb in output["layer_embeddings"]
            ]


register_adapter("specformer", SpecformerAdapter)
