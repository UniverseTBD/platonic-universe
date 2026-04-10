import torch
import numpy as np
from typing import Any, Dict, Iterable
from pu.models.base import ModelAdapter
from pu.models.registry import register_adapter


# LAMOST LRS wavelength grid that SpecCLIP was trained on
LAMOST_RANGE = (3700.0, 9100.0)
LAMOST_N_PIXELS = 1462


class PreprocessSpecCLIP:
    """Preprocessor that resamples DESI spectra to the LAMOST wavelength grid.

    SpecCLIP was trained on LAMOST LRS spectra (3700-9100 A, 1462 pixels).
    DESI spectra (3600-9800 A, 7781 pixels) are resampled via linear
    interpolation onto the LAMOST grid.  This is an intentional
    domain-mismatch control experiment.
    """

    def __init__(self, modes):
        self.modes = modes
        self.target_wavelengths = np.linspace(
            *LAMOST_RANGE, LAMOST_N_PIXELS
        )

    def __call__(self, idx):
        result = {}
        for mode in self.modes:
            if mode in ("desi", "sdss"):
                spectrum = idx["spectrum"]
                flux = np.array(spectrum["flux"], dtype=np.float32)
                wavelengths = np.array(spectrum["lambda"], dtype=np.float32)
                resampled = np.interp(self.target_wavelengths, wavelengths, flux)
                result["spectra"] = resampled.astype(np.float32)[..., np.newaxis]
        return result


class SpecCLIPAdapter(ModelAdapter):
    """Adapter for the SpecCLIP LAMOST LRS encoder.

    SpecCLIP is a masked transformer trained on LAMOST low-resolution
    spectra.  Running it on DESI spectra (with wavelength resampling) is
    an intentional out-of-domain control experiment -- the paper predicts
    this won't produce meaningful embeddings due to instrument systematics.
    """

    def __init__(self, model_name: str, size: str, alias: str = None):
        super().__init__(model_name, size, alias)
        self.model = None

    def load(self, compile_model: bool = False) -> None:
        from huggingface_hub import hf_hub_download
        from pu.models.specclip_arch import SpecCLIPEncoder

        checkpoint_path = hf_hub_download(
            repo_id=self.model_name, filename="encoders/lrs_encoder.ckpt"
        )
        checkpoint = torch.load(
            checkpoint_path, weights_only=False, map_location="cuda"
        )
        self.model = SpecCLIPEncoder(**checkpoint["hyper_parameters"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model = self.model.to("cuda").eval()

        if compile_model:
            self.model = torch.compile(
                self.model, mode="reduce-overhead", fullgraph=False
            )

    def get_preprocessor(
        self, modes: Iterable[str], resize: bool = False, resize_mode: str = "fill"
    ):
        return PreprocessSpecCLIP(modes)

    def embed_for_mode(self, batch: Dict[str, Any], mode: str):
        spectra = batch["spectra"].to("cuda")
        with torch.no_grad():
            with torch.amp.autocast(
                "cuda", enabled=self._use_amp, dtype=torch.float16
            ):
                output = self.model(spectra)
                # Mean-pool over tokens, excluding stats token at position 0
                embedding = output["embedding"][:, 1:, :].mean(dim=1)
        return embedding.float().detach()


register_adapter("specclip", SpecCLIPAdapter)
