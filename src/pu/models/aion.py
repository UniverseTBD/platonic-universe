import torch
import numpy as np
from typing import Any, Dict, Iterable
from pu.models.base import ModelAdapter
from pu.models.registry import register_adapter


class PreprocessAION:
    """Preprocessor that extracts DESI spectrum fields for AION processing."""

    def __init__(self, modes):
        self.modes = modes

    def __call__(self, idx):
        result = {}
        for mode in self.modes:
            if mode in ("desi", "sdss"):
                spectrum = idx["spectrum"]
                result["flux"] = np.array(spectrum["flux"], dtype=np.float32)
                result["ivar"] = np.array(spectrum["ivar"], dtype=np.float32)
                result["mask"] = np.array(spectrum["mask"], dtype=np.bool_)
                result["wavelength"] = np.array(spectrum["lambda"], dtype=np.float32)
        return result


class AIONAdapter(ModelAdapter):
    """Adapter for the AION multimodal foundation model (spectrum encoder).

    AION uses a ConvNeXt1d codec to tokenize spectra into discrete tokens,
    then a transformer encoder to produce continuous embeddings.  Trained on
    DESI spectra, so it is in-domain for the existing DESI data pipeline.

    Available sizes: base (300M), large (900M), xlarge (3B).
    """

    def __init__(self, model_name: str, size: str, alias: str = None):
        super().__init__(model_name, size, alias)
        self.model = None
        self.codec_manager = None

    def load(self, compile_model: bool = False) -> None:
        from aion import AION
        from aion.codecs import CodecManager

        self.model = AION.from_pretrained(self.model_name).to("cuda").eval()
        self.codec_manager = CodecManager(device="cuda")

        if compile_model:
            self.model = torch.compile(
                self.model, mode="reduce-overhead", fullgraph=False
            )

    def get_preprocessor(
        self, modes: Iterable[str], resize: bool = False, resize_mode: str = "fill"
    ):
        return PreprocessAION(modes)

    def _tokenize(self, batch):
        """Tokenize a batch of spectra using the AION codec."""
        from aion.modalities import DESISpectrum

        spectrum = DESISpectrum(
            flux=batch["flux"].to("cuda"),
            ivar=batch["ivar"].to("cuda"),
            mask=batch["mask"].bool().to("cuda"),
            wavelength=batch["wavelength"].to("cuda"),
        )
        return self.codec_manager.encode(spectrum)

    def embed_for_mode(self, batch: Dict[str, Any], mode: str):
        with torch.no_grad():
            with torch.amp.autocast(
                "cuda", enabled=self._use_amp, dtype=torch.float16
            ):
                tokens = self._tokenize(batch)
                embeddings = self.model.encode(
                    tokens, num_encoder_tokens=273
                )
                # Mean-pool over token dimension
                embedding = embeddings.mean(dim=1)
        return embedding.float().detach()


register_adapter("aion", AIONAdapter)
