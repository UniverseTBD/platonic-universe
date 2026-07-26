"""Moirai adapter (time-series modality).

Adds Salesforce Moirai (Woo et al., https://github.com/SalesforceAIResearch/uni2ts)
as a masked-encoder, any-variate time-series foundation model. Moirai patchifies
a series and runs a masked transformer encoder; we take the encoder's per-patch
representation, mean-pooled over patches, as the embedding.

Moirai's ``forward`` consumes a packed representation (target / observed_mask /
sample_id / time_id / variate_id / prediction_mask / patch_size) and returns a
forecast distribution, so the embedding is captured from the encoder via a hook
during a context-only (no-prediction) forward. This adapter handles univariate
series ``(B, C, T)`` by taking the first channel.

Install with ``pip install ".[moirai]"`` (uni2ts).
"""

import importlib.util
from typing import Any, Dict

import torch

from pu.models.base import ModelAdapter, resolve_device
from pu.models.registry import register_adapter

MOIRAI_AVAILABLE = importlib.util.find_spec("uni2ts") is not None


class MoiraiAdapter(ModelAdapter):
    """Adapter for Salesforce Moirai (``uni2ts`` ``MoiraiModule``)."""

    _PREFERRED_PATCH = 16

    def __init__(self, model_name: str, size: str, alias: str = None):
        super().__init__(model_name, size, alias)
        self.model = None
        self.device = None
        self.patch_size = None
        self.max_patch_size = None

    def load(self, compile_model: bool = False) -> None:
        try:
            from uni2ts.model.moirai import MoiraiModule
        except ImportError as exc:
            raise ImportError(
                "uni2ts is not installed. Install the extra with:\n"
                '  pip install ".[moirai]"   (or: pip install uni2ts)'
            ) from exc

        self.device = resolve_device()
        self.model = MoiraiModule.from_pretrained(self.model_name).to(self.device).eval()
        sizes = list(self.model.patch_sizes)
        self.max_patch_size = max(sizes)
        self.patch_size = self._PREFERRED_PATCH if self._PREFERRED_PATCH in sizes else sizes[len(sizes) // 2]
        if compile_model:
            self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)

    def get_preprocessor(self, modes, resize: bool = False, resize_mode: str = "fill"):
        from pu.models.timeseries import PreprocessTimeSeries
        return PreprocessTimeSeries(modes)

    def _get_hookable_model(self):
        return self.model.encoder

    def _pack(self, series: torch.Tensor):
        """Patchify a ``(B, C, T)`` (or ``(B, T)``) batch into Moirai's packed inputs."""
        x = series[:, 0, :] if series.dim() == 3 else series
        x = x.to(self.device).float()
        B, T = x.shape
        psz, mps = self.patch_size, self.max_patch_size
        n_patch = T // psz
        if n_patch < 1:
            raise ValueError(f"series length {T} shorter than patch size {psz}")
        x = x[:, : n_patch * psz].reshape(B, n_patch, psz)

        dev = self.device
        target = torch.zeros(B, n_patch, mps, device=dev)
        target[..., :psz] = x
        observed = torch.zeros(B, n_patch, mps, dtype=torch.bool, device=dev)
        observed[..., :psz] = True
        sample_id = torch.ones(B, n_patch, dtype=torch.long, device=dev)
        time_id = torch.arange(n_patch, device=dev).unsqueeze(0).repeat(B, 1)
        variate_id = torch.zeros(B, n_patch, dtype=torch.long, device=dev)
        prediction_mask = torch.zeros(B, n_patch, dtype=torch.bool, device=dev)
        patch_size = torch.full((B, n_patch), psz, dtype=torch.long, device=dev)
        return (target, observed, sample_id, time_id, variate_id, prediction_mask, patch_size)

    def _run_capturing_encoder(self, args):
        captured = {}
        handle = self.model.encoder.register_forward_hook(
            lambda mod, inp, out: captured.__setitem__("enc", out)
        )
        try:
            with torch.no_grad():
                self.model(*args)
        finally:
            handle.remove()
        return captured.get("enc")

    def embed_for_mode(self, batch: Dict[str, Any], mode: str):
        enc = self._run_capturing_encoder(self._pack(batch[mode]))
        return enc.mean(dim=1).float().detach()  # (B, d_model)

    def supports_layerwise(self) -> bool:
        return True

    def get_layer_names(self, granularity: str = "blocks") -> list:
        names = super().get_layer_names(granularity=granularity)
        names.append("encoder_out")
        return names

    def embed_all_layers_for_mode(self, batch, mode, granularity: str = "blocks"):
        args = self._pack(batch[mode])
        captured = {}

        def forward_fn():
            handle = self.model.encoder.register_forward_hook(
                lambda mod, inp, out: captured.__setitem__("encoder_out", out)
            )
            try:
                self.model(*args)
            finally:
                handle.remove()

        results = self._capture_module_outputs(
            forward_fn, model=self._get_hookable_model(),
            pool_fn=self._generic_pool, granularity=granularity,
        )
        if "encoder_out" in captured:
            results["encoder_out"] = self._generic_pool(captured["encoder_out"]).float().detach()
        return results


MOIRAI_MODEL_MAP = {
    "moirai": (
        ["small", "base", "large"],
        [f"Salesforce/moirai-1.0-R-{s}" for s in ["small", "base", "large"]],
    ),
}

register_adapter("moirai", MoiraiAdapter)
