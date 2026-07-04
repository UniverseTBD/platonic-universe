"""PatchTST / PatchTSMixer adapters (time-series modality, transformers-native).

Adds two more time-series foundation models that ship inside ``transformers``
itself, using IBM's Granite pretrained checkpoints:

- ``patchtst``      -> ``transformers.PatchTSTModel``   (``ibm-granite/granite-timeseries-patchtst``)
- ``patchtsmixer``  -> ``transformers.PatchTSMixerModel`` (``ibm-granite/granite-timeseries-patchtsmixer``)

Both are standard ``nn.Module`` encoders with a ``past_values`` forward and a
``last_hidden_state`` of shape ``(B, n_channels, n_patches, d_model)``, so they
plug straight into the generic layer-wise extraction. Unlike Chronos/MOMENT
these need no extra dependency (they are part of the pinned ``transformers``).

Contract note: these checkpoints are multivariate (the Granite models expect
``config.num_input_channels`` channels). The adapter fits an incoming
``(B, C, T)`` batch to ``(B, context_length, num_input_channels)`` by
truncating/padding time and tiling/truncating channels, so a univariate light
curve still runs. For a faithful comparison, feed data whose channel count
matches the checkpoint.
"""

from typing import Any, Dict

import torch

from pu.models.base import ModelAdapter, resolve_device
from pu.models.registry import register_adapter
from pu.models.timeseries import PreprocessTimeSeries, fit_length


class PatchTSAdapter(ModelAdapter):
    """Adapter for transformers-native PatchTST and PatchTSMixer encoders."""

    _MODEL_CLASSES = {
        "patchtst": "PatchTSTModel",
        "patchtsmixer": "PatchTSMixerModel",
    }

    def __init__(self, model_name: str, size: str, alias: str = None):
        super().__init__(model_name, size, alias)
        self.model = None
        self.device = None
        self.context_length = None
        self.num_channels = None

    def load(self, compile_model: bool = False) -> None:
        import transformers

        cls_name = self._MODEL_CLASSES.get(self.alias)
        if cls_name is None:
            raise ValueError(f"Unknown PatchTS alias '{self.alias}'.")
        self.device = resolve_device()
        cls = getattr(transformers, cls_name)
        self.model = cls.from_pretrained(self.model_name).to(self.device).eval()
        cfg = self.model.config
        self.context_length = int(cfg.context_length)
        self.num_channels = int(cfg.num_input_channels)
        if compile_model:
            self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)

    def get_preprocessor(self, modes, resize: bool = False, resize_mode: str = "fill"):
        return PreprocessTimeSeries(modes)

    def _prepare(self, series: torch.Tensor) -> torch.Tensor:
        """Fit ``(B, C, T)`` to ``(B, context_length, num_channels)`` (channels-last)."""
        x = series.to(self.device).float()
        if x.dim() != 3:
            raise ValueError(f"expected (B, C, T) series, got {tuple(x.shape)}")
        B, C, T = x.shape
        if C != self.num_channels:
            if C > self.num_channels:
                x = x[:, : self.num_channels]
            else:  # tile channels up to the model's expected count
                reps = (self.num_channels + C - 1) // C
                x = x.repeat(1, reps, 1)[:, : self.num_channels]
        x, _ = fit_length(x, self.context_length)  # (B, num_channels, context_length)
        return x.transpose(1, 2).contiguous()      # (B, context_length, num_channels)

    @staticmethod
    def _pool(t: torch.Tensor) -> torch.Tensor:
        """Pool to (B, d): mean over channel + patch dims for the 4D hidden state."""
        if t.dim() == 4:      # (B, C, P, d)
            return t.mean(dim=(1, 2))
        if t.dim() == 3:      # (B, P, d)
            return t.mean(dim=1)
        if t.dim() == 2:
            return t
        return t.reshape(t.shape[0], -1)

    def embed_for_mode(self, batch: Dict[str, Any], mode: str):
        x = self._prepare(batch[mode])
        with torch.no_grad():
            out = self.model(past_values=x)
        return self._pool(out.last_hidden_state).float().detach()

    def supports_layerwise(self) -> bool:
        return True

    def get_layer_names(self, granularity: str = "blocks") -> list:
        names = super().get_layer_names(granularity=granularity)
        names.append("last_hidden_state")
        return names

    def embed_all_layers_for_mode(self, batch, mode, granularity: str = "blocks"):
        x = self._prepare(batch[mode])
        captured = {}

        def forward_fn():
            out = self.model(past_values=x)
            if getattr(out, "last_hidden_state", None) is not None:
                captured["last_hidden_state"] = out.last_hidden_state

        results = self._capture_module_outputs(
            forward_fn, model=self.model, pool_fn=self._pool, granularity=granularity
        )
        if "last_hidden_state" in captured:
            results["last_hidden_state"] = self._pool(captured["last_hidden_state"]).float().detach()
        return results


PATCHTS_MODEL_MAP = {
    "patchtst": (["granite"], ["ibm-granite/granite-timeseries-patchtst"]),
    "patchtsmixer": (["granite"], ["ibm-granite/granite-timeseries-patchtsmixer"]),
}

register_adapter("patchtst", PatchTSAdapter)
register_adapter("patchtsmixer", PatchTSAdapter)
