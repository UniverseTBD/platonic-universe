"""``light-curve`` feature-extractor adapter (time-series / light-curve modality).

Brings the `light-curve <https://pypi.org/project/light-curve/>`_ package
(Malanchev et al. 2021) into platonic-universe as a **feature-extractor**
"foundation model" for astronomical light curves. Unlike the neural adapters,
``light-curve`` produces a *deterministic, hand-engineered* feature vector per
series: half-amplitude, Stetson K, von Neumann eta, linear-trend slope, Otsu
split statistics, and dozens more. Those vectors are exactly the classical
baseline the Platonic Representation Hypothesis wants to compare learned
time-series embeddings against, so this adapter slots into the same
``get_adapter(...)`` / MKNN / CKA path as every other model.

Why this belongs here
----------------------
``timeseries.py`` deliberately defined the light-curve *input contract* and left
"wiring in real light-curve data" as the downstream step (see its module
docstring and ``ARAVIND.md``). ``light-curve`` is the natural first consumer of
that contract: it takes the same ``(B, C, T)`` batches and returns a ``(B, D)``
embedding, but needs **no weights, no network, and no GPU**, so unlike TabPFN /
MOMENT it is verified end-to-end on CPU in CI.

Input contract
--------------
``batch[mode]`` is a float tensor / array shaped ``(B, C, T)`` (channels-first),
matching the time-series modality. Channels carry astronomical meaning:

- ``C == 1``: magnitudes (or flux). Times default to ``arange(T)`` and
  per-point errors to ones.
- ``C == 2``: ``(time, magnitude)``. Errors default to ones.
- ``C == 3``: ``(time, magnitude, sigma)`` — the full photometric triple.

``(B, T)`` is accepted as a shorthand for a single magnitude channel. Each
series is sorted by time before extraction (``light-curve`` requires monotonic
time), so unordered input is handled safely.

Output contract
---------------
``embed_for_mode(batch, mode) -> (B, D)``: one feature vector per light curve,
where ``D == len(self.feature_names)``. Layer-wise extraction is intentionally
unsupported (a feature vector has no residual stream), so ``supports_layerwise``
returns ``False`` — the same choice ``tabular.py`` makes for TabPFN.

Determinism
-----------
Feature extraction is a pure function of the input arrays: no seeds, no BLAS
non-determinism, ``embed_for_mode`` is bit-reproducible run to run. The only
ordering invariant callers must preserve is the usual MKNN one (row ``i`` of the
two matrices must be the same object).

Install with ``pip install ".[lightcurve]"`` (or ``pip install light-curve``).
"""

import importlib.util
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from pu.models.base import ModelAdapter
from pu.models.registry import register_adapter

LIGHTCURVE_AVAILABLE = importlib.util.find_spec("light_curve") is not None


# --------------------------------------------------------------------------- #
# Feature-set presets                                                         #
#                                                                             #
# Each entry is ``(ClassName, *constructor_args)``. All classes below are     #
# Rust-backed so a single ``Extractor`` exposes ``.names`` and runs in one    #
# pass. Presets are ordered cheap -> rich; ``default`` is a robust, always-   #
# defined 23-feature set that needs no period search.                         #
# --------------------------------------------------------------------------- #

_CHEAP_SPECS: List[Tuple] = [
    ("Amplitude",),
    ("BeyondNStd", 1.0),
    ("Cusum",),
    ("Kurtosis",),
    ("Mean",),
    ("Median",),
    ("Skew",),
    ("StandardDeviation",),
    ("InterPercentileRange", 0.25),
    ("MedianAbsoluteDeviation",),
]

_DEFAULT_EXTRA_SPECS: List[Tuple] = [
    ("BeyondNStd", 2.0),
    ("Eta",),
    ("LinearTrend",),
    ("MagnitudePercentageRatio", 0.4, 0.05),
    ("MaximumSlope",),
    ("MeanVariance",),
    ("MedianBufferRangePercentage", 0.1),
    ("PercentAmplitude",),
    ("PercentDifferenceMagnitudePercentile", 0.05),
    ("StetsonK",),
    ("WeightedMean",),
]

_FULL_EXTRA_SPECS: List[Tuple] = [
    ("AndersonDarlingNormal",),
    ("Duration",),
    ("ObservationCount",),
    ("MaximumTimeInterval",),
    ("MinimumTimeInterval",),
    ("TimeMean",),
    ("TimeStandardDeviation",),
    ("ExcessVariance",),
    ("LinearFit",),
    ("ReducedChi2",),
    ("Chi2Pvar",),
    ("EtaE",),
    ("Roms",),
    ("OtsuSplit",),
]

FEATURE_SETS: Dict[str, List[Tuple]] = {
    "cheap": _CHEAP_SPECS,
    "default": _CHEAP_SPECS + _DEFAULT_EXTRA_SPECS,
    "full": _CHEAP_SPECS + _DEFAULT_EXTRA_SPECS + _FULL_EXTRA_SPECS,
}


class PreprocessLightCurve:
    """Turn a HuggingFace example into the ``(C, T)`` light-curve contract array.

    Reads the raw series from ``series_key`` (default ``"lightcurve"``, falling
    back to the mode name) and returns ``{mode: float32 (C, T)}``. A 1-D series
    is promoted to a single magnitude channel. Downstream light-curve datasets
    (e.g. a ZTF / PLAsTiCC adapter) subclass this to assemble a ``(time, mag,
    sigma)`` stack from their own columns.
    """

    def __init__(self, modes: Sequence[str], series_key: str = "lightcurve"):
        self.modes = list(modes)
        self.series_key = series_key

    def __call__(self, idx: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for mode in self.modes:
            raw = idx.get(self.series_key, idx.get(mode))
            if raw is None:
                continue
            series = np.asarray(raw, dtype=np.float32)
            if series.ndim == 1:
                series = series[np.newaxis, :]  # (T,) -> (1, T)
            result[mode] = series
        return result


class LightCurveAdapter(ModelAdapter):
    """Adapter exposing ``light-curve`` feature vectors as ``(B, D)`` embeddings.

    ``model_name`` is informational; ``size`` selects a feature-set preset
    (``"cheap"`` | ``"default"`` | ``"full"``). Pass an explicit list of
    ``light_curve`` feature *instances* via ``features=`` to override the preset.
    """

    def __init__(self, model_name: str, size: str = "default", alias: str = None,
                 features: Sequence[Any] = None):
        super().__init__(model_name, size, alias)
        self.extractor = None
        self.feature_names: List[str] = []
        self._features = features            # optional explicit feature instances
        self.device = "cpu"                  # feature extraction is CPU-only

    # -- construction ------------------------------------------------------- #

    @staticmethod
    def _build_from_specs(specs: Sequence[Tuple]):
        """Instantiate a ``light_curve.Extractor`` from ``(ClassName, *args)`` specs."""
        import light_curve as lc

        feats = [getattr(lc, spec[0])(*spec[1:]) for spec in specs]
        return lc.Extractor(*feats)

    def load(self, compile_model: bool = False) -> None:
        try:
            import light_curve as lc  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "light-curve is not installed. Install the extra with:\n"
                '  pip install ".[lightcurve]"   (or: pip install light-curve)'
            ) from exc

        if self._features is not None:
            self.extractor = lc.Extractor(*self._features)
        else:
            preset = self.size if self.size in FEATURE_SETS else "default"
            self.extractor = self._build_from_specs(FEATURE_SETS[preset])
        self.feature_names = list(self.extractor.names)

    def get_preprocessor(self, modes, resize: bool = False, resize_mode: str = "fill"):
        return PreprocessLightCurve(modes)

    # -- input handling ----------------------------------------------------- #

    @staticmethod
    def _as_numpy(x) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float64)  # light-curve wants float64

    @staticmethod
    def _split_channels(series: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map a ``(C, T)`` (or ``(T,)``) series to sorted ``(t, m, sigma)`` arrays."""
        if series.ndim == 1:
            series = series[np.newaxis, :]
        C, T = series.shape
        if C == 1:
            t = np.arange(T, dtype=np.float64)
            m = series[0]
            sigma = np.ones(T, dtype=np.float64)
        elif C == 2:
            t, m = series[0], series[1]
            sigma = np.ones(T, dtype=np.float64)
        else:  # C >= 3: (time, magnitude, sigma, [extra bands ignored])
            t, m, sigma = series[0], series[1], series[2]
        order = np.argsort(t, kind="stable")
        return t[order], m[order], sigma[order]

    def _batch_to_triples(self, arr: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Convert a ``(B, C, T)`` (or ``(B, T)``) batch into a list of ``(t, m, s)``."""
        if arr.ndim == 2:                    # (B, T) -> one magnitude channel each
            arr = arr[:, np.newaxis, :]
        if arr.ndim != 3:
            raise ValueError(
                f"expected (B, C, T) or (B, T) light curves, got shape {arr.shape}"
            )
        return [self._split_channels(arr[i]) for i in range(arr.shape[0])]

    # -- embedding ---------------------------------------------------------- #

    def embed_for_mode(self, batch: Dict[str, Any], mode: str):
        if self.extractor is None:
            self.load()
        triples = self._batch_to_triples(self._as_numpy(batch[mode]))
        # ``many`` runs every feature in one pass over each series; fill_value
        # replaces any undefined feature (e.g. a degenerate series) with 0.0 so
        # the embedding matrix is always finite and MKNN never sees a NaN row.
        feats = self.extractor.many(
            triples, sorted=True, fill_value=0.0, n_jobs=1
        )
        return torch.as_tensor(np.asarray(feats), dtype=torch.float32)  # (B, D)

    # A hand-engineered feature vector has no per-layer residual stream.
    def supports_layerwise(self) -> bool:
        return False


# Canonical alias -> (preset sizes, informational id) map for downstream discovery.
LIGHTCURVE_MODEL_MAP = {
    "lightcurve": (
        ["cheap", "default", "full"],
        ["light-curve (Malanchev et al. 2021) feature extractor"],
    ),
}

register_adapter("lightcurve", LightCurveAdapter)
