# Light-curve feature extractor

Adds the [`light-curve`](https://pypi.org/project/light-curve/) package
(Malanchev et al. 2021, *MNRAS* 502, 5147) as a **feature-extractor "foundation
model"** for the light-curve modality. Where `timeseries.py` defined the
light-curve *input contract* and left the real-data wiring as the next step,
this adapter is the first concrete consumer of that contract: same `(B, C, T)`
batch in, a `(B, D)` embedding out.

## What ships

| Alias | Class | Presets | Weights / network | Runs on CPU here |
|-------|-------|---------|-------------------|:---:|
| `lightcurve` | `LightCurveAdapter` | `cheap` (10), `default` (23), `full` (42) | none | yes |

It lives in [`src/pu/models/lightcurve.py`](../../src/pu/models/lightcurve.py)
and registers itself like every other adapter, so `get_adapter("lightcurve")`
and the metric suite work unchanged.

## Why a hand-engineered extractor belongs next to the neural adapters

The Platonic Representation Hypothesis asks whether models converge toward a
*shared* representation. To answer that for light curves you need a fixed,
model-independent reference: `light-curve` is exactly that. Its features
(amplitude, Stetson K, von Neumann eta, linear-trend slope, Otsu split, ...) are
the classical descriptors the variable-star / transient community has used for
years. Comparing a learned time-series embedding (Chronos, MOMENT, Moirai)
against this vector is a direct "did the network rediscover the classical
statistics?" probe — the light-curve analogue of the vision-model comparisons
the repo already runs.

Practical bonus: `light-curve` is Rust-backed, needs no weights, no GPU and no
network, so unlike TabPFN (gated) or MOMENT (transformers-pinned) it is
verified **end to end on CPU in CI**.

## The contract

**Input.** `batch[mode]` is a float tensor `(B, C, T)` (channels-first),
matching the time-series modality. Channels carry photometric meaning:

| `C` | channels | missing channels filled with |
|-----|----------|------------------------------|
| 1 | magnitude | time = `arange(T)`, `sigma = 1` |
| 2 | time, magnitude | `sigma = 1` |
| 3 | time, magnitude, sigma | — |

`(B, T)` is shorthand for a single magnitude channel. Each series is sorted by
time inside the adapter (`light-curve` requires monotonic time), so unordered
input is safe.

**Output.** `embed_for_mode(batch, mode) -> (B, D)`, one feature vector per
light curve, `D = len(adapter.feature_names)`. There is no residual stream to
pool, so `supports_layerwise()` is `False` and `embed_all_layers_for_mode` is
not implemented — the same choice `tabular.py` makes for TabPFN.

**Determinism.** Extraction is a pure function of the input arrays: no seeds, no
BLAS non-determinism. `embed_for_mode` is bit-reproducible run to run; any
feature that is undefined for a degenerate series is filled with `0.0` so the
embedding matrix is always finite (MKNN never sees a NaN row).

## Feature-set presets

`size` selects a preset; the sets are nested (`cheap ⊂ default ⊂ full`).

- **`cheap`** (10): magnitude-only moment/shape statistics. No errors needed.
- **`default`** (23): adds shape ratios, linear trend, and error-aware
  Stetson K / weighted mean. The recommended general-purpose set.
- **`full`** (42): adds time-domain statistics (duration, cadence gaps),
  Anderson–Darling normality, reduced χ², Otsu split, and η_E.

Pass `features=[...]` (a list of `light_curve` feature instances) to override the
preset entirely.

## Usage

```python
import numpy as np
from pu.models import get_adapter

ad = get_adapter("lightcurve")("light-curve", "default", alias="lightcurve")
ad.load()

# One band: (B, 3, T) = (time, magnitude, sigma)
B, T = 32, 200
t = np.sort(np.random.uniform(0, 100, (B, T)), axis=1)
m = 15 + 0.3 * np.sin(t / 5.0) + np.random.normal(0, 0.05, (B, T))
s = np.full((B, T), 0.05)
batch = {"lc": np.stack([t, m, s], axis=1)}

emb = ad.embed_for_mode(batch, "lc")   # (32, 23) torch.float32
print(ad.feature_names)                # ['amplitude', 'beyond_1_std', ...]
```

Compare against a learned embedding with the usual metric path:

```python
from pu.metrics.io import compare
# emb_lightcurve, emb_chronos are (B, D) matrices for the *same* B light curves,
# in the *same* row order.
compare({"lightcurve": emb_lightcurve, "chronos": emb_chronos}, metrics=["mknn", "cka"])
```

## Wiring in real light curves

When a light-curve dataset lands (ZTF, PLAsTiCC, ELAsTiCC, ...), register a
dataset adapter under `pu.pu_datasets` exactly as `desi_spectra.py` does for
spectra, and subclass `PreprocessLightCurve` to stack `(time, mag, sigma)` from
that survey's columns into the `(C, T)` array. Nothing else changes: the
`lightcurve` model adapter, the metrics, and the comparison flow are already in
place.

## Tests

```bash
pip install ".[lightcurve]"
pytest tests/test_lightcurve_adapter.py   # live path runs by default (CPU, no download)
```

The channel-splitting / batching logic is covered by numpy-only tests that run
without the package; the shape, determinism, and preset tests exercise the real
Rust extractor.

## Reference

Malanchev, K. L., et al. 2021, "Anomaly detection in the Zwicky Transient
Facility DR3", *MNRAS*, 502, 5147. Package:
<https://github.com/light-curve/light-curve-python>.
