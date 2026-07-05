# Time-series foundation models

Adds the **time-series modality** to platonic-universe so the Platonic
Representation Hypothesis can be probed on 1D signals (for example astronomical
light curves). This is an **API-contract** drop: the adapters and the
layer-wise MKNN harness are here now; wiring in real light-curve data is the
downstream step.

## What ships

| Alias | Class | Checkpoints | Runs on CPU here |
|-------|-------|-------------|------------------|
| `chronos` | `ChronosAdapter` | `amazon/chronos-t5-{tiny…large}`, `amazon/chronos-bolt-{tiny…base}` | yes |
| `moment`  | `MomentAdapter`  | `AutonLab/MOMENT-1-{small,base,large}` | needs isolated env (see below) |

Both live in [`src/pu/models/timeseries.py`](../../src/pu/models/timeseries.py)
and register themselves like every other adapter, so `get_adapter("chronos")`
and the layer-wise extraction path work unchanged.

## The contract

An adapter is a `pu.models.base.ModelAdapter`. For the time-series modality the
data shapes are fixed as follows.

**Input.** `batch[mode]` is a float tensor `(B, n_channels, seq_len)`
(channels-first). Univariate series use `n_channels == 1`. The adapter
right-pads / truncates to the model's native context window and builds an
`input_mask` where the model accepts one (`fit_length`).

**Output.**
- `embed_for_mode(batch, mode) -> (B, D)`: one embedding per series.
- `embed_all_layers_for_mode(batch, mode, granularity) -> OrderedDict[name, (B, D)]`:
  one pooled representation per hooked layer, ordered by module depth. This is
  what the cross-model layer-wise MKNN consumes.

**Preprocessing.** `PreprocessTimeSeries` maps one HuggingFace example
`{series_key: [...]}` to `{mode: (C, T)}`. Downstream datasets subclass it to
pull their own flux/time columns. When real data lands, register a dataset
adapter under `pu.pu_datasets` exactly as `desi_spectra.py` does for spectra.

## Determinism (read this)

During PRH revision we saw layer-wise MKNN drift run-to-run **even with seeds
set**. The cause was not the seed:

1. `torch` models must be in `.eval()` and run under `no_grad()` (dropout off).
2. Non-associative float reductions in multi-threaded BLAS perturb activations
   at ~1e-6, which is enough to flip *near-tied* cosine neighbours in MKNN.
3. MKNN is **row-order sensitive**: row `i` of the two embedding matrices must
   be the *same* series. Any reshuffle (e.g. `num_workers>0` on a paired
   loader) silently tanks the score.

`pu.models.base.set_deterministic()` handles (1) and (2): it seeds everything,
requests deterministic algorithms, and pins CPU math to one thread. (3) is a
data-ordering invariant the caller must preserve. The harness below asserts
`max|Δ|==0` and self-MKNN `==1.0` before reporting any cross-model number.

## Run the harness

```bash
pip install ".[timeseries]"

# Cross-family (T5 vs Bolt) layer-wise MKNN on dummy data, CPU-only:
python scripts/timeseries_layerwise_mknn.py

# Any two checkpoints / sizes:
python scripts/timeseries_layerwise_mknn.py \
    --repo-a amazon/chronos-t5-small \
    --repo-b amazon/chronos-bolt-small \
    --n 128 --seq-len 256 --k 10
```

Example output (chronos-t5-tiny vs chronos-bolt-tiny, `n=96`, `k=10`):

```
[A] chronos amazon/chronos-t5-tiny    layers=16  max|Δ|=0.00e+00  self-MKNN=1.0000 -> OK
[B] chronos amazon/chronos-bolt-tiny  layers=16  max|Δ|=0.00e+00  self-MKNN=1.0000 -> OK

final-vs-final MKNN : 0.6125   (encoder_embeddings vs encoder_embeddings)
best-aligned pair   : 0.6198   (final_layer_norm vs block.1)
```

Alignment rises with depth (~0.25 at the token embeddings to ~0.62 at the final
encoder layers), the same pattern the paper reports for the vision models.

## MOMENT environment note

`momentfm` pins `transformers==4.33.3`, which is incompatible with this repo's
modern `transformers` (and drags in a `tokenizers==0.13.3` source build). Run
MOMENT in a dedicated environment, extract embeddings to `.npy`/parquet, then
compare offline with `pu.metrics.mknn`: the same save-then-compare flow the
repo already uses for pre-computed embeddings. The adapter code itself is
transformers-version agnostic; only the `momentfm` package carries the pin.

## Tests

```bash
pytest tests/test_timeseries_adapters.py           # stub tests, no network
PU_RUN_SLOW=1 pytest tests/test_timeseries_adapters.py -k live   # real weights
```
