# Aravind: foundation-model adapters for new modalities

This branch extends platonic-universe beyond vision models with foundation
models for three new modalities: **time series**, **spectra**, and **tabular**.
Everything follows the existing self-registering adapter pattern, so
`get_adapter(...)`, the layer-wise extraction path, and the metric suite work
unchanged.

Contributed by Aravind Kannappan. On the fork
(`aravinds-kannappan/platonic-universe`) the same work is split into one PR per
model family for easier review; this branch bundles all of it for upstream.

## New models

| Alias | Modality | Model | Extra | Live-verified here |
|-------|----------|-------|-------|:---:|
| `chronos` | time series | Amazon Chronos-T5 and Chronos-Bolt | `.[timeseries]` | yes |
| `moment` | time series | AutonLab MOMENT-1 | (isolated env, see note) | no |
| `patchtst` / `patchtsmixer` | time series | IBM Granite PatchTST / PatchTSMixer | none (transformers) | yes |
| `moirai` | time series | Salesforce Moirai | `.[moirai]` | yes |
| `spender` | spectra | Spender galaxy-spectrum autoencoder | `.[spectra]` | yes |
| `tabpfn` | tabular | Prior Labs TabPFN v2 | `.[tabular]` | no (gated weights) |

"Live-verified here" means I ran the real weights end to end on CPU while
building this. MOMENT is blocked by its `transformers==4.33.3` pin (incompatible
with this repo); TabPFN weights are gated on HuggingFace. Both have correct
adapters and stub tests; see their module docstrings for the run steps.

## Design

Each adapter is a `pu.models.base.ModelAdapter`:

- **Input contract.** Sequence models take `batch[mode]` as `(B, n_channels, seq_len)`
  (channels-first); tabular takes `(B, n_features)` rows plus optional labels.
- **Output contract.** `embed_for_mode -> (B, D)`; `embed_all_layers_for_mode ->`
  ordered `{layer: (B, D)}` for layer-wise MKNN/CKA (TabPFN is embedding-only).
- **Device + determinism.** New `base.resolve_device()` and
  `base.set_deterministic()`. The older image adapters hardcode `.to("cuda")`;
  these do not, so they run and are tested on CPU.

The time-series API contract and the reproducibility notes live in
[`experiments/timeseries/README.md`](experiments/timeseries/README.md).

## Layer-wise MKNN harness

`scripts/timeseries_layerwise_mknn.py` loads two models, iterates every
extraction layer, pushes one dummy batch through both, and compares with MKNN,
after asserting run-to-run determinism (`max|Δ|==0`, self-MKNN `==1.0`). This
was written against the seeded-but-nondeterministic MKNN we saw during revision;
the fix is single-threaded BLAS plus deterministic algorithms, and the reminder
that MKNN is row-order sensitive.

```bash
pip install ".[timeseries]"
python scripts/timeseries_layerwise_mknn.py        # Chronos-T5 vs Chronos-Bolt
python scripts/timeseries_layerwise_mknn.py \
    --model-a patchtst --repo-a ibm-granite/granite-timeseries-patchtst \
    --model-b patchtsmixer --repo-b ibm-granite/granite-timeseries-patchtsmixer
```

## Tests

```bash
pytest tests/test_timeseries_adapters.py \
       tests/test_patchts_adapters.py \
       tests/test_moirai_adapter.py \
       tests/test_spectra_spender.py \
       tests/test_tabular_tabpfn.py          # 39 stub tests, CPU, no network

PU_RUN_SLOW=1 pytest -m slow                 # opt-in, downloads real weights
```

All new modules pass `ruff check` (E, F, I). Stub tests use tiny `nn.Module`
stand-ins so they never download a model, matching `test_hf_clip_and_llava_ov.py`.

## Not yet wired in

These adapters produce embeddings from arbitrary series/spectra/tables; hooking
up the real astronomical datasets (light curves, catalogue rows) is the next
step and slots into `pu.pu_datasets` exactly like `desi_spectra.py`. The
`*_MODEL_MAP` dicts in each module are the canonical alias to checkpoint maps.
