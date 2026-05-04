# Regression tier

Linear-probe every foundation model's embeddings against the LEPHARE
physics labels (redshift, mass, sSFR, mag_g, mag_r, g–r colour) on the
`<anon>/cosmosweb-hsc-jwst-high-snr-pil2` dataset. The whole pipeline
is two scripts and reproduces the regression numbers in the paper end
to end.

## Files

```
experiments/platonic/regress/
├── 01_extract_and_probe.py  step 1 — distributed worker:
│                              claim → extract → probe → kNN → UMAP → upload → release
├── 02_aggregate_pairs.py    step 2 — local aggregator:
│                              pulls neighbour matrices, computes intramodal +
│                              crossmodal MKNN and Wasserstein
├── 03_plot_umap.py          step 3 — UMAP figure rendering (one PDF per
│                              modality; pages = colour-by property,
│                              grid = family × size)
├── aggregate.py             optional: pull all probe.parquet → one summary table
├── setup_hf.py              one-time bootstrap of the coordination dataset
├── slurm/                   SBATCH wrapper
└── vast/                    tmux launcher (one worker per GPU on a single box)
```

## Coordination model

A private Hugging Face dataset acts as the distributed lock. Workers
atomically claim `running/<tag>.running` markers; stale claims older
than 1 h auto-release; nothing data-shaped is committed to git.

## Required environment

```bash
export HF_TOKEN=<token-with-write-access>
export PU_REGRESS_RESULTS_REPO=<owner>/pu-regress-results   # results land here
```

Optional but recommended (lets later analyses skip GPU re-extraction):

```bash
export PU_EMB_REPO=<owner>/pu-regress-embeddings           # cached .npy land here
```

## One-time setup

```bash
python setup_hf.py "$PU_REGRESS_RESULTS_REPO"
```

## Step 1 — distributed extract + probe

Run on as many GPUs / clusters as you have. Workers don't talk to each
other; they coordinate through `$PU_REGRESS_RESULTS_REPO`.

```bash
# SLURM cluster
for i in {1..16}; do sbatch slurm/pu_regress.sh; done

# Single multi-GPU box (one worker per GPU, in tmux)
bash vast/vast_regress.sh
```

Per-tuple output (uploaded to `done/<tag>/` on the results repo):

| File | One row per | Contents |
|---|---|---|
| `probe.parquet` | physical property | mean ± std R² across 10 random 80/20 splits, sample sizes, knobs |
| `neighbours.parquet` | galaxy | (k,) int indices of k nearest neighbours in embedding space |
| `umap.parquet` | galaxy | 2-D UMAP coords (precomputed kNN, cosine) |

If `$PU_EMB_REPO` is set, the per-tuple `emb_<modality>_<alias>_<size>_<n_use>.npy`
is also uploaded there.

### Probe recipe

For each label `y` and embedding matrix `Z`:

1. **Filter**: drop rows where `y` is not finite. For redshift, also drop
   `y ≤ 0` and clip to `[0, 4]`.
2. **Outlier trim**: clip `y` to its own 1–99 % quantile range.
3. **Repeat 10×** with different random splits:
   1. Random 80/20 train/test split; default test size is 2000 galaxies.
   2. `StandardScaler` fit on the train fold, applied to test fold.
   3. Fit OLS via `torch.linalg.lstsq` on GPU (CPU sklearn fallback).
   4. Score R² on the test fold.
4. Report mean ± std across the 10 splits.

This is the recipe the paper plots. The default knobs reproduce the
published numbers; everything is overridable per `Knobs` below.

## Step 2 — pairwise aggregation (local)

After step 1 finishes, on a laptop:

```bash
python 02_aggregate_pairs.py \
    --pull-from "$PU_REGRESS_RESULTS_REPO" \
    --out-dir   ./derived
```

Output: `derived/regress_pairs.parquet` with one row per pair:

| Column | Meaning |
|---|---|
| `pair_kind` | `intramodal` (adjacent sizes within a family on one modality) or `crossmodal` (HSC vs JWST of the same model size) |
| `mknn` | mean per-row \|N_a ∩ N_b\| / k |
| `wass_<property>` | mean per-row Wasserstein-1 distance between the two sides' physics-label distributions on their kNN sets |

To also dump a flat R² table:

```bash
python aggregate.py \
    --pull-from "$PU_REGRESS_RESULTS_REPO" \
    --out-dir   ./derived \
    --json-out  ./derived/r2_vs_params.json
```

JSON layout: `{ modality: { model_alias: { size: { property: { r2_mean } } } } }`.

## Step 3 — UMAP figures

```bash
python 03_plot_umap.py \
    --pull-from "$PU_REGRESS_RESULTS_REPO" \
    --out-dir   ./derived
```

Default output is **per-tuple PNGs** under
`derived/umap/<modality>/<alias>_<size>__<property>.png` — one file per
(modality, alias, size, property), so a downstream analysis can grab
exactly the panel it wants.

Add `--pdf` to also render multi-page overview PDFs at
`derived/umap_<modality>.pdf`. Each PDF has six pages (one per
property); rows are model families, columns are sizes. The same
random subsample of galaxies is used for every panel so cross-panel
comparisons are visually fair.

## Knobs

| Variable | Default | Notes |
|---|---|---|
| `PU_REGRESS_DATASET` | `<anon>/cosmosweb-hsc-jwst-high-snr-pil2` | source dataset |
| `PU_REGRESS_N_USE` | `45000` | catalog / embedding sample size |
| `PU_REGRESS_BATCH_SIZE` | `16` | inference batch size |
| `PU_REGRESS_N_RUNS` | `10` | random train/test splits per probe |
| `PU_REGRESS_TEST_SIZE` | `2000` | held-out galaxies per split |
| `PU_REGRESS_KNN_K` | `10` | k for the kNN graph |
| `PU_REGRESS_STALE_S` | `3600` | running-marker auto-release threshold |
| `PU_REGRESS_TARGET` | unset | single-tuple mode: `<modality>/<alias>_<size>` |

## Cost

Per `(modality, alias, size)` tuple, on one A100:

- small (ViT-base, DINO-small, ConvNeXt-tiny): a few minutes
- medium (ViT-large, ConvNeXt-large): tens of minutes
- large (ViT-huge, DINO-giant, ViT-MAE-huge): ~1 hour
- VLMs (PaliGemma, LLaVA): a few hours; PaliGemma-28B alone is half a day

Full grid (62 tuples): roughly 25–35 GPU-hours. On 8 GPUs in parallel
via `vast/vast_regress.sh`: ~3–5 hours wall.
