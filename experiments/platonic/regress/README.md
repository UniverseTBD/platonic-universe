# Regression tier

Runs a 5-fold linear probe of every model's embeddings against the
LEPHARE physics labels (redshift, mass, sSFR, mag_g, mag_r, g-r colour)
on the `Ashodkh/cosmosweb-hsc-jwst-high-snr-pil2` dataset. Output is a
small parquet per `(modality, model_alias, model_size)` tuple.

Same coordination model as `compute/`: a private Hugging Face dataset
acts as a distributed lock. Workers atomically claim
`running/<tag>.running` markers; stale claims older than 1 hour are
auto-released; nothing data-shaped is committed to git.

## Required environment

```bash
export HF_TOKEN=<your-hf-token>                              # private datasets / write access
export PU_REGRESS_RESULTS_REPO=<owner>/pu-regress-results    # phase outputs land here
```

No defaults; both scripts fail loudly if either is unset.

## One-time setup

```bash
python setup_hf.py "$PU_REGRESS_RESULTS_REPO"
```

## Running

```bash
# SLURM
for i in {1..16}; do sbatch slurm/pu_regress.sh; done

# vast.ai (or any local multi-GPU box)
bash vast/vast_regress.sh
```

## Aggregating

After every tuple lands in `done/`:

```bash
python aggregate.py \
    --pull-from "$PU_REGRESS_RESULTS_REPO" \
    --out-dir   ./derived \
    --json-out  ./derived/r2_vs_params.json
```

The JSON layout matches what's consumed by Smith42's plotting branch:
`{ modality: { model_alias: { size: { property: { r2_mean } } } } }`.

## Knobs

| Variable | Default | What it does |
|---|---|---|
| `PU_REGRESS_DATASET` | `Ashodkh/cosmosweb-hsc-jwst-high-snr-pil2` | source dataset |
| `PU_REGRESS_N_USE` | `45000` | catalog/embedding sample size |
| `PU_REGRESS_BATCH_SIZE` | `16` | inference batch size |
| `PU_REGRESS_CV_FOLDS` | `5` | linear-probe CV folds |
| `PU_REGRESS_PCA_COMPONENTS` | `0` (off) | optional PCA before regression |
| `PU_REGRESS_STALE_S` | `3600` | running-marker auto-release threshold |
| `PU_REGRESS_TARGET` | unset | single-tuple mode: `<modality>/<alias>_<size>` |

## What's where

```
experiments/platonic/regress/
├── pu_regress.py     worker: claim → extract → probe → upload → release
├── setup_hf.py       one-time bootstrap of the coordination dataset
├── aggregate.py      pull all done/*.parquet -> one summary parquet (+ optional JSON)
├── slurm/            SBATCH wrapper
└── vast/             tmux launcher (one worker per GPU)
```

## Cost

Per `(modality, alias, size)` tuple, on one A100:

- small (ViT-base, DINO-small, ConvNeXt-tiny): a few minutes
- medium (ViT-large, ConvNeXt-large): tens of minutes
- large (ViT-huge, DINO-giant, vit-mae-huge): ~1 hour
- VLMs (paligemma, llava): a few hours; paligemma-28B alone is half a day

Total grid: ~25–35 GPU-hours. With 8 GPUs in parallel via `vast_regress.sh`:
~3–5 hours wall.
