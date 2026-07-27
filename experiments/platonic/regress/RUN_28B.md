# Layerwise physics probe — PaliGemma 2 — all sizes complete

Adds a new analysis the paper doesn't do: physics-probe R² (z / log M⋆ / sSFR)
at **every layer** of PaliGemma 2 (SigLIP vision tower → Gemma-2 decoder), for
HSC and JWST, using the paper's exact linear-probe recipe.

## Status

| size | ordered embeddings parquet | probe results | figure |
|------|----------------------------|---------------|--------|
| 3B   | `data/cosmosweb/paligemma_3b_3b_blocks_layerwise_ordered.parquet` (25 GB) | `data/paligemma_3b_layerwise_probe.parquet` | `figs/paligemma_layerwise_physics.pdf` ✅ committed |
| 10B  | `data/cosmosweb/paligemma_10b_10b_blocks_layerwise_ordered.parquet` (56 GB) | `data/paligemma_10b_layerwise_probe.parquet` | `figs/paligemma_layerwise_physics_10b.pdf` ✅ committed |
| 28B  | `…/paligemma_28b_28b_blocks_layerwise_ordered.parquet` (77 GB) | `data/paligemma_28b_layerwise_probe.parquet` | `figs/paligemma_layerwise_physics_28b.pdf` ✅ committed |

`data/` is gitignored, so the parquets live on scratch only (not in git); the
figures (`figs/`) are committed. All three sizes are validated: the
published-embedding (`hidden_states_last`) R² reproduces paper Table 4 within
~0.01. Headline layerwise result: physics info peaks in **intermediate** layers
(mid-SigLIP, or the vision→language boundary `llm.0`), well above the published
final-layer embedding — e.g. 28B JWST log M⋆ hits R²=0.81 at `llm.0` vs 0.72 at
the last layer.

## ⚠️ The one thing that will bite you: row alignment

The externally-delivered `data/cosmosweb/*_blocks_layerwise.parquet` files had
**globally scrambled row order** (extractor ran with `num_workers>0` on a
streaming dataset), so joining physics labels by index gave R² ≈ **−0.06**
(the `−d/n_train` floor) for *every* layer/property. Unrecoverable — no per-row
id. The fix is `00_extract_cosmosweb_layerwise.py`, which re-extracts with
**`num_workers=0`** so row `i` == cosmosweb catalog row `i`. **Do not** revert to
`num_workers>0`. Diagnosis trick if you suspect scramble: probe `mag_r` on the
last SigLIP layer — it should be R²≈0.9, not ≈0.

## How 28B was actually run (NCSA DeltaAI, GH200)

The environment moved to **NCSA DeltaAI** (aarch64 GH200, 96 GB HBM/card). A 28B
copy (~56 GB bf16) fits on **one** GH200, so the old 3×A100 model-sharding is
gone: instead run **4 independent copies, one per GPU**, each extracting a
contiguous quarter of the 45 000 rows (data parallelism), then concat in order.
Full 28B extraction wall time this way: **~30 min** on 4 GH200s.

Environment (native PyTorch per DeltaAI docs, extra deps layered on top):

```bash
module use /sw/user/modules/python
module load python/miniforge3_pytorch/2.10.0            # py3.12, torch 2.10+cu129
python3 -m venv --system-site-packages .venv_native     # base torch/torchvision/datasets visible
uv pip install --python .venv_native/bin/python \
    "transformers @ git+https://github.com/huggingface/transformers.git" \
    polars astropt accelerate opencv-python scikit-image llvmlite calibrated-similarity
# then: uv pip uninstall torch torchvision  (a dep pulls a CPU torch that shadows the module's)
export HF_HOME=/work/nvme/bfvh/msmith10/pu/hf_cache HF_TOKEN=$(cat ~/.cache/huggingface/token)
export HF_HUB_DISABLE_XET=1 PYTHONPATH=src
PY=.venv_native/bin/python
```

Notes:
- The project's `pu_cka` C++ ext won't build under the Cray `CC` (missing
  `Python.h`); it isn't needed here (the probe uses `torch.linalg.lstsq`), so
  install deps only and import `pu` via `PYTHONPATH=src`.
- `00_extract_cosmosweb_layerwise.py` disables the cuDNN SDPA backend on GH200
  (`torch.backends.cuda.enable_cudnn_sdp(False)`) — it otherwise dies with
  "cuDNN Frontend error: No valid execution plans built" in SigLIP/Gemma attention.

4-GPU extract → concat → probe → plot (all headless):

```bash
SHARD_DIR=/work/nvme/bfvh/msmith10/pu/data/cosmosweb
PREFIX=paligemma_28b_28b_blocks_layerwise; N=45000; Q=$((N/4))
for i in 0 1 2 3; do
  START=$((i*Q)); COUNT=$Q; [ $i -eq 3 ] && COUNT=$((N-START))
  CUDA_VISIBLE_DEVICES=$i PU_EXTRACT_MODEL=paligemma_28b \
    PU_EXTRACT_HFID=google/paligemma2-28b-mix-224 PU_EXTRACT_SIZE=28b \
    PU_EXTRACT_BATCH=16 PU_EXTRACT_START=$START PU_EXTRACT_COUNT=$COUNT \
    PU_EXTRACT_OUT=$SHARD_DIR/${PREFIX}_shard${i}.parquet \
    "$PY" experiments/platonic/regress/00_extract_cosmosweb_layerwise.py \
    > logs/pg28b_shard${i}.out 2>&1 &
done; wait
# concat shard0..3 (in order) -> ${PREFIX}_ordered.parquet  (node has ~430 GB RAM)
ORDERED=$SHARD_DIR/${PREFIX}_ordered.parquet
PU_LAYERWISE_INPUT=$ORDERED PU_LAYERWISE_OUTPUT=data/paligemma_28b_layerwise_probe.parquet \
  "$PY" experiments/platonic/regress/04_layerwise_probe.py
PU_PLOT_INPUT=data/paligemma_28b_layerwise_probe.parquet PU_PLOT_MODEL="PaliGemma 2 (28B)" \
  PU_PLOT_PAPER=28b PU_PLOT_OUT=figs/paligemma_layerwise_physics_28b.pdf \
  "$PY" scripts/plot_paligemma_layerwise.py
```

## Verify 28B — DONE, reproduces Table 4

Sanity-probe of the published column vs **Table 4, PaliGemma 2 28B** (achieved):

|          | HSC z / M⋆ / sSFR       | JWST z / M⋆ / sSFR      |
|----------|-------------------------|-------------------------|
| Table 4  | 0.398 / 0.492 / 0.478   | 0.465 / 0.722 / 0.310   |
| achieved | 0.400 / 0.480 / 0.475   | 0.460 / 0.722 / 0.302   |

All within ~0.01 → row alignment good. (If a future re-extract ever reads
≈ −0.06 everywhere, rows are scrambled — confirm `num_workers=0` and that
`PU_EXTRACT_START` shards were concatenated in index order.)

```bash
PU_LAYERWISE_ONLY=hidden_states_last \
PU_LAYERWISE_INPUT=$SHARD_DIR/paligemma_28b_28b_blocks_layerwise_ordered.parquet \
PU_LAYERWISE_OUTPUT=/tmp/sanity28b.parquet \
"$PY" experiments/platonic/regress/04_layerwise_probe.py
```

## Environment assumptions (DeltaAI; adjust if on a different filesystem)

- Repo on `/projects/bfvh/msmith10/platonic-universe`; native-torch venv `.venv_native`
  built on the `python/miniforge3_pytorch/2.10.0` module (see recipe above).
- Big parquets (shards + ordered, ~77 GB) live on **`/work/nvme/bfvh/msmith10/pu`**,
  not the shared 500 G `/projects` quota. `data/` (probe parquet) is gitignored.
- HF cache **must not** be in `$HOME` (small quota): `HF_HOME=/work/nvme/bfvh/msmith10/pu/hf_cache`.
- PaliGemma repos are **gated** — needs a real HF token (`~/.cache/huggingface/token`, or `export HF_TOKEN=...`).
- `HF_HUB_DISABLE_XET=1` avoids the xet chunk-cache filling the home quota.

## Env-var reference

- Extractor (`00_extract_cosmosweb_layerwise.py`): `PU_EXTRACT_MODEL` (alias, e.g.
  `paligemma_28b`), `PU_EXTRACT_HFID`, `PU_EXTRACT_SIZE`, `PU_EXTRACT_N` (default
  45000), `PU_EXTRACT_BATCH`, `PU_EXTRACT_START`/`PU_EXTRACT_COUNT` (contiguous
  shard for data-parallel extraction), `PU_EXTRACT_LIMIT` (smoke-test cap),
  `PU_EXTRACT_OUT`.
- Probe (`04_layerwise_probe.py`): `PU_LAYERWISE_INPUT`, `PU_LAYERWISE_OUTPUT`,
  `PU_LAYERWISE_N_USE` (45000), `PU_LAYERWISE_ONLY` (substring filter),
  `PU_LAYERWISE_LIMIT`.
- Plot (`scripts/plot_paligemma_layerwise.py`): `PU_PLOT_INPUT`, `PU_PLOT_MODEL`
  (title, e.g. `"PaliGemma 2 (28B)"`), `PU_PLOT_OUT`, `PU_PLOT_PAPER` (`3b`/`10b`/
  `28b` for the dotted Table-4 refs; `""` = none).

## ⚠️ The `kshitijd/platonic-embeddings` cosmosweb basket is ALSO scrambled

Spot-checked the smallest model (`dinov3_vits16`, 1.09 GB) from
`kshitijd/platonic-embeddings/cosmosweb/*_blocks_layerwise.parquet` with
`04_layerwise_probe.py` (`PU_LAYERWISE_PROPS=mag_r,redshift,mass,sSFR`).
**Every layer × property × modality reads the `-d/n` floor** (best R² anywhere
= **−0.001**; `mag_r` should be ≈0.9 on a vision backbone). Same globally
row-shuffled bug PaliGemma had before the `num_workers=0` re-extract — so the
whole HF basket is suspect, not just PaliGemma. To trust any of these models,
re-extract in row order (`00_extract_cosmosweb_layerwise.py`) or have them
re-uploaded. Result parquet: `data/dinov3_vits16_cosmosweb_layerwise_probe.parquet`.
(`04_layerwise_probe.py::parse_col` now also parses single-stack `model.layer.N`
/ `last_hidden_state` columns, so it works on any of these non-PaliGemma models.)
