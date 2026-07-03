# Layerwise physics probe — PaliGemma 2 — handoff (28B remaining)

Adds a new analysis the paper doesn't do: physics-probe R² (z / log M⋆ / sSFR)
at **every layer** of PaliGemma 2 (SigLIP vision tower → Gemma-2 decoder), for
HSC and JWST, using the paper's exact linear-probe recipe.

## Status

| size | ordered embeddings parquet | probe results | figure |
|------|----------------------------|---------------|--------|
| 3B   | `data/cosmosweb/paligemma_3b_3b_blocks_layerwise_ordered.parquet` (25 GB) | `data/paligemma_3b_layerwise_probe.parquet` | `figs/paligemma_layerwise_physics.pdf` ✅ committed |
| 10B  | `data/cosmosweb/paligemma_10b_10b_blocks_layerwise_ordered.parquet` (56 GB) | `data/paligemma_10b_layerwise_probe.parquet` | `figs/paligemma_layerwise_physics_10b.pdf` ✅ committed |
| 28B  | **TODO** | **TODO** | **TODO** |

`data/` is gitignored, so the parquets live on **beegfs only** (not in git); the
figures (`figs/`) are committed. 3B and 10B are validated: the published-embedding
(`hidden_states_last`) R² reproduces paper Table 4 within ~0.01.

## ⚠️ The one thing that will bite you: row alignment

The externally-delivered `data/cosmosweb/*_blocks_layerwise.parquet` files had
**globally scrambled row order** (extractor ran with `num_workers>0` on a
streaming dataset), so joining physics labels by index gave R² ≈ **−0.06**
(the `−d/n_train` floor) for *every* layer/property. Unrecoverable — no per-row
id. The fix is `00_extract_cosmosweb_layerwise.py`, which re-extracts with
**`num_workers=0`** so row `i` == cosmosweb catalog row `i`. **Do not** revert to
`num_workers>0`. Diagnosis trick if you suspect scramble: probe `mag_r` on the
last SigLIP layer — it should be R²≈0.9, not ≈0.

## How to run 28B

On a machine/cluster with **≥3 free A100s** (28B is ~56 GB bf16; `device_map=
"auto"` shards it across the allocated cards — this is fitting, not a speedup):

```bash
sbatch experiments/platonic/regress/pg28b.sbatch     # 3×A100, gpu partition
```

The script is headless and does all three steps → extract → probe → plot →
`figs/paligemma_layerwise_physics_28b.pdf`. Log: `logs/pg28b_<jobid>.out`.
Expect ~5–8 h once it starts. Watch: `squeue -j <jobid>`.

If not using SLURM, run the three commands from the sbatch body directly on a
node with 3 visible A100s (don't set `CUDA_VISIBLE_DEVICES` to one card, or the
weights won't fit).

## Verify 28B (do this before trusting the figure)

Sanity-probe just the published column and compare to **Table 4, PaliGemma 2 28B**:

|      | HSC z / M⋆ / sSFR | JWST z / M⋆ / sSFR |
|------|-------------------|--------------------|
| 28B  | 0.398 / 0.492 / 0.478 | 0.465 / 0.722 / 0.310 |

```bash
PU_LAYERWISE_ONLY=hidden_states_last \
PU_LAYERWISE_INPUT=data/cosmosweb/paligemma_28b_28b_blocks_layerwise_ordered.parquet \
PU_LAYERWISE_OUTPUT=/tmp/sanity28b.parquet \
.venv/bin/python experiments/platonic/regress/04_layerwise_probe.py
```
Match within ~0.01 → alignment good. ≈ −0.06 everywhere → scrambled (re-extract).

## Environment assumptions (adjust if on a different filesystem)

- Repo + venv on beegfs: `/beegfs/general/mjsmith/pu` (paths in the sbatch are absolute).
- HF cache **must not** be in `$HOME` (per-user quota is small): `HF_HOME=/beegfs/general/mjsmith/hf_cache`.
- PaliGemma repos are **gated** — needs a real HF token (`~/.cache/huggingface/token`, or `export HF_TOKEN=...`).
- `HF_HUB_DISABLE_XET=1` avoids the xet chunk-cache filling the home quota.

## Env-var reference

- Extractor (`00_extract_cosmosweb_layerwise.py`): `PU_EXTRACT_MODEL` (alias, e.g.
  `paligemma_28b`), `PU_EXTRACT_HFID`, `PU_EXTRACT_SIZE`, `PU_EXTRACT_N` (default
  45000), `PU_EXTRACT_BATCH`, `PU_EXTRACT_LIMIT` (smoke-test cap), `PU_EXTRACT_OUT`.
- Probe (`04_layerwise_probe.py`): `PU_LAYERWISE_INPUT`, `PU_LAYERWISE_OUTPUT`,
  `PU_LAYERWISE_N_USE` (45000), `PU_LAYERWISE_ONLY` (substring filter),
  `PU_LAYERWISE_LIMIT`.
- Plot (`scripts/plot_paligemma_layerwise.py`): `PU_PLOT_INPUT`, `PU_PLOT_MODEL`
  (title, e.g. `"PaliGemma 2 (28B)"`), `PU_PLOT_OUT`, `PU_PLOT_PAPER` (`3b`/`10b`/
  `28b` for the dotted Table-4 refs; `""` = none).
