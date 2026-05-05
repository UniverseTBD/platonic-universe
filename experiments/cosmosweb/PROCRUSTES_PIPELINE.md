# Procrustes pipeline (HF embeddings → paper figures)

This pipeline reproduces the Procrustes-distance and cosine-similarity
artifacts that drive several paper figures
(`crossarchitectural_procrustes.pdf`,
`crossmodal_procrustes_per_property_ancova.pdf`,
`avg_cos_sim_matrix.pdf`).

The artifacts are not currently checked into the repo. This pipeline
regenerates them deterministically from the embeddings hosted on Hugging
Face by anyone with read access — no rerun of the heavy embedding
extraction is required.

## What the pipeline does

```
HCVYM5w6Gn/pu-embeddings/cosmosweb/*.parquet   ← upstream, on HF
       │
       │  (1) stream_embeddings_to_npy.py
       ▼
analysis/procrustes_pipeline/embeddings/*.npy   ← local
       │
       │  (2) probe_weight_analysis.py (the existing, env-driven)
       ▼
analysis/procrustes_pipeline/
  ├── procrustes_distances_45000galaxies.pkl    ← input to the plotters
  ├── avg_cosine_similarity_45000galaxies.pdf   ← paper Fig 4
  ├── probe_weight_analysis_45000galaxies_*.pdf
  └── probe_weight_cache_45000/                 ← per-model probe weights
       │
       │  (3) drop the pkl into a worktree on the
       │      `crossmodalintramodal` branch
       ▼
<worktree>/data/procrustes_distances_45000galaxies_upsampled.pkl
       │
       │  scripts/plot_crossarchitectural_procrustes.py
       │  scripts/plot_crossmodal_procrustes.py
       │  scripts/plot_crossmodal_procrustes_per_property.py
       ▼
<worktree>/figs/{crossarchitectural,crossmodal}_procrustes*.pdf  ← paper figs
```

## Quick start (full basket, ~30 GB transient bandwidth, ~30 GB peak disk)

```
git worktree add /tmp/pu_plotter origin/<plotting-branch>
RPP_PLOT_WORKTREE=/tmp/pu_plotter \
  python experiments/cosmosweb/run_procrustes_pipeline.py
cd /tmp/pu_plotter
python scripts/plot_crossarchitectural_procrustes.py
python scripts/plot_crossmodal_procrustes.py
python scripts/plot_crossmodal_procrustes_per_property.py
ls figs/*procrustes*.pdf
```

## Quick start (smoke-test subset, <2 GB peak disk)

```
STREAM_SUBSET="dinov3:vits16,dinov3:vitb16,convnext:nano,convnext:tiny,astropt:015M,astropt:095M" \
  python experiments/cosmosweb/stream_embeddings_to_npy.py
PWA_DATASET=HCVYM5w6Gn/cosmosweb-hsc-jwst-high-snr-pil2 \
PWA_OUT_DIR=analysis/probe_smoke \
PWA_EMB_DIR=analysis/probe_smoke/embeddings \
PWA_UPSAMPLE_SUFFIX="" \
  python experiments/cosmosweb/probe_weight_analysis.py
```

## Streaming details (peak local disk)

`stream_embeddings_to_npy.py` downloads one parquet to a scratch dir,
converts it to .npy under `STREAM_OUT_DIR`, then deletes the scratch dir
before pulling the next one. Peak local disk during streaming is bounded
by the largest single embedding parquet (~1 GB for `llava_15_13b` /
`paligemma_28b`). After streaming completes, the .npy directory holds
~30 GB if the full basket was processed.

Use `STREAM_SUBSET` (comma-separated `<family>:<size>` pairs) to limit the
basket — e.g. for smoke tests or for a per-family debugging run.

## Why we needed this

`probe_weight_analysis.py` consumes per-`(telescope, alias, size)` .npy
embedding files. The actual embeddings live on HF as parquet shards under
`HCVYM5w6Gn/pu-embeddings/cosmosweb/`. Previously the parquet → .npy
conversion + the script execution were undocumented manual steps on
contributors' personal machines, which is why the Procrustes pkl was not
reproducible by anyone else. This pipeline removes that dependency on a
specific person's local filesystem.

## Verifying the pkl

```
python -c "
import pickle
with open('analysis/procrustes_pipeline/procrustes_distances_45000galaxies.pkl','rb') as f:
    d = pickle.load(f)
print(list(d.keys()))
"
```

Expected top-level keys include `cross_modal_hsc_vs_jwst`, plus per-(model,
property) entries. the `plot_crossmodal_procrustes.py` reads
`cross_modal_hsc_vs_jwst[<property>]` as a list of
`(family, raw_size, params, distance)` tuples.
