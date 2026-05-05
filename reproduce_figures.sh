#!/usr/bin/env bash
# One-shot reproduction of every paper figure.
# Run from the repo root after `uv sync && uv pip install .`.
set -euo pipefail

# 1. Stream-convert cosmosweb embedding parquets to .npy on local disk.
python experiments/cosmosweb/stream_embeddings_to_npy.py

# 2. Fit linear probes, compute Procrustes distances + cosine similarity,
#    write the procrustes pickle that the next step consumes.
python experiments/cosmosweb/probe_weight_analysis.py

# 3. Render the procrustes paper figures.
python scripts/plot_crossarchitectural_procrustes.py
python scripts/plot_crossmodal_procrustes.py
python scripts/plot_crossmodal_procrustes_per_property.py

# 4. Render every MKNN/CKA figure (these only need the committed JSON).
python scripts/plot_r2_vs_params.py
python scripts/plot_r2_vs_params_per_property.py
python scripts/plot_intramodal.py
python scripts/plot_intramodal_per_property.py
python scripts/plot_crossmodal.py
python scripts/plot_crossmodal_per_property.py

echo
echo "All figures written to figs/."
