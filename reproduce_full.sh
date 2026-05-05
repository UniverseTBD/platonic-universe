#!/usr/bin/env bash
# reproduce_full.sh — full end-to-end reproduction.
#
# Runs reproduce_quick.sh first (cheap figures from the committed JSON),
# then pulls embeddings from Hugging Face, fits linear probes, computes
# Procrustes distances, and renders the remaining paper figures.
#
# ~30 GB of bandwidth, ~30 minutes wall-clock on a typical home connection.
set -euo pipefail

# === Phase A: cheap figures (no HF downloads) ====================
bash "$(dirname "$0")/reproduce_quick.sh"

# === Phase B: HF download + probe analysis =======================
echo
echo "[A] streaming cosmosweb embeddings (~30 GB) → .npy"
python experiments/cosmosweb/stream_embeddings_to_npy.py

echo "[B] linear probes + Procrustes pickle + cosine-similarity matrices"
python experiments/cosmosweb/probe_weight_analysis.py

echo "[C] rendering Procrustes paper figures"
python scripts/plot_crossarchitectural_procrustes.py
python scripts/plot_crossmodal_procrustes.py
python scripts/plot_crossmodal_procrustes_per_property.py

echo
echo "Done. Figures written to figs/:"
ls -1 figs/*.pdf 2>/dev/null | sort
