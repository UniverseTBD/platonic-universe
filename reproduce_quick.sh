#!/usr/bin/env bash
# reproduce_quick.sh — render every paper figure that depends ONLY on the
# committed JSON file `r2_vs_params_45000galaxies_upsampled.json`. No
# downloads, no embedding extraction. Finishes in ~30 seconds.
#
# Use this first to verify your install is correct. The remaining figures
# (Procrustes / cosine-similarity) are produced by reproduce_full.sh.
set -euo pipefail

mkdir -p figs

echo "[1/6] r2_vs_params"
python scripts/plot_r2_vs_params.py
python scripts/plot_r2_vs_params_per_property.py

echo "[2/6] intramodal MKNN/CKA"
python scripts/plot_intramodal.py
python scripts/plot_intramodal_per_property.py

echo "[3/6] crossmodal MKNN/CKA"
python scripts/plot_crossmodal.py
python scripts/plot_crossmodal_per_property.py

echo
echo "Done. Figures written to figs/:"
ls -1 figs/*.pdf 2>/dev/null | sort
