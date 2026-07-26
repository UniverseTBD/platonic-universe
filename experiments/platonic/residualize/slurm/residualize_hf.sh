#!/bin/bash
#SBATCH --job-name=pu_residualize_hf
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=pu_residualize_hf_%j.log
#SBATCH --error=pu_residualize_hf_%j.err
#
# Calibrated CKA/MKNN before/after residualizing astrophysics, using the
# PRE-COMPUTED cosmosweb embeddings already on HF
# (UniverseTBD/pu-embeddings/cosmosweb/.../embeddings/*_45000.parquet),
# row-aligned to the cosmosweb labels by stream order. CPU-only, no extraction.
#
#   A/B         = model stems, e.g. vit_base, convnext_base   (as in the filenames)
#   LA/LB       = display labels
#   MODALITY    = hsc | jwst
#   CONFOUNDS   = redshift,mass,sSFR
set -euo pipefail
REPO="$HOME/repos/platonic-universe"
source "$REPO/.venv/bin/activate"
cd "$REPO/experiments/platonic/residualize"
export HF_TOKEN="${HF_TOKEN:-$(grep '^HF_token=' "$REPO/.env" | cut -d"'" -f2)}"
export HF_HOME="/tmp/hf_${SLURM_JOB_ID:-$$}"; mkdir -p "$HF_HOME"
trap 'rm -rf "$HF_HOME" 2>/dev/null || true' EXIT

EMB_REPO="UniverseTBD/pu-embeddings"
BASE="cosmosweb/pscratch/sd/a/ashodkh/platonic_universe/embeddings"
MOD="${MODALITY:-hsc}"
A="${A:-vit_base}"; B="${B:-convnext_base}"
LA="${LA:-vit/base}"; LB="${LB:-convnext/base}"
CONFOUNDS="${CONFOUNDS:-redshift,mass,sSFR}"

python residualize_align.py \
  --emb-repo "$EMB_REPO" \
  --tag-a "$BASE/${MOD}_embeddings_cosmosweb-hsc-jwst-high-snr-pil2_${A}_45000.parquet" \
  --tag-b "$BASE/${MOD}_embeddings_cosmosweb-hsc-jwst-high-snr-pil2_${B}_45000.parquet" \
  --label-a "$LA" --label-b "$LB" --modality "$MOD" \
  --confounds "$CONFOUNDS" --degree 3 --cv "${CV:-0}" \
  --n-sim 3000 --n-draws 3 --perms 200 \
  --out-dir "/scratch/$USER/residualize_out"
