#!/bin/bash
#SBATCH --job-name=pu_control
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=pu_control_%j.log
#SBATCH --error=pu_control_%j.err
#
# Random-direction control: before / physics-removed / random-removed(matched amplitude)
# for one pair, using the pre-computed cosmosweb embeddings on HF. CPU-only.
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
CONFOUNDS="${CONFOUNDS:-redshift,mass,sSFR}"; NRAND="${NRAND:-10}"

python residualize_control.py \
  --emb-repo "$EMB_REPO" \
  --tag-a "$BASE/${MOD}_embeddings_cosmosweb-hsc-jwst-high-snr-pil2_${A}_45000.parquet" \
  --tag-b "$BASE/${MOD}_embeddings_cosmosweb-hsc-jwst-high-snr-pil2_${B}_45000.parquet" \
  --label-a "$LA" --label-b "$LB" --modality "$MOD" \
  --confounds "$CONFOUNDS" --degree 3 --n-sim 3000 --perms 200 --n-rand "$NRAND" \
  --out-dir "/scratch/$USER/residualize_out"
