#!/bin/bash
# Fan out the before/after calibrated CKA/MKNN residualization across all pairs
# of a model set, one CPU job per pair (normal partition, all parallel), reading
# the pre-computed cosmosweb embeddings from HF. No GPU, no extraction.
#
#   MODELS   space-separated stems as they appear in the parquet filenames
#   MODALITY hsc | jwst
#   CONFOUNDS  redshift,mass,sSFR
set -euo pipefail
cd "$(dirname "$0")"

MODELS=(${MODELS:-vit_base vit_large vit_huge convnext_nano convnext_tiny convnext_base convnext_large})
MODALITY="${MODALITY:-hsc}"
CONFOUNDS="${CONFOUNDS:-redshift,mass,sSFR}"
CV="${CV:-0}"   # 0 = in-sample; K>0 = cross-fitted (leakage-free) with K folds
# CONFOUNDS contains commas, which would break sbatch's comma-separated --export
# list (splitting "redshift,mass,sSFR" into separate entries). Export it so it
# propagates via --export=ALL instead of being an explicit list item.
export CONFOUNDS MODALITY

label() { echo "${1/_//}"; }   # vit_base -> vit/base

echo "=== fanning out pairs over ${#MODELS[@]} models ($MODALITY) ==="
n=0
for ((i=0; i<${#MODELS[@]}; i++)); do
  for ((j=i+1; j<${#MODELS[@]}; j++)); do
    A=${MODELS[$i]}; B=${MODELS[$j]}
    jid=$(sbatch --parsable --time=03:00:00 \
      --job-name="rz${CV:+cv}_${A}_x_${B}" \
      --export="ALL,A=$A,B=$B,LA=$(label "$A"),LB=$(label "$B"),CV=$CV" \
      slurm/residualize_hf.sh)
    echo "  $(label "$A") vs $(label "$B")  -> $jid"
    n=$((n+1))
  done
done
echo "=== submitted $n pair jobs (partition=normal, all parallel) ==="
echo "results -> /scratch/\$USER/residualize_out/residualize_${MODALITY}_*.json"
