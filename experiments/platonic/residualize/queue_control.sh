#!/bin/bash
# Fan out the random-direction control (before / physics / random-matched) across
# all pairs of a model set, one CPU job per pair (normal partition, all parallel).
set -euo pipefail
cd "$(dirname "$0")"

MODELS=(${MODELS:-vit_base vit_large vit_huge convnext_nano convnext_tiny convnext_base convnext_large})
MODALITY="${MODALITY:-hsc}"
CONFOUNDS="${CONFOUNDS:-redshift,mass,sSFR}"
NRAND="${NRAND:-10}"
export CONFOUNDS MODALITY NRAND

label() { echo "${1/_//}"; }

echo "=== control fan-out over ${#MODELS[@]} models ($MODALITY, n_rand=$NRAND) ==="
n=0
for ((i=0; i<${#MODELS[@]}; i++)); do
  for ((j=i+1; j<${#MODELS[@]}; j++)); do
    A=${MODELS[$i]}; B=${MODELS[$j]}
    jid=$(sbatch --parsable --job-name="ctl_${A}_x_${B}" \
      --export="ALL,A=$A,B=$B,LA=$(label "$A"),LB=$(label "$B")" \
      slurm/control_hf.sh)
    echo "  $(label "$A") vs $(label "$B")  -> $jid"
    n=$((n+1))
  done
done
echo "=== submitted $n control jobs ==="
