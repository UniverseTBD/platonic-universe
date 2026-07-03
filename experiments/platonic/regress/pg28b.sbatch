#!/bin/bash
# Headless PaliGemma 2 (28B) layerwise run: extract -> probe -> plot.
# 28B (~56 GB bf16) is sharded across the 3 allocated A100s by device_map="auto"
# in the adapter's load(); SLURM sets CUDA_VISIBLE_DEVICES to exactly those cards.
# Submit with:  sbatch experiments/platonic/regress/pg28b.sbatch
#SBATCH --job-name=pg28b_lw
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --time=16:00:00
#SBATCH --output=/beegfs/general/mjsmith/pu/logs/pg28b_%j.out

set -euo pipefail
cd /beegfs/general/mjsmith/pu

export HF_HOME=/beegfs/general/mjsmith/hf_cache
export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
export HF_HUB_DISABLE_XET=1
PY=/beegfs/general/mjsmith/pu/.venv/bin/python

echo "=== node=$(hostname) job=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

ORDERED=/beegfs/general/mjsmith/pu/data/cosmosweb/paligemma_28b_28b_blocks_layerwise_ordered.parquet
PROBE=/beegfs/general/mjsmith/pu/data/paligemma_28b_layerwise_probe.parquet

echo "=== [1/3] extract (num_workers=0, order-preserving) ==="
PU_EXTRACT_MODEL=paligemma_28b PU_EXTRACT_HFID=google/paligemma2-28b-mix-224 PU_EXTRACT_SIZE=28b \
  PU_EXTRACT_N=45000 PU_EXTRACT_BATCH=32 PU_EXTRACT_OUT="$ORDERED" \
  "$PY" experiments/platonic/regress/00_extract_cosmosweb_layerwise.py

echo "=== [2/3] full-depth probe sweep ==="
PU_LAYERWISE_INPUT="$ORDERED" PU_LAYERWISE_OUTPUT="$PROBE" \
  "$PY" experiments/platonic/regress/04_layerwise_probe.py

echo "=== [3/3] render figure ==="
PU_PLOT_INPUT="$PROBE" PU_PLOT_MODEL="PaliGemma 2 (28B)" \
  PU_PLOT_OUT=/beegfs/general/mjsmith/pu/figs/paligemma_layerwise_physics_28b.pdf \
  PU_PLOT_PAPER=28b \
  "$PY" scripts/plot_paligemma_layerwise.py

echo "=== DONE $(date) ==="
