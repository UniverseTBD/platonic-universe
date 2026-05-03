#!/bin/bash
#SBATCH --job-name=pu_layerwise
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --gpus=1
#SBATCH --partition=<your-gpu-partition>
#SBATCH --account=<your-slurm-account>
#SBATCH --output=pu_layerwise_%j.log
#SBATCH --error=pu_layerwise_%j.err
# =============================================================================
#  extract_layerwise.sh — cooperative layer-wise extractor on COSMOS-Web.
#
#  ONE script, intentionally over-provisioned for the worst-case tuple
#  (PaliGemma-28B → 24 h wall, 320 GB RAM, 80 GB GPU). Coordinated via the
#  HF dataset specified by PU_LW_COORD_REPO — every job claims tuples atomically
#  through `running/<tag>.running` markers so any number of workers across
#  Delta + vast.ai cooperate without speaking to each other.
#
#  Usage — fire one job per family alias (the script picks resources by alias):
#
#    export HF_TOKEN=<token>
#    export PU_LW_OUT_DIR=$WORK/layerwise_cosmosweb     # persistent, shared
#    export PU_LW_LOCK_DIR=$WORK/layerwise_locks         # persistent, shared
#    export PU_LW_COORD_REPO=<owner>/pu-cosmosweb-layerwise
#
#    for m in vit clip convnext vit-mae astropt ijepa dinov3 vjepa \
#             llava_15 paligemma_3b paligemma_10b paligemma_28b; do
#        sbatch experiments/platonic/extract/slurm/extract_layerwise.sh "$m"
#    done
#
#  Spam more jobs than there are tuples — they'll just see "queue exhausted"
#  and exit. Each tuple's parquet lands at:
#    $PU_LW_OUT_DIR/<survey>/<family>_<size>_blocks_layerwise.parquet
#  and (because PU_LW_COORD_REPO is also the upload target) at:
#    https://huggingface.co/datasets/$PU_LW_COORD_REPO
# =============================================================================
#  >>>>>>>>>>>>>>>  PER-CLUSTER CONFIG — EDIT THESE 3 LINES  <<<<<<<<<<<<<<<<<<
# =============================================================================
PER_CLUSTER_ACTIVATE='source /path/to/your/venv/bin/activate'
PER_CLUSTER_MODULE_LOAD=':'
PER_CLUSTER_SCRATCH='/scratch/$USER'
# =============================================================================
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN must be exported}"
: "${PU_LW_OUT_DIR:?PU_LW_OUT_DIR must be set (persistent shared output dir)}"
: "${PU_LW_LOCK_DIR:?PU_LW_LOCK_DIR must be set (persistent shared lock dir)}"
: "${PU_LW_COORD_REPO:?PU_LW_COORD_REPO must be set (HF dataset id)}"

PU_LW_SURVEYS="${PU_LW_SURVEYS:-cosmosweb}"

# Family alias is positional arg 1, or "all" → run every family in this job.
ALIAS="${1:-all}"
ALL_FAMILIES="vit clip convnext vit-mae astropt ijepa dinov3 vjepa \
              llava_15 paligemma_3b paligemma_10b paligemma_28b"
if [ "$ALIAS" = "all" ]; then
    MODELS="$ALL_FAMILIES"
else
    MODELS="$ALIAS"
fi

# Per-alias initial batch size — `extract_layerwise_one` adaptively halves
# on CUDA OOM, so set generous defaults that A100/H100 can sustain and let
# smaller cards (or 28B's worst quirks) auto-fall back.
case "$ALIAS" in
    paligemma_28b)             BATCH=2  ;;
    paligemma_10b|llava_15)    BATCH=8  ;;
    paligemma_3b|dinov3|vjepa) BATCH=32 ;;
    *)                         BATCH=64 ;;
esac
export PU_LW_BATCH_SIZE="$BATCH"

$PER_CLUSTER_MODULE_LOAD
$PER_CLUSTER_ACTIVATE

# Throwaway HF cache lives on compute-node-local scratch so multiple workers
# on the same job never collide.
SCRATCH_BASE="$(eval echo "$PER_CLUSTER_SCRATCH")/lw_${SLURM_JOB_ID:-$$}"
export HF_HOME="$SCRATCH_BASE/hf"
mkdir -p "$HF_HOME" "$PU_LW_OUT_DIR" "$PU_LW_LOCK_DIR"

# Reduce memory churn during forward passes; matters most for PaliGemma-28B.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false

trap 'rm -rf "$SCRATCH_BASE" 2>/dev/null || true' EXIT

cd "${SLURM_SUBMIT_DIR:-$PWD}"

echo "==> alias:        $ALIAS"
echo "==> models:       $MODELS"
echo "==> surveys:      $PU_LW_SURVEYS"
echo "==> batch:        $BATCH"
echo "==> coord_repo:   $PU_LW_COORD_REPO"
echo "==> out_dir:      $PU_LW_OUT_DIR"
echo "==> lock_dir:     $PU_LW_LOCK_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

# shellcheck disable=SC2086
exec python experiments/platonic/extract/extract_layerwise.py \
    --models  $MODELS \
    --surveys $PU_LW_SURVEYS
