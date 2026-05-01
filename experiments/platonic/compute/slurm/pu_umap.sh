#!/bin/bash
#SBATCH --job-name=pu_umap
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --partition=ghx4
#SBATCH --account=<your-slurm-account>
#SBATCH --output=pu_umap_%j.log
#SBATCH --error=pu_umap_%j.err
# =============================================================================
#  pu_umap.sh — cross-cluster cooperative UMAP/kNN-purity worker.
#
#  Same coordination model as pu_solve.sh: workers atomically claim
#  (model, survey) tuples via a private HF dataset, do all candidate
#  blocks for that tuple, upload tiny umap_*.parquet outputs, release.
#
#  USAGE — same on both clusters (after pu_umap_setup_hf.py has been
#  run + candidates.parquet/labels uploaded to static/):
#    chmod +x pu_umap.sh
#    for i in {1..16}; do sbatch pu_umap.sh; done
#
#  ============================================================================
#  >>>>>>>>>>>>>>>  PER-CLUSTER CONFIG — EDIT THESE 4 LINES  <<<<<<<<<<<<<<<<<<
#  ============================================================================
PER_CLUSTER_ACTIVATE='source /path/to/your/venv/bin/activate'
PER_CLUSTER_OUT_DIR='/path/to/persistent/umap_out'
PER_CLUSTER_LOCK_DIR='/path/to/persistent/umap_locks'
PER_CLUSTER_MODULE_LOAD='module load python/miniforge3_pytorch/2.10.0'
#  ============================================================================
#  >>>>>>>>>>>>>>>>>>>>>>>>>>>  END PER-CLUSTER  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# =============================================================================
set -euo pipefail

export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be exported}"
export PU_UMAP_RESULTS_REPO="${PU_UMAP_RESULTS_REPO:?must specify HF dataset id, e.g. <owner>/pu-umap-results}"
export PU_UMAP_EMBED_REPO="${PU_UMAP_EMBED_REPO:?must specify HF embeddings dataset id, e.g. <owner>/platonic-embeddings}"

$PER_CLUSTER_MODULE_LOAD
$PER_CLUSTER_ACTIVATE

# Heavy embedding parquets land on /tmp (compute-node-local) and are
# deleted between tuples. No persistent download cache.
export HF_HOME="/tmp/hf_cache_${SLURM_JOB_ID:-$$}"
export PU_UMAP_DLCACHE="/tmp/pu_umap_dl_${SLURM_JOB_ID:-$$}"
mkdir -p "$HF_HOME" "$PU_UMAP_DLCACHE"

export PU_UMAP_OUT="$PER_CLUSTER_OUT_DIR"
export PU_UMAP_LOCKS="$PER_CLUSTER_LOCK_DIR"
mkdir -p "$PU_UMAP_OUT" "$PU_UMAP_LOCKS"

trap 'rm -rf "$HF_HOME" "$PU_UMAP_DLCACHE" 2>/dev/null || true' EXIT

cd "${SLURM_SUBMIT_DIR:-$PWD}"
exec python pu_umap.py "$@"
