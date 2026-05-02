#!/bin/bash
#SBATCH --job-name=pu_regress
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpus=1
#SBATCH --partition=<your-gpu-partition>
#SBATCH --account=<your-slurm-account>
#SBATCH --output=pu_regress_%j.log
#SBATCH --error=pu_regress_%j.err
# =============================================================================
#  pu_regress.sh — cooperative physics-regression worker.
#
#  Each worker atomically claims (modality, model, size) tuples from a private
#  HF coordination dataset, extracts embeddings on COSMOS-Web, runs a 5-fold
#  linear probe against the LEPHARE physics labels, uploads one tiny parquet
#  per tuple, and moves on.
#
#  USAGE — same on every cluster:
#    chmod +x pu_regress.sh
#    for i in {1..N}; do sbatch pu_regress.sh; done
#  ============================================================================
#  >>>>>>>>>>>>>>>  PER-CLUSTER CONFIG — EDIT THESE 4 LINES  <<<<<<<<<<<<<<<<<<
#  ============================================================================
PER_CLUSTER_ACTIVATE='source /path/to/your/venv/bin/activate'
PER_CLUSTER_OUT_DIR='/path/to/persistent/regress_out'
PER_CLUSTER_LOCK_DIR='/path/to/persistent/regress_locks'
PER_CLUSTER_MODULE_LOAD=':'
#  ============================================================================
set -euo pipefail

export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be exported}"
export PU_REGRESS_RESULTS_REPO="${PU_REGRESS_RESULTS_REPO:?must specify HF dataset id, e.g. <owner>/pu-regress-results}"

$PER_CLUSTER_MODULE_LOAD
$PER_CLUSTER_ACTIVATE

# Heavy throwaway cache on a compute-node-local /tmp.
export HF_HOME="/tmp/hf_cache_${SLURM_JOB_ID:-$$}"
export PU_REGRESS_DLCACHE="/tmp/pu_regress_dl_${SLURM_JOB_ID:-$$}"
mkdir -p "$HF_HOME" "$PU_REGRESS_DLCACHE"

# Persistent state.
export PU_REGRESS_OUT="$PER_CLUSTER_OUT_DIR"
export PU_REGRESS_LOCKS="$PER_CLUSTER_LOCK_DIR"
mkdir -p "$PU_REGRESS_OUT" "$PU_REGRESS_LOCKS"

# Sensible defaults for a single A100. Override via env if needed.
export PU_REGRESS_N_USE="${PU_REGRESS_N_USE:-45000}"
export PU_REGRESS_BATCH_SIZE="${PU_REGRESS_BATCH_SIZE:-16}"

trap 'rm -rf "$HF_HOME" "$PU_REGRESS_DLCACHE" 2>/dev/null || true' EXIT

cd "${SLURM_SUBMIT_DIR:-$PWD}"
exec python pu_regress.py "$@"
