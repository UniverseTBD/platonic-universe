#!/bin/bash
#SBATCH --job-name=pu_solve
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --gpus=1
#SBATCH --partition=ghx4
#SBATCH --account=<your-slurm-account>
#SBATCH --output=pu_solve_%j.log
#SBATCH --error=pu_solve_%j.err
# =============================================================================
#  pu_solve.sh — cross-cluster cooperative worker
#
#  Two clusters share a private HF dataset for task coordination. Both submit
#  this script repeatedly; workers atomically claim tasks via HF and never
#  duplicate work. A worker that finishes early simply picks up whatever's
#  unclaimed at that moment, regardless of which cluster left it.
#
#  USAGE — same on both clusters:
#    chmod +x pu_solve.sh
#    for i in {1..16}; do sbatch pu_solve.sh; done
#
#  ============================================================================
#  >>>>>>>>>>>>>>>  PER-CLUSTER CONFIG — EDIT THESE 6 LINES  <<<<<<<<<<<<<<<<<<
#
#    1. The path to your Python env activation (must have torch+CUDA, polars,
#       pyarrow, huggingface_hub, numpy installed).
#    2. The path to where you want OUTPUT files (small parquets) on a
#       persistent filesystem visible to all your worker nodes.
#    3. The path to where lock files live (also persistent, also shared).
#    4-6. SBATCH partition/account/module-load — SLURM-cluster-specific.
#  ============================================================================
# -- 1. Activate Python env on the compute node --------------------------------
PER_CLUSTER_ACTIVATE='source /path/to/your/venv/bin/activate'
# -- 2. Where final result parquets get written --------------------------------
PER_CLUSTER_OUT_DIR='/path/to/persistent/solve_out'
# -- 3. Where transient coordination lockfiles live ----------------------------
PER_CLUSTER_LOCK_DIR='/path/to/persistent/solve_locks'
# -- 4. Module load (set to ":" if your cluster doesn't use modules) -----------
PER_CLUSTER_MODULE_LOAD='module load python/miniforge3_pytorch/2.10.0'
# -- 5/6. SBATCH directives below. Edit partition/account/walltime/mem ---------
#  ============================================================================
#  >>>>>>>>>>>>>>>>>>>>>>>>>>>  END PER-CLUSTER  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# =============================================================================

set -euo pipefail

# ---- HF auth + cross-cluster coordination dataset ---------------------------
# This token can read+write the shared results dataset. If you have your own
# HF_TOKEN, set it BEFORE sbatch (it'll override): `HF_TOKEN=hf_xxx sbatch ...`
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be exported}"

# Shared dataset id. Both clusters point at the same one; pu_solve.py
# coordinates claims and uploads results here. Don't change unless you're
# starting a new run from scratch.
export PU_SOLVE_RESULTS_REPO="${PU_SOLVE_RESULTS_REPO:?must specify HF dataset id, e.g. <owner>/pu-solve-results}"

# ---- Activate Python env (per-cluster command from CONFIG above) ------------
$PER_CLUSTER_MODULE_LOAD
$PER_CLUSTER_ACTIVATE

# ---- Heavy throwaway I/O on /tmp (compute-node-local, no /work quota) -------
export HF_HOME="/tmp/hf_cache_${SLURM_JOB_ID:-$$}"
export PU_SOLVE_DLCACHE="/tmp/pu_solve_dl_${SLURM_JOB_ID:-$$}"
mkdir -p "$HF_HOME" "$PU_SOLVE_DLCACHE"

# ---- Persistent state on the cluster's filesystem (per-cluster from CONFIG) -
export PU_SOLVE_OUT="$PER_CLUSTER_OUT_DIR"
export PU_SOLVE_LOCKS="$PER_CLUSTER_LOCK_DIR"
mkdir -p "$PU_SOLVE_OUT" "$PU_SOLVE_LOCKS"

# Column cache budget — 200 GB / 512 GB allocated. Fits one entire side of
# paligemma_28b's columns (~131 GB peak). Remaining 312 GB absorbs pyarrow's
# transient combine_chunks doubling + Python overhead + GPU staging.
export PU_SOLVE_COL_CACHE_GB=200

# Clean up our /tmp scratch on exit (success or crash).
trap 'rm -rf "$HF_HOME" "$PU_SOLVE_DLCACHE" 2>/dev/null || true' EXIT

# Run from the submit directory (where pu_solve.py lives alongside this
# wrapper). SLURM copies the .sh to /var/spool/slurmd/, so we can't use $0.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
exec python pu_solve.py "$@"
