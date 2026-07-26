#!/bin/bash
# vast_paligemma.sh — launch N pair-range partition workers for the
# 244-GB legacysurvey/paligemma_28b_28b task on a single multi-GPU box.
#
# Each worker is pinned to one GPU via CUDA_VISIBLE_DEVICES and gets a
# disjoint slice of the 5476-pair grid. All workers share the same local
# parquet (one download), so disk usage is ~480 GB regardless of N.
#
# USAGE:
#   1. Provision a vast.ai box with N GPUs, ≥256 GB RAM, ≥600 GB SSD.
#   2. ssh in.
#   3. scp pu_solve.py merge_partials.py vast_paligemma.sh to ~/pu/
#   4. cd ~/pu && bash vast_paligemma.sh
#
# Tunables:
#   N_GPUS              — number of workers (default 8)
#   N_PAIRS             — total pairs in the grid; the script will split
#                         this into N_GPUS contiguous ranges. Default 5476
#                         (74×74 for paligemma 28b legacysurvey after lm_head
#                         filtering). Override if you confirm a different shape.
#   COL_CACHE_GB_TOTAL  — total RAM budget across all workers. Each worker
#                         gets COL_CACHE_GB_TOTAL / N_GPUS.

set -euo pipefail

# ---- per-box config — EDIT THESE -------------------------------------------
TARGET_TUPLE="legacysurvey/paligemma_28b_28b"
N_GPUS="${N_GPUS:-8}"
N_PAIRS="${N_PAIRS:-5476}"
COL_CACHE_GB_TOTAL="${COL_CACHE_GB_TOTAL:-200}"

WORKDIR="${WORKDIR:-$HOME/pu}"
SCRATCH="${SCRATCH:-$WORKDIR}"          # local SSD root for download + outputs
HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be exported}"
# ----------------------------------------------------------------------------

cd "$WORKDIR"

if [ ! -f pu_solve.py ]; then
    echo "[fatal] pu_solve.py not in $WORKDIR" >&2
    exit 1
fi

mkdir -p "$SCRATCH/solve_out" "$SCRATCH/solve_locks" "$SCRATCH/dl_cache" "$SCRATCH/hf_cache"

PER_WORKER_CACHE=$(( COL_CACHE_GB_TOTAL / N_GPUS ))
PAIR_STEP=$(( (N_PAIRS + N_GPUS - 1) / N_GPUS ))

echo "==> launching $N_GPUS workers"
echo "    target           : $TARGET_TUPLE"
echo "    pair grid        : $N_PAIRS"
echo "    pairs per worker : $PAIR_STEP"
echo "    cache GB / worker: $PER_WORKER_CACHE"
echo "    scratch          : $SCRATCH"
echo

for ((g=0; g<N_GPUS; g++)); do
    START=$(( g * PAIR_STEP ))
    END=$(( START + PAIR_STEP ))
    if (( END > N_PAIRS )); then END=$N_PAIRS; fi
    if (( START >= N_PAIRS )); then break; fi
    NAME="pal_g${g}"
    echo "  worker $g  GPU=$g  range=$START:$END  tmux=$NAME"

    tmux new-session -d -s "$NAME" "
        cd '$WORKDIR' && \
        export HF_TOKEN='$HF_TOKEN' && \
        export PU_SOLVE_TARGET='$TARGET_TUPLE' && \
        export PU_SOLVE_PAIR_RANGE='${START}:${END}' && \
        export CUDA_VISIBLE_DEVICES='$g' && \
        export PU_SOLVE_OUT='$SCRATCH/solve_out' && \
        export PU_SOLVE_LOCKS='$SCRATCH/solve_locks' && \
        export PU_SOLVE_DLCACHE='$SCRATCH/dl_cache' && \
        export HF_HOME='$SCRATCH/hf_cache' && \
        export PU_SOLVE_COL_CACHE_GB='$PER_WORKER_CACHE' && \
        export PU_SOLVE_MAX_BLOCK_DIM=16384 && \
        python pu_solve.py 2>&1 | tee '$SCRATCH/solve_g${g}.log'
        echo --- worker $g done; sleep 3600
    "
done

echo
echo "==> all $N_GPUS workers launched in tmux sessions: pal_g0..pal_g$((N_GPUS-1))"
echo
echo "monitor:"
echo "  tmux ls"
echo "  tmux attach -t pal_g0           # peek into worker 0"
echo "  tail -f $SCRATCH/solve_g*.log"
echo "  watch -n 5 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv'"
echo
echo "after all workers exit:"
echo "  python merge_partials.py legacysurvey paligemma_28b_28b \\"
echo "      --out-dir $SCRATCH/solve_out \\"
echo "      --upload-to-hf <owner>/pu-solve-results"
