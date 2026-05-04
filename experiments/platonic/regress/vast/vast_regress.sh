#!/bin/bash
# vast_regress.sh — launch N pu_regress workers on a single multi-GPU box,
# one worker per visible GPU. Each worker independently claims tuples from
# the shared HF coordination dataset.
#
# USAGE on a freshly-rented box:
#   1. ensure venv is set up and 01_extract_and_probe.py + setup_hf.py are at $WORKDIR
#   2. export HF_TOKEN=... PU_REGRESS_RESULTS_REPO=<owner>/<dataset>
#   3. bash vast_regress.sh

set -euo pipefail

N_GPUS="${N_GPUS:-$(nvidia-smi -L | wc -l)}"
WORKDIR="${WORKDIR:-$HOME/pu_regress}"
SCRATCH="${SCRATCH:-$WORKDIR}"

export HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be exported}"
export PU_REGRESS_RESULTS_REPO="${PU_REGRESS_RESULTS_REPO:?must specify HF dataset id}"

cd "$WORKDIR"

mkdir -p "$SCRATCH/regress_out" "$SCRATCH/regress_locks"

echo "==> launching $N_GPUS pu_regress workers"

for ((g=0; g<N_GPUS; g++)); do
    NAME="reg_g${g}"
    tmux new-session -d -s "$NAME" "
        cd '$WORKDIR' && \
        export HF_TOKEN='$HF_TOKEN' && \
        export PU_REGRESS_RESULTS_REPO='$PU_REGRESS_RESULTS_REPO' && \
        export CUDA_VISIBLE_DEVICES='$g' && \
        export PU_REGRESS_OUT='$SCRATCH/regress_out' && \
        export PU_REGRESS_LOCKS='$SCRATCH/regress_locks' && \
        export PU_REGRESS_DLCACHE='$SCRATCH/regress_dl_g${g}' && \
        export HF_HOME='$SCRATCH/hf_cache_g${g}' && \
        mkdir -p \$PU_REGRESS_DLCACHE \$HF_HOME && \
        python 01_extract_and_probe.py 2>&1 | tee '$SCRATCH/regress_g${g}.log'
        echo --- worker $g done; sleep 3600
    "
    echo "  worker $g  GPU=$g  tmux=$NAME"
done

echo
echo "monitor:"
echo "  tmux ls"
echo "  tail -f $SCRATCH/regress_g*.log"
