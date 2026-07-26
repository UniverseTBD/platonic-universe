#!/bin/bash
# vast_umap.sh — launch N parallel UMAP workers on the same vast.ai box
# that's running the paligemma solve. UMAP is CPU-bound (faiss + UMAP), no
# GPU needed, so it doesn't compete with the solve workers.
#
# Each worker independently claims (model, survey) tuples via HF
# coordination and processes their candidate blocks.

set -euo pipefail

N_WORKERS="${N_WORKERS:-6}"
WORKDIR="${WORKDIR:-/root/pu}"
SCRATCH="${SCRATCH:-$WORKDIR}"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be exported}"

cd "$WORKDIR"

mkdir -p "$SCRATCH/umap_out" "$SCRATCH/umap_locks" "$SCRATCH/umap_dl" "$SCRATCH/umap_hf"

echo "==> launching $N_WORKERS UMAP workers (CPU-only, share box with solve)"

for ((g=0; g<N_WORKERS; g++)); do
    NAME="umap_w${g}"
    # Each worker gets its own DLCACHE so concurrent downloads of different
    # parquets don't collide (HF lock works per-blob, but we want isolation).
    tmux new-session -d -s "$NAME" "
        cd '$WORKDIR' && \
        export HF_TOKEN='$HF_TOKEN' && \
        export PU_UMAP_RESULTS_REPO="${PU_UMAP_RESULTS_REPO:?must specify HF dataset id, e.g. <owner>/pu-umap-results}" && \
        export PU_UMAP_EMBED_REPO="${PU_UMAP_EMBED_REPO:?must specify HF embeddings dataset id}" && \
        export PU_UMAP_OUT='$SCRATCH/umap_out' && \
        export PU_UMAP_LOCKS='$SCRATCH/umap_locks' && \
        export PU_UMAP_DLCACHE='$SCRATCH/umap_dl_w${g}' && \
        export HF_HOME='$SCRATCH/umap_hf_w${g}' && \
        mkdir -p \$PU_UMAP_DLCACHE \$HF_HOME && \
        /root/pu/venv/bin/python pu_umap.py 2>&1 | tee '$SCRATCH/umap_w${g}.log'
        echo --- worker $g done; sleep 3600
    "
    echo "  worker $g  tmux=$NAME"
done

echo
echo "==> all $N_WORKERS UMAP workers launched"
echo "tail -f $SCRATCH/umap_w*.log"
