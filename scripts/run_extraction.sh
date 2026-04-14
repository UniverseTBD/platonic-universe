#!/bin/bash

# ── SLURM directives ─────────────────────────────────────────────────────
#SBATCH --job-name=pu_extract
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=408G
#SBATCH --gpus=1
#SBATCH --partition=ghx4
#SBATCH --account=bfir-dtai-gh
#SBATCH --output=pu_extract_%j.log
#SBATCH --error=pu_extract_%j.err

# ── Environment ──────────────────────────────────────────────────────────
module load python/miniforge3_pytorch/2.10.0
source /work/nvme/bfir/kduraphe/pu/pu_env/bin/activate

BASE_DIR="/work/nvme/bfir/kduraphe/pu"
REPO_DIR="$BASE_DIR/platonic-universe"
OUTPUT_DIR="$BASE_DIR/output"
cd "$REPO_DIR"

export HF_TOKEN="${HF_TOKEN:-hf_REPLACE_ME}"
HF_REPO="kshitijd/platonic-embeddings"
export HF_HOME="$BASE_DIR/.cache/huggingface"

echo "Platonic Universe extraction — $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Output: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

# ── Disk check ───────────────────────────────────────────────────────────
AVAIL_GB=$(df --output=avail -BG "$OUTPUT_DIR" | tail -1 | tr -d ' G')
echo "Available disk: ${AVAIL_GB} GB"
if [ "$AVAIL_GB" -lt 300 ]; then
    echo "WARNING: Less than 300 GB free."
fi

# ── Model × Dataset matrix ──────────────────────────────────────────────
VISION_MODELS=(
    "convnext"
    "hiera"
    "dino"
    "clip"
    "vit"
    "vit-mae"
    "ijepa"
    "vjepa"
    "dinov3"
    "sam2"
    "astropt"
)

VLM_MODELS=(
    "paligemma_3b"
    "paligemma_10b"
    "llava_15"
    "llava_ov"
    "paligemma_28b"
)

DATASETS=("desi" "jwst" "legacysurvey" "sdss")

declare -A BATCH_SIZES
BATCH_SIZES[convnext]=64
BATCH_SIZES[hiera]=64
BATCH_SIZES[dino]=64
BATCH_SIZES[clip]=64
BATCH_SIZES[vit]=64
BATCH_SIZES[vit-mae]=64
BATCH_SIZES[ijepa]=32
BATCH_SIZES[vjepa]=32
BATCH_SIZES[dinov3]=16
BATCH_SIZES[sam2]=32
BATCH_SIZES[astropt]=64
BATCH_SIZES[paligemma_3b]=8
BATCH_SIZES[paligemma_10b]=4
BATCH_SIZES[paligemma_28b]=2
BATCH_SIZES[llava_15]=8
BATCH_SIZES[llava_ov]=8

# ── Run extraction ───────────────────────────────────────────────────────
TOTAL=0
FAILED=0
SKIPPED=0

run_model() {
    local model=$1
    local bs=${BATCH_SIZES[$model]:-32}

    for ds in "${DATASETS[@]}"; do
        TOTAL=$((TOTAL + 1))
        local marker="${OUTPUT_DIR}/.done_${ds}_${model}"
        if [ -f "$marker" ]; then
            echo "[skip] ${model} on ${ds} — already done"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi

        echo ""
        echo "=========================================="
        echo "[${TOTAL}] ${model} on ${ds} (batch_size=${bs}) — $(date)"
        echo "=========================================="

        python3 -m pu extract-layers \
            --model "$model" \
            --mode "$ds" \
            --batch-size "$bs" \
            --hf-repo "$HF_REPO" \
            --hf-token "$HF_TOKEN" \
            --delete-after-upload \
            --output-dir "$OUTPUT_DIR" \
            2>&1

        if [ $? -eq 0 ]; then
            touch "$marker"
            echo "[done] ${model} on ${ds}"
        else
            echo "[FAIL] ${model} on ${ds}"
            FAILED=$((FAILED + 1))
        fi

        AVAIL_GB=$(df --output=avail -BG "$OUTPUT_DIR" | tail -1 | tr -d ' G')
        if [ "$AVAIL_GB" -lt 200 ]; then
            echo "WARNING: Only ${AVAIL_GB} GB free."
        fi
    done
}

echo ""
echo "=== Vision models ==="
for model in "${VISION_MODELS[@]}"; do
    run_model "$model"
done

echo ""
echo "=== VLMs ==="
for model in "${VLM_MODELS[@]}"; do
    run_model "$model"
done

# ── Summary ──────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Extraction complete — $(date)"
echo "Total: $TOTAL, Skipped: $SKIPPED, Failed: $FAILED"
echo "=========================================="

if [ $FAILED -gt 0 ]; then
    echo "Failed jobs — resubmit to retry (done markers prevent re-running successes)"
    exit 1
fi
