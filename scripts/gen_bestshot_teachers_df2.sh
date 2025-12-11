#!/usr/bin/env bash
set -euo pipefail

# Optional: pick a GPU (uncomment if needed)
GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"

# Resolve project root as the parent of this script directory.
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$PROJECT_ROOT"

########################################
# 4) Baseline5: CLIP frozen most, only unfrozen last two transformer blocks, eval_mode=bestshot
########################################

# Paths (can be overridden by environment variables).
SKU_ROOT="${SKU_ROOT:-/data/patrick/10623GenAI/final_proj/data/DeepFashion2_SKU}"
CKPT="${CKPT_OUT:-/data/patrick/10623GenAI/final_proj/checkpoints/baseline5_ft_bestshot_bs64_text0p1_unlock2_ep6_lr3e-4/clip_sku_baseline_final_best.pt}"
OUT="${OUT:-/data/patrick/10623GenAI/final_proj/data/DeepFashion2_SKU/bestshot_teachers/baseline5_ft_bestshot_bs64_text0p1_unlock2_ep6_lr3e-4/bestshot_teachers_with_text_train.npz}"

echo "PROJECT_ROOT = ${PROJECT_ROOT}"
echo "SKU_ROOT     = ${SKU_ROOT}"
echo "CKPT         = ${CKPT}"
echo "OUT          = ${OUT}"
echo

python -m scripts.gen_bestshot_teachers_df2 \
  --sku_root "$SKU_ROOT" \
  --ckpt_path "$CKPT" \
  --split train \
  --text_split train \
  --batch_size 256 \
  --num_workers 8 \
  --device cuda \
  --output "$OUT"
