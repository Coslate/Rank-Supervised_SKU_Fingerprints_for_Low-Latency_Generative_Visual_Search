#!/usr/bin/env bash
set -e

# Optional: pick a GPU (uncomment if needed)
GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"

JSONL_ROOT=${JSONL_ROOT:-"./data"}
SKU_ROOT=${SKU_ROOT:-"/data/patrick/10623GenAI/final_proj/data/DeepFashion2_SKU"}
CKPT="/data/patrick/10623GenAI/final_proj/checkpoints/baseline1_reid_run3/baseline1_reid_epoch40.pt"

python -m eval.eval_reid_df2 \
  --jsonl_root "${JSONL_ROOT}" \
  --sku_root "${SKU_ROOT}" \
  --split "validation" \
  --checkpoint "${CKPT}" \
  --batch_size 64 \
  --eval_cpu_latency
