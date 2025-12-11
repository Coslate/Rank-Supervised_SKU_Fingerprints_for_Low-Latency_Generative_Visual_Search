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
# 3) Extra run: CLIP unfrozen, eval_mode=mean_sku
########################################

SKU_ROOT="${SKU_ROOT:-/data/patrick/10623GenAI/final_proj/data/DeepFashion2_SKU}"
CKPT_OUT="${CKPT_OUT:-/data/patrick/10623GenAI/final_proj/checkpoints/baseline3_clip_sku_mean_sku_bs64/clip_sku_mean_sku_baseline_final.pt}"

echo "PROJECT_ROOT = ${PROJECT_ROOT}"
echo "SKU_ROOT     = ${SKU_ROOT}"
echo "CKPT_OUT     = ${CKPT_OUT}"
echo

python -m train.train_clip_sku_df2 \
  --sku_root "$SKU_ROOT" \
  --train_split train \
  --val_split validation \
  --clip_model ViT-B-16 \
  --clip_pretrained laion2b_s34b_b88k \
  --batch_size 64 \
  --epochs 40 \
  --lr 1e-3 \
  --min_lr 1e-6 \
  --warmup_ratio 0.05 \
  --weight_decay 1e-4 \
  --num_workers 8 \
  --device cuda \
  --eval_every_iter 2000 \
  --save_every_iter 2000 \
  --eval_mode mean_sku \
  --output "$CKPT_OUT" \
  --wandb \
  --wandb_project 10623finalproj_df2_all_baselines \
  --wandb_run_name baseline3_clip_sku_bs64_e40_mean_sku
