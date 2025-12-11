#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/data/patrick/hf_cache
export CUDA_LAUNCH_BLOCKING=1

# Optional: pick a GPU id
GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
echo "[INFO] Using GPU: ${CUDA_VISIBLE_DEVICES}"

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$PROJECT_ROOT"

# DeepFashion2_SKU root directory
SKU_ROOT="${SKU_ROOT:-/data/patrick/10623GenAI/final_proj/data/DeepFashion2_SKU}"

# LoRA checkpoint position
OUT_DIR="/data/patrick/10623GenAI/final_proj/checkpoints/sd_lora_df2_v1_lr1e-4_bs4_ep1_r16"

echo "PROJECT_ROOT = ${PROJECT_ROOT}"
echo "SKU_ROOT     = ${SKU_ROOT}"
echo "OUT_DIR      = ${OUT_DIR}"
echo

python -m train.train_sd_lora_df2 \
  --sku_root "$SKU_ROOT" \
  --split train \
  --pretrained_model runwayml/stable-diffusion-v1-5 \
  --output_dir "$OUT_DIR" \
  --resolution 512 \
  --train_batch_size 4 \
  --num_train_epochs 1 \
  --learning_rate 1e-4 \
  --lr_scheduler_type cosine \
  --lr_warmup_ratio 0.05 \
  --gradient_accumulation_steps 1 \
  --lora_rank 16 \
  --lora_alpha 1.0 \
  --seed 16831 \
  --mixed_precision no \
  --wandb_project sd_lora_df2 \
  --wandb_run_name sd_lora_df2_v1_lr1e-4_bs4_ep1_r16
