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
# 0) Baseline1: CNN-REID, eval_mode=bestshot
########################################

# Paths (can be overridden by environment variables).
JSONL_ROOT=${JSONL_ROOT:-"./data"}
SKU_ROOT=${SKU_ROOT:-"/data/patrick/10623GenAI/final_proj/data/DeepFashion2_SKU"}
OUT_DIR=${OUT_DIR:-"/data/patrick/10623GenAI/final_proj/checkpoints/baseline1_reid_runagain"}

mkdir -p "${OUT_DIR}"

python -m train.train_reid_df2 \
  --jsonl_root "${JSONL_ROOT}" \
  --sku_root "${SKU_ROOT}" \
  --output_dir "${OUT_DIR}" \
  --batch_size 64 \
  --num_instances 4 \
  --epochs 40 \
  --lr 3e-4 \
  --weight_decay 5e-4 \
  --margin 0.3 \
  --img_size 224 \
  --num_workers 8 \
  --save_every 5 \
  --eval_every_iter 2000 \
  --ndcg_k 10 \
  --warmup_ratio 0.05 \
  --min_lr 1e-6  \
  --wandb \
  --wandb_project 10623finalproj_df2_all_baselines \
  --wandb_run_name "baseline1_reid_bs64_e40_bestshot"

########################################
# 1) Baseline3: CLIP unfrozen, eval_mode=bestshot
########################################

# Paths (can be overridden by environment variables).
SKU_ROOT="${SKU_ROOT:-/data/patrick/10623GenAI/final_proj/data/DeepFashion2_SKU}"
CKPT_OUT="${CKPT_OUT:-/data/patrick/10623GenAI/final_proj/checkpoints/baseline3_clip_sku_bs64/clip_sku_baseline_final.pt}"

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
  --eval_mode bestshot \
  --output "$CKPT_OUT" \
  --wandb \
  --wandb_project 10623finalproj_df2_all_baselines \
  --wandb_run_name baseline3_clip_sku_bs64_e40_bestshot

########################################
# 2) Baseline2: CLIP frozen towers, eval_mode=bestshot
########################################

# Paths (can be overridden by environment variables).
SKU_ROOT="${SKU_ROOT:-/data/patrick/10623GenAI/final_proj/data/DeepFashion2_SKU}"
CKPT_OUT="${CKPT_OUT:-/data/patrick/10623GenAI/final_proj/checkpoints/baseline2_clip_sku_frozen_bs64/clip_sku_frozen_baseline_final.pt}"

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
  --freeze_towers \
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
  --eval_mode bestshot \
  --output "$CKPT_OUT" \
  --wandb \
  --wandb_project 10623finalproj_df2_all_baselines \
  --wandb_run_name baseline2_clip_sku_bs64_e40_frozen_bestshot

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

########################################
# 4) Baseline5: CLIP frozen most, only unfrozen last two transformer blocks, eval_mode=bestshot
########################################

# Paths (can be overridden by environment variables).
SKU_ROOT="${SKU_ROOT:-/data/patrick/10623GenAI/final_proj/data/DeepFashion2_SKU}"
CKPT_OUT="${CKPT_OUT:-/data/patrick/10623GenAI/final_proj/checkpoints/baseline5_ft_bestshot_bs64_text0p1_unlock2/clip_sku_baseline_final.pt}"

echo "PROJECT_ROOT = ${PROJECT_ROOT}"
echo "SKU_ROOT     = ${SKU_ROOT}"
echo "CKPT_OUT     = ${CKPT_OUT}"
echo

CUDA_VISIBLE_DEVICES=0 python -m train.train_clip_sku_df2 \
  --sku_root "$SKU_ROOT" \
  --train_split train \
  --val_split validation \
  --clip_model ViT-B-16 \
  --clip_pretrained laion2b_s34b_b88k \
  --batch_size 64 \
  --epochs 20 \
  --lr 5e-4 \
  --min_lr 1e-6 \
  --clip_lr 5e-6 \
  --clip_min_lr 1e-6 \
  --warmup_ratio 0.10 \
  --weight_decay 1e-4 \
  --num_workers 8 \
  --device cuda \
  --partial_finetune \
  --vision_unlocked_groups 2 \
  --text_loss_weight 0.10 \
  --eval_mode bestshot \
  --eval_every_iter 5000 \
  --save_every_iter 5000 \
  --output "$CKPT_OUT" \
  --wandb \
  --wandb_project 10623finalproj_df2_all_baselines \
  --wandb_run_name baseline5_ft_bestshot_bs64_text0p1_unlock2

