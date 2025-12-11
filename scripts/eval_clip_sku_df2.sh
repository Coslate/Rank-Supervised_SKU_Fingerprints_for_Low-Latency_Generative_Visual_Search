#!/usr/bin/env bash
set -euo pipefail

# Optional: pick a GPU (uncomment if needed)
GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$PROJECT_ROOT"

echo
echo "----------------------------------------------------------------------------------"
echo "Now running bestshot evaluation (bestshot fingerprint) on baselin2_clip_sku_frozen"
echo "----------------------------------------------------------------------------------"
echo

SKU_ROOT="${SKU_ROOT:-/data/patrick/10623GenAI/final_proj/data/DeepFashion2_SKU}"
CKPT_PATH="/data/patrick/10623GenAI/final_proj/checkpoints/baseline2_clip_sku_frozen_bs64/clip_sku_frozen_baseline_final.pt"

echo "PROJECT_ROOT = ${PROJECT_ROOT}"
echo "SKU_ROOT     = ${SKU_ROOT}"
echo "CKPT_PATH    = ${CKPT_PATH}"
echo

# -------- bestshot evaluation (image-level gallery, max per SKU in compute_metrics) --------
python -m eval.eval_clip_sku_df2 \
  --sku_root "$SKU_ROOT" \
  --split validation \
  --checkpoint "$CKPT_PATH" \
  --clip_model ViT-B-16 \
  --clip_pretrained laion2b_s34b_b88k \
  --batch_size 256 \
  --num_workers 8 \
  --device cuda \
  --ndcg_k 10 \
  --recall_ks 1 5 10 \
  --eval_all_val_skus \
  --eval_cpu_latency \
  --eval_mode bestshot

echo
echo "---------------------------------------------------------------------------------------"
echo "Now running mean_sku evaluation (per-SKU mean fingerprint) on baseline2_clip_sku_frozen"
echo "---------------------------------------------------------------------------------------"
echo

SKU_ROOT="${SKU_ROOT:-/data/patrick/10623GenAI/final_proj/data/DeepFashion2_SKU}"
CKPT_PATH="/data/patrick/10623GenAI/final_proj/checkpoints/baseline2_clip_sku_frozen_bs64/clip_sku_frozen_baseline_final.pt"

echo "PROJECT_ROOT = ${PROJECT_ROOT}"
echo "SKU_ROOT     = ${SKU_ROOT}"
echo "CKPT_PATH    = ${CKPT_PATH}"
echo

# -------- mean_sku evaluation (per-SKU mean catalog embedding) --------
python -m eval.eval_clip_sku_df2 \
  --sku_root "$SKU_ROOT" \
  --split validation \
  --checkpoint "$CKPT_PATH" \
  --clip_model ViT-B-16 \
  --clip_pretrained laion2b_s34b_b88k \
  --batch_size 256 \
  --num_workers 8 \
  --device cuda \
  --ndcg_k 10 \
  --recall_ks 1 5 10 \
  --eval_all_val_skus \
  --eval_cpu_latency \
  --eval_mode mean_sku

echo
echo "---------------------------------------------------------------------------------------"
echo "Now running bestshot evaluation (bestshot fingerprint) on baseline5_clip_sku_finetuned"
echo "---------------------------------------------------------------------------------------"
echo

SKU_ROOT="${SKU_ROOT:-/data/patrick/10623GenAI/final_proj/data/DeepFashion2_SKU}"
CKPT_PATH="/data/patrick/10623GenAI/final_proj/checkpoints/baseline5_ft_bestshot_bs64_text0p1_unlock2_ep6_lr3e-4/clip_sku_baseline_final_best.pt"

echo "PROJECT_ROOT = ${PROJECT_ROOT}"
echo "SKU_ROOT     = ${SKU_ROOT}"
echo "CKPT_PATH    = ${CKPT_PATH}"
echo

# -------- bestshot evaluation (image-level gallery, max per SKU in compute_metrics) --------
python -m eval.eval_clip_sku_df2 \
  --sku_root "$SKU_ROOT" \
  --split validation \
  --checkpoint "$CKPT_PATH" \
  --clip_model ViT-B-16 \
  --clip_pretrained laion2b_s34b_b88k \
  --batch_size 64 \
  --num_workers 8 \
  --device cuda \
  --ndcg_k 10 \
  --recall_ks 1 5 10 \
  --eval_all_val_skus \
  --eval_cpu_latency \
  --eval_mode bestshot

echo
echo "---------------------------------------------------------------------------------------"
echo "Now running mean_sku evaluation (per-SKU mean fingerprint) on baseline5_clip_sku_finetuned"
echo "---------------------------------------------------------------------------------------"
echo

SKU_ROOT="${SKU_ROOT:-/data/patrick/10623GenAI/final_proj/data/DeepFashion2_SKU}"
CKPT_PATH="/data/patrick/10623GenAI/final_proj/checkpoints/baseline5_ft_bestshot_bs64_text0p1_unlock2_ep6_lr3e-4/clip_sku_baseline_final_best.pt"

echo "PROJECT_ROOT = ${PROJECT_ROOT}"
echo "SKU_ROOT     = ${SKU_ROOT}"
echo "CKPT_PATH    = ${CKPT_PATH}"
echo

# -------- mean_sku evaluation (per-SKU mean catalog embedding) --------
python -m eval.eval_clip_sku_df2 \
  --sku_root "$SKU_ROOT" \
  --split validation \
  --checkpoint "$CKPT_PATH" \
  --clip_model ViT-B-16 \
  --clip_pretrained laion2b_s34b_b88k \
  --batch_size 64 \
  --num_workers 8 \
  --device cuda \
  --ndcg_k 10 \
  --recall_ks 1 5 10 \
  --eval_all_val_skus \
  --eval_cpu_latency \
  --eval_mode mean_sku