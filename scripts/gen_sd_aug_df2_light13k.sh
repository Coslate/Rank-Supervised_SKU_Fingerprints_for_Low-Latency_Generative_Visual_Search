#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/data/patrick/hf_cache

# Optional: pick a GPU
GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
echo "[INFO] Using GPU: ${CUDA_VISIBLE_DEVICES}"

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$PROJECT_ROOT"

SKU_ROOT="${SKU_ROOT:-/data/patrick/10623GenAI/final_proj/data/DeepFashion2_SKU}"

echo "PROJECT_ROOT = ${PROJECT_ROOT}"
echo "SKU_ROOT     = ${SKU_ROOT}"
echo

# ---------- Pretrained SD augmentation (multi-view only) ----------
python -m gen.gen_sd_aug_df2 \
  --sku_root "$SKU_ROOT" \
  --split train \
  --num_views 4 \
  --num_counterfactual 0 \
  --max_skus 13000 \
  --shuffle_skus \
  --seed 16831 \
  --out_suffix sd_clipsku_sub13k_nv4 \
  --mv_subdir catalog_sd_light_sub13k_nv4 \
  --mv_suffix aug_light \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --clip_model ViT-B-16 \
  --clip_pretrained laion2b_s34b_b88k \
  --clip_sku_ckpt /data/patrick/10623GenAI/final_proj/checkpoints/baseline5_ft_bestshot_bs64_text0p1_unlock2_ep6_lr3e-4/clip_sku_baseline_final_best.pt \
  --device cuda