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

# ----------------------------------------------------------
# Resume SD-based multiview augmentation from existing images
# ----------------------------------------------------------
python -m gen.gen_sd_aug_df2_resume \
  --sku_root "$SKU_ROOT" \
  --split train \
  --num_views 4 \
  --num_counterfactual 0 \
  --mv_subdir catalog_sd \
  --mv_suffix aug \
  --out_suffix sd_pretrained_aug \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --clip_model ViT-B-16 \
  --clip_pretrained laion2b_s34b_b88k \
  --device cuda \
  --resume_from_disk
  # fine-tuned CLIP-SKU
  # --clip_sku_ckpt /data/patrick/10623GenAI/final_proj/checkpoints/baseline5_ft_bestshot_bs64_text0p1_unlock2_ep6_lr3e-4/clip_sku_baseline_final_best.pt
