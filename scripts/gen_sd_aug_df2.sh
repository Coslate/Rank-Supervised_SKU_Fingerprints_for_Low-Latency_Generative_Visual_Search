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

# ---------- Pretrained DiT/SD augmentation (multi-view only) ----------
python -m gen.gen_sd_aug_df2 \
  --sku_root "$SKU_ROOT" \
  --split train \
  --num_views 4 \
  --num_counterfactual 0 \
  --out_suffix sd_pretrained_aug \
  --sd_model runwayml/stable-diffusion-v1-5 \
  --clip_model ViT-B-16 \
  --clip_pretrained laion2b_s34b_b88k \
  --device cuda  
  #--clip_sku_ckpt /data/patrick/10623GenAI/final_proj/checkpoints/baseline3_clip_sku_bs64/clip_sku_baseline_final.pt \

# if you want to fine-tuned DiT, generating finetuned version, run with：
# python -m gen.gen_dit_aug_df2 \
#   --sku_root "$SKU_ROOT" \
#   --split train \
#   --num_views 2 \
#   --num_counterfactual 0 \
#   --out_suffix dit_finetuned_aug \
#   --sd_model /path/to/your_finetuned_dit_or_sd \
#   --device cuda

