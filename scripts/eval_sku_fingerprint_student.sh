#!/usr/bin/env bash
set -euo pipefail

# Optional: pick a GPU via first CLI arg
GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
echo "[INFO] Using GPU: ${CUDA_VISIBLE_DEVICES}"

# Resolve project root (repo root = parent of this script directory)
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$PROJECT_ROOT"

# Paths (adapt to your environment if needed)
SKU_ROOT="${SKU_ROOT:-/data/patrick/10623GenAI/final_proj/data/DeepFashion2_SKU}"

# Baseline5 CLIP-SKU checkpoint (same one used in Step 2/3)
CLIP_SKU_CKPT="${CLIP_SKU_CKPT:-/data/patrick/10623GenAI/final_proj/checkpoints/baseline5_ft_bestshot_bs64_text0p1_unlock2_ep6_lr3e-4/clip_sku_baseline_final_best.pt}"

# (1) Student without SD multi-view
#STUDENT_CKPT="${STUDENT_CKPT:-$PROJECT_ROOT/checkpoints/sku_fingerprint_student/sku_fingerprint_student_baseline5_ft_bestshot_bs64_text0p1_unlock2_ep6_lr3e-4_e20_both_txtimg.pt}"
#STUDENT_CKPT="${STUDENT_CKPT:-$PROJECT_ROOT/checkpoints/sku_fingerprint_student/sku_fingerprint_student_baseline5_ft_bestshot_bs64_text0p1_unlock2_ep6_lr3e-4_e40_both_txtimg.pt}"
#STUDENT_CKPT="${STUDENT_CKPT:-$PROJECT_ROOT/checkpoints/sku_fingerprint_student/sku_fingerprint_student_baseline5_ft_bestshot_bs64_text0p1_unlock2_ep6_lr3e-4_e60_both_txtimg.pt}"
#STUDENT_CKPT="${STUDENT_CKPT:-$PROJECT_ROOT/checkpoints/sku_fingerprint_student/sku_fingerprint_student_baseline5_ft_bestshot_bs64_text0p1_unlock2_ep6_lr3e-4_e100_both_txtimg.pt}"
#STUDENT_CKPT="${STUDENT_CKPT:-$PROJECT_ROOT/checkpoints/sku_fingerprint_student/sku_fingerprint_student_baseline5_ft_bestshot_bs64_text0p1_unlock2_ep6_lr3e-4_e200_both_txtimg.pt}"

# (2) Student with SD multi-view
#STUDENT_CKPT="${STUDENT_CKPT:-$PROJECT_ROOT/checkpoints/sku_fingerprint_student/sku_fingerprint_student_baseline5_ft_bestshot_bs64_text0p1_unlock2_ep6_lr3e-4_e20_both_txtimg_sd_sub3k_nv4.pt}"
#STUDENT_CKPT="${STUDENT_CKPT:-$PROJECT_ROOT/checkpoints/sku_fingerprint_student/sku_fingerprint_student_baseline5_ft_bestshot_bs64_text0p1_unlock2_ep6_lr3e-4_e40_both_txtimg_sd_sub3k_nv4.pt}"
#STUDENT_CKPT="${STUDENT_CKPT:-$PROJECT_ROOT/checkpoints/sku_fingerprint_student/sku_fingerprint_student_baseline5_ft_bestshot_bs64_text0p1_unlock2_ep6_lr3e-4_e60_both_txtimg_sd_sub3k_nv4.pt}"
#STUDENT_CKPT="${STUDENT_CKPT:-$PROJECT_ROOT/checkpoints/sku_fingerprint_student/sku_fingerprint_student_baseline5_ft_bestshot_bs64_text0p1_unlock2_ep6_lr3e-4_e100_both_txtimg_sd_sub3k_nv4.pt}"
#STUDENT_CKPT="${STUDENT_CKPT:-$PROJECT_ROOT/checkpoints/sku_fingerprint_student/sku_fingerprint_student_baseline5_ft_bestshot_bs64_text0p1_unlock2_ep6_lr3e-4_e200_both_txtimg_sd_sub3k_nv4.pt}"

#STUDENT_CKPT="${STUDENT_CKPT:-$PROJECT_ROOT/checkpoints/sku_fingerprint_student/sku_fingerprint_student_baseline5_ft_bestshot_bs64_text0p1_unlock2_ep6_lr3e-4_e20_both_txtimg_sd_sub5p5k_nv4.pt}"
#STUDENT_CKPT="${STUDENT_CKPT:-$PROJECT_ROOT/checkpoints/sku_fingerprint_student/sku_fingerprint_student_baseline5_ft_bestshot_bs64_text0p1_unlock2_ep6_lr3e-4_e60_both_txtimg_sd_sub5p5k_nv4.pt}"
#STUDENT_CKPT="${STUDENT_CKPT:-$PROJECT_ROOT/checkpoints/sku_fingerprint_student/sku_fingerprint_student_baseline5_ft_bestshot_bs64_text0p1_unlock2_ep6_lr3e-4_e100_both_txtimg_sd_sub5p5k_nv4.pt}"
#STUDENT_CKPT="${STUDENT_CKPT:-$PROJECT_ROOT/checkpoints/sku_fingerprint_student/sku_fingerprint_student_baseline5_ft_bestshot_bs64_text0p1_unlock2_ep6_lr3e-4_e200_both_txtimg_sd_sub5p5k_nv4.pt}"

# (3) LoRA fineuning
#STUDENT_CKPT="${STUDENT_CKPT:-$PROJECT_ROOT/checkpoints/sku_fingerprint_student/sku_fingerprint_student_baseline5_ft_bestshot_bs64_text0p1_unlock2_ep6_lr3e-4_e20_both_txtimg_sd_ft1ep_sub5p5k_nv4.pt}"
STUDENT_CKPT="${STUDENT_CKPT:-$PROJECT_ROOT/checkpoints/sku_fingerprint_student/sku_fingerprint_student_baseline5_ft_bestshot_bs64_text0p1_unlock2_ep6_lr3e-4_e20_both_txtimg_sd_ft3ep_sub5p5k_nv4.pt}"

# Validation split image_text JSONL (usually no multiview for val)
VAL_IMAGE_TEXT="${VAL_IMAGE_TEXT:-$SKU_ROOT/validation_image_text.jsonl}"
# If you ever have validation multiview, change to the dit_* file, e.g.:
#VAL_IMAGE_TEXT="${VAL_IMAGE_TEXT:-$SKU_ROOT/validation_image_text.dit_clipsku_sub3k_nv4.jsonl}"

echo "PROJECT_ROOT   = ${PROJECT_ROOT}"
echo "SKU_ROOT       = ${SKU_ROOT}"
echo "CLIP_SKU_CKPT  = ${CLIP_SKU_CKPT}"
echo "STUDENT_CKPT   = ${STUDENT_CKPT}"
echo "VAL_IMAGE_TEXT = ${VAL_IMAGE_TEXT}"
echo

python -m eval.eval_sku_fingerprint_student \
  --sku_root "$SKU_ROOT" \
  --val_image_text "$VAL_IMAGE_TEXT" \
  --clip_model ViT-B-16 \
  --clip_pretrained laion2b_s34b_b88k \
  --clip_sku_ckpt "$CLIP_SKU_CKPT" \
  --student_ckpt "$STUDENT_CKPT" \
  --batch_size 256 \
  --precompute_sku_batch_size 256 \
  --num_workers 8 \
  --device cuda \
  --ndcg_k 10 \
  --recall_ks 1 5 10 \
  --eval_cpu_latency \
  --eval_all_val_skus \
  --wandb_project 10623_SKU_Fingerprint \
  --wandb_run_name "eval_distilled_skufingerprint_e20_sd_sub5p5k_nv4_allvalskus_vla"
  #--eval_cpu_latency \
  #--eval_all_val_skus \
  #--use_vla \
  #--vla_checkpoint $PROJECT_ROOT/VLA_model/checkpoint_vla/vla_policy.pt \
  #--wandb_run_name "eval_distilled_skufingerprint_e200"
  #--wandb_run_name "eval_distilled_skufingerprint_e20_sd_sub5p5k_nv4_allvalskus"
  #--wandb_run_name "eval_distilled_skufingerprint_e20"
  #--wandb_run_name "eval_distilled_skufingerprint_e100_sd_sub3k_nv4_allvalskus"
  #--wandb_run_name "eval_distilled_skufingerprint_e200_sd_sub3k_nv4"
  #--wandb_run_name "eval_distilled_skufingerprint_e200_sd_sub5p5k_nv4"
  #--wandb_run_name "eval_distilled_skufingerprint_e100_sd_sub5p5k_nv4"
