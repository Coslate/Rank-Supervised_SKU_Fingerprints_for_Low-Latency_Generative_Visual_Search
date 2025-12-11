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

# Teacher npz from Step 2 (best-shot teacher + sku_ids)
TEACHER_NPZ="${TEACHER_NPZ:-$SKU_ROOT/bestshot_teachers/baseline5_ft_bestshot_bs64_text0p1_unlock2_ep6_lr3e-4/bestshot_teachers_with_text_train.npz}"

# Fine-tuned CLIP-SKU checkpoint (baseline5)
CLIP_SKU_CKPT="${CLIP_SKU_CKPT:-/data/patrick/10623GenAI/final_proj/checkpoints/baseline5_ft_bestshot_bs64_text0p1_unlock2_ep6_lr3e-4/clip_sku_baseline_final_best.pt}"

# Train jsonl including DiT multi-view images
TRAIN_IMAGE_TEXT="${TRAIN_IMAGE_TEXT:-$SKU_ROOT/train_image_text.sd_clipsku_sub5p5k_nv4.jsonl}"
#TRAIN_IMAGE_TEXT="${TRAIN_IMAGE_TEXT:-$SKU_ROOT/train_image_text.sd_clipsku_sub3k_nv4.jsonl}"

# Output path for student checkpoint
STUDENT_CKPT_OUT="${STUDENT_CKPT_OUT:-$PROJECT_ROOT/checkpoints/sku_fingerprint_student/sku_fingerprint_student_baseline5_ft_bestshot_bs64_text0p1_unlock2_ep6_lr3e-4_e60_both_txtimg_sd_sub5p5k_nv4.pt}"
#STUDENT_CKPT_OUT="${STUDENT_CKPT_OUT:-$PROJECT_ROOT/checkpoints/sku_fingerprint_student/sku_fingerprint_student_baseline5_ft_bestshot_bs64_text0p1_unlock2_ep6_lr3e-4_e100_both_txtimg_sd_sub5p5k_nv4.pt}"
#STUDENT_CKPT_OUT="${STUDENT_CKPT_OUT:-$PROJECT_ROOT/checkpoints/sku_fingerprint_student/sku_fingerprint_student_baseline5_ft_bestshot_bs64_text0p1_unlock2_ep6_lr3e-4_e200_both_txtimg_sd_sub5p5k_nv4.pt}"

echo "PROJECT_ROOT       = ${PROJECT_ROOT}"
echo "SKU_ROOT           = ${SKU_ROOT}"
echo "TEACHER_NPZ        = ${TEACHER_NPZ}"
echo "CLIP_SKU_CKPT      = ${CLIP_SKU_CKPT}"
echo "TRAIN_IMAGE_TEXT   = ${TRAIN_IMAGE_TEXT}"
echo "STUDENT_CKPT_OUT   = ${STUDENT_CKPT_OUT}"
echo

python -m train.train_sku_fingerprint_distill \
  --sku_root "$SKU_ROOT" \
  --train_image_text "$TRAIN_IMAGE_TEXT" \
  --teacher_npz "$TEACHER_NPZ" \
  --clip_model ViT-B-16 \
  --clip_pretrained laion2b_s34b_b88k \
  --clip_sku_ckpt "$CLIP_SKU_CKPT" \
  --batch_size 64 \
  --precompute_sku_batch_size 256 \
  --text_query_prob 0.5 \
  --epochs 60 \
  --lr 4e-4 \
  --min_lr 1e-6 \
  --weight_decay 1e-2 \
  --warmup_ratio 0.05 \
  --hidden_dim 512 \
  --num_layers 2 \
  --num_heads 8 \
  --device cuda \
  --num_workers 8 \
  --checkpoint_out "$STUDENT_CKPT_OUT" \
  --wandb_project 10623_SKU_Fingerprint \
  --wandb_run_name "student_baseline5_sd_sub5p5k_e60_both_txtimg"
