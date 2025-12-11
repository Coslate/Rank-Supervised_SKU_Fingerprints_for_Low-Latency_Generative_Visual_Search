#!/usr/bin/env bash
set -e

# Optional: pick a GPU (uncomment if needed)
GPU_ID="${1:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
echo "Using GPU: ${CUDA_VISIBLE_DEVICES}"

JSONL_ROOT=${JSONL_ROOT:-"./data"}
SKU_ROOT=${SKU_ROOT:-"/data/patrick/10623GenAI/final_proj/data/DeepFashion2_SKU"}
OUT_DIR=${OUT_DIR:-"/data/patrick/10623GenAI/final_proj/checkpoints/baseline1_reid_run3"}

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
  --wandb_project 10623finalproj_df2_reid_baseline1 \
  --wandb_run_name "baseline1_reid_bs64_e40_run3"  