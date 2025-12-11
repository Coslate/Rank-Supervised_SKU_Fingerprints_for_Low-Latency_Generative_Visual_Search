#!/usr/bin/env bash
set -e

# Root where DeepFashion1_SKU crops and *_sku_metadata.json live
SKU_ROOT=${SKU_ROOT:-"/data/patrick/10622GenAI/final_proj/data/DeepFashion2_SKU"}
OUT_DIR=${OUT_DIR:-"./data"}

echo "Building DF1 re-id JSONL splits under ${OUT_DIR} using SKU_ROOT=${SKU_ROOT}"

python data/build_df2_reid_splits.py \
  --sku_root "${SKU_ROOT}" \
  --out_dir "${OUT_DIR}"