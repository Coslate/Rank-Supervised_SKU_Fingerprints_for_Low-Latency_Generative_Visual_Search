#!/usr/bin/env bash
set -euo pipefail

# Root of original DeepFashion2 (must contain train/validation/test with image/annos).
# You can override via environment variable:
#   DF2_ROOT=/your/path/to/DeepFashion2_original ./prepare_deepfashion2_sku.sh
DF2_ROOT=${DF2_ROOT:-"/data/patrick/10623GenAI/final_proj/data/DeepFashion2_original"}

# Output root for SKU crops and metadata.
# You can override via:
#   SKU_ROOT=/your/output/path ./prepare_deepfashion2_sku.sh
SKU_ROOT=${SKU_ROOT:-"/data/patrick/10623GenAI/final_proj/data/DeepFashion2_SKU"}

# Splits to process. Default: train validation test
# Override via:
#   SPLITS="train validation" ./prepare_deepfashion2_sku.sh
SPLITS=${SPLITS:-"train validation test"}

echo "=== DeepFashion2 SKU preparation ==="
echo "DF2_ROOT = ${DF2_ROOT}"
echo "SKU_ROOT = ${SKU_ROOT}"
echo "SPLITS   = ${SPLITS}"
echo

# 1) Build SKU-level crops and metadata (includes occlusion + viewpoint).
python dataset/build_deepfashion2_sku_crops.py \
  --df2_root "${DF2_ROOT}" \
  --out_root "${SKU_ROOT}" \
  --splits ${SPLITS} \
  --only_pairs

echo
echo "=== Done cropping. Now building image-text JSONL ==="
echo

# 2) Build image-text JSONL (prompts include occlusion + viewpoint).
python dataset/build_deepfashion2_text_prompts.py \
  --sku_root "${SKU_ROOT}" \
  --splits ${SPLITS} \
  --sku_token_dropout 0.5

echo
echo "=== All done. Outputs are under ${SKU_ROOT} ==="
