# Prepare Dataset for all
./scripts/prepare_deepfashion2_sku.sh

# Prepare Dataset for Baseline1
./scripts/prepare_df2_reid_splits.sh

# Train Baseline1
./scripts/train_baseline1_reid.sh

# Eval Baseline1
./scripts/eval_baseline1_reid_val.sh

# Train Baseline5 only
./scripts/train_baseline.only4.sh 5

# Train Baseline2/5
./scripts/train_clip_sku_df2.sh

# Eval Baseline2/5
./scripts/eval_clip_sku_df2.sh

# Train LoRA on SD v1.5 Img2Img
./scripts/train_sd_lora_df2.sh 2
./scripts/train_sd_lora_df2_ep1.sh 2

# SD v1.5 Img2Img Generation Augmented Multi-view
# ./scripts/gen_sd_aug_df2.sh
./scripts/gen_sd_aug_df2_resume.sh 1
./scripts/gen_sd_aug_df2_light13k.sh 0
./scripts/gen_sd_aug_df2_light5p5k.sh 3
./scripts/gen_sd_aug_df2_light3k.sh 3

# ./scripts/gen_dit_aug_df2.sh 2
./scripts/gen_sd_aug_df2_light5p5k_sdloraftep1.sh 6
./scripts/gen_sd_aug_df2_light5p5k_sdloraftep3.sh 2
./scripts/gen_sd_aug_df2_light13k_sdloraftep1.sh 5
./scripts/gen_sd_aug_df2_light13k_sdloraftep3.sh 7


# Generate Teacher Embedding for Fingerprint Distillation
./scripts/gen_bestshot_teachers_df2.sh 2

# Train Student for generating Fingerprint
./scripts/train_sku_fingerprint_student.sh 2
./scripts/train_sku_fingerprint_student_multiview3k.sh 2
./scripts/train_sku_fingerprint_student_multiview5p5k.sh 2

./scripts/train_sku_fingerprint_student_multiview5p5k_sd1ep.sh 2
./scripts/train_sku_fingerprint_student_multiview5p5k_sd3ep.sh 2

# Eval using distilled SKU fingerprint
./scripts/eval_sku_fingerprint_student.sh 2

# Demo
./scripts/eval_sku_fingerprint_student_demo.sh 5

# Rename catalog folder in json file
# python ./scripts/replace_catalog_prefix.py \
#   --input  /data/patrick/10623GenAI/final_proj/data/DeepFashion2_SKU/train_image_text.sd_clipsku_sub3k_nv4.jsonl \
#   --output /data/patrick/10623GenAI/final_proj/data/DeepFashion2_SKU/train_image_text.sd_clipsku_sub3k_nv4.new.jsonl

# python ./scripts/replace_catalog_prefix.py \
#  --input  /data/patrick/10623GenAI/final_proj/data/DeepFashion2_SKU/train_image_text.sd_clipsku_sub5p5k_nv4.jsonl \
#  --old_prefix "train/catalog_dit_light_sub5p5k_nv4" \
#  --new_prefix "train/catalog_sd_light_sub5p5k_nv4" \
#  --output /data/patrick/10623GenAI/final_proj/data/DeepFashion2_SKU/train_image_text.sd_clipsku_sub5p5k_nv4.new.jsonl

# python ./scripts/replace_catalog_crop_path_prefix.py \
  # --input  /data/patrick/10623GenAI/final_proj/data/DeepFashion2_SKU/train_sku_metadata.sd_clipsku_sub3k_nv4.json \
  # --output /data/patrick/10623GenAI/final_proj/data/DeepFashion2_SKU/train_sku_metadata.sd_clipsku_sub3k_nv4.new.json \
  # --old_prefix train/catalog_dit_light_sub3k_nv4 \
  # --new_prefix train/catalog_sd_light_sub3k_nv4

# python ./scripts/replace_catalog_crop_path_prefix.py \
  # --input  /data/patrick/10623GenAI/final_proj/data/DeepFashion2_SKU/train_sku_metadata.sd_clipsku_sub5p5k_nv4.json \
  # --output /data/patrick/10623GenAI/final_proj/data/DeepFashion2_SKU/train_sku_metadata.sd_clipsku_sub5p5k_nv4.new.json \
  # --old_prefix train/catalog_dit_light_sub5p5k_nv4 \
  # --new_prefix train/catalog_sd_light_sub5p5k_nv4