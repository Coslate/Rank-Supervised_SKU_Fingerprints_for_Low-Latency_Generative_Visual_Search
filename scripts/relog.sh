#!/usr/bin/env bash
set -e

python scripts/relog_wandb_runs.py \
  --src_runs \
    bochunc-carnegie-mellon-university/10623finalproj_df2_reid_baseline1/runs/1i3b7x04 \
  --dst_entity bochunc-carnegie-mellon-university \
  --dst_project 10623finalproj_df2_all_baselines \
  --run_name_prefix "relog_"