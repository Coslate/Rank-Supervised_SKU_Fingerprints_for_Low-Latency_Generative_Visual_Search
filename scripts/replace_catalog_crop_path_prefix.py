#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replace crop_path prefix in DeepFashion2_SKU sku_metadata JSON.\n"
            "Example: train/catalog_dit_light_sub3k_nv4 -> train/catalog_sd_light_sub3k_nv4"
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input sku_metadata JSON file (e.g., train_sku_metadata.sd_clipsku_sub3k_nv4.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file path.",
    )
    parser.add_argument(
        "--old_prefix",
        type=str,
        default="train/catalog_dit_light_sub3k_nv4",
        help="Old crop_path prefix to replace.",
    )
    parser.add_argument(
        "--new_prefix",
        type=str,
        default="train/catalog_sd_light_sub3k_nv4",
        help="New crop_path prefix.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[INFO] Loading metadata from {args.input}")
    with args.input.open("r") as f:
        data = json.load(f)

    skus = data.get("skus", {})
    num_entries = 0
    num_modified = 0

    for sku_id, sku_info in skus.items():
        catalog = sku_info.get("catalog", [])
        for entry in catalog:
            if not isinstance(entry, dict):
                continue
            crop_path = entry.get("crop_path")
            if isinstance(crop_path, str):
                num_entries += 1
                if crop_path.startswith(args.old_prefix):
                    new_path = crop_path.replace(args.old_prefix, args.new_prefix, 1)
                    entry["crop_path"] = new_path
                    num_modified += 1

    print(f"[STATS] Catalog entries inspected: {num_entries}")
    print(f"[STATS] crop_path modified:       {num_modified}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"[DONE] Written updated metadata to {args.output}")


if __name__ == "__main__":
    main()
