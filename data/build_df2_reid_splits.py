# data/build_df2_reid_splits.py

import argparse
import json
from pathlib import Path


def build_split(split: str, sku_root: Path, out_dir: Path) -> None:
    """
    Build a JSONL file listing all crops for this split, with a SKU label.

    Input:
        {split}_sku_metadata.json under sku_root (already created by your SKU scripts)

    Output:
        out_dir/df2_reid_{split}.jsonl
        Each line:
          {
            "split": "train" | "validation" | "test",
            "sku_id": str,
            "label": int,          # per-split SKU label (0 .. num_skus-1)
            "domain": "catalog" | "query",
            "crop_path": "train/catalog/....jpg"  # relative to sku_root
          }
    """
    meta_path = sku_root / f"{split}_sku_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    skus = meta["skus"]

    # Only keep SKUs with style > 0
    valid_sku_ids = [sku_id for sku_id, info in skus.items() if info["style"] > 0]
    valid_sku_ids = sorted(valid_sku_ids)

    sku_to_label = {sku_id: i for i, sku_id in enumerate(valid_sku_ids)}
    num_skus = len(valid_sku_ids)
    print(f"[{split}] num_skus (style>0): {num_skus}")

    out_path = out_dir / f"df2_reid_{split}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_samples = 0
    with open(out_path, "w") as f_out:
        for sku_id in valid_sku_ids:
            info = skus[sku_id]
            label = sku_to_label[sku_id]

            for domain in ["catalog", "query"]:
                for entry in info[domain]:
                    rec = {
                        "split": split,
                        "sku_id": sku_id,
                        "label": label,
                        "domain": domain,
                        "crop_path": entry["crop_path"],
                    }
                    f_out.write(json.dumps(rec) + "\n")
                    num_samples += 1

    print(f"[{split}] wrote {num_samples} samples to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build DF2 re-id splits (train/validation/test) JSONL files."
    )
    parser.add_argument(
        "--sku_root",
        type=Path,
        default=Path("data/DeepFashion2_SKU"),
        help="Root containing *_sku_metadata.json and crops.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data"),
        help="Directory to write df2_reid_{split}.jsonl files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        help="Splits to process.",
    )

    args = parser.parse_args()
    for split in args.splits:
        build_split(split, args.sku_root, args.out_dir)


if __name__ == "__main__":
    main()
