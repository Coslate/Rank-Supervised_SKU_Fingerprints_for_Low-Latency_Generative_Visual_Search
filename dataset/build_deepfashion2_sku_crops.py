# dataset/build_deepfashion2_sku_crops.py

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x


def clamp_bbox(bbox, w, h):
    """Clamp DeepFashion2 bbox [x1, y1, x2, y2] into image range."""
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return [int(x1), int(y1), int(x2), int(y2)]


def process_split(split, df2_root: Path, out_root: Path, only_pairs: bool):
    """
    Convert one DeepFashion2 split into SKU-level crops and metadata.

    Visual SKU is defined as (pair_id, style > 0, category_id).
    Items with style == 0 are discarded.
    Crops with source == "shop" are saved under "catalog/",
    and crops with source == "user" are saved under "query/".
    """
    img_dir = df2_root / split / "image"
    anno_dir = df2_root / split / "annos"
    out_split_root = out_root / split
    out_split_root.mkdir(parents=True, exist_ok=True)

    # sku_meta[sku_id] structure:
    # {
    #   "pair_id": int,
    #   "style": int,
    #   "category_id": int,
    #   "category_name": str,
    #   "catalog": [ {crop_entry}, ... ],
    #   "query":   [ {crop_entry}, ... ],
    # }
    #
    # crop_entry structure:
    # {
    #   "crop_path": str (relative to out_root),
    #   "orig_image_path": str (relative to df2_root),
    #   "image_id": str,
    #   "item_idx": int,
    #   "bbox": [x1, y1, x2, y2],
    #   "occlusion": int,   # 1 = slight/none, 2 = medium, 3 = heavy
    #   "viewpoint": int,   # 1 = no wear, 2 = frontal, 3 = side/back
    # }
    sku_meta = defaultdict(
        lambda: {
            "pair_id": None,
            "style": None,
            "category_id": None,
            "category_name": None,
            "catalog": [],
            "query": [],
        }
    )

    anno_files = sorted(anno_dir.glob("*.json"))
    print(f"[{split}] found {len(anno_files)} annotation files")

    for anno_path in tqdm(anno_files, desc=f"{split} annos"):
        image_id = anno_path.stem
        img_path = img_dir / f"{image_id}.jpg"
        if not img_path.exists():
            continue

        with open(anno_path, "r") as f:
            ann = json.load(f)

        source = ann["source"]          # "shop" or "user"
        pair_id = int(ann["pair_id"])

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        items = [(k, v) for k, v in ann.items() if k.startswith("item")]
        if not items:
            img.close()
            continue

        for item_key, item in items:
            style = int(item["style"])
            if style <= 0:
                # Cannot form positive commercial-consumer pairs; skip.
                continue

            category_id = int(item["category_id"])
            category_name = item["category_name"]
            bbox = clamp_bbox(item["bounding_box"], w, h)
            occlusion = int(item.get("occlusion", 1))
            viewpoint = int(item.get("viewpoint", 1))

            # Visual SKU id used as folder name.
            sku_id = f"{pair_id:06d}_{style:02d}_{category_id:02d}"

            x1, y1, x2, y2 = bbox
            crop = img.crop((x1, y1, x2, y2))

            domain = "catalog" if source == "shop" else "query"
            dst_dir = out_split_root / domain / sku_id
            dst_dir.mkdir(parents=True, exist_ok=True)

            item_idx = int(item_key[len("item"):])  # "item3" -> 3
            dst_name = f"{image_id}_item{item_idx}.jpg"
            dst_path = dst_dir / dst_name
            crop.save(dst_path, quality=95)

            crop_rel = dst_path.relative_to(out_root)
            orig_rel = img_path.relative_to(df2_root)

            crop_entry = {
                "crop_path": str(crop_rel),
                "orig_image_path": str(orig_rel),
                "image_id": image_id,
                "item_idx": item_idx,
                "bbox": bbox,
                "occlusion": occlusion,
                "viewpoint": viewpoint,
            }

            meta = sku_meta[sku_id]
            if meta["pair_id"] is None:
                meta["pair_id"] = pair_id
                meta["style"] = style
                meta["category_id"] = category_id
                meta["category_name"] = category_name
            meta[domain].append(crop_entry)

        img.close()

    # Optionally keep only SKUs that have both catalog and query crops.
    if only_pairs:
        filtered = {}
        for sku_id, info in sku_meta.items():
            if info["catalog"] and info["query"]:
                filtered[sku_id] = info
        sku_meta = filtered

    sku_meta = dict(sku_meta)
    out_json = out_root / f"{split}_sku_metadata.json"
    with open(out_json, "w") as f:
        json.dump(
            {
                "split": split,
                "df2_root": str(df2_root),
                "out_root": str(out_root),
                "num_skus": len(sku_meta),
                "skus": sku_meta,
            },
            f,
            indent=2,
        )

    print(f"[{split}] wrote {len(sku_meta)} SKUs to {out_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Build DeepFashion2 SKU-level crops and metadata."
    )
    parser.add_argument(
        "--df2_root",
        type=Path,
        default=Path("data/DeepFashion2_original"),
        help="Root directory of original DeepFashion2 dataset.",
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path("data/DeepFashion2_SKU"),
        help="Output root directory for SKU crops.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        help="Splits to process (default: train validation test).",
    )
    parser.add_argument(
        "--only_pairs",
        action="store_true",
        help="Keep only SKUs with both catalog and query images.",
    )

    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        process_split(split, args.df2_root, args.out_root, args.only_pairs)


if __name__ == "__main__":
    main()
