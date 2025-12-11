#!/usr/bin/env python
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(
        description="Replace image_path prefix from catalog_dit_light_sub3k_nv4 to catalog_sd_light_sub3k_nv4 in a JSONL file."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--old_prefix",
        type=str,
        default="train/catalog_dit_light_sub3k_nv4",
        help="Old image_path prefix to replace.",
    )
    parser.add_argument(
        "--new_prefix",
        type=str,
        default="train/catalog_sd_light_sub3k_nv4",
        help="New image_path prefix.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    num_total = 0
    num_modified = 0

    with open(args.input, "r") as fin, open(args.output, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            num_total += 1
            rec = json.loads(line)

            img_path = rec.get("image_path", "")
            if isinstance(img_path, str) and img_path.startswith(args.old_prefix):
                rec["image_path"] = img_path.replace(args.old_prefix, args.new_prefix, 1)
                num_modified += 1

            fout.write(json.dumps(rec) + "\n")

    print(f"[DONE] Processed {num_total} lines, modified {num_modified} image_path entries.")
    print(f"[OUT]  Written to {args.output}")


if __name__ == "__main__":
    main()
