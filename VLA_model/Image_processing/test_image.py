import argparse
import sys
from pathlib import Path
from PIL import Image

from image_process import ACTION_FUNCS, VLAAction


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test VLA image-processing actions."
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input image."
    )

    parser.add_argument(
        "--action",
        type=str,
        required=True,
        choices=[a.name for a in VLAAction],
        help="Action to apply. Options: " + ", ".join([a.name for a in VLAAction])
    )

    parser.add_argument(
        "--output",
        type=str,
        default="output.jpg",
        help="Where to save the processed image."
    )

    return parser.parse_args()


def main():
    args = parse_args()
    # Load image
    img_path = Path(args.input)
    if not img_path.exists():
        raise FileNotFoundError(f"Input image does not exist: {img_path}")

    img = Image.open(img_path).convert("RGB")

    # Get action enum
    action_enum = VLAAction[args.action]

    # Get corresponding function
    func = ACTION_FUNCS[action_enum]

    print(f"Applying action: {action_enum.name}")

    # Apply transform
    result = func(img)

    # Save output
    out_path = Path(args.output)
    result.save(out_path)
    print(f"Saved output to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
