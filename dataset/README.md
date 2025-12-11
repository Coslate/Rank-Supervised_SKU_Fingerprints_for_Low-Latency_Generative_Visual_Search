# Rank-Supervised_SKU_Fingerprints_for_Low-Latency_Generative_Visual_Search

## DeepFashion2 → SKU Preprocessing Pipeline

This document explains how to run the preprocessing pipeline that converts the
original **DeepFashion2** dataset into a SKU-centric representation used by
our project:

1. Convert the original DeepFashion2 images into **SKU-level crops**.
2. Build **metadata per SKU** (including occlusion and viewpoint).
3. Build **image–text JSONL files** with prompts for training vision-language
   models (CLIP / Q-Former / DiT / etc.).

The pipeline is driven by two Python scripts plus one shell script.

---

### 1. Directory layout and prerequisites

#### 1.1 Original DeepFashion2 layout

You should first download and unzip the *original* DeepFashion2 dataset.
This pipeline expects the following structure (default root:
`data/DeepFashion2_original`):

```text
data/DeepFashion2_original/
  train/
    image/          # 000001.jpg, 000002.jpg, ...
    annos/          # 000001.json, 000002.json, ...
  validation/
    image/
    annos/
  test/
    image/
    annos/
```

If your dataset is stored somewhere else (for example:
`/data/patrick/10623GenAI/final_proj/data/DeepFashion2_original`), you can
override the path using the `DF2_ROOT` environment variable when running
the shell script (see Section 3).

#### 1.2 Scripts used in this pipeline

The repository should contain the following scripts:

```text
dataset/
  build_deepfashion2_sku_crops.py
  build_deepfashion2_text_prompts.py

scripts/
  prepare_deepfashion2_sku.sh
```

- `build_deepfashion2_sku_crops.py`  
  Reads the original DeepFashion2 annotations, defines **visual SKUs** as  
  `(pair_id, style > 0, category_id)`, crops items using their bounding boxes,
  and writes both cropped images and SKU-level metadata (including
  occlusion and viewpoint).

- `build_deepfashion2_text_prompts.py`  
  Reads the SKU-level metadata and produces a **flat JSONL** file of
  image–text samples with prompts that encode category, occlusion and
  viewpoint.

- `prepare_deepfashion2_sku.sh`  
  Convenience wrapper that calls both scripts in the correct order.

#### 1.3 Python dependencies

- `python >= 3.8`
- `Pillow`
- `tqdm` (optional; if missing, the scripts still run but without progress bars)

You can install them via:

```bash
pip install pillow tqdm
```

---

### 2. Shell script: one-command preprocessing

From the **project root**, make the script executable and run it:

```bash
chmod +x ./scripts/prepare_deepfashion2_sku.sh
./scripts/prepare_deepfashion2_sku.sh
```

By default the script assumes:

- `DF2_ROOT = data/DeepFashion2_original`
- `SKU_ROOT = data/DeepFashion2_SKU`
- `SPLITS   = "train validation test"`

You can override any of these via environment variables:

```bash
DF2_ROOT=/data/patrick/10623GenAI/final_proj/data/DeepFashion2_original SKU_ROOT=/data/patrick/10623GenAI/final_proj/data/DeepFashion2_SKU SPLITS="train validation" ./scripts/prepare_deepfashion2_sku.sh
```

Where:

- `DF2_ROOT` – root directory of the original DeepFashion2 dataset
- `SKU_ROOT` – output directory for SKU crops + metadata (+ JSONL)
- `SPLITS` – one or more splits to process (space-separated)

---

### 3. What the pipeline produces

All outputs are written under `${SKU_ROOT}`
(default: `data/DeepFashion2_SKU`).

#### 3.1 Cropped images (SKU-level)

For each split (`train`, `validation`, `test`) and for each **visual SKU**

\`\`\`
SKU = (pair_id, style > 0, category_id)
\`\`\`

the script crops clothing items from the original images using the
bounding box for each item. The cropped images are saved in:

```text
${SKU_ROOT}/
  train/
    catalog/<sku_id>/*.jpg    # crops from source = "shop"
    query/<sku_id>/*.jpg      # crops from source = "user"
  validation/
    catalog/<sku_id>/*.jpg
    query/<sku_id>/*.jpg
  test/
    catalog/<sku_id>/*.jpg
    query/<sku_id>/*.jpg
```

Here:

- `sku_id` is a deterministic string of the form
  `"{pair_id:06d}_{style:02d}_{category_id:02d}"`.
- Only items with `style > 0` are kept.
- Because `prepare_deepfashion2_sku.sh` uses `--only_pairs`, only SKUs that
  have **both catalog and query** crops are retained.

#### 3.2 SKU-level metadata (`*_sku_metadata.json`)

For each split, the cropping script also writes a JSON file describing
every SKU and all of its associated crops:

```text
${SKU_ROOT}/train_sku_metadata.json
${SKU_ROOT}/validation_sku_metadata.json
${SKU_ROOT}/test_sku_metadata.json
```

Each file has the following high-level structure (simplified example):

```jsonc
{
  "split": "train",
  "df2_root": "/abs/path/to/DeepFashion2_original",
  "out_root": "data/DeepFashion2_SKU",
  "num_skus": 12345,
  "skus": {
    "010001_01_01": {
      "pair_id": 1,
      "style": 1,
      "category_id": 1,
      "category_name": "short sleeve top",
      "catalog": [
        {
          "crop_path": "train/catalog/010001_01_01/000002_item1.jpg",
          "orig_image_path": "train/image/000002.jpg",
          "image_id": "000002",
          "item_idx": 1,
          "bbox": [x1, y1, x2, y2],
          "occlusion": 2,
          "viewpoint": 2
        }
      ],
      "query": [
        {
          "crop_path": "train/query/010001_01_01/000001_item1.jpg",
          "orig_image_path": "train/image/000001.jpg",
          "image_id": "000001",
          "item_idx": 1,
          "bbox": [x1, y1, x2, y2],
          "occlusion": 3,
          "viewpoint": 3
        }
      ]
    }
  }
}
```

This file answers questions such as:

- “Which catalog and query crops belong to each SKU?”
- “Where are the cropped images stored?”
- “Which original DeepFashion2 image and item does each crop come from?”
- “What are the occlusion and viewpoint labels for each crop?”

Fields:

- `pair_id`, `style`, `category_id`, `category_name` – SKU-level attributes.
- `catalog` / `query` – lists of crop entries from shop/user images.
- Each crop entry contains:
  - `crop_path` – path to the cropped image, relative to `${SKU_ROOT}`
  - `orig_image_path` – path to the original image, relative to `${DF2_ROOT}`
  - `image_id` – six-digit DeepFashion2 image id (string)
  - `item_idx` – item index within that image (1, 2, ...)
  - `bbox` – `[x1, y1, x2, y2]` in original image coordinates
  - `occlusion` – 1: slight/none, 2: medium, 3: heavy
  - `viewpoint` – 1: no wear, 2: frontal, 3: side/back

This metadata is useful when you want to reason at the SKU level, build
query–catalog pairs or triplets, or inspect occlusion/viewpoint
statistics.

#### 3.3 Image–text JSONL (`*_image_text.jsonl`)

For training vision-language models, it is convenient to have a flat
sample list where each entry corresponds to a single (image, text)
pair. The text-prompt script reads the SKU-level metadata and writes:

```text
${SKU_ROOT}/train_image_text.jsonl
${SKU_ROOT}/validation_image_text.jsonl
${SKU_ROOT}/test_image_text.jsonl
```

Each line is a JSON object like:

```json
{
  "split": "train",
  "sku_id": "010001_01_01",
  "pair_id": 1,
  "style": 1,
  "category_id": 1,
  "category_name": "short sleeve top",
  "domain": "query",
  "image_path": "train/query/010001_01_01/000001_item1.jpg",
  "orig_image_path": "train/image/000001.jpg",
  "image_id": "000001",
  "item_idx": 1,
  "bbox": [x1, y1, x2, y2],
  "occlusion": 3,
  "viewpoint": 3,
  "text": "A user photo of a person wearing a heavily occluded short sleeve top from the side or back."
}
```

Notes:

- `image_path` is relative to `${SKU_ROOT}` and points to the cropped
  image that should be fed into the model.
- `orig_image_path` is relative to `${DF2_ROOT}` and can be used if you
  ever need the full original image.
- `domain` indicates whether the sample comes from a catalog (shop) image
  or a query (user) image.
- The `text` prompt automatically reflects:
  - the clothing **category** (`category_name`),
  - the **occlusion** level, and
  - the **viewpoint** (no wear / frontal / side-or-back).

This JSONL file is usually what you load directly in your Dataset
implementation for training.

---

### 4. Example: PyTorch Dataset using the JSONL

A very small sketch of how you might use the JSONL file in PyTorch:

```python
import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class DeepFashion2ImageTextDataset(Dataset):
    def __init__(self, sku_root, split, transform=None, text_key="text"):
        self.sku_root = Path(sku_root)
        self.transform = transform
        self.text_key = text_key

        jsonl_path = self.sku_root / f"{split}_image_text.jsonl"
        self.samples = []
        with open(jsonl_path, "r") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        img_path = self.sku_root / rec["image_path"]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        text = rec[self.text_key]
        sku_id = rec["sku_id"]
        domain = rec["domain"]  # "catalog" or "query"

        return {
            "image": img,
            "text": text,
            "sku_id": sku_id,
            "domain": domain,
            "meta": rec,
        }
```

You can then plug this dataset into any DataLoader and train your
vision-language model on DeepFashion2 in its SKU-centric form.

