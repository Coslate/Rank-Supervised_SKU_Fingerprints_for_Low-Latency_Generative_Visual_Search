# dataset/df2_clip_sku_dataset.py

from __future__ import annotations
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset


class DeepFashion2ImageTextSkuTrainDataset(Dataset):
    """
    Training dataset for CLIP/SigLIP SKU baseline.

    Reads {split}_image_text.jsonl records like:
      {
        "sku_id": "...",
        "domain": "catalog" or "query",
        "image_path": "train/catalog/....jpg",
        "text": "A catalog product photo of ..."
      }
    """

    def __init__(
        self,
        sku_root: Path | str,
        jsonl_path: Path | str,
        preprocess: Callable,
        tokenizer: Callable,
        sku2idx: Dict[str, int],
        domain_filter: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        """
        Args:
            sku_root: Root directory for DeepFashion2_SKU crops.
            jsonl_path: Path to {split}_image_text.jsonl.
            preprocess: Image transform from open_clip.create_model_and_transforms.
            tokenizer: Text tokenizer from open_clip.get_tokenizer.
            sku2idx: Mapping from sku_id string to integer index [0, num_skus).
            domain_filter: If "catalog" or "query", keep only that domain. If None, use both.
            max_samples: Optional cap on number of samples (for debugging).
        """
        self.sku_root = Path(sku_root)
        self.jsonl_path = Path(jsonl_path)
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.sku2idx = sku2idx
        self.domain_filter = domain_filter

        self.samples: List[Dict] = []

        with open(self.jsonl_path, "r") as f:
            for line in f:
                rec = json.loads(line)
                domain = rec["domain"]
                if self.domain_filter is not None and domain != self.domain_filter:
                    continue

                sku_id = rec["sku_id"]
                if sku_id not in self.sku2idx:
                    continue
                sku_idx = self.sku2idx[sku_id]

                image_rel = rec["image_path"]
                text = rec["text"]

                self.samples.append(
                    {
                        "image_path": self.sku_root / image_rel,
                        "text": text,
                        "sku_idx": sku_idx,
                        "sku_id": sku_id,
                        "domain": domain,
                    }
                )
                if max_samples is not None and len(self.samples) >= max_samples:
                    break

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples loaded from {self.jsonl_path}")

        print(
            f"[TrainDataset] Loaded {len(self.samples)} samples from {self.jsonl_path} "
            f"(domain_filter={self.domain_filter})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        rec = self.samples[idx]
        img_path = rec["image_path"]
        text = rec["text"]
        sku_idx = rec["sku_idx"]
        domain = rec["domain"]

        img = Image.open(img_path).convert("RGB")
        img_tensor = self.preprocess(img)

        tokens = self.tokenizer(text)
        if tokens.ndim == 2:
            tokens = tokens[0]

        domain_id = 0 if domain == "catalog" else 1

        return (
            img_tensor,
            tokens.long(),
            torch.tensor(sku_idx, dtype=torch.long),
            torch.tensor(domain_id, dtype=torch.long),
        )


class DeepFashion2ImageSkuEvalDataset(Dataset):
    """
    Evaluation dataset for CLIP/SigLIP SKU baseline.

    Only returns image + sku_idx, compatible with the ReID-style compute_embeddings:

        imgs, labels, sku_ids, dummy

    so that we can reuse a similar pipeline.
    """

    def __init__(
        self,
        sku_root: Path | str,
        jsonl_path: Path | str,
        preprocess: Callable,
        sku2idx: Dict[str, int],
        domain_filter: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        self.sku_root = Path(sku_root)
        self.jsonl_path = Path(jsonl_path)
        self.preprocess = preprocess
        self.sku2idx = sku2idx
        self.domain_filter = domain_filter

        self.samples: List[Dict] = []

        with open(self.jsonl_path, "r") as f:
            for line in f:
                rec = json.loads(line)
                domain = rec["domain"]
                if self.domain_filter is not None and domain != self.domain_filter:
                    continue

                sku_id = rec["sku_id"]
                if sku_id not in self.sku2idx:
                    continue
                sku_idx = self.sku2idx[sku_id]

                image_rel = rec["image_path"]
                self.samples.append(
                    {
                        "image_path": self.sku_root / image_rel,
                        "sku_idx": sku_idx,
                        "sku_id": sku_id,
                        "domain": domain,
                    }
                )
                if max_samples is not None and len(self.samples) >= max_samples:
                    break

        if len(self.samples) == 0:
            raise RuntimeError(f"No eval samples loaded from {self.jsonl_path}")

        print(
            f"[EvalDataset] Loaded {len(self.samples)} samples from {self.jsonl_path} "
            f"(domain_filter={self.domain_filter})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        rec = self.samples[idx]
        img_path = rec["image_path"]
        sku_idx = rec["sku_idx"]
        sku_id = rec["sku_id"]

        img = Image.open(img_path).convert("RGB")
        img_tensor = self.preprocess(img)

        label = torch.tensor(sku_idx, dtype=torch.long)
        dummy = torch.tensor(0, dtype=torch.long)
        return img_tensor, label, sku_id, dummy


def build_sku_mapping(jsonl_paths: List[Path]) -> Dict[str, int]:
    """
    Build a global mapping sku_id -> integer index by scanning multiple *_image_text.jsonl files.
    """
    sku2idx: Dict[str, int] = {}
    for p in jsonl_paths:
        with open(p, "r") as f:
            for line in f:
                rec = json.loads(line)
                sku_id = rec["sku_id"]
                if sku_id not in sku2idx:
                    sku2idx[sku_id] = len(sku2idx)
    print(f"Built sku2idx with {len(sku2idx)} SKUs")
    return sku2idx
