# dataset/df2_reid_dataset.py

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms


def load_jsonl(path: Path) -> List[dict]:
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


class DeepFashion2ReIDDataset(Dataset):
    """
    Re-ID dataset built from df2_reid_{split}.jsonl.

    Each record:
      {
        "split": str,
        "sku_id": str,
        "label": int,
        "domain": "catalog" | "query",
        "crop_path": str  # relative to sku_root
      }
    """

    def __init__(
        self,
        jsonl_path: Path,
        sku_root: Path,
        transform=None,
        domain_filter: Optional[str] = None,
    ):
        self.sku_root = Path(sku_root)
        self.records = load_jsonl(jsonl_path)
        if domain_filter is not None:
            self.records = [
                r for r in self.records if r["domain"] == domain_filter
            ]
        if not self.records:
            raise RuntimeError(f"No records after filtering in {jsonl_path}")

        self.transform = transform
        self.labels = [r["label"] for r in self.records]
        self.sku_ids = [r["sku_id"] for r in self.records]
        self.domains = [r["domain"] for r in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img_path = self.sku_root / rec["crop_path"]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        label = rec["label"]
        return img, label, rec["sku_id"], rec["domain"]


def build_train_transform(img_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def build_eval_transform(img_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class RandomIdentitySampler(Sampler[int]):
    """
    Random sampler that yields batches with P identities * K images each.

    labels: list of int, same length as dataset.
    batch_size = P * num_instances.
    """

    def __init__(self, labels: List[int], batch_size: int, num_instances: int):
        self.labels = np.asarray(labels)
        self.batch_size = batch_size
        self.num_instances = num_instances

        assert batch_size % num_instances == 0, "batch_size must be divisible by num_instances"
        self.num_pids_per_batch = batch_size // num_instances

        # Build index mapping: label -> list of indices
        self.index_dic = {}
        for index, label in enumerate(self.labels):
            self.index_dic.setdefault(label, []).append(index)

        self.index_dic = {
            pid: idxs
            for pid, idxs in self.index_dic.items()
            if len(idxs) >= self.num_instances
        }

        self.pids = list(self.index_dic.keys())
        if len(self.pids) == 0:
            raise ValueError(
                f"No pid has at least {self.num_instances} instances. "
                "Decrease num_instances or check your dataset."
            )

        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            # number of samples from this pid that will be used
            self.length += len(idxs) - len(idxs) % self.num_instances

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        batch_idxs_dict = {}
        for pid in self.pids:
            idxs = np.random.permutation(self.index_dic[pid])
            batch_idxs_dict[pid] = list(idxs)

        avai_pids = self.pids.copy()
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(
                avai_pids, self.num_pids_per_batch, replace=False
            )
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid]
                selected = batch_idxs[: self.num_instances]
                batch_idxs_dict[pid] = batch_idxs[self.num_instances :]
                final_idxs.extend(selected)
                if len(batch_idxs_dict[pid]) < self.num_instances:
                    avai_pids.remove(pid)

        return iter(final_idxs)
