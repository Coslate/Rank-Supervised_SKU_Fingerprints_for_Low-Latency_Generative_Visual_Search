# eval/eval_reid_df2.py

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.df2_reid_dataset import (
    DeepFashion2ReIDDataset,
    build_eval_transform,
)
from models.reid_resnet50_bnneck import ReIDResNet50BNNeck


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DF2 re-id baseline on validation/test split."
    )
    parser.add_argument(
        "--jsonl_root",
        type=Path,
        default=Path("data"),
        help="Directory containing df2_reid_{split}.jsonl.",
    )
    parser.add_argument(
        "--sku_root",
        type=Path,
        default=Path("data/DeepFashion2_SKU"),
        help="Root with DeepFashion2 SKU crops.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
        help="Split to evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained checkpoint (.pt).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for embedding extraction.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Image size for evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use.",
    )
    parser.add_argument(
        "--ndcg_k",
        type=int,
        default=10,
        help="K for NDCG@K (default: 10)",
    )
    parser.add_argument(
        "--recall_ks",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="List of K values for Recall@K",
    )
    parser.add_argument(
        "--eval_cpu_latency",
        action="store_true",
        help=(
            "If set, latency will be evaluated on cpu. "
        ),
    )

    return parser.parse_args()


@torch.no_grad()
def compute_embeddings(
    model: torch.nn.Module,
    dataset: DeepFashion2ReIDDataset,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    all_embs = []
    all_labels = []
    all_sku_ids: List[str] = []

    for imgs, labels, sku_ids, _ in tqdm(loader, desc="Embedding"):
        imgs = imgs.to(device, non_blocking=True)
        _, emb = model(imgs, normalize=True)
        all_embs.append(emb.cpu())
        all_labels.append(labels.clone())
        all_sku_ids.extend(list(sku_ids))

    embs = torch.cat(all_embs, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return embs, labels, all_sku_ids


def compute_metrics(
    gallery_embs: torch.Tensor,
    gallery_labels: torch.Tensor,
    query_embs: torch.Tensor,
    query_labels: torch.Tensor,
    ndcg_k: int = 10,
    recall_ks = (1, 5, 10),
) -> dict:
    """
    gallery_embs: (Ng, D), L2-normalized
    gallery_labels: (Ng,)
    query_embs: (Nq, D), L2-normalized
    query_labels: (Nq,)
    ndcg_k: K of NDCG (default: 10)
    recall_ks: Recall@K
    """
    device = gallery_embs.device
    Ng, D = gallery_embs.shape
    Nq = query_embs.shape[0]

    num_skus = int(gallery_labels.max().item() + 1)

    gallery_labels = gallery_labels.to(device)
    query_embs = query_embs.to(device)

    ranks = []
    ndcgs = []
    reciprocal_ranks = []
    latencies_ms = []

    for i in tqdm(range(Nq), desc="Retrieval"):
        q = query_embs[i : i + 1]  # (1, D)
        label = int(query_labels[i].item())

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        # cosine similarity since embeddings are L2-normalized
        sims = torch.matmul(gallery_embs, q.t()).squeeze(1)  # (Ng,)

        # Aggregate max similarity per SKU using scatter_reduce
        per_sku_scores = torch.full(
            (num_skus,), -1e9, device=device, dtype=sims.dtype
        )
        per_sku_scores = per_sku_scores.scatter_reduce(
            0,
            gallery_labels,
            sims,
            reduce="amax",
            include_self=True,
        )

        score_gt = per_sku_scores[label]
        # Rank: 1 + number of SKUs with strictly higher score
        rank = int((per_sku_scores > score_gt).sum().item() + 1)

        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        latencies_ms.append((end - start) * 1000.0)

        ranks.append(rank)
        reciprocal_ranks.append(1.0 / rank)

        # NDCG@ndcg_k, here IDCG@K = 1 / log2(1+1) = 1
        if rank <= ndcg_k:
            ndcg = 1.0 / math.log2(rank + 1)
        else:
            ndcg = 0.0
        ndcgs.append(ndcg)

    ranks = np.array(ranks, dtype=np.int64)
    reciprocal_ranks = np.array(reciprocal_ranks, dtype=np.float64)
    ndcgs = np.array(ndcgs, dtype=np.float64)
    latencies_ms = np.array(latencies_ms, dtype=np.float64)

    metrics = {}
    # Recall@K
    for k in recall_ks:
        metrics[f"Recall@{k}"] = float((ranks <= k).mean())
    metrics["MRR"] = float(reciprocal_ranks.mean())
    metrics[f"NDCG@{ndcg_k}"] = float(ndcgs.mean())
    metrics["p95_latency_ms"] = float(np.percentile(latencies_ms, 95))
    metrics["mean_latency_ms"] = float(latencies_ms.mean())
    return metrics


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # we need weights_only=False in PyTorch 2.6+.
    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        # For older PyTorch versions that don't support weights_only
        ckpt = torch.load(args.checkpoint, map_location="cpu")

    num_classes = ckpt["num_classes"]
    model = ReIDResNet50BNNeck(num_classes=num_classes, feat_dim=512)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()

    transform = build_eval_transform(args.img_size)

    split_jsonl = args.jsonl_root / f"df2_reid_{args.split}.jsonl"

    gallery_dataset = DeepFashion2ReIDDataset(
        jsonl_path=split_jsonl,
        sku_root=args.sku_root,
        transform=transform,
        domain_filter="catalog",
    )
    query_dataset = DeepFashion2ReIDDataset(
        jsonl_path=split_jsonl,
        sku_root=args.sku_root,
        transform=transform,
        domain_filter="query",
    )

    print(
        f"[{args.split}] gallery={len(gallery_dataset)} "
        f"query={len(query_dataset)}"
    )

    gallery_embs, gallery_labels, _ = compute_embeddings(
        model, gallery_dataset, args.batch_size, device
    )
    query_embs, query_labels, _ = compute_embeddings(
        model, query_dataset, args.batch_size, device
    )

    if args.eval_cpu_latency:
        gallery_embs = gallery_embs.to('cpu')
        gallery_labels = gallery_labels.to('cpu')
        query_embs = query_embs.to('cpu')
        query_labels = query_labels.to('cpu')
    else:
        gallery_embs = gallery_embs.to('cuda')
        gallery_labels = gallery_labels.to('cuda')
        query_embs = query_embs.to('cuda')
        query_labels = query_labels.to('cuda')


    metrics = compute_metrics(
        gallery_embs,
        gallery_labels,
        query_embs,
        query_labels,
        ndcg_k=args.ndcg_k,
        recall_ks=tuple(args.recall_ks),
    )

    print("=== Evaluation results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
