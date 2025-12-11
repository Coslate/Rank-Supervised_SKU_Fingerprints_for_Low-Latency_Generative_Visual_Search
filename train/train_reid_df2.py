# train/train_reid_df2.py

from __future__ import annotations

import argparse
import time
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from dataset.df2_reid_dataset import (
    DeepFashion2ReIDDataset,
    RandomIdentitySampler,
    build_train_transform,
    build_eval_transform,
)
from models.reid_resnet50_bnneck import ReIDResNet50BNNeck
from eval.eval_reid_df2 import compute_embeddings, compute_metrics


class BatchHardTripletLoss(nn.Module):
    """
    Batch-hard triplet loss with margin.

    embeddings: (B, D), L2-normalized
    labels: (B,)
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(embeddings, embeddings, p=2).pow(2)  # (B, B)

        labels = labels.unsqueeze(1) #(B, 1)
        mask_pos = labels.eq(labels.t())  # (B, B)
        mask_neg = ~mask_pos

        # Remove self-distance for positives
        # Hardest positive: max distance among positives
        dist_pos = dist.clone()
        dist_pos[~mask_pos] = -1e9
        dist_pos.fill_diagonal_(-1e9)
        hardest_pos, _ = dist_pos.max(dim=1)

        # Hardest negative: min distance among negatives
        dist_neg = dist.clone()
        dist_neg[~mask_neg] = 1e9
        hardest_neg, _ = dist_neg.min(dim=1)

        loss = F.relu(hardest_pos - hardest_neg + self.margin)
        return loss.mean()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DF2 re-id baseline (image-level index)."
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
        "--output_dir",
        type=Path,
        default=Path("outputs/baseline1_reid"),
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Global batch size (P * K).",
    )
    parser.add_argument(
        "--num_instances",
        type=int,
        default=4,
        help="Number of images per identity in a batch (K).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.3,
        help="Triplet loss margin.",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Image size for training.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Dataloader workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs.",
    )
    parser.add_argument(
        "--eval_every_iter",
        type=int,
        default=2000,
        help=(
            "Run retrieval evaluation every N training iterations. "
            "Set <= 0 to disable eval during training."
        ),
    )
    parser.add_argument(
        "--ndcg_k",
        type=int,
        default=10,
        help="K for NDCG@K in retrieval metrics.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="Fraction of total training steps used for linear warmup (0~1).",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate for cosine LR schedule.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="10623finalproj_df2_reid_baseline1",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Optional Weights & Biases run name.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----------------- W&B init -----------------
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    # ----------------- Train dataset -----------------
    train_jsonl = args.jsonl_root / "df2_reid_train.jsonl"
    if not train_jsonl.exists():
        raise FileNotFoundError(f"Train jsonl not found: {train_jsonl}")
    transform = build_train_transform(args.img_size)

    train_dataset = DeepFashion2ReIDDataset(
        jsonl_path=train_jsonl,
        sku_root=args.sku_root,
        transform=transform,
        domain_filter=None,  # use both catalog + query for supervision
    )
    num_classes = len(set(train_dataset.labels))
    print(f"Train samples: {len(train_dataset)}, num_classes={num_classes}")

    sampler = RandomIdentitySampler(
        labels=train_dataset.labels,
        batch_size=args.batch_size,
        num_instances=args.num_instances,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ----------------- Validation datasets (gallery/query) -----------------
    val_jsonl = args.jsonl_root / "df2_reid_validation.jsonl"
    if not val_jsonl.exists():
        print(
            f"[WARN] Validation jsonl not found at {val_jsonl}. "
            "Validation metrics will be skipped."
        )
        val_gallery_dataset = None
        val_query_dataset = None
    else:
        eval_transform = build_eval_transform(args.img_size)
        val_gallery_dataset = DeepFashion2ReIDDataset(
            jsonl_path=val_jsonl,
            sku_root=args.sku_root,
            transform=eval_transform,
            domain_filter="catalog",
        )
        val_query_dataset = DeepFashion2ReIDDataset(
            jsonl_path=val_jsonl,
            sku_root=args.sku_root,
            transform=eval_transform,
            domain_filter="query",
        )

        # ReID-style validation loader (both catalog + query) for val loss
        val_reid_dataset = DeepFashion2ReIDDataset(
            jsonl_path=val_jsonl,
            sku_root=args.sku_root,
            transform=eval_transform,
            domain_filter=None,
        )
        val_reid_loader = DataLoader(
            val_reid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        print(
            f"Validation gallery samples: {len(val_gallery_dataset)}, "
            f"query samples: {len(val_query_dataset)}"
        )

    # ----------------- Model & optim -----------------
    model = ReIDResNet50BNNeck(num_classes=num_classes, feat_dim=512)
    model.to(device)

    ce_loss = nn.CrossEntropyLoss()
    tri_loss = BatchHardTripletLoss(margin=args.margin)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ----------------- Cosine LR schedule settings -----------------
    num_train_batches = len(train_loader)
    total_steps = args.epochs * num_train_batches
    warmup_steps = int(total_steps * args.warmup_ratio)
    warmup_steps = max(warmup_steps, 1)  # at least 1 step to avoid dividing by 0
    base_lr = args.lr
    min_lr = args.min_lr
    print(
        f"Total steps = {total_steps}, warmup_steps = {warmup_steps} "
        f"({args.warmup_ratio*100:.1f}% of total)"
    )

    global_step = 0
    best_recall_at1 = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_id_loss = 0.0
        running_tri_loss = 0.0

        start_epoch = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for imgs, labels, _, _ in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits, emb = model(imgs, normalize=True)

            loss_id = ce_loss(logits, labels)
            loss_tri = tri_loss(emb, labels)
            loss = loss_id + loss_tri

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_id_loss += loss_id.item()
            running_tri_loss += loss_tri.item()
            global_step += 1

            # ----- cosine LR with warmup -----
            if global_step <= warmup_steps:
                # linear warmup from 0 -> base_lr
                lr = base_lr * float(global_step) / float(warmup_steps)
            else:
                # cosine decay from base_lr -> min_lr
                progress = float(global_step - warmup_steps) / float(
                    max(1, total_steps - warmup_steps)
                )

                # clup [0, 1]
                progress = min(max(progress, 0.0), 1.0)
                lr = min_lr + 0.5 * (base_lr - min_lr) * (
                    1.0 + math.cos(math.pi * progress)
                )

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            # ----------------------------------

            avg_loss = running_loss / global_step
            avg_id = running_id_loss / global_step
            avg_tri = running_tri_loss / global_step

            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                id=f"{avg_id:.4f}",
                tri=f"{avg_tri:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                step=global_step,
            )

            # ---- W&B train logging (per iteration) ----
            if args.wandb:
                wandb.log(
                    {
                        "train/loss": float(loss.item()),
                        "train/loss_id": float(loss_id.item()),
                        "train/loss_tri": float(loss_tri.item()),
                        "train/lr": float(optimizer.param_groups[0]["lr"]),
                        "epoch": epoch,
                    },
                    step=global_step,
                )

            # ------------- periodic validation eval by iteration -------------
            do_eval = (
                args.eval_every_iter > 0
                and val_gallery_dataset is not None
                and val_query_dataset is not None
                and global_step > 0
                and (global_step % args.eval_every_iter == 0)
            )

            if do_eval:
                print(
                    f"\n[Eval] Global step {global_step}: running retrieval eval "
                    f"on validation split..."
                )
                # Switch to eval mode for inference
                model.eval()

                # 1) Retrieval metrics on validation (catalog/query)
                gallery_embs, gallery_labels, _ = compute_embeddings(
                    model=model,
                    dataset=val_gallery_dataset,
                    batch_size=args.batch_size,
                    device=device,
                )
                query_embs, query_labels, _ = compute_embeddings(
                    model=model,
                    dataset=val_query_dataset,
                    batch_size=args.batch_size,
                    device=device,
                )

                gallery_embs = gallery_embs.to(device)
                gallery_labels = gallery_labels.to(device)
                query_embs = query_embs.to(device)

                val_metrics = compute_metrics(
                    gallery_embs=gallery_embs,
                    gallery_labels=gallery_labels,
                    query_embs=query_embs,
                    query_labels=query_labels,
                    ndcg_k=args.ndcg_k,
                )

                print("[Eval] Validation retrieval metrics:")
                for k, v in val_metrics.items():
                    print(f"  {k}: {v:.4f}")

                if val_metrics.get("Recall@1", 0.0) > best_recall_at1:
                    best_recall_at1 = val_metrics["Recall@1"]
                    print(f"[Eval] New best Recall@1: {best_recall_at1:.4f}")

                # 2) Validation Metrics
                # W&B validation logging
                if args.wandb:
                    log_dict = {}
                    # retrieval metrics
                    for k, v in val_metrics.items():
                        log_dict[f"val/{k}"] = float(v)
                    log_dict["epoch"] = epoch
                    wandb.log(log_dict, step=global_step)

                # Switch back to train mode
                model.train()

        epoch_time = time.time() - start_epoch
        print(f"Epoch {epoch} finished in {epoch_time:.1f}s")

        # ----------------- Save checkpoint by epoch -----------------
        if (epoch % args.save_every == 0) or (epoch == args.epochs):
            ckpt = {
                "model": model.state_dict(),
                "num_classes": num_classes,
                "epoch": epoch,
                "args": vars(args),
            }
            ckpt_path = args.output_dir / f"baseline1_reid_epoch{epoch}.pt"
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
