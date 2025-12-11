# train/train_clip_sku_df2.py

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import open_clip

from dataset.df2_clip_sku_dataset import (
    DeepFashion2ImageTextSkuTrainDataset,
    DeepFashion2ImageSkuEvalDataset,
    build_sku_mapping,
)
from models.clip_sku_baseline import ClipSkuBaseline
from eval.eval_reid_df2 import compute_metrics  # reuse metrics (no FAISS)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CLIP/SigLIP SKU baseline on DeepFashion2_SKU."
    )
    parser.add_argument(
        "--sku_root",
        type=Path,
        required=True,
        help="Root of DeepFashion2_SKU (contains *_image_text.jsonl and crops).",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Train split name (default: train).",
    )
    parser.add_argument(
        "--val_split",
        type=str,
        default="validation",
        help="Validation split name (default: validation).",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-B-16",
        help="open_clip model name (e.g., ViT-B-16).",
    )
    parser.add_argument(
        "--clip_pretrained",
        type=str,
        default="laion2b_s34b_b88k",
        help="open_clip pretrained tag.",
    )
    parser.add_argument(
        "--freeze_towers",
        action="store_true",
        help="Freeze CLIP towers and train only SKU embeddings + logit_scale.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Base learning rate.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate for cosine schedule.",
    )
    parser.add_argument(
        "--clip_lr",
        type=float,
        default=1e-5,
        help="Base learning rate for CLIP image encoder when partially finetuning.",
    )
    parser.add_argument(
        "--clip_min_lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate for CLIP image encoder cosine schedule.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="Fraction of total training steps used for linear warmup (0-1).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device, e.g., cuda or cpu.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/clip_sku_baseline_final.pt"),
        help="Path to final checkpoint (last epoch).",
    )
    parser.add_argument(
        "--save_every_iter",
        type=int,
        default=0,
        help="Save checkpoint every N training iterations (<=0 disables).",
    )
    parser.add_argument(
        "--eval_every_iter",
        type=int,
        default=0,
        help="Run validation (loss + retrieval) every N training iterations (<=0 disables).",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Optional max training samples (for debugging).",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="10623finalproj_df2_clip_sku",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Optional Weights & Biases run name.",
    )
    parser.add_argument(
        "--ndcg_k",
        type=int,
        default=10,
        help="K for NDCG@K in retrieval metrics.",
    )
    parser.add_argument(
        "--recall_ks",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="List of K values for Recall@K.",
    )

    parser.add_argument(
        "--eval_mode",
        type=str,
        default="bestshot",
        choices=["bestshot", "mean_sku"],
        help=(
            "Evaluation mode: 'bestshot' uses max over gallery images per SKU "
            "inside compute_metrics; 'mean_sku' uses mean catalog embedding per SKU."
        ),
    )
    parser.add_argument(
        "--image_text_suffix",
        type=str,
        default="",
        help=(
            "Optional suffix for image_text jsonl filenames. "
            "Example: --image_text_suffix .dit_pretrained_aug "
            "will read 'train_image_text.dit_pretrained_aug.jsonl' "
            "instead of 'train_image_text.jsonl'."
        ),
    )
    parser.add_argument(
        "--partial_finetune",
        action="store_true",
        help=(
            "If set, freeze the CLIP text encoder and only lightly finetune "
            "the last few groups of the image encoder."
        ),
    )
    parser.add_argument(
        "--vision_unlocked_groups",
        type=int,
        default=1,
        help=(
            "Number of vision groups to unlock when --partial_finetune is set. "
            "For ViT-B-16, 1 is usually enough."
        ),
    )
    parser.add_argument(
        "--text_loss_weight",
        type=float,
        default=0.25,
        help="Weight of the text->SKU cross-entropy loss in the total loss.",
    )
    parser.add_argument(
        "--resume_from",
        type=Path,
        default=None,
        help="Optional checkpoint path to resume training from.",
    )
    return parser.parse_args()


@torch.no_grad()
def compute_embeddings_clip(
    model: ClipSkuBaseline,
    dataset: DeepFashion2ImageSkuEvalDataset,
    batch_size: int,
    device: torch.device,
    num_workers: int,
):
    """
    Extract image embeddings and labels for retrieval evaluation.

    Returns:
        embs:   (N, D)
        labels: (N,)
        sku_ids: list of string SKU ids (len N)
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    all_embs = []
    all_labels = []
    all_sku_ids: List[str] = []

    model.eval()
    for imgs, labels, sku_ids, _dummy in tqdm(loader, desc="Embedding"):
        imgs = imgs.to(device, non_blocking=True)
        emb = model.encode_image(imgs)
        all_embs.append(emb.cpu())
        all_labels.append(labels.clone())
        all_sku_ids.extend(list(sku_ids))

    embs = torch.cat(all_embs, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return embs, labels, all_sku_ids

def build_mean_sku_embeddings(
    gallery_embs: torch.Tensor,
    gallery_labels: torch.Tensor,
    num_skus: int,
) -> torch.Tensor:
    """
    Compute mean catalog embedding per SKU.

    gallery_embs:   (Ng, D), L2-normalized image features
    gallery_labels: (Ng,), int64 in [0, num_skus)
    returns:
        sku_embs: (num_skus, D), L2-normalized mean embeddings per SKU
    """
    device = gallery_embs.device
    Ng, D = gallery_embs.shape

    sums = torch.zeros(num_skus, D, device=device)
    counts = torch.zeros(num_skus, 1, device=device)

    sums.index_add_(0, gallery_labels, gallery_embs)
    ones = torch.ones(Ng, 1, device=device)
    counts.index_add_(0, gallery_labels, ones)

    counts = counts.clamp_min(1.0)
    sku_embs = sums / counts
    sku_embs = F.normalize(sku_embs, dim=1)
    return sku_embs

@torch.no_grad()
def compute_val_loss(
    model: ClipSkuBaseline,
    val_loader: DataLoader,
    device: torch.device,
    text_loss_weight: float,
):
    """
    Compute validation loss and accuracy (image->SKU and text->SKU).
    """
    model.eval()
    total_loss = 0.0
    total_img_acc = 0.0
    total_txt_acc = 0.0
    n_batches = 0

    alpha = text_loss_weight

    for imgs, text_tokens, sku_idx, _domain in val_loader:
        imgs = imgs.to(device, non_blocking=True)
        text_tokens = text_tokens.to(device, non_blocking=True)
        sku_idx = sku_idx.to(device, non_blocking=True)

        img_emb, txt_emb, sku_emb_all, logit_scale = model(imgs, text_tokens)
        logits_img = logit_scale * img_emb @ sku_emb_all.t()
        logits_txt = logit_scale * txt_emb @ sku_emb_all.t()

        loss_img = F.cross_entropy(logits_img, sku_idx)
        loss_txt = F.cross_entropy(logits_txt, sku_idx)
        loss = (1.0 - alpha) * loss_img + alpha * loss_txt

        img_pred = logits_img.argmax(dim=-1)
        txt_pred = logits_txt.argmax(dim=-1)
        img_acc = (img_pred == sku_idx).float().mean().item()
        txt_acc = (txt_pred == sku_idx).float().mean().item()

        total_loss += loss.item()
        total_img_acc += img_acc
        total_txt_acc += txt_acc
        n_batches += 1

    avg_loss = total_loss / max(1, n_batches)
    avg_img_acc = total_img_acc / max(1, n_batches)
    avg_txt_acc = total_txt_acc / max(1, n_batches)
    return avg_loss, avg_img_acc, avg_txt_acc


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = args.output.parent / f"{args.output.stem}_best.pt"

    # ----------------- W&B init -----------------
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    sku_root = args.sku_root
    # Decide which image_text jsonl we read, e.g.
    #   ""                    -> train_image_text.jsonl
    #   ".dit_pretrained_aug" -> train_image_text.dit_pretrained_aug.jsonl
    #   "dit_finetuned_aug"   -> train_image_text.dit_finetuned_aug.jsonl
    suffix = args.image_text_suffix or ""
    if suffix and not suffix.startswith("."):
        suffix = "." + suffix

    train_jsonl = sku_root / f"{args.train_split}_image_text{suffix}.jsonl"
    val_jsonl   = sku_root / f"{args.val_split}_image_text{suffix}.jsonl"

    # 1) Build global sku_id -> index mapping from train + val.
    sku2idx: Dict[str, int] = build_sku_mapping([train_jsonl, val_jsonl])
    num_skus = len(sku2idx)

    # 2) Create CLIP model, transforms, tokenizer.
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.clip_model)

    # 3) Datasets and loaders (image + text + sku_idx).
    train_ds = DeepFashion2ImageTextSkuTrainDataset(
        sku_root=sku_root,
        jsonl_path=train_jsonl,
        preprocess=preprocess,
        tokenizer=tokenizer,
        sku2idx=sku2idx,
        domain_filter=None,  # use both catalog + query
        max_samples=args.max_train_samples,
    )
    val_ds = DeepFashion2ImageTextSkuTrainDataset(
        sku_root=sku_root,
        jsonl_path=val_jsonl,
        preprocess=preprocess,
        tokenizer=tokenizer,
        sku2idx=sku2idx,
        domain_filter=None,
        max_samples=None,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"Number of SKUs: {num_skus}")

    # 4) Eval datasets for retrieval metrics (catalog as gallery, query as query).
    eval_gallery_ds = DeepFashion2ImageSkuEvalDataset(
        sku_root=sku_root,
        jsonl_path=val_jsonl,
        preprocess=preprocess,
        sku2idx=sku2idx,
        domain_filter="catalog",
    )
    eval_query_ds = DeepFashion2ImageSkuEvalDataset(
        sku_root=sku_root,
        jsonl_path=val_jsonl,
        preprocess=preprocess,
        sku2idx=sku2idx,
        domain_filter="query",
    )
    print(
        f"Validation gallery images: {len(eval_gallery_ds)}, "
        f"query images: {len(eval_query_ds)}"
    )

    # 5) CLIP + SKU embedding model.
    model = ClipSkuBaseline(
        clip_model=clip_model,
        num_skus=num_skus,
        freeze_towers=args.freeze_towers,
        partial_finetune=args.partial_finetune,
        vision_unlocked_groups=args.vision_unlocked_groups,
    )
    model.to(device)

    # 6) Optimizer with separate parameter groups for CLIP and SKU head.
    clip_params: List[torch.nn.Parameter] = []
    head_params: List[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Everything that lives inside the CLIP model goes to the CLIP group.
        # Adjust the prefix ("clip_model") if your attribute name is different.
        if name.startswith("clip_model."):
            clip_params.append(param)
        else:
            # SKU prototypes, logit_scale, and any extra heads go here.
            head_params.append(param)

    optimizer = optim.AdamW(
        [
            {
                "params": clip_params,
                "lr": args.clip_lr,
                "weight_decay": args.weight_decay,
            },
            {
                "params": head_params,
                "lr": args.lr,
                "weight_decay": 0.0,  # often better to not decay prototypes / logit_scale
            },
        ]
    )

    total_trainable = sum(p.numel() for p in clip_params + head_params)
    print(f"Trainable params: {total_trainable:,}")
    print(f"  - CLIP params: {sum(p.numel() for p in clip_params):,}")
    print(f"  - Head params: {sum(p.numel() for p in head_params):,}")

    # --- Optional: resume from checkpoint ---
    global_step = 0
    start_epoch = 1
    best_recall_at1 = 0.0

    if args.resume_from is not None:
        ckpt_path = args.resume_from
        print(f"[Resume] Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")

        # Restore model weights
        model.load_state_dict(ckpt["model_state"])

        # Restore optimizer state if present
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        else:
            print("[Resume] Warning: optimizer_state not found in checkpoint; "
                  "optimizer will start from fresh state.")

        # Sanity check: sku2idx mapping must match
        if "sku2idx" in ckpt and ckpt["sku2idx"] != sku2idx:
            raise ValueError(
                "[Resume] sku2idx mapping in checkpoint does not match current data."
            )

        global_step = ckpt.get("global_step", 0)
        # ckpt["epoch"] is the epoch index at the time of saving
        prev_epoch = ckpt.get("epoch", 0)
        start_epoch = prev_epoch + 1
        best_recall_at1 = ckpt.get("best_recall_at1", 0.0)

        print(
            f"[Resume] Resumed from epoch={prev_epoch}, "
            f"next start_epoch={start_epoch}, global_step={global_step}, "
            f"best_recall_at1={best_recall_at1:.4f}"
        )

        # If someone resumes from a "final" checkpoint and keeps the same epochs,
        # we avoid starting beyond the configured number of epochs.
        if start_epoch > args.epochs:
            print(
                f"[Resume] start_epoch ({start_epoch}) > args.epochs ({args.epochs}), "
                "clamping start_epoch to args.epochs."
            )
            start_epoch = args.epochs

    # 7) Cosine LR schedule with warmup (like train_reid_df2.py).
    num_train_batches = len(train_loader)
    total_steps = args.epochs * num_train_batches
    warmup_steps = int(total_steps * args.warmup_ratio)
    warmup_steps = max(warmup_steps, 1)  # avoid division by zero
    base_lr = args.lr
    min_lr = args.min_lr
    print(
        f"Total steps = {total_steps}, warmup_steps = {warmup_steps} "
        f"({args.warmup_ratio * 100:.1f}% of total)"
    )

    # NOTE: global_step, start_epoch, best_recall_at1 were already
    # initialized above and may have been overwritten by resume logic.
    # Do NOT reset them here.

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_img_loss = 0.0
        running_txt_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, (imgs, text_tokens, sku_idx, _domain) in enumerate(pbar):
            imgs = imgs.to(device, non_blocking=True)
            text_tokens = text_tokens.to(device, non_blocking=True)
            sku_idx = sku_idx.to(device, non_blocking=True)

            optimizer.zero_grad()
            img_emb, txt_emb, sku_emb_all, logit_scale = model(imgs, text_tokens)

            logits_img = logit_scale * img_emb @ sku_emb_all.t()
            logits_txt = logit_scale * txt_emb @ sku_emb_all.t()

            loss_img = F.cross_entropy(logits_img, sku_idx)
            loss_txt = F.cross_entropy(logits_txt, sku_idx)

            alpha = args.text_loss_weight  # e.g. 0.25
            loss = (1.0 - alpha) * loss_img + alpha * loss_txt

            loss.backward()
            optimizer.step()

            global_step += 1

            # ---- cosine LR with warmup (separate for CLIP vs head) ----
            if global_step <= warmup_steps:
                # Linear warmup
                clip_lr = args.clip_lr * float(global_step) / float(warmup_steps)
                head_lr = args.lr * float(global_step) / float(warmup_steps)
            else:
                progress = float(global_step - warmup_steps) / float(
                    max(1, total_steps - warmup_steps)
                )
                progress = min(max(progress, 0.0), 1.0)

                clip_lr = args.clip_min_lr + 0.5 * (args.clip_lr - args.clip_min_lr) * (
                    1.0 + math.cos(math.pi * progress)
                )
                head_lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (
                    1.0 + math.cos(math.pi * progress)
                )

            # param_groups[0] is CLIP, param_groups[1] is head
            optimizer.param_groups[0]["lr"] = clip_lr
            optimizer.param_groups[1]["lr"] = head_lr
            # -----------------------------------------------------------

            running_loss += loss.item()
            running_img_loss += loss_img.item()
            running_txt_loss += loss_txt.item()

            avg_loss = running_loss / global_step
            avg_img_loss = running_img_loss / global_step
            avg_txt_loss = running_txt_loss / global_step

            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                loss_img=f"{avg_img_loss:.4f}",
                loss_txt=f"{avg_txt_loss:.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                step=global_step,
            )

            # ---- W&B train logging (per iteration) ----
            if args.wandb:
                wandb.log(
                    {
                        "train/loss": float(loss.item()),
                        "train/loss_img": float(loss_img.item()),
                        "train/loss_txt": float(loss_txt.item()),
                        "train/clip_lr": float(optimizer.param_groups[0]["lr"]),
                        "train/head_lr": float(optimizer.param_groups[1]["lr"]),
                        "epoch": epoch,
                    },
                    step=global_step,
                )

            # ---- periodic validation (loss + retrieval) ----
            do_eval = (
                args.eval_every_iter > 0
                and global_step > 0
                and (global_step % args.eval_every_iter == 0)
            )
            if do_eval:
                print(
                    f"\n[Eval] Global step {global_step}: "
                    f"running validation loss + retrieval metrics..."
                )

                # 1) Validation loss / accuracy on val_loader
                val_loss, val_img_acc, val_txt_acc = compute_val_loss(
                    model=model,
                    val_loader=val_loader,
                    device=device,
                    text_loss_weight=args.text_loss_weight,
                )

                print(
                    f"[Eval] Validation loss={val_loss:.4f}, "
                    f"img_acc={val_img_acc:.4f}, txt_acc={val_txt_acc:.4f}"
                )

                # 2) Retrieval metrics on validation gallery/query
                gallery_embs, gallery_labels, _ = compute_embeddings_clip(
                    model=model,
                    dataset=eval_gallery_ds,
                    batch_size=args.batch_size,
                    device=device,
                    num_workers=args.num_workers,
                )
                query_embs, query_labels, _ = compute_embeddings_clip(
                    model=model,
                    dataset=eval_query_ds,
                    batch_size=args.batch_size,
                    device=device,
                    num_workers=args.num_workers,
                )

                gallery_embs = gallery_embs.to(device)
                query_embs = query_embs.to(device)
                gallery_labels = gallery_labels.to(device)
                query_labels = query_labels.to(device)

                if args.eval_mode == "mean_sku":
                    sku_embs = build_mean_sku_embeddings(
                        gallery_embs=gallery_embs,
                        gallery_labels=gallery_labels,
                        num_skus=num_skus,
                    )
                    sku_labels = torch.arange(
                        num_skus, device=device, dtype=torch.long
                    )
                    val_metrics = compute_metrics(
                        gallery_embs=sku_embs,
                        gallery_labels=sku_labels,
                        query_embs=query_embs,
                        query_labels=query_labels,
                        ndcg_k=args.ndcg_k,
                        recall_ks=tuple(args.recall_ks),
                    )
                else:
                    # bestshot: use image-level gallery embeddings; compute_metrics
                    # will take max over images per SKU internally.
                    val_metrics = compute_metrics(
                        gallery_embs=gallery_embs,
                        gallery_labels=gallery_labels,
                        query_embs=query_embs,
                        query_labels=query_labels,
                        ndcg_k=args.ndcg_k,
                        recall_ks=tuple(args.recall_ks),
                    )

                print("[Eval] Validation retrieval metrics:")
                for k, v in val_metrics.items():
                    if "latency" in k:
                        print(f"  {k}: {v:.2f}")
                    else:
                        print(f"  {k}: {v:.4f}")

                current_recall1 = val_metrics.get("Recall@1", 0.0)
                if current_recall1 > best_recall_at1:
                    best_recall_at1 = current_recall1
                    print(f"[Eval] New best Recall@1: {best_recall_at1:.4f}")

                    # ---- save best checkpoint ----
                    best_ckpt = {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "sku2idx": sku2idx,
                        "args": vars(args),
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_recall_at1": best_recall_at1,
                    }
                    torch.save(best_ckpt, best_ckpt_path)
                    print(f"[Eval] Saved new best checkpoint to {best_ckpt_path}")

                # W&B validation logging
                if args.wandb:
                    log_dict = {
                        "val/loss": float(val_loss),
                        "val/img_acc": float(val_img_acc),
                        "val/txt_acc": float(val_txt_acc),
                        "epoch": epoch,
                    }
                    for k, v in val_metrics.items():
                        log_dict[f"val/{k}"] = float(v)
                    wandb.log(log_dict, step=global_step)

                # switch back to train mode
                model.train()

            # ---- periodic checkpoint saving ----
            do_save = args.save_every_iter > 0 and global_step % args.save_every_iter == 0
            if do_save:
                ckpt_dir = args.output.parent
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = ckpt_dir / f"{args.output.stem}_step{global_step}.pt"
                ckpt = {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "sku2idx": sku2idx,
                    "args": vars(args),
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_recall_at1": best_recall_at1,
                }
                torch.save(ckpt, ckpt_path)
                print(f"[Checkpoint] Saved: {ckpt_path}")

        # end of epoch
        print(f"Epoch {epoch} finished. Global step = {global_step}")

    # ----------------- Final checkpoint (last epoch, last iteration) -----------------
    final_ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "sku2idx": sku2idx,
        "args": vars(args),
        "epoch": args.epochs,
        "global_step": global_step,
        "best_recall_at1": best_recall_at1,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final_ckpt, args.output)
    print(f"[Final] Saved final checkpoint to {args.output}")


if __name__ == "__main__":
    main()
