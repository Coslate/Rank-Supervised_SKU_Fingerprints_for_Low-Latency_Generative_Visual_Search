# eval/eval_clip_sku_df2.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

import open_clip
import torch.nn.functional as F

from dataset.df2_clip_sku_dataset import (
    DeepFashion2ImageSkuEvalDataset,
)
from models.clip_sku_baseline import ClipSkuBaseline

# Reuse the metrics function from your existing ReID eval.
from eval.eval_reid_df2 import compute_metrics  # type: ignore
from train.train_clip_sku_df2 import build_mean_sku_embeddings, compute_embeddings_clip

# Reuse text-query helpers from eval_sku_fingerprint_student
from eval.eval_sku_fingerprint_student import (
    compute_text_query_embeddings,
    build_sku2idx_from_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CLIP/SigLIP SKU baseline on DeepFashion2_SKU (no FAISS)."
    )
    parser.add_argument(
        "--sku_root",
        type=Path,
        required=True,
        help="Root of DeepFashion2_SKU.",
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
        help="Checkpoint from train_clip_sku_df2.py.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for embedding extraction.",
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
        help="Device.",
    )
    parser.add_argument(
        "--ndcg_k",
        type=int,
        default=10,
        help="K for NDCG@K.",
    )
    parser.add_argument(
        "--recall_ks",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="List of K values for Recall@K.",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default=None,
        help="Optional: override CLIP model name (else load from checkpoint args).",
    )
    parser.add_argument(
        "--clip_pretrained",
        type=str,
        default=None,
        help="Optional: override CLIP pretrained tag (else load from checkpoint args).",
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
        default=None,
        help=(
            "Optional suffix for image_text jsonl filename. "
            "If not set, will fall back to the suffix stored in the checkpoint args."
        ),
    )
    parser.add_argument(
        "--eval_all_val_skus",
        action="store_true",
        help=(
            "If set, build sku2idx from the validation image_text JSONL and evaluate "
            "on all SKUs that appear there (including SKUs unseen during student training). "
            "By default (flag not set), only SKUs present in the student checkpoint's "
            "sku_ids are evaluated."
        ),
    )
    parser.add_argument(
        "--eval_cpu_latency",
        action="store_true",
        help=(
            "If set, latency will be evaluated on cpu. "
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    sku_root = args.sku_root

    # Load checkpoint first (we may need its args to know the suffix)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args: Dict = ckpt.get("args", {})

    # Decide which image_text suffix to use:
    #   1) if user passes --image_text_suffix, use that
    #   2) else fall back to what was used during training (stored in ckpt_args)
    suffix = args.image_text_suffix
    if suffix is None:
        suffix = ckpt_args.get("image_text_suffix", "")
    if suffix and not suffix.startswith("."):
        suffix = "." + suffix

    split_jsonl = sku_root / f"{args.split}_image_text{suffix}.jsonl"

    # Decide SKU universe for evaluation
    if args.eval_all_val_skus:
        # new mode: use validation jsonl SKU
        sku2idx: Dict[str, int] = build_sku2idx_from_jsonl(split_jsonl)
        num_skus_total = len(sku2idx)
        print(
            "[INFO] SKU universe = ALL SKUs appearing in validation image_text "
            f"(including unseen SKUs); num_skus_total={num_skus_total}"
        )
    else:
        # old mode：only student ckpt SKU
        sku_ids: Dict[str, int] = ckpt["sku2idx"]
        sku2idx = {sku_id: i for i, sku_id in enumerate(sku_ids)}
        num_skus_total = len(sku2idx)
        print(
            "[INFO] SKU universe = SKUs stored in student checkpoint only "
            f"(seen-SKU evaluation); num_skus_total={num_skus_total}"
        )

    # Restore clip model/pretrained from checkpoint args (with optional override).
    clip_model_name = args.clip_model or ckpt_args.get("clip_model", "ViT-B-16")
    clip_pretrained_tag = args.clip_pretrained or ckpt_args.get(
        "clip_pretrained", "laion2b_s34b_b88k"
    )

    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"clip_model         = {clip_model_name}, pretrained = {clip_pretrained_tag}")
    print(f"num_skus_total     = {num_skus_total}")
    print(f"eval_mode          = {args.eval_mode}")
    print(f"image_text_suffix  = '{suffix}'")
    print(f"split_jsonl        = {split_jsonl}")

    # Rebuild CLIP model and preprocess.
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name, pretrained=clip_pretrained_tag
    )
    tokenizer = open_clip.get_tokenizer(clip_model_name)

    # Wrap with ClipSkuBaseline and load weights.
    model = ClipSkuBaseline(
        clip_model=clip_model,
        num_skus=len({sku_id: i for i, sku_id in enumerate(ckpt['sku2idx'])}),
        #num_skus=num_skus_total,
        freeze_towers=False,
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()

    # Build gallery/query datasets: catalog as gallery, query as query.
    gallery_ds = DeepFashion2ImageSkuEvalDataset(
        sku_root=sku_root,
        jsonl_path=split_jsonl,
        preprocess=preprocess,
        sku2idx=sku2idx,
        domain_filter="catalog",
    )
    query_ds = DeepFashion2ImageSkuEvalDataset(
        sku_root=sku_root,
        jsonl_path=split_jsonl,
        preprocess=preprocess,
        sku2idx=sku2idx,
        domain_filter="query",
    )

    print(
        f"[{args.split}] gallery images={len(gallery_ds)}, "
        f"query images={len(query_ds)}"
    )

    # --------------------------------------------------
    # 1) Image → SKU retrieval (original behavior)
    # --------------------------------------------------
    gallery_embs, gallery_labels, _ = compute_embeddings_clip(
        model, gallery_ds, args.batch_size, device, args.num_workers
    )
    query_embs, query_labels, _ = compute_embeddings_clip(
        model, query_ds, args.batch_size, device, args.num_workers
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

    # Choose evaluation mode.
    if args.eval_mode == "mean_sku":
        # Per-SKU mean catalog embedding as gallery.
        sku_embs = build_mean_sku_embeddings(
            gallery_embs=gallery_embs,
            gallery_labels=gallery_labels,
            num_skus=num_skus_total,
        )
        sku_labels = torch.arange(num_skus_total, device=device, dtype=torch.long)

        if args.eval_cpu_latency:
            sku_embs   = sku_embs.to('cpu')
            sku_labels = sku_labels.to('cpu')
        else:
            sku_embs   = sku_embs.to('cuda')
            sku_labels = sku_labels.to('cuda')

        metrics_img = compute_metrics(
            gallery_embs=sku_embs,
            gallery_labels=sku_labels,
            query_embs=query_embs,
            query_labels=query_labels,
            ndcg_k=args.ndcg_k,
            recall_ks=tuple(args.recall_ks),
        )
    else:
        # bestshot: image-level gallery; compute_metrics will aggregate per SKU by max.
        metrics_img = compute_metrics(
            gallery_embs=gallery_embs,
            gallery_labels=gallery_labels,
            query_embs=query_embs,
            query_labels=query_labels,
            ndcg_k=args.ndcg_k,
            recall_ks=tuple(args.recall_ks),
        )

    print(f"=== CLIP-SKU Evaluation results ({args.split}) - Image→SKU ===")
    for k, v in metrics_img.items():
        if "latency" in k:
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v:.4f}")

    # --------------------------------------------------
    # 2) Text → SKU retrieval (reusing compute_text_query_embeddings)
    # --------------------------------------------------
    text_query_embs, text_query_labels = compute_text_query_embeddings(
        clip_model=model,  # ClipSkuBaseline exposes encode_text()
        tokenizer=tokenizer,
        val_image_text=split_jsonl,
        sku2idx=sku2idx,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if text_query_embs.numel() == 0:
        print("[INFO] Skipping text→SKU evaluation because no text queries were found.")
        return

    if args.eval_cpu_latency:
        text_query_embs = text_query_embs.to('cpu')
        text_query_labels = text_query_labels.to('cpu')
    else:
        text_query_embs = text_query_embs.to('cuda')
        text_query_labels = text_query_labels.to('cuda')

    if args.eval_mode == "mean_sku":
        gallery_for_text = sku_embs
        gallery_labels_for_text = sku_labels
    else:
        gallery_for_text = gallery_embs
        gallery_labels_for_text = gallery_labels

    metrics_txt = compute_metrics(
        gallery_embs=gallery_for_text,
        gallery_labels=gallery_labels_for_text,
        query_embs=text_query_embs,
        query_labels=text_query_labels,
        ndcg_k=args.ndcg_k,
        recall_ks=tuple(args.recall_ks),
    )

    print(f"=== CLIP-SKU Evaluation results ({args.split}) - Text→SKU ===")
    for k, v in metrics_txt.items():
        if "latency" in k:
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
