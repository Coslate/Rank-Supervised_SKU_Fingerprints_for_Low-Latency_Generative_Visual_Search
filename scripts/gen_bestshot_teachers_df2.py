#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import open_clip

from dataset.df2_clip_sku_dataset import (
    DeepFashion2ImageSkuEvalDataset,
    DeepFashion2ImageTextSkuTrainDataset,
)
from models.clip_sku_baseline import ClipSkuBaseline


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-SKU best-shot image teacher embeddings (teacher_z_s) "
            "and per-SKU text teacher embeddings (sku_text_embs) from a trained "
            "ClipSkuBaseline checkpoint."
        )
    )
    parser.add_argument(
        "--sku_root",
        type=Path,
        required=True,
        help="Root of DeepFashion2_SKU (contains *_image_text*.jsonl and crops).",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        required=True,
        help="Path to ClipSkuBaseline checkpoint (.pt) from Stage-1 training.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split name to use for image-based teacher selection (default: train).",
    )
    parser.add_argument(
        "--text_split",
        type=str,
        default=None,
        help="Split name for text averaging (default: same as --split).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .npz file to save teacher embeddings.",
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
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device, e.g., cuda or cpu.",
    )
    return parser.parse_args()


@torch.no_grad()
def collect_image_embs_by_sku(
    model: ClipSkuBaseline,
    dataset: DeepFashion2ImageSkuEvalDataset,
    device: torch.device,
    num_skus: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[List[List[torch.Tensor]], int]:
    """
    For a given dataset (catalog or query), collect image embeddings grouped by SKU index.

    Returns:
        by_sku: list of length num_skus; each entry is a list of (D,) tensors.
        feat_dim: embedding dimension D.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    by_sku: List[List[torch.Tensor]] = [[] for _ in range(num_skus)]
    feat_dim = None

    model.eval()
    for imgs, labels, sku_ids, _domain in tqdm(
        loader, desc=f"Embedding ({dataset.domain_filter})"
    ):
        imgs = imgs.to(device, non_blocking=True)
        embs = model.encode_image(imgs)  # (B, D)
        embs = embs.detach().cpu()

        if feat_dim is None:
            feat_dim = embs.shape[1]

        for e, lbl in zip(embs, labels):
            idx = int(lbl.item())
            if 0 <= idx < num_skus:
                by_sku[idx].append(e)

    if feat_dim is None:
        raise RuntimeError("No embeddings were collected from the dataset.")

    return by_sku, feat_dim


@torch.no_grad()
def collect_text_embs_by_sku(
    model: ClipSkuBaseline,
    text_dataset: DeepFashion2ImageTextSkuTrainDataset,
    device: torch.device,
    num_skus: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[List[List[torch.Tensor]], int]:
    """
    Collect text embeddings grouped by SKU index using the Stage-1 model's forward path.
    For each batch, we pass both images and text_tokens through model(...),
    but only use txt_emb.

    Returns:
        by_sku: list of length num_skus; each entry is a list of (D_txt,) tensors.
        feat_dim_txt: text embedding dimension.
    """
    loader = DataLoader(
        text_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    by_sku: List[List[torch.Tensor]] = [[] for _ in range(num_skus)]
    feat_dim_txt = None

    model.eval()
    for imgs, text_tokens, sku_idx, _domain in tqdm(loader, desc="Embedding (text)"):
        imgs = imgs.to(device, non_blocking=True)
        text_tokens = text_tokens.to(device, non_blocking=True)

        # Use the same forward path as Stage-1 training to get txt_emb.
        _, txt_emb, _, _ = model(imgs, text_tokens)  # (B, D_txt)
        txt_emb = txt_emb.detach().cpu()

        if feat_dim_txt is None:
            feat_dim_txt = txt_emb.shape[1]

        for e, lbl in zip(txt_emb, sku_idx):
            idx = int(lbl.item())
            if 0 <= idx < num_skus:
                by_sku[idx].append(e)

    if feat_dim_txt is None:
        raise RuntimeError("No text embeddings were collected from the dataset.")

    return by_sku, feat_dim_txt


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    text_split = args.text_split or args.split

    print(f"[Config] sku_root    = {args.sku_root}")
    print(f"[Config] ckpt_path   = {args.ckpt_path}")
    print(f"[Config] split       = {args.split} (image teachers)")
    print(f"[Config] text_split  = {text_split} (text teachers)")
    print(f"[Config] output      = {args.output}")
    print("")

    # ------------------------------------------------------------------
    # 1) Load checkpoint and recover training-time arguments + sku2idx
    # ------------------------------------------------------------------
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args: Dict = ckpt.get("args", {})
    sku2idx: Dict[str, int] = ckpt["sku2idx"]
    num_skus = len(sku2idx)
    print(f"[Checkpoint] Loaded checkpoint from: {args.ckpt_path}")
    print(f"[Checkpoint] Number of SKUs in mapping: {num_skus}")

    # Build inverse mapping: index -> SKU string id
    inv_sku_ids: List[str] = [""] * num_skus
    for sku_str, idx in sku2idx.items():
        inv_sku_ids[idx] = sku_str

    # ------------------------------------------------------------------
    # 2) Rebuild CLIP + ClipSkuBaseline model and load weights
    # ------------------------------------------------------------------
    clip_model_name = ckpt_args.get("clip_model", "ViT-B-16")
    clip_pretrained = ckpt_args.get("clip_pretrained", "laion2b_s34b_b88k")
    print(f"[Model] clip_model      = {clip_model_name}")
    print(f"[Model] clip_pretrained = {clip_pretrained}")

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name,
        pretrained=clip_pretrained,
    )

    freeze_towers = ckpt_args.get("freeze_towers", False)
    partial_finetune = ckpt_args.get("partial_finetune", False)
    vision_unlocked_groups = ckpt_args.get("vision_unlocked_groups", 0)

    model = ClipSkuBaseline(
        clip_model=clip_model,
        num_skus=num_skus,
        freeze_towers=freeze_towers,
        partial_finetune=partial_finetune,
        vision_unlocked_groups=vision_unlocked_groups,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    print("[Model] ClipSkuBaseline loaded and moved to device.")

    # Prepare tokenizer for text dataset
    tokenizer = open_clip.get_tokenizer(clip_model_name)

    # ------------------------------------------------------------------
    # 3) Build split datasets for image teachers: catalog + query (image side)
    # ------------------------------------------------------------------
    image_text_suffix = ckpt_args.get("image_text_suffix", "") or ""
    if image_text_suffix and not image_text_suffix.startswith("."):
        image_text_suffix = "." + image_text_suffix

    img_split_jsonl = args.sku_root / f"{args.split}_image_text{image_text_suffix}.jsonl"
    if not img_split_jsonl.is_file():
        raise FileNotFoundError(f"JSONL file not found: {img_split_jsonl}")

    print(f"[Dataset] Using {img_split_jsonl} for split='{args.split}' (image teachers)")

    catalog_ds = DeepFashion2ImageSkuEvalDataset(
        sku_root=args.sku_root,
        jsonl_path=img_split_jsonl,
        preprocess=preprocess,
        sku2idx=sku2idx,
        domain_filter="catalog",
    )
    query_ds = DeepFashion2ImageSkuEvalDataset(
        sku_root=args.sku_root,
        jsonl_path=img_split_jsonl,
        preprocess=preprocess,
        sku2idx=sku2idx,
        domain_filter="query",
    )

    print(f"[Dataset] Catalog images: {len(catalog_ds)}")
    print(f"[Dataset] Query images:   {len(query_ds)}")

    # ------------------------------------------------------------------
    # 4) Extract image embeddings per SKU for catalog + query
    # ------------------------------------------------------------------
    catalog_by_sku, feat_dim_cat = collect_image_embs_by_sku(
        model=model,
        dataset=catalog_ds,
        device=device,
        num_skus=num_skus,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    query_by_sku, feat_dim_q = collect_image_embs_by_sku(
        model=model,
        dataset=query_ds,
        device=device,
        num_skus=num_skus,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    if feat_dim_cat != feat_dim_q:
        raise RuntimeError(
            f"Feature dimensions for catalog ({feat_dim_cat}) and query ({feat_dim_q}) do not match."
        )
    feat_dim = feat_dim_cat
    print(f"[Embedding] Image feature dimension D_img = {feat_dim}")

    # ------------------------------------------------------------------
    # 5) Build text dataset and collect text embeddings per SKU
    # ------------------------------------------------------------------
    text_split_jsonl = args.sku_root / f"{text_split}_image_text{image_text_suffix}.jsonl"
    if not text_split_jsonl.is_file():
        raise FileNotFoundError(f"JSONL file not found for text_split: {text_split_jsonl}")

    print(f"[Dataset] Using {text_split_jsonl} for text_split='{text_split}' (text teachers)")

    text_ds = DeepFashion2ImageTextSkuTrainDataset(
        sku_root=args.sku_root,
        jsonl_path=text_split_jsonl,
        preprocess=preprocess,
        tokenizer=tokenizer,
        sku2idx=sku2idx,
        domain_filter=None,  # use both catalog + query texts
        max_samples=None,
    )

    print(f"[Dataset] Text samples: {len(text_ds)}")

    text_by_sku, feat_dim_txt = collect_text_embs_by_sku(
        model=model,
        text_dataset=text_ds,
        device=device,
        num_skus=num_skus,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"[Embedding] Text feature dimension D_txt = {feat_dim_txt}")

    # ------------------------------------------------------------------
    # 6) For each SKU, pick best-shot image teacher using intra-SKU Recall@1
    #    and compute per-SKU text teacher via mean text embedding.
    # ------------------------------------------------------------------
    teacher_embs = torch.zeros(num_skus, feat_dim, dtype=torch.float32)
    sku_text_embs = torch.zeros(num_skus, feat_dim_txt, dtype=torch.float32)

    missing_both_img = 0
    missing_query = 0
    missing_catalog = 0
    missing_text = 0

    # Image-side teacher (best-shot)
    for s in tqdm(range(num_skus), desc="Best-shot image teacher per SKU"):
        q_list = query_by_sku[s]
        c_list = catalog_by_sku[s]

        if len(q_list) == 0 and len(c_list) == 0:
            # No image data at all for this SKU in the chosen split.
            # Leave zero vector (can be handled downstream).
            missing_both_img += 1
            continue
        elif len(q_list) == 0:
            # No queries, only catalog images: fall back to mean catalog embedding.
            missing_query += 1
            C_s = torch.stack(c_list, dim=0)  # (K_s, D)
            C_s = F.normalize(C_s, dim=1)
            teacher = C_s.mean(dim=0, keepdim=True)
            teacher = F.normalize(teacher, dim=1)[0]
        elif len(c_list) == 0:
            # No catalog images, only queries: fall back to mean query embedding.
            missing_catalog += 1
            Q_s = torch.stack(q_list, dim=0)  # (N_s, D)
            Q_s = F.normalize(Q_s, dim=1)
            teacher = Q_s.mean(dim=0, keepdim=True)
            teacher = F.normalize(teacher, dim=1)[0]
        else:
            # Normal case: both queries and catalog images exist.
            Q_s = torch.stack(q_list, dim=0)  # (N_s, D)
            C_s = torch.stack(c_list, dim=0)  # (K_s, D)

            Q_s = F.normalize(Q_s, dim=1)
            C_s = F.normalize(C_s, dim=1)

            # Cosine similarity matrix (N_s, K_s)
            S = Q_s @ C_s.t()

            # For each query, get the index of the top-1 catalog view.
            top1_idx = S.argmax(dim=1)  # (N_s,)
            hits = torch.bincount(top1_idx, minlength=C_s.size(0))
            best_k = hits.argmax().item()

            teacher = C_s[best_k]

        teacher_embs[s] = teacher

    print(f"[Stats/Image] SKUs with no query and no catalog: {missing_both_img}")
    print(f"[Stats/Image] SKUs with no query (used mean catalog): {missing_query}")
    print(f"[Stats/Image] SKUs with no catalog (used mean query): {missing_catalog}")

    # Text-side teacher (mean text embedding per SKU)
    for s in tqdm(range(num_skus), desc="Best-shot text teacher per SKU"):
        t_list = text_by_sku[s]
        if len(t_list) == 0:
            # No text for this SKU: leave zero (can be handled downstream,
            # or you could optionally copy image teacher here).
            missing_text += 1
            continue

        T_s = torch.stack(t_list, dim=0)  # (M_s, D_txt)
        T_s = F.normalize(T_s, dim=1)
        t_mean = T_s.mean(dim=0, keepdim=True)
        t_mean = F.normalize(t_mean, dim=1)[0]
        sku_text_embs[s] = t_mean

    print(f"[Stats/Text] SKUs with no text samples: {missing_text}")

    # L2-normalize all image teachers (for safety).
    teacher_embs = F.normalize(teacher_embs, dim=1)

    # ------------------------------------------------------------------
    # 7) Save to .npz
    # ------------------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)

    teacher_np = teacher_embs.numpy()
    sku_text_np = sku_text_embs.numpy()
    sku_ids_np = np.array(inv_sku_ids, dtype=object)

    np.savez(
        args.output,
        teacher_embs=teacher_np,
        sku_text_embs=sku_text_np,
        sku_ids=sku_ids_np,
    )

    print(f"[Output] Saved best-shot image teachers and text teachers for {num_skus} SKUs.")
    print(f"[Output] teacher_embs shape   = {teacher_np.shape}")
    print(f"[Output] sku_text_embs shape  = {sku_text_np.shape}")
    print(f"[Output] sku_ids length       = {len(sku_ids_np)}")
    print(f"[Output] File written to      = {args.output}")


if __name__ == "__main__":
    main()
