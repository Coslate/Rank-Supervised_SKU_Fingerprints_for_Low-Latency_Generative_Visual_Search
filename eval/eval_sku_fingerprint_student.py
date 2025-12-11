# eval/eval_sku_fingerprint_student.py
from __future__ import annotations

import argparse
import json
import random  # [DEMO] for random sampling
import shutil  # [DEMO] for copying catalog images
from pathlib import Path
from typing import Dict, List, Tuple, Any
from PIL import Image  # local import to avoid global dependency

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import open_clip
import wandb

from models.clip_sku_baseline import ClipSkuBaseline
from models.sku_fingerprint_student import SkuFingerprintStudent
from dataset.df2_clip_sku_dataset import DeepFashion2ImageSkuEvalDataset
from eval.eval_reid_df2 import compute_metrics

# ---------------------------
# VLA imports (policy + actions + quality features)
# ---------------------------
from VLA_model.model.policy import VLAPolicy
from VLA_model.Image_processing.image_feature import compute_quality_features
from VLA_model.Image_processing.image_process import VLAAction, ACTION_FUNCS


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Step 4: Evaluate SKU fingerprint student on DeepFashion2_SKU "
            "by using student fingerprints as gallery and CLIP-SKU as query encoder."
        )
    )
    parser.add_argument(
        "--sku_root",
        type=Path,
        required=True,
        help="DeepFashion2_SKU root (contains *_image_text*.jsonl).",
    )
    parser.add_argument(
        "--val_split",
        type=str,
        default="validation",
        help="Validation split name (default: validation).",
    )
    parser.add_argument(
        "--val_image_text",
        type=Path,
        default=None,
        help=(
            "Optional explicit path to <split>_image_text*.jsonl for validation. "
            "If None, constructed from sku_root / <val_split>_image_text[<suffix>].jsonl"
        ),
    )
    parser.add_argument(
        "--image_text_suffix",
        type=str,
        default="",
        help=(
            "Optional suffix for image_text filenames, e.g. '.dit_clipsku_sub3k_nv4'. "
            "Only used when --val_image_text is None."
        ),
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
        "--clip_sku_ckpt",
        type=Path,
        required=True,
        help="Fine-tuned CLIP-SKU checkpoint (e.g., baseline5 best.pt).",
    )
    parser.add_argument(
        "--student_ckpt",
        type=Path,
        required=True,
        help="SKU fingerprint student checkpoint (.pt) from train_sku_fingerprint_distill.py.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for query embedding extraction.",
    )
    parser.add_argument(
        "--precompute_sku_batch_size",
        type=int,
        default=256,
        help="Batch size for precomputing validation catalog view embeddings.",
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
        "--wandb_project",
        type=str,
        default=None,
        help="If set, enable Weights & Biases logging with this project name.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Optional Weights & Biases run name.",
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
    # [DEMO] Image demo options
    parser.add_argument(
        "--demo_image_num",
        type=int,
        default=0,
        help="If > 0, randomly sample this many image queries for a qualitative demo.",
    )
    parser.add_argument(
        "--demo_image_output_dir",
        type=Path,
        default=None,
        help="Output directory for image-query demo examples.",
    )
    # [DEMO] Text demo options
    parser.add_argument(
        "--demo_text_num",
        type=int,
        default=0,
        help="If > 0, randomly sample this many text queries for a qualitative demo.",
    )
    parser.add_argument(
        "--demo_text_output_dir",
        type=Path,
        default=None,
        help="Output directory for text-query demo examples.",
    )
    # VLA options
    parser.add_argument(
        "--use_vla",
        action="store_true",
        help=(
            "If set, apply the trained VLA policy to enhance query images "
            "before encoding them with CLIP-SKU."
        ),
    )
    parser.add_argument(
        "--vla_checkpoint",
        type=Path,
        default=None,
        help=(
            "Path to VLA policy checkpoint (vla_policy.pt). "
            "Required if --use_vla is set."
        ),
    )
    return parser.parse_args()


# ---------------------------
# Helpers: load CLIP-SKU student teacher
# ---------------------------
def load_clip_sku_model(
    device: torch.device,
    clip_model_name: str,
    clip_pretrained: str,
    ckpt_path: Path,
):
    """
    Build open_clip backbone, wrap it with ClipSkuBaseline, and load the
    fine-tuned baseline checkpoint.

    Returns:
        model: ClipSkuBaseline; use encode_image() to get image embeddings.
        preprocess: image preprocessing transform from open_clip.
    """
    raw_clip_model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name,
        pretrained=clip_pretrained,
    )

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    sku2idx = ckpt["sku2idx"]
    num_skus = len(sku2idx)

    model = ClipSkuBaseline(
        clip_model=raw_clip_model,
        num_skus=num_skus,
        freeze_towers=False,
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()

    return model, preprocess

# ---------------------------
# Helpers: load VLA policy model
# ---------------------------
def load_vla_model(
    device: torch.device,
    clip_model_name: str,
    clip_pretrained: str,
    ckpt_path: Path,
    clip_model: Any = None,
    preprocess: Any = None
):
    """
    Load the VLA policy model and its CLIP visual backbone.

    Returns:
        policy_model: VLAPolicy
        clip_model:   open_clip CLIP model used for visual features
        preprocess:   preprocessing transform for VLA CLIP
    """
    print(f"[INFO] Loading VLA policy from {ckpt_path}...")

    # CLIP backbone for VLA (same config as training)
    if clip_model is None:
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            clip_model_name,
            pretrained=clip_pretrained,
        )
        clip_model = clip_model.to(device)
        clip_model.eval()

    visual_dim = clip_model.visual.output_dim
    quality_dim = 10  # fixed in VLA code
    num_actions = len(VLAAction)

    policy_model = VLAPolicy(
        visual_dim=visual_dim,
        quality_dim=quality_dim,
        num_actions=num_actions,
    )

    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)

    policy_model.load_state_dict(state)
    policy_model.to(device)
    policy_model.eval()

    print(
        f"[INFO] VLA loaded: visual_dim={visual_dim}, "
        f"quality_dim={quality_dim}, num_actions={num_actions}"
    )
    return policy_model, clip_model, preprocess

# ---------------------------
# Text query dataset & embeddings
# ---------------------------
class TextQueryDataset(torch.utils.data.Dataset):
    """
    Dataset for text queries in <split>_image_text.jsonl.

    Each item:
        - raw text string
        - sku index (global index based on sku2idx)
    We typically filter to domain == "query" so that this represents
    user-like queries.
    """

    def __init__(
        self,
        val_image_text: Path,
        sku2idx: Dict[str, int],
        use_domain_filter: bool = True,
        domain_value: str = "query",
    ) -> None:
        super().__init__()
        self.samples: List[Tuple[str, int]] = []

        with val_image_text.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)

                if use_domain_filter:
                    if rec.get("domain", "catalog") != domain_value:
                        continue

                raw_text = rec.get("text", "")
                if not raw_text:
                    continue

                text = raw_text.strip()
                if not text:
                    continue

                sku_id = rec["sku_id"]
                idx = sku2idx.get(sku_id)
                if idx is None:
                    continue

                self.samples.append((text, idx))

        print(
            f"[INFO] TextQueryDataset: loaded {len(self.samples)} text samples "
            f"from {val_image_text}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        text, sku_idx = self.samples[i]
        return text, sku_idx


def text_query_collate_fn(batch):
    """
    Collate function for TextQueryDataset.

    batch: list of (text, sku_idx)
    returns:
        texts:  list[str]
        labels: LongTensor of shape (B,)
    """
    texts = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return texts, labels

@torch.no_grad()
def compute_text_query_embeddings(
    clip_model,
    tokenizer,
    val_image_text: Path,
    sku2idx: Dict[str, int],
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 4,
):
    """
    Compute CLIP text embeddings for validation text queries.

    We build a Dataset/DataLoader so that JSON parsing and batching
    are handled efficiently (similar to the image pipeline).
    """
    dataset = TextQueryDataset(
        val_image_text=val_image_text,
        sku2idx=sku2idx,
        use_domain_filter=True,
        domain_value="query",
    )

    if len(dataset) == 0:
        print("[WARN] No text queries found in validation file.")
        return torch.empty(0, 0), torch.empty(0, dtype=torch.long)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=text_query_collate_fn,
    )

    clip_model.eval()
    clip_model.to(device)

    all_embs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for texts, labels in tqdm(loader, desc="Embed text queries"):
        # tokenize in batch, same as image pipeline doing preprocess() in batch
        tokens = tokenizer(texts).to(device, non_blocking=True)
        feats = clip_model.encode_text(tokens)  # (B, D)
        feats = feats / feats.norm(dim=-1, keepdim=True)

        all_embs.append(feats.cpu())
        all_labels.append(labels.clone())

    embs = torch.cat(all_embs, dim=0)          # (Nq, D)
    labels_t = torch.cat(all_labels, dim=0)    # (Nq,)
    return embs, labels_t

# ---------------------------
# Helpers: compute query embeddings (image)
# ---------------------------
@torch.no_grad()
def compute_query_embeddings_clip(
    model: ClipSkuBaseline,
    dataset: DeepFashion2ImageSkuEvalDataset,
    batch_size: int,
    device: torch.device,
    num_workers: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Similar to compute_embeddings_clip in train_clip_sku_df2.py,
    but only used for query-side embeddings.
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
    for imgs, labels, sku_ids, _dom in tqdm(loader, desc="Embed queries"):
        imgs = imgs.to(device, non_blocking=True)
        emb = model.encode_image(imgs)  # (B, D)
        all_embs.append(emb.cpu())
        all_labels.append(labels.clone())
        all_sku_ids.extend(list(sku_ids))

    embs = torch.cat(all_embs, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return embs, labels, all_sku_ids

# ---------------------------
# Helpers: compute query embeddings with VLA-enhanced images
# ---------------------------
@torch.no_grad()
def compute_query_embeddings_clip_with_vla(
    sku_root: Path,
    val_image_text: Path,
    sku2idx: Dict[str, int],
    clip_sku_model: ClipSkuBaseline,
    sku_preprocess,
    vla_policy: VLAPolicy,
    vla_clip_model,
    vla_preprocess,
    batch_size: int,
    device: torch.device,
    num_workers: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Two-stage version (faster):

      Stage 1 (parallel, CPU-heavy):
        - read query records from JSONL into a list of (img_path, sku_idx)
        - DataLoader with num_workers>0:
            * open image
            * vla_preprocess(img)
            * compute_quality_features(img)
        - run VLA to decide action_id for each image

      Stage 2 (simpler, GPU-bound):
        - reopen images
        - apply ACTION_FUNCS[action_id]
        - sku_preprocess(processed_img)
        - encode with CLIP-SKU

    Returns:
        embs:   (Nq, D) tensor of query embeddings
        labels: (Nq,)  LongTensor of global SKU indices (same index space as sku2idx)
    """

    # ---------- Collect all query samples (paths + labels) ----------
    samples: List[Tuple[Path, int]] = []
    with val_image_text.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("domain", "catalog") != "query":
                continue

            sku_id = rec.get("sku_id")
            img_rel = rec.get("image_path")
            if not sku_id or not img_rel:
                continue

            idx = sku2idx.get(sku_id)
            if idx is None:
                continue

            img_path = sku_root / img_rel
            if not img_path.exists():
                continue

            samples.append((img_path, idx))

    print(f"[INFO] VLA-Query: collected {len(samples)} image queries from {val_image_text}")
    if len(samples) == 0:
        return torch.empty(0, 0), torch.empty(0, dtype=torch.long)

    # ---------- Stage 1: precompute VLA actions with DataLoader ----------
    class VLAActionDataset(torch.utils.data.Dataset):
        def __init__(self, samples, vla_preprocess):
            self.samples = samples
            self.vla_preprocess = vla_preprocess

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, i: int):
            img_path, sku_idx = self.samples[i]
            img = Image.open(img_path).convert("RGB")
            # vla image tensor
            vla_input = self.vla_preprocess(img)
            # quality features (cv2-heavy)
            qfeat = torch.tensor(
                compute_quality_features(img),
                dtype=torch.float32,
            )
            img.close()
            return vla_input, qfeat, sku_idx, str(img_path)

    vla_dataset = VLAActionDataset(samples, vla_preprocess)
    vla_loader = DataLoader(
        vla_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    vla_policy.eval().to(device)
    vla_clip_model.eval().to(device)

    # mapping: img_path(str) -> action_id(int)
    path_to_action: Dict[str, int] = {}
    print("[INFO] Stage 1: precomputing VLA actions...")
    for vla_inputs, qfeats, _sku_labels, img_paths in tqdm(
        vla_loader, desc="VLA policy (stage 1)"
    ):
        vla_inputs = vla_inputs.to(device, non_blocking=True)
        qfeats = qfeats.to(device, non_blocking=True)

        vfeats = vla_clip_model.encode_image(vla_inputs)
        vfeats = vfeats / vfeats.norm(dim=-1, keepdim=True)

        logits = vla_policy(vfeats, qfeats)  # (B, num_actions)
        action_ids = logits.argmax(dim=-1).cpu().tolist()

        for p, a in zip(img_paths, action_ids):
            path_to_action[p] = a

    assert len(path_to_action) == len(samples), \
        f"path_to_action ({len(path_to_action)}) != samples ({len(samples)})"

    # ---------- Stage 2: apply VLA actions + encode with CLIP-SKU ----------
    clip_sku_model.eval().to(device)

    all_embs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    num_samples = len(samples)
    print("[INFO] Stage 2: encoding VLA-enhanced queries with CLIP-SKU...")
    for start in tqdm(range(0, num_samples, batch_size), desc="VLA-enhanced query encoding (stage 2)"):
        batch = samples[start : start + batch_size]
        paths, labels = zip(*batch)

        # load + apply chosen VLA action
        processed_imgs: List[Image.Image] = []
        for p in paths:
            p_str = str(p)
            action_id = path_to_action[p_str]
            action = VLAAction(action_id)

            # Use a context manager and COPY the processed image
            with Image.open(p).convert("RGB") as img:
                proc_img = ACTION_FUNCS[action](img)
                proc_img = proc_img.copy()   # <-- detach from the underlying file

            processed_imgs.append(proc_img)

        # encode with CLIP-SKU
        sku_inputs = torch.stack(
            [sku_preprocess(img) for img in processed_imgs], dim=0
        ).to(device, non_blocking=True)

        for img in processed_imgs:
            img.close()

        emb = clip_sku_model.encode_image(sku_inputs)  # (B, D)
        all_embs.append(emb.cpu())
        all_labels.append(torch.tensor(labels, dtype=torch.long))

    embs = torch.cat(all_embs, dim=0)
    labels_t = torch.cat(all_labels, dim=0)
    return embs, labels_t

# ---------------------------
# Helpers: val catalog embeddings → per-SKU lists
# ---------------------------
class ImageTextEmbedDataset(torch.utils.data.Dataset):
    """
    Dataset that reads <split>_image_text.jsonl and returns image tensors
    plus (sku_idx, is_query) flags.

    It is used only to precompute CLIP image embeddings and split them into:
        - view_embs[s]: catalog / multiview views for SKU s
        - img_query_embs[s]: query images for SKU s (not used here)
    """

    def __init__(
        self,
        sku_root: Path,
        jsonl_path: Path,
        preprocess,
        sku2idx: Dict[str, int],
    ) -> None:
        super().__init__()
        self.sku_root = sku_root
        self.preprocess = preprocess
        self.records: List[Tuple[Path, int, bool]] = []

        with jsonl_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    print(f"line is empty.")
                    continue
                rec = json.loads(line)

                sku_id = rec["sku_id"]
                idx = sku2idx.get(sku_id)
                if idx is None:
                    continue

                img_rel = rec["image_path"]
                img_path = sku_root / img_rel
                if not img_path.exists():
                    print(f"{img_path} does not exist.")
                    continue

                domain = rec.get("domain", "catalog")
                is_query = domain == "query"

                self.records.append((img_path, idx, is_query))

        print(f"[INFO] ImageTextEmbedDataset: loaded {len(self.records)} records from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        img_path, sku_idx, is_query = self.records[i]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.preprocess(img)
        img.close()
        return img_tensor, sku_idx, is_query


@torch.no_grad()
def precompute_val_views(
    sku_root: Path,
    val_image_text: Path,
    clip_model: ClipSkuBaseline,
    preprocess,
    sku2idx: Dict[str, int],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> List[List[torch.Tensor]]:
    """
    Precompute validation catalog view embeddings:

        view_embs_val[s] = list of (D,) tensors for SKU s.

    Only images whose domain != 'query' are treated as catalog views.
    """
    num_skus = len(sku2idx)
    view_embs: List[List[torch.Tensor]] = [[] for _ in range(num_skus)]

    dataset = ImageTextEmbedDataset(
        sku_root=sku_root,
        jsonl_path=val_image_text,
        preprocess=preprocess,
        sku2idx=sku2idx,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    clip_model.eval()
    clip_model.to(device)

    for imgs, sku_idx_batch, is_query_batch in tqdm(
        loader, desc="Precompute val catalog views"
    ):
        imgs = imgs.to(device, non_blocking=True)
        sku_idx_batch = sku_idx_batch.to(device, non_blocking=True)
        is_query_batch = is_query_batch.to(device)

        feats = clip_model.encode_image(imgs)  # (B, D)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats = feats.cpu()

        B = feats.size(0)
        for b in range(B):
            idx = int(sku_idx_batch[b].item())
            if bool(is_query_batch[b].item()):
                # Validation user queries are not part of the gallery fingerprint.
                continue
            view_embs[idx].append(feats[b])

    num_without_views = sum(1 for vs in view_embs if len(vs) == 0)
    print(f"[INFO] In validation, SKUs with NO catalog views: {num_without_views}")
    return view_embs

def build_sku2idx_from_jsonl(jsonl_path: Path) -> Dict[str, int]:
    """
    Build sku2idx mapping from all sku_id values that appear in a given
    <split>_image_text*.jsonl file.

    Used when --eval_all_val_skus is enabled so that evaluation covers
    all SKUs present in the validation split (including unseen SKUs).
    """
    sku_set = set()
    with jsonl_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sku_id = rec.get("sku_id")
            if sku_id is not None:
                sku_set.add(sku_id)

    sku_list = sorted(sku_set)
    print(
        f"[INFO] Eval sku2idx: collected {len(sku_list)} SKUs from {jsonl_path}"
    )
    return {sid: i for i, sid in enumerate(sku_list)}


# [DEMO] Build mapping from SKU index -> catalog image paths
def build_sku_to_catalog_paths(
    sku_root: Path,
    val_image_text: Path,
    sku2idx: Dict[str, int],
) -> Dict[int, List[Path]]:
    """
    Build a mapping from global SKU index -> list of catalog image paths.

    Used only for qualitative demos (image/text).
    """
    sku_to_paths: Dict[int, List[Path]] = {idx: [] for idx in sku2idx.values()}

    with val_image_text.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("domain", "catalog") == "query":
                continue

            sku_id = rec.get("sku_id")
            img_rel = rec.get("image_path")
            if not sku_id or not img_rel:
                continue

            idx = sku2idx.get(sku_id)
            if idx is None:
                continue

            img_path = sku_root / img_rel
            if not img_path.exists():
                continue

            sku_to_paths[idx].append(img_path)

    num_with_catalog = sum(1 for v in sku_to_paths.values() if len(v) > 0)
    print(
        f"[INFO] Catalog image mapping built for {num_with_catalog} SKUs "
        f"(out of {len(sku2idx)})"
    )
    return sku_to_paths


# ---------------------------
# Helpers: aggregate val catalog views with student
# ---------------------------
@torch.no_grad()
def build_student_fingerprints(
    student: SkuFingerprintStudent,
    view_embs: List[List[torch.Tensor]],
    device: torch.device,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Given per-SKU view_embs[s] (catalog views for validation),
    run the student aggregator once per SKU:

        z_s = g_theta({v_s,i})

    Returns:
        gallery_embs: (Neval, D) L2-normalized fingerprints
        valid_indices: list of original SKU indices that had at least one view
    """
    student.eval()
    student.to(device)

    gallery_list: List[torch.Tensor] = []
    valid_indices: List[int] = []

    print("[INFO] Building student fingerprints on validation catalog views...")

    for old_idx, vs in enumerate(tqdm(view_embs, desc="Student fingerprints")):
        if len(vs) == 0:
            continue  # This SKU has no catalog views in validation.

        views_tensor = torch.stack(vs, dim=0).unsqueeze(0)  # (1, V, D)
        views_tensor = views_tensor.to(device)
        mask = torch.ones(1, views_tensor.size(1), dtype=torch.bool, device=device) #(1, V)

        z = student(views_tensor, mask)  # (1, D)
        z = F.normalize(z, dim=-1)
        gallery_list.append(z.squeeze(0).cpu())
        valid_indices.append(old_idx)

    if len(gallery_list) == 0:
        raise RuntimeError("No SKU has catalog views in validation; cannot evaluate.")

    gallery_embs = torch.stack(gallery_list, dim=0)  # (Neval, D)
    return gallery_embs.to(device), valid_indices


# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # init wandb (optional)
    wandb_run = None
    if args.wandb_project is not None:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    # 1) Load student checkpoint and rebuild aggregator
    try:
        s_ckpt = torch.load(args.student_ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        s_ckpt = torch.load(args.student_ckpt, map_location="cpu")

    sku_ids: List[str] = s_ckpt["sku_ids"]
    embed_dim: int = int(s_ckpt["embed_dim"])
    hparams = s_ckpt.get("hparams", {})

    hidden_dim = int(hparams.get("hidden_dim", 512))
    num_layers = int(hparams.get("num_layers", 2))
    num_heads = int(hparams.get("num_heads", 8))

    print(f"[INFO] Student ckpt loaded from {args.student_ckpt}")
    print(
        f"[INFO] embed_dim={embed_dim}, hidden_dim={hidden_dim}, "
        f"num_layers={num_layers}, num_heads={num_heads}"
    )
    print(f"[INFO] Number of SKUs in student ckpt: {len(sku_ids)}")

    student = SkuFingerprintStudent(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
    )
    student.load_state_dict(s_ckpt["model_state"], strict=True)

    # 2) Load CLIP-SKU teacher (baseline5) as encoder
    clip_model, preprocess = load_clip_sku_model(
        device=device,
        clip_model_name=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        ckpt_path=args.clip_sku_ckpt,
    )
    print(f"[INFO] Loaded CLIP-SKU teacher from {args.clip_sku_ckpt}")

    # 3) Resolve validation image_text file
    if args.val_image_text is not None:
        val_image_text = args.val_image_text
    else:
        suffix = args.image_text_suffix or ""
        if suffix and not suffix.startswith("."):
            suffix = "." + suffix
        val_image_text = args.sku_root / f"{args.val_split}_image_text{suffix}.jsonl"

    if not val_image_text.is_file():
        raise FileNotFoundError(f"Validation image_text file not found: {val_image_text}")

    print(f"[INFO] Using validation image_text: {val_image_text}")

    # 3.5) Decide SKU universe for evaluation
    if args.eval_all_val_skus:
        # new mode: use validation jsonl SKU
        sku2idx: Dict[str, int] = build_sku2idx_from_jsonl(val_image_text)
        num_skus_total = len(sku2idx)
        print(
            "[INFO] SKU universe = ALL SKUs appearing in validation image_text "
            f"(including unseen SKUs); num_skus_total={num_skus_total}"
        )
    else:
        # old mode：only student ckpt SKU
        sku2idx = {sku_id: i for i, sku_id in enumerate(sku_ids)}
        num_skus_total = len(sku2idx)
        print(
            "[INFO] SKU universe = SKUs stored in student checkpoint only "
            f"(seen-SKU evaluation); num_skus_total={num_skus_total}"
        )

    # idx2sku for recovering sku_id strings from indices
    idx2sku: List[str] = [None] * num_skus_total
    for sid, idx in sku2idx.items():
        idx2sku[idx] = sid

    # Optionally load VLA policy for query enhancement
    vla_policy = None
    vla_clip_model = None
    vla_preprocess = None
    if args.use_vla:
        if args.vla_checkpoint is None:
            raise ValueError("--use_vla was set but --vla_checkpoint is None.")
        print(f"[INFO] VLA enhancement enabled. Checkpoint: {args.vla_checkpoint}")
        vla_policy, vla_clip_model, vla_preprocess = load_vla_model(
            device=device,
            clip_model_name=args.clip_model,
            clip_pretrained=args.clip_pretrained,
            ckpt_path=args.vla_checkpoint,
            clip_model=clip_model.clip_model,
            preprocess=preprocess,
        )

    # [DEMO] Precompute catalog image paths if any demo is requested
    sku_to_catalog_paths: Dict[int, List[Path]] = {}
    if args.demo_image_num > 0 or args.demo_text_num > 0:
        sku_to_catalog_paths = build_sku_to_catalog_paths(
            sku_root=args.sku_root,
            val_image_text=val_image_text,
            sku2idx=sku2idx,
        )

    # 4) Precompute validation catalog view embeddings (CLIP space)
    view_embs_val = precompute_val_views(
        sku_root=args.sku_root,
        val_image_text=val_image_text,
        clip_model=clip_model,
        preprocess=preprocess,
        sku2idx=sku2idx,
        device=device,
        batch_size=args.precompute_sku_batch_size,
        num_workers=args.num_workers,
    )

    # 5) Aggregate per-SKU catalog views using the student → validation fingerprints
    gallery_embs, valid_indices = build_student_fingerprints(
        student=student,
        view_embs=view_embs_val,
        device=device,
    )
    num_eval_skus = len(valid_indices)
    print(f"[INFO] Number of SKUs with at least one catalog view in val: {num_eval_skus}")

    # Map original (global) SKU index → local gallery index [0..Neval-1]
    old2new = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_indices)}

    # Gallery labels are simply 0..Neval-1
    gallery_labels = torch.arange(num_eval_skus, dtype=torch.long, device=device)

    # 6) Build query dataset and compute query embeddings (CLIP-SKU)
    if args.use_vla:
        print("[INFO] Computing query embeddings with VLA-enhanced images.")
        query_embs, query_labels_global = compute_query_embeddings_clip_with_vla(
            sku_root=args.sku_root,
            val_image_text=val_image_text,
            sku2idx=sku2idx,
            clip_sku_model=clip_model,
            sku_preprocess=preprocess,
            vla_policy=vla_policy,
            vla_clip_model=vla_clip_model,
            vla_preprocess=vla_preprocess,
            batch_size=args.batch_size,
            device=device,
            num_workers=args.num_workers,
        )
        print(
            f"[INFO] Validation query images (VLA mode): "
            f"{query_embs.size(0)}"
        )
    else:
        query_ds = DeepFashion2ImageSkuEvalDataset(
            sku_root=args.sku_root,
            jsonl_path=val_image_text,
            preprocess=preprocess,
            sku2idx=sku2idx,
            domain_filter="query",
        )
        print(f"[INFO] Validation query images: {len(query_ds)}")

        query_embs, query_labels_global, _ = compute_query_embeddings_clip(
            model=clip_model,
            dataset=query_ds,
            batch_size=args.batch_size,
            device=device,
            num_workers=args.num_workers,
        )

    # Keep only queries whose SKU has a student fingerprint
    num_queries_total = int(query_labels_global.shape[0])

    keep_mask = []
    mapped_labels = []
    for lbl in query_labels_global:
        old_idx = int(lbl.item())
        if old_idx in old2new:
            keep_mask.append(True)
            mapped_labels.append(old2new[old_idx])
        else:
            keep_mask.append(False)

    keep_mask = torch.tensor(keep_mask, dtype=torch.bool)
    query_embs = query_embs[keep_mask].to(device)
    query_labels = torch.tensor(mapped_labels, dtype=torch.long, device=device)

    print(
        f"[INFO] Kept {query_embs.size(0)} query images whose SKUs have "
        f"validation fingerprints (out of {num_queries_total})."
    )

    # 7) Compute retrieval metrics (Recall@K, NDCG@K, MRR, latency)
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


    metrics_img = compute_metrics(
        gallery_embs=gallery_embs,
        gallery_labels=gallery_labels,
        query_embs=query_embs,
        query_labels=query_labels,
        ndcg_k=args.ndcg_k,
        recall_ks=tuple(args.recall_ks),
    )

    print("=== Student Fingerprint Evaluation (validation image→SKU) ===")
    for k, v in metrics_img.items():
        if "latency" in k:
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v:.4f}")

    # ----- text-based evaluation (new) -----
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    text_query_embs, text_query_labels_global = compute_text_query_embeddings(
        clip_model=clip_model,
        tokenizer=tokenizer,
        val_image_text=val_image_text,
        sku2idx=sku2idx,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    if text_query_embs.numel() > 0:
        # keep only text queries whose SKU has a gallery fingerprint
        keep_mask = []
        mapped_labels = []
        for lbl in text_query_labels_global:
            old_idx = int(lbl.item())
            if old_idx in old2new:
                keep_mask.append(True)
                mapped_labels.append(old2new[old_idx])
            else:
                keep_mask.append(False)

        keep_mask = torch.tensor(keep_mask, dtype=torch.bool)
        text_query_embs = text_query_embs[keep_mask].to(device)
        text_query_labels = torch.tensor(mapped_labels, dtype=torch.long, device=device)

        print(
            f"[INFO] Kept {text_query_embs.size(0)} text queries whose SKUs have "
            f"validation fingerprints."
        )

        if args.eval_cpu_latency:
            text_query_embs = text_query_embs.to('cpu')
            text_query_labels = text_query_labels.to('cpu')
        else:
            text_query_embs = text_query_embs.to('cuda')
            text_query_labels = text_query_labels.to('cuda')

        metrics_text = compute_metrics(
            gallery_embs=gallery_embs,
            gallery_labels=gallery_labels,
            query_embs=text_query_embs,
            query_labels=text_query_labels,
            ndcg_k=args.ndcg_k,
            recall_ks=tuple(args.recall_ks),
        )

        print("=== Student Fingerprint Evaluation (text→SKU) ===")
        for k, v in metrics_text.items():
            if "latency" in k:
                print(f"text/{k}: {v:.2f}")
            else:
                print(f"text/{k}: {v:.4f}")
    else:
        metrics_text = {}

    # log both to wandb if enabled
    if wandb.run is not None:
        wandb.log({f"eval_image/{k}": float(v) for k, v in metrics_img.items()})
        wandb.log({f"eval_text/{k}": float(v) for k, v in metrics_text.items()})

    # ---------------------------
    # [DEMO] Image-query demo
    # ---------------------------
    if args.demo_image_num > 0:
        if args.demo_image_output_dir is None:
            print("[WARN] demo_image_num > 0 but no demo_image_output_dir was provided; skipping image demo.")
        else:
            out_dir = args.demo_image_output_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[DEMO] Writing image-query demo examples to {out_dir}")

            # Collect candidate image queries from JSONL whose SKUs have fingerprints
            image_query_records: List[Tuple[Path, int, str]] = []
            with val_image_text.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec.get("domain", "catalog") != "query":
                        continue
                    sku_id = rec.get("sku_id")
                    img_rel = rec.get("image_path")
                    if not sku_id or not img_rel:
                        continue
                    global_idx = sku2idx.get(sku_id)
                    if global_idx is None:
                        continue
                    if global_idx not in old2new:
                        continue
                    img_path = args.sku_root / img_rel
                    if not img_path.exists():
                        continue
                    image_query_records.append((img_path, global_idx, sku_id))

            if len(image_query_records) == 0:
                print("[DEMO] No eligible image queries found for demo; skipping.")
            else:
                num_samples = min(args.demo_image_num, len(image_query_records))
                sampled = random.sample(image_query_records, num_samples)
                gallery_device = gallery_embs.device

                # [DEMO][PROGRESS] tqdm for image demo loop
                for i, (img_path, global_idx_gt, sku_id_gt) in enumerate(
                    tqdm(sampled, desc="DEMO image queries", total=num_samples)
                ):
                    # encode query image
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = preprocess(img).unsqueeze(0).to(device)
                    img.close()

                    with torch.no_grad():
                        q = clip_model.encode_image(img_tensor)
                        q = F.normalize(q, dim=-1)
                        q = q.to(gallery_device)
                        scores = torch.matmul(q, gallery_embs.T).squeeze(0)
                        top_idx = int(scores.argmax().item())

                    pred_global_idx = valid_indices[top_idx]
                    pred_sku_id = idx2sku[pred_global_idx]

                    # Save query image
                    query_out = out_dir / f"demo_img_{i:03d}_query_gt-{sku_id_gt}_top1-{pred_sku_id}.jpg"
                    img = Image.open(img_path).convert("RGB")
                    img.save(query_out)
                    img.close()

                    # Save all catalog images for predicted SKU
                    cat_paths = sku_to_catalog_paths.get(pred_global_idx, [])
                    for j, p in enumerate(cat_paths):
                        if not p.exists():
                            continue
                        catalog_out = out_dir / f"demo_img_{i:03d}_top1-{pred_sku_id}_catalog_{j:02d}.jpg"
                        shutil.copy2(p, catalog_out)

    # ---------------------------
    # [DEMO] Text-query demo
    # ---------------------------
    if args.demo_text_num > 0:
        if args.demo_text_output_dir is None:
            print("[WARN] demo_text_num > 0 but no demo_text_output_dir was provided; skipping text demo.")
        else:
            out_dir = args.demo_text_output_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[DEMO] Writing text-query demo examples to {out_dir}")

            # Collect candidate text queries from JSONL whose SKUs have fingerprints
            text_query_records: List[Tuple[str, int, str]] = []
            with val_image_text.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec.get("domain", "catalog") != "query":
                        continue
                    raw_text = rec.get("text", "")
                    text = (raw_text or "").strip()
                    if not text:
                        continue
                    sku_id = rec.get("sku_id")
                    if not sku_id:
                        continue
                    global_idx = sku2idx.get(sku_id)
                    if global_idx is None:
                        continue
                    if global_idx not in old2new:
                        continue
                    text_query_records.append((text, global_idx, sku_id))

            if len(text_query_records) == 0:
                print("[DEMO] No eligible text queries found for demo; skipping.")
            else:
                num_samples = min(args.demo_text_num, len(text_query_records))
                sampled = random.sample(text_query_records, num_samples)
                gallery_device = gallery_embs.device

                # [DEMO][PROGRESS] tqdm for text demo loop
                for i, (text, global_idx_gt, sku_id_gt) in enumerate(
                    tqdm(sampled, desc="DEMO text queries", total=num_samples)
                ):
                    # encode query text
                    tokens = tokenizer([text]).to(device)
                    with torch.no_grad():
                        q = clip_model.encode_text(tokens)
                        q = F.normalize(q, dim=-1)
                        q = q.to(gallery_device)
                        scores = torch.matmul(q, gallery_embs.T).squeeze(0)
                        top_idx = int(scores.argmax().item())

                    pred_global_idx = valid_indices[top_idx]
                    pred_sku_id = idx2sku[pred_global_idx]

                    # Save query text as .txt
                    txt_out = out_dir / f"demo_text_{i:03d}_query_gt-{sku_id_gt}_top1-{pred_sku_id}.txt"
                    with txt_out.open("w", encoding="utf-8") as f_txt:
                        f_txt.write(f"query text: {text}\n")

                    # Save catalog images for predicted SKU
                    cat_paths = sku_to_catalog_paths.get(pred_global_idx, [])
                    for j, p in enumerate(cat_paths):
                        if not p.exists():
                            continue
                        catalog_out = out_dir / f"demo_text_{i:03d}_top1-{pred_sku_id}_catalog_{j:02d}.jpg"
                        shutil.copy2(p, catalog_out)
    wandb.finish()


if __name__ == "__main__":
    # Disable autograd during evaluation
    torch.set_grad_enabled(False)

    main()
