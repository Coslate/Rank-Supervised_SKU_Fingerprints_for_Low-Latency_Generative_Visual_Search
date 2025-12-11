from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import open_clip
import wandb
import random

from models.clip_sku_baseline import ClipSkuBaseline
from models.sku_fingerprint_student import SkuFingerprintStudent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 3: Train SKU fingerprint student aggregator g_theta."
    )
    parser.add_argument(
        "--sku_root",
        type=Path,
        required=True,
        help="DeepFashion2_SKU root (contains *_image_text*.jsonl).",
    )
    parser.add_argument(
        "--train_image_text",
        type=Path,
        default=None,
        help=(
            "Path to train_image_text JSONL. "
            "Recommended to use the version containing DiT views, "
            "e.g., train_image_text.dit_clipsku_sub3k_nv4.jsonl. "
            "If None, uses <sku_root>/train_image_text.jsonl."
        ),
    )
    parser.add_argument(
        "--teacher_npz",
        type=Path,
        required=True,
        help=(
            "NPZ with best-shot teacher embeddings. "
            "Expected keys: teacher_embs, sku_ids, and optionally sku_text_embs."
        ),
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-B-16",
        help="open_clip model name (e.g. ViT-B-16).",
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
        help=(
            "Fine-tuned CLIP-SKU checkpoint from train_clip_sku_df2.py "
            "(e.g., baseline5_ft_bestshot_*.pt). "
            "We load it and use encode_image/text() as teacher encoders."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of SKUs per batch.",
    )
    parser.add_argument(
        "--precompute_sku_batch_size",
        type=int,
        default=128,
        help="Number of sku to precompute embedding per batch.",
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
        default=3e-4,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Final cosine learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="Warmup steps ratio (0-1).",
    )
    parser.add_argument(
        "--lambda_reg",
        type=float,
        default=1.0,
        help="Weight for image-teacher L2 regression loss.",
    )
    parser.add_argument(
        "--lambda_reg_text",
        type=float,
        default=1.0,
        help="Weight for text-teacher L2 regression loss (if sku_text_embs provided).",
    )
    parser.add_argument(
        "--lambda_rank",
        type=float,
        default=1.0,
        help="Weight for ranking distillation loss.",
    )
    parser.add_argument(
        "--lambda_contrast",
        type=float,
        default=1.0,
        help="Weight for contrastive cross-entropy loss.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Temperature for retrieval logits.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Student Transformer hidden dim.",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Student Transformer num layers.",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
        help="Student Transformer num heads.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--text_query_prob",
        type=float,
        default=0.5,
        help=(
            "If a SKU has both image and text queries, probability of sampling "
            "a *text* query for that SKU during training."
        ),
    )
    parser.add_argument(
        "--checkpoint_out",
        type=Path,
        required=True,
        help="Path to save student checkpoint (.pt).",
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
    return parser.parse_args()


def load_teacher_npz(
    path: Path,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[str]]:
    """
    Load teacher NPZ from Step 2.

    Expected keys:
        - teacher_embs: (Ns, D) image best-shot teacher embeddings
        - sku_ids:      list[str] of SKU IDs
        - sku_text_embs (optional): (Ns, D) text teacher embeddings
    """
    data = np.load(path, allow_pickle=True)
    teacher_img = data["teacher_embs"].astype("float32")  # (Ns, D)
    sku_ids = list(data["sku_ids"])

    sku_ids = [
        s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s)
        for s in sku_ids
    ]

    teacher_img = torch.from_numpy(teacher_img)
    teacher_img = F.normalize(teacher_img, dim=-1)

    teacher_txt: Optional[torch.Tensor] = None
    if "sku_text_embs" in data:
        txt_arr = data["sku_text_embs"].astype("float32")
        teacher_txt = torch.from_numpy(txt_arr)
        teacher_txt = F.normalize(teacher_txt, dim=-1)
        print("[INFO] Loaded sku_text_embs from NPZ.")
    else:
        print("[INFO] No sku_text_embs in NPZ; text teacher regression will be disabled.")

    return teacher_img, teacher_txt, sku_ids


def load_clip_sku_model(
    device: torch.device,
    clip_model_name: str,
    clip_pretrained: str,
    ckpt_path: Path,
):
    """
    Build open_clip backbone, wrap it with ClipSkuBaseline, and load
    your fine-tuned baseline5 checkpoint.

    We then use:
        - model.encode_image() for image embeddings
        - model.encode_text() for text embeddings

    Tokenizer is from open_clip.get_tokenizer().
    """
    raw_clip_model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name, pretrained=clip_pretrained
    )
    tokenizer = open_clip.get_tokenizer(clip_model_name)

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
    model.eval().to(device)

    return model, preprocess, tokenizer


@torch.no_grad()
def encode_image(
    clip_model,
    preprocess,
    image: Image.Image,
    device: torch.device,
) -> torch.Tensor:
    x = preprocess(image).unsqueeze(0).to(device)
    feat = clip_model.encode_image(x)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.squeeze(0).cpu()  # (D,)

class ImageTextEmbedDataset(Dataset):
    """
    Dataset for precomputing CLIP *image* embeddings from a train_image_text JSONL.
    Each item is (image_tensor, sku_idx, is_query_flag).
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
                    continue
                rec = json.loads(line)

                sku_id = rec["sku_id"]
                idx = sku2idx.get(sku_id)
                if idx is None:
                    continue

                img_rel = rec["image_path"]
                img_path = sku_root / img_rel
                if not img_path.exists():
                    continue

                domain = rec.get("domain", "catalog")
                is_query = domain == "query"

                self.records.append((img_path, idx, is_query))

        print(f"[INFO] ImageTextEmbedDataset: {len(self.records)} image records loaded.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        img_path, sku_idx, is_query = self.records[i]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.preprocess(img)
        img.close()
        return img_tensor, sku_idx, is_query


@torch.no_grad()
def precompute_image_embeddings(
    sku_root: Path,
    train_image_text: Path,
    clip_model,
    preprocess,
    sku2idx: Dict[str, int],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]]:
    """
    Precompute *image* embeddings.

    Returns:
        view_embs[s]:      list of catalog / augmented view embeddings for SKU s
        img_query_embs[s]: list of image-query embeddings for SKU s (domain == 'query')
    """
    num_skus = len(sku2idx)
    view_embs: List[List[torch.Tensor]] = [[] for _ in range(num_skus)]
    img_query_embs: List[List[torch.Tensor]] = [[] for _ in range(num_skus)]

    dataset = ImageTextEmbedDataset(
        sku_root=sku_root,
        jsonl_path=train_image_text,
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
        loader, desc="Precompute image CLIP embeddings"
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
                img_query_embs[idx].append(feats[b])
            else:
                view_embs[idx].append(feats[b])

    num_without_views = sum(1 for vs in view_embs if len(vs) == 0)
    print(f"[INFO] SKUs with no views: {num_without_views}")

    return view_embs, img_query_embs


@torch.no_grad()
def precompute_text_queries(
    train_image_text: Path,
    clip_model,
    tokenizer,
    sku2idx: Dict[str, int],
    device: torch.device,
    num_skus: int,
    batch_size: int = 256,
) -> List[List[torch.Tensor]]:
    """
    Encode `text` fields with CLIP text encoder and group per-SKU.

    Returns:
        txt_query_embs[s]: list of text-query embeddings for SKU s
    """
    txt_query_embs: List[List[torch.Tensor]] = [[] for _ in range(num_skus)]

    texts: List[str] = []
    sku_indices: List[int] = []

    with train_image_text.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sku_id = rec["sku_id"]
            idx = sku2idx.get(sku_id)
            if idx is None:
                continue

            text = rec.get("text", "")
            if not text:
                continue

            text = text.strip()
            if not text:
                continue

            texts.append(text)
            sku_indices.append(idx)

    if not texts:
        print("[INFO] No text fields found in train_image_text.jsonl; skip text queries.")
        return txt_query_embs

    print(f"[INFO] Encoding {len(texts)} text queries with CLIP text encoder...")

    clip_model.eval()
    clip_model.to(device)

    for start in tqdm(
        range(0, len(texts), batch_size),
        desc="Precompute text CLIP embeddings",
    ):
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]
        batch_idx = sku_indices[start:end]

        tokens = tokenizer(batch_texts).to(device)
        feats = clip_model.encode_text(tokens)  # (B, D)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats = feats.cpu()

        B = feats.size(0)
        for b in range(B):
            idx = int(batch_idx[b])
            txt_query_embs[idx].append(feats[b])

    print("[INFO] Text queries grouped into txt_query_embs.")
    return txt_query_embs


class SkuFingerprintDataset(Dataset):
    def __init__(
        self,
        view_embs: List[List[torch.Tensor]],
        img_query_embs: List[List[torch.Tensor]],
        txt_query_embs: List[List[torch.Tensor]],
        teacher_img_embs: torch.Tensor,
        teacher_txt_embs: Optional[torch.Tensor] = None,
        text_query_prob: float = 0.5,
    ) -> None:
        """
        text_query_prob: If a SKU has both image and text queries, probability
        of sampling a text query for that SKU during training.
        """
        super().__init__()
        self.view_embs = view_embs
        self.img_query_embs = img_query_embs
        self.txt_query_embs = txt_query_embs
        self.teacher_img_embs = teacher_img_embs  # (Ns, D)
        self.teacher_txt_embs = teacher_txt_embs  # (Ns, D) or None
        self.num_skus = len(view_embs)
        assert self.teacher_img_embs.shape[0] == self.num_skus

        if self.teacher_txt_embs is not None:
            assert self.teacher_txt_embs.shape == self.teacher_img_embs.shape

        self.text_query_prob = text_query_prob

    def __len__(self) -> int:
        return self.num_skus

    def __getitem__(self, idx: int):
        views_list = self.view_embs[idx]
        if len(views_list) == 0:
            # Should not happen if filtered beforehand
            raise RuntimeError(f"SKU idx {idx} has no views")

        views = torch.stack(views_list, dim=0)  # (V, D)

        img_qs = self.img_query_embs[idx]
        txt_qs = self.txt_query_embs[idx]

        # query_type: 0=image, 1=text, -1=none
        query_type = -1

        if len(img_qs) > 0 and len(txt_qs) > 0:
            # both exist → probabilistic choice
            if random.random() < self.text_query_prob:
                q = txt_qs[np.random.randint(len(txt_qs))]
                query_type = 1
            else:
                q = img_qs[np.random.randint(len(img_qs))]
                query_type = 0
            queries = q.unsqueeze(0)  # (1, D)
        elif len(img_qs) > 0:
            q = img_qs[np.random.randint(len(img_qs))]
            query_type = 0
            queries = q.unsqueeze(0)
        elif len(txt_qs) > 0:
            q = txt_qs[np.random.randint(len(txt_qs))]
            query_type = 1
            queries = q.unsqueeze(0)
        else:
            queries = torch.empty(0, views.size(-1))
            query_type = -1

        teacher_img = self.teacher_img_embs[idx]  # (D,)

        if self.teacher_txt_embs is not None:
            teacher_txt = self.teacher_txt_embs[idx]  # (D,)
            return views, queries, teacher_img, teacher_txt, idx, query_type
        else:
            return views, queries, teacher_img, idx, query_type


def collate_sku_batch(batch):
    """
    Pad per-SKU variable-length views/queries into a batch.

    Returns:
        views:         (B, V_max, D)
        mask:          (B, V_max) bool
        teachers_img:  (B, D)
        teachers_txt:  (B, D) or None
        sku_indices:   (B,)
        queries:       (Nq, D)   # all queries in the batch (image + text)
        query_labels:  (Nq,)     # each query's target SKU index in batch [0..B-1]
        query_types:   (Nq,)       # 0=image, 1=text
    """
    first = batch[0]
    has_text_teacher = len(first) == 6

    batch_size = len(batch)
    view_lens = [b[0].shape[0] for b in batch]
    D = batch[0][0].shape[1]
    V_max = max(view_lens)

    views = torch.zeros(batch_size, V_max, D, dtype=batch[0][0].dtype)
    mask = torch.zeros(batch_size, V_max, dtype=torch.bool)
    teachers_img = []
    teachers_txt = [] if has_text_teacher else None
    sku_indices = []

    query_list = []
    query_labels = []
    query_types = []

    for i, sample in enumerate(batch):
        if has_text_teacher:
            v, q, t_img, t_txt, idx, q_type = sample
        else:
            v, q, t_img, idx, q_type = sample
            t_txt = None

        V = v.shape[0]
        views[i, :V, :] = v
        mask[i, :V] = True
        teachers_img.append(t_img)
        sku_indices.append(idx)

        if has_text_teacher:
            teachers_txt.append(t_txt)

        if q.size(0) > 0:
            query_list.append(q[0])
            query_labels.append(i)
            query_types.append(q_type)

    teachers_img = torch.stack(teachers_img, dim=0)
    sku_indices = torch.tensor(sku_indices, dtype=torch.long)

    if has_text_teacher:
        teachers_txt_tensor = torch.stack(teachers_txt, dim=0)
    else:
        teachers_txt_tensor = None

    if len(query_list) > 0:
        queries = torch.stack(query_list, dim=0)
        query_labels = torch.tensor(query_labels, dtype=torch.long)
        query_types_tensor = torch.tensor(query_types, dtype=torch.long)
    else:
        queries = torch.empty(0, D)
        query_labels = torch.empty(0, dtype=torch.long)
        query_types_tensor = torch.empty(0, dtype=torch.long)

    return (
        views,
        mask,
        teachers_img,
        teachers_txt_tensor,
        sku_indices,
        queries,
        query_labels,
        query_types_tensor,
    )


def adjust_learning_rate(
    optimizer: optim.Optimizer,
    step: int,
    total_steps: int,
    base_lr: float,
    min_lr: float,
    warmup_ratio: float,
):
    """
    Cosine schedule with linear warmup, warmup_ratio=0.05 by default.
    """
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    if step <= warmup_steps:
        lr = base_lr * float(step) / float(warmup_steps)
    else:
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


def train():
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

    sku_root: Path = args.sku_root
    train_image_text = (
        args.train_image_text
        if args.train_image_text is not None
        else sku_root / "train_image_text.jsonl"
    )
    # 1) Load teacher NPZ (image + optional text teacher)
    teacher_img_embs, teacher_txt_embs, sku_ids = load_teacher_npz(args.teacher_npz)
    num_skus, embed_dim = teacher_img_embs.shape
    print(
        f"[INFO] Loaded teacher image embeddings: {teacher_img_embs.shape}, "
        f"SKUs={num_skus}"
    )
    has_text_teacher = teacher_txt_embs is not None


    # 2) Build sku2idx mapping (teacher NPZ order is the canonical order)
    sku2idx: Dict[str, int] = {sku_id: i for i, sku_id in enumerate(sku_ids)}

    # 3) Load fine-tuned baseline5 CLIP-SKU (torch.load)
    clip_model, preprocess, tokenizer = load_clip_sku_model(
        device=device,
        clip_model_name=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        ckpt_path=args.clip_sku_ckpt,
    )
    print("[INFO] Loaded fine-tuned CLIP-SKU checkpoint for embeddings.")

    # 4a) Precompute image view/query embeddings
    view_embs, img_query_embs = precompute_image_embeddings(
        sku_root=sku_root,
        train_image_text=train_image_text,
        clip_model=clip_model,
        preprocess=preprocess,
        sku2idx=sku2idx,
        device=device,
        batch_size=args.precompute_sku_batch_size,
        num_workers=args.num_workers,
    )

    # 4b) Precompute text queries per SKU
    txt_query_embs = precompute_text_queries(
        train_image_text=train_image_text,
        clip_model=clip_model,
        tokenizer=tokenizer,
        sku2idx=sku2idx,
        device=device,
        num_skus=num_skus,
        batch_size=args.precompute_sku_batch_size,
    )

    # Drop SKUs that have no views at all
    valid_indices = [i for i, vs in enumerate(view_embs) if len(vs) > 0]
    if len(valid_indices) < num_skus:
        print(
            f"[WARN] {num_skus - len(valid_indices)} SKUs have no views; "
            f"they will be dropped from training."
        )
        view_embs = [view_embs[i] for i in valid_indices]
        img_query_embs = [img_query_embs[i] for i in valid_indices]
        txt_query_embs = [txt_query_embs[i] for i in valid_indices]
        teacher_img_embs = teacher_img_embs[valid_indices]
        if has_text_teacher:
            teacher_txt_embs = teacher_txt_embs[valid_indices]
        sku_ids = [sku_ids[i] for i in valid_indices]
        num_skus = len(valid_indices)

    # 5) Dataset / DataLoader
    dataset = SkuFingerprintDataset(
        view_embs=view_embs,
        img_query_embs=img_query_embs,
        txt_query_embs=txt_query_embs,
        teacher_img_embs=teacher_img_embs,
        teacher_txt_embs=teacher_txt_embs if has_text_teacher else None,
        text_query_prob=args.text_query_prob,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_sku_batch,
    )

    # 6) Student model + optimizer
    student = SkuFingerprintStudent(
        embed_dim=embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(device)

    optimizer = optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = args.epochs * max(1, len(loader))
    global_step = 0

    # 7) Training loop (with tqdm progress bar)
    for epoch in range(1, args.epochs + 1):
        student.train()
        running_loss = 0.0
        running_reg_img = 0.0
        running_reg_txt = 0.0
        running_rank = 0.0
        running_rank_img = 0.0
        running_rank_txt = 0.0
        running_ce = 0.0
        running_text_ratio = 0.0
        running_img_ratio = 0.0
        steps_with_queries = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for (
            views,
            mask,
            teachers_img,
            teachers_txt,
            _,
            queries,
            q_labels,
            q_types,
        ) in pbar:
            views = views.to(device)
            mask = mask.to(device)
            teachers_img = teachers_img.to(device)
            teachers_img = F.normalize(teachers_img, dim=-1)

            if teachers_txt is not None and has_text_teacher:
                teachers_txt = teachers_txt.to(device)
                teachers_txt = F.normalize(teachers_txt, dim=-1)

            queries = queries.to(device)
            q_labels = q_labels.to(device)
            q_types = q_types.to(device) #(Nq,)

            optimizer.zero_grad()

            # Forward student
            z_student = student(views, mask)  # (B, D)

            # 1) L2 regression loss to image teacher fingerprint
            loss_reg_img = F.mse_loss(z_student, teachers_img)

            # 2) Text-teacher regression (if available)
            if has_text_teacher and teachers_txt is not None:
                loss_reg_txt = F.mse_loss(z_student, teachers_txt)
            else:
                loss_reg_txt = torch.tensor(0.0, device=device)

            # Init ranking / CE losses
            loss_rank = torch.tensor(0.0, device=device)
            loss_rank_img = torch.tensor(0.0, device=device)
            loss_rank_txt = torch.tensor(0.0, device=device)
            loss_ce = torch.tensor(0.0, device=device)
            text_ratio_batch = 0.0

            # 3) Ranking distillation + 4) contrastive CE using queries (image + text)
            if queries.size(0) > 0:
                # logits: similarity of each query to each SKU in the batch
                logits_student = (queries @ z_student.t()) / args.temperature  # (Nq, B)
                with torch.no_grad():
                    logits_teacher = (queries @ teachers_img.t()) / args.temperature
                    prob_teacher = F.softmax(logits_teacher, dim=-1)

                log_prob_student = F.log_softmax(logits_student, dim=-1)
                '''
                # Distillation: KL( teacher || student )
                loss_rank = F.kl_div(
                    log_prob_student, prob_teacher, reduction="batchmean"
                )
                # Contrastive: CE( student_logits, ground-truth SKU index in Batch)
                loss_ce = F.cross_entropy(logits_student, q_labels)
                '''

                # KL per query → (Nq,)
                kl_matrix = prob_teacher * (
                    torch.log(prob_teacher + 1e-8) - log_prob_student
                )
                kl_per_query = kl_matrix.sum(dim=1)
                loss_rank = kl_per_query.mean()

                # split by query type
                img_mask = q_types == 0
                txt_mask = q_types == 1

                if img_mask.any():
                    loss_rank_img = kl_per_query[img_mask].mean()
                if txt_mask.any():
                    loss_rank_txt = kl_per_query[txt_mask].mean()

                # CE per query
                ce_per_query = F.cross_entropy(
                    logits_student, q_labels, reduction="none"
                )
                loss_ce = ce_per_query.mean()

                # text-query ratio
                num_q = q_types.numel()
                if num_q > 0:
                    text_ratio_batch = (q_types == 1).float().mean().item()
                    img_ratio_batch  = (q_types == 0).float().mean().item()

                    running_text_ratio += text_ratio_batch
                    running_img_ratio += img_ratio_batch
                    steps_with_queries += 1
                else:
                    text_ratio_batch = 0.0
                    img_ratio_batch  = 0.0

            # Total loss
            loss = (
                args.lambda_reg * loss_reg_img
                + args.lambda_reg_text * loss_reg_txt
                + args.lambda_rank * loss_rank
                + args.lambda_contrast * loss_ce
            )

            loss.backward()
            optimizer.step()

            global_step += 1
            lr = adjust_learning_rate(
                optimizer=optimizer,
                step=global_step,
                total_steps=total_steps,
                base_lr=args.lr,
                min_lr=args.min_lr,
                warmup_ratio=args.warmup_ratio,
            )

            running_loss += loss.item()
            running_reg_img += loss_reg_img.item()
            running_reg_txt += loss_reg_txt.item()
            running_rank += loss_rank.item()
            running_rank_img += loss_rank_img.item()
            running_rank_txt += loss_rank_txt.item()
            running_ce += loss_ce.item()

            avg_text_ratio = (
                running_text_ratio / max(1, steps_with_queries)
            )  # average over batches with queries

            avg_img_ratio = (
                running_img_ratio / max(1, steps_with_queries)
            )  # average over batches with queries

            pbar.set_postfix(
                loss=f"{running_loss / global_step:.4f}",
                reg_img=f"{running_reg_img / global_step:.4f}",
                reg_txt=f"{running_reg_txt / max(1, global_step):.4f}",
                rank=f"{running_rank / global_step:.4f}",
                rank_img=f"{running_rank_img / max(1, steps_with_queries):.4f}",
                rank_txt=f"{running_rank_txt / max(1, steps_with_queries):.4f}",
                ce=f"{running_ce / global_step:.4f}",
                txtq=f"{avg_text_ratio:.2f}",
                imgq=f"{avg_img_ratio:.2f}",
                lr=f"{lr:.1e}",
            )

            # wandb logging per step
            if wandb_run is not None:
                wandb.log(
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "train/loss": loss.item(),
                        "train/loss_reg_img": loss_reg_img.item(),
                        "train/loss_reg_txt": loss_reg_txt.item(),
                        "train/loss_rank": loss_rank.item(),
                        "train/loss_rank_img": loss_rank_img.item(),
                        "train/loss_rank_txt": loss_rank_txt.item(),
                        "train/loss_ce": loss_ce.item(),
                        "train/text_query_ratio": text_ratio_batch,
                        "train/img_query_ratio": img_ratio_batch,
                        "train/lr": lr,
                    }
                )

    # 8) Save student checkpoint
    ckpt = {
        "model_state": student.state_dict(),
        "teacher_npz": str(args.teacher_npz),
        "sku_ids": sku_ids,
        "embed_dim": embed_dim,
        "hparams": vars(args),
    }
    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.checkpoint_out)
    print(f"[DONE] Saved student checkpoint to {args.checkpoint_out}")

    # finish wandb
    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    train()
