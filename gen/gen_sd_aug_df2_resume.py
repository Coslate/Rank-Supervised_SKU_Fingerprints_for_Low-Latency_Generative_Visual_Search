# gen/gen_dit_aug_df2_resume.py
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Set

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import open_clip
from models.clip_sku_baseline import ClipSkuBaseline
from tqdm import tqdm
from diffusers import (
    StableDiffusionImg2ImgPipeline,
    DiffusionPipeline,
    PixArtAlphaPipeline,
)

import logging
from train.train_sd_lora_df2 import inject_lora_into_unet

logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# ============================================================
# I/O helpers
# ============================================================


def load_sku_metadata(sku_root: Path, split: str) -> Dict:
    """
    Load ORIGINAL {split}_sku_metadata.json from DeepFashion2_SKU root.
    This function never writes to disk; it just returns a dict.
    """
    meta_path = sku_root / f"{split}_sku_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")
    with meta_path.open("r") as f:
        meta = json.load(f)
    return meta


def save_sku_metadata(meta: Dict, out_path: Path) -> None:
    """
    Save augmented SKU metadata to a NEW file (do NOT overwrite original).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"[META] Saved augmented metadata to {out_path}")


def write_augmented_image_text(
    orig_jsonl_path: Path,
    out_jsonl_path: Path,
    new_records: List[Dict],
) -> None:
    """
    Write a new image_text JSONL that contains:
      1) All original lines from orig_jsonl_path
      2) All new augmented records (appended)

    Original file is never modified.
    """
    if not orig_jsonl_path.exists():
        raise FileNotFoundError(f"Missing original image_text JSONL: {orig_jsonl_path}")

    out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    num_orig = 0
    num_new = len(new_records)

    with orig_jsonl_path.open("r") as f_in, out_jsonl_path.open("w") as f_out:
        # Copy original lines
        for line in f_in:
            f_out.write(line)
            num_orig += 1

        # Append new records
        for rec in new_records:
            f_out.write(json.dumps(rec) + "\n")

    print(
        f"[JSONL] Wrote {num_orig} original + {num_new} new records to {out_jsonl_path}"
    )


def iter_catalog_entries(
    meta: Dict,
    sku_root: Path,
    split: str,
    max_skus: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
    skip_processed: Set[Tuple[str, str, int]] | None = None,
) -> Iterator[Tuple[str, Dict, Dict, Path]]:
    """
    Yield (sku_id, sku_info, catalog_entry, abs_path) over ORIGINAL catalog crops.

    We snapshot both the SKU list and each SKU's catalog list so that
    newly-added augmented entries are not iterated again.

    skip_processed: set of (sku_id, image_id, item_idx) that should be skipped
                    (used for resume_from_disk).
    """
    skus = meta["skus"]
    items = list(skus.items())  # snapshot of SKUs

    if shuffle:
        random.seed(seed)
        random.shuffle(items)

    if max_skus is not None:
        items = items[:max_skus]

    for sku_id, info in items:
        # snapshot of the original catalog list for this SKU
        catalog_list = list(info.get("catalog", []))
        for entry in catalog_list:
            # skip synthetic entries
            if entry.get("dit_aug", False):
                continue

            # skip processed entries when resuming
            if skip_processed is not None:
                image_id = str(entry.get("image_id"))
                try:
                    item_idx = int(entry.get("item_idx", -1))
                except Exception:
                    item_idx = -1
                key = (sku_id, image_id, item_idx)
                if key in skip_processed:
                    continue

            crop_rel = Path(entry["crop_path"])
            crop_abs = sku_root / crop_rel
            yield sku_id, info, entry, crop_abs


def build_sku_catalog_text_map(jsonl_path: Path) -> Dict[str, List[str]]:
    """
    Build sku_id -> list of catalog texts from {split}_image_text.jsonl.
    Only domain == "catalog" is used.
    """
    sku2texts: Dict[str, List[str]] = {}
    if not jsonl_path.exists():
        print(f"[WARN] image_text jsonl not found, skipping text map: {jsonl_path}")
        return sku2texts

    with jsonl_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("domain") != "catalog":
                continue
            sku_id = rec["sku_id"]
            text = rec.get("text", "")
            sku2texts.setdefault(sku_id, []).append(text)

    print(f"[TEXT] Loaded catalog texts for {len(sku2texts)} SKUs from {jsonl_path}")
    return sku2texts


# ============================================================
# Color helpers
# ============================================================


def avg_hsv(img: Image.Image) -> np.ndarray:
    """
    Compute average HSV in [0,1]^3.
    """
    arr = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0  # (H,W,3)
    flat = arr.reshape(-1, 3)
    mean = flat.mean(axis=0)
    return mean  # (3,)


def color_distance(hsv1: np.ndarray, hsv2: np.ndarray) -> float:
    return float(np.linalg.norm(hsv1 - hsv2))


def estimate_color_name(hsv: np.ndarray) -> str:
    """
    Very coarse mapping from hue to a color name for counterfactual prompts.
    hsv: (3,) in [0,1]
    """
    h, s, v = hsv.tolist()
    if v < 0.25:
        return "black"
    if s < 0.2 and v > 0.7:
        return "white"

    hue_deg = (h * 360.0) % 360.0
    if 0 <= hue_deg < 30 or 330 <= hue_deg < 360:
        return "red"
    if 30 <= hue_deg < 90:
        return "yellow"
    if 90 <= hue_deg < 150:
        return "green"
    if 150 <= hue_deg < 210:
        return "cyan"
    if 210 <= hue_deg < 270:
        return "blue"
    if 270 <= hue_deg < 330:
        return "purple"
    return "beige"


def pick_different_color(original: str) -> str:
    candidates = [
        "red",
        "yellow",
        "green",
        "cyan",
        "blue",
        "purple",
        "black",
        "white",
        "beige",
    ]
    candidates = [c for c in candidates if c != original]
    return random.choice(candidates) if candidates else original


# ============================================================
# CLIP helpers
# ============================================================


def setup_clip(
    device: torch.device,
    clip_model_name: str,
    clip_pretrained: str,
    clip_sku_ckpt: Path | None = None,
):
    """
    if clip_sku_ckpt is None: use original open_clip weights.
    if clip_sku_ckpt is not None: use our own pretrained train_clip_sku_df2.py trained CLIP-SKU weights.
    only using the image encoder to do the similarity gating.
    """
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name, pretrained=clip_pretrained
    )

    # open pretrained CLIP
    if clip_sku_ckpt is None:
        print(
            f"[CLIP] Using open_clip pretrained model: {clip_model_name} / {clip_pretrained}"
        )
        clip_model.eval().to(device)
        return clip_model, preprocess

    # our own pretrained CLIP-SKU
    print(f"[CLIP] Loading fine-tuned CLIP-SKU checkpoint from {clip_sku_ckpt}")
    ckpt = torch.load(clip_sku_ckpt, map_location="cpu", weights_only=False)
    sku2idx = ckpt["sku2idx"]
    num_skus = len(sku2idx)

    model = ClipSkuBaseline(
        clip_model=clip_model,
        num_skus=num_skus,
        freeze_towers=False,  # 這裡只是 inference，所以 freeze 與否都 OK
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval().to(device)

    return model, preprocess


@torch.no_grad()
def embed_image(
    clip_model,
    preprocess,
    image: Image.Image,
    device: torch.device,
) -> torch.Tensor:
    x = preprocess(image).unsqueeze(0).to(device)
    feat = clip_model.encode_image(x)
    feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu()  # (1, D)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    a, b: (1, D) L2-normalized.
    """
    return float((a @ b.t()).item())


# ============================================================
# Diffusion (DiT/SD) helpers
# ============================================================

def setup_diffusion(
    device: torch.device,
    model_name: str = "runwayml/stable-diffusion-v1-5",
    lora_ckpt: str | None = None,
    lora_rank: int = 4,
    lora_alpha: float = 1.0,
) -> StableDiffusionImg2ImgPipeline:
    """
    Create an image-to-image diffusion pipeline (SD1.5 img2img).

    If lora_ckpt is provided, we:
      1) inject custom LoRA modules into the UNet (same logic as in train_sd_lora_df2.py)
      2) load the saved LoRA weights from the checkpoint.
    """
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_name)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing("max")

    if lora_ckpt is not None:
        print(f"[INFO] Loading SD LoRA checkpoint from {lora_ckpt}")
        ckpt = torch.load(lora_ckpt, map_location="cpu")

        ckpt_rank = ckpt.get("lora_rank", lora_rank)
        ckpt_alpha = ckpt.get("lora_alpha", lora_alpha)

        # 1) create LoRA modules on the current UNet
        inject_lora_into_unet(
            pipe.unet,
            rank=ckpt_rank,
            alpha=ckpt_alpha,
        )

        # 2) load LoRA weights
        lora_state = ckpt["lora_state_dict"]
        missing, unexpected = pipe.unet.load_state_dict(lora_state, strict=False)
        if missing:
            print(f"[LoRA] Missing keys when loading LoRA weights: {len(missing)}")
        if unexpected:
            print(f"[LoRA] Unexpected keys when loading LoRA weights: {len(unexpected)}")

    pipe = pipe.to(device)
    return pipe


def setup_dit_pipeline(
    device: torch.device,
    model_name: str = "PixArt-alpha/PixArt-XL-2-512x512",
):
    """
    Setup a DiT-based text-to-image pipeline (e.g., PixArt-alpha).
    """
    print(f"[INFO] Using DiT pipeline: {model_name}")

    if "PixArt-alpha" in model_name:
        # PixArtAlphaPipeline, more stable
        pipe = PixArtAlphaPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )
    else:
        # fallback: generic DiffusionPipeline
        pipe = DiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )

    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def is_black_image(img: Image.Image, threshold: float = 0.02) -> bool:
    """
    Return True if the image is nearly black on average.
    This is used to filter out NSFW-blocked outputs from StableDiffusion,
    which are typically returned as fully black images.
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    mean_val = float(arr.mean())
    return mean_val < threshold


def generate_views(
    pipe,
    init_image: Image.Image,
    prompt: str,
    num_samples: int,
    strength_min: float,
    strength_max: float,
    guidance_min: float,
    guidance_max: float,
    backend: str = "sd",
) -> List[Image.Image]:
    """
    Generic generator with random strength and guidance.

    backend:
      - "sd": use StableDiffusionImg2ImgPipeline (img2img, uses init_image)
      - "dit": use a DiT text-to-image DiffusionPipeline (ignores init_image)
    """
    results: List[Image.Image] = []
    backend = backend.lower()

    for _ in range(num_samples):
        strength = random.uniform(strength_min, strength_max)
        guidance = random.uniform(guidance_min, guidance_max)

        if backend == "dit":
            # DiT text-to-image: 不用 init_image
            out = pipe(
                prompt=prompt,
                num_inference_steps=30,
                guidance_scale=guidance,
                num_images_per_prompt=1,
            )
        elif backend == "sd":
            # SD img2img
            out = pipe(
                prompt=prompt,
                image=init_image,
                strength=strength,
                guidance_scale=guidance,
                num_inference_steps=30,
            )
        else:
            raise ValueError(f"Unsupported backend for generate_views: {backend}")

        img = out.images[0]

        # Skip NSFW-blocked black images before returning.
        if is_black_image(img):
            continue

        results.append(img)

    return results


# ============================================================
# Resume helpers (multi-view only)
# ============================================================


def _parse_aug_filename(stem: str, suffix: str) -> Tuple[str, int, int] | None:
    """
    Parse filename like "<image_id>_item<idx>_<suffix>NN" into
    (image_id, item_idx, gen_idx). Returns None if pattern mismatches.
    """
    pattern = rf"^(?P<image_id>.+)_item(?P<item_idx>\d+)_({re.escape(suffix)})(?P<gen_idx>\d+)$"
    m = re.match(pattern, stem)
    if not m:
        return None
    image_id = m.group("image_id")
    item_idx = int(m.group("item_idx"))
    gen_idx = int(m.group("gen_idx"))
    return image_id, item_idx, gen_idx


def restore_existing_multiview(
    sku_root: Path,
    split: str,
    mv_subdir: str,
    mv_suffix: str,
    meta: Dict,
    sku2texts: Dict[str, List[str]],
    mv_backend: str,
) -> Tuple[List[Dict], Set[Tuple[str, str, int]]]:
    """
    Scan existing multi-view augmentation images on disk and reconstruct:

      - metadata entries in meta["skus"][...]["catalog"]
      - JSONL records for those images

    Returns:
      new_records: list of reconstructed JSONL records
      processed_keys: set of (sku_id, image_id, item_idx) that already
                      have multiview images and should be skipped when
                      generating new ones.

    NOTE: currently only handles multiview (not counterfactual), which
          matches the typical setting num_counterfactual=0.
    """
    new_records: List[Dict] = []
    processed_keys: Set[Tuple[str, str, int]] = set()

    for sku_id, sku_info in meta["skus"].items():
        mv_dir = sku_root / split / mv_subdir / sku_id
        if not mv_dir.exists():
            continue

        for img_path in sorted(mv_dir.glob("*.jpg")):
            parsed = _parse_aug_filename(img_path.stem, mv_suffix)
            if parsed is None:
                continue
            image_id, item_idx, gen_idx = parsed

            # 找回原始 catalog entry
            orig_entry = None
            for e in sku_info.get("catalog", []):
                if e.get("dit_aug", False):
                    continue
                if str(e.get("image_id")) == image_id and int(e.get("item_idx", -1)) == item_idx:
                    orig_entry = e
                    break

            if orig_entry is None:
                print(
                    f"[RESUME][WARN] Cannot match restored multiview {img_path} "
                    f"to any original catalog entry; skipping."
                )
                continue

            key = (sku_id, image_id, item_idx)
            processed_keys.add(key)

            crop_rel_path = str(img_path.relative_to(sku_root))
            new_image_id = f"{image_id}_aug{gen_idx:02d}"

            # 避免重複塞進 meta（idempotent）
            already_in_meta = any(
                e.get("crop_path") == crop_rel_path
                for e in sku_info.get("catalog", [])
            )
            if not already_in_meta:
                new_entry = {
                    "crop_path": crop_rel_path,
                    "orig_image_path": orig_entry["orig_image_path"],
                    "image_id": new_image_id,
                    "item_idx": orig_entry["item_idx"],
                    "bbox": orig_entry["bbox"],
                    "occlusion": orig_entry["occlusion"],
                    "viewpoint": orig_entry["viewpoint"],
                    "dit_aug": True,
                    "dit_mode": f"multiview_{mv_backend}",
                }
                sku_info["catalog"].append(new_entry)

            # 重建 JSONL record
            texts_for_sku = sku2texts.get(sku_id, [])
            prompt = build_multiview_prompt(
                category_name=sku_info["category_name"],
                occlusion=orig_entry["occlusion"],
                viewpoint=orig_entry["viewpoint"],
            )
            text_for_aug = texts_for_sku[0] if texts_for_sku else prompt

            rec = {
                "split": split,
                "sku_id": sku_id,
                "pair_id": sku_info["pair_id"],
                "style": sku_info["style"],
                "category_id": sku_info["category_id"],
                "category_name": sku_info["category_name"],
                "domain": "catalog",
                "image_path": crop_rel_path,
                "orig_image_path": orig_entry["orig_image_path"],
                "image_id": new_image_id,
                "item_idx": orig_entry["item_idx"],
                "bbox": orig_entry["bbox"],
                "occlusion": orig_entry["occlusion"],
                "viewpoint": orig_entry["viewpoint"],
                "text": text_for_aug,
            }
            new_records.append(rec)

    print(
        f"[RESUME] Restored {len(new_records)} multiview images from disk across "
        f"{len(processed_keys)} catalog entries."
    )
    return new_records, processed_keys


# ============================================================
# Augmentation logic
# ============================================================


def build_multiview_prompt(
    category_name: str,
    occlusion: int,
    viewpoint: int,
) -> str:
    # Make prompts more product-like (no "on a person") to reduce NSFW triggers.
    if viewpoint == 1:
        view_str = "flat lay product photo"
    elif viewpoint == 2:
        view_str = "front view mannequin product photo"
    else:
        view_str = "side or back view mannequin product photo"

    if occlusion == 1:
        occ_str = "not occluded"
    elif occlusion == 2:
        occ_str = "partially occluded"
    else:
        occ_str = "heavily occluded"

    prompt = (
        f"A catalog product photo of a {occ_str} {category_name} in {view_str}, "
        f"studio lighting, high quality, plain background."
    )
    return prompt


def build_counterfactual_prompt(
    category_name: str,
    orig_color_name: str,
) -> Tuple[str, str]:
    """
    Build a prompt that keeps style but changes color.
    Returns (prompt, new_color_name).
    """
    new_color = pick_different_color(orig_color_name)
    prompt = (
        f"A catalog product photo of a {new_color} {category_name}, "
        f"same overall shape and style as the original but different color, "
        f"studio lighting, high quality, plain background."
    )
    return prompt, new_color


def save_aug_image(
    sku_root: Path,
    split: str,
    sku_id: str,
    entry: Dict,
    gen_idx: int,
    img: Image.Image,
    subdir: str,
    suffix: str = "aug",
) -> str:
    """
    Save generated image under e.g. train/catalog_dit/<sku_id>/xxx_<suffix>NN.jpg
    and return relative crop_path (as used in metadata/image_text.jsonl).
    """
    image_id = entry["image_id"]
    item_idx = entry["item_idx"]
    rel_dir = Path(split) / subdir / sku_id
    out_dir = sku_root / rel_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{image_id}_item{item_idx}_{suffix}{gen_idx:02d}.jpg"
    out_path = out_dir / filename
    img.save(out_path, quality=95)

    crop_rel_path = str(rel_dir / filename)
    return crop_rel_path


def augment_sku_multiview(
    sku_root: Path,
    split: str,
    sku_id: str,
    sku_info: Dict,
    entry: Dict,
    crop_abs: Path,
    pipe,
    clip_model,
    clip_preprocess,
    device: torch.device,
    sku2texts: Dict[str, List[str]],
    meta: Dict,
    jsonl_records: List[Dict],
    num_views: int,
    sim_thresh: float,
    color_max_diff: float,
    mv_subdir: str,
    mv_suffix: str,
    mv_backend: str,
) -> None:
    """
    Multi-view synthesis for a single catalog crop (same SKU id).

    mv_backend: "sd" (SD img2img) or "dit" (DiT text-to-image).
    """
    if num_views <= 0:
        return

    try:
        orig_img = Image.open(crop_abs).convert("RGB")
    except Exception as e:
        print(f"[WARN] Failed to open {crop_abs}: {e}")
        return

    hsv_ref = avg_hsv(orig_img)
    feat_ref = embed_image(clip_model, clip_preprocess, orig_img, device)

    prompt = build_multiview_prompt(
        category_name=sku_info["category_name"],
        occlusion=entry["occlusion"],
        viewpoint=entry["viewpoint"],
    )

    gens = generate_views(
        pipe=pipe,
        init_image=orig_img,
        prompt=prompt,
        num_samples=num_views,
        strength_min=0.2,
        strength_max=0.4,
        guidance_min=5.0,
        guidance_max=8.0,
        backend=mv_backend,
    )

    texts_for_sku = sku2texts.get(sku_id, [prompt])
    text_for_aug = texts_for_sku[0] if texts_for_sku else prompt

    gen_idx = 0
    for gen_img in gens:
        feat_gen = embed_image(clip_model, clip_preprocess, gen_img, device)
        sim = cosine_sim(feat_ref, feat_gen)
        if sim < sim_thresh:
            # too off-distribution
            continue

        hsv_gen = avg_hsv(gen_img)
        diff = color_distance(hsv_ref, hsv_gen)
        if diff > color_max_diff:
            # color drifted too much → 跟 counterfactual 分開
            continue

        gen_idx += 1

        crop_rel_path = save_aug_image(
            sku_root=sku_root,
            split=split,
            sku_id=sku_id,
            entry=entry,
            gen_idx=gen_idx,
            img=gen_img,
            subdir=mv_subdir,
            suffix=mv_suffix,
        )

        new_image_id = f"{entry['image_id']}_aug{gen_idx:02d}"

        # Update metadata: add a new catalog entry.
        new_entry = {
            "crop_path": crop_rel_path,
            "orig_image_path": entry["orig_image_path"],
            "image_id": new_image_id,
            "item_idx": entry["item_idx"],
            "bbox": entry["bbox"],
            "occlusion": entry["occlusion"],
            "viewpoint": entry["viewpoint"],
            "dit_aug": True,
            "dit_mode": f"multiview_{mv_backend}",
        }
        meta["skus"][sku_id]["catalog"].append(new_entry)

        # Update image_text jsonl record.
        rec = {
            "split": split,
            "sku_id": sku_id,
            "pair_id": sku_info["pair_id"],
            "style": sku_info["style"],
            "category_id": sku_info["category_id"],
            "category_name": sku_info["category_name"],
            "domain": "catalog",
            "image_path": crop_rel_path,
            "orig_image_path": entry["orig_image_path"],
            "image_id": new_image_id,
            "item_idx": entry["item_idx"],
            "bbox": entry["bbox"],
            "occlusion": entry["occlusion"],
            "viewpoint": entry["viewpoint"],
            "text": text_for_aug,
        }
        jsonl_records.append(rec)


def augment_sku_counterfactual(
    sku_root: Path,
    split: str,
    sku_id: str,
    sku_info: Dict,
    entry: Dict,
    crop_abs: Path,
    pipe,
    clip_model,
    clip_preprocess,
    device: torch.device,
    sku2texts: Dict[str, List[str]],
    meta: Dict,
    jsonl_records: List[Dict],
    num_cf: int,
    sim_min: float,
    sim_max: float,
    color_min_diff: float,
    cf_subdir: str,
    cf_suffix: str,
    cf_backend: str,
) -> None:
    """
    Counterfactual negatives: change color while preserving shape.
    We create new SKUs like "<sku_id>_cf1" as separate labels.

    cf_backend: "sd" or "dit".
    Recommended to use this mainly on the train split.
    """
    if num_cf <= 0:
        return

    try:
        orig_img = Image.open(crop_abs).convert("RGB")
    except Exception as e:
        print(f"[WARN] Failed to open {crop_abs}: {e}")
        return

    hsv_ref = avg_hsv(orig_img)
    feat_ref = embed_image(clip_model, clip_preprocess, orig_img, device)
    orig_color_name = estimate_color_name(hsv_ref)

    texts_for_sku = sku2texts.get(sku_id, [])
    base_text = texts_for_sku[0] if texts_for_sku else sku_info["category_name"]

    cf_idx = 0
    for _ in range(num_cf):
        prompt, new_color = build_counterfactual_prompt(
            category_name=sku_info["category_name"],
            orig_color_name=orig_color_name,
        )
        gen_imgs = generate_views(
            pipe=pipe,
            init_image=orig_img,
            prompt=prompt,
            num_samples=1,
            strength_min=0.4,
            strength_max=0.7,
            guidance_min=5.0,
            guidance_max=8.0,
            backend=cf_backend,
        )

        # Skip if NSFW filter produced no valid image.
        if len(gen_imgs) == 0:
            continue

        gen_img = gen_imgs[0]

        feat_gen = embed_image(clip_model, clip_preprocess, gen_img, device)
        sim = cosine_sim(feat_ref, feat_gen)
        if sim < sim_min or sim > sim_max:
            # want near-miss: not too close, not too far
            continue

        hsv_gen = avg_hsv(gen_img)
        diff = color_distance(hsv_ref, hsv_gen)
        if diff < color_min_diff:
            # color change not strong enough
            continue

        cf_idx += 1
        cf_sku_id = f"{sku_id}_cf{cf_idx}"

        crop_rel_path = save_aug_image(
            sku_root=sku_root,
            split=split,
            sku_id=cf_sku_id,
            entry=entry,
            gen_idx=cf_idx,
            img=gen_img,
            subdir=cf_subdir,
            suffix=cf_suffix,
        )
        new_image_id = f"{entry['image_id']}_cf{cf_idx:02d}"

        # Metadata: new SKU.
        cf_info = {
            "pair_id": sku_info["pair_id"],
            "style": sku_info["style"],
            "category_id": sku_info["category_id"],
            "category_name": sku_info["category_name"],
            "catalog": [],
            "query": [],
            "counterfactual_of": sku_id,
            "counterfactual_color": new_color,
            "dit_aug": True,
            "dit_mode": f"counterfactual_{cf_backend}",
        }

        new_entry = {
            "crop_path": crop_rel_path,
            "orig_image_path": entry["orig_image_path"],
            "image_id": new_image_id,
            "item_idx": entry["item_idx"],
            "bbox": entry["bbox"],
            "occlusion": entry["occlusion"],
            "viewpoint": entry["viewpoint"],
            "dit_aug": True,
            "dit_mode": f"counterfactual_{cf_backend}",
        }
        cf_info["catalog"].append(new_entry)

        meta["skus"][cf_sku_id] = cf_info

        # JSONL: new catalog entry for cf SKU.
        cf_text = (
            f"{base_text} This is a counterfactual version with {new_color} color."
        )
        rec = {
            "split": split,
            "sku_id": cf_sku_id,
            "pair_id": sku_info["pair_id"],
            "style": sku_info["style"],
            "category_id": sku_info["category_id"],
            "category_name": sku_info["category_name"],
            "domain": "catalog",
            "image_path": crop_rel_path,
            "orig_image_path": entry["orig_image_path"],
            "image_id": new_image_id,
            "item_idx": entry["item_idx"],
            "bbox": entry["bbox"],
            "occlusion": entry["occlusion"],
            "viewpoint": entry["viewpoint"],
            "text": cf_text,
        }
        jsonl_records.append(rec)


# ============================================================
# Argument parsing & main
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Offline DiT/SD-based augmentation for DeepFashion2_SKU "
            "(multi-view & counterfactual catalog images). "
            "Original *_sku_metadata.json / *_image_text.jsonl are never overwritten."
        )
    )
    parser.add_argument(
        "--sku_root",
        type=Path,
        required=True,
        help="Root of DeepFashion2_SKU (contains *_sku_metadata.json and *_image_text.jsonl).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split to augment (train / validation / test).",
    )
    parser.add_argument(
        "--max_skus",
        type=int,
        default=None,
        help="Optional limit on number of SKUs to process (for debugging).",
    )
    # Multi-view params
    parser.add_argument(
        "--num_views",
        type=int,
        default=2,
        help="Number of multi-view synthetic images per catalog crop.",
    )
    parser.add_argument(
        "--mv_sim_thresh",
        type=float,
        default=0.90,
        help="CLIP cosine similarity threshold for multi-view (>=).",
    )
    parser.add_argument(
        "--mv_color_max_diff",
        type=float,
        default=0.20,
        help="Maximum HSV distance allowed for multi-view (<=).",
    )
    parser.add_argument(
        "--mv_subdir",
        type=str,
        default="catalog_dit",
        help="Subdirectory under <split>/ to save multi-view images, e.g. 'catalog_dit'.",
    )
    parser.add_argument(
        "--mv_suffix",
        type=str,
        default="aug",
        help="Filename suffix for multi-view images, e.g. 'aug' -> *_aug01.jpg.",
    )

    # Counterfactual params
    parser.add_argument(
        "--num_counterfactual",
        type=int,
        default=0,
        help="Number of counterfactual images per catalog crop (0 to disable).",
    )
    parser.add_argument(
        "--cf_sim_min",
        type=float,
        default=0.60,
        help="Minimum CLIP similarity for counterfactual.",
    )
    parser.add_argument(
        "--cf_sim_max",
        type=float,
        default=0.90,
        help="Maximum CLIP similarity for counterfactual.",
    )
    parser.add_argument(
        "--cf_color_min_diff",
        type=float,
        default=0.30,
        help="Minimum HSV distance required for counterfactual.",
    )
    parser.add_argument(
        "--cf_subdir",
        type=str,
        default="catalog_cf",
        help="Subdirectory under <split>/ to save counterfactual images, e.g. 'catalog_cf'.",
    )
    parser.add_argument(
        "--cf_suffix",
        type=str,
        default="cf",
        help="Filename suffix for counterfactual images, e.g. 'cf' -> *_cf01.jpg.",
    )
    parser.add_argument(
        "--cf_backend",
        type=str,
        default="sd",
        choices=["sd", "dit"],
        help="Backend for counterfactual generation: 'sd' (img2img) or 'dit' (text-to-image).",
    )

    # Device / model
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for diffusion and CLIP.",
    )
    parser.add_argument(
        "--sd_model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Diffusion model name or path for StableDiffusionImg2ImgPipeline.",
    )
    parser.add_argument(
        "--use_dit_multiview",
        action="store_true",
        help=(
            "If set, use a DiT-based DiffusionPipeline (text-to-image) for multi-view "
            "augmentation instead of StableDiffusionImg2ImgPipeline."
        ),
    )
    parser.add_argument(
        "--dit_model",
        type=str,
        default="PixArt-alpha/PixArt-XL-2-512x512",
        help="DiT text-to-image model name or path (e.g., PixArt-alpha/PixArt-XL-2-512x512).",
    )

    # Resume
    parser.add_argument(
        "--resume_from_disk",
        action="store_true",
        help=(
            "If set, scan existing multiview images on disk (mv_subdir) to reconstruct "
            "metadata / jsonl and resume instead of starting from scratch."
        ),
    )

    # Output naming
    parser.add_argument(
        "--out_suffix",
        type=str,
        default="dit_pretrained_aug",
        help=(
            "Suffix inserted before file extension for augmented outputs. "
            "Example: out_suffix=dit_pretrained_aug -> "
            "train_sku_metadata.dit_pretrained_aug.json "
            "and train_image_text.dit_pretrained_aug.jsonl"
        ),
    )
    parser.add_argument(
        "--out_sku_meta",
        type=Path,
        default=None,
        help=(
            "Optional explicit path for augmented sku metadata JSON. "
            "If None, will use <split>_sku_metadata.<out_suffix>.json under sku_root."
        ),
    )
    parser.add_argument(
        "--out_image_text",
        type=Path,
        default=None,
        help=(
            "Optional explicit path for augmented image_text JSONL. "
            "If None, will use <split>_image_text.<out_suffix>.jsonl under sku_root."
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
        default=None,
        help=(
            "Optional path to a CLIP-SKU checkpoint saved by train_clip_sku_df2.py. "
            "If provided, we will use its fine-tuned image encoder for gating."
        ),
    )
    parser.add_argument(
        "--shuffle_skus",
        action="store_true",
        help="Shuffle SKU order before applying max_skus (for random subset).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for SKU shuffling.",
    )
    parser.add_argument(
        "--sd_lora_ckpt",
        type=str,
        default=None,
        help=(
            "Optional path to SD LoRA checkpoint (.pth) trained by train_sd_lora_df2.py. "
            "If provided, it will be loaded into the SD img2img UNet."
        ),
    )
    parser.add_argument(
        "--sd_lora_rank",
        type=int,
        default=4,
        help="Rank of LoRA adapter.",
    )
    parser.add_argument(
        "--sd_lora_alpha",
        type=float,
        default=1.0,
        help="Scaling factor alpha for LoRA (output = base + alpha/r * BAx).",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    sku_root: Path = args.sku_root
    split: str = args.split

    # 1) Load ORIGINAL metadata + image_text
    meta = load_sku_metadata(sku_root, split)

    orig_jsonl_path = sku_root / f"{split}_image_text.jsonl"
    sku2texts = build_sku_catalog_text_map(orig_jsonl_path)

    # 2) Setup diffusion & CLIP
    pipe_sd = setup_diffusion(
        device=device,
        model_name=args.sd_model,
        lora_ckpt=args.sd_lora_ckpt,
        lora_rank=args.sd_lora_rank,
        lora_alpha=args.sd_lora_alpha
    )

    pipe_dit = None
    if args.use_dit_multiview or args.cf_backend == "dit":
        pipe_dit = setup_dit_pipeline(device=device, model_name=args.dit_model)

    # multiview backend
    if args.use_dit_multiview:
        if pipe_dit is None:
            raise RuntimeError(
                "DiT pipeline not initialized but --use_dit_multiview is set."
            )
        pipe_mv = pipe_dit
        mv_backend = "dit"
    else:
        pipe_mv = pipe_sd
        mv_backend = "sd"

    # counterfactual backend
    if args.cf_backend == "dit":
        if pipe_dit is None:
            raise RuntimeError("DiT pipeline not initialized but --cf_backend dit is set.")
        pipe_cf = pipe_dit
        cf_backend = "dit"
    else:
        pipe_cf = pipe_sd
        cf_backend = "sd"

    clip_model, clip_preprocess = setup_clip(
        device=device,
        clip_model_name=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        clip_sku_ckpt=args.clip_sku_ckpt,
    )

    num_views = args.num_views
    num_cf = args.num_counterfactual

    # ---- Resume from existing multiview images on disk (optional) ----
    new_jsonl_records: List[Dict] = []
    processed_keys: Set[Tuple[str, str, int]] | None = None

    if args.resume_from_disk:
        restored_records, processed_keys = restore_existing_multiview(
            sku_root=sku_root,
            split=split,
            mv_subdir=args.mv_subdir,
            mv_suffix=args.mv_suffix,
            meta=meta,
            sku2texts=sku2texts,
            mv_backend=mv_backend,
        )
        new_jsonl_records.extend(restored_records)
    else:
        processed_keys = None

    # 3) Pre-compute total number of ORIGINAL, UNPROCESSED catalog crops for tqdm
    skus = meta["skus"]
    items = list(skus.items())

    if args.shuffle_skus:
        random.seed(args.seed)
        random.shuffle(items)

    if args.max_skus is not None:
        items = items[: args.max_skus]

    total_catalog = 0
    for sku_id, info in items:
        for e in info.get("catalog", []):
            if e.get("dit_aug", False):
                continue
            image_id = str(e.get("image_id"))
            try:
                item_idx = int(e.get("item_idx", -1))
            except Exception:
                item_idx = -1
            key = (sku_id, image_id, item_idx)
            if processed_keys is not None and key in processed_keys:
                continue
            total_catalog += 1

    print(f"[INFO] Total remaining original catalog crops to process: {total_catalog}")

    with tqdm(
        total=total_catalog,
        desc=f"Augment {split} catalog",
        unit="crop",
    ) as pbar:
        for sku_id, sku_info, entry, crop_abs in iter_catalog_entries(
            meta,
            sku_root,
            split,
            max_skus=args.max_skus,
            shuffle=args.shuffle_skus,
            seed=args.seed,
            skip_processed=processed_keys,
        ):
            if not crop_abs.exists():
                print(f"[WARN] Missing crop image: {crop_abs}")
                pbar.update(1)
                continue

            # (1) Multi-view synthesis (same SKU)
            augment_sku_multiview(
                sku_root=sku_root,
                split=split,
                sku_id=sku_id,
                sku_info=sku_info,
                entry=entry,
                crop_abs=crop_abs,
                pipe=pipe_mv,
                clip_model=clip_model,
                clip_preprocess=clip_preprocess,
                device=device,
                sku2texts=sku2texts,
                meta=meta,
                jsonl_records=new_jsonl_records,
                num_views=num_views,
                sim_thresh=args.mv_sim_thresh,
                color_max_diff=args.mv_color_max_diff,
                mv_subdir=args.mv_subdir,
                mv_suffix=args.mv_suffix,
                mv_backend=mv_backend,
            )

            # (2) Counterfactual negatives (new SKU ids), usually only on train split.
            if num_cf > 0 and split == "train":
                augment_sku_counterfactual(
                    sku_root=sku_root,
                    split=split,
                    sku_id=sku_id,
                    sku_info=sku_info,
                    entry=entry,
                    crop_abs=crop_abs,
                    pipe=pipe_cf,
                    clip_model=clip_model,
                    clip_preprocess=clip_preprocess,
                    device=device,
                    sku2texts=sku2texts,
                    meta=meta,
                    jsonl_records=new_jsonl_records,
                    num_cf=num_cf,
                    sim_min=args.cf_sim_min,
                    sim_max=args.cf_sim_max,
                    color_min_diff=args.cf_color_min_diff,
                    cf_subdir=args.cf_subdir,
                    cf_suffix=args.cf_suffix,
                    cf_backend=cf_backend,
                )

            pbar.update(1)

    # 4) Decide output paths (NEVER overwrite originals)
    if args.out_sku_meta is not None:
        out_meta_path = args.out_sku_meta
    else:
        # e.g. train_sku_metadata.dit_pretrained_aug.json
        out_meta_path = sku_root / f"{split}_sku_metadata.{args.out_suffix}.json"

    if args.out_image_text is not None:
        out_jsonl_path = args.out_image_text
    else:
        # e.g. train_image_text.dit_pretrained_aug.jsonl
        out_jsonl_path = sku_root / f"{split}_image_text.{args.out_suffix}.jsonl"

    # 5) Write augmented metadata + image_text
    save_sku_metadata(meta, out_meta_path)
    write_augmented_image_text(
        orig_jsonl_path=orig_jsonl_path,
        out_jsonl_path=out_jsonl_path,
        new_records=new_jsonl_records,
    )

    print("[DONE] DiT/SD augmentation finished.")
    print(f"[DONE] Metadata   -> {out_meta_path}")
    print(f"[DONE] Image_text -> {out_jsonl_path}")


if __name__ == "__main__":
    main()
