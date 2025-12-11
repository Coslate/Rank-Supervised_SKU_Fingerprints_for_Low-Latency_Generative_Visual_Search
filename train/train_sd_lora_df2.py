from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
)
from transformers import CLIPTokenizer, CLIPTextModel

from tqdm.auto import tqdm
import wandb


# =========================
# Dataset
# =========================

class DeepFashion2CatalogLoraDataset(Dataset):
    """
    Read from <split>_image_text.jsonl.
    Only keep records with domain == 'catalog' whose image file exists.
    """

    def __init__(
        self,
        sku_root: Path,
        split: str,
        tokenizer: CLIPTokenizer,
        resolution: int = 512,
        center_crop: bool = True,
        random_flip: bool = True,
        max_train_samples: int | None = None,
    ):
        super().__init__()
        self.sku_root = sku_root
        self.tokenizer = tokenizer

        jsonl_path = sku_root / f"{split}_image_text.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Missing image_text jsonl: {jsonl_path}")

        examples: List[Dict] = []
        with jsonl_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("domain") != "catalog":
                    continue
                rel_path = rec["image_path"]
                img_path = sku_root / rel_path
                if not img_path.exists():
                    continue

                text = rec.get("text", "").strip()
                if not text:
                    cat = rec.get("category_name", "clothing item")
                    text = f"A catalog product photo of a {cat}."
                examples.append(
                    {
                        "image_path": img_path,
                        "text": text,
                    }
                )

        if max_train_samples is not None:
            examples = examples[: max_train_samples]

        self.examples = examples
        print(f"[DATA] Loaded {len(self.examples)} catalog examples from {jsonl_path}")

        tfs: List[transforms.Transform] = [
            transforms.Resize(resolution, interpolation=Image.BILINEAR),
        ]
        if center_crop:
            tfs.append(transforms.CenterCrop(resolution))
        else:
            tfs.append(transforms.RandomCrop(resolution))
        if random_flip:
            tfs.append(transforms.RandomHorizontalFlip())
        tfs.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.image_transforms = transforms.Compose(tfs)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.examples[idx]
        image_path: Path = rec["image_path"]
        text: str = rec["text"]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] Failed to open {image_path}: {e}")
            image = Image.new("RGB", (512, 512), (0, 0, 0))

        pixel_values = self.image_transforms(image)

        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": tokens.input_ids[0],
            "attention_mask": tokens.attention_mask[0],
        }


# =========================
# Custom LoRA implementation
# =========================

class LoRALinear(torch.nn.Module):
    """
    Simple LoRA wrapper for a Linear layer:
    output = base(x) + alpha / r * (B(A(x)))
    """
    def __init__(self, base: torch.nn.Linear, rank: int = 4, alpha: float | None = None):
        super().__init__()
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank

        # low-rank adapters
        self.lora_down = torch.nn.Linear(self.in_features, rank, bias=False)
        self.lora_up = torch.nn.Linear(rank, self.out_features, bias=False)

        # init
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        assert x.shape[-1] == self.in_features, (
            f"[LoRALinear] mismatch: x.shape[-1]={x.shape[-1]} "
            f"!= in_features={self.in_features}"
        )

        # dtype / device / contiguous consistent
        x = x.to(self.base.weight.dtype)
        x = x.contiguous()

        base_out = self.base(x)  # (B, N, out_features)

        # 把 (B,N,C) 攤平成 (B*N,C) 來走線性層，比較安全
        x_2d = x.view(-1, self.in_features)
        lora_down_out = self.lora_down(x_2d)                 # (B*N, r)
        lora_up_out = self.lora_up(lora_down_out)            # (B*N, out_features)
        lora_up_out = lora_up_out.view(*x.shape[:-1], self.out_features)

        return base_out + self.scaling * lora_up_out

def _get_parent_and_attr(root: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    """
    Given a dotted module name (e.g. 'down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q'),
    return (parent_module, last_attr).
    """
    parts = module_name.split(".")
    parent = root
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    return parent, parts[-1]


def inject_lora_into_unet(
    unet: UNet2DConditionModel,
    rank: int = 4,
    alpha: float = 1.0,
    target_suffixes: Tuple[str, ...] = ("to_q", "to_k", "to_v", "to_out.0"),
) -> List[nn.Parameter]:
    """
    Replace specific Linear layers in UNet with LoRALinear wrappers.
    Returns the list of trainable LoRA parameters.
    """

    # Freeze everything first
    for p in unet.parameters():
        p.requires_grad = False

    num_wrapped = 0
    for name, module in list(unet.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(sfx) for sfx in target_suffixes):
            continue

        parent, attr = _get_parent_and_attr(unet, name)

        if attr.isdigit():
            idx = int(attr)
            base_linear = parent[idx]
            parent[idx] = LoRALinear(base_linear, rank=rank, alpha=alpha)
        else:
            base_linear = getattr(parent, attr)
            setattr(parent, attr, LoRALinear(base_linear, rank=rank, alpha=alpha))

        num_wrapped += 1

    print(f"[LoRA] Injected LoRA into {num_wrapped} Linear layers.")

    lora_params: List[nn.Parameter] = []
    for n, p in unet.named_parameters():
        if "lora_down" in n or "lora_up" in n:
            p.requires_grad = True
            lora_params.append(p)

    total_lora = sum(p.numel() for p in lora_params)
    print(f"[LoRA] Trainable LoRA parameters: {len(lora_params)} tensors, {total_lora:,} params.")

    return lora_params


# =========================
# Argument parsing
# =========================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Finetune Stable Diffusion (SD1.5) with custom LoRA on DeepFashion2_SKU catalog images."
        )
    )
    parser.add_argument(
        "--sku_root",
        type=Path,
        required=True,
        help="Root of DeepFashion2_SKU (contains <split>_image_text.jsonl).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split to use (default: train).",
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Stable Diffusion base model (HF hub id or local path).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save LoRA weights and config.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Training image resolution (will be resized + cropped).",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Per-device batch size.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for LoRA parameters.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        choices=["cosine", "constant"],
        help="LR scheduler type.",
    )
    parser.add_argument(
        "--lr_warmup_ratio",
        type=float,
        default=0.1,
        help="Warmup proportion of total training steps (0~1).",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Accumulate gradients over N steps before optimizer.step().",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Optional limit on number of training samples (for debugging).",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="Rank of LoRA adapter.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=1.0,
        help="Scaling factor alpha for LoRA (output = base + alpha/r * BAx).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16"],
        help="If 'fp16', use torch.cuda.amp for mixed precision.",
    )
    # wandb
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="sd_lora_df2",
        help="WandB project name.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="",
        help="WandB run name (empty = auto).",
    )
    return parser.parse_args()


# =========================
# Training loop
# =========================

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # 1. Load SD components
    print("[INFO] Loading Stable Diffusion base...")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model, subfolder="unet"
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model, subfolder="scheduler"
    )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # 2. Inject custom LoRA into UNet
    print(f"[INFO] Injecting custom LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})...")
    lora_params = inject_lora_into_unet(
        unet,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
    )
    if len(lora_params) == 0:
        raise RuntimeError("No LoRA parameters found; injection may have failed.")

    vae.to(device)
    text_encoder.to(device)
    unet.to(device)

    # Optimizer on LoRA params only
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    # 3. Dataset & Dataloader
    print("[INFO] Building dataset...")
    dataset = DeepFashion2CatalogLoraDataset(
        sku_root=args.sku_root,
        split=args.split,
        tokenizer=tokenizer,
        resolution=args.resolution,
        center_crop=True,
        random_flip=True,
        max_train_samples=args.max_train_samples,
    )

    train_dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # compute warmup steps from ratio
    warmup_steps = int(max_train_steps * args.lr_warmup_ratio)
    warmup_steps = max(warmup_steps, 1)

    print(
        f"[INFO] num_train_epochs={args.num_train_epochs}, "
        f"steps/epoch={num_update_steps_per_epoch}, "
        f"max_train_steps={max_train_steps}, "
        f"warmup_steps={warmup_steps} "
        f"(lr_warmup_ratio={args.lr_warmup_ratio})"
    )

    # LR scheduler
    def lr_lambda(step: int) -> float:
        # linear warmup
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        # cosine / constant decay after warmup
        progress = float(step - warmup_steps) / float(
            max(1, max_train_steps - warmup_steps)
        )
        progress = min(max(progress, 0.0), 1.0)

        if args.lr_scheduler_type == "constant":
            return 1.0

        # cosine decay
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_fp16 = args.mixed_precision == "fp16" and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    # 4. WandB init
    default_run_name = (
        args.wandb_run_name
        if args.wandb_run_name
        else f"sd_lora_df2_lr{args.learning_rate}_bs{args.train_batch_size}_rank{args.lora_rank}"
    )
    wandb.init(
        project=args.wandb_project,
        name=default_run_name,
        config=vars(args),
    )

    global_step = 0
    print("[INFO] Start training LoRA...")

    for epoch in range(args.num_train_epochs):
        unet.train()
        epoch_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{args.num_train_epochs}",
            leave=False,
        )

        for step, batch in enumerate(epoch_bar):
            with torch.cuda.amp.autocast(enabled=use_fp16):
                pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                # attention_mask = batch["attention_mask"].to(device, non_blocking=True)

                # Encode images to latents
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215

                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=device,
                    dtype=torch.long,
                )

                noise = torch.randn_like(latents)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(input_ids)[0]

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                target = noise
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            if use_fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if use_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                lr_scheduler.step()
                global_step += 1

                lr = lr_scheduler.get_last_lr()[0]
                epoch_bar.set_postfix(loss=float(loss.item()), lr=float(lr))

                wandb.log(
                    {
                        "train/loss": float(loss.item()),
                        "train/lr": float(lr),
                        "train/global_step": global_step,
                        "train/epoch": epoch + (step / len(train_dataloader)),
                    },
                    step=global_step,
                )

                if global_step >= max_train_steps:
                    break

        if global_step >= max_train_steps:
            break

    print("[INFO] Training finished, saving LoRA weights...")

    # Only save LoRA weights (parameters whose names contain 'lora_')
    full_state = unet.state_dict()
    lora_state = {k: v.cpu() for k, v in full_state.items() if "lora_" in k}

    out_path = args.output_dir / "sd_lora_df2_lora_only.pth"
    torch.save(
        {
            "lora_state_dict": lora_state,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "pretrained_model": args.pretrained_model,
        },
        out_path,
    )

    print(f"[DONE] LoRA weights saved to: {out_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
