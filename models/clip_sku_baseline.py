# models/clip_sku_baseline.py

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipSkuBaseline(nn.Module):
    """
    CLIP/SigLIP-style bi-encoder + SKU embedding table.

    A pretrained CLIP model is passed in (from open_clip).
    We add an nn.Embedding(num_skus, D) and train it (and optionally unfreeze CLIP).
    """

    def __init__(
        self,
        clip_model,
        num_skus: int,
        freeze_towers: bool = False,
        partial_finetune: bool = False,
        vision_unlocked_groups: int = 1,
    ) -> None:
        super().__init__()

        self.clip_model = clip_model
        self.num_skus = num_skus
        self.partial_finetune = partial_finetune
        self.vision_unlocked_groups = vision_unlocked_groups

        embed_dim = self.clip_model.text_projection.shape[1]

        self.sku_embed = nn.Embedding(num_skus, embed_dim)
        nn.init.normal_(self.sku_embed.weight, std=0.02)

        self.freeze_towers = freeze_towers
        if freeze_towers:
            # Baseline2 behavior: freeze both image and text towers.
            for p in self.clip_model.parameters():
                p.requires_grad = False
        elif partial_finetune:
            # New behavior: freeze text encoder, and only lightly finetune
            # the last few groups of the image encoder.
            self._freeze_text_encoder()
            self._partially_unfreeze_vision(
                unlocked_groups=self.vision_unlocked_groups
            )
        # Trainable temperature for InfoNCE.
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

    def _freeze_text_encoder(self) -> None:
        """
        Freeze all text-side parameters of the CLIP model.
        This keeps the original text embedding space intact.
        """
        text_prefixes = (
            "transformer.",
            "token_embedding.",
            "positional_embedding",
            "ln_final.",
            "text_projection",
        )
        for name, param in self.clip_model.named_parameters():
            if name.startswith(text_prefixes):
                param.requires_grad = False

    def _partially_unfreeze_vision(self, unlocked_groups: int = 1) -> None:
        """
        Freeze most of the vision tower and only leave the last few groups
        trainable. For OpenCLIP ViT models we can use lock_image_tower.
        """
        # First, freeze everything under visual.
        for name, param in self.clip_model.named_parameters():
            if name.startswith("visual."):
                param.requires_grad = False

        # If CLIP exposes lock_image_tower (OpenCLIP), use it.
        if hasattr(self.clip_model, "lock_image_tower"):
            # unlocked_groups=1 means only the last group is trainable.
            self.clip_model.lock_image_tower(
                unlocked_groups=unlocked_groups,
                freeze_bn_stats=True,
            )
            return

        # Fallback: manually unfreeze the last few transformer blocks if needed.
        visual = self.clip_model.visual
        blocks = None

        # ViT-based visual tower
        if hasattr(visual, "blocks"):
            blocks = visual.blocks
        elif hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
            blocks = visual.transformer.resblocks

        if blocks is None:
            # For ResNet-style visual towers we keep everything frozen.
            return

        n = len(blocks)
        k = min(unlocked_groups, n)
        for block in blocks[n - k :]:
            for param in block.parameters():
                param.requires_grad = True

        # Also unfreeze final norm and projection if they exist.
        for attr in ("ln_post", "proj"):
            if hasattr(visual, attr):
                for param in getattr(visual, attr).parameters():
                    param.requires_grad = True

    @property
    def embed_dim(self) -> int:
        return self.clip_model.text_projection.shape[1]

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 3, H, W)
        Returns:
            embs: (B, D) L2-normalized
        """
        feats = self.clip_model.encode_image(images)
        feats = F.normalize(feats, dim=-1)
        return feats

    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_tokens: (B, L)
        Returns:
            embs: (B, D) L2-normalized
        """
        feats = self.clip_model.encode_text(text_tokens)
        feats = F.normalize(feats, dim=-1)
        return feats

    def sku_embeddings(self) -> torch.Tensor:
        """
        Returns:
            (num_skus, D) L2-normalized SKU embeddings.
        """
        w = self.sku_embed.weight
        w = F.normalize(w, dim=-1)
        return w

    def forward(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward used for training.

        Returns:
            img_emb: (B, D)
            txt_emb: (B, D)
            sku_emb_all: (Ns, D)
            logit_scale: scalar exp(temperature)
        """
        img_emb = self.encode_image(images)
        txt_emb = self.encode_text(text_tokens)
        sku_emb_all = self.sku_embeddings()
        logit_scale = self.logit_scale.exp()
        return img_emb, txt_emb, sku_emb_all, logit_scale
