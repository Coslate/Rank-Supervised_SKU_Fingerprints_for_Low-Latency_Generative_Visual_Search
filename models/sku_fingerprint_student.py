from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkuFingerprintStudent(nn.Module):
    """
    Small aggregation network g_θ that aggregates all view embeddings
    (catalog / user / DiT multi-view) of the same SKU into a single
    student fingerprint.

    Input:
        views: (B, V, D)  V view embeddings per SKU
        mask:  (B, V) bool, True for valid view, False for padding

    Output:
        z: (B, D) L2-normalized student SKU fingerprints
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Project CLIP D-dim embeddings to Transformer hidden_dim
        self.input_proj = nn.Linear(embed_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,  # src shape: (B, V, H)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Learnable query for attention pooling
        self.pool_query = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.pool_query, mean=0.0, std=0.02)

        # Final projection back to CLIP embed_dim
        self.output_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, views: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        views: (B, V, D)
        mask:  (B, V) bool (True = valid token)
        """
        # 1) Project to hidden dim
        h = self.input_proj(views)  # (B, V, H)

        # 2) Transformer encoder
        # Transformer src_key_padding_mask: True = padding
        # Our mask=True means valid → invert it
        src_key_padding_mask = ~mask  # (B, V)
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)  # (B, V, H)

        # 3) Attention pooling
        # pool_query: (H,) → broadcast to (B, 1, H)
        q = self.pool_query[None, None, :]  # (1, 1, H)
        # Dot product between each view and the query
        scores = (h * q).sum(dim=-1)        # (B, V)
        # Mask out padding positions
        scores = scores.masked_fill(~mask, -1e9)
        # Softmax over views
        weights = F.softmax(scores, dim=-1)       # (B, V)
        # Weighted sum over views
        pooled = (weights.unsqueeze(-1) * h).sum(dim=1)  # (B, H)

        # 4) Project back to D and L2-normalize
        z = self.output_proj(pooled)              # (B, D)
        z = F.normalize(z, dim=-1)
        return z
