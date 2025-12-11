# models/reid_resnet50_bnneck.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class ReIDResNet50BNNeck(nn.Module):
    """
    ResNet-50 backbone with BNNeck and 512-dim embedding.

    - Global average pooled feature -> BN -> Linear -> BN (feat_bn)
    - Classification uses feat_bn.
    - Embedding is L2-normalized feat_bn for metric learning / retrieval.
    """

    def __init__(self, num_classes: int, feat_dim: int = 512):
        super().__init__()

        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        self.base = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck.bias.requires_grad_(False)

        self.feat = nn.Linear(2048, feat_dim, bias=False)
        self.feat_bn = nn.BatchNorm1d(feat_dim)
        self.feat_bn.bias.requires_grad_(False)

        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)

        self._init_params()

    def _init_params(self):
        nn.init.kaiming_normal_(self.feat.weight, mode="fan_out")
        nn.init.normal_(self.feat_bn.weight, std=0.01)
        nn.init.normal_(self.classifier.weight, std=0.01)

    def forward(self, x, normalize: bool = True):
        x = self.base(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # (B, 2048)

        x = self.bottleneck(x)
        feat = self.feat(x)
        feat = self.feat_bn(feat)  # (B, feat_dim)

        if normalize:
            emb = F.normalize(feat, p=2, dim=1)
        else:
            emb = feat

        logits = self.classifier(feat)
        return logits, emb
