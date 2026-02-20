import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from model.constants import EMBEDDING_DIM


class EmbeddingModel(nn.Module):
    def __init__(self, embedding_dim: int = EMBEDDING_DIM, pretrained: bool = True) -> None:
        super().__init__()

        # Load EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        # EfficientNet-B0 feature extractor produces 1280-dim features
        # Remove the classifier (avgpool is kept as part of backbone)
        self.backbone = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim),
        )

        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x) 
        features = self.pool(features)
        features = features.flatten(1)
        embeddings = self.projection(features) 
        return F.normalize(embeddings, p=2, dim=1)

    # Progressive unfreezing helpers
    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (train projection head only)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze the last 2 feature blocks of EfficientNet-B0.

        EfficientNet-B0 backbone (self.backbone) has 9 children:
          0: stem Conv2dNormActivation
          1–8: MBConv blocks (indices 1..8)
        We unfreeze blocks 7 and 8 — the deepest spatial features.
        """
        # First ensure everything is frozen
        self.freeze_backbone()

        # Unfreeze last 2 blocks
        for block in list(self.backbone.children())[-2:]:
            for param in block.parameters():
                param.requires_grad = True

    def trainable_params(self) -> list[nn.Parameter]:
        """Return list of parameters with requires_grad=True."""
        return [p for p in self.parameters() if p.requires_grad]
