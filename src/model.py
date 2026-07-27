"""EfficientNet-B0 SED model with attention pooling.

Architecture matches the Kaggle training notebook (``BirdSEDModel``):
EfficientNet-B0 (1-ch) → 1×1 SED head → temporal attention pooling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm


class BirdCLEFSED(nn.Module):
    """Multi-label Sound Event Detection network.

    Architecture
    ------------
    - Backbone: EfficientNet-B0 (1-channel mel input, no classifier head)
    - SED head: 1×1 conv bottleneck → class framewise logits
    - Attention pooling over time → clip-level multi-label logits

    Forward returns
    ---------------
    clipwise : (B, num_classes)  logits for multi-label classification
    framewise: (B, num_classes, T) frame-level logits (useful for SED viz)
    """

    def __init__(
        self,
        num_classes: int = 206,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=1,
            num_classes=0,
            global_pool="",
        )

        # Detect feature channels (CPU dummy — same as Kaggle notebook)
        with torch.no_grad():
            dummy = torch.randn(1, 1, 128, 256)
            features = self.backbone(dummy)
            if isinstance(features, tuple):
                features = features[0]
            self.feature_dim = int(features.shape[1])

        self.sed_head = nn.Sequential(
            nn.Conv2d(self.feature_dim, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )
        self.attention = nn.Sequential(
            nn.Linear(num_classes, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor of shape (B, 1, n_mels, time)
        """
        features = self.backbone(x)
        if isinstance(features, tuple):
            features = features[0]

        sed_logits = self.sed_head(features)
        # Pool frequency axis → framewise class scores (B, C, T)
        sed_logits = sed_logits.mean(dim=2)

        # Attention over time
        sed_flat = sed_logits.transpose(1, 2)  # (B, T, C)
        attn_weights = torch.softmax(self.attention(sed_flat), dim=1)  # (B, T, 1)
        pooled = (sed_flat * attn_weights).sum(dim=1)  # (B, C)
        return pooled, sed_logits


# Alias used in the Kaggle notebook
BirdSEDModel = BirdCLEFSED


def load_checkpoint(
    path: str,
    num_classes: int = 206,
    backbone_name: str = "efficientnet_b0",
    device: str | torch.device = "cpu",
) -> BirdCLEFSED:
    """Load a saved state_dict into a fresh BirdCLEFSED model.

    Does not download ImageNet backbone weights — the checkpoint is complete.
    """
    model = BirdCLEFSED(
        num_classes=num_classes,
        backbone_name=backbone_name,
        pretrained=False,
    )
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model
