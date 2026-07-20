"""EfficientNet-B0 SED model with attention pooling."""

from __future__ import annotations

import torch
import torch.nn as nn
import timm


class BirdCLEFSED(nn.Module):
    """Multi-label Sound Event Detection network.

    Architecture
    ------------
    - Backbone: EfficientNet-B0 (1-channel mel input, no classifier head)
    - SED head: 1x1 conv bottleneck → class framewise logits
    - Attention pooling over time → clip-level multi-label logits

    Forward returns
    ---------------
    clipwise : (B, num_classes)  logits for multi-label classification
    framewise: (B, num_classes, T) frame-level logits (useful for SED viz)
    """

    def __init__(
        self,
        num_classes: int = 206,
        backbone_name: str = "tf_efficientnet_b0",
        pretrained: bool = True,
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
        # EfficientNet-B0 feature channels before classifier
        self.sed_head = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )
        # Attend over time using class logits as features (B, T, C)
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
        feat = self.backbone(x)  # (B, 1280, H, W)
        logits_map = self.sed_head(feat)  # (B, C, H, W)
        # Pool frequency axis → framewise class scores
        framewise = logits_map.mean(dim=2)  # (B, C, T)
        x_t = framewise.transpose(1, 2)  # (B, T, C)
        att = torch.softmax(self.attention(x_t), dim=1)  # (B, T, 1)
        clipwise = (x_t * att).sum(dim=1)  # (B, C)
        return clipwise, framewise


def load_checkpoint(
    path: str,
    num_classes: int = 206,
    device: str | torch.device = "cpu",
) -> BirdCLEFSED:
    """Load a saved state_dict into a fresh BirdCLEFSED model.

    Does not download ImageNet backbone weights — the checkpoint is complete.
    """
    model = BirdCLEFSED(num_classes=num_classes, pretrained=False)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model
