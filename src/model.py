"""EfficientNet-B0 SED model with attention pooling.

Architecture matches the Kaggle training notebook (``BirdSEDModel``):
EfficientNet-B0 (1-ch) → 1×1 SED head → temporal attention pooling.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class BirdCLEFModelV3(nn.Module):
    """v3 SED model (based on the 4th-place BirdCLEF+ 2025 solution).

    Architecture
    ------------
    - ``bn0``: BatchNorm2d over the mel axis (their "2D batch normalization")
    - Backbone: timm model (EfficientNet family), drop_rate + drop_path_rate
    - Frequency mean → channel smoothing (max+avg pool) → fc1+ReLU
    - PANNs-style ``AttBlock`` attention pooling → clipwise + framewise logits

    Forward returns
    ---------------
    clipwise : (B, num_classes) logits
    framewise: (B, num_classes, T) frame-level logits
    """

    def __init__(
        self,
        num_classes: int = 206,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        n_mels: int = 256,
        dropout: float = 0.5,
        drop_rate: float = 0.2,
        drop_path_rate: float = 0.5,
        in_chans: int = 1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.n_mels = n_mels

        self.bn0 = nn.BatchNorm2d(n_mels)

        base = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
            global_pool="",
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )
        layers = list(base.children())[:-2]
        self.encoder = nn.Sequential(*layers)
        self.in_features = int(base.num_features)

        self.fc1 = nn.Linear(self.in_features, self.in_features, bias=True)
        self.att_block = AttBlock(self.in_features, num_classes, activation="linear")
        self.dropout_rate = dropout

        self._init_weight()

    def _init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, n_mels, T)
        x = x.transpose(1, 2)          # (B, n_mels, C, T)
        x = self.bn0(x)
        x = x.transpose(1, 2)          # (B, C, n_mels, T)

        x = self.encoder(x)            # (B, C', F', T')

        # Pool frequency axis → (B, C', T')
        x = torch.mean(x, dim=2)

        # Channel smoothing (max + avg pool) — 4th-place trick
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        clipwise_output, _, framewise_output = self.att_block(x)
        return clipwise_output, framewise_output


# --- PANNs-style helpers for the v3 head -------------------------------------

def init_layer(layer: nn.Module) -> None:
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias") and layer.bias is not None:
        layer.bias.data.fill_(0.0)


def init_bn(bn: nn.Module) -> None:
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class AttBlock(nn.Module):
    """Attention block (PANNs / 4th-place solution).

    ``att`` produces softmax attention over time; ``cla`` produces class logits
    per frame; the attention-weighted sum of frame logits is the clipwise output.
    """

    def __init__(self, in_features: int, out_features: int, activation: str = "linear", temperature: float = 1.0):
        super().__init__()
        self.activation = activation
        self.temperature = temperature
        self.att = nn.Conv1d(in_features, out_features, kernel_size=1, bias=True)
        self.cla = nn.Conv1d(in_features, out_features, kernel_size=1, bias=True)
        self.bn_att = nn.BatchNorm1d(out_features)
        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)
        init_bn(self.bn_att)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (B, C, T)
        norm_att = torch.softmax(torch.clamp(self.att(x) / self.temperature, -10, 10), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        pooled = torch.sum(norm_att * cla, dim=2)
        return pooled, norm_att, cla

    def nonlinear_transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "linear":
            return x
        if self.activation == "sigmoid":
            return torch.sigmoid(x)
        raise NotImplementedError(f"activation {self.activation}")


def load_checkpoint(
    path: str,
    num_classes: int = 206,
    backbone_name: str = "efficientnet_b0",
    device: str | torch.device = "cpu",
    model_version: str = "v1",
    n_mels: int = 256,
    dropout: float = 0.5,
) -> nn.Module:
    """Load a saved state_dict into a fresh model (v1 ``BirdCLEFSED`` or v3 ``BirdCLEFModelV3``).

    Does not download ImageNet backbone weights — the checkpoint is complete.
    """
    if model_version in ("v3", "3"):
        model: nn.Module = BirdCLEFModelV3(
            num_classes=num_classes,
            backbone_name=backbone_name,
            pretrained=False,
            n_mels=n_mels,
            dropout=dropout,
        )
    else:
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
