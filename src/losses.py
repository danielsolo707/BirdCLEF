"""Loss functions for BirdCLEF v3.

The core loss is the pairwise ranking loss from the 4th-place solution
(Dylan Liu, BirdCLEF+ 2025): it optimizes the competition metric (macro ROC-AUC)
directly and is very resistant to overfitting.

- AUCLoss      : hard labels (0/1)
- SoftAUCLoss  : soft labels (pseudo-labels, distillation) — pos/neg weighted by
                 how far the label is from 0.5
- FocalBCELoss : Focal BCE (gamma) with optional class pos_weight + label smoothing
                 (borrowed from the 5th-place solution style)

Numerically stable versions (host suggestion): log(1+exp(x)) is computed via
``F.softplus``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AUCLoss(nn.Module):
    """Pairwise ranking loss for hard multi-label targets (4th place)."""

    def __init__(self, margin: float = 1.0, pos_weight: float = 1.0, neg_weight: float = 1.0):
        super().__init__()
        self.margin = margin
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, preds: torch.Tensor, labels: torch.Tensor, sample_weights=None) -> torch.Tensor:
        pos_preds = preds[labels == 1]
        neg_preds = preds[labels == 0]

        if len(pos_preds) == 0 or len(neg_preds) == 0:
            return torch.tensor(0.0, device=preds.device)

        if sample_weights is not None:
            sw = torch.stack([sample_weights] * labels.shape[1], dim=1)
            pos_weights = sw[labels == 1]
            neg_weights = sw[labels == 0]
        else:
            pos_weights = torch.ones_like(pos_preds) * self.pos_weight
            neg_weights = torch.ones_like(neg_preds) * self.neg_weight

        diff = pos_preds.unsqueeze(1) - neg_preds.unsqueeze(0)  # (N_pos, N_neg)
        loss_matrix = F.softplus(-diff * self.margin)  # numerically stable log(1+exp(-d*m))

        weighted_loss = loss_matrix * pos_weights.unsqueeze(1) * neg_weights.unsqueeze(0)
        return weighted_loss.mean()


class SoftAUCLoss(nn.Module):
    """Pairwise ranking loss for SOFT labels (pseudo-labels / distillation).

    Labels above 0.5 are treated as positive, below 0.5 as negative, and their
    distance from 0.5 scales the pair weight (4th-place solution).
    """

    def __init__(self, margin: float = 1.0, pos_weight: float = 1.0, neg_weight: float = 1.0):
        super().__init__()
        self.margin = margin
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, preds: torch.Tensor, labels: torch.Tensor, sample_weights=None) -> torch.Tensor:
        pos_mask = labels > 0.5
        neg_mask = labels < 0.5
        pos_preds = preds[pos_mask]
        neg_preds = preds[neg_mask]
        pos_labels = labels[pos_mask]
        neg_labels = labels[neg_mask]

        if len(pos_preds) == 0 or len(neg_preds) == 0:
            return torch.tensor(0.0, device=preds.device)

        pos_weights = torch.ones_like(pos_preds) * self.pos_weight * (pos_labels - 0.5)
        neg_weights = torch.ones_like(neg_preds) * self.neg_weight * (0.5 - neg_labels)
        if sample_weights is not None:
            sw = torch.stack([sample_weights] * labels.shape[1], dim=1)
            pos_weights = pos_weights * sw[pos_mask]
            neg_weights = neg_weights * sw[neg_mask]

        diff = pos_preds.unsqueeze(1) - neg_preds.unsqueeze(0)  # (N_pos, N_neg)
        loss_matrix = F.softplus(-diff * self.margin)

        weighted_loss = loss_matrix * pos_weights.unsqueeze(1) * neg_weights.unsqueeze(0)
        return weighted_loss.mean()


class FocalBCELoss(nn.Module):
    """Focal BCE with optional pos_weight, label smoothing, and reduction.

    gamma=0 reproduces plain BCEWithLogitsLoss.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing

    def forward(self, preds: torch.Tensor, labels: torch.Tensor, sample_weights=None) -> torch.Tensor:
        # label smoothing: pull hard labels toward 0.5 (softens target)
        if self.label_smoothing > 0:
            labels = labels * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        p = torch.sigmoid(preds)
        # focal modulation on the hard-ish label
        pt = p * labels + (1 - p) * (1 - labels)
        focal = (1 - pt) ** self.gamma

        bce = F.binary_cross_entropy_with_logits(preds, labels, reduction="none")
        if self.pos_weight is not None:
            pw = self.pos_weight.to(preds.device)
            bce = bce * (labels * (pw - 1.0) + 1.0)  # positive samples scaled by pos_weight

        loss = focal * bce
        if sample_weights is not None:
            loss = loss * sample_weights.unsqueeze(1)
        return loss.mean()


def get_criterion(name: str, **kwargs) -> nn.Module:
    """Factory: 'AUCLoss' | 'SoftAUCLoss' | 'FocalBCELoss' | 'BCE'."""
    name = name.strip().lower()
    if name in ("auc", "aucloss"):
        return AUCLoss(**{k: v for k, v in kwargs.items() if k in ("margin", "pos_weight", "neg_weight")})
    if name in ("softauc", "softaucloss"):
        return SoftAUCLoss(**{k: v for k, v in kwargs.items() if k in ("margin", "pos_weight", "neg_weight")})
    if name in ("focal", "focalbce", "focalbceloss"):
        return FocalBCELoss(
            gamma=float(kwargs.get("gamma", 2.0)),
            pos_weight=kwargs.get("pos_weight"),
            label_smoothing=float(kwargs.get("label_smoothing", 0.0)),
        )
    if name in ("bce", "bcewithlogits"):
        return nn.BCEWithLogitsLoss(pos_weight=kwargs.get("pos_weight"))
    raise NotImplementedError(f"Unknown criterion: {name}")
