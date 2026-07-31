# 4th Place Solution (Dylan Liu / dylanliuofficial)

Source: https://www.kaggle.com/competitions/birdclef-2025/writeups/dylan-liu-4th-place-solution
GitHub: https://github.com/dylanliu2/BirdCLEF2025-4th-place-solution

## Core ideas
- **SED solution** inspired by birdclef2023 2nd place solution
- **Custom soft AUC loss** (optimizes the metric directly, very resistant to overfitting — CV worse but LB much better)
- **Semi-supervised learning** on soundscapes
- **NO pretraining** (any kind hurt!), no knowledge distillation, only efficientnet models

## Key results
- Soft AUC loss + semi-supervised: single tf_efficientnetv2_b0 0.850 → **0.901** LB
- Private improvement 11th → 4th attributed mainly to the AUC loss

## Losses
```python
class AUCLoss(nn.Module):  # hard labels
    def __init__(self, margin=1.0, pos_weight=1.0, neg_weight=1.0):
        super().__init__()
        self.margin = margin; self.pos_weight = pos_weight; self.neg_weight = neg_weight
    def forward(self, preds, labels, sample_weights=None):
        pos_preds = preds[labels == 1]; neg_preds = preds[labels == 0]
        if len(pos_preds) == 0 or len(neg_preds) == 0:
            return torch.tensor(0.0, device=preds.device)
        if sample_weights is not None:
            sample_weights = torch.stack([sample_weights]*labels.shape[1], dim=1)
            pos_weights = sample_weights[labels == 1]; neg_weights = sample_weights[labels == 0]
        else:
            pos_weights = torch.ones_like(pos_preds) * self.pos_weight
            neg_weights = torch.ones_like(neg_preds) * self.neg_weight
        diff = pos_preds.unsqueeze(1) - neg_preds.unsqueeze(0)
        loss_matrix = torch.log(1 + torch.exp(-diff * self.margin))
        weighted_loss = loss_matrix * pos_weights.unsqueeze(1) * neg_weights.unsqueeze(0)
        return weighted_loss.mean()

class SoftAUCLoss(nn.Module):  # soft labels (distillation / SSL)
    # same, but pos_preds = preds[labels>0.5], neg_preds = preds[labels<0.5]
    # pos_weights scaled by (pos_labels-0.5), neg_weights by (0.5-neg_labels)
```
- Host comment: use `torch.nn.functional.softplus` (numerically stable) instead of `log(1+exp(x))`

## Other things that helped
- **Semi-supervised**: labeling model = 10 SED models (efficientnet_b0-b4, efficientnetv2_b0-b3, efficientnetv2_s) trained on **first 10s** audio
- **Smaller hop_length (64)** and **larger n_mels (256)**
- **Audio mixup augmentation**: add two audios → new audio, take MAX of their labels as new label (didn't directly help single models but added diversity)

## Didn't help
- Any pretraining; knowledge distillation; non-efficientnet models; data normalizations other than 2D batch normalization

## Final models
- **16 models**: efficientnet_lite0-4, efficientnet_b2-3, efficientnetv2_b2-3, efficientnetv2_s
- 17-25 epochs, LR 5e-4
- 3 types of mel spectrogram parameters
- 2 types of data augmentation
- First-10s and random-10s data
