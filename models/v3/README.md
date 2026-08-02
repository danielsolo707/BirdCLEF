# models/v3 — v3 checkpoints

**Run exp001** (4th-place-style recipe) — new champion, beats v1.

| File | Role |
|------|------|
| `model_best.pth` | Best checkpoint (epoch 15, val macro ROC-AUC **0.9694**) |
| `model_last.pth` | End of training (epoch 20, val AUC 0.9670) |

EfficientNet-B0 backbone, 1 input channel, 206-class AttBlock SED head.
Load with `BirdCLEFModelV3(backbone_name="efficientnet_b0", n_mels=256)` — see `src/model.py`.
Run card: `../results/v3/README.md`.
