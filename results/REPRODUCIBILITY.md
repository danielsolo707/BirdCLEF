# Reproducibility card — BirdCLEF SED

## Versioning

| Run | Location | Notes |
|-----|----------|--------|
| **v1 baseline (frozen)** | [`../artifacts/v1_baseline_auc08529/`](../artifacts/v1_baseline_auc08529/) | First published outcome; do not overwrite |
| Living published checkpoint | [`../models/model.pth`](../models/model.pth) | Same weights as v1 until a newer run is promoted |
| Living config (v1) | [`../configs/config.json`](../configs/config.json) | |
| Labels | [`../labels/`](../labels/) | `classes.json`, `label2id.json` |

Git tag (local until push): **`v1.0.0`**

## Published run (shipped `models/model.pth` = v1)

| Field | Value |
|-------|--------|
| Run id | `v1_baseline_auc08529` |
| Task | Multi-label species presence from 5 s audio |
| Classes | 206 |
| Metric | Macro ROC-AUC |
| Best validation score | **0.8529** |
| Train / val sizes | 22,851 / 5,713 (stratified on primary label, 80/20) |
| Total labeled clips | 28,564 |
| Hardware | NVIDIA Tesla T4 (Kaggle) |
| Framework | PyTorch + timm + librosa |
| Backbone | `efficientnet_b0`, `in_chans=1`, ImageNet pretrained init |
| Head | SED 1×1 conv + temporal attention pooling |
| Loss | BCEWithLogitsLoss |
| Optimizer | AdamW, lr=1e-3, weight_decay=1e-5 |
| Schedule | CosineAnnealingLR, 10 epochs |
| Batch size | 16 (val loader ×2) |
| Seed | 42 (local training script default) |
| Source notebook | https://www.kaggle.com/code/danielsolo1770/notebookeb002d87be |
| Config snapshot | [`../configs/config.json`](../configs/config.json) or frozen [`../artifacts/v1_baseline_auc08529/config.json`](../artifacts/v1_baseline_auc08529/config.json) |
| Training summary | [`training_summary.json`](./training_summary.json) |
| Checkpoint SHA256 | `775CE724E2DC08BAC033FC8E1C23957995B641171BAE0DCE58106C4C51E7FC74` |

## Audio features

| Param | Value |
|-------|------:|
| Sample rate | 32000 Hz |
| Duration | 5 s (middle crop / pad) |
| n_fft | 1024 |
| hop_length | 64 |
| n_mels | 128 |
| fmin / fmax | 50 / 16000 |
| Time width | 256 |
| Normalization | log-mel min-max to [0, 1] |

## How to re-run evaluation

With competition data locally:

```bash
python -m src.evaluate \
  --checkpoint models/model.pth \
  --config configs/config.json \
  --metadata path/to/val_or_train.csv \
  --audio-dir path/to/train_audio
```

Or the frozen v1 paths:

```bash
python -m src.evaluate \
  --checkpoint artifacts/v1_baseline_auc08529/model.pth \
  --config artifacts/v1_baseline_auc08529/config.json \
  --metadata path/to/val_or_train.csv \
  --audio-dir path/to/train_audio
```

Re-training produces **new** checkpoints under `runs/` (or future `artifacts/v2_*`) and will not bit-match `models/model.pth` / the v1 freeze unless the full Kaggle environment, data order, and seeds match.

## What is *not* claimed

- Public competition leaderboard placement  
- Perfect per-species performance (macro average can hide weak rare classes)  
- Bit-identical multi-machine reproducibility without locked CUDA/cuDNN/data snapshots  
