# Reproducibility card — v1 baseline (AUC 0.8529)

Frozen artifact set for the first published BirdCLEF SED outcome.  
Parent portfolio card (living docs): [`../../results/REPRODUCIBILITY.md`](../../results/REPRODUCIBILITY.md)

## Published run

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
| Checkpoint | [`model.pth`](./model.pth) |
| Config snapshot | [`config.json`](./config.json) |
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

With competition data locally, from repo root:

```bash
python -m src.evaluate \
  --checkpoint results/artifacts/v1_baseline_auc08529/model.pth \
  --config results/artifacts/v1_baseline_auc08529/config.json \
  --metadata path/to/val_or_train.csv \
  --audio-dir path/to/train_audio
```

Re-training with current root code produces **new** checkpoints under `runs/` (or future `artifacts/v2_*`) and will not bit-match this freeze unless the full Kaggle environment, data order, and seeds match.

## What is *not* claimed

- Public competition leaderboard placement  
- Perfect per-species performance (macro average can hide weak rare classes)  
- Bit-identical multi-machine reproducibility without locked CUDA/cuDNN/data snapshots  
