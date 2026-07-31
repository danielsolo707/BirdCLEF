# BirdCLEF v3 — 4th-place-style recipe (Soft AUC + 10 s + semi-supervised)

**Goal:** beat **v1 val macro ROC-AUC = 0.8529** by adopting the closest top-5
solution to our codebase: **4th place (Dylan Liu)** — see [`docs/V3_PLAN.md`](./V3_PLAN.md)
for the full comparison and rationale.

> The old v3 (aggressive dual-T4 EfficientNet-B3) was **completely removed**.
> This is the new v3.

## What changed vs v1/v2

| | v1 / v2 | **v3** |
|--|---------|--------|
| Duration | 5 s (middle crop) | **10 s** (random window train / first window val) |
| Mel | log-mel 128×256, min-max | **power mel 256×512** (n_fft 2048, hop 64, n_mels 256, fmin 60), `power_to_db` at load |
| Model | B0 SED + Linear attention | **bn0 + timm backbone + freq-mean + channel smoothing + PANNs AttBlock** |
| Loss | BCE / BCE+pos_weight | **SoftAUCLoss** (metric-aligned, overfitting-resistant; AUCLoss / FocalBCELoss options) |
| Rare classes | pos_weight only | **upsampling to ≥100 samples/class** |
| Mixup | none | **spec mixup** (α=0.5, p=1.0) |
| CV | single 80/20 split | **k-fold StratifiedKFold** (default 1, recommend 5) |
| Semi-supervised | none | **pseudo-labeled train_soundscapes** via `v3.pseudo_label` |
| Inference | single clip | **overlap-TTA + smoothing + post-processing** flags |

## Pipeline

```
train_audio (10 s) ──► power mel 256×512 (precompute_mels_v3.py)
        │
        ▼
BirdCLEFModelV3: bn0(BN over mels) → timm encoder (drop_path) → freq-mean
        → channel smoothing (max+avg) → fc1 → AttBlock → clipwise + framewise
        │
        ▼
SoftAUCLoss (+ spec mixup, spec augment, rare upsampling) → k-fold CV
        │
        ▼
Stage 2 (optional): pseudo_label.py → teacher ensemble predicts soundscapes
        → soft-label chunks → train_v3 --semi-csv
        │
        ▼
Inference: --overlap (sliding 10 s window, 2.5 s stride) + --smooth + --postprocess
        → src.ensemble to blend folds/models
```

## Local (this PC: RTX 4070 8.6 GB)

```bash
# 1. precompute mels (once)
python -m v3.precompute_mels \
  --audio-dir data/train_audio --output-dir data/mels_v3 --config v3/config.json

# 2. train (single fold, small batch for 8 GB)
python -m v3.train \
  --config v3/config.json \
  --metadata data/train.csv --mel-dir data/mels_v3 \
  --output-dir runs/v3_exp001 --fold 0 --folds 5 --batch-size 24

# 3. pseudo-label soundscapes (after stage 1 finishes)
python -m v3.pseudo_label \
  --config v3/config.json \
  --checkpoints runs/v3_exp001/model_fold0_best.pth,runs/v3_exp001/model_fold1_best.pth \
  --soundscapes-dir data/train_soundscapes \
  --mel-dir data/mels_v3_semi --output-csv runs/pseudo/semi_chunks.csv --threshold 0.3

# 4. re-train with pseudo-labels
python -m v3.train \
  --config v3/config.json \
  --metadata data/train.csv --semi-csv runs/pseudo/semi_chunks.csv \
  --mel-dir data/mels_v3 --output-dir runs/v3_semi_exp001 --fold 0 --folds 5
```

## Kaggle (2× T4, full recipe)

- Accelerator: **GPU T4 x2** (or T4 x1 with `--batch-size 16`)
- Attach competition data + this repo as a code pack
- Batch 96 on 2× T4 (batch 48-64 on 1× T4); EfficientNet-B3 (`--backbone efficientnet_b3`) if VRAM allows
- 20 epochs, LR 5e-4, SoftAUCLoss, mixup, 5-fold
- Download `model_fold*_best.pth` + `metrics.json` per fold

## Memory / OOM ladder

1. `--batch-size 16`
2. `--backbone efficientnet_b0` (default) → `efficientnet_b1`
3. `TARGET_TIME 384` (drop temporal resolution)
4. `CHANNELS_LAST: false`

## Sanity

```bash
python v3/smoke_test.py        # tiny synthetic end-to-end on local GPU
```

## Result tracking

- Checkpoints per fold: `runs/<exp>/model_fold{fold}_best.pth`
- Metrics: `runs/<exp>/metrics.json` (mean fold AUC vs v1 baseline 0.8529)
- Promote `models/v1/model.pth` **only if** best AUC > 0.8529
