# BirdCLEF v2 training guide

**Goal:** beat **v1 val macro ROC-AUC = 0.8529** with real training upgrades (not docs-only).

**Style:** simple junior ML code — clear scripts, one config, no heavy experiment framework.

## What v2 changes (for real)

| Change | Why it can help |
|--------|------------------|
| SpecAugment (time/freq masks) | Better than only flip/jitter on mels |
| BCE `pos_weight` | Rare species get more loss weight |
| WeightedRandomSampler on `primary_label` | See rare birds more often |
| 15 epochs, LR 8e-4, dropout 0.35 | A bit more training / regularization |
| Same EfficientNet-B0 SED model | Fair comparison to v1 architecture |

## Metrics (logged every val epoch)

- macro ROC-AUC (still the **main** score vs v1)
- macro PR-AUC
- micro/macro F1, precision, recall @ threshold 0.5
- hamming loss, element accuracy (secondary — accuracy alone is misleading)

## Files (after folder organize)

| Path | Role |
|------|------|
| `v2/config.json` | v2 hyperparameters |
| `v2/train.py` | local CLI training |
| `src/metrics.py` | AUC / F1 / precision / recall |
| `src/dataset.py` | SpecAugment + cfg transforms |
| `v2/kaggle.ipynb` | Kaggle notebook template |
| `v2/smoke_test.py` | fast local sanity checks |
| `results/artifacts/v1_baseline_auc08529/` | frozen first outcome (do not overwrite) |
| `models/v1/model.pth` | published v1 checkpoint |

## Local smoke test (no full train)

```bash
python v2/smoke_test.py
```

## Local train (if you have data)

```bash
python -m v2.train \
  --config v2/config.json \
  --metadata path/to/train.csv \
  --mel-dir path/to/mels \
  --output-dir runs/v2_exp001
```

## Kaggle (already set up)

- Code dataset: https://www.kaggle.com/datasets/danielsolo1770/birdclef-v2-code-pack  
- Train notebook: https://www.kaggle.com/code/danielsolo1770/birdclef-v2-train  

When finished, download:

- `model_best.pth`
- `metrics.json`
- `per_class_metrics.csv`

Then freeze under `results/artifacts/v2_auc0xxxx/` and compare to v1.

## After a good run

Only replace `models/v1/model.pth` if v2 **beats 0.8529** on a fair comparison.

## Honest note

These upgrades usually help, but **no guarantee** of a higher AUC on the first try. If v2 is flat or worse: try without balanced sampler, lower `POS_WEIGHT_MAX`, or train longer.
