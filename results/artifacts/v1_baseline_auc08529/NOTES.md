# v1 baseline — first published outcome

**Frozen run id:** `v1_baseline_auc08529`  
**Git tag (local):** `v1.0.0`  
**Best metric:** validation **macro ROC-AUC = 0.8529**

## What this folder is

Immutable snapshot of the **first portfolio baseline** before accuracy / evaluation upgrades.

- Living code stays organized under `src/`, `configs/`, `labels/`, `models/`.
- This directory is a **run freeze**: checkpoint + config + label maps + metrics for honest before/after comparison.

## Contents

| File | Role |
|------|------|
| `model.pth` | Best checkpoint from the Kaggle T4 run |
| `config.json` | Audio + training hyperparameters used for this run |
| `classes.json` | 206 species label list |
| `label2id.json` | Label → class index map |
| `training_summary.json` | Machine-readable run summary |
| `REPRODUCIBILITY.md` | Human-readable card for this run only |
| `NOTES.md` | This file |

## Checkpoint integrity

| Field | Value |
|-------|--------|
| File | `model.pth` |
| Size (bytes) | 17,971,077 |
| SHA256 | `775CE724E2DC08BAC033FC8E1C23957995B641171BAE0DCE58106C4C51E7FC74` |

## Baseline recipe (short)

- **Task:** multi-label presence of 206 bird species from 5 s audio  
- **Features:** log-mel 128×256, min-max to [0, 1], precomputed `.npy` in original training  
- **Model:** EfficientNet-B0 (`timm`, `in_chans=1`) + SED head + temporal attention  
- **Loss / optim:** BCEWithLogitsLoss, AdamW lr=1e-3, wd=1e-5, cosine 10 epochs, AMP  
- **Split:** single 80/20 stratified on **primary label** (not grouped by site)  
- **Augs:** light image-style (horizontal flip + color jitter on mel)  
- **Hardware:** NVIDIA Tesla T4 (Kaggle)  
- **Source notebook:** https://www.kaggle.com/code/danielsolo1770/notebookeb002d87be  

## Known limitations (why later work exists)

1. Single stratified split → metric variance / possible leakage unknown  
2. Light spectrogram augs → limited noise robustness  
3. No class re-weighting / focal / balanced sampling → rare species may lag  
4. Clip-level multi-hot only → weak time localization supervision  
5. Not a public leaderboard claim  

## How to evaluate this checkpoint later

From repo root (paths relative to `BirdCLEF/`):

```bash
python -m src.evaluate \
  --checkpoint results/artifacts/v1_baseline_auc08529/model.pth \
  --config results/artifacts/v1_baseline_auc08529/config.json \
  --metadata path/to/val_or_train.csv \
  --audio-dir path/to/train_audio
```

`models/v1/model.pth` is the same weights at freeze time; prefer this freeze path when comparing against future `results/artifacts/v2_*` runs.

## Policy

- **Do not overwrite** files in this folder when training improves.  
- New experiments → `artifacts/v2_.../` or `runs/<exp_id>/`.  
- Only replace `models/v1/model.pth` when a new run is deliberately promoted as the public “best” checkpoint.
