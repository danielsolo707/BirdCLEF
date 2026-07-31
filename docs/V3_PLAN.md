# BirdCLEF v3 — Plan & Analysis (based on top-5 solutions)

> Old v3 (aggressive dual-T4 EfficientNet-B3) has been **completely removed**.
> This document defines the NEW v3: an upgrade of our v1/v2 pipeline toward the level of the top-1..5 solutions.

## 1. Which solution is closest to our code? → **4th place (Dylan Liu)**

We compared all 5 writeups against our codebase (EfficientNet-B0 timm + SED head + attention pooling, 5s @32 kHz, log-mel 128×256, hop 64, BCE loss, AdamW + cosine, AMP):

| Criterion | Our code | 1st | 2nd | 3rd | **4th** | 5th |
|---|---|---|---|---|---|---|
| SED + attention pooling | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| timm EfficientNet family | ✅ | ✅ (b0-b4) | v2_s | ✅ | ✅ (b0-b4, v2, lite) | ✅ (v2_s, b3_ns, b0_ns) |
| hop_length = 64 | ✅ | 1252 | ? | 512 | **64 (same!)** | 768 |
| No external data required | ✅ | ❌ Xeno-Canto | ❌ XC pretrain | ❌ 2023+XC+iNat | ✅ (only 2025) | ✅ (only 2025) |
| BCE-family loss | ✅ | CE | Focal+BCE | Focal BCE | custom AUC | Focal |
| Simple PyTorch pipeline | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Precomputed mels + AMP | ✅ | ✅ | ? | ✅ | ✅ | ✅ |

**Decision: 4th place.** It shares our SED paradigm, our exact hop_length (64), the same timm EfficientNet ecosystem, and needs no extra data. Its magic ingredients — **Soft AUC loss** (optimizes the metric directly, overfitting-resistant, supports soft labels) and **semi-supervised learning on soundscapes** — are exactly the upgrades our pipeline lacks. Its code is public (we saved it in `docs/writeups/4th_place_code/`).

Cheap wins borrowed from others (no added complexity):
- **1st place**: overlapping framewise TTA at inference (sliding window over chunks)
- **5th place**: Focal BCE loss option, `log(mel+1e-6)` normalization option
- **2nd place**: post-processing (multiply chunk probs by top prob per class per file)

## 2. What the 4th place does (recap)

- SED model: `bn0` (BatchNorm2d over mel axis) → timm encoder (drop_rate 0.2, drop_path 0.5) → freq mean → channel smoothing (max+avg pool) → fc1+ReLU → PANNs-style `AttBlock` attention → clipwise + framewise logits
- Data: 10 s audio (first-10s or random-10s), power mel (n_fft 2048, hop 64, n_mels 256, fmin 60, fmax 16000) resized to (256, 512), `power_to_db` at load
- Loss: `AUCLoss` (hard labels) / `SoftAUCLoss` (soft labels) — pairwise ranking loss
- Training: 20-25 epochs, AdamW 5e-4, cosine, AMP, batch 32, spec mixup, spec augment (time/freq masking + gain), rare-class upsampling to ≥100, 6-fold CV
- Semi-supervised: teacher ensemble (10 models) predicts soundscapes → soft pseudo-labels added to training (drop some semi samples per epoch)
- Results: single tf_efficientnetv2_b0 went 0.850 → **0.901** LB with SoftAUC + SSL

## 3. New v3 pipeline (what we build)

```
Audio (10 s @32 kHz, random/first segment)
  → power mel (n_fft 2048, hop 64, n_mels 256, fmin 60, fmax 16000)
  → cv2 resize → (256, 512) → power_to_db at load
  → BN over freq → timm backbone (efficientnet_b0..b4 / v2_b0..b3 / v2_s / lite0-4)
  → freq mean → channel smoothing → fc1 → AttBlock attention
  → clipwise logits → SoftAUCLoss (hard) / FocalBCE (option)

Optional Stage 2 (semi-supervised):
  teacher ensemble predicts train_soundscapes → soft pseudo-labels
  → train on audio + pseudo-labeled soundscape chunks (mixup / ratio sampling)
```

## 4. File plan (all local, no push)

| File | Action |
|---|---|
| `src/train_v2.py` → `v2/train.py` | v2 training (same model, better recipe) |
| `src/train_v3.py` → `v3/train.py` | v3 training (4th-place-style) |
| `src/pseudo_label.py` → `v3/pseudo_label.py` | v3 semi-supervised pseudo-labeling |
| `scripts/precompute_mels.py` → `src/precompute_mels.py` | v1/v2 mel cache (shared) |
| `scripts/precompute_mels_v3.py` → `v3/precompute_mels.py` | v3 power-mel cache |
| `scripts/smoke_test_v2.py` → `v2/smoke_test.py` | v2 sanity |
| `scripts/smoke_test_v3.py` → `v3/smoke_test.py` | v3 sanity |
| `configs/config.json` → `v1/config.json` | v1 hyperparameters |
| `configs/config_v2.json` → `v2/config.json` | v2 hyperparameters |
| `configs/config_v3.json` → `v3/config.json` | v3 hyperparameters |
| `notebooks/birdclef_v2_kaggle.ipynb` → `v2/kaggle.ipynb` | v2 Kaggle notebook |
| `models/model.pth` → `models/v1/model.pth` | v1 checkpoint |
| `artifacts/` → `results/artifacts/` | frozen experiment snapshots |

## 5. Expected gains (from 4th place writeup)

| Step | LB gain |
|---|---|
| Baseline (our v1 recipe, 5 s) | 0.8529 val (our number) |
| + 10 s + n_mels 256 + new head + SoftAUC | ~+0.02-0.03 (their B0 jump) |
| + rare upsampling + mixup + folds | further |
| + pseudo-label soundscapes (SSL) | 0.850 → 0.901 single model (their number) |
| + TTA overlap + post-processing + ensemble | +0.005-0.01 |

Target: comfortably beat our 0.8529 validation AUC and reach the 0.88-0.90+ region.
