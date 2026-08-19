# BirdCLEF+ 2025 — Sound Event Detection

Multi-label bird species classification (206 species) with a full **versioned** pipeline:
**v1** clean SED baseline → **v2** failed recipe experiment → **v3** system upgrade (champion).

| | |
|---|---|
| **Task** | Multi-label audio classification (206 species) |
| **Champion** | **v3** — val macro ROC-AUC **0.9694** ([`models/v3/model_best.pth`](./models/v3/model_best.pth)) |
| **v1 baseline** | 0.8529 — first published SED (log-mel 5 s) |
| **v2 result** | 0.8401 — did **not** beat v1 (heavy reweight / sampler) |
| **v3 recipe** | Soft AUC · 10 s · power-mel 256×512 · AttBlock · mixup · rare upsampling |
| **Stack** | PyTorch · timm · librosa · torchvision |
| **Hardware** | RTX 4070 (local champion run) · Tesla T4 / T4×2 (Kaggle) |
| **v1 source notebook** | [danielsolo1770/notebookeb002d87be](https://www.kaggle.com/code/danielsolo1770/notebookeb002d87be) |
| **License** | [MIT](./LICENSE) |

> **How to read v3:** it is a **system upgrade** (features + model head + loss + augs), not “same model, only loss changed.” See [Honest limitations](#honest-limitations--roadmap) below.

---

## Repository layout

```
BirdCLEF/
├── README.md                 # you are here
├── LICENSE
├── requirements.txt
│
├── src/                      # shared library (all versions import from here)
│   ├── model.py              # BirdCLEFSED (v1/v2) + BirdCLEFModelV3 (bn0+AttBlock)
│   ├── audio.py              # mel extraction (v1 log-mel + v3 power-mel)
│   ├── dataset.py            # BirdCLEFDataset + BirdCLEFDatasetV3
│   ├── losses.py             # AUCLoss / SoftAUCLoss / FocalBCELoss
│   ├── metrics.py            # AUC / F1 / precision / recall
│   ├── utils.py              # paths, configs, seeding
│   ├── evaluate.py           # shared eval CLI
│   ├── inference.py          # shared inference CLI (v1 + v3 overlap-TTA flags)
│   ├── ensemble.py           # blend prediction CSVs
│   └── precompute_mels.py    # v1/v2 log-mel cache (shared)
│
├── v1/                       # version 1 — EfficientNet-B0 SED baseline (AUC 0.8529)
│   ├── train.py              # v1 training
│   ├── config.json           # v1 hyperparameters
│   └── __init__.py
│
├── v2/                       # version 2 — same model, better recipe (0.8401)
│   ├── train.py              # v2 training
│   ├── config.json           # v2 hyperparameters
│   ├── kaggle.ipynb          # Kaggle notebook template
│   ├── smoke_test.py         # fast local sanity
│   └── __init__.py
│
├── v3/                       # version 3 — champion (val AUC 0.9694)
│   ├── train.py              # v3 training
│   ├── config.json           # v3 hyperparameters
│   ├── train_local.sh        # one-shot local launcher (mels + train)
│   ├── pseudo_label.py       # pseudo-label train_soundscapes (Stage 2)
│   ├── precompute_mels.py    # v3 power-mel cache
│   ├── smoke_test.py         # v3 end-to-end sanity
│   └── __init__.py
│
├── labels/                   # class maps (206 species)
│   ├── classes.json
│   └── label2id.json
│
├── models/                   # per-version checkpoints
│   ├── README.md
│   ├── v1/model.pth          # v1 best (val AUC 0.8529)
│   ├── v2/model_best.pth     # v2 best (0.8401) + model_last.pth
│   └── v3/model_best.pth     # v3 champion (0.9694) + model_last.pth
│
├── results/                  # per-version run results
│   ├── v1/                   # training_summary.json + REPRODUCIBILITY.md + NOTES.md
│   ├── v2/                   # metrics.json + per_class_metrics.csv + config_used.json
│   └── v3/                   # metrics.json + per_class_metrics.csv + config_used.json
│
└── data/                     # local data only (gitignored except README)
    ├── README.md
    ├── train.csv             # 28,564 labeled clips · 206 species
    ├── train_audio/          # labeled ogg clips (7.3 GB)
    ├── train_soundscapes/    # 9,726 unlabeled recordings (4.4 GB, Stage 2)
    └── mels_v3/              # precomputed v3 power-mels (15 GB)
```

---

## Why this project

BirdCLEF is a realistic bioacoustics challenge: short clips, **multi-label** targets, class imbalance, and spectrogram CNNs. This repo is a **portfolio-style experiment log**:

1. **v1** — clean baseline (val AUC 0.8529)  
2. **v2** — aggressive imbalance tricks that **underperformed** (0.8401)  
3. **v3** — competition-informed **system upgrade** that **won** (0.9694)

**Portfolio note:** Flagship supervised deep-learning project — full pipeline, honest metrics, frozen run cards, runnable CLIs.

---

## Architecture

### v3 champion (current best)

```
Audio (10 s @ 32 kHz; random window train / first window val)
    → power-mel 256×512  (n_fft 2048, hop 64, n_mels 256, fmin 60)
    → power_to_db at load
    → bn0 + EfficientNet-B0 (timm, drop_path) + freq-mean
    → channel smoothing + PANNs-style AttBlock
    → clip-level multi-label logits → sigmoid
```

| Component | v3 choice |
|-----------|-----------|
| Loss | **SoftAUCLoss** (metric-aligned; soft labels OK for Stage 2) |
| Optimizer / schedule | AdamW · CosineAnnealingLR |
| Precision | AMP |
| Augs | Spec mixup (α=0.5) · SpecAugment · rare-class upsample (≥100) |
| Speed | Precomputed power-mel cache (`data/mels_v3/`) |

### v1 baseline (historical)

```
Audio (5 s, middle crop) → log-mel 128×256 → EfficientNet-B0 SED
    → 1×1 SED head → temporal attention → multi-label logits
```

| Component | v1 / v2 |
|-----------|---------|
| Loss | BCE · v2: BCE + `pos_weight` + balanced sampler |
| Features | log-mel 5 s, min-max to [0, 1] |

---

## Results

| Version | Recipe | Val macro ROC-AUC | Macro F1 (best ep) | Δ vs v1 |
|---------|--------|------------------:|-------------------:|--------:|
| v1 | SED baseline, log-mel 128×256 | 0.8529 | (weak @ thr=0.5) | — |
| v2 | same model + SpecAugment + heavy reweight | 0.8401 | ~0.10 | −0.013 |
| **v3** | **full system upgrade (see above)** | **0.9694** | **~0.53** | **+0.117** |

- Run cards: [`results/v1/`](./results/v1/) · [`results/v2/`](./results/v2/) · [`results/v3/`](./results/v3/)  
- Checkpoints: [`models/v1/`](./models/v1/) · [`models/v2/`](./models/v2/) · [`models/v3/`](./models/v3/)  
- v3 best epoch: **15** (0.9694); ep20 last: 0.9670 · PR-AUC ~0.65  

**Metric note:** macro ROC-AUC is **ranking quality**, not accuracy at a fixed threshold.  
**Soft AUC note:** v3 optimizes a soft surrogate of AUC — that *helps* the headline metric by design. Supporting evidence that learning is real: **macro F1 ~0.53** and **macro PR-AUC ~0.65** (v2 was ~0.10 F1 / ~0.18 PR-AUC).  

Validation uses a **random stratified 80/20 on `primary_label`** (same style as v1/v2) — fair **across versions**, **optimistic vs Kaggle LB** (not grouped by site/recording).

### Honest limitations & roadmap

| Caveat | Status | What we do about it |
|--------|--------|---------------------|
| v3 ≠ pure ablation of v1 | **Documented** | Call it a **system upgrade** (features + head + loss + augs), not “only Soft AUC” |
| SoftAUC optimizes AUC | **Documented** | Always report **F1 + PR-AUC** beside AUC |
| Single fold only (champion) | **Code ready** | Run `bash v3/train_5fold.sh` → mean fold AUC for credibility |
| Stage 2 pseudo-labels unused | **Code ready** | `bash v3/stage2_pseudo.sh` after Stage 1 folds exist |
| Split not site-grouped | **Accepted for now** | Same split style as v1/v2 for fair version compare; site-grouped CV is future work |

Details + exact commands: [`results/v3/NEXT_STEPS.md`](./results/v3/NEXT_STEPS.md).

---

## Quick start

```bash
pip install -r requirements.txt
```

### Continuous integration (CPU-only)

GitHub Actions installs the runtime dependencies, compiles the versioned source tree, and verifies the core imports on every push and pull request. It intentionally does **not** download competition data, load checkpoints, train models, or require a GPU; full training and validation remain explicit local or Kaggle runs.

### Inference (champion = v3)

```bash
python -m src.inference \
  --audio path/to/clip.ogg \
  --checkpoint models/v3/model_best.pth \
  --config v3/config.json \
  --model-version v3 \
  --top-k 5
```

v1 weights still work with `--checkpoint models/v1/model.pth --config v1/config.json` (omit `--model-version v3`).

### Evaluate

```bash
# v3
python -m src.evaluate \
  --checkpoint models/v3/model_best.pth \
  --config v3/config.json \
  --metadata path/to/val.csv \
  --mel-dir path/to/mels_v3
```

### Train v1

```bash
python -m v1.train \
  --config v1/config.json \
  --metadata path/to/train.csv \
  --audio-dir path/to/train_audio \
  --mel-dir path/to/mels \
  --output-dir runs/v1_exp
```

### Train v2 (stronger recipe)

```bash
python -m v2.train \
  --config v2/config.json \
  --metadata path/to/train.csv \
  --mel-dir path/to/mels \
  --output-dir runs/v2_exp
```

See [`docs/V2_TRAINING.md`](./docs/V2_TRAINING.md) for the full v2 guide (also running on Kaggle).

### Train v3 (4th-place-style recipe — champion)

```bash
# 1. precompute v3 power-mels (10 s, 256×512) — 4-way parallel ≈ 16 min on 20 cores
python -m v3.precompute_mels \
  --audio-dir data/train_audio --output-dir data/mels_v3 \
  --config v3/config.json --metadata data/train.csv   # 4 procs × --metadata slice_N.csv

# 2. train Stage 1 (champion freeze used single fold; prefer 5-fold for credibility)
bash v3/train_local.sh          # single fold → runs/v3_exp001
# bash v3/train_5fold.sh        # all 5 folds → runs/v3_5fold (long)

# 3. optional Stage 2 (pseudo-label soundscapes → re-train)
bash v3/stage2_pseudo.sh        # needs Stage-1 fold checkpoints

# 4. overlap-TTA inference
python -m src.inference \
  --audio path/to/soundscape.ogg --checkpoint models/v3/model_best.pth \
  --config v3/config.json --model-version v3 --overlap --smooth --postprocess
```

Roadmap for 5-fold + Stage 2: [`results/v3/NEXT_STEPS.md`](./results/v3/NEXT_STEPS.md).

### Precompute mels (v1/v2)

```bash
python -m src.precompute_mels \
  --audio-dir path/to/train_audio \
  --output-dir path/to/mels \
  --config v1/config.json
```

---

## Design choices (short)

1. **SED + attention** — bird calls are sparse in time; attention focuses useful frames.  (v3: PANNs-style `AttBlock`)
2. **Precomputed mels** — large speedup for multi-epoch training.  (v3: power-mel 256×512, db at load)
3. **Min-max log-mel in [0, 1] + middle crop** — train/serve parity with the notebook that produced `models/v1/model.pth`.  (v3: 10 s random window + `power_to_db`)
4. **Per-version folders (`v1/`, `v2/`, `v3/`)** — shared library lives in `src/`; each version keeps its train entry point, config, and tools together. Checkpoints → `models/vN/`, run results → `results/vN/`.
5. **`results/v1|v2|v3/` freezes** — immutable per-version run cards for honest comparison; no duplicated files.
6. **v3 = system upgrade + Soft AUC** — longer context, stronger mels, AttBlock, mixup; Soft AUC aligns training with ranking (report F1/PR-AUC too). Soft labels enable Stage-2 semi-supervised learning.
7. **Honest experiment log** — v2 kept as a failed run; limitations and next steps live under `results/v3/NEXT_STEPS.md`.
