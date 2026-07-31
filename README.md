# BirdCLEF+ 2025 — Sound Event Detection

Multi-label bird species classification from 5-second audio clips using an **EfficientNet-B0** backbone and a **Sound Event Detection (SED)** head with **attention pooling**.

| | |
|---|---|
| **Task** | Multi-label audio classification (206 species) |
| **Best validation AUC** | **0.8529** (macro ROC-AUC) — v1 baseline (still champion) |
| **v2 result** | 0.8401 — did not beat v1 |
| **v3 recipe** | 4th-place-style: Soft AUC loss · 10 s · power-mel 256×512 · AttBlock · semi-supervised (see `docs/V3_TRAINING.md`) |
| **Stack** | PyTorch · timm · librosa · torchvision |
| **Hardware** | Tesla T4 / T4×2 (Kaggle) · RTX 4070 (local) |
| **Checkpoint** | [`models/v1/model.pth`](./models/v1/model.pth) |
| **Source notebook** | [danielsolo1770/notebookeb002d87be](https://www.kaggle.com/code/danielsolo1770/notebookeb002d87be) |
| **v2 Kaggle train** | [danielsolo1770/birdclef-v2-train](https://www.kaggle.com/code/danielsolo1770/birdclef-v2-train) |
| **License** | [MIT](./LICENSE) |

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
├── v3/                       # version 3 — 4th-place-style recipe (Soft AUC + 10s + semi)
│   ├── train.py              # v3 training
│   ├── config.json           # v3 hyperparameters
│   ├── pseudo_label.py       # pseudo-label train_soundscapes (Stage 2)
│   ├── precompute_mels.py    # v3 power-mel cache
│   ├── smoke_test.py         # v3 end-to-end sanity
│   └── __init__.py
│
├── labels/                   # class maps (206 species)
│   ├── classes.json
│   └── label2id.json
│
├── models/                   # published checkpoints
│   └── v1/
│       └── model.pth         # v1 best (val AUC 0.8529)
│
├── results/                  # run cards + frozen experiment snapshots
│   ├── training_summary.json
│   ├── REPRODUCIBILITY.md
│   └── artifacts/            # frozen snapshots (do not overwrite)
│       └── v1_baseline_auc08529/
│
├── docs/                     # guides + top-5 solution writeups (local reference)
│   ├── V2_TRAINING.md
│   ├── V3_PLAN.md            # solution comparison + why 4th place
│   ├── V3_TRAINING.md        # v3 recipe
│   ├── interview/            # local-only interview prep (gitignored)
│   └── writeups/             # 1st-5th place solutions + 4th-place code
│
└── data/                     # local data only (gitignored except README)
    └── README.md
```

---

## Why this project

BirdCLEF is a realistic bioacoustics challenge: short clips, **multi-label** targets, class imbalance, and spectrogram-based CNN modeling. This repo packages the **training / evaluation / inference code** aligned with the Kaggle notebook that produced val AUC **0.8529**, plus the released weights.

**Portfolio note:** Flagship supervised deep-learning project — full pipeline, honest metrics, runnable CLIs.

---

## Architecture

```
Audio (5s, 32 kHz, middle crop)
    → log-mel spectrogram  (1 x 128 x 256), min-max -> [0, 1]
    → EfficientNet-B0 backbone  (timm, in_chans=1)
    → SED head  (1x1 conv → 206 framewise classes)
    → Temporal attention pooling
    → clip-level multi-label logits → sigmoid → probabilities
```

| Component | Choice |
|-----------|--------|
| Loss | `BCEWithLogitsLoss` (v2: + class `pos_weight`) |
| Optimizer | AdamW |
| Schedule | CosineAnnealingLR |
| Precision | Mixed precision (AMP) |
| Speed | Optional precomputed mel `.npy` cache |

---

## Results (v1)

| Split | Clips | Metric |
|-------|------:|--------|
| Train | 22,851 | — |
| Validation | 5,713 | **Macro ROC-AUC = 0.8529** |
| Total labeled clips | 28,564 | 206 species |

- Summary: [`results/training_summary.json`](./results/training_summary.json)
- Reproducibility: [`results/REPRODUCIBILITY.md`](./results/REPRODUCIBILITY.md)
- Frozen pack: [`results/artifacts/v1_baseline_auc08529/`](./results/artifacts/v1_baseline_auc08529/)

**Metric note:** macro ROC-AUC averages per-species ranking quality. It is **not** accuracy at a fixed threshold.

### Limitations (honest)

| Limitation | Natural next step |
|------------|-------------------|
| Single stratified split | Grouped / k-fold CV |
| Light augs (v1) | SpecAugment (v2) |
| No class re-weighting (v1) | `pos_weight` / balanced sampling (v2) |

---

## Quick start

```bash
pip install -r requirements.txt
```

### Inference (shipped weights)

```bash
python -m src.inference \
  --audio path/to/clip.ogg \
  --checkpoint models/v1/model.pth \
  --config v1/config.json \
  --top-k 5
```

### Evaluate

```bash
python -m src.evaluate \
  --checkpoint models/v1/model.pth \
  --config v1/config.json \
  --metadata path/to/val.csv \
  --audio-dir path/to/train_audio
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

### Train v3 (4th-place-style recipe — Soft AUC + 10 s + semi-supervised)

```bash
# 1. precompute v3 power-mels (10 s, 256×512)
python -m v3.precompute_mels \
  --audio-dir path/to/train_audio --output-dir path/to/mels_v3 \
  --config v3/config.json

# 2. train (single fold)
python -m v3.train \
  --config v3/config.json \
  --metadata path/to/train.csv --mel-dir path/to/mels_v3 \
  --output-dir runs/v3_exp001 --fold 0 --folds 5

# 3. (optional Stage 2) pseudo-label soundscapes, then re-train with --semi-csv
python -m v3.pseudo_label \
  --config v3/config.json \
  --checkpoints runs/v3_exp001/model_fold0_best.pth \
  --soundscapes-dir path/to/train_soundscapes \
  --mel-dir path/to/mels_v3_semi --output-csv runs/pseudo/semi_chunks.csv

# 4. overlap-TTA inference with a v3 checkpoint
python -m src.inference \
  --audio path/to/soundscape.ogg --checkpoint runs/v3_exp001/model_fold0_best.pth \
  --config v3/config.json --model-version v3 --overlap --smooth --postprocess
```

See [`docs/V3_TRAINING.md`](./docs/V3_TRAINING.md) for the full v3 guide.

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
4. **Per-version folders (`v1/`, `v2/`, `v3/`)** — shared library lives in `src/`; each version keeps its train entry point, config, and tools together.
5. **`results/artifacts/` freezes** — immutable snapshots for honest v1 vs v2 comparison.
6. **v3 loss = Soft AUC** (4th-place trick) — optimizes the competition metric directly, overfitting-resistant, supports soft labels for semi-supervised learning.
