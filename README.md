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
| **Checkpoint** | [`models/model.pth`](./models/model.pth) |
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
├── configs/                  # hyperparameters
│   ├── config.json           # v1 (matches shipped model)
│   ├── config_v2.json        # v2 (underperformed v1)
│   └── config_v3.json        # v3 (4th-place-style: SoftAUC + 10s + semi) — NEW
│
├── labels/                   # class maps (206 species)
│   ├── classes.json
│   └── label2id.json
│
├── models/                   # published checkpoints
│   └── model.pth             # v1 best (val AUC 0.8529)
│
├── artifacts/                # frozen experiment snapshots (do not overwrite)
│   └── v1_baseline_auc08529/
│
├── results/                  # human-readable run cards + summaries
│   ├── training_summary.json
│   └── REPRODUCIBILITY.md
│
├── src/                      # Python package (train / eval / infer)
│   ├── model.py              # v1 BirdCLEFSED + NEW v3 BirdCLEFModelV3 (bn0+AttBlock)
│   ├── losses.py             # NEW: AUCLoss / SoftAUCLoss / FocalBCELoss
│   ├── train.py              # v1 training
│   ├── train_v2.py           # v2 training
│   ├── train_v3.py           # v3 training (4th-place-style) — REWRITTEN
│   ├── pseudo_label.py       # NEW: pseudo-label train_soundscapes (Stage 2)
│   ├── evaluate.py
│   ├── inference.py          # + overlap-TTA / smooth / post-process flags
│   ├── ensemble.py           # NEW: blend prediction CSVs
│   ├── dataset.py            # + NEW BirdCLEFDatasetV3 (10s, mixup, rare upsample)
│   ├── audio.py              # + NEW v3 power-mel extraction
│   ├── metrics.py
│   └── utils.py
│
├── scripts/                  # one-off utilities
│   ├── precompute_mels.py
│   ├── precompute_mels_v3.py # NEW: v3 power-mel cache
│   ├── smoke_test_v2.py
│   └── smoke_test_v3.py      # NEW: v3 end-to-end sanity
│
├── notebooks/                # Kaggle / exploration notebooks
│   ├── birdclef_v2_kaggle.ipynb
│   └── (v3 notebook planned)
│
├── docs/                     # guides + top-5 solution writeups (local reference)
│   ├── V2_TRAINING.md
│   ├── V3_PLAN.md            # NEW: solution comparison + why 4th place
│   ├── V3_TRAINING.md        # NEW: v3 recipe
│   └── writeups/             # NEW: 1st-5th place solutions + 4th-place code
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
- Frozen pack: [`artifacts/v1_baseline_auc08529/`](./artifacts/v1_baseline_auc08529/)

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
  --checkpoint models/model.pth \
  --config configs/config.json \
  --top-k 5
```

### Evaluate

```bash
python -m src.evaluate \
  --checkpoint models/model.pth \
  --config configs/config.json \
  --metadata path/to/val.csv \
  --audio-dir path/to/train_audio
```

### Train v1

```bash
python -m src.train \
  --config configs/config.json \
  --metadata path/to/train.csv \
  --audio-dir path/to/train_audio \
  --mel-dir path/to/mels \
  --output-dir runs/v1_exp
```

### Train v2 (stronger recipe)

```bash
python -m src.train_v2 \
  --config configs/config_v2.json \
  --metadata path/to/train.csv \
  --mel-dir path/to/mels \
  --output-dir runs/v2_exp
```

See [`docs/V2_TRAINING.md`](./docs/V2_TRAINING.md) for the full v2 guide (also running on Kaggle).

### Train v3 (4th-place-style recipe — Soft AUC + 10 s + semi-supervised)

```bash
# 1. precompute v3 power-mels (10 s, 256×512)
python scripts/precompute_mels_v3.py \
  --audio-dir path/to/train_audio --output-dir path/to/mels_v3 \
  --config configs/config_v3.json

# 2. train (single fold)
python -m src.train_v3 \
  --config configs/config_v3.json \
  --metadata path/to/train.csv --mel-dir path/to/mels_v3 \
  --output-dir runs/v3_exp001 --fold 0 --folds 5

# 3. (optional Stage 2) pseudo-label soundscapes, then re-train with --semi-csv
python -m src.pseudo_label \
  --config configs/config_v3.json \
  --checkpoints runs/v3_exp001/model_fold0_best.pth \
  --soundscapes-dir path/to/train_soundscapes \
  --mel-dir path/to/mels_v3_semi --output-csv runs/pseudo/semi_chunks.csv

# 4. overlap-TTA inference with a v3 checkpoint
python -m src.inference \
  --audio path/to/soundscape.ogg --checkpoint runs/v3_exp001/model_fold0_best.pth \
  --config configs/config_v3.json --model-version v3 --overlap --smooth --postprocess
```

See [`docs/V3_TRAINING.md`](./docs/V3_TRAINING.md) for the full v3 guide.

### Precompute mels

```bash
python scripts/precompute_mels.py \
  --audio-dir path/to/train_audio \
  --output-dir path/to/mels \
  --config configs/config.json
```

---

## Design choices (short)

1. **SED + attention** — bird calls are sparse in time; attention focuses useful frames.  (v3: PANNs-style `AttBlock`)
2. **Precomputed mels** — large speedup for multi-epoch training.  (v3: power-mel 256×512, db at load)
3. **Min-max log-mel in [0, 1] + middle crop** — train/serve parity with the notebook that produced `models/model.pth`.  (v3: 10 s random window + `power_to_db`)
4. **Configs + label maps in folders** — no magic paths buried only in notebooks.
5. **`artifacts/` freezes** — immutable snapshots for honest v1 vs v2 comparison.
6. **v3 loss = Soft AUC** (4th-place trick) — optimizes the competition metric directly, overfitting-resistant, supports soft labels for semi-supervised learning.
