# BirdCLEF+ 2025 — Sound Event Detection

Multi-label bird species classification from 5-second audio clips using an **EfficientNet-B0** backbone and a **Sound Event Detection (SED)** head with **attention pooling**.

| | |
|---|---|
| **Task** | Multi-label audio classification (206 species) |
| **Best validation AUC** | **0.8529** |
| **Stack** | PyTorch · timm · librosa · torchvision |
| **Hardware** | Tesla T4 (Kaggle) |
| **Checkpoint** | [`model.pth`](./model.pth) |
| **Source notebook** | [danielsolo1770/notebookeb002d87be](https://www.kaggle.com/code/danielsolo1770/notebookeb002d87be) |

---

## Why this project

BirdCLEF is a realistic bioacoustics challenge: short clips, **multi-label** targets, class imbalance, and spectrogram-based CNN modeling. This repo packages the **training / evaluation / inference code aligned with the Kaggle notebook** that produced val AUC **0.8529**, plus the released weights.

---

## Architecture

```
Audio (5s, 32 kHz, middle crop)
    │
    ▼
Log-mel spectrogram  (1 × 128 × 256), min-max → [0, 1]
    │
    ▼
EfficientNet-B0 backbone  (timm efficientnet_b0, in_chans=1, no classifier)
    │  features: (B, feature_dim, H, W)  # typically 1280
    ▼
SED head  (1×1 Conv → 256 → BN → ReLU → Dropout(0.3) → 1×1 Conv → 206)
    │  framewise logits: (B, 206, T)
    ▼
Temporal attention pooling  (Linear 206→128 → Tanh → Linear → 1 + softmax over T)
    │
    ▼
Clip-level multi-label logits  (B, 206)  →  sigmoid → species probabilities
```

**Loss:** `BCEWithLogitsLoss`  
**Optimizer:** AdamW (lr=1e-3, weight decay=**1e-5**)  
**Schedule:** CosineAnnealingLR over 10 epochs  
**Other:** mixed precision (AMP), precomputed mels, RandomHorizontalFlip + ColorJitter

---

## Repository layout

```
BirdCLEF/
├── model.pth                 # trained weights (val AUC 0.8529)
├── config.json               # audio + training hyperparameters
├── classes.json              # 206 species labels
├── label2id.json             # label → class index
├── results/
│   └── training_summary.json # frozen metrics from the completed run
├── requirements.txt
├── scripts/
│   └── precompute_mels.py    # optional: cache mels as .npy
└── src/
    ├── model.py              # BirdCLEFSED / BirdSEDModel (matches model.pth)
    ├── audio.py              # mel extraction (Kaggle get_melspec)
    ├── dataset.py            # multi-label Dataset + augs
    ├── train.py              # full training loop
    ├── evaluate.py           # ROC-AUC evaluation
    ├── inference.py          # single-file / folder prediction
    └── utils.py
```

---

## Results

| Split | Clips | Metric |
|-------|------:|--------|
| Train | 22,851 | — |
| Validation | 5,713 | **Macro ROC-AUC = 0.8529** |
| Total labeled clips | 28,564 | 206 species |

Full training configuration is recorded in [`results/training_summary.json`](./results/training_summary.json).

---

## Setup

```bash
git clone https://github.com/danielsolo707/BirdCLEF.git
cd BirdCLEF
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**Data:** download BirdCLEF+ 2025 from [Kaggle](https://www.kaggle.com/competitions/birdclef-2025) (requires competition access). Point the CLI at your local `train.csv` and audio folder — nothing is hardcoded to `/kaggle/input`.

---

## Usage

### Inference (uses shipped `model.pth`)

```bash
python -m src.inference --audio path/to/clip.ogg --top-k 5
python -m src.inference --audio path/to/audio_folder --threshold 0.3
```

### Evaluate a labeled split

```bash
python -m src.evaluate \
  --checkpoint model.pth \
  --metadata path/to/val.csv \
  --audio-dir path/to/audio
```

### Train (optional — weights already provided)

```bash
# Optional speedup: precompute mels once (~10× faster epochs)
python scripts/precompute_mels.py \
  --audio-dir path/to/train_audio \
  --output-dir path/to/mels \
  --metadata path/to/train.csv

python -m src.train \
  --config config.json \
  --metadata path/to/train.csv \
  --audio-dir path/to/train_audio \
  --mel-dir path/to/mels \
  --output-dir runs/exp001
```

Checkpoints (`model_best.pth`, `birdclef_best_model.pth`, `model_last.pth`) and `metrics.json` are written under `--output-dir`.

---

## Design notes

- **SED + attention** keeps frame-level structure instead of global pooling only — better for short, sparse bird calls.
- **1-channel EfficientNet** reuses a strong ImageNet inductive bias on spectrograms.
- **Precomputed mels** trade disk for a large wall-clock win during multi-epoch training (Kaggle cache naming: `folder_file.npy`).
- **Train augs** match the notebook: horizontal flip + mild brightness/contrast jitter on the mel tensor.

---

## What this repo demonstrates

- Multi-label deep learning on real competition-scale audio
- Spectrogram pipelines + CNN transfer learning (`timm`)
- Clean train / eval / inference entrypoints suitable for portfolio review
- Reproducible config + shipped weights (no need to retrain to inspect the system)

---

## Author

**Daniel Soleimani** · [github.com/danielsolo707](https://github.com/danielsolo707)

---

## License

Code in this repository is provided for portfolio / educational use. BirdCLEF data remains under the competition’s own terms.
