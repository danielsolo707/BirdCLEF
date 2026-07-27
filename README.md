# BirdCLEF+ 2025 â€” Sound Event Detection

Multi-label bird species classification from 5-second audio clips using an **EfficientNet-B0** backbone and a **Sound Event Detection (SED)** head with **attention pooling**.

| | |
|---|---|
| **Task** | Multi-label audio classification (206 species) |
| **Best validation AUC** | **0.8529** |
| **Stack** | PyTorch Â· timm Â· librosa Â· torchvision |
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
    â”‚
    â–¼
Log-mel spectrogram  (1 Ã— 128 Ã— 256), min-max â†’ [0, 1]
    â”‚
    â–¼
EfficientNet-B0 backbone  (timm efficientnet_b0, in_chans=1, no classifier)
    â”‚  features: (B, feature_dim, H, W)  # typically 1280
    â–¼
SED head  (1Ã—1 Conv â†’ 256 â†’ BN â†’ ReLU â†’ Dropout(0.3) â†’ 1Ã—1 Conv â†’ 206)
    â”‚  framewise logits: (B, 206, T)
    â–¼
Temporal attention pooling  (Linear 206â†’128 â†’ Tanh â†’ Linear â†’ 1 + softmax over T)
    â”‚
    â–¼
Clip-level multi-label logits  (B, 206)  â†’  sigmoid â†’ species probabilities
```

**Loss:** `BCEWithLogitsLoss`  
**Optimizer:** AdamW (lr=1e-3, weight decay=**1e-5**)  
**Schedule:** CosineAnnealingLR over 10 epochs  
**Other:** mixed precision (AMP), precomputed mels, RandomHorizontalFlip + ColorJitter

---

## Repository layout

```
BirdCLEF/
â”œâ”€â”€ model.pth                 # trained weights (val AUC 0.8529)
â”œâ”€â”€ config.json               # audio + training hyperparameters
â”œâ”€â”€ classes.json              # 206 species labels
â”œâ”€â”€ label2id.json             # label â†’ class index
â”œâ”€â”€ results/
â”‚   â””â”€â”€ training_summary.json # frozen metrics from the completed run
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ scripts/
â”‚   â””â”€â”€ precompute_mels.py    # optional: cache mels as .npy
â””â”€â”€ src/
    â”œâ”€â”€ model.py              # BirdCLEFSED / BirdSEDModel (matches model.pth)
    â”œâ”€â”€ audio.py              # mel extraction (Kaggle get_melspec)
    â”œâ”€â”€ dataset.py            # multi-label Dataset + augs
    â”œâ”€â”€ train.py              # full training loop
    â”œâ”€â”€ evaluate.py           # ROC-AUC evaluation
    â”œâ”€â”€ inference.py          # single-file / folder prediction
    â””â”€â”€ utils.py
```

---

## Results

| Split | Clips | Metric |
|-------|------:|--------|
| Train | 22,851 | â€” |
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

**Data:** download BirdCLEF+ 2025 from [Kaggle](https://www.kaggle.com/competitions/birdclef-2025) (requires competition access). Point the CLI at your local `train.csv` and audio folder â€” nothing is hardcoded to `/kaggle/input`.

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

### Train (optional â€” weights already provided)

```bash
# Optional speedup: precompute mels once (~10Ã— faster epochs)
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

- **SED + attention** keeps frame-level structure instead of global pooling only â€” better for short, sparse bird calls.
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

**Daniel Soleimani** Â· [github.com/danielsolo707](https://github.com/danielsolo707)

---

## License

MIT License — see [LICENSE](./LICENSE). Code is for portfolio / educational use. BirdCLEF competition data remains under the competition's own terms.
