# BirdCLEF+ 2025 — Sound Event Detection

Multi-label bird species classification from 5-second audio clips using an **EfficientNet-B0** backbone and a **Sound Event Detection (SED)** head with **attention pooling**.

| | |
|---|---|
| **Task** | Multi-label audio classification (206 species) |
| **Best validation AUC** | **0.8529** (macro ROC-AUC) |
| **Stack** | PyTorch · timm · librosa · torchvision |
| **Hardware** | Tesla T4 (Kaggle) · ~completed training run |
| **Checkpoint** | [`model.pth`](./model.pth) |
| **Source notebook** | [danielsolo1770/notebookeb002d87be](https://www.kaggle.com/code/danielsolo1770/notebookeb002d87be) |
| **License** | [MIT](./LICENSE) |

---

## Why this project

BirdCLEF is a realistic bioacoustics challenge: short clips, **multi-label** targets, class imbalance, and spectrogram-based CNN modeling. This repo packages the **training / evaluation / inference code aligned with the Kaggle notebook** that produced val AUC **0.8529**, plus the released weights.

**Portfolio note:** This is the flagship supervised deep-learning project in my public profile — full pipeline, honest metrics, runnable CLIs.

---

## Architecture

```
Audio (5s, 32 kHz, middle crop)
    |
    v
Log-mel spectrogram  (1 x 128 x 256), min-max -> [0, 1]
    |
    v
EfficientNet-B0 backbone  (timm efficientnet_b0, in_chans=1, no classifier)
    |  features: (B, feature_dim, H, W)  # typically 1280
    v
SED head  (1x1 Conv -> 256 -> BN -> ReLU -> Dropout(0.3) -> 1x1 Conv -> 206)
    |  framewise logits: (B, 206, T)
    v
Temporal attention pooling  (Linear 206->128 -> Tanh -> Linear -> 1 + softmax over T)
    |
    v
Clip-level multi-label logits  (B, 206)  ->  sigmoid -> species probabilities
```

| Component | Choice |
|-----------|--------|
| Loss | `BCEWithLogitsLoss` (independent multi-label heads) |
| Optimizer | AdamW (lr=1e-3, weight decay=**1e-5**) |
| Schedule | CosineAnnealingLR, 10 epochs |
| Precision | Mixed precision (AMP) |
| Speed | Optional precomputed mel `.npy` cache |
| Train augs | RandomHorizontalFlip + ColorJitter on mel |

---

## Results

| Split | Clips | Metric |
|-------|------:|--------|
| Train | 22,851 | — |
| Validation | 5,713 | **Macro ROC-AUC = 0.8529** |
| Total labeled clips | 28,564 | 206 species |

Machine-readable summary: [`results/training_summary.json`](./results/training_summary.json)  
Reproducibility card: [`results/REPRODUCIBILITY.md`](./results/REPRODUCIBILITY.md)

**How to read the metric:** macro ROC-AUC averages per-species ranking quality so rare and common birds count equally. It is **not** accuracy at a fixed threshold.

### Limitations (honest)

| Limitation | Why it matters | Natural next step |
|------------|----------------|-------------------|
| Single stratified split | Metric variance unknown | Grouped / k-fold CV by site |
| Light spectrogram augs | Less noise robustness | SpecAugment, noise, mixup |
| No class re-weighting | Rare species may lag | `pos_weight` / focal / balanced sampling |
| Clip-level supervision | No strong time stamps | Frame-level SED labels if available |
| Not a public LB claim | Portfolio experiment | Full competition submission stack |

---

## Repository layout

```
BirdCLEF/
├── model.pth                 # trained weights (val AUC 0.8529)
├── config.json               # audio + training hyperparameters
├── classes.json              # 206 species labels
├── label2id.json             # label -> class index
├── LICENSE                   # MIT
├── results/
│   ├── training_summary.json
│   └── REPRODUCIBILITY.md
├── requirements.txt
├── scripts/
│   └── precompute_mels.py
└── src/
    ├── model.py              # BirdCLEFSED (matches model.pth)
    ├── audio.py              # mel extraction
    ├── dataset.py            # multi-label Dataset + augs
    ├── train.py
    ├── evaluate.py
    ├── inference.py
    └── utils.py
```

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

### Inference (shipped `model.pth`)

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

## Design decisions

1. **SED + attention instead of global pool only** — bird calls are sparse in time; attention can focus on informative frames.
2. **1-channel EfficientNet-B0** — strong ImageNet inductive bias on spectrograms with T4-friendly cost.
3. **Min-max log-mel in [0, 1] + middle crop** — matches the training notebook that produced `model.pth` (train/serve parity).
4. **Precomputed mels** — ~10× faster multi-epoch training after a one-time feature pass.
5. **Config + CLI paths** — hyperparameters in `config.json`; data roots never hard-coded to Kaggle only.

---

## What this repo demonstrates

- Multi-label deep learning on competition-scale audio
- Spectrogram pipelines + CNN transfer learning (`timm`)
- Train / eval / inference entrypoints suitable for review
- Reproducible config + shipped weights (no retrain required to inspect the system)

---

## Author

**Daniel Soleimany** · [github.com/danielsolo707](https://github.com/danielsolo707) · [danielsoleimani.ir](https://danielsoleimani.ir/)

---

## License

MIT License — see [LICENSE](./LICENSE).

Code is provided for portfolio / educational use. **BirdCLEF competition data** remains under the competition’s own terms and is **not** redistributed in this repository.
