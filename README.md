# 🐦 BirdCLEF+ 2025 — Sound Event Detection

Multi-label bird species classification from 5-second audio clips.

## Architecture
- **Backbone:** EfficientNet-B0 (1-channel mel-spectrogram input)
- **Head:** CNN-based Sound Event Detection (SED) with attention pooling
- **Input:** 128×256 mel-spectrogram
- **Output:** 206 species probabilities (multi-label)

## Dataset
- 28,564 training clips
- 22,851 train / 5,713 validation split
- 206 bird species

## Training
- Precomputed mel-spectrograms for 10× training speed
- 10 epochs, AdamW, CosineAnnealingLR
- Mixed precision (AMP) on Tesla T4
- BCEWithLogitsLoss for multi-label classification

## Results
- Best validation AUC: 0.8529

## Live Demo
🤗 [Hugging Face Space](https://huggingface.co/spaces/YOUR_USERNAME/birdclef-2025-demo)

## Author
Daniel soleimani
