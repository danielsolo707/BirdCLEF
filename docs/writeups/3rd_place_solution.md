# 3rd Place Solution (Leon Shangguan / shanzhong8)

Source: https://www.kaggle.com/competitions/birdclef-2025/writeups/leon-simon-3rd-place-solution
Final: Private 0.927 (best: 6 CNN + 6 SED models; 20-model ensemble also 0.927)

## Core ideas
- Expand training set with **102 categories from BirdCLEF 2023** (classes with >50/40 instances, more balanced)
- 80% of 2023 + 100% of 2025 for training; remaining 20% of 2023 for **validation** (rough convergence signal)
- Extra Xeno-Canto + iNaturalist data; cleaned CSA (human voices removed via public notebook + manual filtering)
- train_soundscapes with pseudo labels from ensemble trained on above
- **Focal BCE loss**, **model soup** (weight averaging), **rank-aware post-processing** with power adjustment
- ONNX export for fast inference

## Models
- CNN + SED approaches with backbones: tf_efficientnet_b0_ns, tf_efficientnetv2_b3, tf_efficientnetv2_s.in21k_ft_in1k, mnasnet_100, spnasnet_100
- Two mel sets (only n_mels differs: 128 or 96):
```python
mel_spec_params = {
    "sample_rate": 32000,
    "n_mels": 128 or 96,
    "f_min": 0,
    "f_max": 16000,
    "n_fft": 2048,
    "hop_length": 512,
    "normalized": True,
    "center": True,
    "pad_mode": "constant",
    "norm": "slaney",
    "mel_scale": "htk",
}
```

## Training
- **Random sampling** outperformed fixed-first-5s and RMS-based approaches
- Augmentations from prior competitions: **cutmix, mixup, sumix** + added **human voice noise** as background noise
- Focal BCE loss; model soup across checkpoints

## Final submission
- Rank-aware post-processing (power adjustment) for low-confidence predictions
- 20 models = 10 CNN + 10 SED (5 backbones × 2 mel params) → private 0.927
- Best = 6 models each (without mnasnet/spnasnet backbones)
- ONNX: ~2-3× inference speedup

## Key comments from author
- Random sampling best because: with good pseudo labels, use ALL data; noisy samples → improved pseudo labels rather than dropped (RMS/first-5s approaches throw away data)
