# 5th Place — Self-Distillation is All You Need (Zuoli / MYSO / youknow)

Source: https://www.kaggle.com/competitions/birdclef-2025/writeups/noir-5th-place-solution-self-distillation-is-all-y
Final: Public 0.928 / Private 0.924 (Group A + 2.5s overlap + smoothing)

## Core ideas
- **Self-distillation** to enrich train_audio with secondary labels (many bird calls unlabeled in recordings!)
- **Silero VAD + manual cleaning** (~2000 files, remove human voices) + manual segment selection for rare classes (n<~30)
- 3-stage training: (1) 5-fold supervised, (2) iterative self-distillation on train_audio (4-5 rounds), (3) + train_soundscapes (1:1 batch mix)
- SED model with efficientnet family; **FocalLoss (gamma=2)**

## Data prep
- Only 2025 dataset
- Silero VAD detects human voice → Streamlit tool → manually remove segments
- Rare classes: manually select segments containing bird calls
- Cleaned files: first 60s; others: first 30s
- Duplicate files in classes with <20 samples

## Features
- Random 10-second segments
- Mel: sample_rate 32000, **mel_bins 192**, fmin 20, fmax 15000, window_size 2048, **hop_size 768**
- **log(melspec + 1e-6)** (log scale)
- Augmentations: Resampling, Gain, **FilterAugment**, FrequencyMasking, TimeMasking, **Sumix on mel domain**
- Optimizer Adam + Cosine Annealing with warmup; 10 epochs; primary + secondary labels

## Backbones
- 4× tf_efficientnetv2_s, 3× tf_efficientnetv2_b3, 4× tf_efficient_b3_ns, 2× tf_efficient_b0_ns

## Self-distillation recipe (from author comments)
- Stage 2: 5-fold CV; Stage 3: no folds, different seeds
- Teacher = average of previous stage's 5 models
- Mix pseudo with original:
```python
alpha = 0.7
pseudo_labels = pseudo_labels * (pseudo_labels > 0.3) + pseudo_labels**2
pseudo_labels = torch.clamp(pseudo_labels, min=0.0, max=1.0)
labels = alpha * pseudo_labels + (1 - alpha) * labels
```
- Same model arch for teacher & student; weights re-initialized each round; teacher sees augmented data

## Scores (ensemble of 5, Public)
| Model | Stage1 | Distill×2 | Distill×4 | Distill×5 | Stage3 ×1 | ×2 |
|---|---|---|---|---|---|---|
| tf_efficientnetv2_s | 0.839 | 0.863 | 0.880 | 0.884 | 0.915 | **0.921** |
| tf_efficientnetv2_b3 | 0.842 | N/A | 0.872 | - | N/A | 0.918 |
| tf_efficient_b3_ns | N/A | N/A | N/A | - | N/A | 0.921 |
| tf_efficient_b0_ns | 0.836 | 0.871 | 0.879 | 0.883 | 0.905 | 0.912 |

## Inference
- **2.5-second overlap** sliding window; scores weighted & combined (alpha=0.5), similar to 2024 4th place
- Smoothing over neighboring frames: [0.1, 0.8, 0.1]
- Tried power-adjustment postprocessing (public notebook) but dropped it (overfitting risk)
- OpenVINO + ThreadPoolExecutor

## Failed
- CNN-based (non-SED) models; 1D models; too many data augmentations
