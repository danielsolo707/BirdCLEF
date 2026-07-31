# 1st Place — Multi-Iterative Noisy Student Is All You Need (Nikita Babych)

Source: https://www.kaggle.com/competitions/birdclef-2025/writeups/nikita-babych-1st-place-solution-multi-iterative-n
Final scores: Public 0.933 → Private 0.930

## Core ideas
- SED models on **20-second input chunks** (5s=0.842, 10s=0.864, 15s=0.870, **20s=0.872**, 30s=0.872 Public)
- **Multi-Iterative Noisy Student** self-training: MixUp between labeled train data and pseudo-labeled soundscapes
- **Power transform** on pseudo-labels (temperature-like, applied to probs) to reduce label noise
- Pseudo-label **WeightedRandomSampler**: weight = sum of max label probs within each soundscape
- Separate dedicated model for **Amphibia + Insecta** groups using extra Xeno-Canto species
- Final ensemble of 7 models from different training iterations/architectures

## Data
- 5 folds, each fold ≥1 sample per label
- 20s chunks, **absmax normalization**, all secondary labels = 1
- Extra Xeno-Canto: 5489 target-species samples (usually hurt; only 1 model used it) + 17197 extra-species samples (Insecta 16218/544 sp, Amphibia 979/113 sp) for the dedicated group model (700 sp total, 17844 samples, min 1 sample/species)

## Mel parameters
- 20s → image (3, 224, **512**)
- MelSpectrogram(sample_rate=32000, **mel_bins=224**, fmin=0, fmax=16000, n_fft=4096, **hop_size=1252**, top_db=80)
- 0-1 normalization
- Notes: larger hop for speed; **larger n_mels helped** (narrow-band calls of Amphibia/Insecta)

## Model / SED head
- SED head (adaptation of 4th place 2021) + **GeM frequency pooling** + **repeated 3 mel spectrograms as input** (3-channel input)
- Stage 1 backbones: tf_efficientnet_b0.ns_jft_in1k, regnety_008.pycls_in1k
- Iter 1: + tf_efficientnet_b3.ns_jft_in1k, regnety_016.tv2_in1k
- Iter 2-4: tf_efficientnet_b3/b4.ns, regnety_016.tv2_in1k, eca_nfnet_l0.ra2_in1k
- Amphibia/Insecta model: tf_efficientnet_b0.ns_jft_in1k

## Training
- **Loss: CrossEntropy** (CE slightly better than BCE/Focal when LR+epochs tuned; CE handles imbalance better in his tests)
- Labels NOT normalized to sum 1 (harder multi-label samples impact loss more)
- LR 5e-4 → 1e-6, AdamW wd 1e-4, **CosineAnnealingWarmRestarts** (restart every 5 epochs), BS 64, 15 epochs (supervised)
- Mixup p=0.5 on absmax-normalized raw audio, equal sampling weight per species, left-pad 0 so overlap always exists

## Pseudo-labeling (self-training)
- Predict soundscapes with best ensemble; store per-5s max probs or framewise (4 frames/5s)
- **MixUp labeled × pseudo-labeled at constant blend 0.5** (Beta params = inf) — concatenation didn't work; low Beta failed; constant 0.5 worked
- **Stochastic depth drop_path_rate=0.15** — only helps in self-training stages (Noisy Student effect), up to +0.005
- Random 20s interval per soundscape, max-prob across segments/frames, soft labels
- Ratio of pseudo-mixed samples (BS 64): 0→0.872, 0.25→0.883, 0.5→0.887, 0.75→0.890, **1.0→0.898**
- Iterations (power): iter1 pow=1 → 0.909, iter2 pow=1/0.65 → 0.918, iter3 pow=1/0.55 → 0.927, **iter4 pow=1/0.6 → 0.930**; iter5 stopped working
- Training: 25-35 epochs, drop_path 0.15, random padding of short samples within 20s

## Inference
- **Average overlapping framewise predictions from neighboring chunks** (1D sliding window) — +0.002-0.003 (TTA)
- Pad left/right so first/last 5s chunks centered; remove padding predictions
- Smoothing kernel [0.1, 0.2, 0.4, 0.2, 0.1]
- **Delta shift TTA** (from 2nd solution 2023)
- OpenVINO inference, multiprocess soundscape loading, spectrograms computed once

## Final ensemble (7 models)
1. efficientnetb4 (iter 3), 2. efficientnetb3 (iter 3), 3. regnety016 ×2 (iter 4),
4. ecanfnetl0 (iter 3, +Xeno-Canto target species), 5. regnety008 (supervised), 6. efficientnetb0 (Amphibia/Insecta)
- Equal weights best (0.935 private); B3 + eca_nfnet slightly higher weights in some runs

## References
- [1] Self-training with Noisy Student (ImageNet), [2] Design Choices for Enhancing Noisy Student Self-Training, [3] Deep Networks with Stochastic Depth
