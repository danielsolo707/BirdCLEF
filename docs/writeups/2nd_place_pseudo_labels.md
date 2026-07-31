# 2nd Place — Journey Down the Rabbit Hole of Pseudo Labels (Volodymyr Sydorskyi & vialactea)

Source: https://www.kaggle.com/competitions/birdclef-2025/writeups/volodymyr-vialactea-2nd-place-journey-down-the-rab
Final: Private ~0.918 (top solo model 0.917 Public / 0.91 Private)

## Core ideas
- **Pretraining on massive Xeno-Canto** (~7400-7800 species, excluding this year's species) → THE killer feature: 0.83-0.84 → 0.86-0.87
- Pseudo-labeling of train_soundscapes in iterations (soft labels, confidence filtering)
- 5s random segments for training
- Final submission: 3 models (tf_efficientnetv2_s ×2, eca_nfnet_l0 ×1) + postprocessing

## Data
- 7376 Xeno-Canto + 90 previous-competition samples (not in this year's set, proper primary labels)
- iNat/XC parsed data only helped one teammate slightly
- 5s segments: tried random-5s-from-whole / first-7s / first-or-last-7s → last approach better initially, but **skipping non-vocalization periods reduced LB** → ended using whole audio, only removing alien speech (manually or automatically)
- Validation: stratify by primary_label + **group by author**; undersampled species handled 3 ways (add 1 sample to val fold even with tiny leakage / remove from train / move undersampled to train only). Strategies 1 & 3 mostly used. CV-LB correlation weak within ~1% AUC

## Backbones
- tf_efficientnetv2_s, eca_nfnet_l0 (ConvNeXt and ResNeXt failed)
- Heads: SED head (Volodymyr) + MLP head (vialactea)

## Training (common)
- 50 epochs, BS 64, **half of a cosine cycle**, **Focal + BCE loss**, label smoothing 0.005
- eca_nfnet_l0: RAdam lr 1e-4; tf_efficientnetv2_s: AdamW lr 1e-4, eps 1e-8, betas (0.9, 0.999)
- Balancing strategies: Balanced, **Squared** (value_counts ** -0.5), **Upsampling** (repeat rare classes to N, e.g. 100)
- Final ensemble used Balanced and Balanced+Upsampling
- Pretrained checkpoint selection: last / best / avg of 3 best (last two worked best); use backbone only, discard head
- No differential LR between backbone and head

## Augmentations
- Same as 2023 write-up: RandomFiltering + SpecAug (turning off gave better CV but worse LB → keep them)

## Pseudo-labeling
1. Predict all train_soundscapes with best ensemble in submission format (5s windows)
2. Keep only segments with **max prob > 0.5**
3. **Soft labels** (no thresholding) — ensemble distillation + adaptation to soundscape noise
4. **Trim probs < 0.1 to zero** (key step to remove noisy labels)
- Sampling: sample by original strategy; if that class exists in soundscape_df, with prob **0.4** replace with pseudo sample (soft label vector)
- Iter 1: 0.86-0.87 → 0.89-0.895; iter 2 (using previously-failed soundscapes): 0.90-0.91; iter 3: no gains
- **OOF soundscape folds** to avoid leakage on re-prediction → pseudo mix of different generations
- Selected files per iteration: 4430 / 1483 / 1437

## Post-processing
```python
def postprocessing(input_df, top=1):
    only_probs = input_df.iloc[:, 1:].values
    N, F = only_probs.shape
    only_probs = only_probs.reshape((N//12, 12, F))
    mean_ = np.mean(np.sort(only_probs, axis=1)[:, -top:], axis=1, keepdims=True)
    only_probs *= mean_
    input_df.iloc[:, 1:] = only_probs.reshape((N, F))
    return input_df
```
- Multiply all chunk probs by top prob per class per file → +0.005-0.01 (weaker models +0.01)
- **TTA** (overlapping windows, from 2024 top-4): 0.917→0.922 Public / 0.91→0.918 Private, but **didn't combine well with postprocessing**

## Ablation (Write-up Speedrun)
| Improvement | Public |
|---|---|
| Baseline | 0.83-0.84 |
| + Pretrain | 0.86-0.87 |
| + Pseudo iter 1 | 0.89-0.895 |
| + Pseudo iter 2-3 | 0.90-0.91 |
| + TTA | 0.922 |
| Postprocessing | +0.005-0.01 |

## Failed
- Soft labels on main train data; latest XC-snippet pretrain; extra iNat/XC data; Time Flip augmentation

## Resources
- Inference kernels, GitHub: https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place, paper: https://ceur-ws.org/Vol-4038/paper_256.pdf
