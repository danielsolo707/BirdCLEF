# results/v3 — v3 run results (champion freeze)

**Run:** `exp001` — **system upgrade** (not a pure v1 ablation): Soft AUC loss, 10 s power-mel 256×512, AttBlock SED, mixup, rare-class upsampling.  
**Protocol:** single fold (fold 0 only), stratified 80/20 on `primary_label`, 20 epochs, RTX 4070 (~4 h incl. precompute).

| Metric | Value |
|--------|------:|
| **Best val macro ROC-AUC** | **0.9694** |
| Best epoch | 15 |
| Macro PR-AUC (best ep) | ~0.653 |
| Macro F1 (best ep) | ~0.527 |
| Micro F1 (best ep) | ~0.634 |
| Final epoch (20) | AUC 0.9670 · macro_f1 ~0.55 |
| v1 baseline AUC | 0.8529 |
| Δ vs v1 (AUC) | **+0.1165** |

## Files

| File | Role |
|------|------|
| `metrics.json` | Full run card (config, history, deltas) |
| `per_class_metrics.csv` | Per-species ROC-AUC / F1 / P / R (191/206 computable) |
| `config_used.json` | Exact hyperparameters |
| `NEXT_STEPS.md` | Caveats → 5-fold + Stage-2 plan |

## How to talk about this result

1. **System upgrade** — features (10 s, power-mel), head (AttBlock), loss (SoftAUC), augs (mixup, upsample) all changed vs v1.  
2. **SoftAUC optimizes ranking** — headline AUC is partly metric-aligned by design; **F1 / PR-AUC** show the model also improves operating-point quality vs v2.  
3. **Same split style as v1/v2** — fair version comparison; **optimistic vs Kaggle LB** (not site-grouped).  
4. **Single fold** — do not call this a multi-fold mean until `train_5fold.sh` is finished.

## Not yet in this freeze

- 5-fold mean AUC  
- Pseudo-labeled `train_soundscapes` (Stage 2)  
- Fold ensemble / overlap-TTA scorecard on a fixed soundscape set  

See [`NEXT_STEPS.md`](./NEXT_STEPS.md).
