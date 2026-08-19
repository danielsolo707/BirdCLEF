# v3 next steps (from honest caveats)

Recorded reference: **single-fold** SoftAUC system, val macro ROC-AUC **0.9694**.
This file turns remaining caveats into an actionable plan.

## Caveat → plan

| # | Caveat | Action | Status |
|---|--------|--------|--------|
| 1 | Not a pure v1 ablation | Always describe v3 as a **system upgrade** (10 s + power-mel + AttBlock + SoftAUC + mixup) | Done in README / run cards |
| 2 | SoftAUC optimizes AUC | Always ship **F1 + PR-AUC** next to AUC; do not claim “ranking-free” gains | Done in metrics + README |
| 3 | Single fold only | Train **5-fold**, report **mean ± std** fold AUC | Code ready — run below |
| 4 | Stage 2 unused | Pseudo-label `train_soundscapes` → re-train with `--semi-csv` | Code ready — run below |
| 5 | Split not site-grouped | Keep stratified split for fair v1/v2/v3 comparison; optional future: group by location | Recorded limitation |

## A. 5-fold CV (robustness)

**Why:** one fold can be lucky; mean fold AUC is a more representative estimate than a single-fold result.

```bash
# from repo root (mels already in data/mels_v3)
bash v3/train_5fold.sh
# equivalent:
python -m v3.train \
  --config v3/config.json \
  --metadata data/train.csv \
  --mel-dir data/mels_v3 \
  --output-dir runs/v3_5fold \
  --folds 5 \
  --batch-size 24
```

**Expect:** ~5× single-fold wall time (order of ~15–20 h on RTX 4070 @ 20 ep / fold).  
**Outputs:** `runs/v3_5fold/model_fold{0..4}_best.pth` + `metrics.json` with `mean_fold_auc`.

**Update the record:** if the 5-fold run is completed, add mean±std to `results/v3/README.md` and optionally ensemble folds for inference.

## B. Stage 2 — pseudo-labels

**Why:** unlabeled `train_soundscapes` can provide additional training signal for a semi-supervised run.

```bash
bash v3/stage2_pseudo.sh
```

Pipeline:

1. Teacher = Stage-1 fold best checkpoint(s)  
2. `v3.pseudo_label` → `runs/pseudo/semi_chunks.csv`  
3. `v3.train --semi-csv ...` → soft-label mix on labeled + pseudo chunks  

**Update the reference:** only if val AUC / F1 improve on the Stage-1 reference under the **same** split protocol.

## C. Inference polish (cheap)

```bash
python -m src.inference \
  --audio path/to/soundscape.ogg \
  --checkpoint models/v3/model_best.pth \
  --config v3/config.json \
  --model-version v3 \
  --overlap --smooth --postprocess
```

Blend folds (after 5-fold):

```bash
python -m src.ensemble --help   # blend prediction CSVs from each fold
```

## D. What we will not claim

- That 0.9694 equals public LB score  
- That SoftAUC “doesn’t care” about the metric definition  
- That v3 is “the same network as v1 with one hyperparameter change”
