# results/v3 — v3 run results

**Run:** `exp001` — 4th-place-style recipe (Soft AUC loss, 10 s power-mel 256×512, AttBlock SED, mixup, rare-class upsampling). Single fold (0), 80/20 stratified split, 20 epochs on RTX 4070 (~4 h incl. precompute).

| Metric | Value |
|--------|------:|
| **Best val macro ROC-AUC** | **0.9694** |
| v1 baseline | 0.8529 |
| Δ vs v1 | **+0.1165** |
| Best epoch | 15 (val_auc 0.9694) |
| Final epoch (20) | val_auc 0.9670 · macro_f1 0.5515 |

- `metrics.json` — full run card (config snapshot, per-epoch summaries, deltas)
- `per_class_metrics.csv` — per-species ROC-AUC / F1 / precision / recall (191/206 classes computable; 15 rare species have 0–1 val samples)
- `config_used.json` — exact hyperparameters used

**Honest note:** validation uses a random stratified split (not grouped by recording site), the same methodology as v1/v2 — so comparisons across versions are apples-to-apples, but absolute numbers are optimistic vs. the Kaggle leaderboard.

**Next steps (not yet run):** pseudo-label `train_soundscapes` → re-train with `--semi-csv`, overlap-TTA inference, 5-fold ensemble.
