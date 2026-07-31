# results/v2 — v2 run results

**Result:** v2 (EfficientNet-B0 + SpecAugment + pos_weight cap 50 + balanced sampler,
15 epochs, LR 8e-4) finished at **val macro ROC-AUC = 0.8401** — did NOT beat the
v1 baseline (0.8529, delta **-0.0128**).

| File | Role |
|------|------|
| `metrics.json` | Full run: best metrics + per-epoch history (15 epochs) |
| `per_class_metrics.csv` | Per-species AUC / PR-AUC / F1 (206 species) |
| `config_used.json` | Exact config snapshot of the run |

## Lessons (why v2 lost to v1)

- Best AUC at epoch 12 (0.8401); epochs 13-15 plateaued at 0.838-0.840.
- Heavy pos_weight (cap 50) + balanced sampler crushed precision:
  macro_f1 = 0.096, micro_precision = 0.052, micro_recall = 0.466.
- These observations drove the v3 recipe (mild reweighting, no balanced sampler,
  Soft AUC loss) — see `docs/V3_TRAINING.md`.
