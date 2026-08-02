# Models

| Folder | Role |
|--------|------|
| `v1/model.pth` | Published v1 checkpoint (val macro ROC-AUC **0.8529**) |
| `v2/model_best.pth` | v2 best (0.8401) — did not beat v1 |
| `v2/model_last.pth` | v2 end-of-training checkpoint |
| `v3/model_best.pth` | **Champion** — system upgrade (ep15, val macro ROC-AUC **0.9694**, +0.1165 vs v1) |
| `v3/model_last.pth` | v3 end-of-training (epoch 20, val AUC 0.9670) |

Run results and frozen cards: `../results/v1|v2|v3/`.  
v3 caveats / 5-fold + Stage-2 plan: `../results/v3/NEXT_STEPS.md`.

New training runs write to `../runs/` (gitignored). Promote a winner only after it
beats the published metric **and** you update the matching `results/vN/` card.
