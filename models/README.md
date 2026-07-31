# Models

| File | Role |
|------|------|
| `model.pth` | Published best checkpoint (v1, val macro ROC-AUC **0.8529**) |

Frozen copies of experiment runs live under `../artifacts/` (never overwrite those).
New training runs write to `../runs/` (gitignored). Promote a winner here only after it beats the published metric.
