# Models

| Folder | Role |
|--------|------|
| `v1/model.pth` | Published best checkpoint (v1, val macro ROC-AUC **0.8529**) |
| `v2/` | v2 checkpoints (empty — v2 0.8401 did not beat v1; no local checkpoint) |
| `v3/` | v3 checkpoints (empty — ready for the next training run) |

Run results and frozen cards live under `../results/v1|v2|v3/`.
New training runs write to `../runs/` (gitignored). Promote a winner into
`v1/model.pth` (or a new version folder) only after it beats the published metric.
