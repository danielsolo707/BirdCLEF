# models/v2 — v2 checkpoints

**Result:** val macro ROC-AUC **0.8401** (epoch 12) — did NOT beat v1 (0.8529).

| File | Epoch | Val AUC | Role |
|------|-------|---------|------|
| `model_best.pth` | 12 | **0.8401** | Best epoch (checkpointing metric: macro ROC-AUC) |
| `model_last.pth` | 15 | 0.8399 | Final epoch — useful for weight averaging / model soup |

Both load with `src.model.load_checkpoint(..., backbone_name="efficientnet_b0")`
(4.4M params, 371 keys, BirdCLEFSED architecture).

Run metrics: `../results/v2/` (`metrics.json`, `per_class_metrics.csv`, `config_used.json`).
