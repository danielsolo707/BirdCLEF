# models/v3 — champion checkpoints

**Run exp001** — full **system upgrade** vs v1 (not a single-knob ablation).  
Beats v1 val macro ROC-AUC **0.8529** → **0.9694**.

| File | Role |
|------|------|
| `model_best.pth` | Best (epoch 15, val macro ROC-AUC **0.9694**) |
| `model_last.pth` | End of training (epoch 20, val AUC 0.9670) |

## Architecture (load)

- Backbone: EfficientNet-B0, 1 input channel  
- Head: v3 SED (`bn0` + freq-mean + channel smoothing + **AttBlock**)  
- Mel: power-mel **256×512** (10 s), `power_to_db` at load  
- Load with `BirdCLEFModelV3(backbone_name="efficientnet_b0", n_mels=256)` — see `src/model.py`

## Honest scope

| Fact | Implication |
|------|-------------|
| SoftAUC training | Strong AUC expected; also check F1/PR in `results/v3/` |
| Single fold | Not a 5-fold mean — see `results/v3/NEXT_STEPS.md` |
| Stratified clip split | Fair vs v1/v2; optimistic vs public LB |

Run card: [`../../results/v3/README.md`](../../results/v3/README.md).
