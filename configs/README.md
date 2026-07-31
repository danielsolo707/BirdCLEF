# Configs

| File | Version | Notes |
|------|---------|--------|
| `config.json` | v1 | Matches published `models/model.pth` (AUC 0.8529) |
| `config_v2.json` | v2 | SpecAugment, pos_weight, balanced sampler, 15 epochs — **did not beat v1** (0.8401) |
| `config_v3.json` | v3 | **NEW** 4th-place-style recipe: SoftAUCLoss, 10 s power-mel 256×512, mixup, rare upsampling, semi-supervised support (see `docs/V3_TRAINING.md`) |

> Old v3 (aggressive dual-T4 EfficientNet-B3) was removed; this is the replacement v3.
