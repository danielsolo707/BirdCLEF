# Local data (not committed)

Competition data lives here locally (gitignored — only this README is tracked).

```
data/
├── train.csv                  # 28,564 labeled clips (13 cols, 206 species)
├── taxonomy.csv               # species taxonomy (206 rows)
├── sample_submission.csv      # Kaggle submission format
├── recording_location.txt     # site metadata for soundscapes
├── train_audio/               # labeled ogg clips, one folder per species (7.3 GB)
├── train_soundscapes/         # 9,726 unlabeled 10-min recordings (4.4 GB) — Stage 2 semi-supervised
└── mels_v3/                   # precomputed v3 power-mels 256×512, one .npy per clip (15 GB)
```

The v3 mel cache is produced by `python -m v3.precompute_mels --audio-dir data/train_audio --output-dir data/mels_v3 --config v3/config.json` (4-way parallel with `--metadata` slices for ~16 min on a 20-core machine).

To re-download: https://www.kaggle.com/competitions/birdclef-2025/data
