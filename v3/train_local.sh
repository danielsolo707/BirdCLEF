#!/usr/bin/env bash
# v3 training launcher (run via: bash train_v3.sh)
set -e
cd "/c/Users/ASUS/Desktop/DanielEmpire/ML/BirdCLEF"
unset PYTHONPATH
PY="/c/Users/ASUS/AppData/Local/Programs/Python/Python312/python.exe"
echo "=== v3 training start: $(date) ==="
"$PY" -m v3.train \
  --config v3/config.json \
  --metadata data/train.csv \
  --mel-dir data/mels_v3 \
  --output-dir runs/v3_exp001
echo "=== v3 training END: $(date) ==="
