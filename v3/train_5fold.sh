#!/usr/bin/env bash
# Stage-1 5-fold CV — preferred credibility metric (mean fold AUC).
# Usage (from repo root):  bash v3/train_5fold.sh
# Wall time: ~5× single fold (plan for a long run on RTX 4070).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
unset PYTHONPATH

if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi
if [ -x "/c/Users/ASUS/AppData/Local/Programs/Python/Python312/python.exe" ]; then
  PY="/c/Users/ASUS/AppData/Local/Programs/Python/Python312/python.exe"
fi

OUT="${OUT_DIR:-runs/v3_5fold}"
echo "=== v3 5-fold train → ${OUT}  $(date) ==="
"$PY" -m v3.train \
  --config v3/config.json \
  --metadata data/train.csv \
  --mel-dir data/mels_v3 \
  --output-dir "$OUT" \
  --folds 5 \
  --batch-size "${BATCH_SIZE:-24}"
echo "=== done $(date) ==="
echo "Read ${OUT}/metrics.json → mean_fold_auc / folds_best_auc"
echo "Then update results/v3/README.md if you promote a new freeze."
