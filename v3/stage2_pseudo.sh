#!/usr/bin/env bash
# Stage 2: pseudo-label train_soundscapes with Stage-1 teacher(s), then re-train.
# Usage (from repo root):
#   TEACHER=runs/v3_exp001/model_fold0_best.pth bash v3/stage2_pseudo.sh
# Or after 5-fold:
#   TEACHER=runs/v3_5fold/model_fold0_best.pth,runs/v3_5fold/model_fold1_best.pth bash v3/stage2_pseudo.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
unset PYTHONPATH

if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi
if [ -x "/c/Users/ASUS/AppData/Local/Programs/Python/Python312/python.exe" ]; then
  PY="/c/Users/ASUS/AppData/Local/Programs/Python/Python312/python.exe"
fi

TEACHER="${TEACHER:-models/v3/model_best.pth}"
SCAPES="${SOUNDSCAPES:-data/train_soundscapes}"
MEL_SEMI="${MEL_SEMI:-data/mels_v3_semi}"
PSEUDO_CSV="${PSEUDO_CSV:-runs/pseudo/semi_chunks.csv}"
OUT="${OUT_DIR:-runs/v3_semi_exp001}"

if [ ! -d "$SCAPES" ]; then
  echo "ERROR: soundscapes dir not found: $SCAPES"
  exit 1
fi

mkdir -p "$(dirname "$PSEUDO_CSV")" "$OUT"

echo "=== Stage 2a: pseudo-label with teacher(s): $TEACHER ==="
"$PY" -m v3.pseudo_label \
  --config v3/config.json \
  --checkpoints "$TEACHER" \
  --soundscapes-dir "$SCAPES" \
  --mel-dir "$MEL_SEMI" \
  --output-csv "$PSEUDO_CSV" \
  --threshold "${PSEUDO_THR:-0.3}"

echo "=== Stage 2b: re-train with --semi-csv ==="
"$PY" -m v3.train \
  --config v3/config.json \
  --metadata data/train.csv \
  --semi-csv "$PSEUDO_CSV" \
  --mel-dir data/mels_v3 \
  --output-dir "$OUT" \
  --fold 0 \
  --folds 1 \
  --batch-size "${BATCH_SIZE:-24}"

echo "=== Stage 2 done $(date) ==="
echo "Compare $OUT/metrics.json to results/v3 (champion 0.9694) before promoting."
