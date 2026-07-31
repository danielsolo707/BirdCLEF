"""Ensemble / blend of BirdCLEF prediction CSVs (submission format).

Each input CSV has a ``row_id`` column plus one probability column per class
(BirdCLEF submission format). Output is the (optionally weighted) mean.

Example::

    python -m src.ensemble \\
        --pred-csvs runs/preds/fold0.csv,runs/preds/fold1.csv,runs/preds/fold2.csv \\
        --weights 1,1,1.2 \\
        --output runs/preds/ensemble.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Average BirdCLEF prediction CSVs")
    p.add_argument("--pred-csvs", type=str, required=True, help="Comma-separated CSV paths")
    p.add_argument("--weights", type=str, default=None, help="Comma-separated weights (default: equal)")
    p.add_argument("--output", type=str, required=True, help="Output CSV path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(s.strip()) for s in args.pred_csvs.split(",") if s.strip()]
    if len(paths) < 2:
        raise SystemExit("Need at least 2 CSVs to ensemble")

    frames = [pd.read_csv(p) for p in paths]
    id_col = "row_id"
    assert all(id_col in f.columns for f in frames), f"All CSVs must have a '{id_col}' column"

    first = frames[0]
    for f in frames[1:]:
        assert list(f[id_col]) == list(first[id_col]), "row_id order must match across CSVs"

    class_cols = [c for c in first.columns if c != id_col]
    weights = (
        [float(w) for w in args.weights.split(",")]
        if args.weights
        else [1.0] * len(paths)
    )
    assert len(weights) == len(paths), "weights count must match CSVs count"
    wsum = sum(weights)

    probs = np.zeros((len(first), len(class_cols)), dtype=np.float64)
    for f, w in zip(frames, weights):
        probs += w * f[class_cols].to_numpy(dtype=np.float64)
    probs /= wsum

    out = pd.DataFrame({id_col: first[id_col]})
    for i, col in enumerate(class_cols):
        out[col] = probs[:, i]
    out.to_csv(args.output, index=False)
    print(f"Ensembled {len(paths)} CSVs (weights {weights}) → {args.output}")
    print(f"Rows: {len(out)} | Classes: {len(class_cols)}")


if __name__ == "__main__":
    main()
