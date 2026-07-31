"""Evaluate a checkpoint with v2-style multi-label metrics.

Example::

    python -m src.evaluate \\
        --checkpoint models/model.pth \\
        --config configs/config.json \\
        --metadata /path/to/val.csv \\
        --audio-dir /path/to/audio

    python -m src.evaluate \\
        --checkpoint runs/v2_exp001/model_best.pth \\
        --config configs/config_v2.json \\
        --metadata /path/to/train.csv \\
        --mel-dir /path/to/mels \\
        --save-dir runs/v2_exp001/eval
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import BirdCLEFDataset, load_label_maps
from .metrics import compute_metrics, per_class_report, print_metrics
from .model import load_checkpoint
from .utils import (
    default_checkpoint_path,
    default_config_path,
    load_config,
    project_root,
    resolve_device,
    save_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate BirdCLEF SED checkpoint")
    p.add_argument("--checkpoint", type=str, default=str(default_checkpoint_path()))
    p.add_argument("--config", type=str, default=str(default_config_path("v1")))
    p.add_argument("--metadata", type=str, required=True)
    p.add_argument("--audio-dir", type=str, default=None)
    p.add_argument("--mel-dir", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="If set, write metrics.json + per_class_metrics.csv here",
    )
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    root = project_root()
    device = resolve_device(cfg.get("DEVICE", "cuda"))
    classes, label2id = load_label_maps(root)

    df = pd.read_csv(args.metadata)
    ds = BirdCLEFDataset(
        df, label2id, cfg, audio_dir=args.audio_dir, mel_dir=args.mel_dir, train=False
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = load_checkpoint(
        args.checkpoint,
        num_classes=int(cfg.get("NUM_CLASSES", len(label2id))),
        backbone_name=str(cfg.get("BACKBONE", "efficientnet_b0")),
        device=device,
    )

    ys, ps = [], []
    for x, y in tqdm(loader, desc="Evaluating"):
        x = x.to(device)
        logits, _ = model(x)
        ys.append(y.numpy())
        ps.append(torch.sigmoid(logits).float().cpu().numpy())

    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    metrics = compute_metrics(y_true, y_prob, threshold=args.threshold)
    print_metrics(metrics, title="Evaluation")
    print("(v1 reference training run best val AUC: 0.8529)")

    report = per_class_report(
        y_true, y_prob, class_names=classes, threshold=args.threshold
    )

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_json(metrics, save_dir / "metrics.json")
        pd.DataFrame(report).to_csv(save_dir / "per_class_metrics.csv", index=False)
        print(f"Saved metrics to {save_dir}")


if __name__ == "__main__":
    main()
