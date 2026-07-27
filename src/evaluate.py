"""Evaluate a checkpoint on a labeled validation CSV.

Example::

    python -m src.evaluate \\
        --checkpoint model.pth \\
        --metadata /path/to/val.csv \\
        --audio-dir /path/to/audio
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
from .model import load_checkpoint
from .utils import load_config, multilabel_auc, project_root, resolve_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate BirdCLEF SED checkpoint")
    p.add_argument("--checkpoint", type=str, default=str(project_root() / "model.pth"))
    p.add_argument("--config", type=str, default=str(project_root() / "config.json"))
    p.add_argument("--metadata", type=str, required=True)
    p.add_argument("--audio-dir", type=str, default=None)
    p.add_argument("--mel-dir", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    root = project_root()
    device = resolve_device(cfg.get("DEVICE", "cuda"))
    _, label2id = load_label_maps(root)

    df = pd.read_csv(args.metadata)
    ds = BirdCLEFDataset(
        df, label2id, cfg, audio_dir=args.audio_dir, mel_dir=args.mel_dir, train=False
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model = load_checkpoint(
        args.checkpoint,
        num_classes=int(cfg["NUM_CLASSES"]),
        backbone_name=str(cfg.get("BACKBONE", "efficientnet_b0")),
        device=device,
    )

    ys, ps = [], []
    for x, y in tqdm(loader, desc="Evaluating"):
        x = x.to(device)
        logits, _ = model(x)
        ys.append(y.numpy())
        ps.append(torch.sigmoid(logits).cpu().numpy())

    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    auc = multilabel_auc(y_true, y_prob)
    print(f"Samples: {len(ds)}")
    print(f"Macro ROC-AUC: {auc:.4f}")
    print("(Reference training run best val AUC: 0.8529)")


if __name__ == "__main__":
    main()
