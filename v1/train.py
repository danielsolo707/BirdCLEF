"""Train BirdCLEF SED model (EfficientNet-B0 + attention pooling).

Training loop aligned with the Kaggle notebook that produced ``model.pth``
(val macro ROC-AUC ≈ 0.8529):

- stratified 80/20 split on ``primary_label``
- precomputed mels preferred (``--mel-dir``)
- AdamW (lr=1e-3, weight_decay=1e-5), CosineAnnealingLR, AMP
- BCEWithLogitsLoss, macro ROC-AUC checkpointing

Example::

    python -m v1.train \\
        --config v1/config.json \\
        --metadata /path/to/train.csv \\
        --audio-dir /path/to/train_audio \\
        --mel-dir /path/to/precomputed_mels \\
        --output-dir runs/exp001
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import BirdCLEFDataset, labels_from_metadata, load_label_maps
from src.model import BirdCLEFSED
from src.utils import (
    default_config_path,
    labels_dir,
    load_config,
    multilabel_auc,
    project_root,
    resolve_device,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BirdCLEF+ SED model")
    p.add_argument("--config", type=str, default=str(default_config_path("v1")))
    p.add_argument("--metadata", type=str, required=True, help="Path to train.csv")
    p.add_argument("--audio-dir", type=str, default=None)
    p.add_argument("--mel-dir", type=str, default=None, help="Precomputed .npy mels")
    p.add_argument("--output-dir", type=str, default=str(project_root() / "runs" / "default"))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (0 matches Kaggle notebook)",
    )
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--build-labels-from-metadata",
        action="store_true",
        help="Ignore classes.json/label2id.json and build labels from this CSV",
    )
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, criterion) -> tuple[float, float]:
    model.eval()
    losses, ys, ps = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast(enabled=device.type == "cuda"):
            logits, _ = model(x)
            loss = criterion(logits, y)
        losses.append(loss.item())
        ys.append(y.cpu().numpy())
        ps.append(torch.sigmoid(logits).cpu().numpy())
    y_true = np.concatenate(ys, axis=0)
    y_prob = np.concatenate(ps, axis=0)
    return float(np.mean(losses)), multilabel_auc(y_true, y_prob)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    root = project_root()

    seed = args.seed if args.seed is not None else int(cfg.get("SEED", 42))
    epochs = args.epochs if args.epochs is not None else int(cfg.get("EPOCHS", 10))
    batch_size = args.batch_size if args.batch_size is not None else int(cfg.get("BATCH_SIZE", 16))
    lr = args.lr if args.lr is not None else float(cfg.get("LR", 1e-3))
    weight_decay = (
        args.weight_decay
        if args.weight_decay is not None
        else float(cfg.get("WEIGHT_DECAY", 1e-5))
    )
    backbone = str(cfg.get("BACKBONE", "efficientnet_b0"))

    set_seed(seed)
    device = resolve_device(cfg.get("DEVICE", "cuda"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metadata)

    has_labels = (labels_dir() / "label2id.json").exists() or (root / "label2id.json").exists()
    if args.build_labels_from_metadata or not has_labels:
        classes, label2id = labels_from_metadata(df)
        save_json(classes, out_dir / "classes.json")
        save_json(label2id, out_dir / "label2id.json")
    else:
        classes, label2id = load_label_maps(root)

    num_classes = len(label2id)
    cfg = {**cfg, "NUM_CLASSES": num_classes}

    stratify = df["primary_label"] if "primary_label" in df.columns else None
    train_df, val_df = train_test_split(
        df, test_size=args.val_ratio, random_state=seed, stratify=stratify
    )

    train_ds = BirdCLEFDataset(
        train_df, label2id, cfg, audio_dir=args.audio_dir, mel_dir=args.mel_dir, train=True
    )
    val_ds = BirdCLEFDataset(
        val_df, label2id, cfg, audio_dir=args.audio_dir, mel_dir=args.mel_dir, train=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    # Kaggle used batch_size * 2 for validation
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = BirdCLEFSED(
        num_classes=num_classes,
        backbone_name=backbone,
        pretrained=True,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(enabled=device.type == "cuda")

    history = []
    best_auc = -1.0
    best_path = out_dir / "model_best.pth"

    print(f"Device: {device}")
    print(f"Backbone: {backbone} | feature_dim={model.feature_dim}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Classes: {num_classes}")
    print(f"Epochs: {epochs} | Batch: {batch_size} | LR: {lr} | WD: {weight_decay}")

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                logits, _ = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        train_loss = running / max(1, len(train_loader))
        val_loss, val_auc = evaluate(model, val_loader, device, criterion)
        elapsed = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "lr": scheduler.get_last_lr()[0],
            "seconds": elapsed,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_auc={val_auc:.4f} | {elapsed:.1f}s"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_path)
            # Also mirror Kaggle artifact name at output root
            torch.save(model.state_dict(), out_dir / "birdclef_best_model.pth")
            print(f"  -> new best checkpoint -> {best_path} (AUC={best_auc:.4f})")

    torch.save(model.state_dict(), out_dir / "model_last.pth")
    save_json(
        {
            "best_val_auc": best_auc,
            "history": history,
            "config": cfg,
            "backbone": backbone,
            "weight_decay": weight_decay,
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "num_classes": num_classes,
            "source": "Aligned with Kaggle notebook danielsolo1770/notebookeb002d87be",
        },
        out_dir / "metrics.json",
    )
    print(f"\nDone. Best validation AUC: {best_auc:.4f}")
    print(f"Artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()
