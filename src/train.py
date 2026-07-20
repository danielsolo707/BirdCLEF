"""Train BirdCLEF SED model (EfficientNet-B0 + attention pooling).

Example (Kaggle / local data layout)::

    python -m src.train \\
        --config config.json \\
        --metadata /path/to/train.csv \\
        --audio-dir /path/to/train_audio \\
        --mel-dir /path/to/precomputed_mels \\
        --output-dir runs/exp001

Precomputed mels are optional but ~10× faster when available.
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

from .dataset import BirdCLEFDataset, load_label_maps
from .model import BirdCLEFSED
from .utils import load_config, multilabel_auc, project_root, resolve_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BirdCLEF+ SED model")
    p.add_argument("--config", type=str, default=str(project_root() / "config.json"))
    p.add_argument("--metadata", type=str, required=True, help="Path to train.csv")
    p.add_argument("--audio-dir", type=str, default=None)
    p.add_argument("--mel-dir", type=str, default=None, help="Precomputed .npy mels")
    p.add_argument("--output-dir", type=str, default=str(project_root() / "runs" / "default"))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=None)
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
    num_classes = int(cfg.get("NUM_CLASSES", 206))

    set_seed(seed)
    device = resolve_device(cfg.get("DEVICE", "cuda"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, label2id = load_label_maps(root)
    df = pd.read_csv(args.metadata)

    # Stratify on primary label when available
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
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = BirdCLEFSED(num_classes=num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(enabled=device.type == "cuda")

    history = []
    best_auc = -1.0
    best_path = out_dir / "model_best.pth"

    print(f"Device: {device}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Classes: {num_classes}")
    print(f"Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")

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
            print(f"  ↳ new best checkpoint → {best_path} (AUC={best_auc:.4f})")

    # Also dump final weights
    torch.save(model.state_dict(), out_dir / "model_last.pth")
    save_json(
        {
            "best_val_auc": best_auc,
            "history": history,
            "config": cfg,
            "train_size": len(train_ds),
            "val_size": len(val_ds),
        },
        out_dir / "metrics.json",
    )
    print(f"\nDone. Best validation AUC: {best_auc:.4f}")
    print(f"Artifacts written to: {out_dir}")


if __name__ == "__main__":
    main()
