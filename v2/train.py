"""BirdCLEF v2 training — junior-style upgrades that should help for real.

What changed vs v1 (train.py / AUC 0.8529):
  1. SpecAugment on mel spectrograms
  2. BCE with pos_weight (helps rare species a bit)
  3. WeightedRandomSampler by primary_label (see rare birds more often)
  4. Slightly longer training (15 epochs) + slightly lower LR
  5. Richer validation metrics: AUC, PR-AUC, F1, precision, recall

We keep the SAME model (EfficientNet-B0 + SED + attention) so gains come
from training recipe, not a totally different network.

Example (local)::

    python -m v2.train \\
        --config v2/config.json \\
        --metadata path/to/train.csv \\
        --mel-dir path/to/mels \\
        --output-dir runs/v2_exp001

On Kaggle: use v2/kaggle.ipynb (same logic, notebook cells).
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
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.dataset import BirdCLEFDataset, build_target, labels_from_metadata, load_label_maps
from src.metrics import compute_metrics, per_class_report, print_metrics
from src.model import BirdCLEFSED
from src.utils import (
    default_config_path,
    labels_dir,
    load_config,
    project_root,
    resolve_device,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BirdCLEF v2 (better recipe)")
    p.add_argument("--config", type=str, default=str(default_config_path("v2")))
    p.add_argument("--metadata", type=str, required=True, help="Path to train.csv")
    p.add_argument("--audio-dir", type=str, default=None)
    p.add_argument("--mel-dir", type=str, default=None, help="Precomputed .npy mels")
    p.add_argument("--output-dir", type=str, default=str(project_root() / "runs" / "v2_default"))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--val-ratio", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--build-labels-from-metadata",
        action="store_true",
        help="Build classes from this CSV instead of classes.json",
    )
    p.add_argument(
        "--no-pos-weight",
        action="store_true",
        help="Disable pos_weight even if config says use it",
    )
    p.add_argument(
        "--no-balanced-sampler",
        action="store_true",
        help="Disable WeightedRandomSampler",
    )
    p.add_argument(
        "--no-spec-augment",
        action="store_true",
        help="Disable SpecAugment",
    )
    return p.parse_args()


def compute_pos_weight(train_df: pd.DataFrame, label2id: dict, max_weight: float = 50.0) -> torch.Tensor:
    """How rare is each class? Rare classes get higher weight in BCE.

    pos_weight[c] ≈ (#neg / #pos) clipped so it does not explode.
    """
    n = len(train_df)
    num_classes = len(label2id)
    counts = np.zeros(num_classes, dtype=np.float64)

    for _, row in train_df.iterrows():
        t = build_target(row, label2id, num_classes)
        counts += t

    # avoid divide by zero for classes never seen in this split
    pos = np.clip(counts, 1.0, None)
    neg = np.clip(n - counts, 0.0, None)
    w = neg / pos
    w = np.clip(w, 1.0, max_weight).astype(np.float32)
    print(
        f"pos_weight: min={w.min():.2f} max={w.max():.2f} "
        f"median={np.median(w):.2f} | classes with <5 pos: {(counts < 5).sum()}"
    )
    return torch.tensor(w, dtype=torch.float32)


def make_balanced_sampler(train_df: pd.DataFrame) -> WeightedRandomSampler:
    """Sample rare primary labels more often (simple junior trick)."""
    if "primary_label" not in train_df.columns:
        raise KeyError("primary_label column required for balanced sampler")

    counts = train_df["primary_label"].astype(str).value_counts()
    # weight of a row = 1 / count(primary_label)
    weights = train_df["primary_label"].astype(str).map(lambda x: 1.0 / float(counts[x]))
    weights = weights.to_numpy(dtype=np.float64)
    print(
        f"balanced sampler: {len(counts)} primary labels | "
        f"weight range {weights.min():.5f} .. {weights.max():.5f}"
    )
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


@torch.no_grad()
def evaluate_full(model, loader, device, criterion, threshold: float = 0.5) -> dict:
    """Val loss + full metric dict."""
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
        ps.append(torch.sigmoid(logits).float().cpu().numpy())

    y_true = np.concatenate(ys, axis=0)
    y_prob = np.concatenate(ps, axis=0)
    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    metrics["val_loss"] = float(np.mean(losses))
    metrics["y_true"] = y_true  # removed before save
    metrics["y_prob"] = y_prob
    return metrics


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    root = project_root()

    # --- hyperparams (CLI overrides config) ---
    seed = args.seed if args.seed is not None else int(cfg.get("SEED", 42))
    epochs = args.epochs if args.epochs is not None else int(cfg.get("EPOCHS", 15))
    batch_size = (
        args.batch_size if args.batch_size is not None else int(cfg.get("BATCH_SIZE", 16))
    )
    lr = args.lr if args.lr is not None else float(cfg.get("LR", 8e-4))
    weight_decay = (
        args.weight_decay
        if args.weight_decay is not None
        else float(cfg.get("WEIGHT_DECAY", 1e-5))
    )
    val_ratio = (
        args.val_ratio if args.val_ratio is not None else float(cfg.get("VAL_RATIO", 0.2))
    )
    backbone = str(cfg.get("BACKBONE", "efficientnet_b0"))
    dropout = float(cfg.get("DROPOUT", 0.35))
    threshold = float(cfg.get("THRESHOLD", 0.5))
    use_pos_weight = bool(cfg.get("USE_POS_WEIGHT", True)) and not args.no_pos_weight
    use_sampler = bool(cfg.get("USE_BALANCED_SAMPLER", True)) and not args.no_balanced_sampler
    use_spec = bool(cfg.get("USE_SPEC_AUGMENT", True)) and not args.no_spec_augment

    # make sure dataset sees the right flags
    cfg = {
        **cfg,
        "USE_SPEC_AUGMENT": use_spec,
        "USE_POS_WEIGHT": use_pos_weight,
        "USE_BALANCED_SAMPLER": use_sampler,
    }

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

    # same split style as v1 so comparison is fair-ish
    # (true grouped CV can be a later upgrade)
    stratify = df["primary_label"] if "primary_label" in df.columns else None
    train_df, val_df = train_test_split(
        df, test_size=val_ratio, random_state=seed, stratify=stratify
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_ds = BirdCLEFDataset(
        train_df,
        label2id,
        cfg,
        audio_dir=args.audio_dir,
        mel_dir=args.mel_dir,
        train=True,
    )
    val_ds = BirdCLEFDataset(
        val_df,
        label2id,
        cfg,
        audio_dir=args.audio_dir,
        mel_dir=args.mel_dir,
        train=False,
    )

    if use_sampler:
        sampler = make_balanced_sampler(train_df)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=True,
        )
    else:
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
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = BirdCLEFSED(
        num_classes=num_classes,
        backbone_name=backbone,
        pretrained=True,
        dropout=dropout,
    ).to(device)

    if use_pos_weight:
        pos_weight = compute_pos_weight(
            train_df, label2id, max_weight=float(cfg.get("POS_WEIGHT_MAX", 50.0))
        ).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(enabled=device.type == "cuda")

    history = []
    best_auc = -1.0
    best_path = out_dir / "model_best.pth"
    best_metrics = None

    print("======== BirdCLEF v2 training ========")
    print(f"Device: {device}")
    print(f"Backbone: {backbone} | feature_dim={model.feature_dim} | dropout={dropout}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Classes: {num_classes}")
    print(f"Epochs: {epochs} | Batch: {batch_size} | LR: {lr} | WD: {weight_decay}")
    print(f"SpecAugment: {use_spec} | pos_weight: {use_pos_weight} | balanced_sampler: {use_sampler}")
    print(f"v1 baseline to beat: macro ROC-AUC = 0.8529")
    print(f"Output: {out_dir}")

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
        val_metrics = evaluate_full(model, val_loader, device, criterion, threshold=threshold)
        elapsed = time.time() - t0

        # drop big arrays from history
        y_true = val_metrics.pop("y_true")
        y_prob = val_metrics.pop("y_prob")

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "seconds": elapsed,
            "lr": scheduler.get_last_lr()[0],
            **{k: v for k, v in val_metrics.items()},
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['val_loss']:.4f} | "
            f"val_auc={val_metrics['macro_roc_auc']:.4f} | "
            f"macro_f1={val_metrics['macro_f1']:.4f} | "
            f"micro_f1={val_metrics['micro_f1']:.4f} | {elapsed:.1f}s"
        )

        # still select best by macro ROC-AUC (same as v1) for a fair race
        if val_metrics["macro_roc_auc"] > best_auc:
            best_auc = float(val_metrics["macro_roc_auc"])
            best_metrics = dict(val_metrics)
            torch.save(model.state_dict(), best_path)
            torch.save(model.state_dict(), out_dir / "birdclef_v2_best_model.pth")
            print(f"  -> new best checkpoint (AUC={best_auc:.4f})")

            # per-class CSV for the best model
            report = per_class_report(y_true, y_prob, class_names=classes, threshold=threshold)
            pd.DataFrame(report).to_csv(out_dir / "per_class_metrics.csv", index=False)

    torch.save(model.state_dict(), out_dir / "model_last.pth")

    if best_metrics is not None:
        print_metrics(best_metrics, title=f"Best val metrics (AUC={best_auc:.4f})")
        print(f"v1 baseline AUC: 0.8529 | v2 best AUC: {best_auc:.4f} | "
              f"delta: {best_auc - 0.8529:+.4f}")

    summary = {
        "version": "v2",
        "best_val_auc": best_auc,
        "v1_baseline_auc": 0.8529,
        "delta_vs_v1": best_auc - 0.8529 if best_auc >= 0 else None,
        "best_metrics": best_metrics,
        "history": history,
        "config": cfg,
        "backbone": backbone,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "num_classes": num_classes,
        "upgrades": [
            "SpecAugment" if use_spec else None,
            "BCE pos_weight" if use_pos_weight else None,
            "WeightedRandomSampler(primary_label)" if use_sampler else None,
            f"epochs={epochs}",
            f"lr={lr}",
            f"dropout={dropout}",
        ],
        "notes": [
            "Same EfficientNet-B0 SED architecture as v1.",
            "Best checkpoint chosen by validation macro ROC-AUC.",
            "F1/precision/recall use threshold from config (default 0.5).",
            "element_accuracy is easy to look high with sparse multi-label — prefer F1/AUC.",
        ],
    }
    # clean upgrades list
    summary["upgrades"] = [u for u in summary["upgrades"] if u is not None]
    save_json(summary, out_dir / "metrics.json")
    save_json(cfg, out_dir / "config_used.json")

    print(f"\nDone. Best validation AUC: {best_auc:.4f}")
    print(f"Artifacts written to: {out_dir}")
    print("Upload best weights + metrics.json when you freeze artifacts/v2_* locally.")


if __name__ == "__main__":
    main()
