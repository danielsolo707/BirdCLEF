"""BirdCLEF v3 — training aligned with the 4th-place BirdCLEF+ 2025 solution.

Recipe (see docs/V3_PLAN.md and docs/V3_TRAINING.md):
  - 10 s audio, power mel (n_fft 2048, hop 64, n_mels 256, fmin 60, fmax 16000)
    resized to (256, 512), power_to_db at load
  - v3 SED model: bn0 + timm backbone + freq-mean + channel smoothing + AttBlock
  - Loss: SoftAUCLoss (hard labels) — metric-aligned, overfitting-resistant
  - Spec mixup, spec augment, rare-class upsampling, k-fold CV
  - Optional Stage-2: --semi-csv with pseudo-labeled soundscape chunks

Example (local RTX 4070, single fold)::

    python -m v3.train \\
        --config v3/config.json \\
        --metadata path/to/train.csv \\
        --mel-dir path/to/mels_v3 \\
        --output-dir runs/v3_exp001 \\
        --fold 0 --folds 5 --batch-size 24

With pseudo-labels (after running v3.pseudo_label)::

    python -m v3.train \\
        --config v3/config.json \\
        --metadata path/to/train.csv \\
        --semi-csv runs/pseudo/semi_chunks.csv \\
        --mel-dir path/to/mels_v3 \\
        --output-dir runs/v3_semi_exp001 \\
        --fold 0 --folds 5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import BirdCLEFDatasetV3, labels_from_metadata, load_label_maps
from src.losses import get_criterion
from src.metrics import compute_metrics, per_class_report, print_metrics
from src.model import BirdCLEFModelV3
from src.utils import (
    default_config_path,
    labels_dir,
    load_config,
    project_root,
    resolve_device,
    save_json,
    set_seed,
)

V1_BASELINE_AUC = 0.8529


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BirdCLEF v3 (4th-place-style recipe)")
    p.add_argument("--config", type=str, default=str(default_config_path("v3")))
    p.add_argument("--metadata", type=str, required=True, help="Path to train.csv")
    p.add_argument("--audio-dir", type=str, default=None)
    p.add_argument("--mel-dir", type=str, default=None, help="Precomputed v3 power-mel .npy dir")
    p.add_argument("--semi-csv", type=str, default=None, help="Pseudo-labeled chunks CSV (Stage 2)")
    p.add_argument("--output-dir", type=str, default=str(project_root() / "runs" / "v3_default"))
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--backbone", type=str, default=None)
    p.add_argument("--criterion", type=str, default=None, help="SoftAUCLoss|AUCLoss|FocalBCELoss|BCE")
    p.add_argument("--folds", type=int, default=1, help="Number of CV folds (1 = single 80/20 split)")
    p.add_argument("--fold", type=int, default=None, help="Train only this fold index (else all)")
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--val-ratio", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-mixup", action="store_true")
    p.add_argument("--no-upsample", action="store_true")
    p.add_argument("--build-labels-from-metadata", action="store_true")
    return p.parse_args()


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def mixup_batch(inputs: torch.Tensor, targets: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Spec mixup (4th-place style): blend inputs, blend the loss over both targets."""
    lam = float(np.random.beta(alpha, alpha))
    indices = torch.randperm(inputs.size(0), device=inputs.device)
    mixed = lam * inputs + (1 - lam) * inputs[indices]
    return mixed, targets, targets[indices], lam


@torch.no_grad()
def evaluate_full(model, loader, device, channels_last: bool = False) -> dict:
    model.eval()
    ys, ps = [], []
    use_amp = device.type == "cuda"
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if channels_last:
            x = x.to(memory_format=torch.channels_last)
        with autocast(enabled=use_amp):
            logits, _ = model(x)
        ys.append(y.float().cpu().numpy())
        ps.append(torch.sigmoid(logits).float().cpu().numpy())

    y_true = np.concatenate(ys, axis=0)
    y_prob = np.concatenate(ps, axis=0)
    metrics = compute_metrics(y_true, y_prob, threshold=0.5)
    metrics["y_true"] = y_true
    metrics["y_prob"] = y_prob
    return metrics


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    root = project_root()

    seed = args.seed if args.seed is not None else int(cfg.get("SEED", 42))
    epochs = args.epochs if args.epochs is not None else int(cfg.get("EPOCHS", 20))
    batch_size = args.batch_size if args.batch_size is not None else int(cfg.get("BATCH_SIZE", 24))
    lr = args.lr if args.lr is not None else float(cfg.get("LR", 5e-4))
    num_workers = args.num_workers if args.num_workers is not None else int(cfg.get("NUM_WORKERS", 4))
    val_ratio = args.val_ratio if args.val_ratio is not None else float(cfg.get("VAL_RATIO", 0.2))
    backbone = args.backbone if args.backbone is not None else str(cfg.get("BACKBONE", "efficientnet_b0"))
    criterion_name = args.criterion if args.criterion is not None else str(cfg.get("CRITERION", "SoftAUCLoss"))
    n_folds = max(1, args.folds)
    weight_decay = float(cfg.get("WEIGHT_DECAY", 1e-5))
    dropout = float(cfg.get("DROPOUT", 0.5))
    drop_path_rate = float(cfg.get("DROP_PATH_RATE", 0.3))
    mixup_alpha = float(cfg.get("MIXUP_ALPHA", 0.5))
    mixup_rate = float(cfg.get("MIXUP_RATE", 1.0)) if not args.no_mixup else 0.0
    min_samples = 0 if args.no_upsample else int(cfg.get("MIN_SAMPLES_PER_CLASS", 100))
    channels_last = bool(cfg.get("CHANNELS_LAST", True))

    set_seed(seed)
    device = resolve_device(cfg.get("DEVICE", "cuda"))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- labels -----------------------------------------------------------
    has_labels = (labels_dir() / "label2id.json").exists() or (root / "label2id.json").exists()
    if args.build_labels_from_metadata or not has_labels:
        df0 = pd.read_csv(args.metadata)
        classes, label2id = labels_from_metadata(df0)
        save_json(classes, out_dir / "classes.json")
        save_json(label2id, out_dir / "label2id.json")
    else:
        classes, label2id = load_label_maps(root)
    num_classes = len(label2id)
    cfg = {**cfg, "NUM_CLASSES": num_classes, "MIN_SAMPLES_PER_CLASS": min_samples}

    # ---- data -------------------------------------------------------------
    df = pd.read_csv(args.metadata)
    stratify = df["primary_label"] if "primary_label" in df.columns else None

    if args.semi_csv:
        semi = pd.read_csv(args.semi_csv)
        # pseudo rows carry a prebuilt soft 'target' column (list of floats)
        if "target" in semi.columns:
            semi["target"] = semi["target"].map(
                lambda s: np.fromstring(s.strip("[]"), sep=",", dtype=np.float32)
                if isinstance(s, str) else np.asarray(s, dtype=np.float32)
            )
        print(f"[v3] semi-supervised: {len(semi)} pseudo-labeled chunks")
    else:
        semi = None

    folds_aucs = []
    summaries = []

    def build_loaders(train_idx, val_idx):
        train_df = df.iloc[train_idx].copy()
        if semi is not None:
            train_df = pd.concat([train_df, semi], ignore_index=True)
        val_df = df.iloc[val_idx].copy()

        train_ds = BirdCLEFDatasetV3(
            train_df, label2id, cfg, audio_dir=args.audio_dir, mel_dir=args.mel_dir, train=True
        )
        val_ds = BirdCLEFDatasetV3(
            val_df, label2id, cfg, audio_dir=args.audio_dir, mel_dir=args.mel_dir, train=False
        )
        loader_kw = dict(num_workers=num_workers, pin_memory=device.type == "cuda")
        if num_workers > 0:
            loader_kw["persistent_workers"] = True
            loader_kw["prefetch_factor"] = 2
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kw)
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, **loader_kw)
        return train_loader, val_loader

    fold_list = [args.fold] if args.fold is not None else list(range(n_folds))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed) if n_folds > 1 else None

    print("======== BirdCLEF v3 (4th-place-style) ========")
    print(f"Device: {device} | Backbone: {backbone} | Criterion: {criterion_name}")
    print(f"Epochs: {epochs} | Batch: {batch_size} | LR: {lr} | WD: {weight_decay}")
    print(f"Folds: {n_folds} (training folds {fold_list}) | Mixup(α={mixup_alpha}, p={mixup_rate})")
    print(f"Rare-class upsampling to {min_samples} | Semi chunks: {0 if semi is None else len(semi)}")
    print(f"v1 baseline to beat: macro ROC-AUC = {V1_BASELINE_AUC}")
    print(f"Output: {out_dir}")

    for fold in fold_list:
        print(f"\n----- Fold {fold} -----")
        if n_folds > 1:
            train_idx, val_idx = list(skf.split(df, stratify))[fold]
        else:
            from sklearn.model_selection import train_test_split

            train_idx, val_idx = train_test_split(
                np.arange(len(df)), test_size=val_ratio, random_state=seed, stratify=stratify
            )

        train_loader, val_loader = build_loaders(train_idx, val_idx)

        model = BirdCLEFModelV3(
            num_classes=num_classes,
            backbone_name=backbone,
            pretrained=True,
            n_mels=int(cfg.get("TARGET_MELS", 256)),
            dropout=dropout,
            drop_path_rate=drop_path_rate,
        ).to(device)
        if channels_last and device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)

        criterion = get_criterion(criterion_name)
        optimizer = torch.optim.AdamW(unwrap_model(model).parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=float(cfg.get("MIN_LR", 1e-6)))
        scaler = GradScaler(enabled=device.type == "cuda")

        history = []
        best_auc = -1.0
        best_path = out_dir / f"model_fold{fold}_best.pth"

        for epoch in range(1, epochs + 1):
            model.train()
            t0 = time.time()
            running = 0.0
            pbar = tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch}/{epochs}", leave=False)
            for x, y in pbar:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                if channels_last and device.type == "cuda":
                    x = x.to(memory_format=torch.channels_last)
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=device.type == "cuda"):
                    if mixup_rate > 0 and np.random.rand() < mixup_rate:
                        xm, y1, y2, lam = mixup_batch(x, y, mixup_alpha)
                        logits, _ = model(xm)
                        loss = lam * criterion(logits, y1) + (1 - lam) * criterion(logits, y2)
                    else:
                        logits, _ = model(x)
                        loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            scheduler.step()
            train_loss = running / max(1, len(train_loader))
            val_metrics = evaluate_full(model, val_loader, device, channels_last=channels_last)
            y_true = val_metrics.pop("y_true")
            y_prob = val_metrics.pop("y_prob")
            elapsed = time.time() - t0

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
                f"val_auc={val_metrics['macro_roc_auc']:.4f} | "
                f"macro_f1={val_metrics['macro_f1']:.4f} | {elapsed:.1f}s"
            )

            if val_metrics["macro_roc_auc"] > best_auc:
                best_auc = float(val_metrics["macro_roc_auc"])
                torch.save(unwrap_model(model).state_dict(), best_path)
                print(f"  -> new best fold{fold} checkpoint (AUC={best_auc:.4f})")
                report = per_class_report(y_true, y_prob, class_names=classes, threshold=0.5)
                pd.DataFrame(report).to_csv(out_dir / f"per_class_metrics_fold{fold}.csv", index=False)

        torch.save(unwrap_model(model).state_dict(), out_dir / f"model_fold{fold}_last.pth")
        folds_aucs.append(best_auc)
        print(f"Fold {fold} best AUC: {best_auc:.4f}")

        summaries.append(
            {
                "fold": fold,
                "best_val_auc": best_auc,
                "history": history,
                "train_size": len(train_loader.dataset),
                "val_size": len(val_loader.dataset),
            }
        )

    mean_auc = float(np.mean(folds_aucs)) if folds_aucs else -1.0
    summary = {
        "version": "v3",
        "recipe": "4th-place-style (SoftAUC + 10s + power-mel 256x512 + AttBlock + semi)",
        "folds_trained": fold_list,
        "folds_best_auc": folds_aucs,
        "mean_fold_auc": mean_auc,
        "v1_baseline_auc": V1_BASELINE_AUC,
        "delta_vs_v1": mean_auc - V1_BASELINE_AUC,
        "config": cfg,
        "backbone": backbone,
        "criterion": criterion_name,
        "fold_summaries": summaries,
        "honesty": {
            "comparison_type": "system_upgrade_not_pure_ablation",
            "loss_optimizes": "soft_surrogate_of_macro_ROC_AUC",
            "cv_mode": f"folds={n_folds}",
            "split": "stratified_primary_label" if n_folds == 1 else "StratifiedKFold_primary_label",
            "not_site_grouped": True,
        },
        "notes": [
            "v3 is a system upgrade vs v1 (features + AttBlock + SoftAUC + mixup), not a pure ablation.",
            "SoftAUC optimizes ranking — also inspect macro_f1 / macro_pr_auc in fold history.",
            "Best checkpoint per fold: model_fold{fold}_best.pth.",
            "Next credibility step: --folds 5 (bash v3/train_5fold.sh) → mean_fold_auc.",
            "Next LB-style step: bash v3/stage2_pseudo.sh (pseudo soundscapes + --semi-csv).",
            "Inference: src.inference --model-version v3 --overlap --smooth --postprocess; blend with src.ensemble.",
        ],
    }
    save_json(summary, out_dir / "metrics.json")
    print(f"\nDone. Mean fold AUC: {mean_auc:.4f} | v1 baseline: {V1_BASELINE_AUC} | delta: {mean_auc - V1_BASELINE_AUC:+.4f}")
    print(f"Artifacts written to: {out_dir}")
    if n_folds == 1:
        print("Note: single-fold run. For a more credible mean, re-run with --folds 5 (see results/v3/NEXT_STEPS.md).")


if __name__ == "__main__":
    main()
