"""Fast local checks for v2 code — NOT a full training run.

Run from repo root::

    python scripts/smoke_test_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.dataset import SpecAugment, build_train_transform, build_target
from src.metrics import compute_metrics, print_metrics
from src.model import BirdCLEFSED
from src.utils import load_config


def test_spec_augment() -> None:
    x = torch.rand(1, 128, 256)
    aug = SpecAugment(freq_masks=2, time_masks=2, freq_width=16, time_width=32)
    y = aug(x)
    assert y.shape == x.shape
    # should zero something most of the time
    assert torch.isfinite(y).all()
    print("[ok] SpecAugment shape/finite")


def test_build_transform() -> None:
    cfg = load_config(ROOT / "v2" / "config.json")
    t = build_train_transform(cfg)
    x = torch.rand(1, 128, 256)
    y = t(x)
    assert y.shape == x.shape
    print("[ok] build_train_transform(config_v2)")


def test_metrics() -> None:
    rng = np.random.default_rng(0)
    n, c = 64, 10
    y_true = (rng.random((n, c)) > 0.8).astype(np.float32)
    # make sure every class has both labels for AUC
    y_true[0, :] = 0
    y_true[1, :] = 1
    y_prob = np.clip(y_true * 0.7 + rng.random((n, c)) * 0.3, 0, 1)
    m = compute_metrics(y_true, y_prob, threshold=0.5)
    assert "macro_roc_auc" in m
    assert "macro_f1" in m
    assert "macro_precision" in m
    assert "macro_recall" in m
    print_metrics(m, title="smoke metrics")
    print("[ok] compute_metrics")


def test_model_forward() -> None:
    model = BirdCLEFSED(num_classes=8, backbone_name="efficientnet_b0", pretrained=False, dropout=0.35)
    model.eval()
    x = torch.randn(2, 1, 128, 256)
    logits, frame = model(x)
    assert logits.shape == (2, 8)
    assert frame.shape[0] == 2 and frame.shape[1] == 8
    print("[ok] model forward", tuple(logits.shape), tuple(frame.shape))


def test_one_train_step() -> None:
    """One fake batch — proves loss/backward path works with pos_weight."""
    device = torch.device("cpu")
    model = BirdCLEFSED(num_classes=8, backbone_name="efficientnet_b0", pretrained=False, dropout=0.35)
    model.train()
    pos_weight = torch.ones(8) * 2.0
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randn(4, 1, 128, 256)
    y = torch.zeros(4, 8)
    y[:, 0] = 1.0
    y[1, 3] = 1.0

    # apply aug like dataset would
    aug = SpecAugment()
    x_aug = torch.stack([aug(xi) for xi in x])

    opt.zero_grad()
    logits, _ = model(x_aug)
    loss = criterion(logits, y)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss).item()
    print(f"[ok] one train step loss={loss.item():.4f}")


def test_build_target() -> None:
    import pandas as pd

    label2id = {"a": 0, "b": 1, "c": 2}
    row = pd.Series({"primary_label": "a", "secondary_labels": "['b']"})
    t = build_target(row, label2id, 3)
    assert t.tolist() == [1.0, 1.0, 0.0]
    print("[ok] build_target")


def main() -> None:
    print("Running v2 smoke tests (no full training)...")
    test_spec_augment()
    test_build_transform()
    test_metrics()
    test_build_target()
    test_model_forward()
    test_one_train_step()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
