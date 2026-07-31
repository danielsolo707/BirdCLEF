"""Simple multi-label metrics for BirdCLEF.

Junior-style helpers: one function returns a clear dict of numbers.
Primary ranking metric is still macro ROC-AUC (same as v1 = 0.8529).
F1 / precision / recall need a threshold (default 0.5).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def multilabel_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Macro ROC-AUC over classes that have both 0 and 1 in y_true."""
    scores = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].min() == y_true[:, c].max():
            continue
        scores.append(roc_auc_score(y_true[:, c], y_prob[:, c]))
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def multilabel_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Macro average precision (PR-AUC) over non-degenerate classes."""
    scores = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() <= 0:
            continue
        scores.append(average_precision_score(y_true[:, c], y_prob[:, c]))
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute the main numbers we care about for v2.

    Parameters
    ----------
    y_true : (N, C) float/int multi-hot labels
    y_prob : (N, C) predicted probabilities
    threshold : used for F1 / precision / recall / accuracy-style metrics
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(np.float32)

    # Element-wise accuracy (careful: easy to look high with many zeros)
    element_acc = float((y_pred == y_true).mean())

    out = {
        "threshold": float(threshold),
        "macro_roc_auc": multilabel_auc(y_true, y_prob),
        "macro_pr_auc": multilabel_pr_auc(y_true, y_prob),
        "micro_f1": float(
            f1_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "micro_precision": float(
            precision_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "micro_recall": float(
            recall_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "element_accuracy": element_acc,
        "n_samples": int(y_true.shape[0]),
        "n_classes": int(y_true.shape[1]),
    }
    return out


def per_class_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str] | None = None,
    threshold: float = 0.5,
) -> list[dict]:
    """One row per class — useful for finding weak / rare species."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(np.float32)
    n_classes = y_true.shape[1]
    rows = []

    for c in range(n_classes):
        name = class_names[c] if class_names is not None else str(c)
        yt = y_true[:, c]
        yp = y_prob[:, c]
        yd = y_pred[:, c]
        support = int(yt.sum())

        if yt.min() != yt.max():
            try:
                auc = float(roc_auc_score(yt, yp))
            except ValueError:
                auc = float("nan")
        else:
            auc = float("nan")

        if support > 0:
            try:
                ap = float(average_precision_score(yt, yp))
            except ValueError:
                ap = float("nan")
        else:
            ap = float("nan")

        rows.append(
            {
                "class_id": c,
                "class_name": name,
                "support": support,
                "roc_auc": auc,
                "pr_auc": ap,
                "f1": float(f1_score(yt, yd, zero_division=0)),
                "precision": float(precision_score(yt, yd, zero_division=0)),
                "recall": float(recall_score(yt, yd, zero_division=0)),
            }
        )
    return rows


def print_metrics(metrics: dict, title: str = "Metrics") -> None:
    """Pretty print for notebooks / terminal."""
    print(f"\n=== {title} ===")
    order = [
        "macro_roc_auc",
        "macro_pr_auc",
        "micro_f1",
        "macro_f1",
        "micro_precision",
        "macro_precision",
        "micro_recall",
        "macro_recall",
        "hamming_loss",
        "element_accuracy",
        "threshold",
        "n_samples",
        "n_classes",
    ]
    for key in order:
        if key not in metrics:
            continue
        val = metrics[key]
        if isinstance(val, float):
            print(f"  {key:20s}: {val:.4f}")
        else:
            print(f"  {key:20s}: {val}")
