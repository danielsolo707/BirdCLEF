"""Shared helpers: config, paths, seeding, metrics."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def project_root() -> Path:
    """Repo root (parent of ``src/``)."""
    return Path(__file__).resolve().parent.parent


def configs_dir() -> Path:
    return project_root()


def labels_dir() -> Path:
    return project_root() / "labels"


def models_dir() -> Path:
    return project_root() / "models"


def results_dir() -> Path:
    return project_root() / "results"


def default_config_path(version: str = "v1") -> Path:
    if version in ("v3", "3"):
        return project_root() / "v3" / "config.json"
    if version in ("v2", "2"):
        return project_root() / "v2" / "config.json"
    return project_root() / "v1" / "config.json"


def default_checkpoint_path() -> Path:
    """Published best weights (v1 until a newer run is promoted)."""
    return models_dir() / "v1" / "model.pth"


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(preferred: str = "cuda") -> torch.device:
    if preferred.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def multilabel_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Macro ROC-AUC over classes that appear in y_true.

    Kept here so v1 ``train.py`` keeps working. v2 prefers ``src.metrics``.
    """
    scores = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].min() == y_true[:, c].max():
            continue  # skip degenerate classes in this batch/set
        scores.append(roc_auc_score(y_true[:, c], y_prob[:, c]))
    if not scores:
        return float("nan")
    return float(np.mean(scores))
