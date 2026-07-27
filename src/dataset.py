"""Dataset loaders for BirdCLEF+ 2025 multi-label training.

Aligned with the Kaggle notebook:
- multi-hot targets from primary + secondary labels (``ast.literal_eval``)
- precomputed mel cache names: ``folder_file.npy``
- train augs: horizontal flip + brightness/contrast jitter (torchvision v2)
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .audio import get_melspec, mel_cache_name

try:
    from torchvision.transforms import v2 as T

    _HAS_TORCHVISION = True
except ImportError:  # pragma: no cover
    T = None
    _HAS_TORCHVISION = False


def build_target(row: pd.Series, label2id: dict[str, int], num_classes: int) -> np.ndarray:
    """Multi-hot target from primary_label + secondary_labels (Kaggle-style)."""
    labels: list[str] = []

    if "primary_label" in row and pd.notna(row["primary_label"]):
        labels.append(str(row["primary_label"]))

    if "secondary_labels" in row and pd.notna(row["secondary_labels"]):
        sec = row["secondary_labels"]
        if isinstance(sec, (list, tuple)):
            labels.extend(str(s) for s in sec)
        else:
            sec_str = str(sec).strip()
            if sec_str and sec_str.lower() not in ("nan", "none", ""):
                try:
                    parsed = ast.literal_eval(sec_str)
                    if isinstance(parsed, list):
                        labels.extend(str(s) for s in parsed)
                    elif isinstance(parsed, str) and parsed:
                        labels.append(parsed)
                except (ValueError, SyntaxError):
                    clean = sec_str.strip("[]").replace("'", "").replace('"', "")
                    labels.extend(s.strip() for s in clean.split(",") if s.strip())

    if not labels and "label" in row and pd.notna(row["label"]):
        labels.append(str(row["label"]))

    target = np.zeros(num_classes, dtype=np.float32)
    for lab in labels:
        if lab in label2id:
            target[label2id[lab]] = 1.0
    return target


def default_train_transform():
    """Same light augs as the Kaggle notebook (flip + color jitter on mel)."""
    if not _HAS_TORCHVISION:
        return None
    return T.Compose(
        [
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.15, contrast=0.15),
        ]
    )


class BirdCLEFDataset(Dataset):
    """Multi-label dataset over raw audio or precomputed mel ``.npy`` files.

    Expected metadata CSV columns (BirdCLEF-style):
      - ``filename``, ``primary_label``, optional ``secondary_labels``

    Precomputed mels under ``mel_dir`` use Kaggle naming::

        1139490/CSA36385.ogg  →  1139490_CSA36385.npy
    """

    def __init__(
        self,
        df: pd.DataFrame,
        label2id: dict[str, int],
        cfg: dict,
        audio_dir: str | Path | None = None,
        mel_dir: str | Path | None = None,
        train: bool = True,
        transform=None,
    ):
        self.df = df.reset_index(drop=True)
        self.label2id = label2id
        self.num_classes = len(label2id)
        self.cfg = cfg
        self.audio_dir = Path(audio_dir) if audio_dir else None
        self.mel_dir = Path(mel_dir) if mel_dir else None
        self.train = train

        if transform is not None:
            self.transform = transform
        elif train:
            self.transform = default_train_transform()
        else:
            self.transform = None

        if self.audio_dir is None and self.mel_dir is None:
            raise ValueError("Provide at least one of audio_dir or mel_dir")

        # Optional: pre-built multi-hot column (list/ndarray) from notebook-style prep
        self._has_target_col = "target" in self.df.columns

    def __len__(self) -> int:
        return len(self.df)

    def _filename(self, row: pd.Series) -> str:
        for col in ("filename", "file", "audio_path", "path"):
            if col in row and pd.notna(row[col]):
                return str(row[col]).replace("\\", "/")
        raise KeyError("Row is missing a filename column")

    def _resolve_mel_path(self, fname: str) -> Path | None:
        if self.mel_dir is None:
            return None
        candidates = [
            self.mel_dir / mel_cache_name(fname),
            self.mel_dir / f"{Path(fname).stem}.npy",
            self.mel_dir / Path(fname).with_suffix(".npy").name,
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        fname = self._filename(row)

        mel_path = self._resolve_mel_path(fname)
        if mel_path is not None:
            mel = np.load(mel_path).astype(np.float32)
            tw = int(self.cfg.get("TARGET_WIDTH", 256))
            if mel.shape[1] != tw:
                from .audio import fix_time_width

                mel = fix_time_width(mel, target_width=tw)
        else:
            mel = self._compute_mel(fname)

        x = torch.from_numpy(mel).unsqueeze(0)  # (1, M, T)
        if self.train and self.transform is not None:
            x = self.transform(x)

        if self._has_target_col:
            target = row["target"]
            if isinstance(target, torch.Tensor):
                y = target.float()
            else:
                y = torch.tensor(np.asarray(target, dtype=np.float32), dtype=torch.float32)
        else:
            y = torch.from_numpy(build_target(row, self.label2id, self.num_classes))
        return x, y

    def _compute_mel(self, fname: str) -> np.ndarray:
        assert self.audio_dir is not None
        path = self.audio_dir / fname
        if not path.exists():
            matches = list(self.audio_dir.rglob(Path(fname).name))
            if not matches:
                raise FileNotFoundError(path)
            path = matches[0]
        return get_melspec(
            path,
            sr=int(self.cfg["SR"]),
            duration=float(self.cfg["DURATION"]),
            n_fft=int(self.cfg["N_FFT"]),
            hop_length=int(self.cfg["HOP_LENGTH"]),
            n_mels=int(self.cfg["N_MELS"]),
            fmin=int(self.cfg["FMIN"]),
            fmax=int(self.cfg["FMAX"]),
            target_width=int(self.cfg.get("TARGET_WIDTH", 256)),
        )


def load_label_maps(root: str | Path) -> tuple[list[str], dict[str, int]]:
    root = Path(root)
    classes_path = root / "classes.json"
    label2id_path = root / "label2id.json"

    if label2id_path.exists():
        with open(label2id_path, encoding="utf-8") as f:
            label2id = json.load(f)
        label2id = {k: int(v) for k, v in label2id.items()}
        classes: list[str | None] = [None] * len(label2id)
        for k, v in label2id.items():
            classes[v] = k
        return [c if c is not None else f"class_{i}" for i, c in enumerate(classes)], label2id

    with open(classes_path, encoding="utf-8") as f:
        classes = json.load(f)
    label2id = {c: i for i, c in enumerate(classes)}
    return classes, label2id


def labels_from_metadata(df: pd.DataFrame) -> tuple[list[str], dict[str, int]]:
    """Build label maps from a training CSV (sorted unique primary labels)."""
    if "primary_label" not in df.columns:
        raise KeyError("metadata CSV must contain a primary_label column")
    primary_labels = sorted(df["primary_label"].astype(str).unique().tolist())
    label2id = {s: i for i, s in enumerate(primary_labels)}
    return primary_labels, label2id
