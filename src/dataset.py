"""Dataset loaders for BirdCLEF+ 2025 multi-label training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .audio import load_audio, waveform_to_mel


class BirdCLEFDataset(Dataset):
    """Multi-label dataset over raw audio or precomputed mel `.npy` files.

    Expected metadata CSV columns (BirdCLEF-style):
      - filename / primary_label / secondary_labels (flexible; see notes)
      - or a pre-built multi-hot column set

    For precomputed mels, place files under ``mel_dir`` named
    ``{stem}.npy`` with shape ``(n_mels, time)``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        label2id: dict[str, int],
        cfg: dict,
        audio_dir: str | Path | None = None,
        mel_dir: str | Path | None = None,
        train: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.label2id = label2id
        self.num_classes = len(label2id)
        self.cfg = cfg
        self.audio_dir = Path(audio_dir) if audio_dir else None
        self.mel_dir = Path(mel_dir) if mel_dir else None
        self.train = train

        if self.audio_dir is None and self.mel_dir is None:
            raise ValueError("Provide at least one of audio_dir or mel_dir")

    def __len__(self) -> int:
        return len(self.df)

    def _multi_hot(self, row: pd.Series) -> np.ndarray:
        target = np.zeros(self.num_classes, dtype=np.float32)
        labels: list[str] = []

        if "primary_label" in row and pd.notna(row["primary_label"]):
            labels.append(str(row["primary_label"]))

        if "secondary_labels" in row and pd.notna(row["secondary_labels"]):
            sec = row["secondary_labels"]
            if isinstance(sec, str):
                # Kaggle often stores list-like strings: "['a', 'b']"
                sec = sec.strip("[]").replace("'", "").replace('"', "")
                labels.extend([s.strip() for s in sec.split(",") if s.strip()])
            elif isinstance(sec, (list, tuple)):
                labels.extend([str(s) for s in sec])

        # Fallback: single 'label' column
        if not labels and "label" in row and pd.notna(row["label"]):
            labels.append(str(row["label"]))

        for lab in labels:
            if lab in self.label2id:
                target[self.label2id[lab]] = 1.0
        return target

    def _filename(self, row: pd.Series) -> str:
        for col in ("filename", "file", "audio_path", "path"):
            if col in row and pd.notna(row[col]):
                return str(row[col])
        raise KeyError("Row is missing a filename column")

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        fname = self._filename(row)
        stem = Path(fname).stem

        if self.mel_dir is not None:
            mel_path = self.mel_dir / f"{stem}.npy"
            if mel_path.exists():
                mel = np.load(mel_path).astype(np.float32)
            else:
                mel = self._compute_mel(fname)
        else:
            mel = self._compute_mel(fname)

        # Light training-time SpecAugment-style masking
        if self.train:
            mel = self._spec_augment(mel)

        x = torch.from_numpy(mel).unsqueeze(0)  # (1, M, T)
        y = torch.from_numpy(self._multi_hot(row))
        return x, y

    def _compute_mel(self, fname: str) -> np.ndarray:
        assert self.audio_dir is not None
        path = self.audio_dir / fname
        if not path.exists():
            # Some CSVs store only basename; try recursive match
            matches = list(self.audio_dir.rglob(Path(fname).name))
            if not matches:
                raise FileNotFoundError(path)
            path = matches[0]
        y = load_audio(path, sr=self.cfg["SR"], duration=self.cfg["DURATION"])
        return waveform_to_mel(
            y,
            sr=self.cfg["SR"],
            n_fft=self.cfg["N_FFT"],
            hop_length=self.cfg["HOP_LENGTH"],
            n_mels=self.cfg["N_MELS"],
            fmin=self.cfg["FMIN"],
            fmax=self.cfg["FMAX"],
            target_width=256,
        )

    @staticmethod
    def _spec_augment(mel: np.ndarray, freq_masks: int = 2, time_masks: int = 2) -> np.ndarray:
        mel = mel.copy()
        n_mels, n_time = mel.shape
        for _ in range(freq_masks):
            f = np.random.randint(0, max(1, n_mels // 8))
            f0 = np.random.randint(0, max(1, n_mels - f))
            mel[f0 : f0 + f, :] = 0.0
        for _ in range(time_masks):
            t = np.random.randint(0, max(1, n_time // 8))
            t0 = np.random.randint(0, max(1, n_time - t))
            mel[:, t0 : t0 + t] = 0.0
        return mel


def load_label_maps(root: str | Path) -> tuple[list[str], dict[str, int]]:
    root = Path(root)
    classes_path = root / "classes.json"
    label2id_path = root / "label2id.json"

    if label2id_path.exists():
        with open(label2id_path, encoding="utf-8") as f:
            label2id = json.load(f)
        # Ensure int values
        label2id = {k: int(v) for k, v in label2id.items()}
        classes = [None] * len(label2id)
        for k, v in label2id.items():
            classes[v] = k
        return classes, label2id

    with open(classes_path, encoding="utf-8") as f:
        classes = json.load(f)
    label2id = {c: i for i, c in enumerate(classes)}
    return classes, label2id
