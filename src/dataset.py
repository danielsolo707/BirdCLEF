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

import librosa
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


class SpecAugment:
    """Simple SpecAugment for mel spectrograms (junior-style).

    Input tensor shape: (1, n_mels, time)
    We zero-out a few random frequency bands and time bands.
    This is a common audio trick and usually helps more than only ColorJitter.
    """

    def __init__(
        self,
        freq_masks: int = 2,
        time_masks: int = 2,
        freq_width: int = 16,
        time_width: int = 32,
    ):
        self.freq_masks = freq_masks
        self.time_masks = time_masks
        self.freq_width = freq_width
        self.time_width = time_width

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, F, T) or (C, F, T)
        x = x.clone()
        f_dim = x.shape[-2]
        t_dim = x.shape[-1]

        for _ in range(self.freq_masks):
            f = int(np.random.randint(0, self.freq_width + 1))
            if f == 0 or f_dim <= 1:
                continue
            f0 = int(np.random.randint(0, max(1, f_dim - f + 1)))
            x[..., f0 : f0 + f, :] = 0.0

        for _ in range(self.time_masks):
            t = int(np.random.randint(0, self.time_width + 1))
            if t == 0 or t_dim <= 1:
                continue
            t0 = int(np.random.randint(0, max(1, t_dim - t + 1)))
            x[..., :, t0 : t0 + t] = 0.0

        return x


class ComposeTransforms:
    """Tiny compose so we don't need extra libraries."""

    def __init__(self, transforms: list):
        self.transforms = [t for t in transforms if t is not None]

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


def build_train_transform(cfg: dict | None = None):
    """Build train-time mel transforms from config (v1 or v2).

    v1: flip + color jitter only
    v2: same + SpecAugment when USE_SPEC_AUGMENT is true
    """
    cfg = cfg or {}
    parts = []

    # Keep the original light augs unless someone turns them off
    if cfg.get("KEEP_V1_LIGHT_AUGS", True):
        parts.append(default_train_transform())

    if cfg.get("USE_SPEC_AUGMENT", False):
        parts.append(
            SpecAugment(
                freq_masks=int(cfg.get("SPEC_FREQ_MASKS", 2)),
                time_masks=int(cfg.get("SPEC_TIME_MASKS", 2)),
                freq_width=int(cfg.get("SPEC_FREQ_WIDTH", 16)),
                time_width=int(cfg.get("SPEC_TIME_WIDTH", 32)),
            )
        )

    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return ComposeTransforms(parts)


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
            # Prefer cfg-driven transforms (v2 SpecAugment etc.), else v1 defaults
            self.transform = build_train_transform(cfg)
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


def load_label_maps(root: str | Path | None = None) -> tuple[list[str], dict[str, int]]:
    """Load species list + label→id map.

    Looks in this order under ``root`` (default = project root):
    1. ``labels/label2id.json`` + ``labels/classes.json``  (organized layout)
    2. ``label2id.json`` / ``classes.json`` at root          (legacy / Kaggle pack)
    """
    if root is None:
        from .utils import project_root

        root = project_root()
    root = Path(root)

    candidates = [
        (root / "labels" / "label2id.json", root / "labels" / "classes.json"),
        (root / "label2id.json", root / "classes.json"),
    ]
    label2id_path = classes_path = None
    for lid, cls in candidates:
        if lid.exists() or cls.exists():
            label2id_path, classes_path = lid, cls
            break
    if label2id_path is None:
        raise FileNotFoundError(
            f"Could not find label maps under {root} "
            "(expected labels/label2id.json or label2id.json)"
        )

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


# ============================= v3 dataset ====================================
# Based on the 4th-place solution (Dylan Liu, BirdCLEF+ 2025):
#   - 10 s power-mel (precomputed .npy), power_to_db at load
#   - random 10 s window for train, first window for val
#   - rare classes upsampled to a minimum count (default 100)
#   - soft-label rows (pseudo-labeled soundscapes) with a prebuilt 'target'
#     column are mixed into training directly
# =============================================================================

def _to_db(spec: np.ndarray) -> np.ndarray:
    """power_to_db(ref=1.0) — matches 4th-place data pipeline."""
    return librosa.power_to_db(spec, ref=1.0)


class BirdCLEFDatasetV3(Dataset):
    """v3 multi-label dataset over precomputed power-mel .npy files (or raw audio).

    Expected metadata CSV columns (BirdCLEF-style):
      - ``filename`` (e.g. ``1139490/CSA36385.ogg``)
      - ``primary_label``, optional ``secondary_labels``
      - OR a prebuilt soft/hard ``target`` column (list/ndarray of length C)
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
        self.transform = transform  # v3 does spec-aug inside __getitem__; transform is optional

        self.min_samples = int(cfg.get("MIN_SAMPLES_PER_CLASS", 100))
        self.resample_primary_only = bool(cfg.get("RESAMPLE_PRIMARY_ONLY", True))
        self.aug_prob = float(cfg.get("AUG_PROB", 0.5))
        self.spec_len = int(cfg.get("SPEC_LEN", 512))  # time frames after resize

        if self.audio_dir is None and self.mel_dir is None:
            raise ValueError("Provide at least one of audio_dir or mel_dir")

        self._has_target_col = "target" in self.df.columns
        self._build_samplename_maps()

    # ---- helpers ----------------------------------------------------------
    def _filename(self, row: pd.Series) -> str:
        for col in ("filename", "file", "audio_path", "path"):
            if col in row and pd.notna(row[col]):
                return str(row[col]).replace("\\", "/")
        raise KeyError("Row is missing a filename column")

    def _build_samplename_maps(self):
        from .audio import mel_cache_name

        self.df["_samplename"] = [
            Path(mel_cache_name(self._filename(r))).stem for _, r in self.df.iterrows()
        ]
        self.samplename_id_map = {name: i for i, name in enumerate(self.df["_samplename"].values)}

        # map samplename -> list of available mel cache files
        self.df_samplename_dict: dict[str, list] = {sn: [] for sn in self.df["_samplename"].values}
        if self.mel_dir is not None:
            for p in sorted(self.mel_dir.glob("*.npy")):
                pure = p.stem.split("__")[0]
                if pure in self.samplename_id_map:
                    self.df_samplename_dict[pure].append(str(p))

        self.used_df_samplenames: list[str] = []
        if self.train:
            self._init_train_dataset()
        else:
            self.used_df_samplenames = [
                sn for sn in self.df["_samplename"].values if len(self.df_samplename_dict[sn]) > 0
            ]

    def _init_train_dataset(self):
        """Rare-class upsampling: duplicate classes with < min_samples to min_samples."""
        used = []
        used_df = self.df  # (4th place also filters 'removed'; we keep everything)
        for label in self.label2id:
            rows = used_df[used_df["primary_label"] == label]
            sns = [sn for sn in rows["_samplename"].values if len(self.df_samplename_dict.get(sn, [])) > 0]
            if not sns:
                continue
            if len(sns) < self.min_samples:
                rng = np.random.default_rng(0)
                sns = list(rng.choice(sns, size=self.min_samples, replace=True))
            used += sns
        # soft-label rows (primary_label == '__semi__') are appended as-is
        semi = used_df[used_df["primary_label"] == "__semi__"]["_samplename"].tolist()
        semi = [sn for sn in semi if len(self.df_samplename_dict.get(sn, [])) > 0]
        used += semi
        rng = np.random.default_rng(0)
        rng.shuffle(used)
        self.used_df_samplenames = used

    def _resolve_mel_path(self, samplename: str) -> str | None:
        cands = self.df_samplename_dict.get(samplename, [])
        return cands[0] if cands else None

    # ---- item -------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.used_df_samplenames)

    def __getitem__(self, idx: int):
        samplename = self.used_df_samplenames[idx]
        row = self.df.iloc[self.samplename_id_map[self._pure(samplename)]]

        mel_path = self._resolve_mel_path(samplename)
        if mel_path is not None:
            spec = np.load(mel_path).astype(np.float32)
        else:
            spec = self._compute_mel_v3(row)

        # db conversion (4th-place style: power mel -> db at load)
        spec = _to_db(spec)

        # time-window selection
        spec = self._select_window(spec)

        x = torch.from_numpy(spec).unsqueeze(0)  # (1, M, T)
        if self.train and np.random.rand() < self.aug_prob:
            x = self._spec_augment(x)

        if str(row.get("primary_label", "")) == "__semi__":
            # pseudo-labeled chunk: use its prebuilt soft target vector
            t = row["target"]
            if isinstance(t, torch.Tensor):
                y = t.float()
            else:
                y = torch.tensor(np.asarray(t, dtype=np.float32), dtype=torch.float32)
        else:
            y = torch.from_numpy(build_target(row, self.label2id, self.num_classes))

        return x, y

    def _pure(self, samplename: str) -> str:
        """samplename may include a __suffix for pseudo chunks; strip it."""
        return samplename.split("__")[0]

    def _select_window(self, spec: np.ndarray) -> np.ndarray:
        """(M, T) → (M, spec_len): random window (train) or first window (val)."""
        t = spec.shape[-1]
        if t == self.spec_len:
            return spec
        if t > self.spec_len:
            start = np.random.randint(0, t - self.spec_len) if self.train else 0
            return spec[..., start : start + self.spec_len]
        # pad
        pad = np.zeros((spec.shape[0], self.spec_len), dtype=spec.dtype)
        if self.train and self.spec_len - t > 0:
            pad1 = np.random.randint(0, self.spec_len - t + 1)
        else:
            pad1 = 0
        pad[..., pad1 : pad1 + t] = spec
        return pad

    def _spec_augment(self, x: torch.Tensor) -> torch.Tensor:
        """Time/frequency masking + gain (4th-place style)."""
        spec = x.clone()
        # time masking
        if np.random.rand() < 0.5:
            for _ in range(int(np.random.randint(1, 4))):
                width = int(np.random.randint(5, 21))
                start = int(np.random.randint(0, max(1, spec.shape[2] - width + 1)))
                spec[0, :, start : start + width] = 0.0
        # frequency masking
        if np.random.rand() < 0.5:
            for _ in range(int(np.random.randint(1, 4))):
                height = int(np.random.randint(5, 21))
                start = int(np.random.randint(0, max(1, spec.shape[1] - height + 1)))
                spec[0, start : start + height, :] = 0.0
        # gain
        if np.random.rand() < 0.5:
            spec = spec * float(np.random.uniform(0.8, 1.2))
        return spec

    def _compute_mel_v3(self, row: pd.Series) -> np.ndarray:
        assert self.audio_dir is not None
        from .audio import get_melspec_v3

        fname = self._filename(row)
        path = self.audio_dir / fname
        if not path.exists():
            matches = list(self.audio_dir.rglob(Path(fname).name))
            if not matches:
                raise FileNotFoundError(path)
            path = matches[0]
        return get_melspec_v3(
            path,
            sr=int(self.cfg.get("SR", 32000)),
            max_duration=float(self.cfg.get("DURATION", 10.0)),
            n_fft=int(self.cfg.get("N_FFT", 2048)),
            hop_length=int(self.cfg.get("HOP_LENGTH", 64)),
            n_mels=int(self.cfg.get("N_MELS", 256)),
            fmin=int(self.cfg.get("FMIN", 60)),
            fmax=int(self.cfg.get("FMAX", 16000)),
            target_time=int(self.cfg.get("TARGET_TIME", 512)),
            target_mels=int(self.cfg.get("TARGET_MELS", 256)),
        )
