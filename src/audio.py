"""Mel-spectrogram extraction utilities.

Matches the Kaggle training notebook (`get_melspec` + 256-frame crop/pad):
- mono load at 32 kHz, fixed 5 s window
- middle crop when longer than duration
- log-mel, min-max normalized to [0, 1]
- time axis center-cropped / padded to ``target_width`` (256)
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import torch


def load_audio(path: str | Path, sr: int = 32000, duration: float = 5.0) -> np.ndarray:
    """Load mono audio, pad or middle-crop to a fixed duration."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    target = int(sr * duration)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)), mode="constant")
    else:
        # Middle segment (most stable) — matches Kaggle training
        start = (len(y) - target) // 2
        y = y[start : start + target]
    return y.astype(np.float32)


def fix_time_width(mel: np.ndarray, target_width: int = 256) -> np.ndarray:
    """Center-crop or zero-pad the time axis to ``target_width``."""
    t = mel.shape[1]
    if t > target_width:
        start = (t - target_width) // 2
        return mel[:, start : start + target_width]
    if t < target_width:
        return np.pad(mel, ((0, 0), (0, target_width - t)), mode="constant")
    return mel


def waveform_to_mel(
    y: np.ndarray,
    sr: int = 32000,
    n_fft: int = 1024,
    hop_length: int = 64,
    n_mels: int = 128,
    fmin: int = 50,
    fmax: int = 16000,
    target_width: int = 256,
) -> np.ndarray:
    """Convert waveform → log-mel spectrogram (n_mels, target_width) in [0, 1]."""
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    # Log scale + min-max normalize to [0, 1] (Kaggle notebook)
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
    mel = fix_time_width(mel.astype(np.float32), target_width=target_width)
    return mel.astype(np.float32)


def get_melspec(
    path: str | Path,
    sr: int = 32000,
    duration: float = 5.0,
    n_fft: int = 1024,
    hop_length: int = 64,
    n_mels: int = 128,
    fmin: int = 50,
    fmax: int = 16000,
    target_width: int = 256,
) -> np.ndarray:
    """End-to-end path → (n_mels, target_width) mel, matching Kaggle ``get_melspec``."""
    y = load_audio(path, sr=sr, duration=duration)
    return waveform_to_mel(
        y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        target_width=target_width,
    )


def audio_to_tensor(path: str | Path, cfg: dict) -> torch.Tensor:
    """Full path → model-ready tensor of shape (1, 1, n_mels, time)."""
    mel = get_melspec(
        path,
        sr=int(cfg["SR"]),
        duration=float(cfg["DURATION"]),
        n_fft=int(cfg["N_FFT"]),
        hop_length=int(cfg["HOP_LENGTH"]),
        n_mels=int(cfg["N_MELS"]),
        fmin=int(cfg["FMIN"]),
        fmax=int(cfg["FMAX"]),
        target_width=int(cfg.get("TARGET_WIDTH", 256)),
    )
    return torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)  # (1, 1, M, T)


def mel_cache_name(filename: str) -> str:
    """Kaggle-compatible precomputed mel filename.

    ``1139490/CSA36385.ogg`` → ``1139490_CSA36385.npy``
    """
    name = filename.replace("\\", "/").replace("/", "_")
    for ext in (".ogg", ".wav", ".mp3", ".flac", ".m4a", ".OGG", ".WAV"):
        if name.endswith(ext):
            name = name[: -len(ext)] + ".npy"
            break
    else:
        # already stem-like or unknown extension
        if not name.endswith(".npy"):
            name = f"{Path(name).stem}.npy"
    return name
