"""Mel-spectrogram extraction utilities."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import torch


def load_audio(path: str | Path, sr: int = 32000, duration: float = 5.0) -> np.ndarray:
    """Load mono audio, pad/crop to a fixed duration."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    target = int(sr * duration)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]
    return y.astype(np.float32)


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
    """Convert waveform → log-mel spectrogram (n_mels, target_width)."""
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # Normalize to roughly [-1, 1]
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)

    # Resize time axis to a fixed width (matches training: 128x256)
    if log_mel.shape[1] != target_width:
        log_mel = librosa.util.fix_like(
            log_mel,
            size=(n_mels, target_width),
            mode="constant",
            constant_values=0.0,
        )
    return log_mel.astype(np.float32)


def audio_to_tensor(path: str | Path, cfg: dict) -> torch.Tensor:
    """Full path → model-ready tensor of shape (1, 1, n_mels, time)."""
    y = load_audio(path, sr=cfg["SR"], duration=cfg["DURATION"])
    mel = waveform_to_mel(
        y,
        sr=cfg["SR"],
        n_fft=cfg["N_FFT"],
        hop_length=cfg["HOP_LENGTH"],
        n_mels=cfg["N_MELS"],
        fmin=cfg["FMIN"],
        fmax=cfg["FMAX"],
        target_width=256,
    )
    return torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)  # (1, 1, M, T)
