"""Run inference on a single audio file or a directory of clips.

Examples::

    python -m src.inference --audio path/to/clip.ogg --top-k 5
    python -m src.inference --audio path/to/folder --top-k 10
    python -m src.inference --audio path/to/soundscape.ogg --model-version v3 --overlap --smooth --postprocess
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .audio import audio_to_tensor, load_audio_v3
from .model import load_checkpoint
from .utils import (
    default_checkpoint_path,
    default_config_path,
    labels_dir,
    load_config,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BirdCLEF SED inference")
    p.add_argument("--audio", type=str, required=True, help="Audio file or directory")
    p.add_argument("--checkpoint", type=str, default=str(default_checkpoint_path()))
    p.add_argument("--config", type=str, default=str(default_config_path("v1")))
    p.add_argument("--classes", type=str, default=str(labels_dir() / "classes.json"))
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--model-version", type=str, default="v1", choices=["v1", "v3"])
    # v3 / soundscape options
    p.add_argument("--overlap", action="store_true", help="Sliding-window TTA over longer audio")
    p.add_argument("--window-sec", type=float, default=10.0)
    p.add_argument("--stride-sec", type=float, default=2.5)
    p.add_argument("--smooth", action="store_true", help="Smooth frame probs [0.1,0.8,0.1]")
    p.add_argument("--postprocess", action="store_true", help="2nd-place post: x top prob per class")
    return p.parse_args()


def collect_audio(path: Path) -> list[Path]:
    exts = {".ogg", ".wav", ".mp3", ".flac", ".m4a"}
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in exts)


def _smooth1d(probs: np.ndarray, kernel: tuple[float, ...] = (0.1, 0.8, 0.1)) -> np.ndarray:
    """Smooth along the chunk axis (5th-place style). probs: (n_chunks, C)."""
    k = np.asarray(kernel, dtype=np.float32)
    k = k / k.sum()
    out = np.zeros_like(probs)
    n = len(probs)
    for i in range(n):
        lo = max(0, i - len(k) // 2)
        hi = min(n, i + len(k) // 2 + 1)
        w = k[max(0, len(k) // 2 - (i - lo)) : len(k) // 2 + 1 + (hi - 1 - i)]
        out[i] = (probs[lo:hi] * w[:, None]).sum(axis=0) / w.sum()
    return out


def _postprocess_chunks(probs: np.ndarray) -> np.ndarray:
    """2nd-place post-processing: multiply chunk probs by top prob per class."""
    top = probs.max(axis=0, keepdims=True)
    return probs * top


@torch.no_grad()
def predict_one(model, path: Path, cfg: dict, classes: list[str], top_k: int, thr: float):
    x = audio_to_tensor(path, cfg)
    device = next(model.parameters()).device
    x = x.to(device)
    logits, _ = model(x)
    probs = torch.sigmoid(logits)[0].cpu().numpy()
    order = probs.argsort()[::-1]
    top = [
        {"species": classes[i], "probability": float(probs[i])}
        for i in order[:top_k]
    ]
    above = [
        {"species": classes[i], "probability": float(probs[i])}
        for i in order
        if probs[i] >= thr
    ]
    return {"file": str(path), "top_k": top, "above_threshold": above}


@torch.no_grad()
def predict_overlap(model, path: Path, cfg: dict, classes: list[str], top_k: int, thr: float,
                    window_sec: float, stride_sec: float, smooth: bool, postprocess: bool) -> dict:
    """Sliding-window TTA inference for long recordings (soundscapes)."""
    device = next(model.parameters()).device
    sr = int(cfg.get("SR", 32000))
    y = load_audio_v3(path, sr=sr, max_duration=60.0)
    win = int(sr * window_sec)
    step = int(sr * stride_sec)
    starts = list(range(0, max(1, len(y) - win + 1), step)) or [0]

    chunk_probs = []
    for s0 in starts:
        seg = y[s0 : s0 + win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        from .audio import waveform_to_mel_v3

        mel = waveform_to_mel_v3(
            seg,
            sr=sr,
            n_fft=int(cfg.get("N_FFT", 2048)),
            hop_length=int(cfg.get("HOP_LENGTH", 64)),
            n_mels=int(cfg.get("N_MELS", 256)),
            fmin=int(cfg.get("FMIN", 60)),
            fmax=int(cfg.get("FMAX", 16000)),
            target_time=int(cfg.get("TARGET_TIME", 512)),
            target_mels=int(cfg.get("TARGET_MELS", 256)),
        )
        x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(device)
        logits, _ = model(x)
        chunk_probs.append(torch.sigmoid(logits)[0].cpu().numpy())

    probs = np.stack(chunk_probs, axis=0)  # (n_chunks, C)
    if smooth:
        probs = _smooth1d(probs)
    if postprocess:
        probs = _postprocess_chunks(probs)
    clip = probs.mean(axis=0)

    order = clip.argsort()[::-1]
    top = [{"species": classes[i], "probability": float(clip[i])} for i in order[:top_k]]
    above = [{"species": classes[i], "probability": float(clip[i])} for i in order if clip[i] >= thr]
    return {
        "file": str(path),
        "n_windows": len(starts),
        "top_k": top,
        "above_threshold": above,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.get("DEVICE", "cuda"))

    with open(args.classes, encoding="utf-8") as f:
        classes = json.load(f)

    model = load_checkpoint(
        args.checkpoint,
        num_classes=int(cfg["NUM_CLASSES"]),
        backbone_name=str(cfg.get("BACKBONE", "efficientnet_b0")),
        device=device,
        model_version=args.model_version,
        n_mels=int(cfg.get("TARGET_MELS", 256)),
    )

    files = collect_audio(Path(args.audio))
    if not files:
        raise SystemExit(f"No audio files found at {args.audio}")

    results = []
    for f in files:
        if args.overlap:
            results.append(
                predict_overlap(
                    model, f, cfg, classes, args.top_k, args.threshold,
                    args.window_sec, args.stride_sec, args.smooth, args.postprocess,
                )
            )
        else:
            results.append(predict_one(model, f, cfg, classes, args.top_k, args.threshold))

    print(json.dumps(results if len(results) > 1 else results[0], indent=2))


if __name__ == "__main__":
    main()
