"""Run inference on a single audio file or a directory of clips.

Examples::

    python -m src.inference --audio path/to/clip.ogg --top-k 5
    python -m src.inference --audio path/to/folder --top-k 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .audio import audio_to_tensor
from .model import load_checkpoint
from .utils import load_config, project_root, resolve_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BirdCLEF SED inference")
    p.add_argument("--audio", type=str, required=True, help="Audio file or directory")
    p.add_argument("--checkpoint", type=str, default=str(project_root() / "model.pth"))
    p.add_argument("--config", type=str, default=str(project_root() / "config.json"))
    p.add_argument("--classes", type=str, default=str(project_root() / "classes.json"))
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--threshold", type=float, default=0.3)
    return p.parse_args()


def collect_audio(path: Path) -> list[Path]:
    exts = {".ogg", ".wav", ".mp3", ".flac", ".m4a"}
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in exts)


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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.get("DEVICE", "cuda"))

    with open(args.classes, encoding="utf-8") as f:
        classes = json.load(f)

    model = load_checkpoint(
        args.checkpoint, num_classes=int(cfg["NUM_CLASSES"]), device=device
    )

    files = collect_audio(Path(args.audio))
    if not files:
        raise SystemExit(f"No audio files found at {args.audio}")

    results = [
        predict_one(model, f, cfg, classes, args.top_k, args.threshold) for f in files
    ]

    print(json.dumps(results if len(results) > 1 else results[0], indent=2))


if __name__ == "__main__":
    main()
