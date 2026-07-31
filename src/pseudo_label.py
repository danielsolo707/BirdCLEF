"""Generate pseudo-labels for train_soundscapes (semi-supervised stage, v3).

Flow (aligned with the 4th/5th-place solutions):
  1. Load a teacher ensemble (v3 checkpoints) — average of sigmoid probs
  2. Slide 10 s windows over each soundscape (default stride 5 s)
  3. Keep 5 s chunks whose max class probability > ``--threshold``
  4. Write chunk power-mels into ``--mel-dir`` and a CSV of soft-label rows
     (``primary_label='__semi__'`` + a ``target`` column) consumable by
     ``src.train_v3 --semi-csv``

Example::

    python -m src.pseudo_label \\
        --config configs/config_v3.json \\
        --checkpoints runs/v3_exp001/model_fold0_best.pth,runs/v3_exp001/model_fold1_best.pth \\
        --soundscapes-dir /path/to/train_soundscapes \\
        --mel-dir data/mels_v3_semi \\
        --output-csv runs/pseudo/semi_chunks.csv \\
        --threshold 0.3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .audio import get_melspec_v3, load_audio_v3, mel_cache_name
from .model import load_checkpoint
from .utils import default_config_path, labels_dir, load_config, resolve_device, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pseudo-label train_soundscapes with a v3 teacher ensemble")
    p.add_argument("--config", type=str, default=str(default_config_path("v3")))
    p.add_argument("--checkpoints", type=str, required=True, help="Comma-separated teacher .pth paths")
    p.add_argument("--soundscapes-dir", type=str, required=True)
    p.add_argument("--mel-dir", type=str, required=True, help="Where to write chunk power-mels")
    p.add_argument("--output-csv", type=str, required=True, help="semi chunks CSV for --semi-csv")
    p.add_argument("--threshold", type=float, default=0.3, help="Min max-prob to keep a chunk")
    p.add_argument("--power", type=float, default=1.0, help="Power transform on probs (label denoising)")
    p.add_argument("--window-sec", type=float, default=10.0)
    p.add_argument("--stride-sec", type=float, default=5.0)
    p.add_argument("--chunk-sec", type=float, default=5.0, help="Pseudo chunk length written to disk")
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


@torch.no_grad()
def predict_soundscape(model, y: np.ndarray, cfg: dict, device, window_sec: float, stride_sec: float, batch_size: int) -> np.ndarray:
    """Return per-window soft label probs (n_windows, num_classes)."""
    sr = int(cfg.get("SR", 32000))
    win = int(sr * window_sec)
    step = int(sr * stride_sec)
    n = len(y)
    starts = list(range(0, max(1, n - win + 1), step)) or [0]

    probs = []
    for i in range(0, len(starts), batch_size):
        batch_starts = starts[i : i + batch_size]
        xs = []
        for s0 in batch_starts:
            seg = y[s0 : s0 + win]
            if len(seg) < win:
                seg = np.pad(seg, (0, win - len(seg)))
            xs.append(seg)
        mels = np.stack([waveform_to_mel_np(seg, cfg) for seg in xs])
        x = torch.from_numpy(mels).unsqueeze(1).to(device)
        logits, _ = model(x)
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(probs, axis=0)


def waveform_to_mel_np(seg: np.ndarray, cfg: dict) -> np.ndarray:
    """Power mel for a raw segment (target_time from cfg)."""
    from .audio import waveform_to_mel_v3

    return waveform_to_mel_v3(
        seg,
        sr=int(cfg.get("SR", 32000)),
        n_fft=int(cfg.get("N_FFT", 2048)),
        hop_length=int(cfg.get("HOP_LENGTH", 64)),
        n_mels=int(cfg.get("N_MELS", 256)),
        fmin=int(cfg.get("FMIN", 60)),
        fmax=int(cfg.get("FMAX", 16000)),
        target_time=int(cfg.get("TARGET_TIME", 512)),
        target_mels=int(cfg.get("TARGET_MELS", 256)),
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg.get("DEVICE", "cuda"))

    with open(labels_dir() / "classes.json", encoding="utf-8") as f:
        classes = json.load(f)
    num_classes = len(classes)

    ckpt_paths = [s.strip() for s in args.checkpoints.split(",") if s.strip()]
    models = [
        load_checkpoint(
            p,
            num_classes=int(cfg.get("NUM_CLASSES", num_classes)),
            backbone_name=str(cfg.get("BACKBONE", "efficientnet_b0")),
            device=device,
            model_version="v3",
            n_mels=int(cfg.get("TARGET_MELS", 256)),
        )
        for p in ckpt_paths
    ]
    for m in models:
        m.eval()
    print(f"[pseudo] loaded {len(models)} teacher(s) | classes: {num_classes} | device: {device}")

    mel_dir = Path(args.mel_dir)
    mel_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    sound_files = sorted(p for p in Path(args.soundscapes_dir).rglob("*.ogg"))
    sound_files += sorted(p for p in Path(args.soundscapes_dir).rglob("*.wav"))
    if not sound_files:
        raise SystemExit(f"No audio found under {args.soundscapes_dir}")
    print(f"[pseudo] {len(sound_files)} soundscapes to process")

    rows = []
    n_kept = 0
    sr = int(cfg.get("SR", 32000))

    for path in tqdm(sound_files, desc="Pseudo-labeling"):
        y = load_audio_v3(path, sr=sr, max_duration=60.0)  # soundscapes up to 60 s
        all_probs = []
        for m in models:
            all_probs.append(predict_soundscape(m, y, cfg, device, args.window_sec, args.stride_sec, args.batch_size))
        probs = np.mean(all_probs, axis=0)  # (n_windows, C)

        # aggregate to 5 s chunks: max over windows overlapping each chunk
        chunk_sec = args.chunk_sec
        n_chunks = max(1, int((len(y) / sr) // chunk_sec))
        for ci in range(n_chunks):
            c0, c1 = ci * chunk_sec, (ci + 1) * chunk_sec
            cover = [i for i, s0 in enumerate(range(0, max(1, len(y) - int(sr * args.window_sec) + 1), int(sr * args.stride_sec)))
                     if s0 / sr < c1 and (s0 + args.window_sec) / sr > c0]
            if not cover:
                continue
            chunk_prob = probs[cover].max(axis=0)  # max over overlapping windows
            if float(chunk_prob.max()) < args.threshold:
                continue

            # power transform (label denoising, 1st-place trick)
            soft = np.clip(chunk_prob ** args.power, 0.0, 1.0).astype(np.float32)

            # extract the chunk audio → power mel (5 s → target_time scaled)
            seg = y[int(c0 * sr) : int(c1 * sr)]
            if len(seg) < int(sr * chunk_sec):
                seg = np.pad(seg, (0, int(sr * chunk_sec) - len(seg)))
            mel = waveform_to_mel_np(seg, cfg)
            # scale target_time for a chunk_sec window (10 s → 512, 5 s → 256)
            t_full = int(cfg.get("TARGET_TIME", 512))
            t_chunk = max(64, int(t_full * chunk_sec / float(cfg.get("DURATION", 10.0))))
            if mel.shape[-1] > t_chunk:
                mel = mel[..., :t_chunk]

            fake_fname = f"train_soundscapes/{path.parent.name}/{ci:02d}.ogg"
            np.save(mel_dir / mel_cache_name(fake_fname), mel)
            rows.append(
                {
                    "filename": fake_fname,
                    "primary_label": "__semi__",
                    "secondary_labels": "",
                    "target": "[" + ",".join(f"{v:.6f}" for v in soft) + "]",
                }
            )
            n_kept += 1

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[pseudo] kept {n_kept} chunks ({len(sound_files)} soundscapes) → {out_csv}")
    print(f"[pseudo] chunk mels in {mel_dir} | next: python -m src.train_v3 --semi-csv {out_csv}")

    # metadata sidecar
    save_json(
        {
            "n_soundscapes": len(sound_files),
            "n_chunks": n_kept,
            "threshold": args.threshold,
            "power": args.power,
            "checkpoints": ckpt_paths,
            "cfg": cfg,
        },
        out_csv.with_suffix(".meta.json"),
    )


if __name__ == "__main__":
    main()
