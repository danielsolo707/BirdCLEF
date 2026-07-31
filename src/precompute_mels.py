"""Precompute log-mel spectrograms for faster training (~10× speedup).

Matches the Kaggle notebook precompute cell:
- middle-crop audio, min-max log-mel in [0, 1], time width 256
- cache names: ``subdir_file.npy`` (e.g. ``1139490_CSA36385.npy``)

Example::

    python scripts/precompute_mels.py \\
        --audio-dir /path/to/train_audio \\
        --output-dir /path/to/mels \\
        --config v1/config.json

    # Optional: only files listed in train.csv
    python scripts/precompute_mels.py \\
        --audio-dir /path/to/train_audio \\
        --output-dir /path/to/mels \\
        --metadata /path/to/train.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .audio import get_melspec, mel_cache_name
from .utils import default_config_path, load_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--audio-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--config", type=str, default=str(default_config_path("v1")))
    p.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Optional train.csv; if set, only listed filenames are processed",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    audio_dir = Path(args.audio_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.metadata:
        df = pd.read_csv(args.metadata)
        col = "filename" if "filename" in df.columns else None
        if col is None:
            for c in ("file", "audio_path", "path"):
                if c in df.columns:
                    col = c
                    break
        if col is None:
            raise SystemExit("metadata CSV needs a filename column")
        rel_paths = [str(x).replace("\\", "/") for x in df[col].tolist()]
        jobs = []
        for rel in rel_paths:
            path = audio_dir / rel
            if not path.exists():
                matches = list(audio_dir.rglob(Path(rel).name))
                path = matches[0] if matches else path
            jobs.append((rel, path))
    else:
        exts = {".ogg", ".wav", ".mp3", ".flac", ".m4a"}
        files = [f for f in audio_dir.rglob("*") if f.suffix.lower() in exts]
        jobs = []
        for path in files:
            rel = path.relative_to(audio_dir).as_posix()
            jobs.append((rel, path))

    print(f"Found {len(jobs)} audio files")
    tw = int(cfg.get("TARGET_WIDTH", 256))
    saved = 0
    skipped = 0

    for rel, path in tqdm(jobs):
        out_name = mel_cache_name(rel)
        out_path = out_dir / out_name
        if out_path.exists():
            skipped += 1
            continue
        if not path.exists():
            continue
        try:
            mel = get_melspec(
                path,
                sr=int(cfg["SR"]),
                duration=float(cfg["DURATION"]),
                n_fft=int(cfg["N_FFT"]),
                hop_length=int(cfg["HOP_LENGTH"]),
                n_mels=int(cfg["N_MELS"]),
                fmin=int(cfg["FMIN"]),
                fmax=int(cfg["FMAX"]),
                target_width=tw,
            )
            np.save(out_path, mel.astype(np.float32))
            saved += 1
        except Exception:
            # Skip corrupt files (same behavior as Kaggle notebook)
            continue

    print(f"Saved {saved} mels to {out_dir} (skipped existing: {skipped})")


if __name__ == "__main__":
    main()
