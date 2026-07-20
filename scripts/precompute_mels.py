"""Precompute log-mel spectrograms for faster training (~10× speedup).

Example::

    python scripts/precompute_mels.py \\
        --audio-dir /path/to/train_audio \\
        --output-dir /path/to/mels \\
        --config config.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Allow running as a script without installing the package
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.audio import load_audio, waveform_to_mel  # noqa: E402
from src.utils import load_config  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--audio-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--config", type=str, default=str(ROOT / "config.json"))
    args = p.parse_args()

    cfg = load_config(args.config)
    audio_dir = Path(args.audio_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".ogg", ".wav", ".mp3", ".flac", ".m4a"}
    files = [f for f in audio_dir.rglob("*") if f.suffix.lower() in exts]
    print(f"Found {len(files)} audio files")

    for path in tqdm(files):
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
        np.save(out_dir / f"{path.stem}.npy", mel)

    print(f"Saved mels to {out_dir}")


if __name__ == "__main__":
    main()
