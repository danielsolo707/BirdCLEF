"""Precompute v3 power-mel spectrograms (10 s, 256 x 512) as .npy cache.

Naming follows the repo convention (mel_cache_name): ``1139490/CSA36385.ogg``
→ ``1139490_CSA36385.npy``. Stored as RAW POWER mels; ``power_to_db`` is
applied at load time in the v3 dataset (4th-place pipeline).

Example::

    python scripts/precompute_mels_v3.py \\
        --audio-dir path/to/train_audio \\
        --output-dir data/mels_v3 \\
        --config configs/config_v3.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.audio import get_melspec_v3, mel_cache_name  # noqa: E402
from src.utils import default_config_path, load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute v3 power-mel cache")
    p.add_argument("--audio-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--config", type=str, default=str(default_config_path("v3")))
    p.add_argument("--metadata", type=str, default=None, help="Optional train.csv to restrict files")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    audio_dir = Path(args.audio_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.metadata:
        df = pd.read_csv(args.metadata)
        files = [audio_dir / f for f in df["filename"] if (audio_dir / f).exists()]
        files += [audio_dir / f for f in df["filename"] if not (audio_dir / f).exists()]
        files = sorted({p for p in (audio_dir / f for f in df["filename"]) if p.exists()})
    else:
        exts = {".ogg", ".wav", ".mp3", ".flac", ".m4a"}
        files = sorted(p for p in audio_dir.rglob("*") if p.suffix.lower() in exts)

    print(f"[precompute_v3] {len(files)} files → {out_dir}")
    t0 = time.time()
    n = 0
    for p in tqdm(files, desc="mel_v3"):
        try:
            mel = get_melspec_v3(
                p,
                sr=int(cfg["SR"]),
                max_duration=float(cfg["DURATION"]),
                n_fft=int(cfg["N_FFT"]),
                hop_length=int(cfg["HOP_LENGTH"]),
                n_mels=int(cfg["N_MELS"]),
                fmin=int(cfg["FMIN"]),
                fmax=int(cfg["FMAX"]),
                target_time=int(cfg["TARGET_TIME"]),
                target_mels=int(cfg["TARGET_MELS"]),
            )
            np.save(out_dir / mel_cache_name(p.as_posix()), mel)
            n += 1
        except Exception as e:  # noqa: BLE001
            print(f"  skip {p.name}: {e}")
    print(f"[precompute_v3] done: {n} mels in {time.time() - t0:.1f}s → {out_dir}")


if __name__ == "__main__":
    main()
