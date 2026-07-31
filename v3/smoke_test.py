"""Fast local sanity test for the NEW v3 pipeline (4th-place-style).

Creates tiny synthetic audio, precomputes a few v3 power-mels, trains a
micro model for a couple of steps on GPU (or CPU), then runs inference.

Run::

    python scripts/smoke_test_v3.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio import get_melspec_v3  # noqa: E402
from src.dataset import BirdCLEFDatasetV3  # noqa: E402
from src.losses import SoftAUCLoss, FocalBCELoss  # noqa: E402
from src.model import BirdCLEFModelV3, load_checkpoint  # noqa: E402
from v3.train import evaluate_full  # noqa: E402
from src.utils import save_json, set_seed  # noqa: E402

SR = 32000


def make_tone(duration: float, freq: float, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * freq * t)
    y += 0.05 * np.random.randn(len(y))
    return y.astype(np.float32)


def main() -> int:
    set_seed(0)
    tmp = Path(tempfile.mkdtemp(prefix="birdclef_v3_smoke_"))
    try:
        audio_dir = tmp / "audio"
        mel_dir = tmp / "mels"
        audio_dir.mkdir()
        mel_dir.mkdir()

        # 12 synthetic clips across 3 species (10 s each)
        species = ["sp_aloysii", "sp_barbata", "sp_coerul"]
        rows = []
        for i in range(12):
            sp = species[i % 3]
            fname = f"smoke/{i:03d}.wav"
            (audio_dir / "smoke").mkdir(exist_ok=True)
            path = audio_dir / fname
            import soundfile as sf

            sf.write(path, make_tone(10.0, 800 + 400 * (i % 3)), SR)
            rows.append({"filename": fname, "primary_label": sp, "secondary_labels": "[]"})
        df = pd.DataFrame(rows)
        df.to_csv(tmp / "train.csv", index=False)

        label2id = {s: i for i, s in enumerate(species)}
        classes = species
        save_json(classes, tmp / "classes.json")
        save_json(label2id, tmp / "label2id.json")

        # precompute v3 power mels
        cfg = {
            "SR": SR, "DURATION": 10.0, "N_FFT": 2048, "HOP_LENGTH": 64,
            "N_MELS": 256, "FMIN": 60, "FMAX": 16000,
            "TARGET_MELS": 256, "TARGET_TIME": 512, "SPEC_LEN": 512,
            "MIN_SAMPLES_PER_CLASS": 6, "AUG_PROB": 0.5, "DEVICE": "cuda",
        }
        for _, r in df.iterrows():
            mel = get_melspec_v3(
                audio_dir / r["filename"],
                sr=cfg["SR"], max_duration=cfg["DURATION"], n_fft=cfg["N_FFT"],
                hop_length=cfg["HOP_LENGTH"], n_mels=cfg["N_MELS"],
                fmin=cfg["FMIN"], fmax=cfg["FMAX"],
                target_time=cfg["TARGET_TIME"], target_mels=cfg["TARGET_MELS"],
            )
            from src.audio import mel_cache_name

            np.save(mel_dir / mel_cache_name(r["filename"]), mel)
        print(f"[smoke] precomputed {len(list(mel_dir.glob('*.npy')))} mels")

        # dataset
        ds = BirdCLEFDatasetV3(df, label2id, cfg, mel_dir=mel_dir, train=True)
        x, y = ds[0]
        print(f"[smoke] dataset item: x={tuple(x.shape)} y={tuple(y.shape)} (train len={len(ds)})")
        assert x.shape == (1, 256, 512), f"unexpected x shape {x.shape}"
        assert y.shape == (3,), f"unexpected y shape {y.shape}"

        # model + losses forward
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BirdCLEFModelV3(
            num_classes=3, backbone_name="efficientnet_b0", pretrained=False,
            n_mels=256, dropout=0.2, drop_path_rate=0.0,
        ).to(device)
        xb = torch.randn(2, 1, 256, 512, device=device)
        clip, frame = model(xb)
        print(f"[smoke] model: clip={tuple(clip.shape)} frame={tuple(frame.shape)}")
        assert clip.shape == (2, 3) and frame.shape == (2, 3, frame.shape[2])

        yb = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.float32, device=device)
        l_auc = SoftAUCLoss()(clip, yb)
        l_focal = FocalBCELoss(gamma=2.0)(clip, yb)
        print(f"[smoke] SoftAUC loss={l_auc.item():.4f} FocalBCE loss={l_focal.item():.4f}")
        assert torch.isfinite(l_auc) and torch.isfinite(l_focal)

        # one train step
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        opt.zero_grad()
        out, _ = model(xb)
        loss = SoftAUCLoss()(out, yb)
        loss.backward()
        opt.step()
        print(f"[smoke] one train step OK (loss={loss.item():.4f})")

        # evaluate_full on a mini loader
        from torch.utils.data import DataLoader

        val_ds = BirdCLEFDatasetV3(df, label2id, cfg, mel_dir=mel_dir, train=False)
        loader = DataLoader(val_ds, batch_size=4, shuffle=False)
        metrics = evaluate_full(model, loader, device)
        print(f"[smoke] evaluate_full: auc={metrics['macro_roc_auc']:.4f} f1={metrics['macro_f1']:.4f}")

        # save + reload checkpoint (v3 path)
        model.eval()
        with torch.no_grad():
            out, _ = model(xb)
        torch.save(model.state_dict(), tmp / "ckpt.pth")
        model2 = load_checkpoint(
            tmp / "ckpt.pth", num_classes=3, backbone_name="efficientnet_b0",
            device=device, model_version="v3", n_mels=256,
        )
        with torch.no_grad():
            out2, _ = model2(xb)
        assert torch.allclose(out, out2, atol=1e-5), "checkpoint reload mismatch"
        print("[smoke] checkpoint save/reload OK")

        # inference module (v3 + overlap)
        import json
        import subprocess

        smoke_cfg = {
            "SR": SR, "DURATION": 10.0, "N_FFT": 2048, "HOP_LENGTH": 64,
            "N_MELS": 256, "FMIN": 60, "FMAX": 16000,
            "TARGET_MELS": 256, "TARGET_TIME": 512, "NUM_CLASSES": 3,
            "BACKBONE": "efficientnet_b0", "DEVICE": "cuda",
        }
        cfg_path = tmp / "config_smoke.json"
        cfg_path.write_text(json.dumps(smoke_cfg))

        r = subprocess.run(
            [
                sys.executable, "-m", "src.inference",
                "--audio", str(audio_dir / "smoke" / "000.wav"),
                "--checkpoint", str(tmp / "ckpt.pth"),
                "--config", str(cfg_path),
                "--classes", str(tmp / "classes.json"),
                "--top-k", "3", "--model-version", "v3",
            ],
            capture_output=True, text=True,
        )
        print(f"[smoke] inference exit={r.returncode}")
        if r.returncode != 0:
            print(r.stderr[-2000:])
            return 1
        parsed = json.loads(r.stdout)
        assert "top_k" in parsed and len(parsed["top_k"]) == 3
        print(f"[smoke] inference top_k: {[t['species'] for t in parsed['top_k']]}")

        print("\n=== v3 SMOKE TEST PASSED ===")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
