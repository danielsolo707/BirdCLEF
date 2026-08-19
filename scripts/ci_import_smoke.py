"""Lightweight CI check for BirdCLEF imports.

This script intentionally avoids dataset access, checkpoint loading, training, and GPU work.
It verifies that the core modules required by the versioned pipeline import correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    from src import audio, dataset, losses, metrics, model, utils  # noqa: F401
    from v1 import train as v1_train  # noqa: F401
    from v2 import train as v2_train  # noqa: F401
    from v3 import train as v3_train  # noqa: F401

    print("BirdCLEF core imports OK")


if __name__ == "__main__":
    main()
