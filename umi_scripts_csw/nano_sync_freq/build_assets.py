#!/usr/bin/env python3
"""生成 `assets/start.wav` 与 `assets/stop.wav`（可覆盖）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nano_sync_freq.tone_gen import default_asset_paths, ensure_default_assets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--force", action="store_true", help="已存在时仍覆盖"
    )
    args = ap.parse_args()
    ensure_default_assets(overwrite=bool(args.force))
    for k, p in default_asset_paths().items():
        print(f"{k}: {p}", flush=True)


if __name__ == "__main__":
    main()
