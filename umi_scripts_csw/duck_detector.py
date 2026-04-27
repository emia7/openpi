#!/usr/bin/env python3
"""
兼容旧命令名：尖叫/短促声检测已改为云 API（VectorEngine + gpt-4o 系）。

请直接用（推荐）:
  python squeak_audio_gpt4o.py <video.mp4>

本入口等价于上面脚本，并透传所有参数（如 --model --timeout --no-modalities）。
需 `umi_scripts_csw/.env` 中配置 OPENAI_API_KEY 与 OPENAI_BASE_URL。
"""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    target = here / "squeak_audio_gpt4o.py"
    if not target.is_file():
        print(f"未找到 {target}", file=sys.stderr)
        raise SystemExit(1)
    r = subprocess.run([sys.executable, str(target), *sys.argv[1:]])
    raise SystemExit(r.returncode)
