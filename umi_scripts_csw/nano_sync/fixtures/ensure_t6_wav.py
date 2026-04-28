#!/usr/bin/env python3
"""生成本地 T6 用 30s 静音 WAV（不提交仓库）。运行：python3 nano_sync/fixtures/ensure_t6_wav.py"""

from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "t6_stub.wav"


def main() -> None:
    import wave

    sr, sec = 16000, 30
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(OUT), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * (sr * sec))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
