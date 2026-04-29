"""双音（蜂鸣/标定音）标记录制边界，不依赖 ASR 的分段方案。"""

from pathlib import Path

__all__ = ["__version__"]


def _read_version() -> str:
    p = Path(__file__).with_name("VERSION")
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#") and s[0].isdigit():
            return s.split()[0]
    return "0.0.0"


__version__ = _read_version()
