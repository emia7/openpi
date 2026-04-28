#!/usr/bin/env python3
"""
采集中在本**终端**内按 **a=开始段**、**c=结束段** 播标定 chirp（`assets/start.wav` / `stop.wav`），
与 `record_marker_tts_listener` 的键位习惯一致，但**不依赖 pynput**、也**不监听全局键**。

使用方式：在终端里运行本脚本，**让该终端窗口获得焦点**，再按 a/c（macOS/Linux 通常单键即响；
Windows 为每行一个字母后回车）。脚踏若通过 USB 模拟键盘，需把焦点切到**本终端**时踏键才会被收到。

在 ``umi_scripts_csw`` 下::

  python3 nano_sync_freq/pedal_freq_listener.py
  python3 nano_sync_freq/pedal_freq_listener.py --self-test
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_NS = Path(__file__).resolve().parent
_UMI = _NS.parent
if str(_UMI) not in sys.path:
    sys.path.insert(0, str(_UMI))

from nano_sync_freq.play_markers import play_start_marker, play_stop_marker  # noqa: E402


def _validate_key_one_char(s: str) -> str:
    if len(s) != 1:
        raise SystemExit("键位须为单字符，如 a 或 c")
    return s.lower()


def run_terminal_only(start_key: str, stop_key: str) -> None:
    """只读当前 stdin（本终端），不装任何额外包。"""
    s_low = _validate_key_one_char(start_key)
    t_low = _validate_key_one_char(stop_key)
    if s_low == t_low:
        raise SystemExit("开始键与结束键不能相同")

    print(
        f"本终端监听中: 「{start_key!r}」=开始标音  「{stop_key!r}」=结束标音  q=退出\n"
        f"请**先让本终端窗口获得焦点**再按键（脚踏也须把键送到此终端）。",
        flush=True,
    )

    if sys.platform == "win32":
        print("Windows: 每行输入一个字母后回车（a / c / q）", flush=True)
        while True:
            line = (input() or "").strip().lower()
            if not line:
                continue
            c = line[0]
            if c == "q":
                print("已退出", flush=True)
                return
            if c == s_low:
                print("[开始]", flush=True)
                play_start_marker()
            elif c == t_low:
                print("[结束]", flush=True)
                play_stop_marker()
        return

    if not os.isatty(sys.stdin.fileno()):
        print("[非 TTY] 从标准输入按行读首字符", flush=True)
        while True:
            line = sys.stdin.readline()
            if not line:
                return
            ch = (line.strip() or "x")[:1].lower()
            if ch == "q":
                return
            if ch == s_low:
                play_start_marker()
            elif ch == t_low:
                play_stop_marker()
        return

    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            chb = os.read(fd, 1)
            if not chb:
                return
            c = chb.decode("utf-8", errors="ignore").lower()
            if c in ("\x03", "q"):
                print("\n已退出", flush=True)
                return
            if c == s_low:
                print("\n[开始]", end="", flush=True)
                play_start_marker()
            elif c == t_low:
                print("\n[结束]", end="", flush=True)
                play_stop_marker()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main() -> None:
    p = argparse.ArgumentParser(
        description="本终端内 a/c 播标定音（chirp），无 pynput"
    )
    p.add_argument(
        "--start-key",
        default="a",
        help="开始段，默认 a",
    )
    p.add_argument(
        "--stop-key",
        default="c",
        help="结束段，默认 c",
    )
    p.add_argument(
        "--self-test",
        action="store_true",
        help="只连播两次标音，不监听键",
    )
    args = p.parse_args()
    if args.self_test:
        play_start_marker()
        play_stop_marker()
        print("ok: self-test", flush=True)
        return
    run_terminal_only(args.start_key, args.stop_key)


if __name__ == "__main__":
    main()
