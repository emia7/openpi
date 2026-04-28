#!/usr/bin/env python3
"""采集中播放开始/结束标定音（与 `assets/*.wav` 一致）。"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_NS = Path(__file__).resolve().parent
_UMI = _NS.parent
if str(_UMI) not in sys.path:
    sys.path.insert(0, str(_UMI))

from nano_sync_freq.tone_gen import default_asset_paths, ensure_default_assets


def play_start_marker() -> None:
    """播开始标定音（`assets/start.wav`）。"""
    ensure_default_assets()
    _play(default_asset_paths()["start"])


def play_stop_marker() -> None:
    """播结束标定音（`assets/stop.wav`）。"""
    ensure_default_assets()
    _play(default_asset_paths()["stop"])


def _play(path: Path) -> None:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if sys.platform == "darwin" and shutil.which("afplay"):
        subprocess.run(["afplay", str(path)], check=True)
        return
    if shutil.which("ffplay"):
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)],
            check=True,
        )
        return
    try:
        import sounddevice as sd  # type: ignore[import-not-found]
        import soundfile as sf  # type: ignore[import-not-found]

        data, sr = sf.read(str(path), always_2d=False)
        sd.play(data, int(sr), blocking=True)
        return
    except Exception:
        pass
    raise RuntimeError(
        "无法播放：请安装 afplay(mac)、ffplay(ffmpeg)，或 pip install sounddevice soundfile"
    )


def _interactive_line(paths: dict[str, Path]) -> None:
    print("每行输入一个字母后**回车**：s=开始音  t=停/结束音  a=两声连播  q=退出", flush=True)
    while True:
        line = (input() or "").strip().lower()
        if not line or line[0] == "q":
            print("已退出", flush=True)
            return
        c = line[0]
        if c in ("h", "?"):
            print("s=开始  t=停  a=全播  q=退出", flush=True)
        elif c == "s":
            play_start_marker()
        elif c == "t":
            play_stop_marker()
        elif c == "a":
            play_start_marker()
            play_stop_marker()
        else:
            print("未知：请用 s / t / a / q", flush=True)


def _interactive_rawkeys(paths: dict[str, Path]) -> None:
    import termios
    import tty

    fd = sys.stdin.fileno()
    if not os.isatty(fd):
        _interactive_line(paths)
        return
    print(
        "单键**无需回车**：s=开始  t=停  a=全播  q=退出  ?=说明",
        flush=True,
    )
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            ch = os.read(fd, 1)
            if not ch:
                break
            c = ch.decode("utf-8", errors="ignore").lower()
            if c in ("\x03", "q"):
                print("\n已退出", flush=True)
                return
            if c == "\x1b":
                print("\n已退出", flush=True)
                return
            if c in ("h", "?"):
                print("\n(s/t/a/q)", flush=True)
                continue
            if c == "s":
                play_start_marker()
            elif c == "t":
                play_stop_marker()
            elif c == "a":
                play_start_marker()
                play_stop_marker()
            elif c in ("\n", "\r"):
                return
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _run_interactive() -> None:
    paths = default_asset_paths()
    if sys.platform == "win32" or not sys.stdin.isatty():
        _interactive_line(paths)
    else:
        try:
            _interactive_rawkeys(paths)
        except OSError:
            _interactive_line(paths)


def main() -> None:
    ensure_default_assets()
    ap = argparse.ArgumentParser(description="播放开始/结束标定音 WAV")
    ap.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="按键测试：s=开始 t=停 a=连播 q=退出（终端单键模式，无此参数见 --self-test）",
    )
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument("--start", action="store_true", help="播放开始音")
    g.add_argument("--stop", action="store_true", help="播放结束音")
    g.add_argument(
        "--self-test",
        action="store_true",
        help="依次播放 start 与 stop 各一次，用于检查输出设备",
    )
    args = ap.parse_args()
    paths = default_asset_paths()
    if args.interactive:
        _run_interactive()
        return
    if not (args.start or args.stop or args.self_test):
        ap.print_help()
        print(
            "\n快速听两声：  python3 nano_sync_freq/play_markers.py --self-test",
            "\n按键玩：      python3 nano_sync_freq/play_markers.py -i",
            flush=True,
        )
        raise SystemExit(0)
    if args.self_test:
        play_start_marker()
        play_stop_marker()
        print("ok: self-test 已播放开始+结束", flush=True)
        return
    if args.start:
        play_start_marker()
    if args.stop:
        play_stop_marker()


if __name__ == "__main__":
    main()
