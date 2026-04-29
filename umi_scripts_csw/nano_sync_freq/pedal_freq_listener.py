#!/usr/bin/env python3
"""
采集中按 **a=开始段**、**c=结束段** 播标定 chirp（`assets/start.wav` / `stop.wav`）。

- **默认**：只监听**本终端** stdin，无额外依赖，但需该终端获焦。
- **全局**（``--global``）：用 **pynput** 在图形会话中全局听键，无需留在本终端
  （``pip install pynput``；Linux X11 通常可用，Wayland 常受限，见程序启动提示）。

在 ``umi_scripts_csw`` 下::

  python3 nano_sync_freq/pedal_freq_listener.py
  python3 nano_sync_freq/pedal_freq_listener.py --global
  python3 nano_sync_freq/pedal_freq_listener.py --self-test
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

_NS = Path(__file__).resolve().parent
_UMI = _NS.parent
if str(_UMI) not in sys.path:
    sys.path.insert(0, str(_UMI))

from nano_sync_freq.play_markers import play_start_marker, play_stop_marker  # noqa: E402


def _validate_key_one_char(s: str) -> str:
    if len(s) != 1:
        raise SystemExit("键位须为单字符，如 a 或 c")
    return s.lower()


def _read_after_esc_csi_or_ss3(fd: int) -> int | None:
    """
    已读入 0x1b，再读子序列：若为 CSI(ESC[…]) 或 SS3(ESCO…) 等转义，读毕并返回 None
    （避免方向键上为 ESC [ A 时末字节 0x41 被误当成字母 A/a）。

    若次字节不是 '['/'O'，则将其视为 M- / Alt+键 等**单字节后继**，返回该字节作按键用。
    """
    b2 = os.read(fd, 1)
    if not b2:
        return None
    x2 = b2[0]
    if x2 == 0x5B:  # [  CSI
        while True:
            bn = os.read(fd, 1)
            if not bn:
                return None
            c = bn[0]
            if 0x40 <= c <= 0x7E:  # final byte
                return None
    if x2 == 0x4F:  # O  SS3
        _ = os.read(fd, 1)
        return None
    return x2


def run_terminal_only(start_key: str, stop_key: str) -> None:
    """只读当前 stdin（本终端），不装任何额外包。"""
    s_low = _validate_key_one_char(start_key)
    t_low = _validate_key_one_char(stop_key)
    if s_low == t_low:
        raise SystemExit("开始键与结束键不能相同")

    print(
        f"本终端监听中: 「{start_key!r}」=开始标音  「{stop_key!r}」=结束标音  q=退出\n"
        f"请**先让本终端窗口获得焦点**再按键（脚踏也须把键送到此终端）。\n"
        f"说明：开始键为「a」时，若**不按转义处理**，上方向键(ESC[ A)的末字节会误成「a」；\n"
        f" 本程序已忽略方向键/功能键的终端转义。",
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
            b0 = chb[0]
            if b0 == 0x1B:  # ESC: 方向键 CSI，或 M-/Alt+ 后继单字节
                alt = _read_after_esc_csi_or_ss3(fd)
                if alt is None:
                    continue
                c = chr(alt).lower()
            else:
                c = chb.decode("utf-8", errors="ignore").lower()
            if not c:
                continue
            c0 = c[0]
            if c0 in ("\x03",) or c0 == "q":
                print("\n已退出", flush=True)
                return
            if c0 == s_low:
                print("\n[开始]", end="", flush=True)
                play_start_marker()
            elif c0 == t_low:
                print("\n[结束]", end="", flush=True)
                play_stop_marker()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def run_with_pynput(start_key: str, stop_key: str) -> None:
    """用 pynput 在桌面会话中全局监听；与 ``record_marker_tts_listener`` 同思路。"""
    from pynput import keyboard  # type: import-not-found

    s_low = _validate_key_one_char(start_key)
    t_low = _validate_key_one_char(stop_key)
    if s_low == t_low:
        raise SystemExit("开始键与结束键不能相同")

    def on_press(key: Any) -> None:
        if getattr(key, "char", None) is None:
            return
        ch = key.char.lower() if key.char else ""
        if ch == s_low:
            print("[开始]", flush=True)
            play_start_marker()
        elif ch == t_low:
            print("[结束]", flush=True)
            play_stop_marker()

    print(
        f"全局监听(pynput): 「{start_key!r}」=开始  「{stop_key!r}」=结束；"
        f"**英文输入**下按单键；Ctrl+C 退出",
        flush=True,
    )
    if sys.platform == "darwin":
        print(
            "macOS: 系统设置 → 隐私与安全性 → 辅助功能 / 输入监控，"
            "允许当前终端或 Python。",
            flush=True,
        )
    elif sys.platform == "linux":
        print(
            "Linux: 须图形会话。X11 下 pynput 一般可用；**Wayland** 下常无法全局钩键，"
            "可换 Xorg 登录、或继续用**默认的终端内监听**、"
            "或使用 **evdev** 等直接读 ``/dev/input/`` 的方案（本脚本未内置）。\n"
            "若需 uinput 权限: ``sudo usermod -aG input $USER`` 后重登。",
            flush=True,
        )
    elif sys.platform == "win32":
        print("Windows: 若无反应可「以管理员运行」终端，或改回不加 --global。\n", flush=True)
    with keyboard.Listener(on_press=on_press) as ln:
        ln.join()


def main() -> None:
    p = argparse.ArgumentParser(
        description="a/c 播标定音（chirp）：默认本终端，或 --global 用 pynput 全局"
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
    p.add_argument(
        "--global",
        action="store_true",
        dest="pynput_global",
        help="用 pynput 全局听键（需 pip install pynput），勿切回终端",
    )
    args = p.parse_args()
    if args.self_test:
        play_start_marker()
        play_stop_marker()
        print("ok: self-test", flush=True)
        return
    if args.pynput_global:
        try:
            run_with_pynput(args.start_key, args.stop_key)
        except ImportError as e:
            raise SystemExit(
                "需要 pynput:  python3 -m pip install pynput\n"
                "或去掉 --global，用默认的终端内监听。"
            ) from e
        return
    run_terminal_only(args.start_key, args.stop_key)


if __name__ == "__main__":
    main()
