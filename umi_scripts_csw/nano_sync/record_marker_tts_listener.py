#!/usr/bin/env python3
"""
采集中按键盘键（默认与 fastumi 一致：a=开始，c=停止）播放 TTS 口令，便于与录像音轨
「开始录制」「停止录制」锚点对齐（见 record_marker_phrases）。

**TTS**：macOS `say`；Windows **PowerShell + System.Speech**（无需额外 pip）；Linux `espeak-ng` / `espeak` / `spd-say`（中文效果因机而异，建议装系统中文语音包）。

依赖（全局键盘监听）: ``python3 -m pip install pynput``。macOS 若编译 pyobjc 失败见下文 ``--use-stdin``。

免依赖: ``--use-stdin`` 在本终端按键（无需 pynput）。

可选（转发到采集软件，与 fastumi 一致）: ``python3 -m pip install keyboard``
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from collections.abc import Callable
from typing import Any, Literal

# 本文件在 nano_sync/；同目录为 phrase，父目录有 utils/audio_extract 等
_NANO = Path(__file__).resolve().parent
_UMI = _NANO.parent
for d in (_NANO, _UMI):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
from record_marker_phrases import (  # noqa: E402
    DEFAULT_START_PHRASE,
    DEFAULT_STOP_PHRASE,
)

OrderT = Literal["tts-first", "key-first", "tts-only"]


def _speak_windows_sapi(text: str) -> str | None:
    """PowerShell + System.Speech.Synthesis；成功返回 None。"""
    exe = shutil.which("pwsh") or shutil.which("powershell")
    if not exe:
        return "未找到 powershell.exe 或 pwsh"
    safe = text.replace("'", "''")
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.Speak('{safe}')"
    )
    p = subprocess.run(
        [exe, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        timeout=120,
        encoding="utf-8",
        errors="replace",
    )
    if p.returncode == 0:
        return None
    return f"Windows SAPI: {p.stderr or p.stdout or f'exit {p.returncode}'}"


def speak_phrase(text: str) -> str | None:
    """用本机 TTS 播音；成功返回 None，失败返回人可读错误信息。"""
    if sys.platform == "darwin":
        # 优先用系统中文语音；无 -v 时部分机器对中文句子的可听性很差
        for voice in ("Tingting", "Eddy", "Flo", None):
            cmd = ["say", text] if voice is None else ["say", "-v", voice, text]
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
            if p.returncode == 0:
                return None
        return f"say: {p.stderr or p.stdout or '未返回'}"
    if sys.platform == "win32":
        return _speak_windows_sapi(text)
    cands: list[tuple[str, list[str]]] = [
        ("espeak-ng", ["espeak-ng", "-s", "150", text]),
        ("espeak", ["espeak", "-s", "150", text]),
    ]
    for exe, argl in cands:
        pth = shutil.which(exe)
        if not pth:
            continue
        p = subprocess.run([pth, *argl[1:]], capture_output=True, text=True, timeout=120, check=False)
        if p.returncode == 0:
            return None
        return f"{exe}: {p.stderr or p.stdout}"
    if shutil.which("spd-say"):
        p = subprocess.run(
            ["spd-say", "-l", "zh", "-w", text],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if p.returncode == 0:
            return None
        return f"spd-say: {p.stderr}"
    return (
        "无 TTS。macOS: say；Windows: 需 PowerShell + .NET（通常已装）；"
        "Linux: sudo apt install espeak-ng 或 espeak 或 speech-dispatcher"
    )


def forward_start_collect() -> str | None:
    try:
        import keyboard
    except ImportError:
        return "未安装 keyboard，跳过转发( pip install keyboard )"
    keyboard.send("up")
    time.sleep(0.05)
    keyboard.send("enter")
    return None


def forward_stop_collect() -> str | None:
    try:
        import keyboard
    except ImportError:
        return "未安装 keyboard，跳过转发( pip install keyboard )"
    keyboard.send("ctrl+c")
    return None


def _make_tts_actions(
    *,
    start_key: str,
    stop_key: str,
    start_phrase: str,
    stop_phrase: str,
    order: OrderT,
    tts_delay_sec: float,
) -> tuple[Callable[[], None], Callable[[], None], str, str]:
    if len(start_key) != 1 or len(stop_key) != 1:
        raise SystemExit("键位须为单字符，如 a / c")
    s_low = start_key.lower()
    t_low = stop_key.lower()

    def do_start() -> None:
        print(f"[TTS+action] 开始: {start_phrase!r} / key={start_key}", flush=True)
        err: str | None
        if order == "key-first":
            e = forward_start_collect()
            if e:
                print(f"[forward] {e}", flush=True)
            time.sleep(tts_delay_sec)
            err = speak_phrase(start_phrase)
        elif order == "tts-only":
            err = speak_phrase(start_phrase)
        else:
            err = speak_phrase(start_phrase)
            if not err:
                time.sleep(tts_delay_sec)
                e2 = forward_start_collect()
                if e2:
                    print(f"[forward] {e2}", flush=True)
        if err:
            print(f"[TTS] {err}", flush=True)

    def do_stop() -> None:
        print(f"[TTS+action] 停止: {stop_phrase!r} / key={stop_key}", flush=True)
        err: str | None
        if order == "key-first":
            e = forward_stop_collect()
            if e:
                print(f"[forward] {e}", flush=True)
            time.sleep(tts_delay_sec)
            err = speak_phrase(stop_phrase)
        elif order == "tts-only":
            err = speak_phrase(stop_phrase)
        else:
            err = speak_phrase(stop_phrase)
            if not err:
                time.sleep(tts_delay_sec)
                e2 = forward_stop_collect()
                if e2:
                    print(f"[forward] {e2}", flush=True)
        if err:
            print(f"[TTS] {err}", flush=True)

    return do_start, do_stop, s_low, t_low


def run_with_stdin(
    *,
    start_key: str,
    stop_key: str,
    start_phrase: str,
    stop_phrase: str,
    order: OrderT,
    tts_delay_sec: float,
) -> None:
    do_start, do_stop, s_low, t_low = _make_tts_actions(
        start_key=start_key,
        stop_key=stop_key,
        start_phrase=start_phrase,
        stop_phrase=stop_phrase,
        order=order,
        tts_delay_sec=tts_delay_sec,
    )
    print(
        f"[stdin] '{start_key}'=开始 「{start_phrase}」  "
        f"'{stop_key}'=停止 「{stop_phrase}」；q=退出  order={order!r}",
        flush=True,
    )
    if sys.platform == "win32":
        print("[stdin] Windows：每行输入一个字母后回车（a/c），q 退出", flush=True)
        while True:
            line = input().strip().lower()
            if not line:
                continue
            c = line[0]
            if c == "q":
                return
            if c == s_low:
                do_start()
            elif c == t_low:
                do_stop()
    if not sys.stdin.isatty():
        print("[stdin] 非交互终端，改为每行读首字符", flush=True)
        while True:
            line = sys.stdin.readline()
            if not line:
                return
            ch = (line.strip() or " ")[:1].lower()
            if ch == "q":
                return
            if ch == s_low:
                do_start()
            elif ch == t_low:
                do_stop()
        return

    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            ch = sys.stdin.read(1)
            if not ch:
                return
            c = ch.lower()
            if c in ("\x03", "q"):
                print("\n[stdin] 退出", flush=True)
                return
            if c == s_low:
                do_start()
            elif c == t_low:
                do_stop()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def run_with_pynput(
    *,
    start_key: str,
    stop_key: str,
    start_phrase: str,
    stop_phrase: str,
    order: OrderT,
    tts_delay_sec: float,
) -> None:
    from pynput import keyboard  # type: import-not-found

    do_start, do_stop, s_low, t_low = _make_tts_actions(
        start_key=start_key,
        stop_key=stop_key,
        start_phrase=start_phrase,
        stop_phrase=stop_phrase,
        order=order,
        tts_delay_sec=tts_delay_sec,
    )

    def on_press(key: Any) -> None:
        if getattr(key, "char", None) is None:
            return
        ch = key.char.lower() if key.char else ""
        if ch == s_low:
            do_start()
        elif ch == t_low:
            do_stop()

    print(
        f"监听中: '{start_key}'=开始(播「{start_phrase}」) "
        f"'{stop_key}'=停止(播「{stop_phrase}」) order={order!r}；Ctrl+C 退出",
        flush=True,
    )
    if sys.platform == "darwin":
        print(
            "macOS: 请用**英文输入**下按单键（小写 a/c）。若按键完全无反应："
            "系统设置 → 隐私与安全性 → **辅助功能** — 打开你正在用的**终端**或 **Python**；"
            "若仍无反应，再检查 **输入监控** 列表。",
            flush=True,
        )
    elif sys.platform == "win32":
        print(
            "Windows: 若 pynput 无反应，可试「以管理员运行」终端；或改用 --use-stdin。",
            flush=True,
        )
    with keyboard.Listener(on_press=on_press) as ln:
        ln.join()


def main() -> None:
    p = argparse.ArgumentParser(
        description="采集中 a/c: TTS 播报「开始/停止录制」并可转发到 fastumi 同序按键"
    )
    p.add_argument(
        "--start-key",
        default="a",
        help="与 fastumi 默认「开始」一致，默认 a",
    )
    p.add_argument(
        "--stop-key",
        default="c",
        help="与 fastumi 默认「结束」一致，默认 c",
    )
    p.add_argument("--start-phrase", default=DEFAULT_START_PHRASE, help="TTS 文")
    p.add_argument("--stop-phrase", default=DEFAULT_STOP_PHRASE, help="TTS 文")
    p.add_argument(
        "--order",
        default="tts-first",
        choices=("tts-first", "key-first", "tts-only"),
        help="先播再发键/先发再播/仅 TTS 不派键",
    )
    p.add_argument(
        "--tts-delay",
        type=float,
        default=0.15,
        help="TTS 与发键之间间隔(秒)（在 tts-only 时无键）",
    )
    p.add_argument(
        "--self-test",
        action="store_true",
        help="只测两次 TTS，不监听键盘",
    )
    p.add_argument(
        "--use-stdin",
        action="store_true",
        help="不用 pynput：在本终端按键（macOS/Linux 单键；Windows 每行 a/c 回车）。装不上 pynput/pyobjc 时用",
    )
    args = p.parse_args()
    if args.self_test:
        for lab, u in (("开始", args.start_phrase), ("停止", args.stop_phrase)):
            e = speak_phrase(u)
            if e:
                print(f"[TTS] {lab}: 失败: {e}", file=sys.stderr, flush=True)
                raise SystemExit(1)
            print(f"[TTS] {lab}: 已播 {u!r}", flush=True)
        return
    if args.use_stdin:
        run_with_stdin(
            start_key=args.start_key,
            stop_key=args.stop_key,
            start_phrase=args.start_phrase,
            stop_phrase=args.stop_phrase,
            order=args.order,  # type: arg-type
            tts_delay_sec=args.tts_delay,
        )
        return
    try:
        run_with_pynput(
            start_key=args.start_key,
            stop_key=args.stop_key,
            start_phrase=args.start_phrase,
            stop_phrase=args.stop_phrase,
            order=args.order,  # type: arg-type
            tts_delay_sec=args.tts_delay,
        )
    except ImportError as e:
        raise SystemExit(
            "无法导入 pynput。可选方案：\n"
            "  A) 不装依赖，在本终端交互：  python3 nano_sync/record_marker_tts_listener.py --use-stdin\n"
            "  B) 先升级 pip 再装（常能直接下到 wheel，避免编译 pyobjc）：\n"
            "       python3 -m pip install -U pip\n"
            "       python3 -m pip install pynput\n"
            "  C) 若仍从源码编 pyobjc 失败（clang -Werror），可临时放宽编译：\n"
            "       export CFLAGS=\"-Wno-error=default-const-init-var-unsafe -Wno-default-const-init-var-unsafe\"\n"
            "       python3 -m pip install pynput\n"
            "  D) 换用 Homebrew 的较新 Python（3.10+）再装 pynput，一般自带 arm64  wheel。\n"
            "（可选: python3 -m pip install keyboard 用于转发到采集窗）"
        ) from e


if __name__ == "__main__":
    main()
