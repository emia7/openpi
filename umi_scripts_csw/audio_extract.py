"""从视频抽 PCM WAV 并读入 float32 波形（供 squeak_audio_gpt4o 等脚本复用，无传统 DSP 检测）。"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None  # type: ignore


def find_ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    if imageio_ffmpeg is not None:
        return str(imageio_ffmpeg.get_ffmpeg_exe())
    raise FileNotFoundError("未找到 ffmpeg。请安装 ffmpeg 或 pip install imageio-ffmpeg。")


def probe_audio_with_ffmpeg(video: Path) -> dict[str, Any]:
    ff = find_ffmpeg()
    cmd = [ff, "-hide_banner", "-i", str(video)]
    print(f"[probe] 运行: {' '.join(cmd[:3])} <video> ...", flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    out = p.stderr or ""
    m = re.search(
        r"Stream #0:\d+.*: Audio: ([^,]+), ([0-9]+) Hz",
        out,
        re.MULTILINE,
    )
    if not m:
        return {"has_audio": False, "raw_stderr_lines": out.splitlines()[:30]}
    return {"has_audio": True, "codec": m.group(1).strip(), "sample_rate_hz": int(m.group(2))}


def extract_audio_wav(
    video: Path,
    out_wav: Path,
    sample_rate: int = 44100,
    mono: bool = True,
) -> None:
    ff = find_ffmpeg()
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ff,
        "-y",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1" if mono else "2",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(out_wav),
    ]
    print(f"[extract] 抽音中: -> {out_wav.name}", flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if p.returncode != 0:
        print(p.stderr[-4000:] if p.stderr else "", file=sys.stderr)
        raise RuntimeError(f"ffmpeg 抽音失败: exit {p.returncode}")


def load_wav_f32(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        nch = w.getnchannels()
        sw = w.getsampwidth()
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        raise ValueError(f"只支持 16-bit PCM WAV， got sampwidth={sw}")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)
    return x / 32768.0, sr


def print_pcm_stats(x: np.ndarray, sr: int, label: str = "PCM") -> None:
    rms = float(np.sqrt(np.mean(x**2)))
    print(
        f"[{label}] sr={sr} 样本数={x.size} 时长={x.size / sr:.3f}s  "
        f"rms={rms:.6f} std={float(x.std()):.6f} max|x|={float(np.max(np.abs(x))):.6f}",
        flush=True,
    )
    if x.std() < 1e-5:
        print("  警告: 方差接近 0，可能为假静音。", flush=True)
