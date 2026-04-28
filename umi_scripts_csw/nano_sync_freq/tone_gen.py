"""生成/读写开始与结束标定音（线性 chirp + Hann 窗），供播放与互相关检测共用。"""

from __future__ import annotations

import hashlib
import wave
from pathlib import Path
from typing import Literal

import numpy as np

from . import config

Kind = Literal["start", "stop"]


def _hann(n: int) -> np.ndarray:
    if n < 2:
        return np.ones(max(n, 1), dtype=np.float64)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / (n - 1)))


def build_reference_samples(
    kind: Kind,
    *,
    sample_rate: int = config.SAMPLE_RATE_HZ,
    duration_sec: float = config.BURST_DURATION_SEC,
) -> np.ndarray:
    """
    返回 float32 一维数组，值域约 [-0.4, 0.4]。
    """
    n = max(int(round(sample_rate * duration_sec)), 32)
    t = np.arange(n, dtype=np.float64) / sample_rate
    T = max(t[-1], 1.0 / sample_rate)
    if kind == "start":
        f0, f1 = config.START_CHIRP_F0_HZ, config.START_CHIRP_F1_HZ
    else:
        f0, f1 = config.STOP_CHIRP_F0_HZ, config.STOP_CHIRP_F1_HZ
    # 相位: phi(t) = 2pi * (f0*t + (f1-f0)*t^2 / (2T))  线性调频
    k = (f1 - f0) / max(2.0 * T, 1e-9)
    phase = 2.0 * np.pi * (f0 * t + k * t * t)
    w = _hann(n)
    y = 0.38 * np.sin(phase) * w
    return y.astype(np.float32)


def write_wav_f32(
    path: Path,
    samples: np.ndarray,
    sample_rate: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(samples, -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def read_wav_f32(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        sw = w.getsampwidth()
        n = w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        raise ValueError("只支持 16-bit PCM 单/双声道")
    y = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        y = y.reshape(-1, nch).mean(axis=1)
    return y, sr


def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def default_asset_paths() -> dict[Kind, Path]:
    base = _package_dir() / "assets"
    return {
        "start": base / "start.wav",
        "stop": base / "stop.wav",
    }


def ensure_default_assets(overwrite: bool = False) -> dict[Kind, Path]:
    """在 `assets/` 写入与当前 config 一致的开始/结束参考 WAV；已存在时默认跳过。"""
    paths = default_asset_paths()
    for k in ("start", "stop"):
        kind: Kind = k  # type: ignore[assignment]
        p = paths[kind]
        if p.is_file() and not overwrite:
            continue
        y = build_reference_samples(kind)
        write_wav_f32(p, y, config.SAMPLE_RATE_HZ)
    return paths


def stop_template_duration_sec() -> float:
    """与 `build_reference_samples("stop")` 等长，用于切片时包含整段结束标音尾。"""
    y = build_reference_samples("stop")
    return float(len(y)) / float(config.SAMPLE_RATE_HZ)


def stable_reference_digest() -> str:
    """同参数连续两次生成开始/结束波形拼接后的 SHA256 前缀（检测漂移）。"""
    h = hashlib.sha256()
    h.update(int(config.SAMPLE_RATE_HZ).to_bytes(4, "little", signed=False))
    for kind in ("start", "stop"):
        w = build_reference_samples(kind)  # type: ignore[arg-type]
        h.update(np.ascontiguousarray(w, dtype=np.float32).tobytes())
    return h.hexdigest()[:32]
