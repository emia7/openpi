"""E. 写临时 WAV 再跑 detect_wav（需 assets）。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from nano_sync_freq import config
from nano_sync_freq.detect_tones import detect_wav
from nano_sync_freq.tone_gen import build_reference_samples, ensure_default_assets, write_wav_f32


def test_detect_wav_roundtrip(tmp_path: Path) -> None:
    ensure_default_assets(overwrite=True)
    sr = config.SAMPLE_RATE_HZ
    n = int(3.0 * sr)
    y = 0.002 * np.random.default_rng(1).standard_normal(n)
    ts = build_reference_samples("start", sample_rate=sr)
    to = build_reference_samples("stop", sample_rate=sr)
    y[int(0.5 * sr) : int(0.5 * sr) + len(ts)] += 0.9 * ts
    y[int(1.5 * sr) : int(1.5 * sr) + len(to)] += 0.9 * to
    p = tmp_path / "x.wav"
    write_wav_f32(p, y.astype(np.float32), sr)
    d = detect_wav(p)
    assert len(d["start_times"]) == 1
    assert len(d["stop_times"]) == 1
    assert abs(d["start_times"][0] - 0.5) < 0.06
    assert abs(d["stop_times"][0] - 1.5) < 0.06
