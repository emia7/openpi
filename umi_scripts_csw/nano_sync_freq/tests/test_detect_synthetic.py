"""C. 合成长音频 + 多组标定嵌入。"""

from __future__ import annotations

import numpy as np

from nano_sync_freq import config
from nano_sync_freq.detect_tones import detect_marker_times, resample_mono
from nano_sync_freq.tone_gen import build_reference_samples


def _embed_bursts(
    sr: int, duration_s: float, places: list[tuple[str, float]]
) -> np.ndarray:
    n = int(sr * duration_s)
    y = 0.01 * np.random.default_rng(42).standard_normal(n)
    t_start = build_reference_samples("start", sample_rate=sr)
    t_stop = build_reference_samples("stop", sample_rate=sr)
    for kind, t0 in places:
        t0s = t0
        tplt = t_start if kind == "start" else t_stop
        i0 = int(round(t0s * sr))
        sl = min(len(tplt), n - i0)
        if sl > 0:
            y[i0 : i0 + sl] += 0.95 * tplt[:sl]
    return y.astype(np.float64)


def test_detect_two_rounds() -> None:
    sr = config.SAMPLE_RATE_HZ
    x = _embed_bursts(
        sr,
        5.0,
        [
            ("start", 0.5),
            ("stop", 1.2),
            ("start", 2.8),
            ("stop", 3.4),
        ],
    )
    ts = build_reference_samples("start", sample_rate=sr).astype(np.float64)
    to = build_reference_samples("stop", sample_rate=sr).astype(np.float64)
    d = detect_marker_times(x, sr, ts, to)
    s = d["start_times"]
    t = d["stop_times"]
    assert len(s) == 2 and len(t) == 2
    # 20–50 ms
    assert abs(s[0] - 0.5) < 0.05
    assert abs(t[0] - 1.2) < 0.05
    assert abs(s[1] - 2.8) < 0.05
    assert abs(t[1] - 3.4) < 0.05


def test_detect_rescaled_amplitude() -> None:
    sr = config.SAMPLE_RATE_HZ
    x = _embed_bursts(sr, 3.0, [("start", 0.4), ("stop", 1.1)])
    x = x * 0.8
    ts = build_reference_samples("start", sample_rate=sr).astype(np.float64)
    to = build_reference_samples("stop", sample_rate=sr).astype(np.float64)
    d = detect_marker_times(x, sr, ts, to)
    assert abs(d["start_times"][0] - 0.4) < 0.08
    assert abs(d["stop_times"][0] - 1.1) < 0.08


def test_resample_mono_roundtrip() -> None:
    y = np.sin(np.linspace(0, 20, 1000, dtype=np.float64))
    z = resample_mono(resample_mono(y, 10_000, 48_000), 48_000, 10_000)
    assert z.size == 1000
    assert float(np.max(np.abs(z - y))) < 0.02
