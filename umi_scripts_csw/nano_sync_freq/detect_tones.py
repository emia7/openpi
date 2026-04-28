"""在长音频上通过归一化互相关（FFT）检测开始/结束标定音时刻。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from . import tone_gen


def resample_mono(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return y.astype(np.float64)
    n_out = int(max(2, round(len(y) * float(sr_out) / float(sr_in))))
    t_in = np.arange(len(y), dtype=np.float64) / float(sr_in)
    t_out = np.linspace(0.0, (len(y) - 1) / float(sr_in), n_out, endpoint=True)
    return np.interp(t_out, t_in, y.astype(np.float64))


def _fft_xcorr_valid(x: np.ndarray, template: np.ndarray) -> np.ndarray:
    """x 与 template 的 valid 互相关 (sum_m x[k+m]*t[m])，返回长度 len(x)-len(t)+1。"""
    x = np.ascontiguousarray(x, dtype=np.float64)
    t = np.ascontiguousarray(template, dtype=np.float64)
    nt = t.size
    if nt > x.size:
        return np.array([], dtype=np.float64)
    b = t[::-1]
    n = x.size + b.size - 1
    p = 1
    while p < n:
        p *= 2
    xp = np.zeros(p, dtype=np.float64)
    xp[: x.size] = x
    bp = np.zeros(p, dtype=np.float64)
    bp[: b.size] = b
    c = np.fft.irfft(np.fft.rfft(xp) * np.fft.rfft(bp), n=p)
    return c[nt - 1 : x.size]


def _sliding_norm_xcorr(x: np.ndarray, template: np.ndarray) -> np.ndarray:
    """
    对模板 L2 归一化，对 x 的每个 position 的窗做 demean + L2 归一化后点积（近似 NCC）。
    用 O(n) 递推窗能量；实现为滑动 window 的简版：仅 demean 整段后相关再局部归一化近似。
    为速度：先 demean x，再模板已单位化，相关后除以 rolling std（简化用全局 std 或局部）—
    这里用 **整段去均值** + 模板 L2=1，输出用局部滑动 RMS 除（padding same length）。
    """
    t = template.astype(np.float64)
    t = t - t.mean()
    nm = float(np.linalg.norm(t))
    if nm < 1e-12:
        return np.zeros(max(len(x) - len(t) + 1, 0))
    t = t / nm
    x0 = x.astype(np.float64) - np.mean(x)
    raw = _fft_xcorr_valid(x0, t)
    if raw.size == 0:
        return raw
    # 滑动窗 RMS（与 template 等长）
    n = t.size
    win = x0**2
    csum = np.cumsum(np.insert(win, 0, 0.0))
    rms = np.sqrt(np.maximum(csum[n:] - csum[:-n], 1e-18))
    # raw 下标 k 对应 x[k:k+n]
    out = raw / (rms + 1e-9)
    return out


def _find_peaks(
    y: np.ndarray,
    sample_rate: int,
    min_distance_sec: float = config.MIN_PEAK_DISTANCE_SEC,
    snr_ratio: float = config.PEAK_SNR_RATIO,
    abs_floor: float = config.ABS_CORR_FLOOR,
    *,
    snr_cap: float | None = None,
) -> list[int]:
    if y.size < 3:
        return []
    y = np.ascontiguousarray(y, dtype=np.float64)
    m = float(np.max(y))
    if m < abs_floor:
        return []
    thr_raw = m * snr_ratio
    if snr_cap is not None and snr_cap > 0:
        thr_raw = min(thr_raw, float(snr_cap))
    thr = max(thr_raw, abs_floor)
    mind = int(max(1, min_distance_sec * sample_rate))
    idx: list[int] = []
    i = 1
    while i < y.size - 1:
        if y[i] >= thr and y[i] >= y[i - 1] and y[i] >= y[i + 1]:
            # 取局部平台最高
            j = i
            while j + 1 < y.size and y[j + 1] >= y[j] - 1e-9:
                j += 1
            peak = j
            if not idx or peak - idx[-1] >= mind:
                idx.append(peak)
            i = peak + mind
        else:
            i += 1
    return idx


def _indices_to_sec(indices: list[int], sample_rate: int, template_len: int) -> list[float]:
    # 互相关 argmax 对应**模板首样本**在 x 中的对齐下标
    return [i / float(sample_rate) for i in indices]


def detect_marker_times(
    x: np.ndarray,
    sample_rate: int,
    template_start: np.ndarray,
    template_stop: np.ndarray,
    *,
    snr_cap: float | None = None,
) -> dict[str, Any]:
    x = np.ascontiguousarray(x, dtype=np.float64)
    if x.ndim != 1:
        if x.ndim == 2 and x.shape[1] == 1:
            x = x.ravel()
        else:
            raise ValueError("需要单声道一维数组")
    rs = _sliding_norm_xcorr(x, template_start)
    rt = _sliding_norm_xcorr(x, template_stop)
    p_s = _find_peaks(rs, sample_rate, snr_cap=snr_cap)
    p_t = _find_peaks(rt, sample_rate, snr_cap=snr_cap)
    t_s = _indices_to_sec(p_s, sample_rate, template_start.size)
    t_t = _indices_to_sec(p_t, sample_rate, template_stop.size)
    return {
        "sample_rate": int(sample_rate),
        "start_times": t_s,
        "stop_times": t_t,
        "n_start_peaks": len(t_s),
        "n_stop_peaks": len(t_t),
    }


def detect_wav(
    path: Path,
    *,
    assets: dict[str, Path] | None = None,
    snr_cap: float | None = None,
) -> dict[str, Any]:
    """读 WAV（自动转单声道 float），相对 assets 中的 `start`/`stop` 参考检测。

    `snr_cap`：对 ``thr = max_corr * PEAK_SNR_RATIO`` 设上界，避免前段过强标音使门限过高、
    后段真实峰被滤掉（可配 ``config.PEAK_SNR_CAP``）。
    """
    p = Path(path)
    y, sr = tone_gen.read_wav_f32(p)
    y = resample_mono(y, sr, config.SAMPLE_RATE_HZ)
    sr = config.SAMPLE_RATE_HZ
    paths = assets or tone_gen.default_asset_paths()
    ts, srt = tone_gen.read_wav_f32(Path(paths["start"]))
    to, sro = tone_gen.read_wav_f32(Path(paths["stop"]))
    ts = resample_mono(ts, srt, config.SAMPLE_RATE_HZ)
    to = resample_mono(to, sro, config.SAMPLE_RATE_HZ)
    ts = np.ascontiguousarray(ts, dtype=np.float64)
    to = np.ascontiguousarray(to, dtype=np.float64)
    d = detect_marker_times(y, sr, ts, to, snr_cap=snr_cap)
    d["file"] = str(p.resolve())
    d["duration_sec"] = float(len(y)) / sr
    return d


def _parabolic_peak_offset_1d(y: np.ndarray, j: int) -> float:
    """离散序列在 j 处峰值的子样本偏移（约 -0.5~0.5 样本），边界处返回 0。"""
    y = np.ascontiguousarray(y, dtype=np.float64)
    if y.size < 3 or j < 1 or j >= y.size - 1:
        return 0.0
    a, b, c = float(y[j - 1]), float(y[j]), float(y[j + 1])
    den = a - 2.0 * b + c
    if abs(den) < 1e-12:
        return 0.0
    return 0.5 * (a - c) / den


def _scan_start_template_ncc_in_interval(
    x: np.ndarray,
    sample_rate: int,
    template_start: np.ndarray,
    t_lo: float,
    t_hi: float,
    *,
    rel_to_local_max: float = 0.48,
    abs_min: float = 0.10,
) -> tuple[float | None, float]:
    """
    在 [t_lo, t_hi] 内用 start 模板 NCC 对整窗取 **argmax**（须 >= ``abs_min``），
    并对峰位做一维抛物线子样本修正。返回 ``(时间或 None, 窗内最大 NCC)``。
    ``rel_to_local_max`` 仅保留 API 兼容，未使用。
    """
    del rel_to_local_max
    x = np.ascontiguousarray(x, dtype=np.float64)
    nt = int(template_start.size)
    if t_hi - t_lo < 0.08 or t_lo < 0:
        return None, 0.0
    t_lo = max(0.0, float(t_lo))
    t_hi = min(float(t_hi), (len(x) - nt) / float(sample_rate))
    if t_hi <= t_lo + 0.02:
        return None, 0.0
    rs = _sliding_norm_xcorr(x, np.ascontiguousarray(template_start))
    if rs.size < 3:
        return None, 0.0
    k_lo = int(t_lo * sample_rate)
    k_hi = int(t_hi * sample_rate)
    k_lo = max(0, k_lo)
    k_hi = min(k_hi, len(rs) - 1)
    if k_hi - k_lo < 2:
        return None, 0.0
    rseg = rs[k_lo : k_hi + 1]
    gmax = float(np.max(rseg))
    if gmax < float(abs_min):
        return None, gmax
    j = int(np.argmax(rseg))
    if rseg.size >= 3 and 0 < j < rseg.size - 1:
        off = _parabolic_peak_offset_1d(rseg, j)
    else:
        off = 0.0
    return (k_lo + j + off) / float(sample_rate), gmax


def _nearest_idx(sorted_ts: list[float], t: float, eps: float = 0.02) -> int:
    for j, x in enumerate(sorted_ts):
        if abs(x - t) < eps:
            return j
    return -1


def refine_starts_for_unpaired_stops(
    x: np.ndarray,
    sample_rate: int,
    template_start: np.ndarray,
    start_times: list[float],
    stop_times: list[float],
    duration_sec: float | None = None,
    max_rounds: int = 8,
) -> tuple[list[float], list[str]]:
    """
    在 ``len(stop) > len(start)`` 成对后产生的「unpaired stop」所对应的区间内，
    用**局部**门限再扫一次 start 模板，并插入时间戳后**重新**贪心成对，直至无多出来的 stop。
    """
    from .pair_markers import pair_start_stop_times

    st = sorted(float(t) for t in start_times)
    sp = sorted(float(t) for t in stop_times)
    log: list[str] = []
    for _ in range(max_rounds):
        pr = pair_start_stop_times(st, sp, duration_sec=duration_sec)
        to_fill: list[float] = []
        for w in pr.warnings:
            if w.startswith("unpaired_stop:"):
                rest = w.split(":", 1)[1].strip()
                to_fill.append(float(rest))
        if not to_fill:
            break
        u = min(to_fill)
        j = _nearest_idx(sp, u, eps=0.03)
        if j < 0:
            log.append(f"weak_start_skip: unpaired {u:.3f} not matched in stop list")
            break
        prev = sp[j - 1] if j > 0 else 0.0
        from . import config as _cfg

        t_lo = prev + float(_cfg.REFINE_AFTER_PREV_STOP_SEC)
        t_hi = u - float(_cfg.REFINE_BEFORE_STOP_SEC)
        t_new, ncc = _scan_start_template_ncc_in_interval(
            x,
            sample_rate,
            template_start,
            t_lo,
            t_hi,
            abs_min=float(_cfg.WEAK_START_ABS_MIN),
        )
        if t_new is None or any(abs(t_new - t) < 0.04 for t in st):
            log.append(
                f"weak_start_scan_failed: interval=({t_lo:.3f},{t_hi:.3f}) for unpaired_stop={u:.3f} (ncc_max={ncc:.3f})"
            )
            break
        st.append(t_new)
        st.sort()
        wmsg = f"refined_start_inserted: {t_new:.3f}s ncc={ncc:.3f} (unpaired stop {u:.3f})"
        if ncc < float(_cfg.WEAK_START_LO_CONFIDENCE):
            wmsg += " [low NCC, consider re-record or check mic/level]"
        log.append(wmsg)
    return st, log


def write_markers_json(data: dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **data,
        "start_times": [float(t) for t in data.get("start_times", [])],
        "stop_times": [float(t) for t in data.get("stop_times", [])],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
