#!/usr/bin/env python3
"""
尖叫声/短促高频声检测（飞书等 MP4 友好）

优先使用 imageio-ffmpeg 自带的 ffmpeg 将音轨解为 PCM WAV，再在上做带通 + 包络峰值检测。
避免仅依赖 moviepy 稀疏 get_frame（易出现“假静音”）。

后处理为结构合并 + 2–8 kHz 声学量（crest / band_ratio / 相对 quality），**不用**“跳过前 N 秒”
或片尾时间窗；调参请改 --sensitivity 与 postfilter 内门限。

用法:
  python squeak_detector.py /path/to/video.mp4
  python squeak_detector.py /path/to/video.mp4 --sensitivity 0.6
  python squeak_detector.py /path/to/video.mp4 --keep-wav
  python squeak_detector.py /path/to/video.mp4 --legacy-moviepy   # 旧路径（不推荐）
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

# Optional: bundled ffmpeg (no system brew required)
try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None  # type: ignore

try:
    from PIL import Image, ImageDraw
except ImportError as e:
    raise SystemExit("需要 pillow: " + str(e)) from e


@dataclass
class SqueakEvent:
    t_peak: float
    t_start: float
    t_end: float
    score: float


def _find_ffmpeg() -> str:
    """Prefer PATH ffmpeg; else imageio-ffmpeg binary."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    if imageio_ffmpeg is not None:
        return str(imageio_ffmpeg.get_ffmpeg_exe())
    raise FileNotFoundError("未找到 ffmpeg。请安装 ffmpeg 或确保已安装 imageio-ffmpeg。")


def probe_audio_with_ffmpeg(video: Path) -> dict[str, Any]:
    """Run `ffmpeg -i` and parse Audio line (no ffprobe required)."""
    ff = _find_ffmpeg()
    cmd = [ff, "-hide_banner", "-i", str(video)]
    print(f"[probe] 运行: {' '.join(cmd[:3])} <video> ...", flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    out = p.stderr or ""
    # Audio: aac, 48000 Hz, stereo, fltp, ...
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
    ff = _find_ffmpeg()
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
    print(f"[extract] 抽音中（数秒级）: -> {out_wav.name}", flush=True)
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
        print("  警告: 方差接近 0，可能为假静音，后续检测会失败。", flush=True)


def bandpass_squeak_y_2_8k(
    x: np.ndarray,
    sr: int,
    low_hz: float = 2000.0,
    high_hz: float = 8000.0,
) -> np.ndarray:
    """2–8 kHz 带通，用于与尖叫/尖促声同频段一致。"""
    x = x.astype(np.float64, copy=False)
    nyq = 0.5 * float(sr)
    lo = max(low_hz / nyq, 1e-4)
    hi = min(high_hz / nyq, 0.999)
    b, a = signal.butter(4, [lo, hi], btype="band")
    return signal.filtfilt(b, a, x).astype(np.float64, copy=False)


def bandpass_squeak_envelope(
    x: np.ndarray,
    sr: int,
    low_hz: float = 2000.0,
    high_hz: float = 8000.0,
) -> np.ndarray:
    """Return short-time RMS envelope of band-pass filtered signal."""
    y = bandpass_squeak_y_2_8k(x, sr, low_hz=low_hz, high_hz=high_hz)
    # ~20ms window
    win = max(int(0.02 * sr) | 1, 1)
    kernel = np.ones(win) / float(win)
    return np.sqrt(signal.convolve(y**2, kernel, mode="same"))


def acoustic_squeak_metrics(
    y_band: np.ndarray,
    x_full: np.ndarray,
    sr: int,
    t_peak: float,
    pre_s: float = 0.05,
    post_s: float = 0.07,
) -> tuple[float, float]:
    """
    在峰时刻附近估计两类声学量（不依赖在视频中的时间位置）:
    - crest: 2–8k 带内 max|y|/RMS(y) — 尖、短、瞬态通常较高
    - band_ratio: RMS(带通)/RMS(全带) — 尖叫/摩擦在此频段能量占比高
    """
    c = int(t_peak * sr)
    w0 = max(0, c - int(pre_s * sr))
    w1 = min(x_full.size, c + int(post_s * sr))
    if w1 - w0 < 16:
        return 0.0, 0.0
    yb = y_band[w0:w1]
    xb = x_full[w0:w1]
    r_b = float(np.sqrt(np.mean(yb**2)) + 1e-9)
    r_f = float(np.sqrt(np.mean(xb**2)) + 1e-9)
    ax = float(np.max(np.abs(yb)))
    crest = ax / r_b
    band_ratio = r_b / r_f
    return crest, band_ratio


def detect_events_from_envelope(
    t: np.ndarray,
    env: np.ndarray,
    sensitivity: float = 1.0,
) -> list[SqueakEvent]:
    """
    sensitivity: 越小越“敏感”（更多峰）。用分位数和 prominence 调节。
    """
    if env.size < 8:
        return []
    # baseline from noise floor
    med = float(np.median(env))
    p90 = float(np.percentile(env, 90))
    p99 = float(np.percentile(env, 99))
    # sensitivity: 越大越严格（峰更少）；建议 0.3 ~ 2.0
    s = float(sensitivity)
    s = max(s, 0.1)
    h = med + (p99 - med) * (0.06 * s)
    prom = max((p90 - med) * (0.15 * s), 1e-7)
    dt = float(t[1] - t[0]) if t.size > 1 else 0.01
    min_dist = max(int(round(0.12 / dt)), 1)
    peaks, prop = signal.find_peaks(
        env,
        height=h,
        distance=min_dist,
        prominence=prom,
    )
    events: list[SqueakEvent] = []
    for i, pk in enumerate(peaks):
        # rough onset/offset: expand around peak until below half-prominence (fallback window)
        left, right = pk, pk
        target = max(env[pk] * 0.25, med)
        while left > 0 and env[left] > target:
            left -= 1
        while right + 1 < env.size and env[right] > target:
            right += 1
        events.append(
            SqueakEvent(
                t_peak=float(t[pk]),
                t_start=float(t[left]),
                t_end=float(t[right]),
                score=float(prop["prominences"][i] if "prominences" in prop else env[pk]),
            )
        )

    return events


def postfilter_squeak_markers(
    events: list[SqueakEvent],
    x: np.ndarray,
    sr: int,
    *,
    merge_earliest_s: float = 0.5,
    merge_max_in_window_s: float = 1.05,
    # 声学门限：不依赖“片头/片尾第几秒”等时间先验
    min_band_ratio: float = 0.1,
    min_band_ratio_abs: float = 0.40,
    min_crest: float = 1.5,
    max_crest: float = 6.0,
    max_band_ratio: float = 0.80,
    quality_min_vs_best: float = 0.38,
) -> list[SqueakEvent]:
    """
    面向「尖促/尖叫类」声（2–8 kHz）：

    1) 同一动作重复峰：极近（~0.5s）内保留较早的峰点。
    2) 约 1s 内多峰：取 find_peaks 的 prominence 更高者（与弱噪声/回声区分）。
    3) 声学门限：在 `t_peak` 附近 50–70ms 上算
       ``crest=峰值/RMS(带)`` 与 ``band_ratio=RMS(2–8k)/RMS(全带)``，并合成
       ``quality=crest*(band_ratio+ε)^0.4``；**相对本段最强峰**截断（无片头/片尾时间窗）。

    另有几条**纯声学**的边界项（不按时长）：
    - ``min_band_ratio_abs``：2–8k 能量占全带过低则不像本类尖声样例（易与宽带钝声混淆）。
    - ``max_crest``：带内尖度过高时更像**点击/冲激**而非多周期尖叫。
    - ``max_band_ratio``：2–8k 占比**过高**时更像**窄带啸叫/喑鸣**，而非目标尖叫的带宽形态。

    若过严可略低 ``quality_min_vs_best``/``min_band_ratio_abs``，或略高 ``max_crest``/``max_band_ratio``；或略调 ``--sensitivity`` 以多留候选项。
    """
    if not events or x.size == 0:
        return []
    y_band = bandpass_squeak_y_2_8k(x, sr)
    ev = sorted(events, key=lambda e: e.t_peak)
    # 1) 极近取早
    merged: list[SqueakEvent] = [ev[0]]
    for e in ev[1:]:
        if e.t_peak - merged[-1].t_peak <= merge_earliest_s:
            if e.t_peak < merged[-1].t_peak:
                merged[-1] = e
        else:
            merged.append(e)
    # 2) 1s 内多峰取 score
    n = len(merged)
    out: list[SqueakEvent] = []
    i = 0
    while i < n:
        j = i
        run = [merged[i]]
        while j + 1 < n and merged[j + 1].t_peak - merged[i].t_peak <= merge_max_in_window_s:
            j += 1
            run.append(merged[j])
        if len(run) == 1:
            out.append(run[0])
        else:
            out.append(max(run, key=lambda e: e.score))
        i = j + 1
    if not out:
        return []
    # 3) 声学
    rows: list[tuple[SqueakEvent, float, float, float]] = []
    for e in out:
        c, br = acoustic_squeak_metrics(y_band, x, sr, e.t_peak)
        q = float(c) * (float(max(br, 0.0)) + 0.01) ** 0.4
        rows.append((e, c, br, q))
    c_max = max(t[1] for t in rows) if rows else 1.0
    br_max = max(t[2] for t in rows) if rows else 1.0
    q_max = max(t[3] for t in rows) if rows else 1.0
    kept: list[SqueakEvent] = []
    for e, c, br, q in rows:
        if c > max_crest:
            continue
        if br > max_band_ratio or br < min_band_ratio_abs:
            continue
        if br < min_band_ratio * 0.25 * max(br_max, 1e-9):
            continue
        if c < 0.22 * c_max and br < 0.1 * br_max and q < 0.25 * q_max:
            continue
        if c < min_crest * 0.4 and br < 0.35 * br_max and q < quality_min_vs_best * 0.85 * q_max:
            continue
        if q >= quality_min_vs_best * q_max:
            kept.append(e)
        elif c >= 0.4 * c_max and br >= min_band_ratio * br_max:
            kept.append(e)
    if not kept and rows:
        rows.sort(key=lambda t: t[3], reverse=True)
        n_take = min(4, len(rows))
        return sorted([t[0] for t in rows[:n_take]], key=lambda e: e.t_peak)
    return sorted(kept, key=lambda e: e.t_peak)


def run_detection_pcm(
    x: np.ndarray,
    sr: int,
    sensitivity: float,
    postfilter: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[SqueakEvent]]:
    # downsample envelope grid for speed / stability
    frame = int(0.01 * sr)
    step = max(frame // 2, 1)
    env = bandpass_squeak_envelope(x, sr)
    env_ds = np.array([env[i : i + frame].mean() for i in range(0, len(env) - frame, step)])
    t_ds = (np.arange(env_ds.size) * step + frame // 2) / float(sr)
    events = detect_events_from_envelope(t_ds, env_ds, sensitivity=sensitivity)
    if postfilter and events:
        n0 = len(events)
        events = postfilter_squeak_markers(
            events,
            x,
            sr,
            merge_earliest_s=0.5,
            merge_max_in_window_s=1.05,
        )
        print(
            f"[postfilter] {n0} -> {len(events)}  (0.5s 内取早峰, 1.05s 内取强峰, 声学门限/相对 quality)",
            flush=True,
        )
    return t_ds, env_ds, events


def save_debug_png(
    t_ds: np.ndarray,
    env_ds: np.ndarray,
    events: list[SqueakEvent],
    out_png: Path,
    title: str,
) -> None:
    w, h = 1200, 400
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    emax = float(np.max(env_ds)) if env_ds.size else 1.0
    if emax <= 0:
        emax = 1.0
    pad = 20
    for i in range(1, env_ds.size):
        x0 = pad + (i - 1) * (w - 2 * pad) / max(env_ds.size - 1, 1)
        x1 = pad + i * (w - 2 * pad) / max(env_ds.size - 1, 1)
        y0 = h - pad - (env_ds[i - 1] / emax) * (h - 2 * pad)
        y1 = h - pad - (env_ds[i] / emax) * (h - 2 * pad)
        draw.line((x0, y0, x1, y1), fill=(30, 30, 200), width=2)
    for e in events:
        xi = int(pad + (e.t_peak / max(t_ds[-1], 1e-6)) * (w - 2 * pad)) if t_ds.size else pad
        draw.line((xi, pad, xi, h - pad), fill=(220, 40, 40), width=2)
    draw.text((pad, 5), title, fill=(0, 0, 0))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_png))


def analyze_with_ffmpeg(
    video_path: Path,
    out_dir: Path,
    sensitivity: float,
    keep_wav: bool,
    use_postfilter: bool = True,
    *,
    write_envelope_png: bool = True,
) -> tuple[dict[str, Any], Path | None, Path | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prob = probe_audio_with_ffmpeg(video_path)
    print(f"[probe] 摘要: {prob}", flush=True)
    if not prob.get("has_audio"):
        return {"ok": False, "reason": "no_audio_stream", "probe": prob}, None, None

    wav = out_dir / f"{video_path.stem}.wav"
    extract_audio_wav(video_path, wav)
    x, sr = load_wav_f32(wav)
    print_pcm_stats(x, sr, label="WAV")
    t_ds, env_ds, events = run_detection_pcm(
        x,
        sr,
        sensitivity=sensitivity,
        postfilter=use_postfilter,
    )
    plab = f" + post({use_postfilter})" if use_postfilter else " raw"
    if write_envelope_png:
        png = out_dir / f"{video_path.stem}.envelope.png"
        save_debug_png(
            t_ds,
            env_ds,
            events,
            png,
            title=f"{video_path.name}  band(2-8kHz) envelope + peaks  sens={sensitivity}{plab}",
        )
    else:
        png = None
    if not keep_wav:
        try:
            wav.unlink()
        except OSError:
            pass
    return (
        {
            "ok": True,
            "video": str(video_path),
            "sr": sr,
            "n_events": len(events),
            "events": [e.__dict__ for e in events],
        },
        png,
        None if not keep_wav else wav,
    )


def detect_squeak_from_video(
    video_path: Path,
    *,
    out_dir: Path | None = None,
    sensitivity: float = 1.0,
    use_postfilter: bool = True,
    keep_wav: bool = False,
    write_envelope_png: bool = True,
) -> tuple[dict[str, Any], Path | None, Path | None]:
    """
    供 `duck_detector`、其它脚本**接入**的入口：与 CLI 同一条 ffmpeg→PCM→检测
    管线，返回同 ``analyze_with_ffmpeg`` 的 (summary, png 路径或 None, wav 路径或 None)。"""
    od = out_dir or Path(__file__).resolve().parent / "squeak_debug"
    return analyze_with_ffmpeg(
        video_path,
        od,
        sensitivity,
        keep_wav,
        use_postfilter=use_postfilter,
        write_envelope_png=write_envelope_png,
    )


def _legacy_analyze_with_moviepy(video_path: Path, sensitivity: float) -> list[dict[str, Any]]:
    """Previous sparse-sampling path (kept for emergency compare only)."""
    from moviepy.editor import VideoFileClip

    clip = VideoFileClip(str(video_path))
    if not clip.audio:
        return []
    duration = float(clip.duration)
    step = 0.05
    t = np.arange(0, duration, step)
    hfe = []
    for ti in t:
        try:
            f = clip.audio.get_frame(float(ti))
            if isinstance(f, np.ndarray) and f.size > 1:
                hfe.append(float(np.sqrt(np.mean(np.diff(f.ravel()) ** 2))))
            else:
                hfe.append(0.0)
        except Exception:
            hfe.append(0.0)
    clip.close()
    hfe = np.array(hfe, dtype=np.float64)
    thr = float(hfe.mean() + (hfe.std() * (2.0 / max(sensitivity, 0.1))))
    peaks = np.where(hfe > thr)[0]
    return [{"t_peak": float(t[i])} for i in peaks[:50]]


def main() -> None:
    p = argparse.ArgumentParser(description="尖叫声检测（ffmpeg PCM 优先）")
    p.add_argument("video", type=Path, help="输入 MP4/MOV 等")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "squeak_debug",
        help="调试输出目录（WAV/PNG/JSON）",
    )
    p.add_argument("--sensitivity", type=float, default=1.0, help="越小越敏感，默认 1.0")
    p.add_argument("--keep-wav", action="store_true", help="保留解出的中间 WAV")
    p.add_argument("--legacy-moviepy", action="store_true", help="只跑旧版 moviepy 检测（调试用）")
    p.add_argument(
        "--no-postfilter",
        action="store_true",
        help="跳过后处理（不合并、不用 crest/band_ratio 质量门限；只保留包络直检峰，调试用）",
    )
    p.add_argument("--json", type=Path, default=None, help="将事件写 JSON 路径")
    args = p.parse_args()

    video = args.video.expanduser()
    if not video.is_file():
        raise SystemExit(f"文件不存在: {video}")

    out_dir = args.out_dir.expanduser()
    print("=" * 60, flush=True)
    print("squeak_detector: ffmpeg -> PCM -> band(2-8kHz) -> envelope peaks", flush=True)
    print("=" * 60, flush=True)

    if args.legacy_moviepy:
        ev = _legacy_analyze_with_moviepy(video, args.sensitivity)
        print(json.dumps({"ok": True, "legacy_moviepy": True, "n": len(ev), "events": ev}, indent=2, ensure_ascii=False))
        return

    summary, png, kept = analyze_with_ffmpeg(
        video,
        out_dir,
        args.sensitivity,
        args.keep_wav,
        use_postfilter=not args.no_postfilter,
        write_envelope_png=True,
    )
    print(f"[result] 事件数: {summary.get('n_events', 0)}", flush=True)
    if summary.get("events"):
        for i, e in enumerate(summary["events"][:20], 1):
            print(
                f"  #{i:02d} peak={e['t_peak']:.3f}s  start={e['t_start']:.3f}s  end={e['t_end']:.3f}s  score={e['score']:.6f}",
                flush=True,
            )
    if png:
        print(f"[out] 包络图: {png}", flush=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[out] JSON: {args.json}", flush=True)
    if args.keep_wav and kept is not None:
        print(f"[out] 保留 WAV: {kept}", flush=True)
    if not summary.get("ok"):
        print("未得到可用音频/PCM，已在上文打印 probe 信息。", file=sys.stderr)


if __name__ == "__main__":
    main()
