#!/usr/bin/env python3
"""
从视频/音轨检测双标定音并切片（不依赖 ASR）。

示例（在 umi_scripts_csw 下）::

  python nano_sync_freq/segment_by_freq_markers.py video.mp4 --cut-dir ./out_clips
  python nano_sync_freq/segment_by_freq_markers.py video.mp4 --markers-in markers.json --cut-dir ./out_clips
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np
import subprocess

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from audio_extract import extract_audio_wav, find_ffmpeg  # noqa: E402
from nano_sync_freq import config  # noqa: E402
from nano_sync_freq.detect_tones import (  # noqa: E402
    detect_wav,
    refine_starts_for_unpaired_stops,
    resample_mono,
)
from nano_sync_freq.pair_markers import (  # noqa: E402
    apply_stop_beep_tail,
    pair_start_stop_times,
    pair_to_jsonable,
)
from nano_sync_freq.tone_gen import (  # noqa: E402
    default_asset_paths,
    ensure_default_assets,
    read_wav_f32,
    stop_template_duration_sec,
)


def _ffprobe_duration(path: Path) -> float:
    if shutil.which("ffprobe") is None:
        return 0.0
    p = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nk=1:nw=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if p.returncode != 0 or not (p.stdout or "").strip():
        return 0.0
    try:
        return float((p.stdout or "").strip())
    except ValueError:
        return 0.0


def _cut_ffmpeg(in_media: Path, out_path: Path, t0: float, t1: float) -> None:
    ff = find_ffmpeg()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    d = t1 - t0
    if d <= 0:
        raise ValueError("切片时长须为正")
    cmd = [
        ff,
        "-y",
        "-ss",
        f"{t0:.3f}",
        "-i",
        str(in_media),
        "-t",
        f"{d:.3f}",
        "-c",
        "copy",
        str(out_path),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(
            f"ffmpeg 失败: {p.stderr[-2000:] if p.stderr else p.stdout or 'unknown'}"
        )


def _media_duration(path: Path, det: dict[str, Any]) -> float:
    d = float(det.get("duration_sec", 0.0) or 0.0)
    if d > 0:
        return d
    if path.suffix.lower() in (".wav", ".wave"):
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / max(w.getframerate(), 1)
    d2 = _ffprobe_duration(path)
    if d2 > 0:
        return d2
    return 0.0


def _enrich_manual_review_for_clips(
    manual: list[dict[str, Any]],
    clips: list[Any],
    *,
    eps: float = 0.02,
) -> list[dict[str, Any]]:
    """
    将 ``manual_review`` 里每条时刻 ``t`` 与已应用尾音后的 ``pairing`` 切片对应：
    ``in_clip_index`` / ``in_clip_basename`` 表示 t 是否落在某段内；否则可标间隙 ``in_gap_*`` + ``note``。
    """
    if not manual:
        return []
    out: list[dict[str, Any]] = []
    if not clips:
        for m in manual:
            d = dict(m)
            d["in_clip_index"] = None
            d["in_clip_basename"] = None
            d["in_gap_after_clip"] = None
            d["in_gap_before_clip"] = None
            d["note"] = None
            out.append(d)
        return out
    sorted_c = sorted(clips, key=lambda c: int(c.index))
    for m in manual:
        d = dict(m)
        t = float(m.get("t", 0.0))
        d["in_clip_index"] = None
        d["in_clip_basename"] = None
        d["in_gap_after_clip"] = None
        d["in_gap_before_clip"] = None
        d["note"] = None
        for c in sorted_c:
            t0, t1 = float(c.t_start), float(c.t_end)
            if t0 - eps <= t <= t1 + eps:
                d["in_clip_index"] = int(c.index)
                d["in_clip_basename"] = f"clip_{c.index:04d}"
                break
        if d["in_clip_index"] is not None:
            out.append(d)
            continue
        for k in range(len(sorted_c) - 1):
            a, b = sorted_c[k], sorted_c[k + 1]
            if float(a.t_end) - eps < t < float(b.t_start) + eps:
                d["in_gap_after_clip"] = int(a.index)
                d["in_gap_before_clip"] = int(b.index)
                d["note"] = (
                    f"在 clip_{a.index:04d} 与 clip_{b.index:04d} 时间间隙 (t={t:.3f}s)"
                )
                break
        out.append(d)
    return out


def _load_mono_48k(path: Path) -> np.ndarray:
    if path.suffix.lower() in (".wav", ".wave"):
        y, sr = read_wav_f32(path)
        return resample_mono(y, sr, config.SAMPLE_RATE_HZ)
    with tempfile.TemporaryDirectory() as td:
        tw = Path(td) / "t.wav"
        extract_audio_wav(path, tw, sample_rate=config.SAMPLE_RATE_HZ, mono=True)
        y, sr = read_wav_f32(tw)
        return resample_mono(y, sr, config.SAMPLE_RATE_HZ)


def _run_detect(
    path: Path,
    *,
    snr_cap: float | None,
    apply_sequence_prior: bool | None = None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tdir:
        tdirp = Path(tdir)
        tw = tdirp / "t.wav"
        if path.suffix.lower() in (".mp4", ".mov", ".m4v", ".webm", ".mkv", ".m4a"):
            extract_audio_wav(
                path, tw, sample_rate=config.SAMPLE_RATE_HZ, mono=True
            )
        elif path.suffix.lower() in (".wav", ".wave"):
            return detect_wav(
                path,
                snr_cap=snr_cap,
                apply_sequence_prior=apply_sequence_prior,
            )
        else:
            raise SystemExit("不支持的输入，请用 mp4/mov/webm/m4a 或 wav")
        return detect_wav(
            tw,
            snr_cap=snr_cap,
            apply_sequence_prior=apply_sequence_prior,
        )


def main() -> None:
    import argparse

    ensure_default_assets()
    ap = argparse.ArgumentParser(description="双标定音检测 + 视频切片")
    ap.add_argument("path", type=Path, help="含音轨的 mp4 或 .wav")
    ap.add_argument("--cut-dir", type=Path, default=None, help="输出 clip_0000.* 等")
    ap.add_argument(
        "--markers-in", type=Path, default=None, help="已检测 JSON（有 start/stop 则跳过检测）"
    )
    ap.add_argument("--markers-out", type=Path, default=None, help="写出完整结果 JSON")
    ap.add_argument(
        "--write-markers-only", action="store_true", help="只检测/配对并写 --markers-out"
    )
    ap.add_argument(
        "--no-stop-tail",
        action="store_true",
        help="片尾只切到「停」标起音，不包含整段结束 chirp（默认会延长约一帧标音长）",
    )
    ap.add_argument(
        "--stop-tail-sec",
        type=float,
        default=None,
        help="覆盖 config 的尾音秒数；默认与 stop 模板等长",
    )
    ap.add_argument(
        "--no-peak-snr-cap",
        action="store_true",
        help="不限制检峰门限上界（恢复旧版：仅 max_corr×PEAK_SNR_RATIO）",
    )
    ap.add_argument(
        "--peak-snr-cap",
        type=float,
        default=None,
        help="覆盖 config.PEAK_SNR_CAP；避免首段过强使后段标音漏检",
    )
    ap.add_argument(
        "--no-weak-start-refine",
        action="store_true",
        help="当出现「多一个停、少一个起」时，不再在局部窗口补扫弱 start",
    )
    ap.add_argument(
        "--no-sequence-prior",
        action="store_true",
        help="关闭 S/T 交替先验（多馀停丢弃、近阈值进人工表）；与 config 默认相反时显式关",
    )
    args = ap.parse_args()
    path = args.path.expanduser()
    if not path.is_file():
        raise SystemExit(f"file not found: {path}")
    if args.markers_in and args.markers_in.is_file():
        with open(args.markers_in, encoding="utf-8") as f:
            det: dict[str, Any] = json.load(f)
    else:
        snr_cap: float | None
        if args.no_peak_snr_cap:
            snr_cap = None
        else:
            snr_cap = (
                float(args.peak_snr_cap)
                if args.peak_snr_cap is not None
                else config.PEAK_SNR_CAP
            )
        prior_arg: bool | None = False if args.no_sequence_prior else None
        det = _run_detect(
            path, snr_cap=snr_cap, apply_sequence_prior=prior_arg
        )
    st = [float(x) for x in det.get("start_times", [])]
    stp = [float(x) for x in det.get("stop_times", [])]
    dur = _media_duration(path, det)
    refine_notes: list[str] = []
    if (
        config.REFINE_UNPAIRED_STARTS
        and not args.no_weak_start_refine
        and not (args.markers_in and args.markers_in.is_file())
    ):
        pr_probe = pair_start_stop_times(
            st, stp, duration_sec=dur if dur > 0.0 else None
        )
        if any(w.startswith("unpaired_stop:") for w in pr_probe.warnings):
            y = _load_mono_48k(path)
            p_st = Path(default_asset_paths()["start"])
            ts, srt = read_wav_f32(p_st)
            ts = np.ascontiguousarray(
                resample_mono(ts, srt, config.SAMPLE_RATE_HZ), dtype=np.float64
            )
            st, refine_notes = refine_starts_for_unpaired_stops(
                y, config.SAMPLE_RATE_HZ, ts, st, stp, dur if dur > 0.0 else None
            )
    pr = pair_start_stop_times(
        st, stp, duration_sec=dur if dur > 0.0 else None
    )
    if refine_notes:
        pr.warnings = list(pr.warnings) + list(refine_notes)
    tail_sec = 0.0 if args.no_stop_tail else (
        float(args.stop_tail_sec)
        if args.stop_tail_sec is not None
        else (config.INCLUDE_STOP_BEEP_TAIL_SEC
              if config.INCLUDE_STOP_BEEP_TAIL_SEC is not None
              else stop_template_duration_sec())
    )
    pr = apply_stop_beep_tail(
        pr, tail_sec=tail_sec, duration_sec=dur if dur > 0.0 else None
    )
    mr_raw = list(det.get("manual_review", []))
    out_payload: dict[str, Any] = {
        "file": str(path.resolve()),
        "duration_sec": dur,
        "sample_rate": det.get("sample_rate", config.SAMPLE_RATE_HZ),
        "start_times": st,
        "stop_times": stp,
        "stop_beep_tail_sec": tail_sec,
        "pairing": pair_to_jsonable(pr),
        "process_warnings": pr.warnings,
        "manual_review": _enrich_manual_review_for_clips(mr_raw, pr.clips),
        "sequence_prior_log": list(det.get("sequence_prior_log", [])),
    }
    if not (args.markers_in and args.markers_in.is_file()):
        out_payload["peak_snr_cap"] = (
            None
            if args.no_peak_snr_cap
            else (
                float(args.peak_snr_cap)
                if args.peak_snr_cap is not None
                else config.PEAK_SNR_CAP
            )
        )
        if refine_notes:
            out_payload["refine_unpaired_starts"] = refine_notes
    if args.markers_out is not None:
        args.markers_out = args.markers_out.expanduser()
        args.markers_out.parent.mkdir(parents=True, exist_ok=True)
        full = {**out_payload, "raw_detection": det}
        with open(args.markers_out, "w", encoding="utf-8") as f:
            json.dump(
                full, f, ensure_ascii=False, indent=2, default=str
            )
        print(f"已写: {args.markers_out}", flush=True)
    if args.write_markers_only:
        if args.markers_out is None:
            print(json.dumps(out_payload, ensure_ascii=False, indent=2, default=str))
        return
    if not args.cut_dir:
        print(json.dumps(out_payload, ensure_ascii=False, indent=2, default=str))
        return
    out_root = args.cut_dir.expanduser()
    ext = path.suffix.lower() if path.suffix else ".mp4"
    if ext not in (".mp4", ".mov", ".webm", ".mkv", ".wav", ".m4a"):
        ext = ".mp4"
    for c in pr.clips:
        oname = f"clip_{c.index:04d}{ext}"
        _cut_ffmpeg(path, out_root / oname, c.t_start, c.t_end)
    print(
        f"[ffmpeg] 已写 {len(pr.clips)} 个文件到 {out_root.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
