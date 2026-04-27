#!/usr/bin/env python3
"""
尖叫鸡/鸭子样「短促高频声」时间检测（与 squeak 管线一致）

**默认**使用同目录下 ``squeak_detector``：ffmpeg 抽 PCM → 2–8kHz 包络峰
+ 声学后处理，**不**用「片头/片尾剪秒」等时间先验。可选旧版 moviepy 粗能量
路径（--legacy-moviepy，不推荐）。

用法:
    python duck_detector.py <video_path>
    python duck_detector.py <video_path> --sensitivity 0.8
    python duck_detector.py <video_path> --legacy-moviepy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_UMI = Path(__file__).resolve().parent
if str(_UMI) not in sys.path:
    sys.path.insert(0, str(_UMI))


def detect_squeak_duck(
    video_path: str,
    *,
    out_dir: Path | None = None,
    sensitivity: float = 1.0,
    use_postfilter: bool = True,
) -> list[dict]:
    """用 squeak 管线；返回与旧 ``detect_duck_sound`` 兼容的列表结构。"""
    from squeak_detector import detect_squeak_from_video

    print("=" * 60, flush=True)
    print("duck_detector → squeak 管线 (ffmpeg PCM, 2–8kHz, 声学后处理)", flush=True)
    print("=" * 60, flush=True)

    video_file = Path(video_path)
    if not video_file.is_file():
        print(f"文件不存在: {video_path}", file=sys.stderr, flush=True)
        return []
    summary, png, _ = detect_squeak_from_video(
        video_file,
        out_dir=out_dir,
        sensitivity=sensitivity,
        use_postfilter=use_postfilter,
        keep_wav=False,
        write_envelope_png=True,
    )
    if not summary.get("ok") or "events" not in summary:
        if summary.get("reason") == "no_audio_stream":
            pass
        return []
    out: list[dict] = []
    for i, e in enumerate(summary["events"], 1):
        t0, t1, tp = e["t_start"], e["t_end"], e["t_peak"]
        out.append(
            {
                "index": i,
                "start": float(t0),
                "end": float(t1),
                "duration": float(t1) - float(t0),
                "t_peak": float(tp),
                "score": float(e.get("score", 0.0)),
                "avg_energy": float(e.get("score", 0.0)),  # 与旧版字段名兼容
            }
        )
    if png and Path(png).is_file():
        print(f"[duck] 包络调试图: {png}", flush=True)
    return out


def detect_duck_sound(video_path: str, threshold_percentile: int = 75) -> list[dict]:
    """[legacy] 基于 moviepy 粗能量/分位，稀疏采样。仅 ``--legacy-moviepy`` 时调用。"""

    print("=" * 60, flush=True)
    print("🦆 鸭子叫声检测器 [legacy moviepy 粗能量，不推荐]", flush=True)
    print("=" * 60, flush=True)

    video_file = Path(video_path)
    if not video_file.exists():
        print(f"✗ 文件不存在: {video_path}", file=sys.stderr)
        return []

    print(f"\n[1] 分析视频: {video_file.name}")
    print(f"    文件大小: {video_file.stat().st_size / 1024 / 1024:.1f} MB")

    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        print("\n✗ 需要安装 moviepy: pip install moviepy", file=sys.stderr)
        return []

    try:
        print(f"\n[2] 提取音频数据...")
        clip = VideoFileClip(str(video_file))

        duration = clip.duration
        fps = clip.audio.fps if hasattr(clip, "audio") and clip.audio else 44100

        print(f"    ✓ 视频时长: {duration:.2f} 秒")
        print(f"    ✓ 音频采样率: {fps} Hz")

        if not clip.audio:
            print("    ✗ 视频没有音频轨道")
            clip.close()
            return []

        print(f"\n[3] 采样音频能量...")
        sample_times = np.arange(0, duration, 0.1)
        energies: list[float] = []

        for t in sample_times:
            try:
                frame = clip.audio.get_frame(t)
                if isinstance(frame, np.ndarray):
                    energy = float(np.sqrt(np.mean(frame**2)))
                else:
                    energy = float(abs(frame))
                energies.append(energy)
            except OSError:
                energies.append(0.0)

        energies = np.array(energies, dtype=np.float64)
        clip.close()

        print(f"    采集了 {len(energies)} 个样本点")

        print(f"\n[4] 音频能量分析:")
        print(f"    平均能量: {np.mean(energies):.4f}")
        print(f"    最大能量: {np.max(energies):.4f}")
        print(f"    能量中位数: {np.median(energies):.4f}")

        threshold = float(np.percentile(energies, threshold_percentile))
        print(f"    检测阈值 ({threshold_percentile}分位): {threshold:.4f}")

        high_energy_indices = np.where(energies > threshold)[0]

        if len(high_energy_indices) == 0:
            print(f"\n    ✗ 未检测到明显的声音事件")
            return []

        print(f"\n[5] 检测声音事件...")

        events: list[list[int]] = []
        current_event = [int(high_energy_indices[0])]

        for j in range(1, len(high_energy_indices)):
            gap = int(high_energy_indices[j] - high_energy_indices[j - 1])
            if gap <= 3:
                current_event.append(int(high_energy_indices[j]))
            else:
                events.append(current_event)
                current_event = [int(high_energy_indices[j])]
        events.append(current_event)

        print(f"    找到 {len(events)} 个声音事件")

        duck_events: list[dict] = []
        print(f"\n[6] 鸭子叫声候选（持续0.3-1.5秒的高能量事件):")
        print("-" * 60)

        for i, event in enumerate(events, 1):
            start_idx = event[0]
            end_idx = event[-1]

            start_time = float(sample_times[start_idx])
            end_time = float(sample_times[end_idx])
            dur = end_time - start_time

            avg_energy = float(np.mean(energies[start_idx : end_idx + 1]))
            max_energy = float(np.max(energies[start_idx : end_idx + 1]))

            if 0.3 <= dur <= 1.5 and avg_energy > threshold * 0.8:
                duck_events.append(
                    {
                        "index": i,
                        "start": start_time,
                        "end": end_time,
                        "duration": dur,
                        "avg_energy": avg_energy,
                        "max_energy": max_energy,
                    }
                )

                print(f"  事件 {i}: {start_time:.2f}s - {end_time:.2f}s")
                print(f"           持续 {dur:.2f}s, 平均能量 {avg_energy:.4f}")
                print(f"           ✓ 疑似鸭子叫声")
                print()

        print("-" * 60)

        if len(duck_events) == 0:
            print(f"\n✗ 未找到符合鸭子叫声特征的事件")
        else:
            print(f"\n✓ 检测到 {len(duck_events)} 个疑似鸭子叫声")
            print(f"\n建议的分割时间点:")
            for fe in duck_events:
                print(f"   {fe['start']:.2f}s (第 {fe['index']} 个事件)")

        return duck_events

    except Exception as e:
        print(f"\n✗ 分析失败: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return []


def _print_squeak_style(events: list[dict], video: Path) -> None:
    print("=" * 60, flush=True)
    print("🦆 尖叫/短促声 (2–8kHz, squeak 管线) ", flush=True)
    print("=" * 60, flush=True)
    print(f"视频: {video.name}\n", flush=True)
    if not events:
        print("未检测到事件（可试调 --sensitivity 或看 squeak 包络图）", flush=True)
        return
    print(f"共 {len(events)} 个\n", flush=True)
    for e in events:
        print(
            f"  #{e['index']:02d}  peak={e['t_peak']:.3f}s  "
            f"start={e['start']:.3f}s  end={e['end']:.3f}s  score={e['score']:.4f}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="检测视频中尖叫鸡/类似短促声时刻（默认 squeak 管线，可选 legacy moviepy）"
    )
    parser.add_argument("video", type=Path, help="视频文件路径")
    parser.add_argument(
        "--legacy-moviepy",
        action="store_true",
        help="使用旧版粗能量+moviepy（稀疏采样，不推荐；仅对比）",
    )
    parser.add_argument(
        "--threshold", type=int, default=75, help="[legacy] 能量分位阈值，越大越严"
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=1.0,
        help="[squeak] 包络直检：越小越敏感、峰越多。默认 1.0",
    )
    parser.add_argument(
        "--no-postfilter",
        action="store_true",
        help="[squeak] 不做结构合并/声学后处理（直检调试用）",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="[squeak] 调试图/WAV 等输出目录，默认 umi_scripts_csw/squeak_debug",
    )
    args = parser.parse_args()
    video = args.video.expanduser()
    if not video.is_file():
        raise SystemExit(f"文件不存在: {video}")

    if args.legacy_moviepy:
        events = detect_duck_sound(str(video), args.threshold)
    else:
        out_dir = args.out_dir.expanduser() if args.out_dir else None
        events = detect_squeak_duck(
            str(video),
            out_dir=out_dir,
            sensitivity=args.sensitivity,
            use_postfilter=not args.no_postfilter,
        )
        _print_squeak_style(events, video)

    print("\n" + "=" * 60, flush=True)
    if not args.legacy_moviepy and len(events) > 0:
        print("检测完成。可分割片段或再录视频用 squeak/duck 同参对比。", flush=True)
    elif not args.legacy_moviepy and len(events) == 0:
        print("未检到事件。可: 1) 略降 --sensitivity 2) 加 --no-postfilter 看原峰 3) 用 --legacy-moviepy 粗对比", flush=True)
    elif len(events) == 0:
        print("未检测到鸭子叫声，可检查音轨、阈值或试 squeak 默认模式（去掉 --legacy-moviepy）", flush=True)
    else:
        print("检测完成！可以用这些时间点分割视频片段", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
