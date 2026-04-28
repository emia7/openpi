#!/usr/bin/env python3
"""
本脚本**只**用 OpenCV 逐帧显示画面；不经过系统扬声器播放音轨，因此
「看得见画面但听不见声音」是正常现象，不是文件一定无声。

要验证音轨，请用：
1) 先抽 WAV + 打统计
   python umi_scripts_csw/play_video_check_audio.py <video> --extract-only
2)（macOS）抽轨后用扬声器试听中间 WAV
   python umi_scripts_csw/play_video_check_audio.py <video> --afplay

nano 长录后如需按「开始录制 / 停止录制」做句级切分，见 README「nano_sync」与
``nano_sync/segment_by_record_markers.py``（百炼 filetrans + 公网 ``--file-url``）。

仅播放画面（不播放系统音频）时：
  python umi_scripts_csw/play_video_check_audio.py <video>
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import imageio.v3 as iio
import numpy as np

_UMI = Path(__file__).resolve().parent
if str(_UMI) not in sys.path:
    sys.path.insert(0, str(_UMI))

from audio_extract import (
    extract_audio_wav,
    find_ffmpeg,
    load_wav_f32,
    print_pcm_stats,
    probe_audio_with_ffmpeg,
)


def play_video_with_audio_vis(video_path: str) -> None:
    try:
        import cv2
    except ImportError as e:
        raise SystemExit("需要 opencv-python 才能弹窗播画面: pip/uv 安装 opencv-python") from e

    video_file = Path(video_path)
    if not video_file.exists():
        print(f"文件不存在: {video_path}", file=sys.stderr)
        return

    print(f"正在仅显示画面: {video_file.name}")
    print("提示: 本模式**不会**从电脑扬声器播音频；需要听声音请用 --afplay 或外置播放器。")

    reader = iio.imiter(str(video_path))
    meta = iio.immeta(str(video_path))
    fps = float(meta.get("fps", 30) or 30.0)
    if fps < 1e-3:
        fps = 30.0
    print(f"FPS(用于叠字, 近似): {fps}")

    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    frame_idx = 0
    paused = False
    print("\n按键: 空格 暂停/继续 | ESC 退出\n", flush=True)

    for frame in reader:
        if not paused:
            frame_idx += 1
            current_time = float(frame_idx) / fps
            frame_array = np.array(frame, dtype=np.uint8)
            cv2.putText(
                frame_array,
                f"Time(approx): {current_time:.2f}s  no_system_audio",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame_array,
                f"Frame: {frame_idx}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Video", frame_array)
        key = cv2.waitKey(max(int(1000.0 / fps), 1)) & 0xFF
        if key == 27:
            break
        if key == 32:
            paused = not paused
            print("暂停" if paused else "继续", flush=True)

    cv2.destroyAllWindows()
    print("\n播放结束(仅视频窗口)。")


def run_extract_and_stats(video: Path) -> Path:
    print("=" * 60, flush=True)
    print("用 ffmpeg 抽音（imageio-ffmpeg 自带或 PATH 的 ffmpeg）")
    print("ffmpeg =", find_ffmpeg(), flush=True)
    print("=" * 60, flush=True)
    d = Path(tempfile.mkdtemp(prefix="audio_check_"))
    wav = d / "extracted.wav"
    p = probe_audio_with_ffmpeg(video)
    print(f"[probe] {p}", flush=True)
    if not p.get("has_audio"):
        raise RuntimeError("未从 ffmpeg 输出中解析到音轨。若文件有声音，请用 ffmpeg 手工检查。")
    extract_audio_wav(video, wav, sample_rate=44100, mono=True)
    x, sr = load_wav_f32(wav)
    print_pcm_stats(x, sr, label="ffmpeg_WAV")
    return wav


def main() -> None:
    parser = argparse.ArgumentParser(
        description="仅 OpenCV 显示=无扬声器音频; 用 --extract-only / --afplay 验证音轨",
    )
    parser.add_argument("video", type=Path, help="视频文件路径")
    parser.add_argument("--extract-only", action="store_true", help="只抽音轨到临时 WAV 并打印统计，不弹窗")
    parser.add_argument(
        "--afplay",
        action="store_true",
        help="抽轨后用 macOS afplay 播完整 WAV(仅 macOS)",
    )
    args = parser.parse_args()
    video = args.video.expanduser()
    if not video.is_file():
        raise SystemExit(f"文件不存在: {video}")

    if args.extract_only or args.afplay:
        wav = run_extract_and_stats(video)
        if args.afplay and sys.platform == "darwin":
            print("afplay 开始播放（若太长可用 Ctrl+C 停）", flush=True)
            os.system(f"afplay {os.path.abspath(wav)}")
        elif args.afplay:
            print("非 macOS，无 afplay；WAV:", wav, file=sys.stderr)
        return

    print("=" * 60, flush=True)
    print("即将弹 OpenCV(无扬声器); 要统计音轨加 --extract-only", flush=True)
    print("=" * 60, flush=True)
    play_video_with_audio_vis(str(video))


if __name__ == "__main__":
    main()
