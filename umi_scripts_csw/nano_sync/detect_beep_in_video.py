#!/usr/bin/env python3
"""
从 Nano 视频中检测蜂鸣声标记

开始音：1000Hz，0.1s，单声
结束音：1500Hz，0.1s×2，间隔0.2s

使用:
    python detect_beep_in_video.py <video_file>
    
输出:
    - 打印检测到的标记
    - 生成 <video>_marks.json 文件
"""

import numpy as np
import subprocess
import json
import sys
import os
import tempfile
from scipy.signal import butter, filtfilt, find_peaks


def extract_audio(video_path, output_wav=None):
    """
    从视频提取音频为 WAV 格式
    
    Args:
        video_path: 输入视频路径
        output_wav: 输出 WAV 路径（为 None 则创建临时文件）
    
    Returns:
        output_wav 路径
    """
    if output_wav is None:
        # 创建临时文件
        fd, output_wav = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
    
    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-vn',                    # 不处理视频
        '-acodec', 'pcm_s16le',   # 16-bit PCM
        '-ar', '16000',           # 16kHz 采样率
        '-ac', '1',               # 单声道
        output_wav
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"[ERROR] ffmpeg 提取音频失败:")
            print(f"        {result.stderr}")
            sys.exit(1)
    except FileNotFoundError:
        print("[ERROR] 找不到 ffmpeg，请先安装:")
        print("        macOS: brew install ffmpeg")
        print("        Ubuntu: sudo apt-get install ffmpeg")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("[ERROR] ffmpeg 执行超时")
        sys.exit(1)
    
    return output_wav


def load_audio(wav_path):
    """
    加载 WAV 音频文件
    
    Returns:
        y: 音频数据 (numpy array)
        sr: 采样率
    """
    try:
        import wave
        with wave.open(wav_path, 'rb') as f:
            sr = f.getframerate()
            n_channels = f.getnchannels()
            n_frames = f.getnframes()
            
            # 读取数据
            raw_data = f.readframes(n_frames)
            y = np.frombuffer(raw_data, dtype=np.int16)
            
            # 如果是立体声，转为单声道
            if n_channels == 2:
                y = y.reshape(-1, 2).mean(axis=1)
            
            # 归一化到 [-1, 1]
            y = y.astype(np.float32) / 32768.0
            
        return y, sr
    except Exception as e:
        print(f"[ERROR] 加载音频失败: {e}")
        sys.exit(1)


def bandpass_filter(y, sr, low_freq, high_freq):
    """
    带通滤波
    
    Args:
        y: 音频数据
        sr: 采样率
        low_freq: 低频截止
        high_freq: 高频截止
    
    Returns:
        滤波后的音频
    """
    nyquist = sr / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # Butterworth 滤波器
    b, a = butter(4, [low, high], btype='band')
    y_filtered = filtfilt(b, a, y)
    
    return y_filtered


def compute_short_time_energy(y, sr, frame_duration=0.02, hop_duration=0.01):
    """
    计算短时能量
    
    Args:
        y: 音频数据
        sr: 采样率
        frame_duration: 帧长（秒）
        hop_duration: 帧移（秒）
    
    Returns:
        energy: 能量序列
        times: 对应的时间点
    """
    frame_length = int(sr * frame_duration)
    hop_length = int(sr * hop_duration)
    
    energy = []
    for i in range(0, len(y) - frame_length, hop_length):
        frame = y[i:i+frame_length]
        energy.append(np.sum(frame ** 2))
    
    energy = np.array(energy)
    times = np.arange(len(energy)) * hop_duration
    
    return energy, times


def detect_beep_marks(video_path, visualize=False):
    """
    检测视频中的蜂鸣声标记
    
    Args:
        video_path: 视频文件路径
        visualize: 是否可视化结果（调试用）
    
    Returns:
        list: [{'type': 'start'/'end', 'time': float, 'confidence': float}, ...]
    """
    print(f"[1/5] 提取音频: {video_path}")
    temp_wav = None
    try:
        temp_wav = extract_audio(video_path)
        y, sr = load_audio(temp_wav)
        duration = len(y) / sr
        print(f"      音频时长: {duration:.2f}s, 采样率: {sr}Hz")
    finally:
        # 清理临时文件
        if temp_wav and os.path.exists(temp_wav):
            os.remove(temp_wav)
    
    print("[2/5] 带通滤波...")
    # 开始音频段滤波: 800-1200Hz (中心1000Hz)
    y_1k = bandpass_filter(y, sr, 800, 1200)
    # 结束音频段滤波: 1300-1700Hz (中心1500Hz)
    y_1_5k = bandpass_filter(y, sr, 1300, 1700)
    
    print("[3/5] 计算短时能量...")
    energy_1k, times = compute_short_time_energy(y_1k, sr)
    energy_1_5k, _ = compute_short_time_energy(y_1_5k, sr)
    
    # 归一化能量
    if np.max(energy_1k) > 0:
        energy_1k = energy_1k / np.max(energy_1k)
    if np.max(energy_1_5k) > 0:
        energy_1_5k = energy_1_5k / np.max(energy_1_5k)
    
    print("[4/5] 检测峰值...")
    marks = []
    
    # 检测开始音 (1000Hz 单峰值，持续约0.1s)
    # 峰值间隔至少 0.3s，避免重复检测
    min_peak_distance = int(0.3 * 100)  # 0.3s / 0.01s_hop = 30 samples
    peaks_1k, props_1k = find_peaks(
        energy_1k,
        height=0.5,                    # 阈值 0.5
        distance=min_peak_distance
    )
    
    print(f"      1000Hz 频段候选峰值: {len(peaks_1k)} 个")
    
    # 检测结束音 (1500Hz 双峰值，间隔0.15-0.25s)
    peaks_1_5k, props_1_5k = find_peaks(
        energy_1_5k,
        height=0.5,
        distance=int(0.15 * 100)       # 最小间隔 0.15s
    )
    
    print(f"      1500Hz 频段候选峰值: {len(peaks_1_5k)} 个")
    
    print("[5/5] 匹配模式...")
    
    # 识别开始音：1000Hz 有峰值，1500Hz 附近无峰值
    for peak in peaks_1k:
        time_1k = times[peak]
        confidence = energy_1k[peak]
        
        # 检查 1500Hz 频段在附近是否有能量（避免把结束音的第一声当开始音）
        window = int(0.1 * 100)  # ±0.1s
        start_idx = max(0, peak - window)
        end_idx = min(len(energy_1_5k), peak + window)
        nearby_1_5k_energy = np.max(energy_1_5k[start_idx:end_idx]) if end_idx > start_idx else 0
        
        # 如果 1500Hz 附近能量也高，可能是结束音的第一声，跳过
        if nearby_1_5k_energy > 0.3:
            continue
        
        marks.append({
            'type': 'start',
            'time': float(time_1k),
            'confidence': float(confidence),
            'freq_band': '1000Hz'
        })
    
    # 识别结束音：1500Hz 双峰值，间隔 0.15-0.25s
    i = 0
    while i < len(peaks_1_5k):
        peak1 = peaks_1_5k[i]
        time1 = times[peak1]
        
        if i + 1 < len(peaks_1_5k):
            peak2 = peaks_1_5k[i + 1]
            time2 = times[peak2]
            gap = time2 - time1
            
            # 检查间隔是否在 0.15-0.25s 之间
            if 0.15 < gap < 0.25:
                confidence = min(energy_1_5k[peak1], energy_1_5k[peak2])
                marks.append({
                    'type': 'end',
                    'time': float(time1),  # 以第一声为准
                    'confidence': float(confidence),
                    'freq_band': '1500Hz',
                    'gap': float(gap)
                })
                i += 2  # 跳过两个峰值
                continue
        
        i += 1
    
    # 按时间排序
    marks.sort(key=lambda x: x['time'])
    
    # 可视化（调试用）
    if visualize:
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(3, 1, figsize=(12, 8))
            
            # 原始音频
            axes[0].plot(np.arange(len(y)) / sr, y)
            axes[0].set_ylabel('Amplitude')
            axes[0].set_title('Original Audio')
            
            # 1000Hz 能量
            axes[1].plot(times, energy_1k, label='1000Hz band')
            axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
            for mark in marks:
                if mark['type'] == 'start':
                    axes[1].axvline(x=mark['time'], color='g', linestyle='--', alpha=0.7)
                    axes[1].text(mark['time'], 0.9, f'S\n{mark["confidence"]:.2f}',
                               ha='center', fontsize=8)
            axes[1].set_ylabel('Normalized Energy')
            axes[1].set_title('1000Hz Band (Start Beep)')
            axes[1].legend()
            
            # 1500Hz 能量
            axes[2].plot(times, energy_1_5k, label='1500Hz band', color='orange')
            axes[2].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
            for mark in marks:
                if mark['type'] == 'end':
                    axes[2].axvline(x=mark['time'], color='r', linestyle='--', alpha=0.7)
                    axes[2].text(mark['time'], 0.9, f'E\n{mark["confidence"]:.2f}',
                               ha='center', fontsize=8)
            axes[2].set_ylabel('Normalized Energy')
            axes[2].set_xlabel('Time (s)')
            axes[2].set_title('1500Hz Band (End Beep)')
            axes[2].legend()
            
            plt.tight_layout()
            plt.savefig(video_path.replace('.mp4', '_detection.png'), dpi=150)
            print(f"[INFO] 可视化结果保存: {video_path.replace('.mp4', '_detection.png')}")
            plt.close()
        except ImportError:
            pass
    
    return marks


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用: python detect_beep_in_video.py <video_file> [--visualize]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    visualize = '--visualize' in sys.argv
    
    if not os.path.exists(video_path):
        print(f"[ERROR] 文件不存在: {video_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("蜂鸣声检测")
    print("=" * 60)
    print(f"视频: {video_path}")
    print()
    
    # 检测标记
    marks = detect_beep_marks(video_path, visualize=visualize)
    
    # 打印结果
    print()
    print("=" * 60)
    print(f"检测结果: {len(marks)} 个标记")
    print("=" * 60)
    
    start_count = sum(1 for m in marks if m['type'] == 'start')
    end_count = sum(1 for m in marks if m['type'] == 'end')
    
    print(f"开始音: {start_count} 个")
    print(f"结束音: {end_count} 个")
    print()
    
    for i, mark in enumerate(marks, 1):
        mark_type = '开始' if mark['type'] == 'start' else '结束'
        print(f"  {i}. [{mark_type}] @ {mark['time']:.3f}s (置信度: {mark['confidence']:.2f})")
    
    # 保存 JSON
    output_json = video_path.replace('.mp4', '_marks.json')
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump({
            'video_file': video_path,
            'total_marks': len(marks),
            'start_count': start_count,
            'end_count': end_count,
            'marks': marks
        }, f, indent=2, ensure_ascii=False)
    
    print()
    print(f"[OK] 结果已保存: {output_json}")


if __name__ == '__main__':
    main()
