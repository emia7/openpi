#!/usr/bin/env python3
"""
从 Nano 视频中检测蜂鸣声标记（moviepy 版本，无需 ffmpeg 命令行）

开始音：1000Hz，0.1s，单声
结束音：1500Hz，0.1s×2，间隔0.2s

使用:
    python detect_beep_moviepy.py <video_file>
    
输出:
    - 打印检测到的标记
    - 生成 <video>_marks.json 文件
"""

import numpy as np
import json
import sys
import os
from scipy.signal import butter, filtfilt, find_peaks

def extract_audio_with_moviepy(video_path):
    """
    使用 moviepy 从视频提取音频
    """
    try:
        from moviepy import VideoFileClip
    except ImportError:
        try:
            from moviepy.editor import VideoFileClip
        except ImportError:
            print("[ERROR] moviepy 未安装")
            print("安装命令: pip3 install moviepy")
            sys.exit(1)
    
    print(f"[1/5] 加载视频: {video_path}")
    video = VideoFileClip(video_path)
    
    print(f"      视频时长: {video.duration:.2f}s")
    print(f"      视频 FPS: {video.fps}")
    
    print("[2/5] 提取音频...")
    audio = video.audio
    
    if audio is None:
        print("[ERROR] 视频没有音频轨道")
        sys.exit(1)
    
    # 采样音频（降低采样率以加快处理）
    sr = 16000  # 16kHz 足够检测 1000/1500Hz
    
    # 获取音频数据
    print(f"      采样率: {sr}Hz")
    
    # 直接获取整个音频
    print(f"      正在提取音频数据（这可能需要几分钟）...")
    
    # 直接获取整个音频数据
    y = audio.to_soundarray(fps=sr)
    
    # 如果是立体声，转为单声道
    if y.ndim > 1:
        y = y.mean(axis=1)
    
    print(f"      音频数据提取完成: {len(y)} 样本")
    
    # 归一化
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    
    video.close()
    
    return y, sr


def bandpass_filter(y, sr, low_freq, high_freq):
    """带通滤波"""
    nyquist = sr / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    b, a = butter(4, [low, high], btype='band')
    y_filtered = filtfilt(b, a, y)
    
    return y_filtered


def compute_short_time_energy(y, sr, frame_duration=0.02, hop_duration=0.01):
    """计算短时能量"""
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
    """检测视频中的蜂鸣声标记"""
    # 提取音频
    y, sr = extract_audio_with_moviepy(video_path)
    duration = len(y) / sr
    print(f"      音频总时长: {duration:.2f}s")
    
    print("[3/5] 带通滤波...")
    # 开始音频段滤波: 800-1200Hz (中心1000Hz)
    y_1k = bandpass_filter(y, sr, 800, 1200)
    # 结束音频段滤波: 1300-1700Hz (中心1500Hz)
    y_1_5k = bandpass_filter(y, sr, 1300, 1700)
    
    print("[4/5] 计算短时能量...")
    energy_1k, times = compute_short_time_energy(y_1k, sr)
    energy_1_5k, _ = compute_short_time_energy(y_1_5k, sr)
    
    # 归一化能量
    if np.max(energy_1k) > 0:
        energy_1k = energy_1k / np.max(energy_1k)
    if np.max(energy_1_5k) > 0:
        energy_1_5k = energy_1_5k / np.max(energy_1_5k)
    
    print("[5/5] 检测峰值...")
    marks = []
    
    # 检测开始音 (1000Hz 单峰值) - 使用中等阈值 + 宽度约束
    min_peak_distance = int(0.25 * 100)  # 0.25s 最小间隔
    peaks_1k, props_1k = find_peaks(
        energy_1k,
        height=0.4,  # 中等阈值
        distance=min_peak_distance,
        width=[2, 15]  # 峰宽约束：2-15个样本（约0.02-0.15s，匹配0.1s蜂鸣）
    )
    
    print(f"      1000Hz 频段候选峰值: {len(peaks_1k)} 个")
    
    # 检测结束音 (1500Hz 双峰值) - 使用中等阈值 + 宽度约束
    peaks_1_5k, props_1_5k = find_peaks(
        energy_1_5k,
        height=0.4,  # 中等阈值
        distance=int(0.1 * 100),  # 最小间隔 0.1s
        width=[2, 15]  # 峰宽约束
    )
    
    print(f"      1500Hz 频段候选峰值: {len(peaks_1_5k)} 个")
    
    print("      匹配模式...")
    
    # 识别开始音 - 提高判断阈值，避免误判结束音的第一声
    for peak in peaks_1k:
        time_1k = times[peak]
        confidence = energy_1k[peak]
        
        # 检查 1500Hz 频段附近是否有高能量
        window = int(0.15 * 100)  # 扩大检查窗口到 0.15s
        start_idx = max(0, peak - window)
        end_idx = min(len(energy_1_5k), peak + window)
        nearby_1_5k_energy = np.max(energy_1_5k[start_idx:end_idx]) if end_idx > start_idx else 0
        
        # 如果 1500Hz 附近能量很高 (>0.4)，可能是结束音的第一声，跳过
        if nearby_1_5k_energy > 0.4:
            continue
        
        marks.append({
            'type': 'start',
            'time': float(time_1k),
            'confidence': float(confidence),
            'freq_band': '1000Hz'
        })
    
    # 识别结束音 (1500Hz 双峰值) - 使用标准时间窗口
    i = 0
    while i < len(peaks_1_5k):
        peak1 = peaks_1_5k[i]
        time1 = times[peak1]
        conf1 = energy_1_5k[peak1]
        
        if i + 1 < len(peaks_1_5k):
            peak2 = peaks_1_5k[i + 1]
            time2 = times[peak2]
            conf2 = energy_1_5k[peak2]
            gap = time2 - time1
            
            # 严格检查间隔：0.18-0.25s（双蜂鸣间隔是 0.2s）
            # 同时要求两个峰值置信度都足够高
            if 0.18 <= gap <= 0.25 and conf1 >= 0.45 and conf2 >= 0.45:
                confidence = min(conf1, conf2)
                marks.append({
                    'type': 'end',
                    'time': float(time1),
                    'confidence': float(confidence),
                    'freq_band': '1500Hz',
                    'gap': float(gap)
                })
                i += 2
                continue
        
        i += 1
    
    # 按时间排序
    marks.sort(key=lambda x: x['time'])
    
    return marks


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用: python detect_beep_moviepy.py <video_file>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"[ERROR] 文件不存在: {video_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("蜂鸣声检测 (MoviePy 版本)")
    print("=" * 60)
    print(f"视频: {video_path}")
    print()
    
    # 检测标记
    marks = detect_beep_marks(video_path)
    
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
    
    # 显示前 20 个和后 20 个
    display_count = 20
    if len(marks) <= display_count * 2:
        for i, mark in enumerate(marks, 1):
            mark_type = '开始' if mark['type'] == 'start' else '结束'
            print(f"  {i}. [{mark_type}] @ {mark['time']:.3f}s (置信度: {mark['confidence']:.2f})")
    else:
        print(f"  显示前 {display_count} 个和后 {display_count} 个（共 {len(marks)} 个）：")
        print()
        for i, mark in enumerate(marks[:display_count], 1):
            mark_type = '开始' if mark['type'] == 'start' else '结束'
            print(f"  {i}. [{mark_type}] @ {mark['time']:.3f}s (置信度: {mark['confidence']:.2f})")
        print(f"  ... ({len(marks) - display_count * 2} 个省略) ...")
        print()
        for i, mark in enumerate(marks[-display_count:], len(marks) - display_count + 1):
            mark_type = '开始' if mark['type'] == 'start' else '结束'
            print(f"  {i}. [{mark_type}] @ {mark['time']:.3f}s (置信度: {mark['confidence']:.2f})")
    
    # 保存 JSON
    output_json = video_path.replace('.mp4', '_marks_moviepy.json')
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
    
    # 简单统计
    if start_count > 0 and end_count > 0:
        print()
        print("片段统计:")
        pairs = min(start_count, end_count)
        print(f"  预期片段数: {pairs}")
        
        if len(marks) >= 2:
            avg_duration = sum(
                marks[i+1]['time'] - marks[i]['time']
                for i in range(0, min(len(marks)-1, pairs*2-1), 2)
            ) / pairs if pairs > 0 else 0
            print(f"  平均片段时长: {avg_duration:.1f}s")


if __name__ == '__main__':
    main()
