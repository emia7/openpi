#!/usr/bin/env python3
"""
脚踏板声音检测 v2 - 使用差分方法检测瞬态声音

针对"咔哒"机械声特征：
- 短促（<50ms）
- 能量突变（快速上升下降）
- 使用差分而非绝对阈值

使用:
    python detect_pedal_v2.py <video_file> [--sensitivity 3]
"""

import numpy as np
import json
import sys
import os

try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip


def extract_audio(video_path, sr=22050):
    """提取音频，使用更高采样率"""
    print(f"[1/3] 加载视频: {video_path}")
    video = VideoFileClip(video_path)
    
    print(f"      时长: {video.duration:.1f}s")
    print("[2/3] 提取音频...")
    
    audio = video.audio
    y = audio.to_soundarray(fps=sr)
    
    if y.ndim > 1:
        y = y.mean(axis=1)
    
    video.close()
    print(f"      样本数: {len(y)}, 时长: {len(y)/sr:.1f}s")
    return y, sr


def detect_transients(y, sr, sensitivity=3.0, min_gap_ms=300):
    """
    使用差分检测瞬态声音
    
    原理：咔哒声 = 能量的快速突变
    - 计算短时能量的变化率
    - 找突变点（上升沿和下降沿）
    """
    print("[3/3] 检测瞬态事件...")
    
    # 1. 短时能量（5ms窗口）
    window_ms = 5
    window = int(sr * window_ms / 1000)
    hop = window // 2
    
    energy = []
    times = []
    for i in range(0, len(y) - window, hop):
        e = np.sum(y[i:i+window]**2)
        energy.append(e)
        times.append(i / sr)
    
    energy = np.array(energy)
    times = np.array(times)
    
    # 归一化
    if np.max(energy) > 0:
        energy = energy / np.max(energy)
    
    # 2. 计算能量变化率（差分）
    diff = np.diff(energy)
    diff_times = times[1:]
    
    # 3. 找快速上升沿（咔哒声开始）
    # 使用标准差的倍数作为阈值
    mean_diff = np.mean(np.abs(diff))
    std_diff = np.std(diff)
    threshold = mean_diff + sensitivity * std_diff
    
    print(f"      差分统计: 均值={mean_diff:.6f}, 标准差={std_diff:.6f}")
    print(f"      检测阈值: {threshold:.6f} (敏感度={sensitivity})")
    
    # 找正峰值（上升沿）
    from scipy.signal import find_peaks
    peaks, props = find_peaks(
        diff,
        height=threshold,
        distance=int(min_gap_ms / 2.5),  # 最小间隔
        width=[1, 5]  # 窄峰
    )
    
    print(f"      检测到 {len(peaks)} 个瞬态事件")
    
    events = []
    for p in peaks:
        events.append({
            'time': float(diff_times[p]),
            'strength': float(diff[p]),
            'energy': float(energy[p+1])
        })
    
    return events


def pair_events(events, min_duration=2.0, max_duration=60.0):
    """配对为开始-结束"""
    pairs = []
    i = 0
    while i < len(events) - 1:
        t1 = events[i]['time']
        t2 = events[i+1]['time']
        duration = t2 - t1
        
        if min_duration <= duration <= max_duration:
            pairs.append({
                'start': t1,
                'end': t2,
                'duration': duration,
                'start_strength': events[i]['strength'],
                'end_strength': events[i+1]['strength']
            })
            i += 2
        else:
            i += 1
    
    return pairs


def main():
    if len(sys.argv) < 2:
        print("使用: python detect_pedal_v2.py <video> [sensitivity]")
        print("  sensitivity: 敏感度 (默认3.0, 范围1-10)")
        print("  值越大越敏感，越容易检测到")
        sys.exit(1)
    
    video = sys.argv[1]
    sensitivity = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    
    # 提取
    y, sr = extract_audio(video, sr=22050)
    
    # 检测
    events = detect_transients(y, sr, sensitivity=sensitivity)
    
    # 配对
    pairs = pair_events(events)
    
    # 输出
    print()
    print("=" * 60)
    print(f"检测完成: {len(events)} 个事件, {len(pairs)} 个片段")
    print("=" * 60)
    
    if events:
        print(f"\n前20个事件:")
        for i, e in enumerate(events[:20], 1):
            print(f"  {i}. @ {e['time']:.2f}s (强度: {e['strength']:.4f})")
    
    if pairs:
        print(f"\n前10个片段:")
        for i, p in enumerate(pairs[:10], 1):
            print(f"  {i}. [{p['start']:.1f}s - {p['end']:.1f}s] 时长: {p['duration']:.1f}s")
    
    # 保存
    result = {
        'video': video,
        'sensitivity': sensitivity,
        'events': events,
        'pairs': pairs
    }
    
    out = video.replace('.mp4', '_pedal_v2.json')
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n[OK] 结果保存: {out}")
    print(f"提示: 如检测不足，降低敏感度(如2.0,1.5); 如太多，提高(如5.0,8.0)")


if __name__ == '__main__':
    main()
