#!/usr/bin/env python3
"""
检测脚踏板"咔哒"声（瞬态声音事件）

脚踏板特征：
- 短促的机械"咔哒"声
- 持续时间 < 0.1s
- 能量在时间上很尖锐（瞬态）
- 不依赖特定频率

使用:
    python detect_foot_pedal.py <video_file> [--threshold 0.5]
    
输出:
    - 打印检测到的踩踏事件
    - 生成 <video>_pedal_marks.json
"""

import numpy as np
import json
import sys
import os
from scipy.signal import find_peaks

try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip


def extract_audio(video_path, sr=16000):
    """提取音频"""
    print(f"[1/4] 加载视频: {video_path}")
    video = VideoFileClip(video_path)
    
    print(f"      视频时长: {video.duration:.2f}s")
    print("[2/4] 提取音频...")
    
    audio = video.audio
    if audio is None:
        print("[ERROR] 视频没有音频")
        sys.exit(1)
    
    # 提取音频数据
    y = audio.to_soundarray(fps=sr)
    if y.ndim > 1:
        y = y.mean(axis=1)  # 转为单声道
    
    # 归一化
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    
    video.close()
    print(f"      音频样本数: {len(y)}")
    return y, sr


def compute_energy_envelope(y, sr, window_ms=10):
    """
    计算能量包络（短时能量）
    
    window_ms: 窗口大小（毫秒），脚踏板声音短促，用小窗口
    """
    window_size = int(sr * window_ms / 1000)  # 样本数
    hop_size = window_size // 2  # 50% 重叠
    
    energy = []
    times = []
    
    for i in range(0, len(y) - window_size, hop_size):
        frame = y[i:i+window_size]
        # 平方能量
        e = np.sum(frame ** 2)
        energy.append(e)
        times.append(i / sr)
    
    energy = np.array(energy)
    times = np.array(times)
    
    # 归一化
    if np.max(energy) > 0:
        energy = energy / np.max(energy)
    
    return energy, times


def detect_pedal_events(y, sr, threshold=0.5, min_interval_ms=200):
    """
    检测脚踏板事件
    
    策略：
    1. 计算能量包络
    2. 找瞬态峰值（尖锐、短促）
    3. 过滤：峰值宽度要窄（<100ms），表示短促声音
    
    Args:
        threshold: 能量阈值 (0-1)
        min_interval_ms: 最小间隔（毫秒），避免重复检测
    """
    print("[3/4] 计算能量包络...")
    energy, times = compute_energy_envelope(y, sr, window_ms=10)
    
    print("[4/4] 检测瞬态峰值...")
    
    # 找峰值
    min_distance = int(min_interval_ms / 10)  # 转换为样本数（hop=5ms）
    
    peaks, properties = find_peaks(
        energy,
        height=threshold,
        distance=min_distance,
        width=[1, 10]  # 宽度约束：窄峰（5-100ms）
    )
    
    print(f"      检测到 {len(peaks)} 个候选峰值")
    
    # 构建事件列表
    events = []
    for i, peak in enumerate(peaks):
        time = times[peak]
        confidence = energy[peak]
        width = properties['widths'][i] if 'widths' in properties else 0
        width_ms = width * 5  # 转换为毫秒（hop=5ms）
        
        events.append({
            'time': float(time),
            'confidence': float(confidence),
            'width_ms': float(width_ms)
        })
    
    return events


def pair_events(events, max_gap_s=30):
    """
    将事件配对为开始-结束
    
    假设：奇数索引是开始，偶数索引是结束
    或者：根据时间间隔配对（如果间隔<30s则配对）
    """
    if len(events) < 2:
        return []
    
    pairs = []
    i = 0
    
    while i < len(events) - 1:
        start = events[i]
        end = events[i + 1]
        gap = end['time'] - start['time']
        
        # 如果间隔合理（< max_gap_s），视为一对
        if gap < max_gap_s:
            pairs.append({
                'start_time': start['time'],
                'end_time': end['time'],
                'duration': gap,
                'start_confidence': start['confidence'],
                'end_confidence': end['confidence']
            })
            i += 2
        else:
            # 间隔太长，可能是单独的事件
            i += 1
    
    return pairs


def main():
    if len(sys.argv) < 2:
        print("使用: python detect_foot_pedal.py <video_file> [threshold]")
        print("  threshold: 检测阈值 (默认 0.5, 范围 0-1)")
        print("  嘈杂环境建议: 0.6-0.8")
        sys.exit(1)
    
    video_path = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    
    if not os.path.exists(video_path):
        print(f"[ERROR] 文件不存在: {video_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("脚踏板声音检测")
    print("=" * 60)
    print(f"视频: {video_path}")
    print(f"阈值: {threshold}")
    print()
    
    # 提取音频
    y, sr = extract_audio(video_path)
    
    # 检测事件
    events = detect_pedal_events(y, sr, threshold=threshold)
    
    # 配对
    pairs = pair_events(events)
    
    # 打印结果
    print()
    print("=" * 60)
    print(f"检测结果: {len(events)} 个踩踏事件")
    print(f"配对片段: {len(pairs)} 个")
    print("=" * 60)
    
    if len(events) > 0:
        print()
        print("事件列表（前20个）:")
        for i, ev in enumerate(events[:20], 1):
            print(f"  {i}. @ {ev['time']:.2f}s  (强度: {ev['confidence']:.2f}, 宽度: {ev['width_ms']:.1f}ms)")
        
        if len(events) > 20:
            print(f"  ... 还有 {len(events)-20} 个 ...")
    
    if len(pairs) > 0:
        print()
        print("片段列表（前10个）:")
        for i, p in enumerate(pairs[:10], 1):
            print(f"  {i}. [{p['start_time']:.1f}s - {p['end_time']:.1f}s] 时长: {p['duration']:.1f}s")
        
        if len(pairs) > 10:
            print(f"  ... 还有 {len(pairs)-10} 个 ...")
        
        # 统计
        durations = [p['duration'] for p in pairs]
        print()
        print("片段统计:")
        print(f"  总数: {len(pairs)}")
        print(f"  平均时长: {np.mean(durations):.1f}s")
        print(f"  最短: {np.min(durations):.1f}s")
        print(f"  最长: {np.max(durations):.1f}s")
    
    # 保存结果
    output = {
        'video': video_path,
        'threshold': threshold,
        'total_events': len(events),
        'paired_segments': len(pairs),
        'events': events,
        'pairs': pairs
    }
    
    output_file = video_path.replace('.mp4', '_pedal_marks.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print()
    print(f"[OK] 结果已保存: {output_file}")
    print()
    print("提示:")
    print("  - 如果检测到太多事件，提高阈值（如 0.6, 0.7）")
    print("  - 如果漏检，降低阈值（如 0.3, 0.4）")
    print("  - 嘈杂环境建议阈值: 0.6-0.8")


if __name__ == '__main__':
    main()
