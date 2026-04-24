#!/usr/bin/env python3
"""
分析音频能量分布，找到合适的阈值
"""

import numpy as np
import sys
import os

try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip


def analyze_audio(video_path):
    print(f"加载视频: {video_path}")
    video = VideoFileClip(video_path)
    
    print(f"视频时长: {video.duration:.2f}s")
    print("提取音频...")
    
    audio = video.audio
    if audio is None:
        print("[ERROR] 没有音频")
        return
    
    # 提取音频
    sr = 16000
    y = audio.to_soundarray(fps=sr)
    if y.ndim > 1:
        y = y.mean(axis=1)
    
    video.close()
    
    # 计算能量包络
    window_ms = 10
    window_size = int(sr * window_ms / 1000)
    hop_size = window_size // 2
    
    energy = []
    for i in range(0, len(y) - window_size, hop_size):
        frame = y[i:i+window_size]
        e = np.sum(frame ** 2)
        energy.append(e)
    
    energy = np.array(energy)
    
    # 分析能量分布
    print()
    print("=" * 60)
    print("音频能量分析")
    print("=" * 60)
    print(f"总样本数: {len(energy)}")
    print(f"原始能量统计:")
    print(f"  最小值: {np.min(energy):.6f}")
    print(f"  最大值: {np.max(energy):.6f}")
    print(f"  均值: {np.mean(energy):.6f}")
    print(f"  中位数: {np.median(energy):.6f}")
    print(f"  标准差: {np.std(energy):.6f}")
    
    # 百分位数
    percentiles = [50, 90, 95, 99, 99.5, 99.9]
    print()
    print("能量百分位数:")
    for p in percentiles:
        val = np.percentile(energy, p)
        print(f"  {p}%: {val:.6f}")
    
    # 归一化后的分布
    energy_norm = energy / np.max(energy)
    print()
    print("归一化后能量分布:")
    for p in percentiles:
        val = np.percentile(energy_norm, p)
        print(f"  {p}%: {val:.4f}")
    
    # 找潜在的峰值（前100个最大能量）
    top_indices = np.argsort(energy)[-100:][::-1]
    top_times = top_indices * hop_size / sr
    
    print()
    print("能量最高的100个时刻:")
    for i, (idx, t) in enumerate(zip(top_indices[:20], top_times[:20]), 1):
        e = energy_norm[idx]
        print(f"  {i}. @ {t:.2f}s: {e:.4f}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("使用: python analyze_audio.py <video_file>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    analyze_audio(video_path)
