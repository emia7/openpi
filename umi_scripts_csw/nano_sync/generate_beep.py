#!/usr/bin/env python3
"""
生成用于 Nano 同步的蜂鸣声音频文件

开始音：1000Hz，0.1s，单声，80% 音量
结束音：1500Hz，0.1s×2，间隔0.2s，80% 音量

使用:
    python generate_beep.py
    
输出:
    - start_beep.wav (开始音)
    - end_beep.wav (结束音)
"""

import numpy as np
import wave
import os


def generate_single_beep(filename, freq, duration, amplitude=0.8, sample_rate=44100):
    """
    生成单频率蜂鸣声
    
    Args:
        filename: 输出文件名
        freq: 频率 (Hz)
        duration: 时长 (秒)
        amplitude: 音量 (0-1)
        sample_rate: 采样率
    """
    # 生成正弦波
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave_data = amplitude * np.sin(2 * np.pi * freq * t)
    
    # 添加淡入淡出减少爆破音
    fade_samples = int(sample_rate * 0.01)  # 10ms 淡入淡出
    if fade_samples > 0 and len(wave_data) > fade_samples * 2:
        wave_data[:fade_samples] *= np.linspace(0, 1, fade_samples)
        wave_data[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    
    # 转换为 16-bit PCM
    wave_data = (wave_data * 32767).astype(np.int16)
    
    # 保存为 WAV
    with wave.open(filename, 'w') as f:
        f.setnchannels(1)  # 单声道
        f.setsampwidth(2)  # 16-bit
        f.setframerate(sample_rate)
        f.writeframes(wave_data.tobytes())
    
    print(f"生成: {filename}")
    print(f"  频率: {freq}Hz")
    print(f"  时长: {duration}s")
    print(f"  音量: {amplitude * 100}%")
    print(f"  采样率: {sample_rate}Hz")
    print()


def generate_double_beep(filename, freq, duration, gap, amplitude=0.8, sample_rate=44100):
    """
    生成双蜂鸣声（结束音）
    
    Args:
        filename: 输出文件名
        freq: 频率 (Hz)
        duration: 单个蜂鸣时长 (秒)
        gap: 两个蜂鸣间隔 (秒)
        amplitude: 音量 (0-1)
        sample_rate: 采样率
    """
    single_samples = int(sample_rate * duration)
    gap_samples = int(sample_rate * gap)
    
    # 生成两个蜂鸣
    t = np.linspace(0, duration, single_samples, False)
    beep1 = amplitude * np.sin(2 * np.pi * freq * t)
    beep2 = amplitude * np.sin(2 * np.pi * freq * t)
    
    # 中间静音
    silence = np.zeros(gap_samples)
    
    # 拼接
    wave_data = np.concatenate([beep1, silence, beep2])
    
    # 整体淡入淡出
    fade_samples = int(sample_rate * 0.01)
    if fade_samples > 0 and len(wave_data) > fade_samples * 2:
        wave_data[:fade_samples] *= np.linspace(0, 1, fade_samples)
        wave_data[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    
    # 转换为 16-bit PCM
    wave_data = (wave_data * 32767).astype(np.int16)
    
    # 保存
    with wave.open(filename, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(wave_data.tobytes())
    
    total_duration = duration * 2 + gap
    print(f"生成: {filename}")
    print(f"  频率: {freq}Hz")
    print(f"  单声时长: {duration}s")
    print(f"  间隔: {gap}s")
    print(f"  总时长: {total_duration}s")
    print(f"  音量: {amplitude * 100}%")
    print()


def main():
    """主函数：生成两种蜂鸣声"""
    print("=" * 50)
    print("生成 Nano 同步蜂鸣声")
    print("=" * 50)
    print()
    
    # 开始音：1000Hz, 0.1s, 单声
    generate_single_beep(
        filename='start_beep.wav',
        freq=1000,
        duration=0.1,
        amplitude=0.8
    )
    
    # 结束音：1500Hz, 0.1s×2, 间隔0.2s
    generate_double_beep(
        filename='end_beep.wav',
        freq=1500,
        duration=0.1,
        gap=0.2,
        amplitude=0.8
    )
    
    print("=" * 50)
    print("完成！生成的文件：")
    print("  - start_beep.wav (开始音: 1000Hz, 单声)")
    print("  - end_beep.wav (结束音: 1500Hz, 双声)")
    print("=" * 50)
    print()
    print("测试命令:")
    print("  paplay start_beep.wav  # 或 aplay start_beep.wav")
    print("  paplay end_beep.wav    # 或 aplay end_beep.wav")


if __name__ == '__main__':
    main()
