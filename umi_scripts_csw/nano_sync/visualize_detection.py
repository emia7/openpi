#!/usr/bin/env python3
"""
可视化检测结果 - 在视频帧上标记检测点

生成预览图或短视频片段，让你直观验证检测是否正确

使用:
    python visualize_detection.py <video_file> <detection_json> [--output preview.mp4]
    
输出:
    - 预览视频/图片，带红色标记和时间戳
    - 或提取关键帧图片集
"""

import cv2
import json
import numpy as np
import sys
import os
from pathlib import Path


def load_detection(json_path):
    """加载检测结果"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 支持两种格式
    if 'events' in data:
        events = data['events']
    elif 'marks' in data:
        events = data['marks']
    else:
        events = []
    
    return events


def extract_keyframes(video_path, events, output_dir, num_frames=50):
    """
    提取关键帧（检测到事件的时间点）
    
    Args:
        video_path: 视频路径
        events: 事件列表（带 'time' 字段）
        output_dir: 输出目录
        num_frames: 最多提取多少帧（均匀采样）
    """
    print(f"提取关键帧...")
    print(f"  视频: {video_path}")
    print(f"  总事件数: {len(events)}")
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] 无法打开视频: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"  视频时长: {duration:.1f}s, FPS: {fps}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 均匀采样事件（如果太多）
    if len(events) > num_frames:
        indices = np.linspace(0, len(events)-1, num_frames, dtype=int)
        selected_events = [events[i] for i in indices]
        print(f"  采样显示: {num_frames}/{len(events)} 个事件")
    else:
        selected_events = events
        print(f"  显示全部: {len(events)} 个事件")
    
    # 提取帧
    for i, event in enumerate(selected_events, 1):
        time_sec = event.get('time', event.get('start', 0))
        frame_num = int(time_sec * fps)
        
        # 定位到帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # 添加标记
        h, w = frame.shape[:2]
        
        # 红色圆点（中心）
        center = (w // 2, h // 2)
        cv2.circle(frame, center, 30, (0, 0, 255), -1)  # 实心圆
        cv2.circle(frame, center, 30, (255, 255, 255), 3)  # 白边
        
        # 时间戳文字
        text = f"{time_sec:.2f}s"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, (20, 50), font, 1.5, (0, 0, 255), 3)
        
        # 事件序号
        cv2.putText(frame, f"#{i}", (20, h - 20), font, 1.2, (0, 255, 0), 2)
        
        # 保存
        output_path = os.path.join(output_dir, f"frame_{i:04d}_{time_sec:.2f}s.jpg")
        cv2.imwrite(output_path, frame)
        
        if i % 10 == 0:
            print(f"    已保存 {i}/{len(selected_events)} 帧")
    
    cap.release()
    
    print(f"\n[OK] 关键帧已保存到: {output_dir}/")
    print(f"     共 {len(selected_events)} 张图片")
    print(f"\n提示: 浏览这些图片，检查红圈位置是否对应脚踏板声音")


def create_preview_video(video_path, events, output_path, max_duration=60):
    """
    创建预览视频（只包含检测到事件的片段）
    
    适合快速验证，视频小，不上传GitHub
    """
    print(f"创建预览视频...")
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 只取前 max_duration 秒的检测事件
    selected = [e for e in events if e.get('time', 0) < max_duration][:20]
    
    print(f"  包含前 {max_duration}s 的 {len(selected)} 个事件")
    
    # 写入帧
    frame_count = 0
    for event in selected:
        time_sec = event.get('time', 0)
        frame_num = int(time_sec * fps)
        
        # 提取前后各 0.5 秒
        start_frame = max(0, frame_num - int(0.5 * fps))
        end_frame = frame_num + int(0.5 * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for f in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            # 在中心帧添加标记
            if f == frame_num:
                h, w = frame.shape[:2]
                cv2.circle(frame, (w//2, h//2), 40, (0, 0, 255), -1)
                cv2.putText(frame, f"{time_sec:.2f}s", (20, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            
            out.write(frame)
            frame_count += 1
    
    cap.release()
    out.release()
    
    print(f"[OK] 预览视频: {output_path}")
    print(f"     帧数: {frame_count}, 时长: ~{frame_count/fps:.1f}s")
    print(f"\n提示: 播放视频，看红圈是否出现在脚踏板声音时刻")


def main():
    if len(sys.argv) < 3:
        print("使用:")
        print("  提取关键帧: python visualize_detection.py video.mp4 detection.json --frames")
        print("  创建预览视频: python visualize_detection.py video.mp4 detection.json --preview")
        sys.exit(1)
    
    video_path = sys.argv[1]
    json_path = sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else '--frames'
    
    if not os.path.exists(video_path):
        print(f"[ERROR] 视频不存在: {video_path}")
        sys.exit(1)
    
    if not os.path.exists(json_path):
        print(f"[ERROR] 检测结果不存在: {json_path}")
        sys.exit(1)
    
    # 加载检测结果
    events = load_detection(json_path)
    
    print("=" * 60)
    print("检测结果可视化")
    print("=" * 60)
    print(f"视频: {video_path}")
    print(f"检测文件: {json_path}")
    print(f"事件数: {len(events)}")
    print()
    
    if mode == '--frames':
        # 提取关键帧
        output_dir = video_path.replace('.mp4', '_detection_frames')
        extract_keyframes(video_path, events, output_dir)
    elif mode == '--preview':
        # 创建预览视频
        output_path = video_path.replace('.mp4', '_preview.mp4')
        create_preview_video(video_path, events, output_path)
    else:
        print(f"[ERROR] 未知模式: {mode}")
        print("使用 --frames 或 --preview")
        sys.exit(1)
    
    print()
    print("=" * 60)
    print("完成！浏览生成的文件来验证检测准确性。")
    print("=" * 60)


if __name__ == '__main__':
    main()
