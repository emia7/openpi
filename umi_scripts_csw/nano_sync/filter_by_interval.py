#!/usr/bin/env python3
"""
基于时间间隔过滤检测事件

原理：脚踏板踩踏有合理的间隔（比如 5-30 秒）
过滤掉太密集的事件（噪音）

使用:
    python filter_by_interval.py <detection_json> --min-gap 3 --max-gap 30
"""

import json
import sys
import os
import numpy as np


def filter_events_by_interval(events, min_gap=3.0, max_gap=60.0):
    """
    过滤事件，只保留间隔合理的事件
    
    Args:
        events: 事件列表（带'time'字段）
        min_gap: 最小间隔（秒）
        max_gap: 最大间隔（秒）
    """
    if len(events) < 2:
        return events
    
    # 按时间排序
    events = sorted(events, key=lambda x: x['time'])
    
    filtered = [events[0]]  # 保留第一个
    
    for i in range(1, len(events)):
        gap = events[i]['time'] - filtered[-1]['time']
        
        # 如果间隔合理，保留
        if min_gap <= gap <= max_gap:
            filtered.append(events[i])
    
    return filtered


def main():
    if len(sys.argv) < 2:
        print("使用: python filter_by_interval.py <json_file> [min_gap] [max_gap]")
        print("  min_gap: 最小间隔（秒，默认 3.0）")
        print("  max_gap: 最大间隔（秒，默认 60.0）")
        print("\\n示例:")
        print("  python filter_by_interval.py detection.json 5 30")
        sys.exit(1)
    
    json_path = sys.argv[1]
    min_gap = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    max_gap = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
    
    if not os.path.exists(json_path):
        print(f"[ERROR] 文件不存在: {json_path}")
        sys.exit(1)
    
    # 加载
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    events = data.get('events', [])
    
    print("=" * 60)
    print("基于间隔过滤事件")
    print("=" * 60)
    print(f"原始事件数: {len(events)}")
    print(f"过滤条件: 间隔 {min_gap}s - {max_gap}s")
    print()
    
    # 过滤
    filtered = filter_events_by_interval(events, min_gap, max_gap)
    
    print(f"过滤后事件数: {len(filtered)}")
    print(f"过滤比例: {len(filtered)/len(events)*100:.1f}%")
    print()
    
    # 显示前 20 个
    print("保留的事件（前 20 个）:")
    for i, e in enumerate(filtered[:20], 1):
        print(f"  {i}. @ {e['time']:.2f}s (强度: {e['strength']:.4f})")
    
    if len(filtered) > 20:
        print(f"  ... 还有 {len(filtered)-20} 个 ...")
    
    # 配对
    pairs = []
    for i in range(0, len(filtered)-1, 2):
        if i+1 < len(filtered):
            t1 = filtered[i]['time']
            t2 = filtered[i+1]['time']
            pairs.append({
                'start': t1,
                'end': t2,
                'duration': t2 - t1
            })
    
    print()
    print(f"配对片段: {len(pairs)} 个")
    if pairs:
        durations = [p['duration'] for p in pairs]
        print(f"  平均时长: {np.mean(durations):.1f}s")
        print(f"  最短: {min(durations):.1f}s")
        print(f"  最长: {max(durations):.1f}s")
    
    # 保存
    output = {
        'original_file': json_path,
        'min_gap': min_gap,
        'max_gap': max_gap,
        'original_count': len(events),
        'filtered_count': len(filtered),
        'filtered_events': filtered,
        'pairs': pairs
    }
    
    output_path = json_path.replace('.json', '_filtered.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print()
    print(f"[OK] 过滤结果保存: {output_path}")
    print()
    print("下一步:")
    print(f"  python visualize_detection.py test_video.mp4 {output_path} --frames")


if __name__ == '__main__':
    main()
