#!/usr/bin/env python3
"""
批量测试声音检测算法

使用:
    python test_detection.py --video-dir ./test_videos/
    
输出:
    - 详细的测试报告
    - 每个视频的检测结果对比
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path

# 导入检测函数
from detect_beep_in_video import detect_beep_marks


def load_ground_truth(video_path):
    """
    加载标注文件（如果存在）
    
    标注文件格式: <video>_ground_truth.json
    {
        "expected_marks": [
            {"type": "start", "time": 5.0},
            {"type": "end", "time": 15.0}
        ]
    }
    """
    gt_path = video_path.replace('.mp4', '_ground_truth.json')
    if os.path.exists(gt_path):
        with open(gt_path, 'r') as f:
            return json.load(f).get('expected_marks', [])
    return None


def evaluate_detection(detected, expected):
    """
    评估检测结果
    
    Args:
        detected: 检测到的标记列表
        expected: 期望的标记列表（带 time 字段）
    
    Returns:
        dict: 评估指标
    """
    if not expected:
        return None
    
    tolerance = 0.5  # 时间容差 ±0.5s
    
    correct = 0
    false_positives = 0
    false_negatives = 0
    time_errors = []
    
    # 匹配检测到的和期望的
    matched_expected = [False] * len(expected)
    
    for det in detected:
        det_time = det['time']
        det_type = det['type']
        
        # 寻找匹配的期望标记
        matched = False
        for i, exp in enumerate(expected):
            if matched_expected[i]:
                continue
            
            exp_time = exp['time']
            exp_type = exp['type']
            
            # 时间和类型都匹配
            if abs(det_time - exp_time) < tolerance and det_type == exp_type:
                correct += 1
                matched_expected[i] = True
                time_errors.append(abs(det_time - exp_time))
                matched = True
                break
        
        if not matched:
            false_positives += 1
    
    # 未匹配的期望标记是假阴性
    false_negatives = sum(1 for m in matched_expected if not m)
    
    return {
        'correct': correct,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': correct / (correct + false_positives) if (correct + false_positives) > 0 else 0,
        'recall': correct / (correct + false_negatives) if (correct + false_negatives) > 0 else 0,
        'avg_time_error': sum(time_errors) / len(time_errors) if time_errors else 0,
        'max_time_error': max(time_errors) if time_errors else 0
    }


def test_single_video(video_path, visualize=False):
    """
    测试单个视频
    
    Returns:
        dict: 测试结果
    """
    print(f"\n测试: {os.path.basename(video_path)}")
    print("-" * 60)
    
    try:
        # 运行检测
        marks = detect_beep_marks(video_path, visualize=visualize)
        
        # 加载标注
        expected = load_ground_truth(video_path)
        
        # 统计
        start_count = sum(1 for m in marks if m['type'] == 'start')
        end_count = sum(1 for m in marks if m['type'] == 'end')
        
        result = {
            'video': video_path,
            'status': 'success',
            'detected_count': len(marks),
            'start_count': start_count,
            'end_count': end_count,
            'marks': marks
        }
        
        # 如果有标注，进行评估
        if expected:
            evaluation = evaluate_detection(marks, expected)
            result['evaluation'] = evaluation
            
            print(f"检测结果: {len(marks)} 个 (期望: {len(expected)} 个)")
            print(f"  开始音: {start_count}, 结束音: {end_count}")
            print(f"  正确率: {evaluation['correct']}/{len(expected)} ({evaluation['recall']*100:.1f}%)")
            print(f"  误检: {evaluation['false_positives']} 个")
            print(f"  漏检: {evaluation['false_negatives']} 个")
            print(f"  平均时间误差: {evaluation['avg_time_error']*1000:.1f}ms")
            
            # 判断是否通过
            passed = (evaluation['recall'] >= 0.9 and evaluation['avg_time_error'] <= 0.05)
            result['passed'] = passed
            print(f"  结果: {'✓ 通过' if passed else '✗ 不通过'}")
        else:
            print(f"检测结果: {len(marks)} 个标记")
            print(f"  开始音: {start_count}, 结束音: {end_count}")
            print("  (无标注文件，跳过评估)")
            result['passed'] = None
        
        return result
        
    except Exception as e:
        print(f"[ERROR] 测试失败: {e}")
        return {
            'video': video_path,
            'status': 'error',
            'error': str(e)
        }


def generate_report(results, output_file='detection_test_report.json'):
    """
    生成测试报告
    """
    # 统计
    total_videos = len(results)
    success_videos = sum(1 for r in results if r['status'] == 'success')
    error_videos = sum(1 for r in results if r['status'] == 'error')
    
    # 有评估结果的视频
    evaluated_results = [r for r in results if r.get('evaluation')]
    
    if evaluated_results:
        avg_recall = sum(r['evaluation']['recall'] for r in evaluated_results) / len(evaluated_results)
        avg_precision = sum(r['evaluation']['precision'] for r in evaluated_results) / len(evaluated_results)
        avg_time_error = sum(r['evaluation']['avg_time_error'] for r in evaluated_results) / len(evaluated_results)
        passed_count = sum(1 for r in evaluated_results if r.get('passed'))
    else:
        avg_recall = avg_precision = avg_time_error = 0
        passed_count = 0
    
    # 生成报告
    report = {
        'summary': {
            'total_videos': total_videos,
            'success_videos': success_videos,
            'error_videos': error_videos,
            'evaluated_videos': len(evaluated_results),
            'passed_videos': passed_count,
            'avg_recall': f"{avg_recall*100:.1f}%",
            'avg_precision': f"{avg_precision*100:.1f}%",
            'avg_time_error_ms': f"{avg_time_error*1000:.1f}ms"
        },
        'results': results
    }
    
    # 保存报告
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 打印报告
    print()
    print("=" * 60)
    print("测试报告")
    print("=" * 60)
    print(f"测试视频数: {total_videos}")
    print(f"  成功处理: {success_videos}")
    print(f"  处理失败: {error_videos}")
    print()
    
    if evaluated_results:
        print("评估结果:")
        print(f"  召回率: {avg_recall*100:.1f}%")
        print(f"  精确率: {avg_precision*100:.1f}%")
        print(f"  平均时间误差: {avg_time_error*1000:.1f}ms")
        print(f"  通过视频: {passed_count}/{len(evaluated_results)}")
        print()
    
    print("详细结果:")
    for r in results:
        name = os.path.basename(r['video'])
        if r['status'] == 'error':
            print(f"  ✗ {name}: 处理失败 - {r.get('error', 'Unknown')}")
        elif r.get('passed') is None:
            print(f"  ? {name}: {r['detected_count']} 个标记 (未评估)")
        elif r.get('passed'):
            print(f"  ✓ {name}: 通过 ({r['detected_count']} 个标记)")
        else:
            print(f"  ✗ {name}: 不通过 (召回率: {r['evaluation']['recall']*100:.1f}%)")
    
    print()
    print(f"报告已保存: {output_file}")
    print("=" * 60)
    
    return report


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='测试声音检测算法')
    parser.add_argument('--video-dir', required=True, help='测试视频目录')
    parser.add_argument('--visualize', action='store_true', help='生成可视化图像')
    parser.add_argument('--output', default='detection_test_report.json', help='报告输出文件')
    args = parser.parse_args()
    
    # 查找视频文件
    video_dir = Path(args.video_dir)
    video_patterns = ['*.mp4', '*.MP4', '*.mov', '*.MOV']
    
    video_files = []
    for pattern in video_patterns:
        video_files.extend(video_dir.glob(pattern))
    
    video_files = sorted(set(video_files))
    
    if not video_files:
        print(f"[ERROR] 在 {video_dir} 中找不到视频文件")
        print("支持的格式: .mp4, .mov")
        sys.exit(1)
    
    print("=" * 60)
    print("声音检测算法批量测试")
    print("=" * 60)
    print(f"测试目录: {video_dir}")
    print(f"测试视频: {len(video_files)} 个")
    print()
    
    # 逐个测试
    results = []
    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] ", end='')
        result = test_single_video(str(video_path), visualize=args.visualize)
        results.append(result)
    
    # 生成报告
    generate_report(results, output_file=args.output)


if __name__ == '__main__':
    main()
