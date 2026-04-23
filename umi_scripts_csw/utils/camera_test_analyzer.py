#!/usr/bin/env python3
"""
相机测试方案分析脚本
读取 camera_test_records.csv 并生成分析报告

使用方法：
    python camera_test_analyzer.py
"""

import csv
import os
from datetime import datetime
from pathlib import Path


# 8种测试方案定义
ALL_SCHEMES = {
    "DJI-C": {"camera": "大疆Nano", "position": "胸口", "has_depth": False, "expected_fov": "80-90°"},
    "DJI-H": {"camera": "大疆Nano", "position": "头上", "has_depth": False, "expected_fov": "80-90°"},
    "ORB-C": {"camera": "奥比中光DCW2", "position": "胸口", "has_depth": True, "expected_fov": "90-100°"},
    "ORB-H": {"camera": "奥比中光DCW2", "position": "头上", "has_depth": True, "expected_fov": "90-100°"},
    "ZED-C": {"camera": "ZED相机", "position": "胸口", "has_depth": True, "expected_fov": "110-120°"},
    "ZED-H": {"camera": "ZED相机", "position": "头上", "has_depth": True, "expected_fov": "110-120°"},
    "WIDE-C": {"camera": "广角相机(120°)", "position": "胸口", "has_depth": False, "expected_fov": "120°"},
    "WIDE-H": {"camera": "广角相机(120°)", "position": "头上", "has_depth": False, "expected_fov": "120°"},
}


def load_records(csv_path):
    """读取CSV记录文件"""
    records = []
    if not os.path.exists(csv_path):
        return records
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 跳过注释行和空行
            if not row.get('scheme_id') or row['scheme_id'].startswith('#'):
                continue
            
            # 转换数值字段
            try:
                row['fov_coverage_score'] = float(row['fov_coverage_score']) if row.get('fov_coverage_score') else None
                row['stability_score'] = float(row['stability_score']) if row.get('stability_score') else None
                row['comfort_score'] = float(row['comfort_score']) if row.get('comfort_score') else None
                row['aruco_detection_rate'] = float(row['aruco_detection_rate']) if row.get('aruco_detection_rate') else None
                row['depth_quality_score'] = float(row['depth_quality_score']) if row.get('depth_quality_score') else None
                row['overall_score'] = float(row['overall_score']) if row.get('overall_score') else None
                row['test_duration_min'] = float(row['test_duration_min']) if row.get('test_duration_min') else None
            except (ValueError, TypeError):
                pass
            
            records.append(row)
    
    return records


def analyze_records(records):
    """分析记录数据"""
    # 按方案分组
    scheme_stats = {}
    
    for record in records:
        scheme_id = record.get('scheme_id', '').strip().upper()
        if scheme_id not in ALL_SCHEMES:
            continue
        
        if scheme_id not in scheme_stats:
            scheme_stats[scheme_id] = {
                'count': 0,
                'fov_scores': [],
                'stability_scores': [],
                'comfort_scores': [],
                'detection_rates': [],
                'depth_scores': [],
                'overall_scores': [],
                'notes': [],
                'dealbreakers': [],
                'recommendations': [],
            }
        
        stats = scheme_stats[scheme_id]
        stats['count'] += 1
        
        # Helper to safely convert to float
        def safe_float(val):
            if val is None or val == '':
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None
        
        fov_score = safe_float(record.get('fov_coverage_score'))
        stability_score = safe_float(record.get('stability_score'))
        comfort_score = safe_float(record.get('comfort_score'))
        detection_rate = safe_float(record.get('aruco_detection_rate'))
        depth_score = safe_float(record.get('depth_quality_score'))
        overall_score = safe_float(record.get('overall_score'))
        
        if fov_score is not None:
            stats['fov_scores'].append(fov_score)
        if stability_score is not None:
            stats['stability_scores'].append(stability_score)
        if comfort_score is not None:
            stats['comfort_scores'].append(comfort_score)
        if detection_rate is not None:
            stats['detection_rates'].append(detection_rate)
        if depth_score is not None:
            stats['depth_scores'].append(depth_score)
        if overall_score is not None:
            stats['overall_scores'].append(overall_score)
        
        if record.get('fov_coverage_notes'):
            stats['notes'].append(f"视野: {record['fov_coverage_notes']}")
        if record.get('dealbreaker_issues'):
            stats['dealbreakers'].append(record['dealbreaker_issues'])
        if record.get('recommended_for'):
            stats['recommendations'].append(record['recommended_for'])
    
    # 计算平均分
    for scheme_id, stats in scheme_stats.items():
        stats['avg_fov'] = sum(stats['fov_scores']) / len(stats['fov_scores']) if stats['fov_scores'] else None
        stats['avg_stability'] = sum(stats['stability_scores']) / len(stats['stability_scores']) if stats['stability_scores'] else None
        stats['avg_comfort'] = sum(stats['comfort_scores']) / len(stats['comfort_scores']) if stats['comfort_scores'] else None
        stats['avg_detection'] = sum(stats['detection_rates']) / len(stats['detection_rates']) if stats['detection_rates'] else None
        stats['avg_depth'] = sum(stats['depth_scores']) / len(stats['depth_scores']) if stats['depth_scores'] else None
        stats['avg_overall'] = sum(stats['overall_scores']) / len(stats['overall_scores']) if stats['overall_scores'] else None
    
    return scheme_stats


def calculate_composite_score(stats, scheme_info):
    """计算综合评分（加权平均）"""
    weights = {
        'fov': 0.25,      # 视野覆盖最重要
        'stability': 0.20,
        'comfort': 0.15,
        'detection': 0.30,  # 检测率非常重要
        'depth': 0.10,     # 深度质量权重较低
    }
    
    score = 0
    weight_sum = 0
    
    if stats['avg_fov'] is not None:
        score += stats['avg_fov'] * weights['fov']
        weight_sum += weights['fov']
    if stats['avg_stability'] is not None:
        score += stats['avg_stability'] * weights['stability']
        weight_sum += weights['stability']
    if stats['avg_comfort'] is not None:
        score += stats['avg_comfort'] * weights['comfort']
        weight_sum += weights['comfort']
    if stats['avg_detection'] is not None:
        # 检测率转换为5分制
        detection_5scale = stats['avg_detection'] / 20
        score += detection_5scale * weights['detection']
        weight_sum += weights['detection']
    if stats['avg_depth'] is not None and scheme_info['has_depth']:
        score += stats['avg_depth'] * weights['depth']
        weight_sum += weights['depth']
    
    if weight_sum == 0:
        return stats['avg_overall'] if stats['avg_overall'] else 0
    
    return score / weight_sum if weight_sum > 0 else 0


def print_report(scheme_stats):
    """打印分析报告"""
    print("=" * 70)
    print("           相机测试方案分析结果")
    print("=" * 70)
    print()
    
    # 计算综合评分和排名
    scored_schemes = []
    for scheme_id, stats in scheme_stats.items():
        scheme_info = ALL_SCHEMES[scheme_id]
        composite = calculate_composite_score(stats, scheme_info)
        scored_schemes.append((scheme_id, stats, scheme_info, composite))
    
    # 按综合评分排序
    scored_schemes.sort(key=lambda x: x[3], reverse=True)
    
    # 已完成测试的方案
    tested = [s for s in scored_schemes if s[1]['count'] > 0]
    untested = [s for s in scored_schemes if s[1]['count'] == 0]
    
    # 统计
    total_schemes = len(ALL_SCHEMES)
    tested_count = len(tested)
    
    print(f"【测试进度】({tested_count}/{total_schemes})")
    print(f"已完成: {tested_count}个方案 | 待测试: {total_schemes - tested_count}个方案")
    print()
    
    if tested:
        print("【已完成测试方案排名】")
        print("-" * 70)
        print(f"{'排名':<4} {'方案':<8} {'相机':<15} {'位置':<6} {'综合分':<6} {'检测率':<8} {'关键特点'}")
        print("-" * 70)
        
        for rank, (scheme_id, stats, info, composite) in enumerate(tested, 1):
            camera = info['camera']
            position = info['position']
            avg_detection = stats['avg_detection']
            detection_str = f"{avg_detection:.0f}%" if avg_detection else "N/A"
            
            # 关键特点总结
            features = []
            if stats['avg_fov'] and stats['avg_fov'] >= 4.5:
                features.append("视野优秀")
            if stats['avg_comfort'] and stats['avg_comfort'] >= 4.5:
                features.append("佩戴舒适")
            if avg_detection and avg_detection >= 95:
                features.append("检测率高")
            if info['has_depth']:
                features.append("有深度")
            
            feature_str = " | ".join(features) if features else "-"
            
            print(f"{rank:<4} {scheme_id:<8} {camera:<15} {position:<6} {composite:<6.2f} {detection_str:<8} {feature_str}")
        
        print("-" * 70)
        print()
    
    if untested:
        print("【待测试方案】")
        for scheme_id, stats, info, _ in untested:
            camera = info['camera']
            position = info['position']
            fov = info['expected_fov']
            depth = "有深度" if info['has_depth'] else "无深度"
            print(f"  - {scheme_id}: {camera} + {position} (预估视野: {fov}, {depth})")
        print()
    
    # 分项最佳
    if tested:
        print("【分项最佳】")
        
        # 最佳视野
        best_fov = max(tested, key=lambda x: x[1]['avg_fov'] or 0)
        print(f"  最佳视野: {best_fov[0]} ({best_fov[1]['avg_fov']:.1f}分) - {best_fov[2]['camera']}+{best_fov[2]['position']}")
        
        # 最佳稳定
        best_stable = max(tested, key=lambda x: x[1]['avg_stability'] or 0)
        print(f"  最佳稳定: {best_stable[0]} ({best_stable[1]['avg_stability']:.1f}分) - {best_stable[2]['camera']}+{best_stable[2]['position']}")
        
        # 最佳舒适
        best_comfort = max(tested, key=lambda x: x[1]['avg_comfort'] or 0)
        print(f"  最佳舒适: {best_comfort[0]} ({best_comfort[1]['avg_comfort']:.1f}分) - {best_comfort[2]['camera']}+{best_comfort[2]['position']}")
        
        # 最佳检测率
        best_detection = max(tested, key=lambda x: x[1]['avg_detection'] or 0)
        print(f"  最佳检测率: {best_detection[0]} ({best_detection[1]['avg_detection']:.0f}%) - {best_detection[2]['camera']}+{best_detection[2]['position']}")
        
        # 最佳深度（仅深度相机）
        depth_cameras = [s for s in tested if s[2]['has_depth'] and s[1]['avg_depth']]
        if depth_cameras:
            best_depth = max(depth_cameras, key=lambda x: x[1]['avg_depth'] or 0)
            print(f"  最佳深度: {best_depth[0]} ({best_depth[1]['avg_depth']:.1f}分) - {best_depth[2]['camera']}+{best_depth[2]['position']}")
        
        print()
    
    # 关键问题汇总
    all_dealbreakers = []
    for scheme_id, stats, _, _ in tested:
        for issue in stats['dealbreakers']:
            all_dealbreakers.append((scheme_id, issue))
    
    if all_dealbreakers:
        print("【关键问题汇总】")
        for scheme_id, issue in all_dealbreakers:
            print(f"  - {scheme_id}: {issue}")
        print()
    
    # 最终推荐
    if tested:
        print("【最终推荐】")
        best = tested[0]
        print(f"  综合推荐方案: {best[0]} ({best[2]['camera']} + {best[2]['position']})")
        print(f"    综合评分: {best[3]:.2f}/5.0")
        print(f"    适用场景: {', '.join(best[1]['recommendations']) if best[1]['recommendations'] else '待补充'}")
        print()
    
    print("=" * 70)
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


def main():
    """主函数"""
    script_dir = Path(__file__).parent
    csv_path = script_dir / "camera_test_records.csv"
    
    print(f"正在读取测试记录: {csv_path}")
    
    records = load_records(csv_path)
    
    if not records:
        print("警告: 没有找到有效的测试记录")
        print("请在 camera_test_records.csv 中填入测试数据后重新运行")
        return
    
    print(f"已加载 {len(records)} 条测试记录")
    print()
    
    # 初始化所有方案的统计（包括未测试的）
    scheme_stats = {sid: {
        'count': 0,
        'fov_scores': [],
        'stability_scores': [],
        'comfort_scores': [],
        'detection_rates': [],
        'depth_scores': [],
        'overall_scores': [],
        'notes': [],
        'dealbreakers': [],
        'recommendations': [],
        'avg_fov': None,
        'avg_stability': None,
        'avg_comfort': None,
        'avg_detection': None,
        'avg_depth': None,
        'avg_overall': None,
    } for sid in ALL_SCHEMES}
    
    # 分析记录
    analyzed_stats = analyze_records(records)
    scheme_stats.update(analyzed_stats)
    
    # 打印报告
    print_report(scheme_stats)


if __name__ == "__main__":
    main()
