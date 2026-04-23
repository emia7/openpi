#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按照数据合并与清理规范 v1.0 对0422数据进行清理预览
不实际删除文件，只生成清理报告
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import imageio.v3 as iio


def check_file_completeness(episode_dir: Path, episode_name: str) -> Tuple[bool, List[str]]:
    """CHECK-001: 文件完整性检查"""
    required_files = [
        f"{episode_name}_left.mp4",
        f"{episode_name}_left.json",
        f"{episode_name}_right.mp4",
        f"{episode_name}_right.json",
        f"{episode_name}_third.mp4",
    ]
    missing = [f for f in required_files if not (episode_dir / f).exists()]
    return len(missing) == 0, missing


def load_episode_info(json_path: Path) -> Dict:
    """加载episode元信息"""
    try:
        with open(json_path) as f:
            data = json.load(f)
        return {
            "num_frames": data.get("num_frames", 0),
            "fps": data.get("fps", 0),
            "valid": True
        }
    except Exception as e:
        return {"num_frames": 0, "fps": 0, "valid": False, "error": str(e)}


def get_video_info(mp4_path: Path) -> Dict:
    """获取视频信息"""
    try:
        frame = iio.imread(mp4_path, index=0)
        return {
            "resolution": frame.shape[:2],  # (H, W)
            "readable": True
        }
    except Exception as e:
        return {"resolution": (0, 0), "readable": False, "error": str(e)}


def analyze_data(data_dir: Path) -> Dict:
    """分析数据集，生成清理报告预览"""
    
    print(f"[INFO] 分析数据目录: {data_dir}")
    
    # 扫描所有episodes
    left_jsons = sorted(data_dir.glob("episode_*_left.json"))
    print(f"[INFO] 发现 {len(left_jsons)} 个episodes")
    
    # 统计存储
    episodes_info = []
    total_frames_list = []
    resolutions = {"left": [], "right": [], "third": []}
    
    # 各类异常统计
    incomplete_files = []
    json_errors = []
    video_errors = []
    short_episodes = []
    
    for json_file in left_jsons:
        episode_name = json_file.stem.replace("_left", "")
        
        # CHECK-001: 文件完整性
        is_complete, missing = check_file_completeness(data_dir, episode_name)
        if not is_complete:
            incomplete_files.append({
                "episode": episode_name,
                "missing": missing
            })
            continue
        
        # 加载JSON信息
        left_info = load_episode_info(json_file)
        right_info = load_episode_info(data_dir / f"{episode_name}_right.json")
        
        if not left_info["valid"]:
            json_errors.append({"episode": episode_name, "error": left_info.get("error", "unknown")})
            continue
        if not right_info["valid"]:
            json_errors.append({"episode": episode_name, "error": right_info.get("error", "unknown")})
            continue
        
        # 检查视频可读性
        left_video_info = get_video_info(data_dir / f"{episode_name}_left.mp4")
        right_video_info = get_video_info(data_dir / f"{episode_name}_right.mp4")
        third_video_info = get_video_info(data_dir / f"{episode_name}_third.mp4")
        
        if not left_video_info["readable"]:
            video_errors.append({"episode": episode_name, "view": "left"})
            continue
        if not right_video_info["readable"]:
            video_errors.append({"episode": episode_name, "view": "right"})
            continue
        if not third_video_info["readable"]:
            video_errors.append({"episode": episode_name, "view": "third"})
            continue
        
        # 记录分辨率
        resolutions["left"].append(left_video_info["resolution"])
        resolutions["right"].append(right_video_info["resolution"])
        resolutions["third"].append(third_video_info["resolution"])
        
        # 帧数信息
        left_frames = left_info["num_frames"]
        right_frames = right_info["num_frames"]
        third_frames = left_frames  # 假设第三视角与左手相同
        avg_frames = (left_frames + right_frames) / 2
        total_frames_list.append(avg_frames)
        
        episodes_info.append({
            "episode": episode_name,
            "left_frames": left_frames,
            "right_frames": right_frames,
            "third_frames": third_frames,
            "avg_frames": avg_frames,
            "left_resolution": left_video_info["resolution"],
            "right_resolution": right_video_info["resolution"],
            "third_resolution": third_video_info["resolution"],
        })
    
    # 计算统计数据
    if total_frames_list:
        avg_frames_total = np.mean(total_frames_list)
        min_frames = np.min(total_frames_list)
        max_frames = np.max(total_frames_list)
    else:
        avg_frames_total = 0
        min_frames = 0
        max_frames = 0
    
    # CHECK-002: 帧数异常检测 (动态阈值)
    threshold = avg_frames_total * 0.3  # 剔除线
    warning_threshold = avg_frames_total * 0.5  # 警告线
    
    for ep in episodes_info:
        if ep["avg_frames"] < threshold:
            short_episodes.append({
                "episode": ep["episode"],
                "frames": ep["avg_frames"],
                "status": "剔除",
                "reason": f"低于阈值({threshold:.1f}帧)"
            })
        elif ep["avg_frames"] < warning_threshold:
            short_episodes.append({
                "episode": ep["episode"],
                "frames": ep["avg_frames"],
                "status": "警告",
                "reason": f"低于警告线({warning_threshold:.1f}帧)"
            })
    
    # CHECK-003: 分辨率一致性
    left_res_set = set(resolutions["left"])
    right_res_set = set(resolutions["right"])
    third_res_set = set(resolutions["third"])
    
    # CHECK-004: 时间同步性 (帧数相关)
    sync_issues = []
    for ep in episodes_info:
        left_frames = ep["left_frames"]
        right_frames = ep["right_frames"]
        avg_ep_frames = (left_frames + right_frames) / 2
        
        # 左右手差异 < 5%
        if abs(left_frames - right_frames) > 0.05 * avg_ep_frames:
            sync_issues.append({
                "episode": ep["episode"],
                "issue": "左右手帧数差异过大",
                "left": left_frames,
                "right": right_frames,
                "diff": abs(left_frames - right_frames),
                "threshold": 0.05 * avg_ep_frames
            })
    
    # 生成报告
    report = {
        "report_version": "1.0",
        "data_source": str(data_dir),
        "analysis_timestamp": str(np.datetime64('now')),
        "summary": {
            "total_episodes_scanned": len(left_jsons),
            "episodes_passed_basic_check": len(episodes_info),
            "episodes_with_issues": len(incomplete_files) + len(json_errors) + len(video_errors),
        },
        "statistics": {
            "avg_frames_per_episode": float(avg_frames_total),
            "min_frames": int(min_frames),
            "max_frames": int(max_frames),
            "frame_threshold": float(threshold),
            "frame_warning_threshold": float(warning_threshold),
            "resolution_left": list(left_res_set),
            "resolution_right": list(right_res_set),
            "resolution_third": list(third_res_set),
        },
        "issues": {
            "incomplete_files": {
                "count": len(incomplete_files),
                "episodes": incomplete_files[:5]  # 只显示前5个
            },
            "json_errors": {
                "count": len(json_errors),
                "episodes": json_errors[:5]
            },
            "video_errors": {
                "count": len(video_errors),
                "episodes": video_errors[:5]
            },
            "short_episodes": {
                "count": len(short_episodes),
                "to_remove": len([s for s in short_episodes if s["status"] == "剔除"]),
                "to_warn": len([s for s in short_episodes if s["status"] == "警告"]),
                "episodes": short_episodes[:10]  # 显示前10个
            },
            "sync_issues": {
                "count": len(sync_issues),
                "episodes": sync_issues[:5]
            }
        },
        "cleaning_preview": {
            "episodes_to_remove": [s["episode"] for s in short_episodes if s["status"] == "剔除"],
            "episodes_to_warn": [s["episode"] for s in short_episodes if s["status"] == "警告"],
            "final_episode_count": len(episodes_info) - len([s for s in short_episodes if s["status"] == "剔除"]),
        }
    }
    
    return report


def print_report(report: Dict):
    """打印清理报告"""
    print("\n" + "="*70)
    print("数据清理预览报告 (v1.0)")
    print("="*70)
    
    print(f"\n[数据源] {report['data_source']}")
    print(f"[分析时间] {report['analysis_timestamp']}")
    
    print("\n" + "-"*70)
    print("📊 统计概览")
    print("-"*70)
    s = report["summary"]
    print(f"  扫描episodes: {s['total_episodes_scanned']}")
    print(f"  通过基础检查: {s['episodes_passed_basic_check']}")
    print(f"  存在问题: {s['episodes_with_issues']}")
    
    print("\n" + "-"*70)
    print("📈 帧数统计")
    print("-"*70)
    st = report["statistics"]
    print(f"  平均每episode帧数: {st['avg_frames_per_episode']:.1f}")
    print(f"  最少帧数: {st['min_frames']}")
    print(f"  最多帧数: {st['max_frames']}")
    print(f"  剔除阈值 (<{st['frame_threshold']:.1f}帧)")
    print(f"  警告阈值 (<{st['frame_warning_threshold']:.1f}帧)")
    
    print("\n" + "-"*70)
    print("🖼️  分辨率分布")
    print("-"*70)
    print(f"  左手: {st['resolution_left']}")
    print(f"  右手: {st['resolution_right']}")
    print(f"  第三视角: {st['resolution_third']}")
    
    print("\n" + "-"*70)
    print("⚠️  问题详情")
    print("-"*70)
    
    issues = report["issues"]
    
    if issues["incomplete_files"]["count"] > 0:
        print(f"\n  [文件不完整] {issues['incomplete_files']['count']}个")
        for item in issues["incomplete_files"]["episodes"]:
            print(f"    - {item['episode']}: 缺失 {item['missing']}")
    
    if issues["json_errors"]["count"] > 0:
        print(f"\n  [JSON错误] {issues['json_errors']['count']}个")
        for item in issues["json_errors"]["episodes"]:
            print(f"    - {item['episode']}: {item['error']}")
    
    if issues["video_errors"]["count"] > 0:
        print(f"\n  [视频错误] {issues['video_errors']['count']}个")
        for item in issues["video_errors"]["episodes"]:
            print(f"    - {item['episode']} ({item['view']})")
    
    if issues["short_episodes"]["count"] > 0:
        print(f"\n  [帧数异常] {issues['short_episodes']['count']}个")
        print(f"    建议剔除: {issues['short_episodes']['to_remove']}个")
        print(f"    建议警告: {issues['short_episodes']['to_warn']}个")
        print("    详情 (前10个):")
        for item in issues["short_episodes"]["episodes"]:
            print(f"      - {item['episode']}: {item['frames']:.0f}帧 [{item['status']}] {item['reason']}")
    
    if issues["sync_issues"]["count"] > 0:
        print(f"\n  [同步问题] {issues['sync_issues']['count']}个")
        for item in issues["sync_issues"]["episodes"]:
            print(f"    - {item['episode']}: {item['issue']} (差异{item['diff']:.0f}帧)")
    
    print("\n" + "-"*70)
    print("🧹 清理预览")
    print("-"*70)
    cp = report["cleaning_preview"]
    print(f"  当前episodes: {report['summary']['episodes_passed_basic_check']}")
    print(f"  建议剔除: {len(cp['episodes_to_remove'])}")
    print(f"  清理后剩余: {cp['final_episode_count']}")
    
    if cp["episodes_to_remove"]:
        print(f"\n  待剔除列表: {cp['episodes_to_remove'][:10]}")
        if len(cp["episodes_to_remove"]) > 10:
            print(f"  ... 还有 {len(cp['episodes_to_remove'])-10} 个")
    
    print("\n" + "="*70)
    print("注意: 此为预览报告，未实际删除任何文件")
    print("="*70)


def main():
    data_dir = Path("/Users/chenshuaiwen/Downloads/handover_umi_0422")
    
    print("[开始] 按照数据合并与清理规范 v1.0 进行分析...")
    report = analyze_data(data_dir)
    print_report(report)
    
    # 保存报告
    report_path = Path("/Users/chenshuaiwen/Desktop/openpi/docs/cleaning_report_0422_preview.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n[报告已保存] {report_path}")


if __name__ == "__main__":
    main()