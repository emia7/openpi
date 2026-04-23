#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
执行数据清理 - 删除异常episodes并重新编号

按照数据合并与清理规范 v1.1 执行实际清理操作：
1. 读取异常检测报告
2. 删除指定的episodes
3. 重新编号（从1开始连续）
4. 生成清理日志

Usage:
    python execute_data_cleaning.py \
        --data_dir ~/Downloads/handover_umi_0422 \
        --anomaly_report ./anomaly_reports/trajectory_anomaly_report.json \
        --severity critical \
        --output_dir ~/Downloads/handover_umi_0422_cleaned
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import List, Dict
from datetime import datetime


def load_anomaly_report(report_path: Path) -> Dict:
    """加载异常检测报告"""
    with open(report_path) as f:
        return json.load(f)


def get_episodes_to_remove(report: Dict, severity: str) -> List[str]:
    """根据严重程度获取要删除的episodes列表"""
    classified = report.get("classified_reports", {})
    
    if severity == "critical":
        return [r["episode"] for r in classified.get("critical", [])]
    elif severity == "warning":
        return [r["episode"] for r in classified.get("warning", [])]
    elif severity == "minor":
        return [r["episode"] for r in classified.get("minor", [])]
    elif severity == "all":
        # 删除所有非正常的
        critical = [r["episode"] for r in classified.get("critical", [])]
        warning = [r["episode"] for r in classified.get("warning", [])]
        minor = [r["episode"] for r in classified.get("minor", [])]
        return critical + warning + minor
    else:
        return []


def delete_episode(data_dir: Path, episode_name: str, dry_run: bool = True) -> List[str]:
    """
    删除单个episode的所有文件
    
    Returns:
        删除的文件列表
    """
    files_to_delete = [
        f"{episode_name}_left.mp4",
        f"{episode_name}_left.json",
        f"{episode_name}_right.mp4",
        f"{episode_name}_right.json",
        f"{episode_name}_third.mp4",
        f"{episode_name}_alignment.png",
        f"{episode_name}_left_states.png",
        f"{episode_name}_right_states.png",
    ]
    
    deleted_files = []
    for filename in files_to_delete:
        file_path = data_dir / filename
        if file_path.exists():
            if not dry_run:
                file_path.unlink()
                deleted_files.append(str(file_path))
            else:
                deleted_files.append(f"[DRY-RUN] {file_path}")
    
    return deleted_files


def renumber_episodes(data_dir: Path, episodes_to_keep: List[str], 
                      output_dir: Path, dry_run: bool = True) -> Dict:
    """
    重新编号episodes（从1开始连续）
    
    Returns:
        编号映射表 {old_name: new_name}
    """
    mapping = {}
    new_number = 1
    
    # 按原顺序排序
    episodes_to_keep = sorted(episodes_to_keep)
    
    for old_name in episodes_to_keep:
        # 新编号格式: episode_000001
        new_name = f"episode_{new_number:06d}"
        mapping[old_name] = new_name
        
        if not dry_run:
            # 实际重命名文件
            for suffix in [
                "_left.mp4", "_left.json",
                "_right.mp4", "_right.json",
                "_third.mp4",
                "_alignment.png",
                "_left_states.png", "_right_states.png"
            ]:
                old_file = data_dir / f"{old_name}{suffix}"
                new_file = output_dir / f"{new_name}{suffix}"
                if old_file.exists():
                    shutil.copy2(old_file, new_file)
        
        new_number += 1
    
    return mapping


def generate_cleaning_log(args, episodes_removed: List[str], 
                         episodes_kept: List[str], mapping: Dict,
                         total_original: int) -> Dict:
    """生成清理日志"""
    log = {
        "cleaning_version": "1.1",
        "timestamp": datetime.now().isoformat(),
        "original_data_dir": str(args.data_dir),
        "output_data_dir": str(args.output_dir),
        "anomaly_report": str(args.anomaly_report),
        "severity_threshold": args.severity,
        "dry_run": args.dry_run,
        "statistics": {
            "total_original": total_original,
            "removed": len(episodes_removed),
            "retained": len(episodes_kept),
            "final_count": len(episodes_kept)
        },
        "episodes_removed": episodes_removed,
        "renumber_mapping": mapping
    }
    
    return log


def main():
    parser = argparse.ArgumentParser(
        description="执行数据清理 - 删除异常episodes并重新编号",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 预览清理 (不实际删除)
    python execute_data_cleaning.py \\
        --data_dir ~/Downloads/handover_umi_0422 \\
        --anomaly_report ./anomaly_reports/trajectory_anomaly_report.json \\
        --severity critical \\
        --dry_run

    # 正式执行清理 (删除严重异常)
    python execute_data_cleaning.py \\
        --data_dir ~/Downloads/handover_umi_0422 \\
        --anomaly_report ./anomaly_reports/trajectory_anomaly_report.json \\
        --severity critical \\
        --output_dir ~/Downloads/handover_umi_0422_cleaned
        """
    )
    parser.add_argument("--data_dir", required=True, 
                       help="原始数据目录")
    parser.add_argument("--anomaly_report", required=True,
                       help="异常检测报告JSON路径")
    parser.add_argument("--severity", choices=["critical", "warning", "minor", "all"],
                       default="critical",
                       help="删除严重程度级别 (默认: critical)")
    parser.add_argument("--output_dir", 
                       help="输出目录 (默认: 原目录_cleaned)")
    parser.add_argument("--dry_run", action="store_true",
                       help="预览模式，不实际删除文件")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    report_path = Path(args.anomaly_report)
    
    # 验证路径
    if not data_dir.exists():
        print(f"[ERROR] 数据目录不存在: {data_dir}")
        return
    
    if not report_path.exists():
        print(f"[ERROR] 异常报告不存在: {report_path}")
        return
    
    # 加载报告
    print(f"[INFO] 加载异常报告: {report_path}")
    report = load_anomaly_report(report_path)
    
    # 获取要删除的episodes
    episodes_to_remove = get_episodes_to_remove(report, args.severity)
    total_episodes = report["summary"]["total_episodes"]
    
    # 获取所有原始episodes
    all_episodes = []
    for json_file in sorted(data_dir.glob("episode_*_left.json")):
        episode_name = json_file.stem.replace("_left", "")
        all_episodes.append(episode_name)
    
    # 计算保留的episodes
    episodes_to_keep = [ep for ep in all_episodes if ep not in episodes_to_remove]
    
    print("\n" + "="*70)
    print("数据清理执行预览" if args.dry_run else "数据清理执行")
    print("="*70)
    print(f"原始数据目录: {data_dir}")
    print(f"异常报告: {report_path}")
    print(f"严重程度阈值: {args.severity}")
    print(f"\n统计信息:")
    print(f"  总episodes: {total_episodes}")
    print(f"  待删除: {len(episodes_to_remove)}")
    print(f"  保留: {len(episodes_to_keep)}")
    print(f"  清理后: {len(episodes_to_keep)}")
    
    print(f"\n待删除列表 ({len(episodes_to_remove)}个):")
    for ep in episodes_to_remove[:10]:  # 只显示前10个
        print(f"  - {ep}")
    if len(episodes_to_remove) > 10:
        print(f"  ... 还有 {len(episodes_to_remove) - 10} 个")
    
    # 确认执行
    if not args.dry_run:
        print(f"\n[WARNING] 即将删除 {len(episodes_to_remove)} 个episodes并重新编号!")
        confirm = input("确认执行? (yes/no): ")
        if confirm.lower() != "yes":
            print("[CANCELLED] 操作已取消")
            return
    
    # 设置输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(str(data_dir) + "_cleaned")
    
    # 删除操作
    deleted_files_log = []
    for episode in episodes_to_remove:
        deleted = delete_episode(data_dir, episode, dry_run=args.dry_run)
        deleted_files_log.extend(deleted)
    
    print(f"\n[OK] 已删除 {len(episodes_to_remove)} 个episodes")
    
    # 创建输出目录并重新编号
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    mapping = renumber_episodes(
        data_dir, 
        episodes_to_keep, 
        output_dir, 
        dry_run=args.dry_run
    )
    
    print(f"[OK] 已重新编号 {len(episodes_to_keep)} 个episodes")
    print(f"[OK] 输出目录: {output_dir}")
    
    # 生成清理日志
    log = generate_cleaning_log(
        args, episodes_to_remove, episodes_to_keep, 
        mapping, total_episodes
    )
    
    # 保存日志
    if args.dry_run:
        log_path = data_dir.parent / f"cleaning_log_{args.severity}_dryrun.json"
    else:
        log_path = output_dir / "cleaning_log.json"
    
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] 清理日志已保存: {log_path}")
    
    print("\n" + "="*70)
    print("清理完成!" if not args.dry_run else "预览完成 (dry-run)")
    print("="*70)
    
    # 打印映射表预览
    print("\n编号映射预览 (前10个):")
    for old, new in list(mapping.items())[:10]:
        print(f"  {old} -> {new}")
    if len(mapping) > 10:
        print(f"  ... 还有 {len(mapping) - 10} 个")


if __name__ == "__main__":
    main()