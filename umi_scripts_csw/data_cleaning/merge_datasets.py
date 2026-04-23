#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并多个数据集为一个新数据集，自动重新编号

Usage:
    python merge_datasets.py \
        --sources ~/Downloads/handover_umi_0422 ~/Downloads/handover_umi_0422_good_mix \
        --output ~/Downloads/handover_umi_0423_good_mix \
        --dataset_name "handover_umi_0423_good_mix"
"""

import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict


def get_episodes_from_dir(data_dir: Path) -> List[str]:
    """获取目录中的所有episodes"""
    json_files = sorted(data_dir.glob("episode_*_left.json"))
    return [f.stem.replace("_left", "") for f in json_files]


def copy_episode(source_dir: Path, target_dir: Path, 
                 old_name: str, new_name: str) -> Dict:
    """复制单个episode的所有文件到新名称"""
    files_copied = []
    
    for suffix in [
        "_left.mp4", "_left.json",
        "_right.mp4", "_right.json",
        "_third.mp4",
        "_alignment.png",
        "_left_states.png", "_right_states.png"
    ]:
        src_file = source_dir / f"{old_name}{suffix}"
        dst_file = target_dir / f"{new_name}{suffix}"
        if src_file.exists():
            shutil.copy2(src_file, dst_file)
            files_copied.append(str(dst_file))
    
    return {
        "old_name": old_name,
        "new_name": new_name,
        "files_copied": files_copied
    }


def merge_datasets(source_dirs: List[Path], output_dir: Path, 
                   dataset_name: str) -> Dict:
    """合并多个数据集"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集所有episodes
    all_episodes = []
    source_info = []
    
    for i, src_dir in enumerate(source_dirs):
        episodes = get_episodes_from_dir(src_dir)
        source_info.append({
            "source_dir": str(src_dir),
            "source_index": i,
            "num_episodes": len(episodes),
            "episodes": episodes
        })
        all_episodes.extend([(src_dir, ep) for ep in episodes])
    
    print(f"[INFO] 合并 {len(source_dirs)} 个数据源")
    print(f"[INFO] 总计 {len(all_episodes)} 个episodes")
    
    # 重新编号并复制
    mapping = []
    for new_idx, (src_dir, old_name) in enumerate(all_episodes, start=1):
        new_name = f"episode_{new_idx:06d}"
        info = copy_episode(src_dir, output_dir, old_name, new_name)
        mapping.append({
            **info,
            "source_dir": str(src_dir),
            "source_name": old_name
        })
        
        if new_idx % 50 == 0:
            print(f"[PROGRESS] 已处理 {new_idx}/{len(all_episodes)} episodes")
    
    # 生成合并日志
    log = {
        "merge_timestamp": datetime.now().isoformat(),
        "dataset_name": dataset_name,
        "output_dir": str(output_dir),
        "total_episodes": len(all_episodes),
        "sources": source_info,
        "mapping": mapping
    }
    
    log_file = output_dir / "merge_log.json"
    with open(log_file, 'w') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    
    print(f"[OK] 合并完成: {output_dir}")
    print(f"[OK] 合并日志: {log_file}")
    
    return log


def main():
    parser = argparse.ArgumentParser(description="合并多个数据集")
    parser.add_argument("--sources", nargs='+', required=True,
                       help="源数据目录列表 (按顺序)")
    parser.add_argument("--output", required=True,
                       help="输出目录")
    parser.add_argument("--dataset_name", required=True,
                       help="数据集名称")
    
    args = parser.parse_args()
    
    source_dirs = [Path(s) for s in args.sources]
    output_dir = Path(args.output)
    
    # 验证源目录
    for src_dir in source_dirs:
        if not src_dir.exists():
            print(f"[ERROR] 源目录不存在: {src_dir}")
            return
    
    # 执行合并
    log = merge_datasets(source_dirs, output_dir, args.dataset_name)
    
    print("\n" + "="*70)
    print("合并完成!")
    print("="*70)
    print(f"数据集名称: {args.dataset_name}")
    print(f"输出目录: {output_dir}")
    print(f"总episodes: {log['total_episodes']}")
    print(f"\n数据源:")
    for src in log['sources']:
        print(f"  - {src['source_dir']}: {src['num_episodes']} episodes")


if __name__ == "__main__":
    main()