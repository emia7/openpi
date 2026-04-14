"""
检查LeRobot数据集中相邻帧的action变化
帮助诊断模型预测多步相同的问题
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def check_parquet_actions(parquet_path: Path, num_frames: int = 100):
    """检查parquet文件中相邻帧的action变化"""
    print(f"[INFO] Loading parquet file: {parquet_path}")
    
    if not parquet_path.exists():
        print(f"[ERROR] File not found: {parquet_path}")
        return
    
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"[ERROR] Failed to load parquet: {e}")
        return
    
    print(f"[OK] Loaded {len(df)} frames")
    print(f"[INFO] Columns: {list(df.columns)}")
    
    # 检查是否有actions列
    if "actions" not in df.columns:
        print(f"[ERROR] No 'actions' column found!")
        print(f"[INFO] Available columns: {list(df.columns)}")
        return
    
    # 检查actions的形状
    first_act = df["actions"].iloc[0]
    print(f"\n[DEBUG] First frame action:")
    print(f"  Type: {type(first_act)}")
    print(f"  Shape: {np.asarray(first_act).shape}")
    print(f"  Values[:5]: {np.asarray(first_act)[:5]}")
    
    # 打印前10帧的action
    num_check = min(num_frames, len(df))
    print(f"\n[DEBUG] First {num_check} frames actions (first 3 values):")
    for i in range(num_check):
        act = np.asarray(df["actions"].iloc[i])
        print(f"  Frame {i}: {act[:3]}")
    
    # 计算相邻帧变化
    print(f"\n[DEBUG] Adjacent frame differences:")
    diffs = []
    for i in range(min(num_frames - 1, len(df) - 1)):
        act_curr = np.asarray(df["actions"].iloc[i])
        act_next = np.asarray(df["actions"].iloc[i+1])
        
        diff = np.abs(act_next - act_curr)
        max_diff = diff.max()
        mean_diff = diff.mean()
        diffs.append({
            "frame": i,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "action_curr": act_curr[:3],
            "action_next": act_next[:3],
        })
        
        if i < 10:  # 只打印前10个
            print(f"  Frame {i}->{i+1}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
            print(f"    curr[:3]: {act_curr[:3]}")
            print(f"    next[:3]: {act_next[:3]}")
    
    # 统计
    if diffs:
        max_diffs = [d["max_diff"] for d in diffs]
        mean_diffs = [d["mean_diff"] for d in diffs]
        
        print(f"\n[SUMMARY] Action change statistics (first {len(diffs)} frames):")
        print(f"  Max diff range: [{min(max_diffs):.6f}, {max(max_diffs):.6f}]")
        print(f"  Mean max_diff: {np.mean(max_diffs):.6f}")
        print(f"  Mean mean_diff: {np.mean(mean_diffs):.6f}")
        
        # 风险评级
        if np.mean(max_diffs) < 0.001:
            print(f"\n[WARNING] Very small action changes detected!")
            print(f"  This may cause the model to learn 'do nothing' behavior.")
        elif np.mean(max_diffs) < 0.01:
            print(f"\n[CAUTION] Small action changes detected.")
            print(f"  Consider checking data quality or increasing motion threshold.")
        else:
            print(f"\n[OK] Action changes look healthy.")


def find_parquet_files(data_dir: Path):
    """递归查找所有parquet文件"""
    parquet_dir = data_dir / "data"
    if not parquet_dir.exists():
        return []
    
    # 递归查找所有 .parquet 文件
    parquet_files = list(parquet_dir.rglob("*.parquet"))
    return sorted(parquet_files)


def check_episode_actions(data_dir: Path, episode_idx: int = 0):
    """检查特定episode的actions"""
    parquet_files = find_parquet_files(data_dir)
    
    if not parquet_files:
        print(f"[ERROR] No parquet files found in {data_dir / 'data'}")
        return
    
    print(f"[INFO] Found {len(parquet_files)} parquet files")
    print(f"[DEBUG] First few files: {[f.name for f in parquet_files[:3]]}")
    
    # 读取第一个文件
    check_parquet_actions(parquet_files[0])


def main():
    ap = argparse.ArgumentParser(description="检查LeRobot数据集actions")
    ap.add_argument("--data-dir", required=True, help="LeRobot数据集根目录")
    ap.add_argument("--num-frames", type=int, default=100, help="检查多少帧")
    args = ap.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}")
        return
    
    print(f"[INFO] Checking dataset: {data_dir}")
    
    # 检查目录结构
    print(f"\n[DEBUG] Directory structure:")
    for subdir in ["data", "meta", "videos"]:
        path = data_dir / subdir
        if path.exists():
            files = list(path.iterdir())[:5]  # 只显示前5个
            print(f"  {subdir}/: {len(list(path.iterdir()))} items")
            for f in files:
                print(f"    - {f.name}")
        else:
            print(f"  {subdir}/: NOT FOUND")
    
    # 检查parquet
    parquet_files = find_parquet_files(data_dir)
    if parquet_files:
        print(f"\n[INFO] Found {len(parquet_files)} parquet files")
        print(f"[DEBUG] First file: {parquet_files[0]}")
        check_parquet_actions(parquet_files[0], args.num_frames)
    else:
        print(f"[ERROR] No parquet files found in {data_dir / 'data'}")
        # 列出data目录内容帮助调试
        data_dir_path = data_dir / "data"
        if data_dir_path.exists():
            print(f"[DEBUG] Contents of {data_dir_path}:")
            for item in data_dir_path.iterdir():
                print(f"  - {item.name} {'(dir)' if item.is_dir() else '(file)'}")


if __name__ == "__main__":
    main()
