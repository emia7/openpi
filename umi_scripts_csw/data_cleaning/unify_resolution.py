#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一数据集视频分辨率

将低分辨率视频提升到目标分辨率 (使用最近邻插值保持清晰度)

Usage:
    python unify_resolution.py \
        --data_dir ~/Downloads/handover_umi_0423_mix \
        --target_resolution 720 1280 \
        --views third \
        --output_dir ~/Downloads/handover_umi_0423_mix_unified
"""

import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple
import imageio.v3 as iio
import numpy as np
from tqdm import tqdm


def get_video_resolution(video_path: Path) -> Tuple[int, int]:
    """获取视频分辨率 (H, W)"""
    try:
        frame = iio.imread(video_path, index=0)
        return frame.shape[:2]  # (H, W)
    except:
        return (0, 0)


def needs_resize(video_path: Path, target_h: int, target_w: int) -> bool:
    """检查视频是否需要调整分辨率"""
    current_h, current_w = get_video_resolution(video_path)
    return (current_h, current_w) != (target_h, target_w)


def resize_video_python(input_path: Path, output_path: Path,
                       target_h: int, target_w: int) -> bool:
    """
    使用Python (imageio + numpy) 调整视频分辨率
    不依赖ffmpeg和skimage
    """
    try:
        import imageio.v3 as iio
        import numpy as np
        
        # 读取视频
        frames = iio.imread(input_path)
        
        # 处理每一帧 - 使用简单的双线性插值
        resized_frames = []
        for frame in frames:
            # 转换为float进行插值
            frame_float = frame.astype(np.float32)
            
            # 简单的双线性插值
            h, w = frame.shape[:2]
            
            # 计算缩放比例
            row_scale = h / target_h
            col_scale = w / target_w
            
            # 生成目标坐标
            row_idx = (np.arange(target_h) * row_scale).astype(np.int32)
            col_idx = (np.arange(target_w) * col_scale).astype(np.int32)
            
            # 限制范围
            row_idx = np.clip(row_idx, 0, h - 1)
            col_idx = np.clip(col_idx, 0, w - 1)
            
            # 采样
            resized = frame_float[row_idx[:, None], col_idx[None, :]]
            resized_frames.append(resized.astype(np.uint8))
        
        # 保存视频
        iio.imwrite(output_path, resized_frames, fps=10.0)
        
        return True
    except Exception as e:
        print(f"[ERROR] Python resize failed: {e}")
        return False


def unify_dataset_resolution(data_dir: Path, output_dir: Path,
                            target_resolution: Tuple[int, int],
                            views: List[str]) -> dict:
    """
    统一数据集分辨率
    
    Returns:
        处理统计信息
    """
    target_h, target_w = target_resolution
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有episodes
    json_files = sorted(data_dir.glob("episode_*_left.json"))
    total_episodes = len(json_files)
    
    print(f"[INFO] 处理 {total_episodes} 个episodes")
    print(f"[INFO] 目标分辨率: {target_h}×{target_w}")
    print(f"[INFO] 需要处理的视角: {views}")
    
    stats = {
        "total_episodes": total_episodes,
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "resized_videos": 0
    }
    
    for json_file in tqdm(json_files, desc="处理episodes"):
        episode_name = json_file.stem.replace("_left", "")
        
        # 复制所有非视频文件
        for suffix in ["_left.json", "_right.json", "_alignment.png", 
                      "_left_states.png", "_right_states.png"]:
            src = data_dir / f"{episode_name}{suffix}"
            dst = output_dir / f"{episode_name}{suffix}"
            if src.exists():
                # 对于JSON文件，直接复制
                import shutil
                shutil.copy2(src, dst)
        
        # 处理各个视角的视频
        for view in ["left", "right", "third"]:
            src_video = data_dir / f"{episode_name}_{view}.mp4"
            dst_video = output_dir / f"{episode_name}_{view}.mp4"
            
            if not src_video.exists():
                continue
            
            # 检查是否需要调整分辨率
            if view in views and needs_resize(src_video, target_h, target_w):
                # 需要调整分辨率
                success = resize_video_python(src_video, dst_video, target_h, target_w)
                if success:
                    stats["resized_videos"] += 1
                else:
                    stats["failed"] += 1
                    # 失败时复制原文件
                    import shutil
                    shutil.copy2(src_video, dst_video)
            else:
                # 不需要调整，直接复制
                import shutil
                shutil.copy2(src_video, dst_video)
        
        stats["processed"] += 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="统一数据集视频分辨率")
    parser.add_argument("--data_dir", required=True, help="输入数据目录")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--target_resolution", nargs=2, type=int, 
                       default=[720, 1280], help="目标分辨率 (H W)")
    parser.add_argument("--views", nargs='+', default=["third"],
                       help="需要处理的视角 (left right third)")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    target_resolution = tuple(args.target_resolution)
    
    if not data_dir.exists():
        print(f"[ERROR] 数据目录不存在: {data_dir}")
        return
    
    # 检查必要的Python包
    try:
        import numpy as np
    except ImportError:
        print("[ERROR] 需要安装numpy")
        return
    
    # 执行统一分辨率
    stats = unify_dataset_resolution(
        data_dir, output_dir, target_resolution, args.views
    )
    
    print("\n" + "="*70)
    print("处理完成!")
    print("="*70)
    print(f"总episodes: {stats['total_episodes']}")
    print(f"成功处理: {stats['processed']}")
    print(f"调整分辨率: {stats['resized_videos']} 个视频")
    print(f"失败: {stats['failed']}")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()