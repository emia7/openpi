#!/usr/bin/env python3
"""Check internal consistency of datasets"""
import os
from pathlib import Path
import cv2

def get_video_info(video_path):
    """Get video resolution and size"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        size_mb = video_path.stat().st_size / (1024 * 1024)
        return {
            'resolution': (width, height),
            'fps': fps,
            'frames': frames,
            'size_mb': size_mb,
            'duration': frames / fps if fps > 0 else 0
        }
    except Exception as e:
        return None

def check_dataset_consistency(data_dir, name):
    """Check consistency within a dataset"""
    print(f"\n{'='*70}")
    print(f"Dataset: {name}")
    print(f"Path: {data_dir}")
    print('='*70)
    
    if not os.path.exists(data_dir):
        print(f"[ERROR] Directory not found!")
        return
    
    # Find all episodes
    left_mp4s = sorted(Path(data_dir).glob("episode_*_left.mp4"))
    total_eps = len(left_mp4s)
    print(f"Total episodes: {total_eps}\n")
    
    if total_eps == 0:
        return
    
    # Collect stats
    left_res = set()
    right_res = set()
    third_res = set()
    
    left_sizes = []
    right_sizes = []
    third_sizes = []
    
    anomalies = []
    
    for mp4 in left_mp4s:
        stem = mp4.stem.replace("_left", "")
        
        # Check left
        info = get_video_info(mp4)
        if info:
            left_res.add(info['resolution'])
            left_sizes.append(info['size_mb'])
        
        # Check right
        right_mp4 = mp4.parent / f"{stem}_right.mp4"
        if right_mp4.exists():
            info = get_video_info(right_mp4)
            if info:
                right_res.add(info['resolution'])
                right_sizes.append(info['size_mb'])
        
        # Check third
        third_mp4 = mp4.parent / f"{stem}_third.mp4"
        if third_mp4.exists():
            info = get_video_info(third_mp4)
            if info:
                third_res.add(info['resolution'])
                third_sizes.append(info['size_mb'])
    
    # Print resolution consistency
    print("--- Resolution Consistency ---")
    print(f"Left videos:  {len(left_res)} unique resolutions - {left_res}")
    print(f"Right videos: {len(right_res)} unique resolutions - {right_res}")
    print(f"Third videos: {len(third_res)} unique resolutions - {third_res}")
    
    if len(left_res) == 1:
        print("✅ Left: All same resolution")
    else:
        print("⚠️ Left: INCONSISTENT resolutions!")
        
    if len(right_res) == 1:
        print("✅ Right: All same resolution")
    else:
        print("⚠️ Right: INCONSISTENT resolutions!")
        
    if len(third_res) == 1:
        print("✅ Third: All same resolution")
    else:
        print("⚠️ Third: INCONSISTENT resolutions!")
    
    # Check size anomalies
    print("\n--- Size Anomalies ---")
    
    def check_anomalies(sizes, name):
        if not sizes:
            return
        avg = sum(sizes) / len(sizes)
        std = (sum((x - avg)**2 for x in sizes) / len(sizes))**0.5
        threshold = avg - 2*std  # 2 std below mean
        
        small_count = sum(1 for s in sizes if s < threshold)
        
        print(f"{name}: avg={avg:.2f}MB, std={std:.2f}MB")
        print(f"  Range: {min(sizes):.2f}MB ~ {max(sizes):.2f}MB")
        
        if small_count > 0:
            print(f"⚠️ {small_count} files unusually small (<{threshold:.2f}MB)")
            # Find the specific files
            for i, mp4 in enumerate(left_mp4s):
                if i < len(sizes) and sizes[i] < threshold:
                    print(f"    Small: {mp4.name} ({sizes[i]:.2f}MB)")
        else:
            print(f"✅ No size anomalies detected")
    
    check_anomalies(left_sizes, "Left")
    check_anomalies(right_sizes, "Right")
    check_anomalies(third_sizes, "Third")

def main():
    downloads = Path("/Users/chenshuaiwen/Downloads")
    
    datasets = [
        ("handover_umi_0416", "0416 Original"),
        ("handover_umi_0421", "0421 Original"),
        ("handover_umi_0420_mix", "0420 Mix"),
        ("handover_umi_0422_good_mix", "0422 Good Mix"),
        ("handover_umi_0422_mix", "0422 Mix"),
    ]
    
    for folder, name in datasets:
        path = downloads / folder
        if path.exists():
            check_dataset_consistency(path, name)
        else:
            print(f"\n[SKIP] {name}: {folder} not found")
    
    print("\n" + "="*70)
    print("SUMMARY: Check if each camera type has consistent resolution within dataset")
    print("="*70)

if __name__ == "__main__":
    main()