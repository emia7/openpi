#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize ArUco detection results on third-view camera frames.

Usage:
    python visualize_aruco_results.py --data_dir ~/Downloads/handover_umi_0422 --output_dir ./preview --sample 5
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import imageio.v3 as iio


def draw_axis(img, cam_matrix, dist_coeffs, rvec, tvec, axis_length=0.02):
    """Draw 3D coordinate axes on image."""
    points = np.float32([
        [0, 0, 0],           # origin
        [axis_length, 0, 0], # X axis (red)
        [0, axis_length, 0], # Y axis (green)
        [0, 0, axis_length]  # Z axis (blue)
    ]).reshape(-1, 3)
    
    imgpts, _ = cv2.projectPoints(points, rvec, tvec, cam_matrix, dist_coeffs)
    imgpts = imgpts.astype(int)
    
    origin = tuple(imgpts[0].ravel())
    
    # X axis - Red
    cv2.line(img, origin, tuple(imgpts[1].ravel()), (0, 0, 255), 3)
    # Y axis - Green
    cv2.line(img, origin, tuple(imgpts[2].ravel()), (0, 255, 0), 3)
    # Z axis - Blue
    cv2.line(img, origin, tuple(imgpts[3].ravel()), (255, 0, 0), 3)
    
    return img


def visualize_episode(data_dir: Path, episode_name: str, output_dir: Path):
    """Visualize ArUco detection for a single episode."""
    
    third_mp4 = data_dir / f"{episode_name}_third.mp4"
    third_json = data_dir / f"{episode_name}_third.json"
    
    if not third_mp4.exists():
        print(f"[SKIP] {episode_name}: third.mp4 not found")
        return None
    
    if not third_json.exists():
        print(f"[SKIP] {episode_name}: third.json not found")
        return None
    
    # Load JSON
    with open(third_json, 'r') as f:
        data = json.load(f)
    
    # Check if ArUco data exists
    has_left = "left_in_cam" in data
    has_right = "right_in_cam" in data
    
    if not has_left and not has_right:
        print(f"[SKIP] {episode_name}: No ArUco detection data")
        return None
    
    # Get detection frames
    metadata = data.get("metadata", {})
    detection_frame_left = metadata.get("detection_frame_left", -1)
    detection_frame_right = metadata.get("detection_frame_right", -1)
    
    # Camera matrix
    camera_intrinsics = metadata.get("camera_intrinsics", {
        "fx": 919.6441040039062,
        "fy": 919.8379516601562,
        "cx": 648.0823974609375,
        "cy": 344.23870849609375,
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
    })
    
    fx = camera_intrinsics["fx"]
    fy = camera_intrinsics["fy"]
    cx = camera_intrinsics["cx"]
    cy = camera_intrinsics["cy"]
    cam_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    dist_coeffs = np.array(camera_intrinsics.get("distortion", [0,0,0,0,0]), dtype=np.float32)
    
    # Read detection frames
    frames_to_capture = []
    if has_left and detection_frame_left >= 0:
        frames_to_capture.append((detection_frame_left, "left"))
    if has_right and detection_frame_right >= 0:
        frames_to_capture.append((detection_frame_right, "right"))
    
    if not frames_to_capture:
        print(f"[SKIP] {episode_name}: No valid detection frames")
        return None
    
    # Read video frames
    frames_dict = {}
    reader = iio.imiter(third_mp4)
    for i, frame in enumerate(reader):
        for target_frame, hand in frames_to_capture:
            if i == target_frame:
                frames_dict[hand] = np.asarray(frame, dtype=np.uint8)
        if len(frames_dict) == len(frames_to_capture):
            break
    
    if not frames_dict:
        print(f"[WARN] {episode_name}: Could not read detection frames")
        return None
    
    # Create visualization
    results = []
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for hand, img_rgb in frames_dict.items():
        img = img_rgb.copy()
        
        # Get pose data
        pose_data = data.get(f"{hand}_in_cam", {})
        if not pose_data:
            continue
        
        # Extract pose
        translation = np.array(pose_data.get("translation", [0,0,0]), dtype=np.float32)
        quaternion = np.array(pose_data.get("quaternion", [0,0,0,1]), dtype=np.float32)  # [qx,qy,qz,qw]
        
        # Convert to rvec/tvec
        tvec = translation.reshape(3, 1)
        
        # Quaternion to rotation matrix
        qx, qy, qz, qw = quaternion
        R_mat = np.array([
            [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
            [2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)],
            [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)]
        ], dtype=np.float32)
        
        rvec, _ = cv2.Rodrigues(R_mat)
        
        # Draw coordinate axes
        img = draw_axis(img, cam_matrix, dist_coeffs, rvec, tvec, axis_length=0.02)
        
        # Draw label
        label = f"{hand.upper()} (ID={1 if hand=='left' else 2})"
        origin_proj, _ = cv2.projectPoints(
            np.float32([[0,0,0]]), rvec, tvec, cam_matrix, dist_coeffs
        )
        origin_px = tuple(origin_proj[0].ravel().astype(int))
        
        # Draw label background
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, 
                     (origin_px[0], origin_px[1] - text_h - 10),
                     (origin_px[0] + text_w, origin_px[1]),
                     (255, 255, 255), -1)
        cv2.putText(img, label, 
                   (origin_px[0], origin_px[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 0, 255) if hand == "left" else (255, 0, 0), 2)
        
        # Add frame number
        frame_num = detection_frame_left if hand == "left" else detection_frame_right
        cv2.putText(img, f"Frame {frame_num}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Save
        output_path = output_dir / f"{episode_name}_{hand}_frame{frame_num:03d}.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        results.append((hand, frame_num, output_path))
    
    return {
        "episode": episode_name,
        "results": results,
        "has_left": has_left,
        "has_right": has_right,
    }


def main():
    parser = argparse.ArgumentParser(description="Visualize ArUco detection results")
    parser.add_argument("--data_dir", required=True, help="Directory containing MP4+JSON files")
    parser.add_argument("--output_dir", default="./preview_aruco", help="Output directory for visualizations")
    parser.add_argument("--sample", type=int, default=10, help="Only process N episodes (0=all)")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Find all episodes with third.json
    third_jsons = sorted(data_dir.glob("episode_*_third.json"))
    
    if args.sample > 0:
        third_jsons = third_jsons[:args.sample]
    
    print(f"[INFO] Found {len(third_jsons)} episodes with ArUco data to process")
    print(f"[INFO] Output directory: {output_dir}")
    
    success_count = 0
    
    for third_json in third_jsons:
        episode_name = third_json.stem.replace("_third", "")
        result = visualize_episode(data_dir, episode_name, output_dir)
        if result:
            success_count += 1
            hands_str = ",".join([r[0] for r in result["results"]])
            print(f"[OK] {episode_name}: {hands_str}")
    
    print(f"\n{'='*60}")
    print(f"Visualization Summary")
    print(f"{'='*60}")
    print(f"Total processed: {len(third_jsons)}")
    print(f"Successfully visualized: {success_count}")
    print(f"Images saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()