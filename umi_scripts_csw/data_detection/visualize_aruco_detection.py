#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize ArUco detection on third-view camera.

This script reads third.mp4 and third.json to visualize:
- ArUco marker detection boxes
- Coordinate axes (X=red, Y=green, Z=blue)
- Left (ID=1) and Right (ID=2) hand labels
- Detection frame numbers

Usage:
    python visualize_aruco_detection.py --data_dir ~/Downloads/handover_umi_0422_aruco --output_dir ./preview
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import imageio.v3 as iio


def draw_axis(img, cam_matrix, dist_coeffs, rvec, tvec, axis_length=0.05):
    """Draw 3D coordinate axes on image."""
    points = np.float32([
        [0, 0, 0],           # origin
        [axis_length, 0, 0], # X axis
        [0, axis_length, 0], # Y axis
        [0, 0, axis_length]  # Z axis
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


def visualize_episode(episode_dir: Path, episode_name: str, output_dir: Path, camera_intrinsics: dict):
    """Visualize ArUco detection for a single episode."""
    
    # Paths
    third_mp4 = episode_dir / f"{episode_name}_third.mp4"
    third_json = episode_dir / f"{episode_name}_third.json"
    
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
    if "left_in_cam" not in data or "right_in_cam" not in data:
        print(f"[SKIP] {episode_name}: No ArUco data in third.json")
        return None
    
    # Get detection frames
    metadata = data.get("metadata", {})
    detection_frame_left = metadata.get("detection_frame_left", 0)
    detection_frame_right = metadata.get("detection_frame_right", 0)
    
    # Extract poses [x,y,z,qx,qy,qz,qw] from structured JSON
    left_data = data["left_in_cam"]
    right_data = data["right_in_cam"]
    
    # Build pose7: [x,y,z, qx,qy,qz,qw]
    left_pose = np.array([
        *left_data["translation"],  # x,y,z
        *left_data["quaternion"]     # qx,qy,qz,qw
    ], dtype=np.float32)
    
    right_pose = np.array([
        *right_data["translation"],  # x,y,z
        *right_data["quaternion"]    # qx,qy,qz,qw
    ], dtype=np.float32)
    
    # Camera matrix
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
    frames_to_capture = list(set([detection_frame_left, detection_frame_right]))
    frames_dict = {}
    
    reader = iio.imiter(third_mp4)
    for i, frame in enumerate(reader):
        if i in frames_to_capture:
            frames_dict[i] = np.asarray(frame, dtype=np.uint8)
            frames_to_capture.remove(i)
            if not frames_to_capture:
                break
    
    if not frames_dict:
        print(f"[WARN] {episode_name}: Could not read detection frames")
        return None
    
    # Create visualization
    results = []
    
    for frame_idx in sorted(frames_dict.keys()):
        img = frames_dict[frame_idx].copy()
        
        # Determine which hands are detected in this frame
        hands_in_frame = []
        if frame_idx == detection_frame_left:
            hands_in_frame.append(("left", left_pose))
        if frame_idx == detection_frame_right:
            hands_in_frame.append(("right", right_pose))
        
        for hand_name, pose in hands_in_frame:
            # Convert pose7 to rvec/tvec
            pos = pose[:3]
            quat = pose[3:]  # [qx, qy, qz, qw]
            
            # Convert quaternion to rotation matrix then to rotation vector
            # scipy might not be available, use simple method
            qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
            
            # Rotation matrix from quaternion
            R = np.array([
                [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
                [2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)],
                [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)]
            ], dtype=np.float32)
            
            # Convert to rotation vector
            rvec, _ = cv2.Rodrigues(R)
            tvec = pos.reshape(3, 1)
            
            # Draw coordinate axes
            img = draw_axis(img, cam_matrix, dist_coeffs, rvec, tvec, axis_length=0.02)
            
            # Draw label
            label = f"{hand_name.upper()} (ID={1 if hand_name=='left' else 2})"
            # Get projected position for label
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
                       (0, 0, 255) if hand_name == "left" else (255, 0, 0), 2)
        
        # Add frame number
        cv2.putText(img, f"Frame {frame_idx}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        results.append((frame_idx, img))
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for frame_idx, img in results:
        output_path = output_dir / f"{episode_name}_frame{frame_idx:03d}.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    # Create combined view if both hands detected on different frames
    if len(results) == 2 and detection_frame_left != detection_frame_right:
        # Create side-by-side comparison
        img_left = results[0][1] if results[0][0] == detection_frame_left else results[1][1]
        img_right = results[0][1] if results[0][0] == detection_frame_right else results[1][1]
        
        # Resize to same height
        h = min(img_left.shape[0], img_right.shape[0])
        img_left = cv2.resize(img_left, (img_left.shape[1] * h // img_left.shape[0], h))
        img_right = cv2.resize(img_right, (img_right.shape[1] * h // img_right.shape[0], h))
        
        combined = np.hstack([img_left, img_right])
        combined_path = output_dir / f"{episode_name}_combined.png"
        cv2.imwrite(str(combined_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    
    return {
        "episode": episode_name,
        "left_frame": detection_frame_left,
        "right_frame": detection_frame_right,
        "saved_images": len(results)
    }


def main():
    parser = argparse.ArgumentParser(description="Visualize ArUco detection")
    parser.add_argument("--data_dir", required=True, help="Directory containing MP4+JSON files")
    parser.add_argument("--output_dir", default="./preview_aruco", help="Output directory for visualizations")
    parser.add_argument("--sample", type=int, default=0, help="Only process N episodes (0=all)")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Camera intrinsics (D435, 1280x720)
    camera_intrinsics = {
        "fx": 919.6441040039062,
        "fy": 919.8379516601562,
        "cx": 648.0823974609375,
        "cy": 344.23870849609375,
        "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
    }
    
    # Find all episodes
    left_jsons = sorted(data_dir.glob("episode_*_left.json"))
    
    if args.sample > 0:
        left_jsons = left_jsons[:args.sample]
    
    print(f"[INFO] Found {len(left_jsons)} episodes to process")
    print(f"[INFO] Output directory: {output_dir}")
    
    results = []
    for left_json in left_jsons:
        episode_name = left_json.stem.replace("_left", "")
        result = visualize_episode(data_dir, episode_name, output_dir, camera_intrinsics)
        if result:
            results.append(result)
            print(f"[OK] {episode_name}: left@{result['left_frame']}, right@{result['right_frame']} ({result['saved_images']} images)")
    
    # Summary
    print("\n" + "="*60)
    print("ArUco Detection Summary")
    print("="*60)
    for r in results:
        same_frame = "SAME" if r['left_frame'] == r['right_frame'] else "DIFF"
        print(f"{r['episode']}: left@frame{r['left_frame']:3d}, right@frame{r['right_frame']:3d} [{same_frame}]")
    print("="*60)
    print(f"Total: {len(results)} episodes with ArUco data")
    print(f"Images saved to: {output_dir}")


if __name__ == "__main__":
    main()