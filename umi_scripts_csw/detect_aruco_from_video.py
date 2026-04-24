#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ArUco detection from third-view video (no third.json required).

This script processes third-view MP4 videos to detect ArUco markers
directly without requiring pre-existing third.json timestamp files.

Usage:
    python detect_aruco_from_video.py --stage1_dir /path/to/data [--max_frames 100]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import cv2
import imageio.v3 as iio

# Add src to path for pose utilities
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from openpi.policies.pose_util import pose7_to_mat, mat_to_pose7


# Camera intrinsics (D435, 1280x720)
CAMERA_INTRINSICS = {
    "fx": 919.6441040039062,
    "fy": 919.8379516601562,
    "cx": 648.0823974609375,
    "cy": 344.23870849609375,
    "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
}

# ArUco configuration
ARUCO_CONFIG = {
    "dictionary": cv2.aruco.DICT_4X4_50,
    "marker_size_m": 0.02,  # 2cm
    "left_id": 1,
    "right_id": 2,
}


def get_camera_matrix():
    """Build camera intrinsic matrix."""
    K = np.array([
        [CAMERA_INTRINSICS["fx"], 0, CAMERA_INTRINSICS["cx"]],
        [0, CAMERA_INTRINSICS["fy"], CAMERA_INTRINSICS["cy"]],
        [0, 0, 1]
    ], dtype=np.float32)
    return K


def detect_aruco_in_frame(frame, aruco_dict, aruco_params, K, dist_coeffs, marker_size):
    """Detect ArUco markers in a single frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Create detector (new OpenCV API)
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    # Detect markers
    corners, ids, rejected = detector.detectMarkers(gray)
    
    if ids is None or len(ids) == 0:
        return None, None
    
    # Estimate pose for each marker using solvePnP
    # Create 3D marker corners (in marker coordinate system)
    half_size = marker_size / 2.0
    obj_points = np.array([
        [-half_size, half_size, 0],
        [half_size, half_size, 0],
        [half_size, -half_size, 0],
        [-half_size, -half_size, 0]
    ], dtype=np.float32)
    
    rvecs = []
    tvecs = []
    for corner in corners:
        img_points = corner[0].astype(np.float32)
        _, rvec, tvec = cv2.solvePnP(
            obj_points, img_points, K, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        rvecs.append(rvec)
        tvecs.append(tvec)
    
    results = {}
    for i, marker_id in enumerate(ids.flatten()):
        rvec = rvecs[i].flatten()
        tvec = tvecs[i].flatten()
        
        # Convert to pose7: [x, y, z, qx, qy, qz, qw]
        R_mat, _ = cv2.Rodrigues(rvec)
        
        # Rotation matrix to quaternion
        q = rotation_matrix_to_quaternion(R_mat)
        
        results[int(marker_id)] = {
            "translation": tvec.tolist(),
            "quaternion": q.tolist(),  # [qx, qy, qz, qw]
            "rotation_matrix": R_mat.tolist(),
        }
    
    return results, corners


def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion [qx, qy, qz, qw]."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return np.array([qx, qy, qz, qw], dtype=np.float32)


def process_episode(stage1_dir: Path, episode_name: str, max_frames: int = 100, debug: bool = False):
    """Process a single episode."""
    third_mp4 = stage1_dir / f"{episode_name}_third.mp4"
    
    if not third_mp4.exists():
        print(f"  [SKIP] {episode_name}: third.mp4 not found")
        return None
    
    # Read video
    try:
        reader = iio.imiter(third_mp4)
    except Exception as e:
        print(f"  [ERROR] {episode_name}: Failed to read video - {e}")
        return None
    
    # Setup ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_CONFIG["dictionary"])
    aruco_params = cv2.aruco.DetectorParameters()
    K = get_camera_matrix()
    dist_coeffs = np.array(CAMERA_INTRINSICS["distortion"], dtype=np.float32)
    marker_size = ARUCO_CONFIG["marker_size_m"]
    
    left_id = ARUCO_CONFIG["left_id"]
    right_id = ARUCO_CONFIG["right_id"]
    
    left_data = None
    right_data = None
    detection_frame_left = -1
    detection_frame_right = -1
    
    # Process frames
    for frame_idx, frame in enumerate(reader):
        if frame_idx >= max_frames:
            break
        
        frame_rgb = np.asarray(frame, dtype=np.uint8)
        
        # Detect markers
        results, corners = detect_aruco_in_frame(
            frame_rgb, aruco_dict, aruco_params, K, dist_coeffs, marker_size
        )
        
        if results is None:
            continue
        
        # Check for left marker
        if left_id in results and left_data is None:
            left_data = results[left_id]
            detection_frame_left = frame_idx
            print(f"  [{episode_name}] Left marker (ID={left_id}) detected at frame {frame_idx}")
        
        # Check for right marker
        if right_id in results and right_data is None:
            right_data = results[right_id]
            detection_frame_right = frame_idx
            print(f"  [{episode_name}] Right marker (ID={right_id}) detected at frame {frame_idx}")
        
        # Stop if both found
        if left_data is not None and right_data is not None:
            print(f"  [{episode_name}] Both markers found, stopping at frame {frame_idx}")
            break
    
    # Prepare output
    if left_data is None and right_data is None:
        print(f"  [WARN] {episode_name}: No markers detected in {max_frames} frames")
        return None
    
    output = {}
    
    if left_data:
        output["left_in_cam"] = left_data
    if right_data:
        output["right_in_cam"] = right_data
    
    output["metadata"] = {
        "detection_frame_left": detection_frame_left,
        "detection_frame_right": detection_frame_right,
        "marker_size_m": marker_size,
        "camera_intrinsics": CAMERA_INTRINSICS,
    }
    
    return output


def main(stage1_dir: str, max_frames: int = 100, debug: bool = False):
    stage1_path = Path(stage1_dir)
    
    # Find all episodes
    left_jsons = sorted(stage1_path.glob("episode_*_left.json"))
    
    print(f"[INFO] Found {len(left_jsons)} episodes to process")
    print(f"[INFO] Max frames to check per episode: {max_frames}")
    
    success_count = 0
    fail_count = 0
    
    for left_json in left_jsons:
        episode_name = left_json.stem.replace("_left", "")
        
        # Check if already processed
        third_json = stage1_path / f"{episode_name}_third.json"
        if third_json.exists():
            print(f"[SKIP] {episode_name}: Already has third.json")
            success_count += 1
            continue
        
        # Process
        result = process_episode(stage1_path, episode_name, max_frames, debug)
        
        if result:
            # Save to third.json
            third_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"[OK] {episode_name}: Saved to {third_json.name}")
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n{'='*60}")
    print(f"ArUco Detection Summary")
    print(f"{'='*60}")
    print(f"Total episodes: {len(left_jsons)}")
    print(f"Success: {success_count}")
    print(f"Failed/No markers: {fail_count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Detect ArUco markers from video")
    p.add_argument("--stage1_dir", required=True, help="Directory containing episode files")
    p.add_argument("--max_frames", type=int, default=100, help="Max frames to check per episode")
    p.add_argument("--debug", action="store_true", help="Save debug visualizations")
    args = p.parse_args()
    
    main(
        stage1_dir=args.stage1_dir,
        max_frames=args.max_frames,
        debug=args.debug,
    )