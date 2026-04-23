#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add ArUco marker detection results to episode JSON files.

This script processes the first frame of third-view videos to detect ArUco markers
on left and right grippers, estimates their 6D poses, and stores the results.

Usage:
    python add_aruco_pose_to_json.py --stage1_dir /path/to/stage1_data

Requirements:
    - OpenCV with aruco module
    - Camera intrinsics calibrated (see CAMERA_CALIBRATION below)
    - ArUco markers printed and attached to grippers:
        * Left gripper: ID=1, size=5cm
        * Right gripper: ID=2, size=5cm

Camera Calibration:
    The script needs camera intrinsics (K matrix and distortion coefficients).
    Options:
    1. Use RealSense SDK intrinsic from D435
    2. Run OpenCV checkerboard calibration (see calibrate_camera.py template below)
    3. Use default if already calibrated
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import cv2

# Add src to path for pose utilities
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from openpi.policies.pose_util import pose7_to_mat, mat_to_pose7


# =============================================================================
# CAMERA CONFIGURATION - MODIFY THIS SECTION
# =============================================================================

# Intel RealSense D435 calibrated intrinsics (1280x720, from ROS topic)
# Updated 2024-04-22 with actual camera parameters
CAMERA_INTRINSICS = {
    "fx": 919.6441040039062,    # focal length x (from /camera/color/camera_info)
    "fy": 919.8379516601562,    # focal length y
    "cx": 648.0823974609375,    # principal point x (image center)
    "cy": 344.23870849609375,   # principal point y
    "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],  # plumb_bob, minimal distortion
}

# ArUco marker configuration
ARUCO_CONFIG = {
    "dictionary": cv2.aruco.DICT_4X4_50,  # 4x4 marker, 50 IDs
    "marker_size_m": 0.02,  # 2cm marker size (reduced from 5cm for smaller grippers)
    "left_id": 1,  # ID for left gripper
    "right_id": 2,  # ID for right gripper
}

# Marker-to-gripper transform (if marker is not at gripper center)
# Default: identity (marker center = gripper center)
# If marker is offset from gripper center, specify the transform here
MARKER_TO_GRIPPER = {
    "left": np.eye(4),   # Replace with actual offset if needed
    "right": np.eye(4),  # Replace with actual offset if needed
}


def get_camera_matrix():
    """Build camera intrinsic matrix from configuration."""
    K = np.array([
        [CAMERA_INTRINSICS["fx"], 0, CAMERA_INTRINSICS["cx"]],
        [0, CAMERA_INTRINSICS["fy"], CAMERA_INTRINSICS["cy"]],
        [0, 0, 1]
    ], dtype=np.float32)
    return K


def detect_aruco_pose(
    frame: np.ndarray,
    marker_id: int,
    K: np.ndarray,
    dist_coeffs: np.ndarray
) -> tuple[np.ndarray, bool]:
    """
    Detect ArUco marker and estimate 6D pose.
    
    Args:
        frame: RGB image (H, W, 3)
        marker_id: ArUco marker ID to detect
        K: Camera intrinsic matrix (3x3)
        dist_coeffs: Distortion coefficients (5,)
    
    Returns:
        pose7: [x, y, z, qx, qy, qz, qw] in camera frame, or None if not detected
        success: bool
    """
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Load ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_CONFIG["dictionary"])
    
    # Create detector parameters (handle different OpenCV versions)
    try:
        # OpenCV 4.7+
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
    except AttributeError:
        # Older OpenCV versions
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    if ids is None or len(ids) == 0:
        return None, False
    
    # Find target marker
    target_idx = None
    for i, detected_id in enumerate(ids):
        if detected_id[0] == marker_id:
            target_idx = i
            break
    
    if target_idx is None:
        return None, False
    
    # Estimate pose
    marker_size = ARUCO_CONFIG["marker_size_m"]
    
    try:
        # OpenCV 4.7+
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            [corners[target_idx]],
            marker_size,
            K,
            dist_coeffs
        )
        rvec, tvec = rvecs[0], tvecs[0]
    except AttributeError:
        # Older versions - return format is different
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners[target_idx],
            marker_size,
            K,
            dist_coeffs
        )
        rvec, tvec = rvecs[0], tvecs[0]
    
    # Convert rvec/tvec to 4x4 matrix
    R_mat, _ = cv2.Rodrigues(rvec)
    T_cam_marker = np.eye(4, dtype=np.float32)
    T_cam_marker[:3, :3] = R_mat
    T_cam_marker[:3, 3] = tvec.flatten()
    
    # Apply marker-to-gripper offset if needed
    if marker_id == ARUCO_CONFIG["left_id"]:
        T_cam_gripper = T_cam_marker @ MARKER_TO_GRIPPER["left"]
    elif marker_id == ARUCO_CONFIG["right_id"]:
        T_cam_gripper = T_cam_marker @ MARKER_TO_GRIPPER["right"]
    else:
        T_cam_gripper = T_cam_marker
    
    # Convert to pose7
    pose7 = mat_to_pose7(T_cam_gripper)
    
    return pose7, True


def process_episode(
    stem: str,
    stage1_dir: Path,
    K: np.ndarray,
    dist_coeffs: np.ndarray,
    debug: bool = False
):
    """
    Process a single episode: detect ArUco markers and update JSON.
    
    Args:
        stem: Episode stem (e.g., "episode_000001")
        stage1_dir: Directory containing episode files
        K: Camera intrinsic matrix (3x3)
        dist_coeffs: Distortion coefficients
        debug: If True, save visualization images
    
    Returns:
        success: bool
    """
    third_mp4 = stage1_dir / f"{stem}_third.mp4"
    third_json = stage1_dir / f"{stem}_third.json"
    left_json = stage1_dir / f"{stem}_left.json"
    right_json = stage1_dir / f"{stem}_right.json"
    
    # Check required files exist
    if not third_mp4.exists():
        print(f"[SKIP] {stem}: Missing {third_mp4.name}")
        return False
    if not third_json.exists():
        print(f"[SKIP] {stem}: Missing {third_json.name}")
        return False
    if not left_json.exists():
        print(f"[SKIP] {stem}: Missing {left_json.name}")
        return False
    if not right_json.exists():
        print(f"[SKIP] {stem}: Missing {right_json.name}")
        return False
    
    # Load first frame from video
    cap = cv2.VideoCapture(str(third_mp4))
    ret, first_frame = cap.read()
    cap.release()
    
    if not ret or first_frame is None:
        print(f"[ERROR] {stem}: Failed to read first frame from {third_mp4.name}")
        return False
    
    # Convert BGR to RGB
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    
    try:
        # Detect left gripper marker
        print(f"[INFO] {stem}: Detecting left gripper (ID={ARUCO_CONFIG['left_id']})...")
        left_pose7, left_success = detect_aruco_pose(
            first_frame_rgb,
            ARUCO_CONFIG["left_id"],
            K,
            dist_coeffs
        )
        
        if not left_success:
            print(f"[WARN] {stem}: Left gripper marker not detected!")
            return False
        
        # Detect right gripper marker
        print(f"[INFO] {stem}: Detecting right gripper (ID={ARUCO_CONFIG['right_id']})...")
        right_pose7, right_success = detect_aruco_pose(
            first_frame_rgb,
            ARUCO_CONFIG["right_id"],
            K,
            dist_coeffs
        )
        
        if not right_success:
            print(f"[WARN] {stem}: Right gripper marker not detected!")
            return False
        
        print(f"[INFO] {stem}: Detected - Left: {left_pose7[:3]}, Right: {right_pose7[:3]}")
        
        # Load first frame SLAM poses from left/right JSONs
        with open(left_json, 'r') as f:
            left_data = json.load(f)
        with open(right_json, 'r') as f:
            right_data = json.load(f)
        
        T_left_slam_0 = np.array(left_data["records"][0]["pose"], dtype=np.float32)
        T_right_slam_0 = np.array(right_data["records"][0]["pose"], dtype=np.float32)
        
        # Ensure pose7 format
        if T_left_slam_0.shape == (4, 4):
            T_left_slam_0 = mat_to_pose7(T_left_slam_0)
        if T_right_slam_0.shape == (4, 4):
            T_right_slam_0 = mat_to_pose7(T_right_slam_0)
        
        # Convert to 4x4 matrices
        T_left_slam_0_mat = pose7_to_mat(T_left_slam_0)
        T_right_slam_0_mat = pose7_to_mat(T_right_slam_0)
        T_left_cam_0 = pose7_to_mat(left_pose7)
        T_right_cam_0 = pose7_to_mat(right_pose7)
        
        # Compute SLAM to camera transforms
        T_slam_to_cam_left = T_left_cam_0 @ np.linalg.inv(T_left_slam_0_mat)
        T_slam_to_cam_right = T_right_cam_0 @ np.linalg.inv(T_right_slam_0_mat)
        
        # Prepare foundation_pose data (same format as before!)
        foundation_pose_data = {
            "left_in_cam": left_pose7.tolist(),
            "right_in_cam": right_pose7.tolist(),
            "T_slam0_to_cam_left": mat_to_pose7(T_slam_to_cam_left).tolist(),
            "T_slam0_to_cam_right": mat_to_pose7(T_slam_to_cam_right).tolist(),
            "metadata": {
                "detection_source": "aruco",
                "marker_size_m": ARUCO_CONFIG["marker_size_m"],
                "left_marker_id": ARUCO_CONFIG["left_id"],
                "right_marker_id": ARUCO_CONFIG["right_id"],
                "camera_intrinsics": CAMERA_INTRINSICS,
                "frame_index": 0,
                "video_file": str(third_mp4.name),
            }
        }
        
        # Update third.json
        with open(third_json, 'r') as f:
            third_data = json.load(f)
        
        third_data["foundation_pose"] = foundation_pose_data
        
        with open(third_json, 'w') as f:
            json.dump(third_data, f, indent=2)
        
        print(f"[OK] {stem}: Added foundation_pose (via ArUco) to {third_json.name}")
        
        # Debug: Save visualization
        if debug:
            vis_dir = stage1_dir / "aruco_vis"
            vis_dir.mkdir(exist_ok=True)
            
            # Draw detected markers on frame
            vis_frame = first_frame.copy()
            
            # Re-detect to get corners for visualization
            gray = cv2.cvtColor(first_frame_rgb, cv2.COLOR_RGB2GRAY)
            aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_CONFIG["dictionary"])
            
            try:
                parameters = cv2.aruco.DetectorParameters()
                detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
                corners, ids, _ = detector.detectMarkers(gray)
            except AttributeError:
                parameters = cv2.aruco.DetectorParameters_create()
                corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            
            if ids is not None:
                try:
                    cv2.aruco.drawDetectedMarkers(vis_frame, corners, ids)
                except AttributeError:
                    vis_frame = cv2.aruco.drawDetectedMarkers(vis_frame, corners, ids)
            
            vis_path = vis_dir / f"{stem}_aruco_detection.png"
            cv2.imwrite(str(vis_path), vis_frame)
            print(f"[DEBUG] Saved visualization: {vis_path}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] {stem}: Detection failed - {e}")
        import traceback
        traceback.print_exc()
        return False


def main(stage1_dir: str, debug: bool = False):
    """
    Main function to process all episodes in a directory.
    
    Args:
        stage1_dir: Directory containing episode files
        debug: Enable debug visualization
    """
    stage1_dir = Path(stage1_dir)
    if not stage1_dir.exists():
        raise FileNotFoundError(f"Directory not found: {stage1_dir}")
    
    # Find all left JSON files to identify episodes
    left_jsons = sorted(stage1_dir.glob("episode_*_left.json"))
    
    if not left_jsons:
        print(f"[WARN] No episode files found in {stage1_dir}")
        return
    
    print(f"[INFO] Found {len(left_jsons)} episodes to process")
    print(f"[INFO] Using ArUco marker size: {ARUCO_CONFIG['marker_size_m']*100:.1f}cm")
    print(f"[INFO] Left marker ID: {ARUCO_CONFIG['left_id']}, Right marker ID: {ARUCO_CONFIG['right_id']}")
    
    # Get camera intrinsics
    K = get_camera_matrix()
    dist_coeffs = np.array(CAMERA_INTRINSICS["distortion"], dtype=np.float32)
    
    print(f"[INFO] Camera intrinsics:")
    print(f"  fx={CAMERA_INTRINSICS['fx']:.2f}, fy={CAMERA_INTRINSICS['fy']:.2f}")
    print(f"  cx={CAMERA_INTRINSICS['cx']:.2f}, cy={CAMERA_INTRINSICS['cy']:.2f}")
    
    # Process each episode
    success_count = 0
    for left_json in left_jsons:
        stem = left_json.stem.replace("_left", "")
        success = process_episode(stem, stage1_dir, K, dist_coeffs, debug)
        if success:
            success_count += 1
    
    print(f"\n[SUMMARY] Processed {len(left_jsons)} episodes, {success_count} successful")
    
    if success_count < len(left_jsons):
        print("[WARN] Some episodes failed. Check:")
        print("  1. Camera intrinsics are correct (see CAMERA_INTRINSICS at top of script)")
        print("  2. ArUco markers are clearly visible in first frame")
        print("  3. Marker IDs and sizes match configuration")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Add ArUco marker detection to episode JSONs")
    p.add_argument("--stage1_dir", required=True, help="Directory containing episode files")
    p.add_argument("--debug", action="store_true", help="Save debug visualizations with detected markers")
    args = p.parse_args()
    
    main(
        stage1_dir=args.stage1_dir,
        debug=args.debug,
    )


# =============================================================================
# CAMERA CALIBRATION HELPER (optional)
# =============================================================================
"""
To calibrate your D435 camera, you can:

Option 1: Use RealSense SDK
    import pyrealsense2 as rs
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    # Then update CAMERA_INTRINSICS above

Option 2: OpenCV checkerboard calibration
    See: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

Option 3: Use factory defaults (approximate)
    D435 1920x1080 defaults:
    fx ≈ 1380, fy ≈ 1380
    cx ≈ 960, cy ≈ 540
    distortion ≈ [0, 0, 0, 0, 0] (RealSense does internal rectification)
"""