#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add Foundation Pose detection results to episode JSON files.

This script processes the first frame of third-view videos to detect left and right
hand poses using Foundation Pose, then stores the results in the episode JSON.

Usage:
    python add_foundation_pose_to_json.py --stage1_dir /path/to/stage1_data

Requirements:
    - Foundation Pose installed and configured
    - Third-view videos: episode_XXX_third.mp4
    - Left/right JSONs: episode_XXX_left.json, episode_XXX_right.json
    - Output: Updates episode_XXX_third.json with foundation_pose field
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


def load_foundation_pose_model(obj_name: str, model_path: str = None):
    """
    Load Foundation Pose model for a specific object.
    
    Args:
        obj_name: "left_gripper" or "right_gripper"
        model_path: Path to the object model/mesh (optional)
    
    Returns:
        Foundation Pose model instance
    """
    try:
        # Import Foundation Pose (adjust based on your installation)
        from foundation_pose import FoundationPose
        
        # Initialize model
        # NOTE: Adjust parameters based on your Foundation Pose setup
        model = FoundationPose(
            model_path=model_path,
            obj_name=obj_name,
            # Add other necessary parameters
        )
        return model
    except ImportError:
        print("[ERROR] Foundation Pose not installed. Please install it first.")
        print("  pip install foundation-pose  # or your installation method")
        raise


def detect_gripper_pose(model, frame: np.ndarray, obj_name: str) -> np.ndarray:
    """
    Detect gripper pose in a frame using Foundation Pose.
    
    Args:
        model: Foundation Pose model instance
        frame: RGB image (H, W, 3)
        obj_name: "left_gripper" or "right_gripper"
    
    Returns:
        pose7: [x, y, z, qx, qy, qz, qw] in camera coordinate system
    """
    # Run Foundation Pose detection
    # NOTE: Adjust based on your Foundation Pose API
    pose_mat = model.predict(frame)  # Returns 4x4 transformation matrix
    
    # Convert to pose7 format
    pose7 = mat_to_pose7(pose_mat)
    return pose7


def process_episode(
    stem: str,
    stage1_dir: Path,
    left_model,
    right_model,
    debug: bool = False
):
    """
    Process a single episode: detect poses in first frame and update JSON.
    
    Args:
        stem: Episode stem (e.g., "episode_000001")
        stage1_dir: Directory containing episode files
        left_model: Foundation Pose model for left gripper
        right_model: Foundation Pose model for right gripper
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
    
    # Convert BGR to RGB (OpenCV default is BGR)
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    
    try:
        # Detect left gripper pose
        print(f"[INFO] {stem}: Detecting left gripper...")
        left_pose7 = detect_gripper_pose(left_model, first_frame_rgb, "left_gripper")
        
        # Detect right gripper pose
        print(f"[INFO] {stem}: Detecting right gripper...")
        right_pose7 = detect_gripper_pose(right_model, first_frame_rgb, "right_gripper")
        
        # Load first frame SLAM poses from left/right JSONs
        with open(left_json, 'r') as f:
            left_data = json.load(f)
        with open(right_json, 'r') as f:
            right_data = json.load(f)
        
        T_left_slam_0 = np.array(left_data["records"][0]["pose"], dtype=np.float32)
        T_right_slam_0 = np.array(right_data["records"][0]["pose"], dtype=np.float32)
        
        # Ensure pose7 format (if SLAM outputs 4x4, convert; if 7D, use directly)
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
        
        # Prepare foundation_pose data
        foundation_pose_data = {
            "left_in_cam": left_pose7.tolist(),
            "right_in_cam": right_pose7.tolist(),
            "T_slam0_to_cam_left": mat_to_pose7(T_slam_to_cam_left).tolist(),
            "T_slam0_to_cam_right": mat_to_pose7(T_slam_to_cam_right).tolist(),
            "metadata": {
                "detection_source": "foundation_pose",
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
        
        print(f"[OK] {stem}: Added foundation_pose to {third_json.name}")
        
        # Debug: Save visualization
        if debug:
            vis_dir = stage1_dir / "foundation_pose_vis"
            vis_dir.mkdir(exist_ok=True)
            
            # Draw detected poses on frame (if Foundation Pose provides visualization)
            # Or just save the first frame for verification
            vis_path = vis_dir / f"{stem}_first_frame.png"
            cv2.imwrite(str(vis_path), first_frame)
            print(f"[DEBUG] Saved visualization: {vis_path}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] {stem}: Detection failed - {e}")
        import traceback
        traceback.print_exc()
        return False


def main(stage1_dir: str, left_model_path: str = None, right_model_path: str = None, debug: bool = False):
    """
    Main function to process all episodes in a directory.
    
    Args:
        stage1_dir: Directory containing episode files
        left_model_path: Path to left gripper model (optional)
        right_model_path: Path to right gripper model (optional)
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
    
    # Load Foundation Pose models
    print("[INFO] Loading Foundation Pose models...")
    left_model = load_foundation_pose_model("left_gripper", left_model_path)
    right_model = load_foundation_pose_model("right_gripper", right_model_path)
    
    # Process each episode
    success_count = 0
    for left_json in left_jsons:
        stem = left_json.stem.replace("_left", "")
        success = process_episode(stem, stage1_dir, left_model, right_model, debug)
        if success:
            success_count += 1
    
    print(f"\n[SUMMARY] Processed {len(left_jsons)} episodes, {success_count} successful")
    
    if success_count < len(left_jsons):
        print("[WARN] Some episodes failed. Check errors above.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Add Foundation Pose detection to episode JSONs")
    p.add_argument("--stage1_dir", required=True, help="Directory containing episode files")
    p.add_argument("--left_model", default=None, help="Path to left gripper Foundation Pose model")
    p.add_argument("--right_model", default=None, help="Path to right gripper Foundation Pose model")
    p.add_argument("--debug", action="store_true", help="Save debug visualizations")
    args = p.parse_args()
    
    main(
        stage1_dir=args.stage1_dir,
        left_model_path=args.left_model,
        right_model_path=args.right_model,
        debug=args.debug,
    )