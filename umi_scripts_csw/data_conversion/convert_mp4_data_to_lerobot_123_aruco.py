#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert ArUco-detected MP4+JSON data to LeRobot format.

Based on convert_mp4_data_to_lerobot_123.py with modifications:
1. Reads ArUco detection from third.json (root level, not foundation_pose field)
2. Uses detection_frame_left/right for multi-frame detection support
3. Computes inter-gripper state in unified world coordinate system
4. Simplified features: only 9D hands_rel state (removes raw SLAM proprioception)

Usage:
    python convert_mp4_data_to_lerobot_123_aruco.py \
        --stage1_dir ~/Downloads/handover_umi_0422_aruco \
        --repo handover_umi_0422_aruco_lerobot \
        --robot_type XV_DUAL
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import imageio.v3 as iio
from scipy.spatial.transform import Rotation as R

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

# Import pose utilities for inter-gripper coordinate transformations
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from openpi.policies.pose_util import (
    mat_to_rot6d,
    pose7_to_mat,
    mat_to_pose7,
)


def pose7_to_pos_rotvec(pose7: np.ndarray):
    """pose7: [x,y,z,qx,qy,qz,qw] -> pos(3,), rotvec(3,)"""
    pos = pose7[:3].astype(np.float32)
    quat = pose7[3:7].astype(np.float32)
    rotvec = R.from_quat(quat).as_rotvec().astype(np.float32)
    return pos, rotvec


def load_episode_json(path: Path):
    """Load episode JSON with pose and clamp data."""
    meta = json.loads(path.read_text(encoding="utf-8"))
    records = meta["records"]
    if len(records) < 2:
        raise ValueError(f"{path} has <2 records.")
    poses = np.asarray([r["pose"] for r in records], dtype=np.float32)  # (T,7)
    clamps = np.asarray([r["clamp"] for r in records], dtype=np.float32).reshape(-1, 1)  # (T,1)
    fps = float(meta.get("fps", 10.0))
    return poses, clamps, fps, len(records)


def main(stage1_dir: str, repo: str, robot_type: str, task: str, fps_override: int = 0):
    stage1_dir = Path(stage1_dir)
    if not stage1_dir.exists():
        raise FileNotFoundError(stage1_dir)

    left_jsons = sorted(stage1_dir.glob("episode_*_left.json"))
    if not left_jsons:
        raise FileNotFoundError(f"No episode_*_left.json found in {stage1_dir}")

    # Inspect first episode to get video dimensions
    first_left_json = left_jsons[0]
    stem = first_left_json.stem.replace("_left", "")  # episode_000001

    p_left = stage1_dir / f"{stem}_left.mp4"
    p_right = stage1_dir / f"{stem}_right.mp4"
    p_third = stage1_dir / f"{stem}_third.mp4"
    p_right_json = stage1_dir / f"{stem}_right.json"

    if not (p_left.exists() and p_right.exists() and p_third.exists() and p_right_json.exists()):
        raise FileNotFoundError(f"Missing files for first episode: {stem}")

    poses_l0, clamp_l0, fps0, _ = load_episode_json(first_left_json)

    # Dataset fps
    fps_ds = int(round(fps0))
    if fps_override and fps_override > 0:
        fps_ds = int(fps_override)

    # Get image shapes from first frames
    frame_left0 = np.asarray(iio.imread(p_left, index=0), dtype=np.uint8)
    frame_right0 = np.asarray(iio.imread(p_right, index=0), dtype=np.uint8)
    frame_third0 = np.asarray(iio.imread(p_third, index=0), dtype=np.uint8)
    Hl, Wl, _ = frame_left0.shape
    Hr, Wr, _ = frame_right0.shape
    Ht, Wt, _ = frame_third0.shape

    out_path = HF_LEROBOT_HOME / repo
    if out_path.exists():
        print(f"[INFO] Removing existing repo at {out_path}")
        shutil.rmtree(out_path)

    # Create LeRobot dataset with simplified features (only 9D hands_rel state)
    dataset = LeRobotDataset.create(
        repo_id=repo,
        robot_type=robot_type,
        fps=fps_ds,
        features={
            "left_view":  {"dtype": "video", "shape": (Hl, Wl, 3), "names": ["h", "w", "c"]},
            "right_view": {"dtype": "video", "shape": (Hr, Wr, 3), "names": ["h", "w", "c"]},
            "third_view": {"dtype": "video", "shape": (Ht, Wt, 3), "names": ["h", "w", "c"]},

            # Simplified state: only 9D inter-gripper relative pose
            # No raw SLAM proprioception (left_eef_pos, right_eef_pos, etc.)
            "hands_rel_xyz": {"dtype": "float32", "shape": (3,), "names": ["dx", "dy", "dz"]},
            "hands_rel_rot6d": {"dtype": "float32", "shape": (6,), "names": ["r0", "r1", "r2", "r3", "r4", "r5"]},

            # Actions: next-step absolute pose+grip for each hand
            "left_action": {"dtype": "float32", "shape": (7,), "names": ["x", "y", "z", "rx", "ry", "rz", "g"]},
            "right_action": {"dtype": "float32", "shape": (7,), "names": ["x", "y", "z", "rx", "ry", "rz", "g"]},
            "actions": {"dtype": "float32", "shape": (14,), "names": [
                "left_x", "left_y", "left_z", "left_rx", "left_ry", "left_rz", "left_g",
                "right_x", "right_y", "right_z", "right_rx", "right_ry", "right_rz", "right_g",
            ]},

        },
        image_writer_threads=8,
        image_writer_processes=4,
    )

    total_eps, total_frames = 0, 0

    for left_json in left_jsons:
        stem = left_json.stem.replace("_left", "")  # episode_000001
        right_json = stage1_dir / f"{stem}_right.json"
        left_mp4 = stage1_dir / f"{stem}_left.mp4"
        right_mp4 = stage1_dir / f"{stem}_right.mp4"
        third_mp4 = stage1_dir / f"{stem}_third.mp4"

        if not (right_json.exists() and left_mp4.exists() and right_mp4.exists() and third_mp4.exists()):
            print(f"[SKIP] Missing files for {stem}")
            continue

        # Load states (pose7 is [x,y,z,qx,qy,qz,qw] in stage1 json)
        poses_l7, clamp_l, _, T_l = load_episode_json(left_json)
        poses_r7, clamp_r, _, T_r = load_episode_json(right_json)

        # Load ArUco data from third.json
        third_json = stage1_dir / f"{stem}_third.json"
        fp_data = None
        detection_frame_left = 0
        detection_frame_right = 0
        
        if third_json.exists():
            with open(third_json, 'r') as f:
                third_meta = json.load(f)
                # ArUco data is at root level (not under foundation_pose field)
                if "left_in_cam" in third_meta:
                    fp_data = third_meta
                    metadata = fp_data.get("metadata", {})
                    detection_frame_left = metadata.get("detection_frame_left", 0)
                    detection_frame_right = metadata.get("detection_frame_right", 0)
        
        # Pre-compute coordinate transforms (if ArUco data available)
        T_world_cam = None
        T_slam_to_cam_left = None
        T_slam_to_cam_right = None
        
        if fp_data is not None:
            # ArUco poses in camera frame at detection frames
            # JSON format: {"translation": [x,y,z], "quaternion": [qx,qy,qz,qw]}
            left_in_cam = fp_data["left_in_cam"]
            right_in_cam = fp_data["right_in_cam"]
            
            # Build pose7: [x,y,z, qx,qy,qz,qw]
            left_pose7 = np.array([
                *left_in_cam["translation"],
                *left_in_cam["quaternion"]
            ], dtype=np.float32)
            right_pose7 = np.array([
                *right_in_cam["translation"],
                *right_in_cam["quaternion"]
            ], dtype=np.float32)
            
            T_left_cam_detect = pose7_to_mat(left_pose7)
            T_right_cam_detect = pose7_to_mat(right_pose7)
            
            # World coordinate system: detected left hand position as origin
            T_world_cam = np.linalg.inv(T_left_cam_detect)
            
            # Get SLAM poses at the actual detection frames (not frame 0)
            # This is the key difference for multi-frame detection support
            T_left_slam_detect = pose7_to_mat(poses_l7[detection_frame_left])
            T_right_slam_detect = pose7_to_mat(poses_r7[detection_frame_right])
            
            # Pre-compute SLAM to camera transforms
            # T_slam_to_cam = T_cam_detect @ inv(T_slam_detect)
            T_slam_to_cam_left = T_left_cam_detect @ np.linalg.inv(T_left_slam_detect)
            T_slam_to_cam_right = T_right_cam_detect @ np.linalg.inv(T_right_slam_detect)
            
            print(f"[{stem}] ArUco: left@frame{detection_frame_left}, right@frame{detection_frame_right}")

        # Iterate through videos
        it_l = iio.imiter(left_mp4)
        it_r = iio.imiter(right_mp4)
        it_t = iio.imiter(third_mp4)

        # Truncate length by json
        T_json = min(T_l, T_r, len(clamp_l), len(clamp_r))
        written = 0

        try:
            for i, (fl, fr, ft) in enumerate(zip(it_l, it_r, it_t)):
                if i >= T_json:
                    break

                fl = np.asarray(fl, dtype=np.uint8)
                fr = np.asarray(fr, dtype=np.uint8)
                ft = np.asarray(ft, dtype=np.uint8)

                # Compute actions (next step absolute)
                j = i + 1 if (i + 1) < T_json else i
                pos_l2, rot_l2 = pose7_to_pos_rotvec(poses_l7[j])
                pos_r2, rot_r2 = pose7_to_pos_rotvec(poses_r7[j])
                g_l2 = np.array([float(clamp_l[j].item())], dtype=np.float32)
                g_r2 = np.array([float(clamp_r[j].item())], dtype=np.float32)

                left_action = np.concatenate([pos_l2, rot_l2, g_l2], axis=0).astype(np.float32)   # (7,)
                right_action = np.concatenate([pos_r2, rot_r2, g_r2], axis=0).astype(np.float32)  # (7,)
                actions = np.concatenate([left_action, right_action], axis=0).astype(np.float32)  # (14,)

                # Base frame data
                frame_data = {
                    "left_view": fl,
                    "right_view": fr,
                    "third_view": ft,

                    "left_action": left_action,
                    "right_action": right_action,
                    "actions": actions,

                    "task": task,
                }

                # Compute inter-gripper state (if ArUco data available)
                if fp_data is not None and T_world_cam is not None:
                    # Get current SLAM poses at frame i
                    T_left_slam_i = pose7_to_mat(poses_l7[i])
                    T_right_slam_i = pose7_to_mat(poses_r7[i])
                    
                    # Transform to camera coordinate system
                    # T_cam_i = T_slam_to_cam @ T_slam_i
                    T_left_cam_i = T_slam_to_cam_left @ T_left_slam_i
                    T_right_cam_i = T_slam_to_cam_right @ T_right_slam_i
                    
                    # Transform to world coordinate system
                    # T_world_i = T_world_cam @ T_cam_i
                    T_left_world_i = T_world_cam @ T_left_cam_i
                    T_right_world_i = T_world_cam @ T_right_cam_i
                    
                    # Compute inter-gripper relative pose (right hand in left hand frame)
                    # T_right_in_left = inv(T_left_world) @ T_right_world
                    T_right_in_left_i = np.linalg.inv(T_left_world_i) @ T_right_world_i
                    
                    # Extract 9D features
                    hands_rel_xyz = T_right_in_left_i[:3, 3].astype(np.float32)           # (3,)
                    hands_rel_rot6d = mat_to_rot6d(T_right_in_left_i[:3, :3]).astype(np.float32)  # (6,)
                    
                    frame_data["hands_rel_xyz"] = hands_rel_xyz
                    frame_data["hands_rel_rot6d"] = hands_rel_rot6d
                else:
                    # Fill with zeros if ArUco data not available
                    frame_data["hands_rel_xyz"] = np.zeros(3, dtype=np.float32)
                    frame_data["hands_rel_rot6d"] = np.zeros(6, dtype=np.float32)

                dataset.add_frame(frame_data)
                written += 1

        except Exception as e:
            print(f"[WARN] {stem}: error at frame {written}: {e}. Truncating this episode.")

        dataset.save_episode()
        total_eps += 1
        total_frames += written
        print(f"[OK] {stem} -> frames={written}")

    print(f"\nDONE -> repo saved at: {out_path}")
    print(f"Total Episodes: {total_eps}, Total Frames: {total_frames}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_dir", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--robot_type", default="XV_DUAL")
    p.add_argument("--task", default="handover with aruco")
    p.add_argument("--fps", type=int, default=0, help="override dataset fps (0=use json fps rounded)")
    args = p.parse_args()

    main(
        stage1_dir=args.stage1_dir,
        repo=args.repo,
        robot_type=args.robot_type,
        task=args.task,
        fps_override=args.fps,
    )