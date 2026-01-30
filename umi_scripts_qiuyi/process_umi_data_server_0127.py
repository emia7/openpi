"""
[Server Machine] Reconstruct LeRobot dataset from UMI-style lightweight export (MP4 + JSON).
Adapts to Franka Teleop format for unified training.

Usage:
    python process_server_umi.py --input_dir=./umi_data_upload --repo_id=local/umi_demo
"""
import os
import json
import shutil
import numpy as np
import cv2
from absl import app, flags
from tqdm import tqdm
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import imageio.v3 as iio

FLAGS = flags.FLAGS
flags.DEFINE_string("input_dir", "./umi_data_upload", "Directory containing uploaded episode mp4/json.")
flags.DEFINE_string("output_dir", None, "Output directory for LeRobot dataset.")
flags.DEFINE_string("repo_id", "local/umi_demo", "LeRobot repo ID.")
flags.DEFINE_float("target_fps", 10.0, "Target FPS for downsampling (default 10Hz).")
flags.DEFINE_string("task_text", "do the task", "Task description for all episodes.")

def pose7_to_pos_rotvec(pose7: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """pose7: [x,y,z,qx,qy,qz,qw] -> (pos(3,), rotvec(3,))"""
    pos = pose7[:3].astype(np.float32)
    quat = pose7[3:7].astype(np.float32)  # (qx,qy,qz,qw)
    # Convert quaternion to rotation vector
    rotvec = R.from_quat(quat).as_rotvec().astype(np.float32)
    return pos, rotvec

def compute_clamp_delta(current_clamp, next_clamp, threshold=0.05):
    """
    Calculate clamp change direction: -1 (closing), 1 (opening), 0 (static).
    Args:
        current_clamp: float (width or normalized value)
        next_clamp: float
        threshold: float, change threshold
    """
    diff = next_clamp - current_clamp
    if diff > threshold:
        return 1.0  # Opening
    elif diff < -threshold:
        return -1.0 # Closing
    else:
        return 0.0  # Static

def load_episode_json(json_path: Path):
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    records = meta["records"]
    if len(records) < 2:
        raise ValueError(f"{json_path} has <2 records, cannot build episode.")
    
    poses = np.asarray([r["pose"] for r in records], dtype=np.float32)  # (T,7)

    if "clamp" in records[0]:
        clamp = np.asarray([r["clamp"] for r in records], dtype=np.float32) # (T,)
    else:
        clamp = np.zeros(len(records), dtype=np.float32)

    fps = float(meta.get("fps", 60.0))
    return poses, clamp, fps, len(records)

def main(_):
    input_dir = Path(FLAGS.input_dir)
    repo_id = FLAGS.repo_id
    output_dir = FLAGS.output_dir or (input_dir / "lerobot_dataset")
    
    # 1. Clean up existing dataset
    output_path = Path(output_dir) / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)
    
    # 2. Find MP4 files
    mp4_files = sorted(input_dir.glob("episode*.mp4"))
    if not mp4_files:
        raise FileNotFoundError(f"No episode*.mp4 found in {input_dir}")

    # 3. Determine Dataset Schema from first episode
    # We need H, W to init dataset
    first_json = input_dir / (mp4_files[0].stem + ".json")
    if not first_json.exists():
        raise FileNotFoundError(f"Missing json for first mp4: {first_json}")
    
    poses0, clamp0, fps0, _ = load_episode_json(first_json)
    
    # Downsample calculation
    stride = int(round(fps0 / FLAGS.target_fps))
    stride = max(1, stride)
    achieved_fps = fps0 / stride
    print(f"[INFO] Downsampling: {fps0}Hz -> {achieved_fps:.2f}Hz (stride={stride})")

    # Read first frame for dims
    # Using imageio to peek
    first_frame = iio.imread(mp4_files[0], index=0)
    H, W, C = first_frame.shape

    # Define Features (Matches Franka Teleop format)
    features = {
        "observation.state.tcp_pose": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "qx", "qy", "qz", "qw"]
        },
        "observation.state.gripper_pose": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["gripper"]
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "rx", "ry", "rz", "gripper_delta"]
        },
        "task_idx": {"dtype": "int64", "shape": (1,), "names": None},
        "subtask_idx": {"dtype": "int64", "shape": (1,), "names": None},
        "is_failed": {"dtype": "bool", "shape": (1,), "names": None},
        "reward": {"dtype": "float32", "shape": (1,), "names": None},
        "return": {"dtype": "float32", "shape": (1,), "names": None},
        "done": {"dtype": "bool", "shape": (1,), "names": None},
    }
    
    # Hardcode camera name to match your Franka policy config
    cam_key = "fish_eye_front"
    features[f"observation.images.{cam_key}"] = {
        "dtype": "image", 
        "shape": (H, W, C), 
        "names": ["height", "width", "channel"]
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(round(achieved_fps)),
        robot_type="franka_research_3",
        features=features,
        root=str(output_dir)
    )

    # 4. Processing Loop
    total_frames = 0
    
    for mp4_path in tqdm(mp4_files, desc="Processing episodes"):
        json_path = input_dir / (mp4_path.stem + ".json")
        if not json_path.exists():
            continue

        poses, clamp, _, T = load_episode_json(json_path)
        
        # Calculate indices for downsampling
        indices = np.arange(0, T, stride)
        
        # Read video frames
        # imageio is generally more robust for frame-accurate seeking than cv2
        reader = iio.imiter(mp4_path)
        
        frame_iter = enumerate(reader)
        current_idx = 0
        
        for t_idx in indices:
            # 1. Seek/Read correct frame
            # Skip frames until we reach t_idx
            try:
                while current_idx < t_idx:
                    next(frame_iter)
                    current_idx += 1
                _, frame_img = next(frame_iter)
                current_idx += 1
            except StopIteration:
                break

            # 2. Extract State (Observation)
            # TCP Pose (7D: pos + quat)
            # Use original quaternion format to match franka_policy expectation
            # NOTE: Your franka_policy expects [x,y,z, qx,qy,qz,qw] in state
            tcp_pose = poses[t_idx] 
            
            # Gripper Pose (1D: Absolute Width)
            gripper_pose = np.array([clamp[t_idx]], dtype=np.float32)

            # 3. Construct Action (Target for next step)
            # We use next downsampled step as target
            next_t_idx = t_idx + stride
            if next_t_idx >= T:
                next_t_idx = t_idx # Last frame repeats
            
            # Action Pose: Absolute Position + Rotation Vector (6D)
            # NOTE: Your franka_policy action processing expects [x,y,z, rx,ry,rz, g]
            # where rx,ry,rz is rotation vector.
            pos_next, rotvec_next = pose7_to_pos_rotvec(poses[next_t_idx])
            
            # Action Gripper: Delta {-1, 0, 1}
            # Compare current clamp vs next clamp
            clamp_curr = clamp[t_idx]
            clamp_next = clamp[next_t_idx]
            gripper_delta = compute_clamp_delta(clamp_curr, clamp_next)
            
            action_vec = np.concatenate([pos_next, rotvec_next, [gripper_delta]]).astype(np.float32)

            # 4. Build Frame
            frame_dict = {
                "observation.state.tcp_pose": tcp_pose,
                "observation.state.gripper_pose": gripper_pose,
                "action": action_vec,
                "task_idx": np.array([0], dtype=np.int64), # Dummy
                "subtask_idx": np.array([0], dtype=np.int64), # Dummy
                "is_failed": np.array([False], dtype=bool), # Assume UMI demos are successful
                "reward": np.array([0.0], dtype=np.float32),
                "return": np.array([0.0], dtype=np.float32),
                "done": np.array([t_idx == indices[-1]], dtype=bool),
                f"observation.images.{cam_key}": frame_img,
                "task": FLAGS.task_text,
            }
            
            # Add task description using the new method (save_episode arg)
            # We store it temporarily and pass it later
            
            dataset.add_frame(frame_dict)
            total_frames += 1

        dataset.save_episode()

    print(f"\nDataset created at: {output_path}")
    print(f"Total frames: {total_frames}")

if __name__ == "__main__":
    app.run(main)