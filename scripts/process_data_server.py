"""
[Server Machine] Reconstruct LeRobot dataset from lightweight export (Parquet + MP4).

Usage:
    python process_server.py --input_dir=./uploaded_data --repo_id=local/franka_demo
"""
import os
import json
import shutil
import numpy as np
import pandas as pd
import cv2
from absl import app, flags
from tqdm import tqdm
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from pathlib import Path

FLAGS = flags.FLAGS
flags.DEFINE_string("input_dir", "./ready_to_upload", "Directory containing uploaded processed data.")
flags.DEFINE_string("output_dir", None, "Output directory for LeRobot dataset.")
flags.DEFINE_string("repo_id", "local/franka_demo", "LeRobot repo ID.")
flags.DEFINE_bool("skip_failed", True, "Skip failed episodes.")

def scale_action_for_lerobot(action, action_scale):
    """Scale action for LeRobot format (gripper mapped to [0,1])."""
    scaled = np.zeros(7, dtype=np.float32)
    scaled[:3] = action[:3] * action_scale[0]
    scaled[3:6] = action[3:6] * action_scale[1]
    scaled[6] = (action[6] + 1) / 2
    return scaled

def main(_):
    input_dir = FLAGS.input_dir
    repo_id = FLAGS.repo_id
    output_dir = FLAGS.output_dir or os.path.join(input_dir, "lerobot_dataset")
    
    # 1. Clean up existing dataset
    output_path = Path(output_dir) / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)
    
    # 2. Scan for episodes
    tasks = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    all_episodes = []
    
    for task in tasks:
        task_path = os.path.join(input_dir, task)
        # Find all parquet files (excluding meta/plots if any)
        files = [f for f in os.listdir(task_path) if f.endswith(".parquet")]
        for f in files:
            base_name = os.path.splitext(f)[0]
            meta_path = os.path.join(task_path, f"{base_name}_meta.json")
            
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as mf:
                    meta = json.load(mf)
                
                if FLAGS.skip_failed and meta.get("episode_failed", False):
                    continue
                    
                all_episodes.append({
                    "task": task,
                    "parquet_path": os.path.join(task_path, f),
                    "meta": meta,
                    "base_name": base_name,
                    "dir": task_path
                })

    if not all_episodes:
        print("No episodes found.")
        return

    # 3. Initialize LeRobot Dataset (using info from first episode)
    first_ep = all_episodes[0]
    cameras = first_ep["meta"]["cameras"]
    
    # Read first video to get dims
    cap = cv2.VideoCapture(os.path.join(first_ep["dir"], f"{first_ep['base_name']}_{cameras[0]}.mp4"))
    ret, frame = cap.read()
    h, w, c = frame.shape
    cap.release()
    
    features = {
        "observation.state.tcp_pose": {"dtype": "float32", "shape": (7,), "names": ["x", "y", "z", "qx", "qy", "qz", "qw"]},
        "observation.state.gripper_pose": {"dtype": "float32", "shape": (1,), "names": ["gripper"]},
        "action": {"dtype": "float32", "shape": (7,), "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]},
        "task_idx": {"dtype": "int64", "shape": (1,), "names": None},
        "subtask_idx": {"dtype": "int64", "shape": (1,), "names": None},
        "is_failed": {"dtype": "bool", "shape": (1,), "names": None},
        "reward": {"dtype": "float32", "shape": (1,), "names": None},
        "return": {"dtype": "float32", "shape": (1,), "names": None},
        "done": {"dtype": "bool", "shape": (1,), "names": None},
    }
    
    for cam in cameras:
        features[f"observation.images.{cam}"] = {"dtype": "image", "shape": (h, w, c), "names": ["height", "width", "channel"]}

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=first_ep["meta"]["fps"],
        robot_type="franka_research_3",
        features=features,
        root=output_dir
    )

    # 4. Processing Loop
    total_frames = 0
    for ep in tqdm(all_episodes, desc="Reconstructing episodes"):
        # Load State Data
        df = pd.read_parquet(ep["parquet_path"])
        meta = ep["meta"]
        action_scale = meta["action_scale"]
        
        # Load Video Data generators
        video_caps = {}
        for cam in cameras:
            vid_path = os.path.join(ep["dir"], f"{ep['base_name']}_{cam}.mp4")
            video_caps[cam] = cv2.VideoCapture(vid_path)
        
        # Iterate steps
        for idx in range(len(df)):
            row = df.iloc[idx]
            
            # Prepare State & Action
            tcp_pose = np.array(row["observation/state/tcp_pose"], dtype=np.float32)
            gripper_pose = np.array([row["observation/state/gripper_pose"]], dtype=np.float32)
            
            raw_action = np.array(row["action"], dtype=np.float32)
            scaled_action = scale_action_for_lerobot(raw_action, action_scale)
            
            # Prepare Frame Dict
            frame = {
                "observation.state.tcp_pose": tcp_pose,
                "observation.state.gripper_pose": gripper_pose,
                "action": scaled_action,
                "task_idx": np.array([int(row.get("task_idx", 0))], dtype=np.int64),
                "subtask_idx": np.array([int(row.get("subtask_idx", 0))], dtype=np.int64),
                "is_failed": np.array([meta["episode_failed"]], dtype=bool),
                "reward": np.array([row.get("computed_reward", 0.0)], dtype=np.float32),
                # Note: 'return' calc omitted for brevity, can be recomputed or loaded if saved in process_local
                "return": np.array([0.0], dtype=np.float32), 
                "done": np.array([idx == len(df)-1], dtype=bool),
            }
            
            # Extract Images
            for cam in cameras:
                ret, img = video_caps[cam].read()
                if not ret:
                    raise RuntimeError(f"Video {cam} ended prematurely at step {idx}/{len(df)}")
                # OpenCV is BGR, LeRobot/PyTorch expects RGB
                frame[f"observation.images.{cam}"] = img[..., ::-1]
            
            dataset.add_frame(frame, task=meta["task_description"])
        
        # Cleanup videos
        for cap in video_caps.values():
            cap.release()
            
        dataset.save_episode()
        total_frames += len(df)

    print(f"\nDataset created at: {output_path}")
    print(f"Total episodes: {len(all_episodes)}")
    print(f"Total frames: {total_frames}")

if __name__ == "__main__":
    app.run(main)