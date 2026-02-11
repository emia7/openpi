"""
[Server Machine] Reconstruct LeRobot dataset from lightweight export (Parquet + MP4).
Supports multiple input directories and episode sampling.
"""
import os
import json
import shutil
import numpy as np
import pandas as pd
import cv2
from absl import app, flags
from tqdm import tqdm
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from pathlib import Path
from unittest.mock import patch
import random

FLAGS = flags.FLAGS
# Update flags to support list of input dirs and sampling
flags.DEFINE_multi_string("input_dirs", ["./ready_to_upload"], "List of directories containing uploaded processed data (task folders inside).")
flags.DEFINE_string("output_dir", None, "Output directory for LeRobot dataset.")
flags.DEFINE_string("repo_id", "local/franka_demo", "LeRobot repo ID.")
flags.DEFINE_bool("skip_failed", True, "Skip failed episodes.")
# New sampling flags
flags.DEFINE_integer("num_episodes", None, "Number of episodes to sample. If None, use all.", lower_bound=1)
flags.DEFINE_integer("seed", 42, "Random seed for sampling.")

def scale_action_for_lerobot(action, action_scale):
    """Scale action for LeRobot format (gripper mapped to [0,1])."""
    scaled = np.zeros(7, dtype=np.float32)
    scaled[:3] = action[:3] * action_scale[0]
    scaled[3:6] = action[3:6] * action_scale[1]
    scaled[6] = (action[6] + 1) / 2
    return scaled

def gather_all_episodes(input_dirs, skip_failed):
    """Scan all directories and collect valid episode info."""
    all_episodes = []
    
    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            print(f"[WARN] Input directory not found: {input_dir}")
            continue

        files = sorted([f for f in os.listdir(input_dir) if f.endswith(".parquet")])
        
        for f in files:
            base_name = os.path.splitext(f)[0]
            meta_path = os.path.join(input_dir, f"{base_name}_meta.json")
            
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as mf:
                    meta = json.load(mf)
                
                if skip_failed and meta.get("episode_failed", False):
                    continue
                    
                all_episodes.append({
                    "parquet_path": os.path.join(input_dir, f),
                    "meta": meta,
                    "base_name": base_name,
                    "dir": input_dir
                })

    return all_episodes

def main(_):
    if FLAGS.output_dir:
        output_dir = Path(FLAGS.output_dir)
    else:
        output_dir = Path(HF_LEROBOT_HOME) # Ensure it's a Path object

    repo_id = FLAGS.repo_id
    
    # 1. Gather Episodes
    print(f"Scanning input directories: {FLAGS.input_dirs}")
    all_episodes = gather_all_episodes(FLAGS.input_dirs, FLAGS.skip_failed)
    total_found = len(all_episodes)
    print(f"Total valid episodes found: {total_found}")

    if not all_episodes:
        print("No episodes found.")
        return

    # 2. Sampling Logic
    rng = np.random.default_rng(FLAGS.seed)
    
    indices = np.arange(total_found)
    rng.shuffle(indices)
    shuffled_episodes = [all_episodes[i] for i in indices]

    if FLAGS.num_episodes is not None:
        if FLAGS.num_episodes < total_found:
            selected_episodes = shuffled_episodes[:FLAGS.num_episodes]
            print(f"Sampling: Selected {FLAGS.num_episodes} episodes out of {total_found} (Seed: {FLAGS.seed})")
        else:
            selected_episodes = shuffled_episodes
            print(f"Sampling: Requested {FLAGS.num_episodes} but only {total_found} available. Using all.")
    else:
        selected_episodes = shuffled_episodes
        print(f"Using all {total_found} episodes.")

    # 3. Clean up existing dataset
    output_path = output_dir / repo_id
    if output_path.exists():
        print(f"Removing existing dataset at: {output_path}")
        shutil.rmtree(output_path)

    # 4. Initialize LeRobot Dataset (using info from first selected episode)
    first_ep = selected_episodes[0]
    cameras = first_ep["meta"]["cameras"]
    
    # Read first video to get dims
    cap = cv2.VideoCapture(os.path.join(first_ep["dir"], f"{first_ep['base_name']}_{cameras[0]}.mp4"))
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"Could not read video to determine shape: {first_ep['base_name']}")
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


    # Patch mkdir for existing directory support
    original_mkdir = Path.mkdir
    def patched_mkdir(self, mode=0o777, parents=False, exist_ok=False):
        return original_mkdir(self, mode, parents, exist_ok=True)

    with patch('pathlib.Path.mkdir', side_effect=patched_mkdir, autospec=True):
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=first_ep["meta"]["fps"],
            robot_type="franka_research_3",
            features=features,
        )

    # 5. Processing Loop
    total_frames = 0
    for ep in tqdm(selected_episodes, desc="Reconstructing episodes"):
        df = pd.read_parquet(ep["parquet_path"])
        meta = ep["meta"]
        action_scale = meta["action_scale"]
        
        video_caps = {}
        for cam in cameras:
            vid_path = os.path.join(ep["dir"], f"{ep['base_name']}_{cam}.mp4")
            video_caps[cam] = cv2.VideoCapture(vid_path)
        
        for idx in range(len(df)):
            row = df.iloc[idx]
            
            tcp_pose = np.array(row["observation/state/tcp_pose"], dtype=np.float32)
            gripper_pose = np.array([row["observation/state/gripper_pose"]], dtype=np.float32)
            
            raw_action = np.array(row["action"], dtype=np.float32)
            scaled_action = scale_action_for_lerobot(raw_action, action_scale)
            
            frame = {
                "observation.state.tcp_pose": tcp_pose,
                "observation.state.gripper_pose": gripper_pose,
                "action": scaled_action,
                "task_idx": np.array([int(row.get("task_idx", 0))], dtype=np.int64),
                "subtask_idx": np.array([int(row.get("subtask_idx", 0))], dtype=np.int64),
                "is_failed": np.array([meta["episode_failed"]], dtype=bool),
                "reward": np.array([row.get("computed_reward", 0.0)], dtype=np.float32),
                "return": np.array([0.0], dtype=np.float32), 
                "done": np.array([idx == len(df)-1], dtype=bool),
                "task": meta["task_description"],
            }
            
            for cam in cameras:
                ret, img = video_caps[cam].read()
                if not ret:
                    raise RuntimeError(f"Video {cam} ended prematurely at step {idx}/{len(df)}")
                frame[f"observation.images.{cam}"] = img[..., ::-1]
            
            dataset.add_frame(frame)
        
        for cap in video_caps.values():
            cap.release()
            
        dataset.save_episode()
        total_frames += len(df)

    print(f"\nDataset created at: {output_path}")
    print(f"Total episodes: {len(selected_episodes)}")
    print(f"Total frames: {total_frames}")

if __name__ == "__main__":
    app.run(main)