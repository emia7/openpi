"""
Convert external dual_franka LeRobot v2.1 format (embedded images in parquet)
to intermediate format: parquet + meta.json + separate MP4 videos.

This prepares the data for convert_dual_franka_data_to_lerobot.py.

Input format (from external source):
  - data/chunk-000/episode_XXXXXX.parquet (images embedded, state 68D with rot6d)
  - meta/episodes.jsonl, tasks.jsonl
  
Output format (for dual_franka converter):
  - episode_XXXXXX.parquet (with observation/state/* columns, using quat instead of rot6d)
  - episode_XXXXXX_meta.json
  - episode_XXXXXX_third_d455.mp4
  - episode_XXXXXX_left_wrist_d435.mp4
  - episode_XXXXXX_right_wrist_d435.mp4

Usage:
  uv run python dual_scripts/convert_external_to_intermediate.py \
    --input_dir=/Users/chenshuaiwen/Downloads/dual_franka_cylinder_handover_20260424 \
    --output_dir=/Users/chenshuaiwen/Downloads/dual_franka_cylinder_handover_20260424_intermediate
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None
try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    R = None
from absl import app, flags
from tqdm import tqdm
import cv2
from PIL import Image

FLAGS = flags.FLAGS

flags.DEFINE_string("input_dir", None, "Input directory containing external LeRobot format data", required=True)
flags.DEFINE_string("output_dir", None, "Output directory for intermediate format", required=True)
flags.DEFINE_bool("overwrite", False, "Overwrite existing output directory")

# Camera mapping: external field -> our camera name
# Mapping: external left -> our right, external right -> our left
# Using _lumos suffix to match training config expectations
CAMERA_MAPPING = {
    "image": "right_wrist_lumos",  # external left -> our right
    "extra_view_image-0": "third_d455",
    "extra_view_image-1": "left_wrist_lumos",  # external right -> our left
}


def decode_image(img_data) -> np.ndarray:
    """
    Decode image from parquet data.
    
    Handles various formats:
    - dict with 'bytes' key (PNG/JPEG bytes)
    - numpy array (already decoded)
    - PIL Image
    """
    if isinstance(img_data, dict) and 'bytes' in img_data:
        # PNG/JPEG bytes
        img_bytes = img_data['bytes']
        img = Image.open(io.BytesIO(img_bytes))
        return np.array(img)
    elif isinstance(img_data, np.ndarray):
        # Already a numpy array
        return img_data
    elif hasattr(img_data, 'shape'):
        # Some other array-like object
        return np.array(img_data)
    else:
        # Try converting to numpy
        return np.array(img_data)


def rot6d_to_quat(rot6d: np.ndarray) -> np.ndarray:
    """
    Convert 6D rotation representation to quaternion [qx, qy, qz, qw].
    
    Uses Gram-Schmidt orthonormalization to reconstruct rotation matrix.
    rot6d: [6] array - first 2 columns of rotation matrix (flattened)
    Returns: [4] quaternion [qx, qy, qz, qw]
    """
    a1, a2 = rot6d[:3], rot6d[3:6]
    
    # Gram-Schmidt orthonormalization
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    
    # Build rotation matrix (3x3)
    rot_mat = np.stack([b1, b2, b3], axis=1)
    
    # Convert to quaternion [qx, qy, qz, qw]
    quat = R.from_matrix(rot_mat).as_quat()
    return quat.astype(np.float32)


def extract_state_from_68d(state_68: np.ndarray) -> dict:
    """
    Extract and convert state from 68D external format.
    
    External format (first 20 dims):
      [0]: L_gripper
      [1]: R_gripper
      [2:5]: L_xyz
      [5:11]: L_rot6d
      [11:14]: R_xyz
      [14:20]: R_rot6d
    
    SWAP: external left <-> external right
      - external left -> our right
      - external right -> our left
    
    Returns dict with:
      - left_ee_pos_quat: [7] xyz + quat (from external RIGHT)
      - right_ee_pos_quat: [7] xyz + quat (from external LEFT)
      - gripper_pose: [2] R_grip, L_grip
    """
    state = np.asarray(state_68, dtype=np.float32).reshape(-1)
    
    # Extract first 20 dims
    s = state[:20]
    
    # Extract components (external naming)
    ext_l_grip = s[0]  # external left
    ext_r_grip = s[1]  # external right
    ext_l_xyz = s[2:5]
    ext_l_rot6d = s[5:11]
    ext_r_xyz = s[11:14]
    ext_r_rot6d = s[14:20]
    
    # Convert rot6d to quat
    ext_l_quat = rot6d_to_quat(ext_l_rot6d)
    ext_r_quat = rot6d_to_quat(ext_r_rot6d)
    
    # SWAP: external left -> our right, external right -> our left
    left_ee_pos_quat = np.concatenate([ext_r_xyz, ext_r_quat]).astype(np.float32)  # ext right -> our left
    right_ee_pos_quat = np.concatenate([ext_l_xyz, ext_l_quat]).astype(np.float32)  # ext left -> our right
    gripper_pose = np.array([ext_r_grip, ext_l_grip], dtype=np.float32)  # R, L order
    
    return {
        "left_ee_pos_quat": left_ee_pos_quat,
        "right_ee_pos_quat": right_ee_pos_quat,
        "gripper_pose": gripper_pose,
    }


def process_single_episode(
    parquet_path: Path,
    episode_meta: dict,
    task_description: str,
    output_dir: Path,
) -> bool:
    """
    Process a single episode from external format to intermediate format.
    
    Returns True if successful, False otherwise.
    """
    episode_idx = episode_meta["episode_index"]
    base_name = f"episode_{episode_idx:06d}"
    
    try:
        # Read parquet
        df = pd.read_parquet(parquet_path)
        n_frames = len(df)
        
        if n_frames == 0:
            print(f"[WARN] Empty episode: {base_name}")
            return False
        
        # Extract state data
        obs_data = []
        
        for idx in range(n_frames):
            row = df.iloc[idx]
            
            # Extract state from 68D
            state_68 = row["state"]
            state_dict = extract_state_from_68d(state_68)
            
            # Build observation row
            obs_row = {
                "observation/state/left_ee_pos_quat": state_dict["left_ee_pos_quat"],
                "observation/state/right_ee_pos_quat": state_dict["right_ee_pos_quat"],
                "observation/state/gripper_pose": state_dict["gripper_pose"],
            }
            
            # Build next_observation (from next row, or same row for last frame)
            if idx < n_frames - 1:
                next_row = df.iloc[idx + 1]
                next_state_68 = next_row["state"]
            else:
                next_state_68 = state_68  # Last frame: use current state
            
            next_state_dict = extract_state_from_68d(next_state_68)
            obs_row["next_observation/state/left_ee_pos_quat"] = next_state_dict["left_ee_pos_quat"]
            obs_row["next_observation/state/right_ee_pos_quat"] = next_state_dict["right_ee_pos_quat"]
            obs_row["next_observation/state/gripper_pose"] = next_state_dict["gripper_pose"]
            
            obs_data.append(obs_row)
        
        # Create output dataframe
        out_df = pd.DataFrame(obs_data)
        
        # Save parquet
        out_parquet_path = output_dir / f"{base_name}.parquet"
        out_df.to_parquet(out_parquet_path, index=False)
        
        # Save meta.json
        meta = {
            "fps": 10,
            "cameras": ["third_d455", "left_wrist_lumos", "right_wrist_lumos"],
            "task_description": task_description,
            "episode_failed": False,
        }
        meta_path = output_dir / f"{base_name}_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        
        # Extract and save videos
        for ext_field, cam_name in CAMERA_MAPPING.items():
            video_path = output_dir / f"{base_name}_{cam_name}.mp4"
            
            # Get image shape from first frame (decode first)
            first_img_raw = df[ext_field].iloc[0]
            first_img = decode_image(first_img_raw)
            h, w = first_img.shape[:2]
            
            # Write video
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (w, h))
            
            for idx in range(n_frames):
                img_raw = df[ext_field].iloc[idx]
                img = decode_image(img_raw)
                
                # Ensure uint8
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                
                # Convert RGB to BGR for OpenCV
                if img.shape[-1] == 3:
                    img_bgr = img[..., ::-1]
                else:
                    img_bgr = img
                
                writer.write(img_bgr)
            
            writer.release()
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to process {base_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main(_: list[str]) -> None:
    input_dir = Path(FLAGS.input_dir)
    output_dir = Path(FLAGS.output_dir)
    
    # Check input directory
    if not input_dir.exists():
        raise ValueError(f"Input directory not found: {input_dir}")
    
    # Create output directory
    if output_dir.exists():
        if FLAGS.overwrite:
            import shutil
            shutil.rmtree(output_dir)
        else:
            raise ValueError(f"Output directory exists: {output_dir}. Use --overwrite to replace.")
    output_dir.mkdir(parents=True)
    
    # Load meta information
    meta_dir = input_dir / "meta"
    
    # Load tasks
    tasks_path = meta_dir / "tasks.jsonl"
    tasks = {}
    if tasks_path.exists():
        with open(tasks_path) as f:
            for line in f:
                task = json.loads(line)
                tasks[task["task_index"]] = task["task"]
    
    # Load episodes
    episodes_path = meta_dir / "episodes.jsonl"
    episodes = []
    if episodes_path.exists():
        with open(episodes_path) as f:
            for line in f:
                episodes.append(json.loads(line))
    
    if not episodes:
        raise ValueError(f"No episodes found in {episodes_path}")
    
    print(f"Found {len(episodes)} episodes to process")
    print(f"Output directory: {output_dir}")
    
    # Process each episode
    success_count = 0
    
    for episode_meta in tqdm(episodes, desc="Converting episodes"):
        episode_idx = episode_meta["episode_index"]
        chunk_idx = episode_idx // 1000  # Assuming chunk size 1000
        
        parquet_path = input_dir / f"data/chunk-{chunk_idx:03d}/episode_{episode_idx:06d}.parquet"
        
        if not parquet_path.exists():
            print(f"[WARN] Parquet not found: {parquet_path}")
            continue
        
        # Get task description
        # episodes.jsonl has "tasks" as a list of task description strings directly
        task_list = episode_meta.get("tasks", [])
        if task_list and isinstance(task_list[0], str):
            # tasks is already a list of description strings
            task_description = task_list[0]
        elif task_list and isinstance(task_list[0], int):
            # tasks is a list of task indices (fallback for other formats)
            task_description = tasks.get(task_list[0], "unknown task")
        else:
            task_description = "unknown task"
        
        # Process episode
        if process_single_episode(parquet_path, episode_meta, task_description, output_dir):
            success_count += 1
    
    print(f"\nConversion complete: {success_count}/{len(episodes)} episodes successfully converted")
    print(f"Output location: {output_dir}")
    
    # Print sample file list
    files = sorted(os.listdir(output_dir))[:6]
    if files:
        print("\nSample output files:")
        for f in files:
            print(f"  {f}")


if __name__ == "__main__":
    app.run(main)
