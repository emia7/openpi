import argparse
import json
import shutil
from pathlib import Path
import random

import numpy as np
import imageio.v3 as iio
from scipy.spatial.transform import Rotation as R

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


def pose7_to_pos_rotvec(pose7: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """pose7: [x,y,z,qx,qy,qz,qw] -> (pos(3,), rotvec(3,))"""
    pos = pose7[:3].astype(np.float32)
    quat = pose7[3:7].astype(np.float32)  # (qx,qy,qz,qw)
    rotvec = R.from_quat(quat).as_rotvec().astype(np.float32)
    return pos, rotvec


def load_episode_json(json_path: Path):
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    records = meta["records"]
    if len(records) < 2:
        raise ValueError(f"{json_path} has <2 records, cannot build episode.")
    poses = np.asarray([r["pose"] for r in records], dtype=np.float32)  # (T,7)

    if "clamp" in records[0]:
        clamp = np.asarray([r["clamp"] for r in records], dtype=np.float32).reshape(-1, 1)  # (T,1)
    else:
        clamp = None

    fps = float(meta.get("fps", 60.0))
    return poses, clamp, fps, len(records)


def compute_stride(orig_fps: float, target_fps: float) -> tuple[int, float]:
    """Return (stride, achieved_fps)."""
    if target_fps <= 0:
        raise ValueError("target_fps must be > 0")
    stride = int(round(orig_fps / target_fps))
    stride = max(1, stride)
    achieved_fps = orig_fps / stride
    return stride, achieved_fps


def gather_all_episodes(input_dirs: list[str]) -> list[Path]:
    """Scan all directories and collect valid episode mp4 paths."""
    all_mp4_files = []
    for d in input_dirs:
        path = Path(d)
        if not path.exists():
            print(f"[WARN] Input directory not found: {d}")
            continue
        
        # Sort to ensure deterministic order before shuffling
        files = sorted(list(path.glob("episode*.mp4")))
        print(f"  - Found {len(files)} episodes in {d}")
        all_mp4_files.extend(files)
    
    return all_mp4_files


def main(
    input_dirs: list[str],
    repo_name: str,
    robot_type: str = "XV",
    target_fps: float = 10.0,
    task_text: str = "put the purple cup into the plate",
    num_episodes: int = None,
    seed: int = 42,
):
    # 1. Gather all episodes from multiple directories
    print(f"Scanning input directories: {input_dirs}")
    all_mp4_files = gather_all_episodes(input_dirs)
    
    if not all_mp4_files:
        raise FileNotFoundError("No episode*.mp4 files found in any input directory.")

    total_found = len(all_mp4_files)
    print(f"Total episodes found: {total_found}")

    # 2. Random Sampling
    # Set seed for reproducibility
    rng = np.random.default_rng(seed)
    
    # Shuffle the list
    # Convert Path objects to strings for shuffling to be safe, though numpy handles objects ok
    # using an index array to shuffle is safer for maintaining type
    indices = np.arange(total_found)
    rng.shuffle(indices)
    shuffled_files = [all_mp4_files[i] for i in indices]

    # Slice the list if num_episodes is set
    if num_episodes is not None and num_episodes > 0:
        if num_episodes < total_found:
            selected_files = shuffled_files[:num_episodes]
            print(f"Sampling: Selected {num_episodes} episodes out of {total_found} (Seed: {seed})")
        else:
            selected_files = shuffled_files
            print(f"Sampling: Requested {num_episodes} but only {total_found} available. Using all.")
    else:
        selected_files = shuffled_files
        print(f"Using all {total_found} episodes.")

    # 3. Initialization using the first selected episode
    first_mp4 = selected_files[0]
    first_json = first_mp4.with_suffix(".json")
    
    if not first_json.exists():
        raise FileNotFoundError(f"Missing json for first selected episode: {first_json}")

    poses0, clamp0, fps0, _ = load_episode_json(first_json)

    # Downsample config
    stride, achieved_fps = compute_stride(fps0, float(target_fps))
    fps_int = int(round(achieved_fps))
    fps_int = max(1, fps_int)

    if abs(achieved_fps - target_fps) / max(target_fps, 1e-6) > 0.05:
        print(
            f"[WARN] target_fps={target_fps}Hz not exactly achievable from orig_fps={fps0}Hz "
            f"with integer stride. Using stride={stride} => achieved_fps={achieved_fps:.4f}Hz "
            f"(dataset fps set to {fps_int})."
        )
    else:
        print(f"[INFO] Downsample: {fps0}Hz -> {achieved_fps:.4f}Hz (stride={stride}), dataset fps={fps_int}")

    # Peek first frame for H,W
    first_frame = iio.imread(first_mp4, index=0)
    if first_frame.ndim != 3 or first_frame.shape[2] != 3:
        raise ValueError(f"Unexpected frame shape: {first_frame.shape}")
    H, W, _ = first_frame.shape

    # Prepare output repo (clean once)
    out_path = HF_LEROBOT_HOME / repo_name
    if out_path.exists():
        print(f"Removing existing dataset at {out_path}")
        shutil.rmtree(out_path)

    # Decide if clamp exists (dataset schema)
    use_clamp = clamp0 is not None

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type=robot_type,
        fps=fps_int,
        features={
            "image": {
                "dtype": "image",
                "shape": (H, W, 3),
                "names": ["height", "width", "channel"],
            },
            "eef_pos": {
                "dtype": "float32",
                "shape": (3,),
                "names": ["x", "y", "z"],
            },
            "eef_rot_axis_angle": {
                "dtype": "float32",
                "shape": (3,),
                "names": ["rx", "ry", "rz"],
            },
            "gripper_width": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_width"],
            },
            "demo_start_pose": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["x", "y", "z", "rx", "ry", "rz"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["x", "y", "z", "rx", "ry", "rz", "gripper_width"],
            },
            "episode_start_idx": {
                "dtype": "int64",
                "shape": (1,),
                "names": ["episode_start_idx"],
            },
            "episode_end_idx": {
                "dtype": "int64",
                "shape": (1,),
                "names": ["episode_end_idx"],
            },
        },
        image_writer_threads=8,
        image_writer_processes=4,
    )

    total_eps = 0
    total_frames = 0
    
    # 4. Processing Loop
    import tqdm
    for mp4_path in tqdm.tqdm(selected_files, desc="Processing Episodes"):
        json_path = mp4_path.with_suffix(".json")
        if not json_path.exists():
            print(f"[WARN] Missing json for {mp4_path.name}, skip")
            continue

        poses, clamp, fps_meta, T = load_episode_json(json_path)

        if abs(fps_meta - fps0) > 1e-3:
            # Note: We stick to the stride calculated from the first episode for consistency
            pass 

        if use_clamp and clamp is None:
            raise ValueError(f"{json_path} has no clamp but dataset expects clamp (first episode had clamp).")

        # Strict downsample indices and strict T_ds
        ds_indices = np.arange(0, T, stride, dtype=np.int64)
        T_ds = int(ds_indices.shape[0])

        # Episode metadata (strict)
        episode_start_idx = np.array([0], dtype=np.int64)
        episode_end_idx = np.array([T_ds], dtype=np.int64)

        # Demo start pose from original first record (t=0)
        pos0, rotvec0 = pose7_to_pos_rotvec(poses[0])
        demo_start_pose = np.concatenate([pos0, rotvec0], axis=0).astype(np.float32)  # (6,)

        # Use pointer approach for efficiency
        ds_ptr = 0
        written = 0

        for i, frame in enumerate(iio.imiter(str(mp4_path))):
            if i >= T:
                break
            if ds_ptr >= T_ds or i != ds_indices[ds_ptr]:
                continue

            # Current frame index
            i_obs = int(ds_indices[ds_ptr])

            # Next frame index (for action)
            if ds_ptr + 1 < T_ds:
                i_next = int(ds_indices[ds_ptr + 1])
            else:
                i_next = i_obs  # Last frame repeats

            frame = np.asarray(frame, dtype=np.uint8)

            # ===== observation from i_obs =====
            pos, rotvec = pose7_to_pos_rotvec(poses[i_obs])
            if use_clamp:
                gripper_width = np.array([float(clamp[i_obs].item())], dtype=np.float32)
            else:
                gripper_width = np.array([0.0], dtype=np.float32)

            # ===== action = next-state (absolute) from i_next =====
            pos_n, rotvec_n = pose7_to_pos_rotvec(poses[i_next])
            if use_clamp:
                grip_n = np.array([float(clamp[i_next].item())], dtype=np.float32)
            else:
                grip_n = np.array([0.0], dtype=np.float32)

            action7 = np.concatenate([pos_n, rotvec_n, grip_n], axis=0).astype(np.float32)

            dataset.add_frame(
                {
                    "image": frame,
                    "eef_pos": pos,
                    "eef_rot_axis_angle": rotvec,
                    "gripper_width": gripper_width,
                    "demo_start_pose": demo_start_pose,
                    "actions": action7,
                    "episode_start_idx": episode_start_idx,
                    "episode_end_idx": episode_end_idx,
                    "task": task_text,
                }
            )

            written += 1
            ds_ptr += 1

        # Strict sanity check
        if written != T_ds:
            print(f"[WARN] {mp4_path.name}: Written {written} frames, expected {T_ds}. Metadata might be slightly off.")
            # We save anyway as long as something was written
        
        if written > 0:
            dataset.save_episode()
            total_eps += 1
            total_frames += written
        else:
            print(f"[ERROR] Failed to write any frames for {mp4_path.name}")

    print(f"\nDONE -> repo saved at: {out_path}")
    print(
        f"episodes={total_eps}, total_frames_written={total_frames}, "
        f"orig_fps≈{fps0}, stride={stride}, achieved_fps={achieved_fps:.4f}, dataset_fps={fps_int}, clamp={use_clamp}"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # Accept multiple directories
    p.add_argument("--input_dirs", nargs='+', required=True, help="List of directories containing episodeXXXXX.mp4/json")
    p.add_argument("--repo", required=True, help="Output LeRobot repo name")
    p.add_argument("--robot_type", default="XV")
    p.add_argument("--target_fps", type=float, default=10.0, help="Downsample target fps, e.g. 10")
    p.add_argument("--task", type=str, default="put the purple cup into the plate")
    # New sampling arguments
    p.add_argument("--num_episodes", type=int, default=None, help="Number of episodes to sample. If not set, use all.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    
    args = p.parse_args()

    main(
        input_dirs=args.input_dirs,
        repo_name=args.repo,
        robot_type=args.robot_type,
        target_fps=args.target_fps,
        task_text=args.task,
        num_episodes=args.num_episodes,
        seed=args.seed,
    )