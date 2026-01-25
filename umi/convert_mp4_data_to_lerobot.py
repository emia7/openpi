import json
import shutil
from pathlib import Path

import numpy as np
import imageio.v3 as iio

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

from scipy.spatial.transform import Rotation as R
from openpi.policies.pose_util import pose7_to_pos_rotvec


def load_episode_json(json_path: Path):
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    records = meta["records"]
    if len(records) < 2:
        raise ValueError(f"{json_path} has <2 records, cannot build next-state actions.")
    poses = np.asarray([r["pose"] for r in records], dtype=np.float32)  # (T,7)

    # clamp is optional
    if "clamp" in records[0]:
        clamp = np.asarray([r["clamp"] for r in records], dtype=np.float32).reshape(-1, 1)  # (T,1)
    else:
        clamp = None

    fps = meta.get("fps", 60.0)
    return poses, clamp, float(fps), len(records)


def main(stage1_dir: str, repo_name: str, robot_type: str = "XV", fps: int | None = None):
    stage1_dir = Path(stage1_dir)
    assert stage1_dir.exists(), f"stage1_dir not found: {stage1_dir}"

    # Find episodes
    mp4_files = sorted(stage1_dir.glob("episode*.mp4"))
    if not mp4_files:
        raise FileNotFoundError(f"No episode*.mp4 found in {stage1_dir}")

    # Determine fps (integer) to keep video encoder happy
    # If fps not provided, take first json's fps and round.
    first_json = stage1_dir / (mp4_files[0].stem + ".json")
    if not first_json.exists():
        raise FileNotFoundError(f"Missing json for first mp4: {first_json}")

    _, _, fps0, _ = load_episode_json(first_json)
    if fps is None:
        fps_int = int(round(fps0))
        if fps_int <= 0:
            fps_int = 60
    else:
        fps_int = int(fps)

    # Peek first frame for H,W
    first_frame = iio.imread(mp4_files[0], index=0)
    if first_frame.ndim != 3 or first_frame.shape[2] != 3:
        raise ValueError(f"Unexpected frame shape: {first_frame.shape}")
    H, W, _ = first_frame.shape

    # Prepare output repo (clean once)
    out_path = HF_LEROBOT_HOME / repo_name
    if out_path.exists():
        shutil.rmtree(out_path)

    # Decide state/action dims: pose7 or pose7+clamp1
    # Look at first episode json to decide
    poses0, clamp0, _, _ = load_episode_json(first_json)
    use_clamp = clamp0 is not None
    state_dim = 8 if use_clamp else 7

    state_names = ["x", "y", "z", "qx", "qy", "qz", "qw"] + (["clamp"] if use_clamp else [])
    action_names = state_names  # next-state

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

    for mp4_path in mp4_files:
        json_path = stage1_dir / (mp4_path.stem + ".json")
        if not json_path.exists():
            print(f"[WARN] Missing json for {mp4_path.name}, skip")
            continue

        poses, clamp, _, T = load_episode_json(json_path)

        # start pose for this episode (pos+rotvec)
        pos0, rotvec0 = pose7_to_pos_rotvec(poses[0])
        demo_start_pose = np.concatenate([pos0, rotvec0], axis=0).astype(np.float32)  # (6,)

        episode_start_idx = np.array([0], dtype=np.int64)
        episode_end_idx = np.array([T], dtype=np.int64)


        # Build states (T,dim)
        if use_clamp:
            if clamp is None:
                raise ValueError(f"{json_path} has no clamp but dataset expects clamp.")
            states = np.concatenate([poses, clamp], axis=1).astype(np.float32)  # (T,8)
        else:
            states = poses.astype(np.float32)  # (T,7)

        # Stream frames, write T-1 steps
        frame_iter = iio.imiter(mp4_path)
        written = 0
        frame_iter = iio.imiter(mp4_path)
        written = 0
        for i, frame in enumerate(frame_iter):
            if i+1 >= T:
                break
            frame = np.asarray(frame, dtype=np.uint8)

            pos, rotvec = pose7_to_pos_rotvec(poses[i])
            action_pos, action_rotvec = pose7_to_pos_rotvec(poses[i+1])  # next-state

            if use_clamp:
                if clamp is None:
                    raise ValueError(f"{json_path} has no clamp but dataset expects clamp.")
                gripper_width = np.array([clamp[i].item()], dtype=np.float32)  # (1,)
                action_gripper_width = np.array([clamp[i+1].item()], dtype=np.float32)  # (1,)
            else:
                gripper_width = np.array([0.0], dtype=np.float32)
                action_gripper_width = np.array([0.0], dtype=np.float32)

            action7 = np.concatenate([action_pos, action_rotvec, action_gripper_width], axis=0).astype(np.float32)  # (7,)

            # 注意现在的action是取了下一时刻的位姿
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
                    "task": "put the purple cup into the plate",
                }
            )
            written += 1


        if written <= 0:
            print(f"[WARN] {mp4_path.name}: wrote 0 frames, skip save_episode")
            continue

        dataset.save_episode()
        total_eps += 1
        total_frames += written
        print(f"[OK] {mp4_path.name} -> episode_saved, frames_used={written}")

    print(f"\nDONE -> repo saved at: {out_path}")
    print(f"episodes={total_eps}, total_steps={total_frames}, fps={fps_int}, clamp_in_state={use_clamp}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--stage1_dir", required=True, help="Directory containing episodeXXXXX.mp4/json")
    p.add_argument("--repo", required=True, help="Output LeRobot repo name")
    p.add_argument("--robot_type", default="XV")
    p.add_argument("--fps", type=int, default=None, help="Force integer fps (recommended). If omitted, round from json.")
    args = p.parse_args()

    main(args.stage1_dir, args.repo, args.robot_type, args.fps)
