"""
Build a LeRobot dataset from dual-arm Franka exports: Parquet + per-camera MP4 + meta JSON.

Expected layout (same stem for all files of one episode):
  {base}.parquet
  {base}_meta.json   with keys: fps, cameras (list of strings), task_description, episode_failed, ...
  {base}_{camera}.mp4 for each name in cameras

Parquet columns used:
  - observation/state/left_ee_pos_quat, right_ee_pos_quat (7 = xyz + qx,qy,qz,qw)
  - observation/state/gripper_pose (2 = left, right)
  - next_observation/state/left_ee_pos_quat, right_ee_pos_quat, gripper_pose
  Parquet "action" columns are ignored. Labels are built from next_observation; on the last
  frame, left_action/right_action copy the current observation (stay).

Run from repo root, e.g.:
  uv run python dual_scripts/convert_dual_franka_data_to_lerobot.py \\
    --input_dirs=/path/to/handover_high --repo_id=local/handover_high_dual
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
from absl import app, flags
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_multi_string(
    "input_dirs",
    None,
    "Directories containing parquet + meta + mp4 (one or more paths).",
    required=True,
)
flags.DEFINE_string("output_dir", None, "Output root; default HF_LEROBOT_HOME.")
flags.DEFINE_string("repo_id", "local/dual_franka_demo", "LeRobot repo id (folder name under output).")
flags.DEFINE_string("robot_type", "dual_franka", "LeRobot robot_type string.")
flags.DEFINE_bool("skip_failed", True, "Skip episodes whose meta has episode_failed=true.")
flags.DEFINE_integer("num_episodes", None, "Random subsample size; None = all.", lower_bound=1)
flags.DEFINE_integer("seed", 42, "RNG seed for subsampling.")

COL_OBS_L = "observation/state/left_ee_pos_quat"
COL_OBS_R = "observation/state/right_ee_pos_quat"
COL_OBS_G = "observation/state/gripper_pose"
COL_NEXT_L = "next_observation/state/left_ee_pos_quat"
COL_NEXT_R = "next_observation/state/right_ee_pos_quat"
COL_NEXT_G = "next_observation/state/gripper_pose"

REQUIRED_COLS = [COL_OBS_L, COL_OBS_R, COL_OBS_G, COL_NEXT_L, COL_NEXT_R, COL_NEXT_G]


def pose7_quat_to_pose6(pose7: np.ndarray) -> np.ndarray:
    """[x,y,z,qx,qy,qz,qw] -> [x,y,z,rx,ry,rz] rotvec."""
    p = np.asarray(pose7, dtype=np.float64).reshape(-1)
    if p.size != 7:
        raise ValueError(f"Expected pose7, got shape {p.shape}")
    pos = p[:3].astype(np.float32)
    quat = p[3:7].astype(np.float64)
    rotvec = R.from_quat(quat).as_rotvec().astype(np.float32)
    return np.concatenate([pos, rotvec], axis=0)


def pose7_quat_grip_to_action7(pose7: np.ndarray, gripper: float) -> np.ndarray:
    """Absolute target: pos3 + rotvec3 + gripper1."""
    p6 = pose7_quat_to_pose6(pose7)
    g = np.array([float(gripper)], dtype=np.float32)
    return np.concatenate([p6, g], axis=0).astype(np.float32)


def parse_gripper2(row: pd.Series) -> tuple[float, float]:
    gp = np.asarray(row[COL_OBS_G], dtype=np.float32).reshape(-1)
    if gp.size < 2:
        raise ValueError(f"Expected gripper_pose length >= 2, got {gp.size}")
    return float(gp[0]), float(gp[1])


def parse_gripper2_next(row: pd.Series) -> tuple[float, float]:
    gp = np.asarray(row[COL_NEXT_G], dtype=np.float32).reshape(-1)
    if gp.size < 2:
        raise ValueError(f"Expected next gripper_pose length >= 2, got {gp.size}")
    return float(gp[0]), float(gp[1])


def observation_vectors_from_row(row: pd.Series) -> dict[str, np.ndarray]:
    """Current ee poses and grippers from observation/state."""
    l7 = np.asarray(row[COL_OBS_L], dtype=np.float32).reshape(-1)
    r7 = np.asarray(row[COL_OBS_R], dtype=np.float32).reshape(-1)
    gl, gr = parse_gripper2(row)
    l6 = pose7_quat_to_pose6(l7)
    r6 = pose7_quat_to_pose6(r7)
    return {
        "observation.state.left_eef_pos": l6[:3].astype(np.float32),
        "observation.state.left_eef_rotvec": l6[3:6].astype(np.float32),
        "observation.state.left_gripper": np.array([gl], dtype=np.float32),
        "observation.state.right_eef_pos": r6[:3].astype(np.float32),
        "observation.state.right_eef_rotvec": r6[3:6].astype(np.float32),
        "observation.state.right_gripper": np.array([gr], dtype=np.float32),
    }


def left_right_action7_from_next(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    l7 = np.asarray(row[COL_NEXT_L], dtype=np.float32).reshape(-1)
    r7 = np.asarray(row[COL_NEXT_R], dtype=np.float32).reshape(-1)
    gl, gr = parse_gripper2_next(row)
    return pose7_quat_grip_to_action7(l7, gl), pose7_quat_grip_to_action7(r7, gr)


def left_right_action7_from_obs(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    l7 = np.asarray(row[COL_OBS_L], dtype=np.float32).reshape(-1)
    r7 = np.asarray(row[COL_OBS_R], dtype=np.float32).reshape(-1)
    gl, gr = parse_gripper2(row)
    return pose7_quat_grip_to_action7(l7, gl), pose7_quat_grip_to_action7(r7, gr)


def gather_all_episodes(input_dirs: list[str], skip_failed: bool) -> list[dict]:
    all_episodes: list[dict] = []
    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            print(f"[WARN] Input directory not found: {input_dir}")
            continue

        files = sorted(f for f in os.listdir(input_dir) if f.endswith(".parquet"))
        for f in files:
            base_name = os.path.splitext(f)[0]
            meta_path = os.path.join(input_dir, f"{base_name}_meta.json")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path, encoding="utf-8") as mf:
                meta = json.load(mf)
            if skip_failed and meta.get("episode_failed", False):
                continue
            all_episodes.append(
                {
                    "parquet_path": os.path.join(input_dir, f),
                    "meta": meta,
                    "base_name": base_name,
                    "dir": input_dir,
                }
            )
    return all_episodes


def camera_shapes_from_episode(ep: dict, cameras: list[str]) -> dict[str, tuple[int, int, int]]:
    shapes: dict[str, tuple[int, int, int]] = {}
    for cam in cameras:
        vid_path = os.path.join(ep["dir"], f"{ep['base_name']}_{cam}.mp4")
        cap = cv2.VideoCapture(vid_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Could not read first frame: {vid_path}")
        h, w, c = frame.shape
        shapes[cam] = (h, w, c)
    return shapes


def assert_parquet_columns(df: pd.DataFrame, base_name: str) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{base_name}: missing columns {missing}")


def build_features(cameras: list[str], cam_shapes: dict[str, tuple[int, int, int]]) -> dict:
    features: dict = {
        "observation.state.left_eef_pos": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["x", "y", "z"],
        },
        "observation.state.left_eef_rotvec": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["rx", "ry", "rz"],
        },
        "observation.state.left_gripper": {"dtype": "float32", "shape": (1,), "names": ["gripper"]},
        "observation.state.right_eef_pos": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["x", "y", "z"],
        },
        "observation.state.right_eef_rotvec": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["rx", "ry", "rz"],
        },
        "observation.state.right_gripper": {"dtype": "float32", "shape": (1,), "names": ["gripper"]},
        "observation.state.demo_start_pose_left": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["x", "y", "z", "rx", "ry", "rz"],
        },
        "observation.state.demo_start_pose_right": {
            "dtype": "float32",
            "shape": (6,),
            "names": ["x", "y", "z", "rx", "ry", "rz"],
        },
        "left_action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
        },
        "right_action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
        },
        "task_idx": {"dtype": "int64", "shape": (1,), "names": None},
        "subtask_idx": {"dtype": "int64", "shape": (1,), "names": None},
        "done": {"dtype": "bool", "shape": (1,), "names": None},
    }
    for cam in cameras:
        h, w, c = cam_shapes[cam]
        features[f"observation.images.{cam}"] = {
            "dtype": "image",
            "shape": (h, w, c),
            "names": ["height", "width", "channel"],
        }
    return features


def main(_: list[str]) -> None:
    output_dir = Path(FLAGS.output_dir) if FLAGS.output_dir else Path(HF_LEROBOT_HOME)
    repo_id = FLAGS.repo_id

    print(f"Scanning input directories: {FLAGS.input_dirs}")
    all_episodes = gather_all_episodes(list(FLAGS.input_dirs), FLAGS.skip_failed)
    total_found = len(all_episodes)
    print(f"Total valid episodes found: {total_found}")
    if not all_episodes:
        print("No episodes found.")
        return

    rng = np.random.default_rng(FLAGS.seed)
    indices = np.arange(total_found)
    rng.shuffle(indices)
    shuffled = [all_episodes[i] for i in indices]

    if FLAGS.num_episodes is not None and FLAGS.num_episodes < total_found:
        selected = shuffled[: FLAGS.num_episodes]
        print(f"Sampling: {FLAGS.num_episodes} / {total_found} (seed={FLAGS.seed})")
    else:
        selected = shuffled
        if FLAGS.num_episodes is not None:
            print(f"Requested {FLAGS.num_episodes} but only {total_found} available; using all.")
        else:
            print(f"Using all {total_found} episodes.")

    output_path = output_dir / repo_id
    if output_path.exists():
        print(f"Removing existing dataset at: {output_path}")
        shutil.rmtree(output_path)

    first_ep = selected[0]
    cameras = list(first_ep["meta"]["cameras"])
    if len(cameras) == 0:
        raise ValueError("meta.cameras is empty")

    cam_shapes = camera_shapes_from_episode(first_ep, cameras)
    features = build_features(cameras, cam_shapes)

    for ep in selected[1:]:
        cams_ep = list(ep["meta"]["cameras"])
        if cams_ep != cameras:
            raise ValueError(
                f"Camera list mismatch: {ep['base_name']} has {cams_ep}, expected {cameras}"
            )
        shapes_ep = camera_shapes_from_episode(ep, cameras)
        for cam in cameras:
            if shapes_ep[cam] != cam_shapes[cam]:
                raise ValueError(
                    f"Frame shape mismatch for {cam}: {ep['base_name']} {shapes_ep[cam]} "
                    f"vs first episode {cam_shapes[cam]}"
                )

    fps = int(first_ep["meta"]["fps"])

    original_mkdir = Path.mkdir

    def patched_mkdir(self: Path, mode: int = 0o777, parents: bool = False, exist_ok: bool = False) -> None:
        original_mkdir(self, mode, parents, exist_ok=True)

    dataset_root = output_dir / repo_id

    with patch("pathlib.Path.mkdir", side_effect=patched_mkdir, autospec=True):
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            root=dataset_root,
            robot_type=FLAGS.robot_type,
            features=features,
        )

    total_frames = 0
    for ep in tqdm(selected, desc="Reconstructing episodes"):
        df = pd.read_parquet(ep["parquet_path"])
        meta = ep["meta"]
        if len(df) == 0:
            print(f"[WARN] Skip empty parquet: {ep['base_name']}")
            continue

        assert_parquet_columns(df, ep["base_name"])

        row0 = df.iloc[0]
        l0 = np.asarray(row0[COL_OBS_L], dtype=np.float32).reshape(-1)
        r0 = np.asarray(row0[COL_OBS_R], dtype=np.float32).reshape(-1)
        demo_left = pose7_quat_to_pose6(l0)
        demo_right = pose7_quat_to_pose6(r0)

        video_caps = {
            cam: cv2.VideoCapture(os.path.join(ep["dir"], f"{ep['base_name']}_{cam}.mp4"))
            for cam in cameras
        }

        n = len(df)
        for idx in range(n):
            row = df.iloc[idx]
            obs_vecs = observation_vectors_from_row(row)

            if idx < n - 1:
                la, ra = left_right_action7_from_next(row)
            else:
                la, ra = left_right_action7_from_obs(row)

            lerobot_frame = {
                **obs_vecs,
                "observation.state.demo_start_pose_left": demo_left.astype(np.float32),
                "observation.state.demo_start_pose_right": demo_right.astype(np.float32),
                "left_action": la,
                "right_action": ra,
                "task_idx": np.array([int(row.get("task_idx", 0))], dtype=np.int64),
                "subtask_idx": np.array([int(row.get("subtask_idx", 0))], dtype=np.int64),
                "done": np.array([idx == n - 1], dtype=bool),
                "task": meta["task_description"],
            }

            for cam in cameras:
                ret, img = video_caps[cam].read()
                if not ret:
                    raise RuntimeError(
                        f"Video {cam} ended early at step {idx}/{n} ({ep['base_name']})"
                    )
                lerobot_frame[f"observation.images.{cam}"] = img[..., ::-1]

            dataset.add_frame(lerobot_frame)

        for cap in video_caps.values():
            cap.release()
        dataset.save_episode()
        total_frames += n

    print(f"\nDataset created at: {output_path}")
    print(f"Total episodes: {len(selected)}")
    print(f"Total frames: {total_frames}")


if __name__ == "__main__":
    app.run(main)
