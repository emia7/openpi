"""
[Server Machine] High-Performance LeRobot Dataset Converter for UMI Data.
Features:
1. Schema matches Franka Teleop (tcp_pose, gripper_pose, fish_eye_front).
2. Action: 7D [pos(3), rotvec(3), absolute_gripper_width(1)].
3. Extremely fast video reading using imageio iterator.

Usage:
    python process_umi_data_fast.py --stage1_dir=./umi_data --repo=local/umi_demo
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import imageio.v3 as iio
from scipy.spatial.transform import Rotation as R
from unittest.mock import patch

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
    achieved_fps = orig_fps / target_fps  # 修正：之前这里除错了，应该是 orig/stride
    achieved_fps = orig_fps / stride
    return stride, achieved_fps

def main(
    stage1_dir: str,
    repo_name: str,
    robot_type: str = "franka_research_3",
    target_fps: float = 10.0,
    task_text: str = "put the purple cup into the plate",
):
    stage1_dir = Path(stage1_dir)
    assert stage1_dir.exists(), f"stage1_dir not found: {stage1_dir}"

    mp4_files = sorted(stage1_dir.glob("episode*.mp4"))
    if not mp4_files:
        raise FileNotFoundError(f"No episode*.mp4 found in {stage1_dir}")

    first_json = stage1_dir / (mp4_files[0].stem + ".json")
    if not first_json.exists():
        raise FileNotFoundError(f"Missing json for first mp4: {first_json}")

    poses0, clamp0, fps0, _ = load_episode_json(first_json)

    # Downsample config
    stride, achieved_fps = compute_stride(fps0, float(target_fps))
    fps_int = int(round(achieved_fps))
    fps_int = max(1, fps_int)

    print(f"[INFO] Downsample: {fps0}Hz -> {achieved_fps:.4f}Hz (stride={stride}), dataset fps={fps_int}")

    # Peek first frame for H,W
    first_frame = iio.imread(mp4_files[0], index=0)
    if first_frame.ndim != 3 or first_frame.shape[2] != 3:
        raise ValueError(f"Unexpected frame shape: {first_frame.shape}")
    H, W, _ = first_frame.shape

    # Prepare output repo (clean once)
    out_path = HF_LEROBOT_HOME / repo_name
    if out_path.exists():
        shutil.rmtree(out_path)

    use_clamp = clamp0 is not None
    cam_key = "fish_eye_front"

    # =========================================================================
    # [修改 1] 对齐 Franka Teleop 格式的 Features
    # =========================================================================
    features = {
        f"observation.images.{cam_key}": {
            "dtype": "image",
            "shape": (H, W, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state.tcp_pose": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "qx", "qy", "qz", "qw"],
        },
        "observation.state.gripper_pose": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["gripper"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["x", "y", "z", "rx", "ry", "rz", "gripper_width"],
        },
        "task_idx": {"dtype": "int64", "shape": (1,), "names": None},
        "subtask_idx": {"dtype": "int64", "shape": (1,), "names": None},
        "is_failed": {"dtype": "bool", "shape": (1,), "names": None},
        "reward": {"dtype": "float32", "shape": (1,), "names": None},
        "return": {"dtype": "float32", "shape": (1,), "names": None},
        "done": {"dtype": "bool", "shape": (1,), "names": None},
    }

    # === [修改 2] Monkey Patch mkdir 以解决 FileExistsError ===
    original_mkdir = Path.mkdir
    def patched_mkdir(self, mode=0o777, parents=False, exist_ok=False):
        return original_mkdir(self, mode, parents, exist_ok=True)

    with patch('pathlib.Path.mkdir', side_effect=patched_mkdir, autospec=True):
        dataset = LeRobotDataset.create(
            repo_id=repo_name,
            robot_type=robot_type,
            fps=fps_int,
            features=features,
            image_writer_threads=8,
            image_writer_processes=4,
        )
    # =========================================================================

    total_eps = 0
    total_frames = 0

    for mp4_path in mp4_files:
        json_path = stage1_dir / (mp4_path.stem + ".json")
        if not json_path.exists():
            print(f"[WARN] Missing json for {mp4_path.name}, skip")
            continue

        poses, clamp, fps_meta, T = load_episode_json(json_path)

        if use_clamp and clamp is None:
            raise ValueError(f"{json_path} has no clamp but dataset expects clamp.")

        # =========================================================================
        # [核心继承] 保留了你第一个脚本的高效指针遍历逻辑
        # =========================================================================
        ds_indices = np.arange(0, T, stride, dtype=np.int64)
        T_ds = int(ds_indices.shape[0])
        ds_ptr = 0
        written = 0

        # 流式读取视频，不把整个视频加载进内存
        for i, frame in enumerate(iio.imiter(mp4_path)):
            if i >= T:
                break
            if ds_ptr >= T_ds or i != ds_indices[ds_ptr]:
                continue

            i_obs = int(ds_indices[ds_ptr])
            i_next = int(ds_indices[ds_ptr + 1]) if ds_ptr + 1 < T_ds else i_obs

            frame = np.asarray(frame, dtype=np.uint8)

            # -------------------------------------------------------------
            # [修改 3] 映射到标准 State (tcp_pose 7D, gripper_pose 1D)
            # -------------------------------------------------------------
            # tcp_pose: 直接使用 poses[i_obs] (里面是 x,y,z, qx,qy,qz,qw)
            tcp_pose = poses[i_obs].astype(np.float32)
            
            if use_clamp:
                gripper_pose = np.array([float(clamp[i_obs].item())], dtype=np.float32)
            else:
                gripper_pose = np.array([0.0], dtype=np.float32)

            # -------------------------------------------------------------
            # [修改 4] 映射到标准 Action (pos 3D + rotvec 3D + ABSOLUTE clamp 1D)
            # -------------------------------------------------------------
            pos_n, rotvec_n = pose7_to_pos_rotvec(poses[i_next])
            
            if use_clamp:
                # 动作直接用下一帧的绝对宽度
                grip_n = np.array([float(clamp[i_next].item())], dtype=np.float32)
            else:
                grip_n = np.array([0.0], dtype=np.float32)

            action7 = np.concatenate([pos_n, rotvec_n, grip_n], axis=0).astype(np.float32)

            # -------------------------------------------------------------
            # [修改 5] 构建符合 Franka Teleop 格式的 Frame Dict
            # -------------------------------------------------------------
            dataset.add_frame(
                {
                    f"observation.images.{cam_key}": frame,
                    "observation.state.tcp_pose": tcp_pose,
                    "observation.state.gripper_pose": gripper_pose,
                    "action": action7,
                    "task_idx": np.array([0], dtype=np.int64),
                    "subtask_idx": np.array([0], dtype=np.int64),
                    "is_failed": np.array([False], dtype=bool),
                    "reward": np.array([0.0], dtype=np.float32),
                    "return": np.array([0.0], dtype=np.float32),
                    "done": np.array([ds_ptr == T_ds - 1], dtype=bool),
                    "task": task_text,
                }
            )

            written += 1
            ds_ptr += 1

        if written != T_ds:
            raise RuntimeError(f"[STRICT ERROR] {mp4_path.name}: written={written} != T_ds={T_ds}.")

        dataset.save_episode()
        total_eps += 1
        total_frames += written
        print(f"[OK] {mp4_path.name} -> episode_saved, frames_used={written}")

    print(f"\nDONE -> repo saved at: {out_path}")
    print(f"episodes={total_eps}, total_frames_written={total_frames}, dataset_fps={fps_int}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True, help="Directory containing episodeXXXXX.mp4/json")
    p.add_argument("--repo_id", required=True, help="Output LeRobot repo name")
    p.add_argument("--robot_type", default="franka_research_3")
    p.add_argument("--target_fps", type=float, default=10.0, help="Downsample target fps, e.g. 10")
    p.add_argument("--task_text", type=str, default="put cubes into the cabinet")
    args = p.parse_args()

    main(
        stage1_dir=args.input_dir,
        repo_name=args.repo_id,
        robot_type=args.robot_type,
        target_fps=args.target_fps,
        task_text=args.task_text,
    )