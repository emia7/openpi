import argparse
import json
import shutil
from pathlib import Path

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

    fps = float(meta.get("fps", 30.0))
    return poses, clamp, fps, len(records)


def compute_stride(orig_fps: float, target_fps: float) -> tuple[int, float]:
    """Return (stride, achieved_fps)."""
    if target_fps <= 0:
        raise ValueError("target_fps must be > 0")
    stride = int(round(orig_fps / target_fps))
    stride = max(1, stride)
    achieved_fps = orig_fps / stride
    return stride, achieved_fps


def main(
    stage1_dir: str,
    repo_name: str,
    robot_type: str = "XV",
    target_fps: float = 10.0,
    task_text: str = "put the purple cup into the plate",
):
    stage1_dir = Path(stage1_dir)
    assert stage1_dir.exists(), f"stage1_dir not found: {stage1_dir}"

    # 1. 寻找 Episodes
    json_files = sorted(stage1_dir.glob("episode*.json"))
    if not json_files:
        raise FileNotFoundError(f"No episode*.json found in {stage1_dir}")

    # 检查第一个 Episode，获取基础信息
    first_json = json_files[0]
    ep_stem = first_json.stem
    
    path_head_0 = stage1_dir / f"{ep_stem}_head.mp4"
    path_left_0 = stage1_dir / f"{ep_stem}_left.mp4"

    if not path_head_0.exists() or not path_left_0.exists():
        raise FileNotFoundError(f"Missing mp4 files for {ep_stem}")

    poses0, clamp0, fps0, _ = load_episode_json(first_json)

    # 计算降采样步长
    stride, achieved_fps = compute_stride(fps0, float(target_fps))
    fps_int = int(round(achieved_fps))
    fps_int = max(1, fps_int)

    if abs(achieved_fps - target_fps) / max(target_fps, 1e-6) > 0.05:
        print(f"[WARN] Stride={stride} => achieved_fps={achieved_fps:.4f}Hz (dataset fps set to {fps_int}).")
    else:
        print(f"[INFO] Downsample: {fps0}Hz -> {achieved_fps:.4f}Hz (stride={stride})")

    # 获取图像尺寸
    frame_head = iio.imread(path_head_0, index=0)
    H_h, W_h, _ = frame_head.shape
    frame_left = iio.imread(path_left_0, index=0)
    H_l, W_l, _ = frame_left.shape

    print(f"[INFO] Head: {frame_head.shape}, Left: {frame_left.shape}")

    # 准备输出仓库
    out_path = HF_LEROBOT_HOME / repo_name
    if out_path.exists():
        print(f"[INFO] Removing existing repo at {out_path}")
        shutil.rmtree(out_path)

    use_clamp = clamp0 is not None

    # 创建数据集结构
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type=robot_type,
        fps=fps_int,
        features={
            "head_view": {
                "dtype": "video",
                "shape": (H_h, W_h, 3),
                "names": ["height", "width", "channel"],
            },
            "left_view": {
                "dtype": "video",
                "shape": (H_l, W_l, 3),
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

    for json_path in json_files:
        ep_stem = json_path.stem
        path_head = stage1_dir / f"{ep_stem}_head.mp4"
        path_left = stage1_dir / f"{ep_stem}_left.mp4"

        if not path_head.exists() or not path_left.exists():
            print(f"[SKIP] Missing videos for {ep_stem}")
            continue

        poses, clamp, fps_meta, T_json = load_episode_json(json_path)

        if use_clamp and clamp is None:
            raise ValueError(f"{json_path} has no clamp but dataset expects clamp.")

        # 理论上的目标帧数 (基于 JSON)
        ds_indices = np.arange(0, T_json, stride, dtype=np.int64)
        T_target = len(ds_indices)

        # Demo Start Pose
        pos0, rotvec0 = pose7_to_pos_rotvec(poses[0])
        demo_start_pose = np.concatenate([pos0, rotvec0], axis=0).astype(np.float32)

        # Video Iterators
        iter_head = iio.imiter(path_head)
        iter_left = iio.imiter(path_left)

        ds_ptr = 0
        written = 0
        
        # === 核心修改区 ===
        try:
            # zip 保证了只要有一个视频流结束，循环就结束 -> 自动避免越界
            for i, (frame_h, frame_l) in enumerate(zip(iter_head, iter_left)):
                
                # 1. 超过 JSON 长度则停止
                if i >= T_json:
                    break
                
                # 2. 降采样逻辑
                if ds_ptr >= T_target or i != ds_indices[ds_ptr]:
                    continue

                # === Observation ===
                i_obs = int(ds_indices[ds_ptr])
                
                frame_h = np.asarray(frame_h, dtype=np.uint8)
                frame_l = np.asarray(frame_l, dtype=np.uint8)

                pos, rotvec = pose7_to_pos_rotvec(poses[i_obs])
                
                # [修复] 显式构建 array，防止 0-d tensor 错误
                gw = float(clamp[i_obs].item()) if use_clamp else 0.0
                gripper_width = np.array([gw], dtype=np.float32)

                # === Action (Next State) ===
                if ds_ptr + 1 < T_target:
                    i_next = int(ds_indices[ds_ptr + 1])
                else:
                    i_next = i_obs

                pos_n, rotvec_n = pose7_to_pos_rotvec(poses[i_next])
                gw_n = float(clamp[i_next].item()) if use_clamp else 0.0
                grip_n = np.array([gw_n], dtype=np.float32)

                action7 = np.concatenate([pos_n, rotvec_n, grip_n], axis=0).astype(np.float32)

                # [注意] episode_end_idx 填入的是 T_target (理论值)
                # LeRobot 在 save_episode 时会自动处理实际写入的 written 数量
                dataset.add_frame({
                    "head_view": frame_h,
                    "left_view": frame_l,
                    "eef_pos": pos,
                    "eef_rot_axis_angle": rotvec,
                    "gripper_width": gripper_width,
                    "demo_start_pose": demo_start_pose,
                    "actions": action7,
                    "episode_start_idx": np.array([0], dtype=np.int64), 
                    "episode_end_idx": np.array([T_target], dtype=np.int64),
                    "task": task_text,
                })

                written += 1
                ds_ptr += 1

        except Exception as e:
            print(f"[WARN] Error reading {ep_stem} at frame {i}: {e}. Truncating episode here.")

        # 检查是否截断
        if written != T_target:
            print(f"[WARN] {ep_stem}: Truncated. JSON has {T_target} frames, but only wrote {written}.")
        
        # 这一步会提交所有 add_frame 的数据，自动更新索引
        dataset.save_episode()
        
        total_eps += 1
        total_frames += written
        print(f"[OK] {ep_stem} -> frames={written}")

    print(f"\nDONE -> repo saved at: {out_path}")
    print(f"Total Episodes: {total_eps}, Total Frames: {total_frames}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_dir", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--robot_type", default="XV")
    p.add_argument("--target_fps", type=float, default=10.0)
    p.add_argument("--task", type=str, default="put the purple cup into the plate")
    args = p.parse_args()

    main(
        stage1_dir=args.stage1_dir,
        repo_name=args.repo,
        robot_type=args.robot_type,
        target_fps=args.target_fps,
        task_text=args.task,
    )
