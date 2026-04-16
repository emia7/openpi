import argparse
import shutil
from pathlib import Path

import numpy as np
import imageio.v3 as iio

import stage2_core
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


def main(
    stage1_dir: str,
    repo_name: str,
    robot_type: str = "XV",
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

    poses0, clamp0, fps0, _ = stage2_core.load_episode_json(first_json, default_fps=60.0, require_clamp=False)

    # Downsample config
    stride, achieved_fps = stage2_core.compute_stride(fps0, float(target_fps))
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
    first_frame = iio.imread(mp4_files[0], index=0)
    if first_frame.ndim != 3 or first_frame.shape[2] != 3:
        raise ValueError(f"Unexpected frame shape: {first_frame.shape}")
    H, W, _ = first_frame.shape

    # Prepare output repo (clean once)
    out_path = HF_LEROBOT_HOME / repo_name
    if out_path.exists():
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

    for mp4_path in mp4_files:
        json_path = stage1_dir / (mp4_path.stem + ".json")
        if not json_path.exists():
            print(f"[WARN] Missing json for {mp4_path.name}, skip")
            continue

        poses, clamp, fps_meta, T = stage2_core.load_episode_json(json_path, default_fps=60.0, require_clamp=False)

        if abs(fps_meta - fps0) > 1e-3:
            print(f"[WARN] {json_path.name}: fps={fps_meta} differs from first fps={fps0}. Using stride={stride} anyway.")

        if use_clamp and clamp is None:
            raise ValueError(f"{json_path} has no clamp but dataset expects clamp (first episode had clamp).")

        # Strict downsample indices and strict T_ds
        ds_indices = np.arange(0, T, stride, dtype=np.int64)
        T_ds = int(ds_indices.shape[0])

        # Episode metadata (strict)
        episode_start_idx = np.array([0], dtype=np.int64)
        episode_end_idx = np.array([T_ds], dtype=np.int64)

        # Demo start pose from original first record (t=0)
        pos0, rotvec0 = stage2_core.pose7_to_pos_rotvec(poses[0])
        demo_start_pose = np.concatenate([pos0, rotvec0], axis=0).astype(np.float32)  # (6,)

        # Stream frames strictly: only keep i in ds_indices
        ds_set = set(ds_indices.tolist())
        written = 0

                # 先把下采样索引做成 list，保持顺序
        ds_indices = np.arange(0, T, stride, dtype=np.int64)
        T_ds = int(ds_indices.shape[0])

        # 用指针而不是 set，更清晰也更快
        ds_ptr = 0

        for i, frame in enumerate(iio.imiter(mp4_path)):
            if i >= T:
                break
            if ds_ptr >= T_ds or i != ds_indices[ds_ptr]:
                continue

            # 当前帧 index（observation）
            i_obs = int(ds_indices[ds_ptr])

            # 下一帧 index（action 指向下一帧 state）
            if ds_ptr + 1 < T_ds:
                i_next = int(ds_indices[ds_ptr + 1])
            else:
                i_next = i_obs  # 最后一帧：没有 next，就用自己（或用全 0，见下方建议）

            frame = np.asarray(frame, dtype=np.uint8)

            # ===== observation from i_obs =====
            pos, rotvec = stage2_core.pose7_to_pos_rotvec(poses[i_obs])
            if use_clamp:
                gripper_width = np.array([float(clamp[i_obs].item())], dtype=np.float32)
            else:
                gripper_width = np.array([0.0], dtype=np.float32)

            # ===== action = next-state (absolute) from i_next =====
            pos_n, rotvec_n = stage2_core.pose7_to_pos_rotvec(poses[i_next])
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


        # Strict sanity check: must match T_ds
        if written != T_ds:
            raise RuntimeError(
                f"[STRICT ERROR] {mp4_path.name}: written={written} != T_ds={T_ds}. "
                f"If you truly guarantee alignment, this should not happen."
            )

        dataset.save_episode()
        total_eps += 1
        total_frames += written
        print(f"[OK] {mp4_path.name} -> episode_saved, frames_used={written}")

    print(f"\nDONE -> repo saved at: {out_path}")
    print(
        f"episodes={total_eps}, total_frames_written={total_frames}, "
        f"orig_fps≈{fps0}, stride={stride}, achieved_fps={achieved_fps:.4f}, dataset_fps={fps_int}, clamp={use_clamp}"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_dir", required=True, help="Directory containing episodeXXXXX.mp4/json")
    p.add_argument("--repo", required=True, help="Output LeRobot repo name")
    p.add_argument("--robot_type", default="XV")
    p.add_argument("--target_fps", type=float, default=10.0, help="Downsample target fps, e.g. 10")
    p.add_argument("--task", type=str, default="put cubes into the cabinet")
    args = p.parse_args()

    main(
        stage1_dir=args.stage1_dir,
        repo_name=args.repo,
        robot_type=args.robot_type,
        target_fps=args.target_fps,
        task_text=args.task,
    )
