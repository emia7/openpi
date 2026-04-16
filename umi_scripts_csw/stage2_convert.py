#!/usr/bin/env python3
"""Unified Stage2 converter: mp4/json -> LeRobot (views 1/2/3)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

def _to_frame(img):
    import numpy as np

    return np.asarray(img, dtype=np.uint8)


def _convert_views1(stage1_dir: Path, repo: str, robot_type: str, task: str, target_fps: float) -> None:
    import imageio.v3 as iio
    import numpy as np
    import stage2_core
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    mp4_files = sorted(stage1_dir.glob("episode*.mp4"))
    if not mp4_files:
        raise FileNotFoundError(f"No episode*.mp4 found in {stage1_dir}")

    first_json = stage1_dir / f"{mp4_files[0].stem}.json"
    poses0, clamp0, fps0, _ = stage2_core.load_episode_json(first_json, default_fps=60.0, require_clamp=False)
    stride, achieved_fps = stage2_core.compute_stride(fps0, target_fps)
    fps_int = max(1, int(round(achieved_fps)))
    first_frame = _to_frame(iio.imread(mp4_files[0], index=0))
    h, w, _ = first_frame.shape
    use_clamp = clamp0 is not None

    out_path = HF_LEROBOT_HOME / repo
    if out_path.exists():
        shutil.rmtree(out_path)

    dataset = LeRobotDataset.create(
        repo_id=repo,
        robot_type=robot_type,
        fps=fps_int,
        features={
            "image": {"dtype": "image", "shape": (h, w, 3), "names": ["height", "width", "channel"]},
            "eef_pos": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
            "eef_rot_axis_angle": {"dtype": "float32", "shape": (3,), "names": ["rx", "ry", "rz"]},
            "gripper_width": {"dtype": "float32", "shape": (1,), "names": ["gripper_width"]},
            "demo_start_pose": {"dtype": "float32", "shape": (6,), "names": ["x", "y", "z", "rx", "ry", "rz"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["x", "y", "z", "rx", "ry", "rz", "gripper_width"]},
            "episode_start_idx": {"dtype": "int64", "shape": (1,), "names": ["episode_start_idx"]},
            "episode_end_idx": {"dtype": "int64", "shape": (1,), "names": ["episode_end_idx"]},
        },
        image_writer_threads=8,
        image_writer_processes=4,
    )

    for mp4_path in mp4_files:
        json_path = stage1_dir / f"{mp4_path.stem}.json"
        if not json_path.exists():
            continue
        poses, clamp, _, t = stage2_core.load_episode_json(json_path, default_fps=60.0, require_clamp=False)
        if use_clamp and clamp is None:
            raise ValueError(f"{json_path} has no clamp but dataset expects clamp.")

        ds_indices = np.arange(0, t, stride, dtype=np.int64)
        t_ds = int(ds_indices.shape[0])
        pos0, rot0 = stage2_core.pose7_to_pos_rotvec(poses[0])
        demo_start = np.concatenate([pos0, rot0], axis=0).astype(np.float32)
        ds_ptr = 0
        written = 0
        for i, frame in enumerate(iio.imiter(mp4_path)):
            if i >= t:
                break
            if ds_ptr >= t_ds or i != ds_indices[ds_ptr]:
                continue
            i_obs = int(ds_indices[ds_ptr])
            i_next = int(ds_indices[ds_ptr + 1]) if (ds_ptr + 1) < t_ds else i_obs
            pos, rot = stage2_core.pose7_to_pos_rotvec(poses[i_obs])
            pos_n, rot_n = stage2_core.pose7_to_pos_rotvec(poses[i_next])
            g = np.array([float(clamp[i_obs].item())], dtype=np.float32) if use_clamp else np.array([0.0], dtype=np.float32)
            g_n = np.array([float(clamp[i_next].item())], dtype=np.float32) if use_clamp else np.array([0.0], dtype=np.float32)
            action = np.concatenate([pos_n, rot_n, g_n], axis=0).astype(np.float32)
            dataset.add_frame(
                {
                    "image": _to_frame(frame),
                    "eef_pos": pos,
                    "eef_rot_axis_angle": rot,
                    "gripper_width": g,
                    "demo_start_pose": demo_start,
                    "actions": action,
                    "episode_start_idx": np.array([0], dtype=np.int64),
                    "episode_end_idx": np.array([t_ds], dtype=np.int64),
                    "task": task,
                }
            )
            written += 1
            ds_ptr += 1
        if written != t_ds:
            raise RuntimeError(f"{mp4_path.name}: written={written} != expected={t_ds}")
        dataset.save_episode()


def _convert_views2(stage1_dir: Path, repo: str, robot_type: str, task: str, target_fps: float) -> None:
    import imageio.v3 as iio
    import numpy as np
    import stage2_core
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    json_files = sorted(stage1_dir.glob("episode*.json"))
    if not json_files:
        raise FileNotFoundError(f"No episode*.json found in {stage1_dir}")
    first = json_files[0].stem
    frame_h = _to_frame(iio.imread(stage1_dir / f"{first}_head.mp4", index=0))
    frame_l = _to_frame(iio.imread(stage1_dir / f"{first}_left.mp4", index=0))
    h_h, w_h, _ = frame_h.shape
    h_l, w_l, _ = frame_l.shape
    _, clamp0, fps0, _ = stage2_core.load_episode_json(stage1_dir / f"{first}.json", default_fps=30.0, require_clamp=False)
    use_clamp = clamp0 is not None
    stride, achieved_fps = stage2_core.compute_stride(fps0, target_fps)
    fps_int = max(1, int(round(achieved_fps)))

    out_path = HF_LEROBOT_HOME / repo
    if out_path.exists():
        shutil.rmtree(out_path)
    dataset = LeRobotDataset.create(
        repo_id=repo,
        robot_type=robot_type,
        fps=fps_int,
        features={
            "head_view": {"dtype": "video", "shape": (h_h, w_h, 3), "names": ["height", "width", "channel"]},
            "left_view": {"dtype": "video", "shape": (h_l, w_l, 3), "names": ["height", "width", "channel"]},
            "eef_pos": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
            "eef_rot_axis_angle": {"dtype": "float32", "shape": (3,), "names": ["rx", "ry", "rz"]},
            "gripper_width": {"dtype": "float32", "shape": (1,), "names": ["gripper_width"]},
            "demo_start_pose": {"dtype": "float32", "shape": (6,), "names": ["x", "y", "z", "rx", "ry", "rz"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["x", "y", "z", "rx", "ry", "rz", "gripper_width"]},
            "episode_start_idx": {"dtype": "int64", "shape": (1,), "names": ["episode_start_idx"]},
            "episode_end_idx": {"dtype": "int64", "shape": (1,), "names": ["episode_end_idx"]},
        },
        image_writer_threads=8,
        image_writer_processes=4,
    )
    for json_path in json_files:
        stem = json_path.stem
        head_mp4 = stage1_dir / f"{stem}_head.mp4"
        left_mp4 = stage1_dir / f"{stem}_left.mp4"
        if not head_mp4.exists() or not left_mp4.exists():
            continue
        poses, clamp, _, t_json = stage2_core.load_episode_json(json_path, default_fps=30.0, require_clamp=False)
        ds_indices = np.arange(0, t_json, stride, dtype=np.int64)
        t_target = len(ds_indices)
        pos0, rot0 = stage2_core.pose7_to_pos_rotvec(poses[0])
        demo_start = np.concatenate([pos0, rot0], axis=0).astype(np.float32)
        ds_ptr = 0
        for i, (fh, fl) in enumerate(zip(iio.imiter(head_mp4), iio.imiter(left_mp4))):
            if i >= t_json:
                break
            if ds_ptr >= t_target or i != ds_indices[ds_ptr]:
                continue
            i_obs = int(ds_indices[ds_ptr])
            i_next = int(ds_indices[ds_ptr + 1]) if (ds_ptr + 1) < t_target else i_obs
            pos, rot = stage2_core.pose7_to_pos_rotvec(poses[i_obs])
            pos_n, rot_n = stage2_core.pose7_to_pos_rotvec(poses[i_next])
            g = np.array([float(clamp[i_obs].item())], dtype=np.float32) if use_clamp else np.array([0.0], dtype=np.float32)
            g_n = np.array([float(clamp[i_next].item())], dtype=np.float32) if use_clamp else np.array([0.0], dtype=np.float32)
            action = np.concatenate([pos_n, rot_n, g_n], axis=0).astype(np.float32)
            dataset.add_frame(
                {
                    "head_view": _to_frame(fh),
                    "left_view": _to_frame(fl),
                    "eef_pos": pos,
                    "eef_rot_axis_angle": rot,
                    "gripper_width": g,
                    "demo_start_pose": demo_start,
                    "actions": action,
                    "episode_start_idx": np.array([0], dtype=np.int64),
                    "episode_end_idx": np.array([t_target], dtype=np.int64),
                    "task": task,
                }
            )
            ds_ptr += 1
        dataset.save_episode()


def _convert_views3(stage1_dir: Path, repo: str, robot_type: str, task: str, fps_override: int) -> None:
    import imageio.v3 as iio
    import numpy as np
    import stage2_core
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    left_jsons = sorted(stage1_dir.glob("episode_*_left.json"))
    if not left_jsons:
        raise FileNotFoundError(f"No episode_*_left.json found in {stage1_dir}")
    stem = left_jsons[0].stem.replace("_left", "")
    fl0 = _to_frame(iio.imread(stage1_dir / f"{stem}_left.mp4", index=0))
    fr0 = _to_frame(iio.imread(stage1_dir / f"{stem}_right.mp4", index=0))
    ft0 = _to_frame(iio.imread(stage1_dir / f"{stem}_third.mp4", index=0))
    h_l, w_l, _ = fl0.shape
    h_r, w_r, _ = fr0.shape
    h_t, w_t, _ = ft0.shape
    _, _, fps0, _ = stage2_core.load_episode_json(left_jsons[0], default_fps=10.0, require_clamp=True)
    fps_ds = int(round(fps0)) if fps_override <= 0 else int(fps_override)

    out_path = HF_LEROBOT_HOME / repo
    if out_path.exists():
        shutil.rmtree(out_path)
    dataset = LeRobotDataset.create(
        repo_id=repo,
        robot_type=robot_type,
        fps=fps_ds,
        features={
            "left_view": {"dtype": "video", "shape": (h_l, w_l, 3), "names": ["h", "w", "c"]},
            "right_view": {"dtype": "video", "shape": (h_r, w_r, 3), "names": ["h", "w", "c"]},
            "third_view": {"dtype": "video", "shape": (h_t, w_t, 3), "names": ["h", "w", "c"]},
            "left_eef_pos": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
            "left_eef_rotvec": {"dtype": "float32", "shape": (3,), "names": ["rx", "ry", "rz"]},
            "left_gripper": {"dtype": "float32", "shape": (1,), "names": ["g"]},
            "right_eef_pos": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
            "right_eef_rotvec": {"dtype": "float32", "shape": (3,), "names": ["rx", "ry", "rz"]},
            "right_gripper": {"dtype": "float32", "shape": (1,), "names": ["g"]},
            "demo_start_pose_left": {"dtype": "float32", "shape": (6,), "names": ["x", "y", "z", "rx", "ry", "rz"]},
            "demo_start_pose_right": {"dtype": "float32", "shape": (6,), "names": ["x", "y", "z", "rx", "ry", "rz"]},
            "left_action": {"dtype": "float32", "shape": (7,), "names": ["x", "y", "z", "rx", "ry", "rz", "g"]},
            "right_action": {"dtype": "float32", "shape": (7,), "names": ["x", "y", "z", "rx", "ry", "rz", "g"]},
            "actions": {"dtype": "float32", "shape": (14,), "names": [
                "left_x", "left_y", "left_z", "left_rx", "left_ry", "left_rz", "left_g",
                "right_x", "right_y", "right_z", "right_rx", "right_ry", "right_rz", "right_g",
            ]},
        },
        image_writer_threads=8,
        image_writer_processes=4,
    )
    for left_json in left_jsons:
        stem = left_json.stem.replace("_left", "")
        right_json = stage1_dir / f"{stem}_right.json"
        left_mp4 = stage1_dir / f"{stem}_left.mp4"
        right_mp4 = stage1_dir / f"{stem}_right.mp4"
        third_mp4 = stage1_dir / f"{stem}_third.mp4"
        if not (right_json.exists() and left_mp4.exists() and right_mp4.exists() and third_mp4.exists()):
            continue
        poses_l, clamp_l, _, t_l = stage2_core.load_episode_json(left_json, default_fps=10.0, require_clamp=True)
        poses_r, clamp_r, _, t_r = stage2_core.load_episode_json(right_json, default_fps=10.0, require_clamp=True)
        t_json = min(t_l, t_r, len(clamp_l), len(clamp_r))
        pos_l0, rot_l0 = stage2_core.pose7_to_pos_rotvec(poses_l[0])
        pos_r0, rot_r0 = stage2_core.pose7_to_pos_rotvec(poses_r[0])
        demo_l = np.concatenate([pos_l0, rot_l0], axis=0).astype(np.float32)
        demo_r = np.concatenate([pos_r0, rot_r0], axis=0).astype(np.float32)
        for i, (fl, fr, ft) in enumerate(zip(iio.imiter(left_mp4), iio.imiter(right_mp4), iio.imiter(third_mp4))):
            if i >= t_json:
                break
            j = i + 1 if (i + 1) < t_json else i
            pos_l, rot_l = stage2_core.pose7_to_pos_rotvec(poses_l[i])
            pos_r, rot_r = stage2_core.pose7_to_pos_rotvec(poses_r[i])
            pos_l2, rot_l2 = stage2_core.pose7_to_pos_rotvec(poses_l[j])
            pos_r2, rot_r2 = stage2_core.pose7_to_pos_rotvec(poses_r[j])
            g_l = np.array([float(clamp_l[i].item())], dtype=np.float32)
            g_r = np.array([float(clamp_r[i].item())], dtype=np.float32)
            g_l2 = np.array([float(clamp_l[j].item())], dtype=np.float32)
            g_r2 = np.array([float(clamp_r[j].item())], dtype=np.float32)
            left_action = np.concatenate([pos_l2, rot_l2, g_l2], axis=0).astype(np.float32)
            right_action = np.concatenate([pos_r2, rot_r2, g_r2], axis=0).astype(np.float32)
            actions = np.concatenate([left_action, right_action], axis=0).astype(np.float32)
            dataset.add_frame(
                {
                    "left_view": _to_frame(fl),
                    "right_view": _to_frame(fr),
                    "third_view": _to_frame(ft),
                    "left_eef_pos": pos_l,
                    "left_eef_rotvec": rot_l,
                    "left_gripper": g_l,
                    "right_eef_pos": pos_r,
                    "right_eef_rotvec": rot_r,
                    "right_gripper": g_r,
                    "demo_start_pose_left": demo_l,
                    "demo_start_pose_right": demo_r,
                    "left_action": left_action,
                    "right_action": right_action,
                    "actions": actions,
                    "task": task,
                }
            )
        dataset.save_episode()


def run_stage2(
    *,
    views: int,
    stage1_dir: str,
    repo: str,
    robot_type: str | None,
    task: str,
    target_fps: float,
    fps: int,
) -> int:
    if views == 1:
        _convert_views1(Path(stage1_dir), repo, robot_type or "XV", task, target_fps)
        return 0

    if views == 2:
        _convert_views2(Path(stage1_dir), repo, robot_type or "XV", task, target_fps)
        return 0

    _convert_views3(Path(stage1_dir), repo, robot_type or "XV_DUAL", task, fps if fps > 0 else 0)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified stage2 converter launcher")
    parser.add_argument("--views", type=int, choices=[1, 2, 3], required=True, help="Number of camera views")
    parser.add_argument("--stage1_dir", required=True, help="Stage1 output directory")
    parser.add_argument("--repo", required=True, help="LeRobot repo id/name")
    parser.add_argument("--robot_type", default=None, help="Override robot_type")
    parser.add_argument("--task", default="put the purple cup into the plate", help="Task text")
    parser.add_argument("--target_fps", type=float, default=10.0, help="Target fps for views=1/2")
    parser.add_argument("--fps", type=int, default=0, help="Override dataset fps for views=3")
    args = parser.parse_args()

    return run_stage2(
        views=args.views,
        stage1_dir=args.stage1_dir,
        repo=args.repo,
        robot_type=args.robot_type,
        task=args.task,
        target_fps=args.target_fps,
        fps=args.fps,
    )


if __name__ == "__main__":
    raise SystemExit(main())
