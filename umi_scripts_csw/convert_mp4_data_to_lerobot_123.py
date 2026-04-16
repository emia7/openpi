#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import shutil
from pathlib import Path

import numpy as np
import imageio.v3 as iio

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
import stage2_core


def pose7_to_pos_rotvec(pose7: np.ndarray):
    return stage2_core.pose7_to_pos_rotvec(pose7)


def load_episode_json(path: Path):
    return stage2_core.load_episode_json(path, default_fps=10.0, require_clamp=True)


def main(stage1_dir: str, repo: str, robot_type: str, task: str, fps_override: int = 0):
    stage1_dir = Path(stage1_dir)
    if not stage1_dir.exists():
        raise FileNotFoundError(stage1_dir)

    left_jsons = sorted(stage1_dir.glob("episode_*_left.json"))
    if not left_jsons:
        raise FileNotFoundError(f"No episode_*_left.json found in {stage1_dir}")

    # inspect first episode
    first_left_json = left_jsons[0]
    stem = first_left_json.stem.replace("_left", "")  # episode_000001

    p_left = stage1_dir / f"{stem}_left.mp4"
    p_right = stage1_dir / f"{stem}_right.mp4"
    p_third = stage1_dir / f"{stem}_third.mp4"
    p_right_json = stage1_dir / f"{stem}_right.json"

    if not (p_left.exists() and p_right.exists() and p_third.exists() and p_right_json.exists()):
        raise FileNotFoundError(f"Missing files for first episode: {stem}")

    poses_l0, clamp_l0, fps0, _ = load_episode_json(first_left_json)

    # dataset fps
    fps_ds = int(round(fps0))
    if fps_override and fps_override > 0:
        fps_ds = int(fps_override)

    # image shapes
    frame_left0 = np.asarray(iio.imread(p_left, index=0), dtype=np.uint8)
    frame_right0 = np.asarray(iio.imread(p_right, index=0), dtype=np.uint8)
    frame_third0 = np.asarray(iio.imread(p_third, index=0), dtype=np.uint8)
    Hl, Wl, _ = frame_left0.shape
    Hr, Wr, _ = frame_right0.shape
    Ht, Wt, _ = frame_third0.shape

    out_path = HF_LEROBOT_HOME / repo
    if out_path.exists():
        print(f"[INFO] Removing existing repo at {out_path}")
        shutil.rmtree(out_path)

    # dataset
    dataset = LeRobotDataset.create(
        repo_id=repo,
        robot_type=robot_type,
        fps=fps_ds,
        features={
            "left_view":  {"dtype": "video", "shape": (Hl, Wl, 3), "names": ["h", "w", "c"]},
            "right_view": {"dtype": "video", "shape": (Hr, Wr, 3), "names": ["h", "w", "c"]},
            "third_view": {"dtype": "video", "shape": (Ht, Wt, 3), "names": ["h", "w", "c"]},

            "left_eef_pos": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
            "left_eef_rotvec": {"dtype": "float32", "shape": (3,), "names": ["rx", "ry", "rz"]},
            "left_gripper": {"dtype": "float32", "shape": (1,), "names": ["g"]},

            "right_eef_pos": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
            "right_eef_rotvec": {"dtype": "float32", "shape": (3,), "names": ["rx", "ry", "rz"]},
            "right_gripper": {"dtype": "float32", "shape": (1,), "names": ["g"]},

            "demo_start_pose_left": {"dtype": "float32", "shape": (6,), "names": ["x", "y", "z", "rx", "ry", "rz"]},
            "demo_start_pose_right": {"dtype": "float32", "shape": (6,), "names": ["x", "y", "z", "rx", "ry", "rz"]},

            # actions: next-step absolute pose+grip for each hand
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

    total_eps, total_frames = 0, 0

    for left_json in left_jsons:
        stem = left_json.stem.replace("_left", "")  # episode_000001
        right_json = stage1_dir / f"{stem}_right.json"
        left_mp4 = stage1_dir / f"{stem}_left.mp4"
        right_mp4 = stage1_dir / f"{stem}_right.mp4"
        third_mp4 = stage1_dir / f"{stem}_third.mp4"

        if not (right_json.exists() and left_mp4.exists() and right_mp4.exists() and third_mp4.exists()):
            print(f"[SKIP] Missing files for {stem}")
            continue

        # load states (pose7 is [x,y,z,qx,qy,qz,qw] in stage1 json)
        poses_l7, clamp_l, _, T_l = load_episode_json(left_json)
        poses_r7, clamp_r, _, T_r = load_episode_json(right_json)

        # demo start pose (6D) from first frame
        pos_l0, rot_l0 = pose7_to_pos_rotvec(poses_l7[0])
        pos_r0, rot_r0 = pose7_to_pos_rotvec(poses_r7[0])
        demo_left6 = np.concatenate([pos_l0, rot_l0], axis=0).astype(np.float32)
        demo_right6 = np.concatenate([pos_r0, rot_r0], axis=0).astype(np.float32)

        # iter videos
        it_l = iio.imiter(left_mp4)
        it_r = iio.imiter(right_mp4)
        it_t = iio.imiter(third_mp4)

        # truncate length by json (and zip truncation)
        T_json = min(T_l, T_r, len(clamp_l), len(clamp_r))
        written = 0

        try:
            for i, (fl, fr, ft) in enumerate(zip(it_l, it_r, it_t)):
                if i >= T_json:
                    break

                fl = np.asarray(fl, dtype=np.uint8)
                fr = np.asarray(fr, dtype=np.uint8)
                ft = np.asarray(ft, dtype=np.uint8)

                # obs pose7 -> obs pos+rotvec
                pos_l, rot_l = pose7_to_pos_rotvec(poses_l7[i])
                pos_r, rot_r = pose7_to_pos_rotvec(poses_r7[i])
                g_l = np.array([float(clamp_l[i].item())], dtype=np.float32)
                g_r = np.array([float(clamp_r[i].item())], dtype=np.float32)

                # action = next step absolute (pos+rotvec+g)
                j = i + 1 if (i + 1) < T_json else i
                pos_l2, rot_l2 = pose7_to_pos_rotvec(poses_l7[j])
                pos_r2, rot_r2 = pose7_to_pos_rotvec(poses_r7[j])
                g_l2 = np.array([float(clamp_l[j].item())], dtype=np.float32)
                g_r2 = np.array([float(clamp_r[j].item())], dtype=np.float32)

                left_action = np.concatenate([pos_l2, rot_l2, g_l2], axis=0).astype(np.float32)   # (7,)
                right_action = np.concatenate([pos_r2, rot_r2, g_r2], axis=0).astype(np.float32)  # (7,)

                actions = np.concatenate([left_action, right_action], axis=0).astype(np.float32)  # (14,)

                dataset.add_frame({
                    "left_view": fl,
                    "right_view": fr,
                    "third_view": ft,

                    "left_eef_pos": pos_l,
                    "left_eef_rotvec": rot_l,
                    "left_gripper": g_l,

                    "right_eef_pos": pos_r,
                    "right_eef_rotvec": rot_r,
                    "right_gripper": g_r,

                    "demo_start_pose_left": demo_left6,
                    "demo_start_pose_right": demo_right6,

                    "left_action": left_action,
                    "right_action": right_action,
                    "actions": actions,

                    "task": task,
                })
                written += 1

        except Exception as e:
            print(f"[WARN] {stem}: error at frame {written}: {e}. Truncating this episode.")

        dataset.save_episode()
        total_eps += 1
        total_frames += written
        print(f"[OK] {stem} -> frames={written}")

    print(f"\nDONE -> repo saved at: {out_path}")
    print(f"Total Episodes: {total_eps}, Total Frames: {total_frames}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_dir", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--robot_type", default="XV_DUAL")
    p.add_argument("--task", default="demo task")
    p.add_argument("--fps", type=int, default=0, help="override dataset fps (0=use json fps rounded)")
    args = p.parse_args()

    main(
        stage1_dir=args.stage1_dir,
        repo=args.repo,
        robot_type=args.robot_type,
        task=args.task,
        fps_override=args.fps,
    )
