#!/usr/bin/env python3
"""Unified relative-eval script for single/dual setups."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from openpi.policies import policy_config
from openpi.training import config as _config


class IdentityTransform:
    def __call__(self, data):
        return data


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _get_fps(ds, default=10.0):
    for path in [("meta", "info", "fps"), ("meta", "fps")]:
        cur = ds
        ok = True
        for k in path:
            try:
                cur = cur[k] if isinstance(cur, dict) else getattr(cur, k)
            except Exception:
                ok = False
                break
        if ok:
            try:
                return float(cur)
            except Exception:
                pass
    return float(default)


def _find_episode_indices_by_scan(ds, episode_index: int, length: int):
    ep_key_candidates = ["episode_index", "episode_id", "episode"]
    found_start = None
    found_key = None
    for i in range(len(ds)):
        s = ds[i]
        for k in ep_key_candidates:
            if k in s and int(_to_numpy(s[k])) == int(episode_index):
                found_start = i
                found_key = k
                break
        if found_start is not None:
            break
    if found_start is None:
        raise ValueError(f"Episode {episode_index} not found.")
    idxs = []
    for j in range(found_start, len(ds)):
        s = ds[j]
        if found_key not in s or int(_to_numpy(s[found_key])) != int(episode_index):
            break
        idxs.append(j)
        if len(idxs) >= length:
            break
    return idxs


def _build_single_raw(sample: dict):
    return {
        "image": _to_numpy(sample["image"]),
        "eef_pos": np.asarray(sample["eef_pos"], np.float32),
        "eef_rot_axis_angle": np.asarray(sample["eef_rot_axis_angle"], np.float32),
        "gripper_width": np.asarray(sample["gripper_width"], np.float32),
        "demo_start_pose": np.asarray(sample["demo_start_pose"], np.float32),
        "prompt": str(sample.get("task", "")),
    }


def _build_dual_raw(sample: dict):
    return {
        "third_view": _to_numpy(sample["third_view"]),
        "left_view": _to_numpy(sample["left_view"]),
        "right_view": _to_numpy(sample["right_view"]),
        "left_eef_pos": np.asarray(sample["left_eef_pos"], np.float32),
        "left_eef_rotvec": np.asarray(sample["left_eef_rotvec"], np.float32),
        "left_gripper": np.asarray(sample["left_gripper"], np.float32).reshape(1),
        "right_eef_pos": np.asarray(sample["right_eef_pos"], np.float32),
        "right_eef_rotvec": np.asarray(sample["right_eef_rotvec"], np.float32),
        "right_gripper": np.asarray(sample["right_gripper"], np.float32).reshape(1),
        "demo_start_pose_left": np.asarray(sample["demo_start_pose_left"], np.float32),
        "demo_start_pose_right": np.asarray(sample["demo_start_pose_right"], np.float32),
        "task": str(sample.get("task", "")),
    }


def _collect_gt_sequence(ds, idxs, k: int, horizon: int, mode: str):
    if mode == "dual":
        left_seq, right_seq = [], []
        for h in range(horizon):
            target_k = min(k + h, len(idxs) - 1)
            target_gi = idxs[target_k]
            left_seq.append(_to_numpy(ds[target_gi]["left_action"]).astype(np.float32))
            right_seq.append(_to_numpy(ds[target_gi]["right_action"]).astype(np.float32))
        return np.stack(left_seq), np.stack(right_seq)

    actions = []
    for h in range(horizon):
        target_k = min(k + h, len(idxs) - 1)
        target_gi = idxs[target_k]
        actions.append(_to_numpy(ds[target_gi]["actions"]).astype(np.float32))
    return np.stack(actions), None


def _plot_single_step(
    out_dir: Path,
    t: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    episode: int,
    step_idx: int,
    use_time_axis: bool,
):
    for d in range(pred.shape[1]):
        plt.figure(figsize=(10, 4))
        plt.plot(t, gt[:, d], label="GT", color="black", alpha=0.8)
        plt.plot(t, pred[:, d], label="Pred", color="red", alpha=0.8, linestyle="--")
        plt.title(f"Episode {episode} | Dim {d} | Step+{step_idx}")
        plt.xlabel("Time (s)" if use_time_axis else "Step")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"ep{episode:04d}_dim{d:02d}_step{step_idx:02d}.png", dpi=150)
        plt.close()


def main():
    ap = argparse.ArgumentParser(description="Unified relative eval for single/dual modes")
    ap.add_argument("--mode", choices=["single", "dual"], default="single")
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp-name", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--repo", required=True, help="LeRobot dataset root path")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--out-dir", default="eval_relative_results")
    ap.add_argument("--dims", type=int, default=20)
    ap.add_argument("--compare-steps", default="0", help="e.g. 0,1,2 or all")
    ap.add_argument("--max-plots-per-dim", type=int, default=4)
    ap.add_argument("--use-time-axis", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-base", default="/mnt/public1/chenshuaiwen/checkpoints")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = _config.get_config(args.config)
    train_cfg = dataclasses.replace(train_cfg, exp_name=args.exp_name)
    ckpt_dir = Path(args.ckpt_base) / args.config / args.exp_name / str(args.step)
    policy = policy_config.create_trained_policy(train_cfg, str(ckpt_dir))
    input_transform = policy._input_transform
    if input_transform is None:
        raise RuntimeError("No input transform found.")
    horizon = int(getattr(input_transform, "action_horizon", 16))
    policy._output_transform = IdentityTransform()

    if args.compare_steps.lower() == "all":
        compare_steps = list(range(horizon))
    else:
        compare_steps = [int(x.strip()) for x in args.compare_steps.split(",") if x.strip()]
        compare_steps = [s for s in compare_steps if 0 <= s < horizon]
        if not compare_steps:
            compare_steps = [0]

    ds = LeRobotDataset(repo_id="local_eval", root=args.repo)
    fps = _get_fps(ds, default=10.0)
    episodes = ds.meta["episodes"] if isinstance(ds.meta, dict) else ds.meta.episodes
    ep_meta = episodes[args.episode] if isinstance(episodes, dict) else episodes.iloc[args.episode].to_dict()
    length = int(ep_meta.get("length", 0))
    idxs = _find_episode_indices_by_scan(ds, args.episode, length)

    rng = jax.random.key(args.seed)
    t_list = []
    pred_by_step = {s: [] for s in compare_steps}
    gt_by_step = {s: [] for s in compare_steps}

    for k, gi in enumerate(idxs):
        sample = ds[gi]
        raw_data = _build_dual_raw(sample) if args.mode == "dual" else _build_single_raw(sample)

        gt_seq_a, gt_seq_b = _collect_gt_sequence(ds, idxs, k, horizon, args.mode)
        input_for_gt = raw_data.copy()
        if args.mode == "dual":
            input_for_gt["left_action"] = gt_seq_a
            input_for_gt["right_action"] = gt_seq_b
        else:
            input_for_gt["actions"] = gt_seq_a

        processed_gt = input_transform(input_for_gt)
        gt_chunk = _to_numpy(processed_gt["actions"]).astype(np.float32)
        rng, sub = jax.random.split(rng)
        try:
            out = policy.infer(raw_data, sub)
        except TypeError:
            out = policy.infer(raw_data)
        pred_chunk = _to_numpy(out["actions"]).astype(np.float32)
        if pred_chunk.ndim == 1:
            pred_chunk = pred_chunk[None, :]
        if gt_chunk.ndim == 1:
            gt_chunk = gt_chunk[None, :]

        t = (k / fps) if args.use_time_axis else k
        t_list.append(t)
        for step_idx in compare_steps:
            sidx = min(step_idx, pred_chunk.shape[0] - 1, gt_chunk.shape[0] - 1)
            gt_step = gt_chunk[sidx]
            pred_step = pred_chunk[sidx]
            dim = min(args.dims, pred_step.shape[0], gt_step.shape[0])
            pred_by_step[step_idx].append(pred_step[:dim])
            gt_by_step[step_idx].append(gt_step[:dim])

    t = np.array(t_list)
    save_dict = {"t": t, "compare_steps": np.array(compare_steps)}
    for step_idx in compare_steps:
        pred_arr = np.array(pred_by_step[step_idx])
        gt_arr = np.array(gt_by_step[step_idx])
        save_dict[f"pred_step{step_idx}"] = pred_arr
        save_dict[f"gt_step{step_idx}"] = gt_arr
        _plot_single_step(out_dir, t, pred_arr, gt_arr, args.episode, step_idx, args.use_time_axis)

    np.savez(out_dir / f"ep{args.episode:04d}_{args.mode}.npz", **save_dict)
    print(f"[OK] Saved eval outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
