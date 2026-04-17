#!/usr/bin/env python3
"""Unified eval entrypoint for relative + transform export."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import jax
import numpy as np
import matplotlib.pyplot as plt
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


def _pick_active_image_key(image_dict: dict, image_mask_dict: dict) -> str:
    true_keys = []
    for k, v in image_mask_dict.items():
        try:
            if bool(v):
                true_keys.append(k)
        except Exception:
            pass
    if true_keys:
        return sorted(true_keys)[0]
    return sorted(list(image_dict.keys()))[0]


def _ensure_uint8_hwc(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _run_transform_export(args, ds, idxs, input_transform, horizon, fps, out_dir):
    import imageio.v2 as imageio

    mp4_name = args.mp4_name or f"ep{args.episode:04d}_transformed.mp4"
    mp4_path = out_dir / mp4_name
    writer = imageio.get_writer(str(mp4_path), fps=fps)

    t_list, prompt_list, state_list, actions_list = [], [], [], []
    image_key_used = None
    for k, gi in enumerate(idxs):
        sample = ds[gi]
        raw_data = _build_single_raw(sample)
        gt_seq, _ = _collect_gt_sequence(ds, idxs, k, horizon, "single")
        raw_data["actions"] = gt_seq
        transformed = input_transform(raw_data)

        img_dict = transformed["image"]
        mask_dict = transformed["image_mask"]
        if image_key_used is None:
            image_key_used = _pick_active_image_key(img_dict, mask_dict)
        frame = _ensure_uint8_hwc(img_dict[image_key_used])
        writer.append_data(frame)

        t_list.append(k / fps)
        prompt_list.append(str(transformed.get("prompt", "")))
        state = transformed.get("state", np.zeros((0,), dtype=np.float32))
        state_list.append(np.asarray(_to_numpy(state), dtype=np.float32))
        acts = transformed.get("actions")
        actions_list.append(None if acts is None else _to_numpy(acts).astype(np.float32))

    writer.close()
    t = np.asarray(t_list, dtype=np.float32)
    state_arr = np.stack(state_list, axis=0).astype(np.float32)
    actions_arr = (
        np.array(actions_list, dtype=object)
        if any(a is None for a in actions_list)
        else np.stack(actions_list, axis=0).astype(np.float32)
    )
    np.savez(
        out_dir / f"ep{args.episode:04d}_transformed_inputs.npz",
        t=t,
        state=state_arr,
        actions=actions_arr,
        prompt=np.array(prompt_list, dtype=object),
        image_key=np.array([image_key_used], dtype=object),
    )
    print(f"[OK] Saved transformed mp4+npz to: {out_dir.resolve()}")


def _run_relative_eval(args, ds, idxs, policy, input_transform, horizon, fps, out_dir):
    if args.compare_steps.lower() == "all":
        compare_steps = list(range(horizon))
    else:
        compare_steps = [int(x.strip()) for x in args.compare_steps.split(",") if x.strip()]
        compare_steps = [s for s in compare_steps if 0 <= s < horizon]
        if not compare_steps:
            compare_steps = [0]

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

        gt_chunk = _to_numpy(input_transform(input_for_gt)["actions"]).astype(np.float32)
        rng, sub = jax.random.split(rng)
        try:
            pred_chunk = _to_numpy(policy.infer(raw_data, sub)["actions"]).astype(np.float32)
        except TypeError:
            pred_chunk = _to_numpy(policy.infer(raw_data)["actions"]).astype(np.float32)
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

    npz_path = out_dir / f"ep{args.episode:04d}_{args.mode}.npz"
    np.savez(npz_path, **save_dict)
    print(f"[OK] Saved relative eval outputs to: {npz_path}")


def main():
    ap = argparse.ArgumentParser(description="Unified eval entrypoint")
    ap.add_argument("--mode", choices=["single", "dual", "transform"], default="single")
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
    ap.add_argument("--mp4-name", default=None, help="transform mode only")
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

    ds = LeRobotDataset(repo_id="local_eval", root=args.repo)
    fps = _get_fps(ds, default=10.0)
    episodes = ds.meta["episodes"] if isinstance(ds.meta, dict) else ds.meta.episodes
    ep_meta = episodes[args.episode] if isinstance(episodes, dict) else episodes.iloc[args.episode].to_dict()
    length = int(ep_meta.get("length", 0))
    idxs = _find_episode_indices_by_scan(ds, args.episode, length)

    if args.mode == "transform":
        _run_transform_export(args, ds, idxs, input_transform, horizon, fps, out_dir)
    else:
        _run_relative_eval(args, ds, idxs, policy, input_transform, horizon, fps, out_dir)


if __name__ == "__main__":
    main()
