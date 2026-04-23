import argparse
import dataclasses
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import jax

from openpi.training import config as _config
from openpi.policies import policy_config
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


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


def _get_fps(ds, default=60.0):
    # robust fps extraction across versions
    for path in [
        ("meta", "info", "fps"),
        ("meta", "fps"),
    ]:
        cur = ds
        ok = True
        for k in path:
            try:
                if isinstance(cur, dict):
                    cur = cur[k]
                else:
                    cur = getattr(cur, k)
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
    """
    Scan dataset to find the first occurrence of episode_index, then return a list of global indices
    of length `length`. Requires samples to contain episode_index/episode_id.
    """
    ep_key_candidates = ["episode_index", "episode_id", "episode"]
    found_start = None
    found_key = None

    # Scan until we find the episode
    for i in range(len(ds)):
        s = ds[i]
        for k in ep_key_candidates:
            if k in s:
                if int(_to_numpy(s[k])) == int(episode_index):
                    found_start = i
                    found_key = k
                    break
        if found_start is not None:
            break

    if found_start is None:
        raise ValueError(
            f"Could not find episode_index={episode_index} by scanning dataset. "
            f"Try printing ds[0].keys() to see episode id field."
        )

    # Collect consecutive indices that belong to this episode, until we have `length`
    idxs = []
    for j in range(found_start, len(ds)):
        s = ds[j]
        if found_key not in s or int(_to_numpy(s[found_key])) != int(episode_index):
            break
        idxs.append(j)
        if len(idxs) >= length:
            break

    if len(idxs) < length:
        raise ValueError(
            f"Episode {episode_index} expected length={length}, but only found {len(idxs)} consecutive steps "
            f"starting at global index {found_start}. (Maybe episode is not stored contiguously.)"
        )

    return idxs, found_key, found_start





def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp-name", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--out-dir", default="eval_results")
    ap.add_argument("--dims", type=int, default=8)
    ap.add_argument("--use-time-axis", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-scan", type=int, default=None, help="optional: cap scanning length for debugging")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) train config
    train_cfg = _config.get_config(args.config)
    train_cfg = dataclasses.replace(train_cfg, exp_name=args.exp_name)

    # 2) policy
    ckpt_dir = Path("/share/chenshuaiwen-local/checkpoints") / args.config / args.exp_name / str(args.step)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir.resolve()}")
    policy = policy_config.create_trained_policy(train_cfg, str(ckpt_dir))

    # 3) dataset
    ds = LeRobotDataset(args.repo)
    fps = _get_fps(ds, default=60.0)

    # meta episode length
    episodes = ds.meta["episodes"] if isinstance(ds.meta, dict) else ds.meta.episodes
    if isinstance(episodes, dict):
        ep_meta = episodes[args.episode]
    else:
        ep_meta = episodes.iloc[args.episode].to_dict()

    length = int(ep_meta.get("length", 0))
    if length <= 0:
        raise ValueError(f"Episode meta has no valid length: {ep_meta}")

    print(f"[INFO] episode={args.episode}, length={length}, fps={fps}")

    # 4) find global indices for this episode by scanning
    idxs, ep_key, start = _find_episode_indices_by_scan(ds, args.episode, length)
    print(f"[INFO] found episode key='{ep_key}', start_global_index={start}, idxs_len={len(idxs)}")

    # 5) step-by-step inference
    rng = jax.random.key(args.seed)
    t_list, pred_list, gt_list = [], [], []

    for k, gi in enumerate(idxs):
        sample = ds[gi]

        wrist_view = _to_numpy(sample["wrist_view"])
        state = _to_numpy(sample["state"]).astype(np.float32)
        gt_action = _to_numpy(sample["actions"]).astype(np.float32)[: args.dims]

        prompt = sample.get("prompt", None)
        if prompt is None:
            prompt = sample.get("task", "")
        if prompt is None:
            prompt = ""

        example = {
            "wrist_view": wrist_view,
            "state": state,
            "prompt": prompt,
        }

        rng, sub = jax.random.split(rng)
        try:
            out = policy.infer(example, sub)   # rng 作为位置参数
        except TypeError:
            out = policy.infer(example)        # infer 不需要 rng

        chunk = _to_numpy(out["actions"]).astype(np.float32)

        pred_action = chunk[0, : args.dims]

        t = (k / fps) if args.use_time_axis else k
        t_list.append(t)
        pred_list.append(pred_action)
        gt_list.append(gt_action)

    t = np.asarray(t_list)
    pred = np.asarray(pred_list)  # (T,dims)
    gt = np.asarray(gt_list)      # (T,dims)

    # 6) plots
    for d in range(args.dims):
        plt.figure(figsize=(12, 4))
        plt.plot(t, gt[:, d], label="GT")
        plt.plot(t, pred[:, d], label="Pred")
        plt.xlabel("time (s)" if args.use_time_axis else "step")
        plt.ylabel(f"action[{d}]")
        plt.title(f"episode={args.episode} dim={d} | {args.config}/{args.exp_name} step={args.step}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"episode{args.episode:04d}_dim{d:02d}.png", dpi=200)
        plt.close()

    np.savez(out_dir / f"episode{args.episode:04d}_pred_gt.npz", t=t, pred=pred, gt=gt)
    print(f"[OK] Saved plots + npz to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
