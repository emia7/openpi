import argparse
import dataclasses
from pathlib import Path

import numpy as np
import jax
import imageio.v2 as imageio

from openpi.training import config as _config
from openpi.policies import policy_config
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    try:
        return x.detach().cpu().numpy()
    except Exception:
        return np.asarray(x)


def _get_fps(ds, default=10.0):
    for path in [("meta", "info", "fps"), ("meta", "fps")]:
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


def _pick_active_image_key(image_dict: dict, image_mask_dict: dict) -> str:
    """
    Pick which image stream is active according to image_mask.
    Prefer the key with mask True. If multiple True, pick the first in sorted order.
    Fallback: first key in image_dict.
    """
    true_keys = []
    for k, v in image_mask_dict.items():
        try:
            if bool(v):
                true_keys.append(k)
        except Exception:
            pass

    if len(true_keys) > 0:
        true_keys = sorted(true_keys)
        return true_keys[0]

    # fallback
    keys = sorted(list(image_dict.keys()))
    return keys[0]


def _ensure_uint8_hwc(img: np.ndarray) -> np.ndarray:
    """
    Your UMIRelInputs._parse_image already returns uint8 HWC for the base_image,
    and then it constructs inputs["image"][...]=base_image / zeros_like(base_image).
    So here we mainly enforce type/shape safety.
    """
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[0] == 3:  # CHW -> HWC (just in case)
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp-name", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--out-dir", default="eval_transformed")
    ap.add_argument("--seed", type=int, default=0)

    # always make mp4
    ap.add_argument("--mp4-name", default=None, help="Optional mp4 filename (default: epXXXX_transformed.mp4)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load Policy -> get input_transform
    train_cfg = _config.get_config(args.config)
    train_cfg = dataclasses.replace(train_cfg, exp_name=args.exp_name)
    ckpt_dir = Path("/share/public-local/checkpoints") / args.config / args.exp_name / str(args.step)
    print(f"[INFO] Loading policy: {ckpt_dir}")
    policy = policy_config.create_trained_policy(train_cfg, str(ckpt_dir))

    input_transform = policy._input_transform
    if input_transform is None:
        raise RuntimeError("No input transform found in policy.")

    try:
        horizon = int(input_transform.action_horizon)
        print(f"[INFO] Detected Action Horizon: {horizon}")
    except Exception:
        horizon = 16
        print(f"[WARN] Could not detect horizon, defaulting to {horizon}")

    # 2) Load Dataset
    print(f"[INFO] Loading dataset: {args.repo}")
    ds = LeRobotDataset(repo_id="local_eval", root=args.repo)
    fps = _get_fps(ds, default=10.0)

    episodes = ds.meta["episodes"] if isinstance(ds.meta, dict) else ds.meta.episodes
    if isinstance(episodes, dict):
        ep_meta = episodes[args.episode]
    else:
        ep_meta = episodes.iloc[args.episode].to_dict()
    length = int(ep_meta.get("length", 0))
    idxs = _find_episode_indices_by_scan(ds, args.episode, length)
    print(f"[INFO] Episode length: {len(idxs)}")

    # 3) Prepare MP4 writer (always)
    mp4_name = args.mp4_name or f"ep{args.episode:04d}_transformed.mp4"
    mp4_path = out_dir / mp4_name
    writer = imageio.get_writer(str(mp4_path), fps=fps)
    print(f"[INFO] Writing MP4: {mp4_path}")

    # 4) Iterate and collect transformed contents
    rng = jax.random.key(args.seed)

    # transformed recordings
    t_list = []
    prompt_list = []
    state_list = []       # (T, state_dim)
    actions_list = []     # (T, H, 32) if present, else None per step
    image_key_used = None

    print("[INFO] Loop: exporting transform outputs (state/actions) + mp4 frames...")

    for k, gi in enumerate(idxs):
        sample = ds[gi]

        # --- build raw_data that UMIRelInputs expects ---
        raw_data = {
            "image": _to_numpy(sample["image"]),
            "eef_pos": np.asarray(sample["eef_pos"], np.float32),
            "eef_rot_axis_angle": np.asarray(sample["eef_rot_axis_angle"], np.float32),
            "gripper_width": np.asarray(sample["gripper_width"], np.float32),
            "demo_start_pose": np.asarray(sample["demo_start_pose"], np.float32),
            "prompt": sample.get("task", "do the task"),
        }
        
        # --- build sliding window raw actions (H,7) ONLY to feed transform so it can produce padded (H,32) ---
        future_actions = []
        for h in range(horizon):
            target_k = k + h
            if target_k >= len(idxs):
                target_k = len(idxs) - 1
            target_gi = idxs[target_k]
            act = _to_numpy(ds[target_gi]["actions"]).astype(np.float32)
            future_actions.append(act)
        raw_data["actions"] = np.stack(future_actions, axis=0)  # (H,7)
        

        # --- apply input_transform (this is exactly what you asked for) ---
        transformed = input_transform(raw_data)

        # --- choose the active image stream and write to mp4 ---
        img_dict = transformed["image"]
        mask_dict = transformed["image_mask"]
        if image_key_used is None:
            image_key_used = _pick_active_image_key(img_dict, mask_dict)
            print(f"[INFO] Using image stream for mp4: {image_key_used}")

        frame = _ensure_uint8_hwc(img_dict[image_key_used])
        writer.append_data(frame)

        # --- record transformed fields ---
        t_list.append(k / fps)
        prompt_list.append(str(transformed.get("prompt", "")))

        state = _to_numpy(transformed.get("state"))
        if state is None:
            # In case some config doesn't output state
            state = np.zeros((0,), dtype=np.float32)
        state_list.append(np.asarray(state, dtype=np.float32))

        acts = transformed.get("actions", None)
        print(acts[0][21])
        if acts is None:
            # Some transforms might not include actions if not provided
            actions_list.append(None)
        else:
            acts = _to_numpy(acts).astype(np.float32)  # expected (H,32)
            actions_list.append(acts)

        if k % 10 == 0:
            print(f"Step {k}...", end="\r")

        # (optional) advance rng if you later want to also run policy.infer
        rng, _ = jax.random.split(rng)

    writer.close()
    print(f"\n[OK] MP4 saved: {mp4_path}")

    # stack arrays
    t = np.asarray(t_list, dtype=np.float32)

    # state: stack to (T, D)
    # (UMIRelInputs uses 10D state: 9D pose + gripper)
    state_arr = np.stack(state_list, axis=0).astype(np.float32)

    # actions: if any None exists, we will not stack; but in your use-case it should always exist
    if any(a is None for a in actions_list):
        # Save as object array (rare)
        actions_arr = np.array(actions_list, dtype=object)
    else:
        actions_arr = np.stack(actions_list, axis=0).astype(np.float32)  # (T,H,32)

    # Save ONLY transformed content
    npz_path = out_dir / f"ep{args.episode:04d}_transformed_inputs.npz"
    np.savez(
        npz_path,
        t=t,
        state=state_arr,
        actions=actions_arr,
        prompt=np.array(prompt_list, dtype=object),
        image_key=np.array([image_key_used], dtype=object),
    )

    print(f"[OK] Saved transformed inputs to: {npz_path.resolve()}")
    print(f"[INFO] state shape: {state_arr.shape}")
    if isinstance(actions_arr, np.ndarray) and actions_arr.dtype != object:
        print(f"[INFO] actions shape: {actions_arr.shape}  (expected: T x H x 32)")
    else:
        print("[WARN] actions saved as object array (some steps missing actions).")


if __name__ == "__main__":
    main()
