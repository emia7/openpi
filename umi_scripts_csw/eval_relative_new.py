import argparse
import dataclasses
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import jax

from openpi.training import config as _config
from openpi.policies import policy_config
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# --- Hack Class ---
class IdentityTransform:
    def __call__(self, data):
        return data

def _to_numpy(x):
    if isinstance(x, np.ndarray): return x
    try: return x.detach().cpu().numpy()
    except: return np.asarray(x)

def _get_fps(ds, default=10.0):
    for path in [("meta", "info", "fps"), ("meta", "fps")]:
        cur = ds
        ok = True
        for k in path:
            try:
                if isinstance(cur, dict): cur = cur[k]
                else: cur = getattr(cur, k)
            except Exception:
                ok = False; break
        if ok:
            try: return float(cur)
            except: pass
    return float(default)

def _find_episode_indices_by_scan(ds, episode_index: int, length: int):
    ep_key_candidates = ["episode_index", "episode_id", "episode"]
    found_start = None; found_key = None
    
    for i in range(len(ds)):
        s = ds[i]
        for k in ep_key_candidates:
            if k in s and int(_to_numpy(s[k])) == int(episode_index):
                found_start = i; found_key = k; break
        if found_start is not None: break
            
    if found_start is None:
        raise ValueError(f"Episode {episode_index} not found.")

    idxs = []
    for j in range(found_start, len(ds)):
        s = ds[j]
        if found_key not in s or int(_to_numpy(s[found_key])) != int(episode_index):
            break
        idxs.append(j)
        if len(idxs) >= length: break
            
    return idxs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp-name", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--out-dir", default="eval_results_horizon")
    ap.add_argument("--dims", type=int, default=11)
    ap.add_argument("--use-time-axis", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Policy
    train_cfg = _config.get_config(args.config)
    train_cfg = dataclasses.replace(train_cfg, exp_name=args.exp_name)
    ckpt_dir = Path("/share/chenshuaiwen-local/checkpoints") / args.config / args.exp_name / str(args.step)
    
    print(f"[INFO] Loading policy: {ckpt_dir}")
    policy = policy_config.create_trained_policy(train_cfg, str(ckpt_dir))
    
    # 获取 Input Transform
    input_transform = policy._input_transform
    if input_transform is None:
        raise RuntimeError("No input transform found.")
        
    # [CRITICAL] 获取 Horizon 长度
    # 你的 XVInputs 类里有 action_horizon 属性
    try:
        horizon = input_transform.action_horizon
        print(f"[INFO] Detected Action Horizon: {horizon}")
    except AttributeError:
        horizon = 16 # Fallback if not found
        print(f"[WARN] Could not detect horizon, defaulting to {horizon}")

    # Hack Output Transform
    policy._output_transform = IdentityTransform()

    # 2. Load Dataset
    print(f"[INFO] Loading dataset: {args.repo}")
    ds = LeRobotDataset(repo_id="local_eval", root=args.repo)
    fps = _get_fps(ds, default=10.0)

    # Get Episode Indices
    episodes = ds.meta["episodes"] if isinstance(ds.meta, dict) else ds.meta.episodes
    if isinstance(episodes, dict): ep_meta = episodes[args.episode]
    else: ep_meta = episodes.iloc[args.episode].to_dict()
    length = int(ep_meta.get("length", 0))
    
    idxs = _find_episode_indices_by_scan(ds, args.episode, length)
    print(f"[INFO] Episode length: {len(idxs)}")

    # 3. Inference Loop
    rng = jax.random.key(args.seed)
    t_list, pred_list, gt_list = [], [], []

    print("[INFO] Starting Loop with Sliding Window GT generation...")

    for k, gi in enumerate(idxs):
        sample = ds[gi]
        
        # --- A. 基础 Input ---
        raw_data = {
            "image": _to_numpy(sample["image"]),
            "eef_pos": np.asarray(sample["eef_pos"], np.float32),
            "eef_rot_axis_angle": np.asarray(sample["eef_rot_axis_angle"], np.float32),
            "gripper_width": np.asarray(sample["gripper_width"], np.float32),
            "demo_start_pose": np.asarray(sample["demo_start_pose"], np.float32),
            "prompt": str(sample.get("task", "")),
        }

        # --- B. 构建未来的 Action Sequence (Sliding Window) ---
        # 我们需要从 dataset 里提取 [gi, gi+1, ..., gi+H-1] 的 actions
        future_actions = []
        for h in range(horizon):
            # 计算目标帧在当前 episode 中的索引 (k + h)
            target_k = k + h
            
            # 边界处理：如果超出当前 episode 长度，就取最后一帧 (Padding)
            if target_k >= len(idxs):
                target_k = len(idxs) - 1
            
            # 获取对应的全局索引
            target_gi = idxs[target_k]
            
            # 读取该帧动作
            act = _to_numpy(ds[target_gi]["actions"]).astype(np.float32)
            future_actions.append(act)
        
        # 堆叠成 (Horizon, 7) 的数组
        gt_actions_seq = np.stack(future_actions) # Shape: (H, 7)
        
        # --- C. 生成 GT ---
        input_for_gt = raw_data.copy()
        input_for_gt["actions"] = gt_actions_seq # 传入序列！
        
        # XVInputs 现在会看到 (H, 7) 的数据，不仅能正确计算 Relative，还能计算未来的轨迹
        processed_gt = input_transform(input_for_gt)
        gt_chunk = _to_numpy(processed_gt["actions"]) # (H, 11)

        # --- D. 生成 Pred ---
        # 这里的 input 不需要 actions 字段（或者给 dummy），因为 infer 不看它
        rng, sub = jax.random.split(rng)
        try:
            out = policy.infer(raw_data, sub)
        except TypeError:
            out = policy.infer(raw_data)
        
        pred_chunk = _to_numpy(out["actions"]).astype(np.float32) # (H, 11)

        # --- E. 选择要绘图的数据 ---
        # 既然我们有了 Horizon，我们可以选择画第几步？
        # 通常画 chunk[0] 代表“下一步预测”
        # 如果你想看更明显的动作趋势，可以试着画 chunk[5] 甚至 chunk[-1]
        
        step_idx = 0
        
        gt_step = gt_chunk[step_idx]
        pred_step = pred_chunk[step_idx]

        eval_dim = min(args.dims, pred_step.shape[0], gt_step.shape[0])
        t = (k / fps) if args.use_time_axis else k
        t_list.append(t)
        pred_list.append(pred_step[:eval_dim])
        gt_list.append(gt_step[:eval_dim])
        
        if k % 10 == 0: print(f"Step {k}...", end="\r")

    t = np.array(t_list)
    pred = np.array(pred_list)
    gt = np.array(gt_list)

    # 4. Plot
    for d in range(pred.shape[1]):
        plt.figure(figsize=(10, 4))
        plt.plot(t, gt[:, d], label="GT (Rel Seq)", color='black', alpha=0.8)
        plt.plot(t, pred[:, d], label="Pred (Raw)", color='red', alpha=0.8, linestyle='--')
        
        title = f"Dim {d}"
        if d == 9: title += " (Scale/Rot?)"
        if d == 10: title += " (Gripper)"
            
        plt.title(f"Episode {args.episode} | {title} (Step+{step_idx})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"ep{args.episode:04d}_dim{d:02d}.png")
        plt.close()

    np.savez(out_dir / f"ep{args.episode:04d}_horizon.npz", t=t, pred=pred, gt=gt)
    print(f"[OK] Saved to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
