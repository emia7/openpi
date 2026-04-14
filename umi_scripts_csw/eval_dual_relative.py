"""
双臂XV/UMI离线评估脚本
对比模型预测（Pred）vs 真实数据（GT）在训练集上的拟合效果

适用于: pi05_xv_dual_finetune 配置
数据格式: left_view, right_view, third_view + 左右手状态
"""

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import jax

from openpi.training import config as _config
from openpi.policies import policy_config
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


class IdentityTransform:
    """绕过output transform，直接比较模型输出"""
    def __call__(self, data):
        return data


def _to_numpy(x):
    """转换为numpy数组"""
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
    """从数据集获取FPS"""
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
    """
    扫描数据集找到指定episode的所有帧索引
    """
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
        if found_key not in s or int(_to_numpy(s[k])) != int(episode_index):
            break
        idxs.append(j)
        if len(idxs) >= length:
            break

    return idxs


def main():
    ap = argparse.ArgumentParser(description="双臂XV/UMI模型离线评估")
    ap.add_argument("--config", required=True, help="训练配置名 (如 pi05_xv_dual_finetune)")
    ap.add_argument("--exp-name", required=True, help="实验名称")
    ap.add_argument("--step", type=int, required=True, help="检查点步数")
    ap.add_argument("--repo", required=True, help="LeRobot数据集路径")
    ap.add_argument("--episode", type=int, default=0, help="要评估的episode索引")
    ap.add_argument("--out-dir", default="eval_dual_results", help="输出目录")
    ap.add_argument("--dims", type=int, default=20, help="评估维度数 (默认20=双臂各10维)")
    ap.add_argument("--use-time-axis", action="store_true", help="使用时间轴而非帧索引")
    ap.add_argument("--seed", type=int, default=0, help="随机种子")
    ap.add_argument("--ckpt-base", default="/share/chenshuaiwen-local/checkpoints",
                    help="检查点基础目录")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载训练配置和模型
    print(f"[INFO] Loading config: {args.config}")
    train_cfg = _config.get_config(args.config)
    train_cfg = dataclasses.replace(train_cfg, exp_name=args.exp_name)

    ckpt_dir = Path(args.ckpt_base) / args.config / args.exp_name / str(args.step)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir}")
    print(f"[INFO] Loading checkpoint: {ckpt_dir}")

    policy = policy_config.create_trained_policy(train_cfg, str(ckpt_dir))

    # 获取input transform和action horizon
    input_transform = policy._input_transform
    if input_transform is None:
        raise RuntimeError("No input transform found in policy.")

    try:
        horizon = input_transform.action_horizon
        print(f"[INFO] Detected action horizon: {horizon}")
    except AttributeError:
        horizon = 10
        print(f"[WARN] Could not detect horizon, using default: {horizon}")

    # 绕过output transform，直接比较模型原始输出
    policy._output_transform = IdentityTransform()
    print("[INFO] Output transform bypassed (IdentityTransform)")

    # 2. 加载数据集
    print(f"[INFO] Loading dataset: {args.repo}")
    ds = LeRobotDataset(repo_id="local_eval", root=args.repo)
    fps = _get_fps(ds, default=10.0)
    print(f"[INFO] Dataset FPS: {fps}")

    # 获取episode信息
    episodes = ds.meta["episodes"] if isinstance(ds.meta, dict) else ds.meta.episodes
    if isinstance(episodes, dict):
        ep_meta = episodes[args.episode]
    else:
        ep_meta = episodes.iloc[args.episode].to_dict()
    length = int(ep_meta.get("length", 0))
    print(f"[INFO] Episode {args.episode} length: {length}")

    idxs = _find_episode_indices_by_scan(ds, args.episode, length)
    print(f"[INFO] Found {len(idxs)} frames for episode {args.episode}")

    # 3. 推理循环
    rng = jax.random.key(args.seed)
    t_list, pred_list, gt_list = [], [], []

    print("[INFO] Starting inference with sliding window GT generation...")

    for k, gi in enumerate(idxs):
        sample = ds[gi]

        # --- A. 准备双臂输入数据 ---
        raw_data = {
            # 三个视角图像
            "third_view": _to_numpy(sample["third_view"]),
            "left_view": _to_numpy(sample["left_view"]),
            "right_view": _to_numpy(sample["right_view"]),
            # 左手状态
            "left_eef_pos": np.asarray(sample["left_eef_pos"], np.float32),
            "left_eef_rotvec": np.asarray(sample["left_eef_rotvec"], np.float32),
            "left_gripper": np.asarray(sample["left_gripper"], np.float32).reshape(1),
            # 右手状态
            "right_eef_pos": np.asarray(sample["right_eef_pos"], np.float32),
            "right_eef_rotvec": np.asarray(sample["right_eef_rotvec"], np.float32),
            "right_gripper": np.asarray(sample["right_gripper"], np.float32).reshape(1),
            # Demo起始位姿（用于计算相对坐标）
            "demo_start_pose_left": np.asarray(sample["demo_start_pose_left"], np.float32),
            "demo_start_pose_right": np.asarray(sample["demo_start_pose_right"], np.float32),
            # Prompt
            "task": str(sample.get("task", "")),
        }

        # --- B. 构建未来的Action序列 (Sliding Window) ---
        future_left_actions = []
        future_right_actions = []

        for h in range(horizon):
            target_k = k + h
            if target_k >= len(idxs):
                target_k = len(idxs) - 1  # padding with last frame

            target_gi = idxs[target_k]

            # 获取该帧的左右手action (各7D: pos3 + rotvec3 + gripper1)
            left_act = _to_numpy(ds[target_gi]["left_action"]).astype(np.float32)
            right_act = _to_numpy(ds[target_gi]["right_action"]).astype(np.float32)

            future_left_actions.append(left_act)
            future_right_actions.append(right_act)

        gt_left_seq = np.stack(future_left_actions)  # (H, 7)
        gt_right_seq = np.stack(future_right_actions)  # (H, 7)

        # --- C. 生成GT：通过Input Transform处理 ---
        input_for_gt = raw_data.copy()
        input_for_gt["left_action"] = gt_left_seq
        input_for_gt["right_action"] = gt_right_seq

        processed_gt = input_transform(input_for_gt)
        gt_chunk = _to_numpy(processed_gt["actions"])  # (H, 20)

        # --- D. 生成Pred：模型推理 ---
        rng, sub = jax.random.split(rng)
        try:
            out = policy.infer(raw_data, sub)
        except TypeError:
            out = policy.infer(raw_data)

        pred_chunk = _to_numpy(out["actions"]).astype(np.float32)  # (H, 20)

        # --- E. 选择要对比的时间步 (默认第0步=下一步) ---
        step_idx = 0

        gt_step = gt_chunk[step_idx]      # (20,)
        pred_step = pred_chunk[step_idx]  # (20,)

        eval_dim = min(args.dims, pred_step.shape[0], gt_step.shape[0])
        t = (k / fps) if args.use_time_axis else k

        t_list.append(t)
        pred_list.append(pred_step[:eval_dim])
        gt_list.append(gt_step[:eval_dim])

        if k % 10 == 0:
            print(f"Step {k}/{len(idxs)}...", end="\r")

    print(f"\n[INFO] Processed {len(idxs)} steps")

    # 4. 转换为数组并保存
    t = np.array(t_list)
    pred = np.array(pred_list)  # (T, dims)
    gt = np.array(gt_list)      # (T, dims)

    np.savez(out_dir / f"ep{args.episode:04d}_dual.npz", t=t, pred=pred, gt=gt)
    print(f"[OK] Saved data to: {out_dir / f'ep{args.episode:04d}_dual.npz'}")

    # 5. 绘制对比图
    dim_names = [
        # Left hand (10 dims)
        "L_pos_x", "L_pos_y", "L_pos_z",
        "L_rot6d_0", "L_rot6d_1", "L_rot6d_2", "L_rot6d_3", "L_rot6d_4", "L_rot6d_5",
        "L_gripper",
        # Right hand (10 dims)
        "R_pos_x", "R_pos_y", "R_pos_z",
        "R_rot6d_0", "R_rot6d_1", "R_rot6d_2", "R_rot6d_3", "R_rot6d_4", "R_rot6d_5",
        "R_gripper",
    ]

    for d in range(pred.shape[1]):
        plt.figure(figsize=(12, 4))
        plt.plot(t, gt[:, d], label="GT", color='black', alpha=0.8, linewidth=1.5)
        plt.plot(t, pred[:, d], label="Pred", color='red', alpha=0.8, linewidth=1.5, linestyle='--')

        title = dim_names[d] if d < len(dim_names) else f"Dim {d}"
        if d == 9:
            title += " (Left Gripper)"
        elif d == 19:
            title += " (Right Gripper)"

        plt.title(f"Episode {args.episode} | {title} (Step+{step_idx})\n{args.config}/{args.exp_name} step={args.step}")
        plt.xlabel("Time (s)" if args.use_time_axis else "Step")
        plt.ylabel("Action Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"ep{args.episode:04d}_dim{d:02d}_{title.replace(' ', '_')}.png", dpi=150)
        plt.close()

    print(f"[OK] Saved {pred.shape[1]} plots to: {out_dir.resolve()}")

    # 6. 计算并显示统计信息
    mse = np.mean((pred - gt) ** 2, axis=0)
    mae = np.mean(np.abs(pred - gt), axis=0)
    corr = np.array([np.corrcoef(pred[:, d], gt[:, d])[0, 1] for d in range(pred.shape[1])])

    print("\n" + "=" * 60)
    print("Evaluation Statistics (per dimension):")
    print("=" * 60)
    for d in range(pred.shape[1]):
        name = dim_names[d] if d < len(dim_names) else f"Dim {d}"
        print(f"{name:15s} | MSE: {mse[d]:.6f} | MAE: {mae[d]:.6f} | Corr: {corr[d]:.4f}")
    print("=" * 60)
    print(f"Overall MSE: {np.mean(mse):.6f} | Overall MAE: {np.mean(mae):.6f}")

    # 特别检查gripper维度
    if pred.shape[1] >= 20:
        left_grip_pred = pred[:, 9]
        left_grip_gt = gt[:, 9]
        right_grip_pred = pred[:, 19]
        right_grip_gt = gt[:, 19]

        print("\n[Gripper Check]")
        print(f"Left  Gripper - Pred range: [{left_grip_pred.min():.2f}, {left_grip_pred.max():.2f}] | GT range: [{left_grip_gt.min():.2f}, {left_grip_gt.max():.2f}]")
        print(f"Right Gripper - Pred range: [{right_grip_pred.min():.2f}, {right_grip_pred.max():.2f}] | GT range: [{right_grip_gt.min():.2f}, {right_grip_gt.max():.2f}]")


if __name__ == "__main__":
    main()
