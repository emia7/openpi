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
    ap.add_argument("--ckpt-base", default="/mnt/public1/chenshuaiwen/checkpoints",
                    help="检查点基础目录")
    ap.add_argument("--compare-steps", type=str, default="0",
                    help="要对比的时间步，逗号分隔，如'0,1,2,3'或'all'表示全部")
    ap.add_argument("--max-plots-per-dim", type=int, default=4,
                    help="汇总图中每个维度最多显示几个时间步")
    args = ap.parse_args()

    # 解析compare-steps参数
    if args.compare_steps.lower() == "all":
        # 暂时使用占位符，等知道horizon后再解析
        compare_steps = None
    else:
        try:
            compare_steps = [int(x.strip()) for x in args.compare_steps.split(",")]
            if not compare_steps:
                raise ValueError("Empty compare-steps")
        except Exception as e:
            print(f"[ERROR] Invalid compare-steps format: {args.compare_steps}")
            print(f"        Use comma-separated integers like '0,1,2,3' or 'all'")
            raise

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

    # 如果compare_steps是"all"，现在生成所有时间步
    if compare_steps is None:
        compare_steps = list(range(horizon))
        print(f"[INFO] Comparing all {horizon} steps: {compare_steps}")
    else:
        # 验证时间步是否有效
        valid_steps = [s for s in compare_steps if 0 <= s < horizon]
        if len(valid_steps) != len(compare_steps):
            invalid = set(compare_steps) - set(valid_steps)
            print(f"[WARN] Ignoring invalid steps {invalid} (horizon={horizon})")
            compare_steps = valid_steps
        print(f"[INFO] Comparing steps: {compare_steps}")

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
    
    # 为每个时间步初始化存储列表
    t_list = []
    pred_by_step = {s: [] for s in compare_steps}
    gt_by_step = {s: [] for s in compare_steps}

    print("[INFO] Starting inference with sliding window GT generation...")
    print(f"[INFO] Will collect data for steps: {compare_steps}")

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
        
        # Debug: 检查pred_chunk的形状和内容
        if k == 0:  # 只在第一步打印调试信息
            print(f"\n[DEBUG] Step {k}:")
            print(f"  pred_chunk shape: {pred_chunk.shape}")
            print(f"  pred_chunk dtype: {pred_chunk.dtype}")
            print(f"  gt_chunk shape: {gt_chunk.shape}")
            if len(pred_chunk.shape) == 1:
                print(f"  WARNING: pred_chunk is 1D! Expected 2D (H, 20)")
                print(f"  This means model only outputs single step, not chunk.")
            elif len(pred_chunk.shape) == 2:
                H, D = pred_chunk.shape
                print(f"  pred_chunk[0]: {pred_chunk[0][:5]}...")  # 前5个值
                if H > 1:
                    print(f"  pred_chunk[1]: {pred_chunk[1][:5]}...")
                    print(f"  pred_chunk[{H-1}]: {pred_chunk[H-1][:5]}...")
                    # 检查所有行是否相同
                    if np.allclose(pred_chunk[0], pred_chunk[min(1,H-1)]):
                        print(f"  WARNING: All rows in pred_chunk are identical!")
                        print(f"  Model may not be predicting different steps.")

        # --- E. 收集所有要对比的时间步 ---
        t = (k / fps) if args.use_time_axis else k
        t_list.append(t)

        for step_idx in compare_steps:
            gt_step = gt_chunk[step_idx]      # (20,)
            pred_step = pred_chunk[step_idx]  # (20,)
            
            eval_dim = min(args.dims, pred_step.shape[0], gt_step.shape[0])
            pred_by_step[step_idx].append(pred_step[:eval_dim])
            gt_by_step[step_idx].append(gt_step[:eval_dim])

        if k % 10 == 0:
            print(f"Step {k}/{len(idxs)}...", end="\r")

    print(f"\n[INFO] Processed {len(idxs)} steps")

    # 4. 转换为数组并保存
    t = np.array(t_list)  # (T,)
    
    # 转换每个时间步的数据为数组
    pred_arrays = {}
    gt_arrays = {}
    for step_idx in compare_steps:
        pred_arrays[step_idx] = np.array(pred_by_step[step_idx])  # (T, dims)
        gt_arrays[step_idx] = np.array(gt_by_step[step_idx])      # (T, dims)
    
    # 获取第一个时间步的维度数（所有时间步维度相同）
    first_step = compare_steps[0]
    pred = pred_arrays[first_step]  # (T, dims) - 用于后续兼容性
    gt = gt_arrays[first_step]        # (T, dims)

    # 构建保存字典，包含所有时间步的数据
    save_dict = {"t": t, "compare_steps": np.array(compare_steps)}
    for step_idx in compare_steps:
        save_dict[f"pred_step{step_idx}"] = pred_arrays[step_idx]
        save_dict[f"gt_step{step_idx}"] = gt_arrays[step_idx]
    
    np.savez(out_dir / f"ep{args.episode:04d}_dual.npz", **save_dict)
    print(f"[OK] Saved data to: {out_dir / f'ep{args.episode:04d}_dual.npz'}")
    print(f"[INFO] Saved data for steps: {compare_steps}")

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

    # 5A. 为每个时间步生成单独的对比图
    print(f"[INFO] Generating individual comparison plots for each step...")
    total_plots = 0
    for step_idx in compare_steps:
        pred_step = pred_arrays[step_idx]
        gt_step = gt_arrays[step_idx]
        
        for d in range(pred_step.shape[1]):
            plt.figure(figsize=(12, 4))
            plt.plot(t, gt_step[:, d], label="GT", color='black', alpha=0.8, linewidth=1.5)
            plt.plot(t, pred_step[:, d], label="Pred", color='red', alpha=0.8, linewidth=1.5, linestyle='--')

            title = dim_names[d] if d < len(dim_names) else f"Dim {d}"
            if d == 9:
                title += " (Left Gripper)"
            elif d == 19:
                title += " (Right Gripper)"

            plt.title(f"Episode {args.episode} | {title} | Step+{step_idx}\n{args.config}/{args.exp_name} step={args.step}")
            plt.xlabel("Time (s)" if args.use_time_axis else "Step")
            plt.ylabel("Action Value")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / f"ep{args.episode:04d}_dim{d:02d}_step{step_idx:02d}.png", dpi=150)
            plt.close()
            total_plots += 1
        
        print(f"  Step+{step_idx}: {pred_step.shape[1]} plots generated")

    print(f"[OK] Saved {total_plots} individual plots to: {out_dir.resolve()}")

    # 5B. 为每个维度生成时间步汇总对比图（subplot）
    print(f"[INFO] Generating multi-step summary plots...")
    n_steps = len(compare_steps)
    n_dims = pred.shape[1]
    
    for d in range(n_dims):
        title = dim_names[d] if d < len(dim_names) else f"Dim {d}"
        
        # 创建子图：每个时间步一个子图
        fig, axes = plt.subplots(n_steps, 1, figsize=(14, 3 * n_steps), sharex=True)
        if n_steps == 1:
            axes = [axes]  # 统一处理为列表
        
        for i, step_idx in enumerate(compare_steps):
            ax = axes[i]
            gt_step = gt_arrays[step_idx]
            pred_step = pred_arrays[step_idx]
            
            ax.plot(t, gt_step[:, d], label="GT", color='black', alpha=0.8, linewidth=1.5)
            ax.plot(t, pred_step[:, d], label="Pred", color='red', alpha=0.8, linewidth=1.5, linestyle='--')
            ax.set_ylabel(f"Step+{step_idx}")
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # 只在第一个子图添加标题
            if i == 0:
                ax.set_title(f"Episode {args.episode} | {title} | Multi-Step Comparison\n{args.config}/{args.exp_name} step={args.step}")
        
        # 最后一个子图添加xlabel
        axes[-1].set_xlabel("Time (s)" if args.use_time_axis else "Step")
        
        plt.tight_layout()
        plt.savefig(out_dir / f"ep{args.episode:04d}_dim{d:02d}_all_steps.png", dpi=150)
        plt.close()
    
    print(f"[OK] Saved {n_dims} multi-step summary plots")

    # 6. 计算并显示统计信息（每个时间步）
    print("\n" + "=" * 80)
    print("Evaluation Statistics by Prediction Step:")
    print("=" * 80)
    
    # 为每个时间步计算统计信息
    stats_by_step = {}
    for step_idx in compare_steps:
        pred_step = pred_arrays[step_idx]
        gt_step = gt_arrays[step_idx]
        
        mse_step = np.mean((pred_step - gt_step) ** 2, axis=0)
        mae_step = np.mean(np.abs(pred_step - gt_step), axis=0)
        corr_step = np.array([np.corrcoef(pred_step[:, d], gt_step[:, d])[0, 1] 
                              for d in range(pred_step.shape[1])])
        
        stats_by_step[step_idx] = {
            "mse": mse_step,
            "mae": mae_step,
            "corr": corr_step,
            "overall_mse": np.mean(mse_step),
            "overall_mae": np.mean(mae_step),
        }
        
        print(f"\nStep+{step_idx}: Overall MSE={stats_by_step[step_idx]['overall_mse']:.6f} | "
              f"MAE={stats_by_step[step_idx]['overall_mae']:.6f}")
    
    print("\n" + "=" * 80)
    print("Per-Dimension Statistics (averaged across all steps):")
    print("=" * 80)
    
    # 计算跨时间步的平均统计
    avg_mse = np.mean([stats_by_step[s]["mse"] for s in compare_steps], axis=0)
    avg_mae = np.mean([stats_by_step[s]["mae"] for s in compare_steps], axis=0)
    avg_corr = np.mean([stats_by_step[s]["corr"] for s in compare_steps], axis=0)
    
    for d in range(n_dims):
        name = dim_names[d] if d < len(dim_names) else f"Dim {d}"
        print(f"{name:15s} | Avg MSE: {avg_mse[d]:.6f} | Avg MAE: {avg_mae[d]:.6f} | Avg Corr: {avg_corr[d]:.4f}")
    
    print("=" * 80)
    print(f"Overall (across all steps & dims): MSE={np.mean(avg_mse):.6f} | MAE={np.mean(avg_mae):.6f}")

    # 6B. 生成误差随时间步变化的趋势图
    print("\n[INFO] Generating error vs horizon trend plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Overall MSE vs Step
    ax = axes[0, 0]
    overall_mse_by_step = [stats_by_step[s]["overall_mse"] for s in compare_steps]
    ax.plot(compare_steps, overall_mse_by_step, marker='o', linewidth=2, markersize=8)
    ax.set_xlabel("Prediction Step")
    ax.set_ylabel("Overall MSE")
    ax.set_title("Overall Error vs Prediction Horizon")
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Overall MAE vs Step
    ax = axes[0, 1]
    overall_mae_by_step = [stats_by_step[s]["overall_mae"] for s in compare_steps]
    ax.plot(compare_steps, overall_mae_by_step, marker='s', color='orange', linewidth=2, markersize=8)
    ax.set_xlabel("Prediction Step")
    ax.set_ylabel("Overall MAE")
    ax.set_title("Overall MAE vs Prediction Horizon")
    ax.grid(True, alpha=0.3)
    
    # Plot 3: 关键维度的MSE（位置）
    ax = axes[1, 0]
    key_pos_dims = [0, 1, 2, 10, 11, 12]  # L_pos_x,y,z, R_pos_x,y,z
    key_pos_names = ["L_x", "L_y", "L_z", "R_x", "R_y", "R_z"]
    for d, name in zip(key_pos_dims, key_pos_names):
        if d < n_dims:
            mse_by_step = [stats_by_step[s]["mse"][d] for s in compare_steps]
            ax.plot(compare_steps, mse_by_step, marker='o', label=name, linewidth=1.5)
    ax.set_xlabel("Prediction Step")
    ax.set_ylabel("MSE")
    ax.set_title("Position Error vs Horizon")
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Gripper MSE
    ax = axes[1, 1]
    if n_dims >= 20:
        left_grip_mse = [stats_by_step[s]["mse"][9] for s in compare_steps]
        right_grip_mse = [stats_by_step[s]["mse"][19] for s in compare_steps]
        ax.plot(compare_steps, left_grip_mse, marker='o', label="Left Gripper", linewidth=2)
        ax.plot(compare_steps, right_grip_mse, marker='s', label="Right Gripper", linewidth=2)
        ax.set_xlabel("Prediction Step")
        ax.set_ylabel("MSE")
        ax.set_title("Gripper Error vs Horizon")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"Error Analysis by Prediction Horizon\n{args.config}/{args.exp_name} step={args.step}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / f"ep{args.episode:04d}_error_by_horizon.png", dpi=150)
    plt.close()
    print("[OK] Saved error trend plot")

    # 特别检查gripper维度（显示所有时间步的范围）
    if n_dims >= 20:
        print("\n[Gripper Check - All Steps]")
        for step_idx in compare_steps[:3]:  # 只显示前3个时间步
            pred_step = pred_arrays[step_idx]
            gt_step = gt_arrays[step_idx]
            print(f"  Step+{step_idx}: Left=[{pred_step[:,9].min():.2f}, {pred_step[:,9].max():.2f}] | "
                  f"Right=[{pred_step[:,19].min():.2f}, {pred_step[:,19].max():.2f}]")


if __name__ == "__main__":
    main()
