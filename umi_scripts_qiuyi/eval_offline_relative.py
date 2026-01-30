import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from scipy.spatial.transform import Rotation

# OpenPi / LeRobot imports
from openpi.training import config as _config
from openpi.policies import policy_config
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 复用你在 policy 里的函数，保证逻辑一致性
def compute_relative_pose(curr_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
    """计算 target 相对于 curr 的变换 (T_rel = T_curr^-1 * T_target)"""
    p_curr = curr_pose[:3]
    r_curr = Rotation.from_quat(curr_pose[3:])
    p_target = target_pose[:3]
    r_target = Rotation.from_quat(target_pose[3:])
    
    # 相对位置 (在 curr 局部坐标系)
    p_rel = r_curr.inv().apply(p_target - p_curr)
    
    # 相对旋转
    r_rel = r_curr.inv() * r_target
    r_rel_vec = r_rel.as_rotvec()
    
    return np.concatenate([p_rel, r_rel_vec])

def get_args():
    parser = argparse.ArgumentParser(description="Offline evaluation for Pi0 Relative Action Policy")
    parser.add_argument("--config", type=str, required=True, help="Train config name (e.g., pi05_franka_rel_finetune)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--repo_id", type=str, required=True, help="LeRobot dataset repo id")
    parser.add_argument("--episode_idx", type=int, default=0, help="Index of the episode to evaluate")
    parser.add_argument("--prompt", type=str, default="do the task", help="Text prompt")
    parser.add_argument("--root_dir", type=str, default=None, help="Root directory for LeRobot datasets")
    return parser.parse_args()

def prepare_input(item, prompt):
    """构建推理输入"""
    # Image: (C,H,W) float -> (H,W,C) uint8
    image = item["observation.images.fish_eye_front"]
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
    if image.dtype == np.float32:
        image = (image * 255).astype(np.uint8)

    # State
    tcp_pose = item["observation.state.tcp_pose"]
    gripper_pose = item["observation.state.gripper_pose"]
    
    if isinstance(tcp_pose, torch.Tensor): tcp_pose = tcp_pose.numpy()
    if isinstance(gripper_pose, torch.Tensor): gripper_pose = gripper_pose.numpy()

    # 注意：根据你的 FrankaRelInputs，输入只需要单帧数据，不需要序列
    # 因为 Policy 内部不负责构建序列，只负责根据当前状态预测未来
    return {
        "tcp_pose": tcp_pose, # (7,)
        "gripper_pose": gripper_pose, # (1,)
        "image": image,
        "prompt": prompt
        # 不需要传入 "actions" 或 "raw_actions"，因为是推理模式
    }

def main():
    args = get_args()

    # 1. Load Policy
    print(f"Loading config: {args.config}")
    config = _config.get_config(args.config)
    print(f"Loading checkpoint: {args.checkpoint}")
    policy = policy_config.create_trained_policy(config, Path(args.checkpoint))
    print("Policy loaded.")

    # 2. Load Dataset
    print(f"Loading dataset: {args.repo_id}")
    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root_dir)
    
    # 获取 Episode 范围
    start_idx = dataset.episode_data_index["from"][args.episode_idx].item()
    end_idx = dataset.episode_data_index["to"][args.episode_idx].item()
    length = end_idx - start_idx
    print(f"Evaluating Episode {args.episode_idx} (Length: {length})")

    # 3. 准备 Ground Truth 序列
    # 我们需要一次性取出整个 Episode 的绝对位姿序列，以便计算 GT Relative Action
    # 假设 LeRobot 里存的是绝对位姿 "observation.state.tcp_pose"
    # 如果 LeRobot 里存的是 Delta Action，你需要先积分还原成绝对位姿，或者确保 dataset 里有绝对位姿列
    
    # 预加载所有帧的数据以加速
    all_tcp_poses = []
    all_gripper_actions = []
    
    print("Pre-loading dataset trajectory...")
    for i in range(length):
        item = dataset[start_idx + i]
        tcp = item["observation.state.tcp_pose"]
        if isinstance(tcp, torch.Tensor): tcp = tcp.numpy()
        all_tcp_poses.append(tcp)
        
        # 获取 gripper action (来自 action 列的最后一位)
        act = item["action"]
        if isinstance(act, torch.Tensor): act = act.numpy()
        all_gripper_actions.append(act[-1])
        
    all_tcp_poses = np.array(all_tcp_poses) # (T, 7)
    
    pred_rel_actions = []
    gt_rel_actions = []

    # 4. Inference Loop
    print("Running inference...")
    # 我们只对比 Step 0 的预测 (Closed-loop 假设)
    # 对于每个时间点 t，模型预测 t+1 的位置相对于 t 的位姿 (Relative)
    # GT 也是计算 pose[t+1] 相对于 pose[t]
    
    # 注意：由于需要 t+1，循环只能到 length - 1
    for t in tqdm.tqdm(range(length - 1)):
        # A. 获取当前帧数据进行推理
        item = dataset[start_idx + t]
        obs = prepare_input(item, args.prompt)
        
        with torch.no_grad():
            result = policy.infer(obs)
        
        # 模型输出：(Horizon, 7) 的相对动作序列
        # 我们取第0步：预测下一帧相对于当前帧的运动
        # [dx, dy, dz, drx, dry, drz, gripper]
        pred_action_step0 = result["actions"][0] 
        pred_rel_actions.append(pred_action_step0)
        
        # B. 计算 Ground Truth Relative Action
        # 当前绝对位姿
        curr_pose_abs = all_tcp_poses[t]
        # 下一步绝对位姿
        next_pose_abs = all_tcp_poses[t+1]
        
        # 计算相对变换 (这是模型应该学到的目标)
        rel_pose_gt = compute_relative_pose(curr_pose_abs, next_pose_abs)
        
        # 拼接 Gripper Action GT (使用 t 时刻记录的 action，还是 t+1 时刻的状态？)
        # 通常 Action 预测的是下一步怎么做，所以用 t 时刻数据里的 action 指令最准
        gripper_gt = all_gripper_actions[t]
        
        gt_action_step0 = np.concatenate([rel_pose_gt, [gripper_gt]])
        gt_rel_actions.append(gt_action_step0)

    # 转为 numpy
    pred_rel_actions = np.array(pred_rel_actions) # (T-1, 7)
    gt_rel_actions = np.array(gt_rel_actions)     # (T-1, 7)

    # 5. Visualization
    mse = np.mean((pred_rel_actions - gt_rel_actions) ** 2)
    print(f"\nMean Squared Error (Relative): {mse:.6f}")
    
    print("Plotting...")
    dim_names = ["Rel X", "Rel Y", "Rel Z", "Rel Rx", "Rel Ry", "Rel Rz", "Gripper"]
    fig, axes = plt.subplots(7, 1, figsize=(10, 20), sharex=True)
    steps = range(len(pred_rel_actions))

    for dim in range(7):
        ax = axes[dim]
        # GT 画黑虚线
        ax.plot(steps, gt_rel_actions[:, dim], label="GT (Relative)", color="black", linestyle="--", alpha=0.7)
        # Pred 画红实线
        ax.plot(steps, pred_rel_actions[:, dim], label="Pred (Relative)", color="red", linewidth=1.5)
        
        ax.set_ylabel(dim_names[dim])
        ax.grid(True, alpha=0.3)
        if dim == 0: ax.legend()

    plt.xlabel("Time Step")
    plt.suptitle(f"Offline Eval (Relative Action) - Ep {args.episode_idx}\nMSE: {mse:.5f}")
    plt.tight_layout()
    
    out_file = f"eval_rel_ep{args.episode_idx}.png"
    plt.savefig(out_file)
    print(f"Saved plot to {out_file}")

if __name__ == "__main__":
    main()