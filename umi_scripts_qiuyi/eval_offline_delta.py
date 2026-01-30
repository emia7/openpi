import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import einops

# OpenPi / LeRobot imports
from openpi.training import config as _config
from openpi.policies import policy_config
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def get_args():
    parser = argparse.ArgumentParser(description="Offline evaluation for Pi0.5 Delta Action Policy")
    parser.add_argument("--config", type=str, required=True, help="Train config name (e.g., pi05_franka_finetune)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint directory (e.g., .../2000)")
    parser.add_argument("--repo_id", type=str, required=True, help="LeRobot dataset repo id (e.g., local/franka_pick_place_0112)")
    parser.add_argument("--episode_idx", type=int, default=0, help="Index of the episode to evaluate")
    parser.add_argument("--prompt", type=str, default="do the task", help="Text prompt for the model")
    parser.add_argument("--horizon", type=int, default=10, help="Action horizon (chunk size) of the model")
    parser.add_argument("--root_dir", type=str, default=None, help="Root directory for LeRobot datasets")
    return parser.parse_args()

def prepare_input(item, prompt):
    """
    将 LeRobot Dataset 的单帧数据转换为 OpenPi Policy 需要的输入格式。
    参考 franka_policy.py 中的 FrankaInputs
    """
    # 1. Image: LeRobot 返回的是 (C, H, W) float [0,1]，我们需要 (H, W, C) uint8 [0,255]
    image = item["observation.images.fish_eye_front"] # 根据你的数据集 key 修改
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
    
    # 转换为 uint8 [0, 255]
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)

    # 2. State
    tcp_pose = item["observation.state.tcp_pose"]
    gripper_pose = item["observation.state.gripper_pose"]
    
    if isinstance(tcp_pose, torch.Tensor): tcp_pose = tcp_pose.numpy()
    if isinstance(gripper_pose, torch.Tensor): gripper_pose = gripper_pose.numpy()

    return {
        "tcp_pose": tcp_pose,
        "gripper_pose": gripper_pose,
        "image": image,
        "prompt": prompt
    }

def main():
    args = get_args()

    # 1. Load Policy
    print(f"Loading config: {args.config}")
    config = _config.get_config(args.config)
    
    print(f"Loading checkpoint: {args.checkpoint}")
    # 注意：这里假设 checkpoint 路径是本地的，如果是 gs:// 需要用 download.maybe_download
    policy = policy_config.create_trained_policy(config, Path(args.checkpoint))
    print("Policy loaded successfully.")

    # 2. Load Dataset
    print(f"Loading dataset: {args.repo_id}")
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.root_dir
    )
    print(f"Dataset loaded. Total episodes: {dataset.num_episodes}")

    # 3. Get Episode Data
    # 获取指定 episode 的帧索引范围
    start_idx = dataset.episode_data_index["from"][args.episode_idx].item()
    end_idx = dataset.episode_data_index["to"][args.episode_idx].item()
    length = end_idx - start_idx
    print(f"Evaluating Episode {args.episode_idx} (Length: {length} steps)")

    # 存储结果
    # 我们只记录每个时刻模型预测的 Chunk 的第 0 步动作 (Closed-loop simulation assumption)
    # 或者你可以记录整个 Chunk 进行更复杂的分析。这里为了绘图清晰，记录第0步动作。
    all_pred_actions = [] 
    all_gt_actions = []

    # 4. Inference Loop
    # 遍历该 episode 的每一帧
    for i in tqdm.tqdm(range(length)):
        # 获取数据
        item = dataset[start_idx + i]
        
        # 准备模型输入
        obs = prepare_input(item, args.prompt)
        
        # 推理
        # infer 返回的是 {"actions": (Horizon, 7)}
        with torch.no_grad(): # 虽然是 JAX 模型，但防止数据转换产生的梯度
            result = policy.infer(obs)
        
        # 获取模型预测的动作 (Chunk)
        # 你的 FrankaOutputs 只取前 7 维，所以 shape 是 (Horizon, 7)
        pred_chunk = result["actions"]
        
        # 获取 GT 动作
        # 注意：GT 也是 delta action
        gt_action = item["action"]
        if isinstance(gt_action, torch.Tensor):
            gt_action = gt_action.numpy()

        # 策略 1: 仅记录当前步的预测 (只看 chunk 的第 0 帧)
        # 这代表了如果每一帧都重新推理，模型想怎么走
        all_pred_actions.append(pred_chunk[0])
        all_gt_actions.append(gt_action)

    all_pred_actions = np.array(all_pred_actions) # (T, 7)
    all_gt_actions = np.array(all_gt_actions)     # (T, 7)

    # 5. Analysis
    # 计算 MSE
    mse = np.mean((all_pred_actions - all_gt_actions) ** 2)
    print(f"\nMean Squared Error (MSE): {mse:.6f}")

    # 6. Visualization
    print("Plotting results...")
    dim_names = ["Delta X", "Delta Y", "Delta Z", "Delta Rx", "Delta Ry", "Delta Rz", "Gripper"]
    
    fig, axes = plt.subplots(7, 1, figsize=(10, 20), sharex=True)
    steps = range(length)

    for dim in range(7):
        ax = axes[dim]
        ax.plot(steps, all_gt_actions[:, dim], label="Ground Truth", color="black", linestyle="--", alpha=0.7)
        ax.plot(steps, all_pred_actions[:, dim], label="Prediction (Step 0)", color="red", linewidth=1.5)
        
        ax.set_ylabel(dim_names[dim])
        ax.grid(True, alpha=0.3)
        if dim == 0:
            ax.legend()

    plt.xlabel("Time Step")
    plt.suptitle(f"Offline Evaluation - Episode {args.episode_idx}\nModel: {args.config} | MSE: {mse:.5f}")
    plt.tight_layout()
    
    output_filename = f"eval_ep{args.episode_idx}_delta.png"
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")

if __name__ == "__main__":
    # 为了避免 JAX 抢占所有显存导致 OOM，限制显存或使用 CPU (如果只是跑单条推理)
    # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    main()