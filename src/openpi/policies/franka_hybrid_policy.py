import dataclasses
import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model

from openpi.policies.pose_util import (
    mat_to_pose10d, pose10d_to_mat, mat_to_pose6, pose6_to_mat, pose7_to_mat,
    pos_rot_to_mat, mat_to_pos_rot
)
from openpi.policies.pose_repr_util import convert_pose_mat_rep

def _parse_image(image) -> np.ndarray:
    """Helper to convert images to uint8 HWC format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image

# ==============================================================================
# Common Base Logic
# ==============================================================================
def _process_common_inputs(data: dict, model_type: _model.ModelType):
    """处理图像、Prompt 等通用输入"""
    base_image = _parse_image(data["image"])
    
    match model_type:
        case _model.ModelType.PI0 | _model.ModelType.PI05:
            names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            images = (base_image, np.zeros_like(base_image), np.zeros_like(base_image))
            image_masks = (np.True_, np.False_, np.False_)
        case _model.ModelType.PI0_FAST:
            names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
            images = (base_image, np.zeros_like(base_image), np.zeros_like(base_image))
            image_masks = (np.True_, np.False_, np.False_)
        case _:
            raise ValueError(f"Unsupported model type: {model_type}")

    inputs = {
        "image": dict(zip(names, images, strict=True)),
        "image_mask": dict(zip(names, image_masks, strict=True)),
    }

    if "prompt" in data:
        if isinstance(data["prompt"], bytes):
            data["prompt"] = data["prompt"].decode("utf-8")
        inputs["prompt"] = data["prompt"]
        
    return inputs

def _get_tcp_seq(data: dict):
    """提取 tcp_pose 序列和当前帧"""
    tcp_pose_in = np.asarray(data["tcp_pose"], dtype=np.float32)
    
    if tcp_pose_in.ndim == 1:
        # 推理模式: (7,)
        curr_tcp = tcp_pose_in
        tcp_seq = tcp_pose_in[None, :]
    else:
        # 训练模式: (H, 7)
        if tcp_pose_in.ndim == 3 and tcp_pose_in.shape[0] == 1:
            tcp_pose_in = tcp_pose_in[0]
        curr_tcp = tcp_pose_in[0]
        tcp_seq = tcp_pose_in
        
    return curr_tcp, tcp_seq

def _get_gripper_pose(data: dict):
    gripper = np.asarray(data["gripper_pose"], dtype=np.float32)
    if gripper.ndim == 0: gripper = gripper[None]
    return gripper


# ==============================================================================
# Mode 1: Relative 6D (Rel6d -> Rel6d)
# ==============================================================================
@dataclasses.dataclass(frozen=True)
class FrankaRelInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        inputs = _process_common_inputs(data, self.model_type)
        curr_tcp, tcp_seq = _get_tcp_seq(data)
        gripper = _get_gripper_pose(data)
        
        # --- 1. Relative State Calculation ---
        # 计算当前 TCP 相对于 Demo Start 的变换
        # 假设 data["demo_start_tcp_pose"] 是 (7,)
        if "demo_start_tcp_pose" in data:
            demo_start = np.asarray(data["demo_start_tcp_pose"], dtype=np.float32)
            
            # 使用矩阵运算计算相对变换 (T_rel = T_start^-1 * T_curr)
            curr_mat = pose7_to_mat(curr_tcp)
            start_mat = pose7_to_mat(demo_start)
            
            rel_state_mat = convert_pose_mat_rep(
                curr_mat, 
                start_mat, 
                pose_rep='relative', 
                backward=False
            )
            # 编码为 9D (Pos + Ortho6D)
            rel_state_9d = mat_to_pose10d(rel_state_mat)
            
            # 拼接 Gripper -> 10D State
            state = np.concatenate([rel_state_9d, gripper], axis=-1)
        else:
            # Fallback: 如果没有 demo_start，使用绝对坐标 (不推荐)
            print("[WARN] Missing demo_start_tcp_pose, using absolute state.")
            curr_mat = pose7_to_mat(curr_tcp)
            abs_state_9d = mat_to_pose10d(curr_mat)
            state = np.concatenate([abs_state_9d, gripper], axis=-1)
            
        inputs["state"] = state.astype(np.float32)

        # --- 2. Relative Action Calculation ---
        # 目标是计算 T_curr^-1 * T_next
        raw_action_seq = np.asarray(data.get("raw_actions", []), dtype=np.float32)
        if raw_action_seq.size > 0:
            horizon = len(tcp_seq)
            
            # 目标 Pose 矩阵序列 (H, 4, 4)
            target_mat_seq = pose7_to_mat(tcp_seq) # tcp_seq 是绝对位姿
            
            # 当前 Pose 矩阵 (4, 4)
            curr_mat = pose7_to_mat(curr_tcp)
            
            # 批量计算相对变换
            rel_action_mat_seq = convert_pose_mat_rep(
                target_mat_seq,
                curr_mat,
                pose_rep='relative',
                backward=False
            ) # (H, 4, 4)
            
            # 编码为 9D Pose
            rel_action_9d = mat_to_pose10d(rel_action_mat_seq) # (H, 9)
            
            # 提取 Gripper Action
            # raw_actions 最后一维是 gripper
            target_gripper = raw_action_seq[:, -1:] # (H, 1)
            if len(target_gripper) < horizon:
                 target_gripper = np.zeros((horizon, 1), dtype=np.float32)
            
            # 拼接 -> 10D Action
            actions = np.concatenate([rel_action_9d, target_gripper], axis=-1)
            
            # Padding to 32
            inputs["actions"] = transforms.pad_to_dim(actions, 32).astype(np.float32)

        return inputs


# ==============================================================================
# Mode 2: Absolute 6D (Abs6d -> Abs6d)
# ==============================================================================
@dataclasses.dataclass(frozen=True)
class FrankaAbsInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        inputs = _process_common_inputs(data, self.model_type)
        curr_tcp, tcp_seq = _get_tcp_seq(data)
        gripper = _get_gripper_pose(data)

        # --- 1. Absolute State Calculation ---
        # 要求：10D with 6d rot
        # 当前绝对 Pose -> Matrix -> 10D Encoding
        curr_mat = pose7_to_mat(curr_tcp)
        abs_state_9d = mat_to_pose10d(curr_mat) # (9,)
        
        state = np.concatenate([abs_state_9d, gripper], axis=-1)
        inputs["state"] = state.astype(np.float32)

        # --- 2. Absolute Action Calculation ---
        # 目标：从 raw_actions (Delta) 恢复出 Absolute Pose
        # raw_actions 前6维是 Delta (Base Frame)，第7维是 Gripper
        
        raw_action_seq = np.asarray(data.get("raw_actions", []), dtype=np.float32)
        if raw_action_seq.size > 0:
            # 直接使用 Dataset 提供的 tcp_pose 序列，把 tcp_seq 转成 10D 格式
            
            # 目标 Pose 矩阵序列 (H, 4, 4)
            target_mat_seq = pose7_to_mat(tcp_seq) 
            
            # 编码为 9D Absolute Pose
            abs_action_9d = mat_to_pose10d(target_mat_seq) # (H, 9)
            
            # 提取 Gripper
            target_gripper = raw_action_seq[:, -1:]
            if len(target_gripper) < len(abs_action_9d):
                 target_gripper = np.zeros((len(abs_action_9d), 1), dtype=np.float32)

            # 拼接 -> 10D Action
            actions = np.concatenate([abs_action_9d, target_gripper], axis=-1)
            
            # Padding to 32
            inputs["actions"] = transforms.pad_to_dim(actions, 32).astype(np.float32)

        return inputs


# ==============================================================================
# Outputs (Inference Decode)
# ==============================================================================
@dataclasses.dataclass(frozen=True)
class FrankaRelOutputs(transforms.DataTransformFn):
    """解码 Relative 10D -> 7D Delta (Base/EEF Frame)"""
    def __call__(self, data: dict) -> dict:
        act = np.asarray(data["actions"], dtype=np.float32)
        pose9d = act[..., :9]
        gripper = act[..., 9:10]
        
        mat_rel = pose10d_to_mat(pose9d)
        pose6d_delta = mat_to_pose6(mat_rel) # [dx, dy, dz, drx, dry, drz]
        
        action_7d = np.concatenate([pose6d_delta, gripper], axis=-1)
        return {"actions": action_7d}

@dataclasses.dataclass(frozen=True)
class FrankaAbsOutputs(transforms.DataTransformFn):
    """解码 Absolute 10D -> 7D Absolute Pose"""
    def __call__(self, data: dict) -> dict:
        act = np.asarray(data["actions"], dtype=np.float32)
        pose9d = act[..., :9]
        gripper = act[..., 9:10]
        
        mat_abs = pose10d_to_mat(pose9d)
        pose6d_abs = mat_to_pose6(mat_abs) # [x, y, z, rx, ry, rz]
        
        action_7d = np.concatenate([pose6d_abs, gripper], axis=-1)
        return {"actions": action_7d}