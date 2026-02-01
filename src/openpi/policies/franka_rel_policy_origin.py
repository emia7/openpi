import dataclasses
import einops
import numpy as np
from scipy.spatial.transform import Rotation

from openpi import transforms
from openpi.models import model as _model

def make_franka_example() -> dict:
    """Creates a random input example for the Franka policy (for debugging)."""
    return {
        "tcp_pose": np.random.rand(7).astype(np.float32),
        "gripper_pose": np.random.rand(1).astype(np.float32),
        "image": np.random.randint(256, size=(240, 420, 3), dtype=np.uint8),
        "prompt": "do something",
    }

def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image

def _compute_relative_pose(curr_pose: np.ndarray, target_pose: np.ndarray) -> np.ndarray:
    """
    计算 target_pose 相对于 curr_pose 的变换 (T_rel = T_curr^-1 * T_target)。
    结果表示在 curr_pose 局部坐标系下的运动。
    
    Args:
        curr_pose: [x, y, z, qx, qy, qz, qw] (7,) - Base Frame
        target_pose: [x, y, z, qx, qy, qz, qw] (7,) - Base Frame
    Returns:
        relative_pose: [dx, dy, dz, drx, dry, drz] (6,)
        其中 drx, dry, drz 是旋转向量 (Rotation Vector)
    """
    # 1. 提取位置和旋转
    p_curr = curr_pose[:3]
    r_curr = Rotation.from_quat(curr_pose[3:])
    
    p_target = target_pose[:3]
    r_target = Rotation.from_quat(target_pose[3:])
    
    # 2. 计算相对位置 (Transform target pos into current local frame)
    # p_rel = R_curr^T * (p_target - p_curr)
    p_rel = r_curr.inv().apply(p_target - p_curr)
    
    # 3. 计算相对旋转
    # R_rel = R_curr^T * R_target
    r_rel = r_curr.inv() * r_target
    r_rel_vec = r_rel.as_rotvec()
    
    return np.concatenate([p_rel, r_rel_vec]) 

@dataclasses.dataclass(frozen=True)
class FrankaRelInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # =========================================================
        # 1. 解析输入数据 & 处理维度差异 (Inference vs Training)
        # =========================================================
        tcp_pose_in = np.asarray(data["tcp_pose"], dtype=np.float32)
        
        # [核心修复] 判断是单帧(推理)还是序列(训练)
        if tcp_pose_in.ndim == 1:
            # === 推理模式 ===
            # 输入是 (7,)，直接作为当前位姿
            curr_tcp_pose = tcp_pose_in
            # 为了后续逻辑兼容，伪造一个序列维度 (1, 7)
            tcp_pose_seq = tcp_pose_in[None, :]
        else:
            # === 训练模式 ===
            # 输入可能是 (H, 7) 或 (1, H, 7)
            if tcp_pose_in.ndim == 3 and tcp_pose_in.shape[0] == 1:
                tcp_pose_in = tcp_pose_in[0] # (1, H, 7) -> (H, 7)
            
            # 取序列的第一帧作为当前位姿
            curr_tcp_pose = tcp_pose_in[0] # (H, 7) -> (7,)
            tcp_pose_seq = tcp_pose_in

        # 处理 raw_actions (只有训练时有)
        raw_action_seq = np.asarray(data.get("raw_actions", []), dtype=np.float32) 
        if raw_action_seq.size == 0:
             raw_action_seq = np.zeros_like(tcp_pose_seq)
        elif raw_action_seq.ndim == 3 and raw_action_seq.shape[0] == 1:
            raw_action_seq = raw_action_seq[0]

        # 处理 gripper (单帧)
        gripper_pose = np.asarray(data["gripper_pose"], dtype=np.float32)
        if gripper_pose.ndim == 0: gripper_pose = gripper_pose[None]

        # =========================================================
        # 2. 构建 State
        # =========================================================
        # 此时 curr_tcp_pose 是 (7,), gripper_pose 是 (1,)
        # print(f"DEBUG: curr_tcp {curr_tcp_pose.shape}, gripper {gripper_pose.shape}") 
        state = np.concatenate([curr_tcp_pose, gripper_pose], axis=-1).astype(np.float32)

        # =========================================================
        # 3. 构建 Actions (计算相对动作目标) - 仅训练时有效
        # =========================================================
        if raw_action_seq.size > 0 and "raw_actions" in data:
            rel_actions_list = []
            horizon = len(tcp_pose_seq)
            
            for i in range(horizon):
                # A. 目标绝对位姿 (来自序列)
                target_tcp_abs = tcp_pose_seq[i]
                
                # B. 目标夹爪 (来自 raw_actions 序列末位)
                if len(raw_action_seq) > i:
                    target_gripper_action = raw_action_seq[i, -1]
                else:
                    target_gripper_action = 0.0

                # C. 计算相对变换 (始终相对于 t=0 时刻的 curr_tcp_pose)
                rel_pose_6d = _compute_relative_pose(curr_tcp_pose, target_tcp_abs)
                
                # D. 组合
                rel_action = np.concatenate([rel_pose_6d, [target_gripper_action]])
                rel_actions_list.append(rel_action)
            
            processed_actions = np.array(rel_actions_list, dtype=np.float32)
            
            # E. Padding
            target_dim = 32
            current_dim = processed_actions.shape[-1]
            if current_dim < target_dim:
                padding = np.zeros(
                    processed_actions.shape[:-1] + (target_dim - current_dim,), 
                    dtype=np.float32
                )
                inputs_actions = np.concatenate([processed_actions, padding], axis=-1)
            else:
                inputs_actions = processed_actions
        else:
            inputs_actions = None

        # =========================================================
        # 4. 组装返回
        # =========================================================
        base_image = _parse_image(data["image"])
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, np.zeros_like(base_image), np.zeros_like(base_image))
                image_masks = (np.True_, np.False_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base_image, np.zeros_like(base_image), np.zeros_like(base_image))
                image_masks = (np.True_, np.False_, np.False_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }
        
        if inputs_actions is not None:
            inputs["actions"] = inputs_actions

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs

@dataclasses.dataclass(frozen=True)
class FrankaRelOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # 推理输出：截取前 7 维
        # 含义：[dx, dy, dz, drx, dry, drz, gripper] (相对于当前EEF坐标系)
        return {"actions": np.asarray(data["actions"][..., :7])}