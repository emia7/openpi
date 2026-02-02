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
    """Helper to convert images to uint8 HWC format."""
    image = np.asarray(image)
    # LeRobot/HF Datasets sometimes return float [0,1], convert to uint8 [0,255]
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    
    # Handle Channel-First vs Channel-Last: (C, H, W) -> (H, W, C)
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
    p_rel = r_curr.inv().apply(p_target - p_curr)
    
    # 3. 计算相对旋转
    r_rel = r_curr.inv() * r_target
    r_rel_vec = r_rel.as_rotvec()
    
    return np.concatenate([p_rel, r_rel_vec]) 

@dataclasses.dataclass(frozen=True)
class FrankaRelInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # ---------------------------------------------------------
        # 0. Print Input Data Keys Shapes
        # ---------------------------------------------------------
        # for key, value in data.items():
        #     print(f"Input key: {key}, shape: {np.asarray(value).shape}")
        #     if key == "gripper_pose" or key =="prompt" or key =="tcp_pose":
        #         print(f" {key} value: {value}")

        # ---------------------------------------------------------
        # 1. State Processing
        # ---------------------------------------------------------
        tcp_pose_in = np.asarray(data["tcp_pose"], dtype=np.float32)
        if tcp_pose_in.ndim == 1:
            # 推理模式：输入是 (7,)，直接作为当前位姿，为了后续逻辑兼容，伪造一个序列维度 (1, 7)
            curr_tcp_pose = tcp_pose_in
            tcp_pose_seq = tcp_pose_in[None, :]
        else:
            # 训练模式：输入是 (action_horizon, 7)，取序列的第一帧作为当前位姿
            curr_tcp_pose = tcp_pose_in[0]
            tcp_pose_seq = tcp_pose_in

        gripper_pose = np.asarray(data["gripper_pose"], dtype=np.float32)
        if gripper_pose.ndim == 0: gripper_pose = gripper_pose[None]

        state = np.concatenate([curr_tcp_pose, gripper_pose], axis=-1).astype(np.float32)

        # ---------------------------------------------------------
        # 2. Image Processing
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # 3. Action Processing
        # ---------------------------------------------------------
         
        raw_action_seq = np.asarray(data.get("raw_actions", []), dtype=np.float32) 
        if raw_action_seq.size == 0:
             raw_action_seq = np.zeros_like(tcp_pose_seq)

        # 计算相对动作序列
        rel_actions_list = []
        horizon = len(tcp_pose_seq)
        for i in range(horizon):
            # 目标绝对位姿 (来自序列)
            target_tcp_abs = tcp_pose_seq[i]
            
            # 目标夹爪 (来自raw_actions序列末位)
            if len(raw_action_seq) > i:
                target_gripper_action = raw_action_seq[i, -1]
            else:
                target_gripper_action = 0.0

            # 计算相对变换 (始终相对于curr_tcp_pose)
            rel_pose_6d = _compute_relative_pose(curr_tcp_pose, target_tcp_abs)
            
            # 组合
            rel_action = np.concatenate([rel_pose_6d, [target_gripper_action]])
            rel_actions_list.append(rel_action)
        
        processed_actions = np.array(rel_actions_list, dtype=np.float32)
        
        # Padding
        target_dim = 32
        current_dim = processed_actions.shape[-1]
        if current_dim < target_dim:
            padding = np.zeros(
                processed_actions.shape[:-1] + (target_dim - current_dim,), 
                dtype=np.float32
            )
            padded_actions = np.concatenate([processed_actions, padding], axis=-1)
            inputs["actions"] = padded_actions
        else:
            inputs["actions"] = processed_actions


        # ---------------------------------------------------------
        # 5. Prompt Processing
        # ---------------------------------------------------------
        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs

@dataclasses.dataclass(frozen=True)
class FrankaRelOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # 模型输出是 32 维的，只需要前7维给机器人
        # 含义：[dx, dy, dz, drx, dry, drz, gripper] (相对于当前EEF坐标系)
        return {"actions": np.asarray(data["actions"][..., :7])}