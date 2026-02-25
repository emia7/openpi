import dataclasses
import einops
import numpy as np
from scipy.spatial.transform import Rotation as R

from openpi import transforms
from openpi.policies.pose_util import pose6_to_mat, mat_to_pose10d, mat_to_pose6, pose10d_to_mat
from openpi.policies.pose_repr_util import convert_pose_mat_rep
from openpi.models import model as _model


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
    elif "task" in data:
        if isinstance(data["task"], bytes):
            data["task"] = data["task"].decode("utf-8")
        inputs["prompt"] = data["task"]

    return inputs

# ==============================================================================
# Mode 1: Relative 6D (Rel6d -> Rel6d)
# ==============================================================================
@dataclasses.dataclass(frozen=True)
class UMIRelInputs(transforms.DataTransformFn):
    model_type: _model.ModelType
    action_horizon: int              # should match model_config.action_horizon

    def __call__(self, data: dict) -> dict:
        inputs = _process_common_inputs(data, self.model_type)

        curr_pos = np.asarray(data["eef_pos"], np.float32)                  # (3,) 
        curr_rot = np.asarray(data["eef_rot_axis_angle"], np.float32)       # (3,) 
        gripper = np.asarray(data["gripper_width"], np.float32)            # (1,) 

        # --- 1. Relative State Calculation ---
        # 计算当前 TCP 相对于 Demo Start 的变换
        if "demo_start_pose" in data:
            demo_start = np.asarray(data["demo_start_pose"], dtype=np.float32)

            # 使用矩阵运算计算相对变换 (T_rel = T_start^-1 * T_curr)
            curr_mat = pose6_to_mat(np.concatenate([curr_pos, curr_rot], axis=-1))
            start_mat = pose6_to_mat(demo_start)

            rel_state_mat = convert_pose_mat_rep(
                curr_mat,
                start_mat,
                pose_rep='relative',
                backward=False,
            )

            # 编码为 9D (Pos + Ortho6D)
            rel_state_9d = mat_to_pose10d(rel_state_mat)

            # 拼接 Gripper -> 10D State
            state = np.concatenate([rel_state_9d, gripper], axis=-1)
        else:
            # Fallback: 如果没有 demo_start，使用绝对坐标 (不推荐)
            print("[WARN] Missing demo_start_tcp_pose, using absolute state.")
            curr_mat = pose6_to_mat(np.concatenate([curr_pos, curr_rot], axis=-1))
            abs_state_9d = mat_to_pose10d(curr_mat)
            state = np.concatenate([abs_state_9d, gripper], axis=-1)

        inputs["state"] = state.astype(np.float32)

        # --- 2. Relative Action Calculation ---
        # 目标是计算 T_curr^-1 * T_next
        raw_action_seq = data.get("actions", None)
        if raw_action_seq is not None:
            # Make it (H,7)
            if raw_action_seq.ndim == 1:
                # single-step fallback: repeat to horizon, all pad=True except first
                raw_action_seq = raw_action_seq[None, :]
                pad_mask = np.ones((self.action_horizon,), dtype=np.bool_)
                pad_mask[0] = False
                raw_action_seq = np.repeat(raw_action_seq, self.action_horizon, axis=0)
            else:
                # assume already a sequence (T,7); pad/truncate to action_horizon
                pad_mask = np.zeros((self.action_horizon,), dtype=np.bool_)
                T = raw_action_seq.shape[0]
                if T < self.action_horizon:
                    pad_mask[T:] = True
                    raw_action_seq = np.concatenate(
                        [raw_action_seq, np.repeat(raw_action_seq[-1:], self.action_horizon - T, axis=0)], axis=0
                    )
                else:
                    raw_action_seq = raw_action_seq[: self.action_horizon]

            # 目标 Pose 矩阵序列 (H, 4, 4)
            target_mat_seq = pose6_to_mat(np.concatenate([raw_action_seq[:, :3], raw_action_seq[:, 3:6]], axis=-1))
            
            # 批量计算相对变换
            rel_action_mat_seq = convert_pose_mat_rep(
                target_mat_seq,
                curr_mat,
                pose_rep='relative',
                backward=False,
            )  # (H, 4, 4)
            
            # 编码为 9D Pose
            rel_action_9d = mat_to_pose10d(rel_action_mat_seq)

            # 提取 Gripper Action
            target_gripper = raw_action_seq[:, 6:7]  # (H,1)
            
            # 拼接 -> 10D Action
            actions = np.concatenate([rel_action_9d, target_gripper], axis=-1).astype(np.float32)

            inputs["actions"] = actions

        
        return inputs


# ==============================================================================
# Outputs (Inference Decode)
# ==============================================================================
@dataclasses.dataclass(frozen=True)
class UMIRelOutputs(transforms.DataTransformFn):
    """解码 Relative 10D -> 7D Delta (Base/EEF Frame)"""
    def __call__(self, data: dict) -> dict:
        act = np.asarray(data["actions"], dtype=np.float32)  # (H,11)
        pose9d = act[..., :9]   # (H,9)
        gripper = act[..., 9:10].astype(np.float32) / 88 # (H,1)

        mat = pose10d_to_mat(pose9d)
        pose6d = mat_to_pose6(mat)

        action_7d = np.concatenate([pose6d, gripper], axis=-1).astype(np.float32)

        return {"actions": action_7d}        # (H,7)
