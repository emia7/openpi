import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

def make_franka_example() -> dict:
    """Creates a random input example for the Franka policy (for debugging)."""
    return {
        "tcp_pose": np.random.rand(7).astype(np.float32),
        "gripper_pose": np.random.rand(1).astype(np.float32),
        "image": np.random.randint(256, size=(240, 420, 3), dtype=np.uint8),
        "prompt": "do the task",
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

@dataclasses.dataclass(frozen=True)
class FrankaInputs(transforms.DataTransformFn):
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
        tcp = np.asarray(data["tcp_pose"])
        gripper = np.asarray(data["gripper_pose"])
        if gripper.ndim == 0: gripper = gripper[None]
        state = np.concatenate([tcp, gripper]).astype(np.float32)
        
        # ---------------------------------------------------------
        # 2. Image Processing
        # ---------------------------------------------------------
        base_image = _parse_image(data["image"])

        # 期望的输入槽位是base_0_rgb，其他槽位 (wrist) 用全0填充并Mask掉
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, np.zeros_like(base_image), np.zeros_like(base_image))
                # Mask: True 表示有效，False 表示 Padding (忽略)
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
        if "actions" in data:
            # 原始动作 (7维)
            raw_actions = np.asarray(data["actions"], dtype=np.float32)
            
            # Padding
            target_dim = 32
            current_dim = raw_actions.shape[-1]
            if current_dim < target_dim:
                padding = np.zeros(
                    raw_actions.shape[:-1] + (target_dim - current_dim,), 
                    dtype=np.float32
                )
                # 拼接: [Action(7) | Zeros(25)]
                padded_actions = np.concatenate([raw_actions, padding], axis=-1)
                inputs["actions"] = padded_actions
            else:
                inputs["actions"] = raw_actions

        # ---------------------------------------------------------
        # 4. Prompt Processing
        # ---------------------------------------------------------
        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs

@dataclasses.dataclass(frozen=True)
class FrankaOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # 模型输出是 32 维的，只需要前7维给机器人
        # 含义：[dx, dy, dz, drx, dry, drz, gripper] (相对于franka base坐标系)
        return {"actions": np.asarray(data["actions"][..., :7])}