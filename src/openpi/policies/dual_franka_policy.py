"""
Dual Franka bimanual VLA transforms (standalone; does not subclass `xv_dual_policy.XVDualInputs`).

After `RepackTransform`, each frame uses:
  - `left_view`, `right_view`, `third_view` (uint8 HWC)
  - `left_eef_pos`, `left_eef_rotvec`, `left_gripper`, and right counterparts
  - `demo_start_pose_left`, `demo_start_pose_right` (6D pos+rotvec)
  - training: `left_action`, `right_action` (7D absolute next state per hand)

Four `Inputs` classes differ only by **documentation / future hooks** (same tensor math today):
  - high third camera + Lumos wrists vs RealSense wrists
  - masked third camera + Lumos vs RealSense

`DualFrankaDualHandOutputs` inverts the 20-D model action chunk to 14-D env actions (7 per hand).
"""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.policies.pose_repr_util import convert_pose_mat_rep
from openpi.policies.pose_util import mat_to_pose10d, mat_to_pose6, pose10d_to_mat, pose6_to_mat


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _dual_franka_bimanual_inputs(
    data: dict,
    *,
    model_type: _model.ModelType,
    action_dim: int,
    action_horizon: int,
) -> dict:
    """Shared forward: images + 12-D relative rot6d state + optional (H,20) actions + prompt."""
    del model_type  # reserved for parity with other policies
    del action_dim

    third_img = _parse_image(data["third_view"])
    left_img = _parse_image(data["left_view"])
    right_img = _parse_image(data["right_view"])

    inputs: dict = {
        "image": {
            "base_0_rgb": third_img,
            "left_wrist_0_rgb": left_img,
            "right_wrist_0_rgb": right_img,
        },
        "image_mask": {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        },
    }

    l_pos = np.asarray(data["left_eef_pos"], np.float32)
    l_rot = np.asarray(data["left_eef_rotvec"], np.float32)
    l_g = np.asarray(data["left_gripper"], np.float32).reshape(1,)

    r_pos = np.asarray(data["right_eef_pos"], np.float32)
    r_rot = np.asarray(data["right_eef_rotvec"], np.float32)
    r_g = np.asarray(data["right_gripper"], np.float32).reshape(1,)

    l_cur_mat = pose6_to_mat(np.concatenate([l_pos, l_rot], axis=-1))
    r_cur_mat = pose6_to_mat(np.concatenate([r_pos, r_rot], axis=-1))

    l_start = np.asarray(data["demo_start_pose_left"], np.float32)
    l_start_mat = pose6_to_mat(l_start)
    r_start = np.asarray(data["demo_start_pose_right"], np.float32)
    r_start_mat = pose6_to_mat(r_start)

    l_rel_mat = convert_pose_mat_rep(l_cur_mat, l_start_mat, pose_rep="relative", backward=False)
    r_rel_mat = convert_pose_mat_rep(r_cur_mat, r_start_mat, pose_rep="relative", backward=False)

    l_rel_pose9 = mat_to_pose10d(l_rel_mat)
    r_rel_pose9 = mat_to_pose10d(r_rel_mat)
    l_rel_rot6 = l_rel_pose9[3:]
    r_rel_rot6 = r_rel_pose9[3:]

    inputs["state"] = np.concatenate([l_rel_rot6, r_rel_rot6], axis=-1).astype(np.float32)

    raw_left = data.get("left_action", None)
    raw_right = data.get("right_action", None)

    if raw_left is not None and raw_right is not None:
        raw_left = np.asarray(raw_left, np.float32)
        raw_right = np.asarray(raw_right, np.float32)

        if raw_left.ndim == 1:
            raw_left = raw_left[None, :]
        if raw_right.ndim == 1:
            raw_right = raw_right[None, :]

        t = min(raw_left.shape[0], raw_right.shape[0])
        if t < action_horizon:
            raw_left = np.concatenate(
                [raw_left[:t], np.repeat(raw_left[t - 1 : t], action_horizon - t, axis=0)], axis=0
            )
            raw_right = np.concatenate(
                [raw_right[:t], np.repeat(raw_right[t - 1 : t], action_horizon - t, axis=0)], axis=0
            )
        else:
            raw_left = raw_left[:action_horizon]
            raw_right = raw_right[:action_horizon]

        left_tgt_mat = pose6_to_mat(np.concatenate([raw_left[:, :3], raw_left[:, 3:6]], axis=-1))
        right_tgt_mat = pose6_to_mat(np.concatenate([raw_right[:, :3], raw_right[:, 3:6]], axis=-1))

        left_rel_mat = convert_pose_mat_rep(left_tgt_mat, l_cur_mat, pose_rep="relative", backward=False)
        right_rel_mat = convert_pose_mat_rep(right_tgt_mat, r_cur_mat, pose_rep="relative", backward=False)

        left_pose9 = mat_to_pose10d(left_rel_mat).astype(np.float32)
        right_pose9 = mat_to_pose10d(right_rel_mat).astype(np.float32)
        left_grip = raw_left[:, 6:7].astype(np.float32)
        right_grip = raw_right[:, 6:7].astype(np.float32)

        left_act10 = np.concatenate([left_pose9, left_grip], axis=-1)
        right_act10 = np.concatenate([right_pose9, right_grip], axis=-1)
        inputs["actions"] = np.concatenate([left_act10, right_act10], axis=-1).astype(np.float32)

    if "prompt" in data:
        inputs["prompt"] = str(data["prompt"])
    elif "task" in data:
        inputs["prompt"] = str(data["task"])

    return inputs


@dataclasses.dataclass(frozen=True)
class DualFrankaHighThirdLumosWristInputs(transforms.DataTransformFn):
    """High (non-masked) third-person camera; wrists from Lumos after Repack."""

    model_type: _model.ModelType
    action_dim: int
    action_horizon: int

    def __call__(self, data: dict) -> dict:
        return _dual_franka_bimanual_inputs(
            data,
            model_type=self.model_type,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
        )


@dataclasses.dataclass(frozen=True)
class DualFrankaHighThirdRealSenseWristInputs(transforms.DataTransformFn):
    """High third-person camera; wrists from RealSense D435 after Repack."""

    model_type: _model.ModelType
    action_dim: int
    action_horizon: int

    def __call__(self, data: dict) -> dict:
        return _dual_franka_bimanual_inputs(
            data,
            model_type=self.model_type,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
        )


@dataclasses.dataclass(frozen=True)
class DualFrankaMaskThirdLumosWristInputs(transforms.DataTransformFn):
    """Masked third-person camera; wrists from Lumos after Repack."""

    model_type: _model.ModelType
    action_dim: int
    action_horizon: int

    def __call__(self, data: dict) -> dict:
        return _dual_franka_bimanual_inputs(
            data,
            model_type=self.model_type,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
        )


@dataclasses.dataclass(frozen=True)
class DualFrankaMaskThirdRealSenseWristInputs(transforms.DataTransformFn):
    """Masked third-person camera; wrists from RealSense D435 after Repack."""

    model_type: _model.ModelType
    action_dim: int
    action_horizon: int

    def __call__(self, data: dict) -> dict:
        return _dual_franka_bimanual_inputs(
            data,
            model_type=self.model_type,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
        )


@dataclasses.dataclass(frozen=True)
class DualFrankaDualHandOutputs(transforms.DataTransformFn):
    """
    Model actions (H,20): left pose9d+grip + right pose9d+grip -> (H,14) concat left7, right7.
    """

    def __call__(self, data: dict) -> dict:
        act = np.asarray(data["actions"], dtype=np.float32)
        if act.ndim == 1:
            act = act[None, :]
        if act.shape[-1] != 20:
            raise ValueError(f"Expected model actions last-dim=20, got {act.shape}")

        l = act[..., :10]
        r = act[..., 10:20]
        l_pose9 = l[..., :9]
        l_grip = l[..., 9:10]
        r_pose9 = r[..., :9]
        r_grip = r[..., 9:10]

        l_mat = pose10d_to_mat(l_pose9)
        r_mat = pose10d_to_mat(r_pose9)
        l_pose6 = mat_to_pose6(l_mat)
        r_pose6 = mat_to_pose6(r_mat)

        l7 = np.concatenate([l_pose6, l_grip], axis=-1).astype(np.float32)
        r7 = np.concatenate([r_pose6, r_grip], axis=-1).astype(np.float32)
        both14 = np.concatenate([l7, r7], axis=-1).astype(np.float32)
        return {"actions": both14}
