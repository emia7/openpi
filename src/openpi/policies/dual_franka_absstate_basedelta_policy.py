from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.policies.pose_util import mat_to_pose10d, mat_to_pose6, pose10d_to_mat, pose6_to_mat


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _delta_pose6_in_base(prev_pose6: np.ndarray, next_pose6: np.ndarray) -> np.ndarray:
    prev_pose6 = np.asarray(prev_pose6, dtype=np.float32)
    next_pose6 = np.asarray(next_pose6, dtype=np.float32)

    prev_mat = pose6_to_mat(prev_pose6)
    next_mat = pose6_to_mat(next_pose6)

    prev_rot = prev_mat[..., :3, :3]
    next_rot = next_mat[..., :3, :3]
    delta_rot = next_rot @ np.swapaxes(prev_rot, -1, -2)
    delta_pos = next_pose6[..., :3] - prev_pose6[..., :3]

    delta_mat = np.broadcast_to(np.eye(4, dtype=np.float32), prev_mat.shape).copy()
    delta_mat[..., :3, :3] = delta_rot
    delta_mat[..., :3, 3] = delta_pos
    return mat_to_pose6(delta_mat).astype(np.float32)


def _integrate_base_delta(prev_pose6: np.ndarray, delta_pose6: np.ndarray) -> np.ndarray:
    prev_pose6 = np.asarray(prev_pose6, dtype=np.float32)
    delta_pose6 = np.asarray(delta_pose6, dtype=np.float32)

    prev_mat = pose6_to_mat(prev_pose6)
    delta_mat = pose6_to_mat(delta_pose6)

    prev_rot = prev_mat[..., :3, :3]
    delta_rot = delta_mat[..., :3, :3]

    next_pos = prev_pose6[..., :3] + delta_pose6[..., :3]
    next_rot = delta_rot @ prev_rot

    next_mat = np.broadcast_to(np.eye(4, dtype=np.float32), prev_mat.shape).copy()
    next_mat[..., :3, :3] = next_rot
    next_mat[..., :3, 3] = next_pos
    return mat_to_pose6(next_mat).astype(np.float32)


def _dual_franka_absstate_basedelta_inputs(
    data: dict,
    *,
    model_type: _model.ModelType,
    action_dim: int,
    action_horizon: int,
    mask_third_view: bool = False,
) -> dict:
    del model_type
    del action_dim

    third_img = _parse_image(data["third_view"])
    if mask_third_view:
        third_img = np.zeros_like(third_img)
    left_img = _parse_image(data["left_view"])
    right_img = _parse_image(data["right_view"])

    l_pos = np.asarray(data["left_eef_pos"], np.float32)
    l_rot = np.asarray(data["left_eef_rotvec"], np.float32)
    l_g = np.asarray(data["left_gripper"], np.float32).reshape(1,)
    r_pos = np.asarray(data["right_eef_pos"], np.float32)
    r_rot = np.asarray(data["right_eef_rotvec"], np.float32)
    r_g = np.asarray(data["right_gripper"], np.float32).reshape(1,)

    l_cur_pose6 = np.concatenate([l_pos, l_rot], axis=-1).astype(np.float32)
    r_cur_pose6 = np.concatenate([r_pos, r_rot], axis=-1).astype(np.float32)
    l_cur_pose9 = mat_to_pose10d(pose6_to_mat(l_cur_pose6)).astype(np.float32)
    r_cur_pose9 = mat_to_pose10d(pose6_to_mat(r_cur_pose6)).astype(np.float32)

    inputs: dict = {
        "image": {
            "base_0_rgb": third_img,
            "left_wrist_0_rgb": left_img,
            "right_wrist_0_rgb": right_img,
        },
        "image_mask": {
            "base_0_rgb": np.False_ if mask_third_view else np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        },
        # Absolute state in franka base frame.
        "state": np.concatenate([l_cur_pose9, l_g, r_cur_pose9, r_g], axis=-1).astype(np.float32),
    }

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

        prev_left_pose6 = np.concatenate([l_cur_pose6[None, :], raw_left[:-1, :6]], axis=0)
        prev_right_pose6 = np.concatenate([r_cur_pose6[None, :], raw_right[:-1, :6]], axis=0)

        left_delta_pose6 = _delta_pose6_in_base(prev_left_pose6, raw_left[:, :6])
        right_delta_pose6 = _delta_pose6_in_base(prev_right_pose6, raw_right[:, :6])
        left_delta_pose9 = mat_to_pose10d(pose6_to_mat(left_delta_pose6)).astype(np.float32)
        right_delta_pose9 = mat_to_pose10d(pose6_to_mat(right_delta_pose6)).astype(np.float32)

        left_grip_abs = raw_left[:, 6:7].astype(np.float32)
        right_grip_abs = raw_right[:, 6:7].astype(np.float32)

        left_act10 = np.concatenate([left_delta_pose9, left_grip_abs], axis=-1)
        right_act10 = np.concatenate([right_delta_pose9, right_grip_abs], axis=-1)
        inputs["actions"] = np.concatenate([left_act10, right_act10], axis=-1).astype(np.float32)

    if "prompt" in data:
        inputs["prompt"] = str(data["prompt"])
    elif "task" in data:
        inputs["prompt"] = str(data["task"])

    return inputs


@dataclasses.dataclass(frozen=True)
class DualFrankaAbsStateBaseDeltaHighLumosInputs(transforms.DataTransformFn):
    model_type: _model.ModelType
    action_dim: int
    action_horizon: int

    def __call__(self, data: dict) -> dict:
        return _dual_franka_absstate_basedelta_inputs(
            data,
            model_type=self.model_type,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            mask_third_view=False,
        )


@dataclasses.dataclass(frozen=True)
class DualFrankaAbsStateBaseDeltaHighRSInputs(transforms.DataTransformFn):
    model_type: _model.ModelType
    action_dim: int
    action_horizon: int

    def __call__(self, data: dict) -> dict:
        return _dual_franka_absstate_basedelta_inputs(
            data,
            model_type=self.model_type,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            mask_third_view=False,
        )


@dataclasses.dataclass(frozen=True)
class DualFrankaAbsStateBaseDeltaMaskLumosInputs(transforms.DataTransformFn):
    model_type: _model.ModelType
    action_dim: int
    action_horizon: int

    def __call__(self, data: dict) -> dict:
        return _dual_franka_absstate_basedelta_inputs(
            data,
            model_type=self.model_type,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            mask_third_view=True,
        )


@dataclasses.dataclass(frozen=True)
class DualFrankaAbsStateBaseDeltaMaskRSInputs(transforms.DataTransformFn):
    model_type: _model.ModelType
    action_dim: int
    action_horizon: int

    def __call__(self, data: dict) -> dict:
        return _dual_franka_absstate_basedelta_inputs(
            data,
            model_type=self.model_type,
            action_dim=self.action_dim,
            action_horizon=self.action_horizon,
            mask_third_view=True,
        )


@dataclasses.dataclass(frozen=True)
class DualFrankaAbsStateBaseDeltaOutputs(transforms.DataTransformFn):
    """Return model delta actions directly as 14D (delta pose + absolute gripper)."""

    def __call__(self, data: dict) -> dict:
        act = np.asarray(data["actions"], dtype=np.float32)

        l = act[..., :10]
        r = act[..., 10:20]
        l_delta_pose6 = mat_to_pose6(pose10d_to_mat(l[..., :9])).astype(np.float32)
        r_delta_pose6 = mat_to_pose6(pose10d_to_mat(r[..., :9])).astype(np.float32)
        l_grip_abs = l[..., 9:10].astype(np.float32)
        r_grip_abs = r[..., 9:10].astype(np.float32)

        l7 = np.concatenate([l_delta_pose6, l_grip_abs], axis=-1).astype(np.float32)
        r7 = np.concatenate([r_delta_pose6, r_grip_abs], axis=-1).astype(np.float32)
        return {"actions": np.concatenate([l7, r7], axis=-1).astype(np.float32)}
