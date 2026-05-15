import dataclasses
import einops
import numpy as np

from openpi import transforms
from openpi.policies.pose_util import (
    pose6_to_mat,
    mat_to_pose10d,     # NOTE: returns 9D: pos3 + rot6d6
    mat_to_pose6,       # returns 6D: pos3 + rotvec3
    pose10d_to_mat,     # NOTE: accepts 9D pose (pos3+rot6d6)
)
from openpi.policies.pose_repr_util import convert_pose_mat_rep
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    # allow CHW
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _ensure_horizon(x: np.ndarray, H: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Ensure x is (H, D). Return (x_H, pad_mask) where pad_mask is (H,) bool.
    If x is (D,), we repeat it to H and mark all padded except first.
    If x is (T, D), we pad/clip to H; padded steps repeat last row.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[None, :]
        pad_mask = np.ones((H,), dtype=np.bool_)
        pad_mask[0] = False
        xH = np.repeat(x, H, axis=0)
        return xH, pad_mask

    T = x.shape[0]
    pad_mask = np.zeros((H,), dtype=np.bool_)
    if T < H:
        pad_mask[T:] = True
        xH = np.concatenate([x, np.repeat(x[-1:], H - T, axis=0)], axis=0)
    else:
        xH = x[:H]
    return xH, pad_mask


def _pose7_to_pose6(pose7: np.ndarray) -> np.ndarray:
    """
    pose7: [x,y,z,qx,qy,qz,qw] -> pose6: [x,y,z,rx,ry,rz] (rotvec)
    We do NOT need quaternion math here because your dataset already stores rotvec in obs.
    This helper remains for compatibility if you ever switch.
    """
    raise NotImplementedError("This policy expects rotvec obs already (left_eef_rotvec/right_eef_rotvec).")


def abs_next_targets_to_policy20_row(
    left_eef_pose6: np.ndarray,
    right_eef_pose6: np.ndarray,
    left_action_abs7: np.ndarray,
    right_action_abs7: np.ndarray,
) -> np.ndarray:
    """
    One timestep of model supervision (20,), matching ``XVDualInputs`` for the first
    horizon row: absolute next-state (7D per hand) → relative pose9d+grip per hand.

    Current pose must come from ``left_eef_*`` / ``right_eef_*`` (pose6 = pos3+rotvec3),
    same as the action branch in ``XVDualInputs``.
    """
    l6 = np.asarray(left_eef_pose6, dtype=np.float32).reshape(6)
    r6 = np.asarray(right_eef_pose6, dtype=np.float32).reshape(6)
    la = np.asarray(left_action_abs7, dtype=np.float32).reshape(7)
    ra = np.asarray(right_action_abs7, dtype=np.float32).reshape(7)

    l_cur_mat = pose6_to_mat(np.concatenate([l6[:3], l6[3:6]], axis=-1))
    r_cur_mat = pose6_to_mat(np.concatenate([r6[:3], r6[3:6]], axis=-1))

    left_tgt_mat = pose6_to_mat(np.concatenate([la[:3], la[3:6]], axis=-1))
    right_tgt_mat = pose6_to_mat(np.concatenate([ra[:3], ra[3:6]], axis=-1))

    left_rel_mat = convert_pose_mat_rep(left_tgt_mat, l_cur_mat, pose_rep="relative", backward=False)
    right_rel_mat = convert_pose_mat_rep(right_tgt_mat, r_cur_mat, pose_rep="relative", backward=False)

    left_pose9 = mat_to_pose10d(left_rel_mat).astype(np.float32)
    right_pose9 = mat_to_pose10d(right_rel_mat).astype(np.float32)
    left_act10 = np.concatenate([left_pose9, la[6:7]], axis=-1)
    right_act10 = np.concatenate([right_pose9, ra[6:7]], axis=-1)
    return np.concatenate([left_act10, right_act10], axis=-1).astype(np.float32)


@dataclasses.dataclass(frozen=True)
class XVDualInputs(transforms.DataTransformFn):
    """
    Expect LeRobot raw keys (per frame):
      - left_view, right_view, third_view: uint8 HWC
      - left_eef_pos(3), left_eef_rotvec(3), left_gripper(1)
      - right_eef_pos(3), right_eef_rotvec(3), right_gripper(1)
      - (optional) left_action(7), right_action(7) OR sequences (T,7)
      - (optional) task or prompt
      - (optional) demo_start_pose_left(6), demo_start_pose_right(6)
        If not provided, we assume the demo-start pose equals current pose (relative=identity).
    Produces OpenPI model inputs:
      - image dict with base/left/right
      - state (14,) = left_rel_rot6d(6) + right_rel_rot6d(6) + left_grip(1) + right_grip(1)
      - actions (H,20) when training: per-hand (pose9d+grip) => 10D each
    """

    model_type: _model.ModelType
    action_dim: int
    action_horizon: int

    def __call__(self, data: dict) -> dict:
        # --------------------------
        # 1) Images
        # --------------------------
        third_img = _parse_image(data["third_view"])
        left_img = _parse_image(data["left_view"])
        right_img = _parse_image(data["right_view"])

        inputs = {
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

        # --------------------------
        # 2) Low-dim obs state (UMI-style inter-gripper proprioception)
        # --------------------------
        # New design: Use Foundation Pose based inter-gripper state
        # State: hands_rel_xyz (3D) + hands_rel_rot6d (6D) = 9D total
        
        # Check if new features are available (Foundation Pose based)
        if "hands_rel_xyz" in data and "hands_rel_rot6d" in data:
            # Use new inter-gripper state (preferred)
            hands_rel_xyz = np.asarray(data["hands_rel_xyz"], np.float32)      # (3,)
            hands_rel_rot6d = np.asarray(data["hands_rel_rot6d"], np.float32)  # (6,)
            
            # State: 9D (hands_rel_xyz + hands_rel_rot6d)
            state9 = np.concatenate([hands_rel_xyz, hands_rel_rot6d], axis=-1).astype(np.float32)
            inputs["state"] = state9
        else:
            # Fallback: Use legacy relative-to-demo-start rotation state (12D)
            l_pos = np.asarray(data["left_eef_pos"], np.float32)
            l_rot = np.asarray(data["left_eef_rotvec"], np.float32)
            r_pos = np.asarray(data["right_eef_pos"], np.float32)
            r_rot = np.asarray(data["right_eef_rotvec"], np.float32)
            
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
            
            # Legacy state: only rotations, no position info (12D)
            state12 = np.concatenate([l_rel_rot6, r_rel_rot6], axis=-1).astype(np.float32)
            inputs["state"] = state12  # model_transforms will pad to action_dim if needed


                # --------------------------
        # 3) action chunk:
        #    dataset raw actions are ABS next-state per-hand:
        #      left_action/right_action: [pos3, rotvec3, gripper] (7D)
        #    We convert to RELATIVE-to-current pose:
        #      pose9d = [pos3, rot6d6] (9D)  + gripper(1) => 10D per hand
        #    Final model actions: concat(left10, right10) => (H,20)
        # --------------------------
        raw_left = data.get("left_action", None)
        raw_right = data.get("right_action", None)

        if raw_left is not None and raw_right is not None:
            raw_left = np.asarray(raw_left, np.float32)
            raw_right = np.asarray(raw_right, np.float32)

            # ---- Make to (H,7) with pad_mask (same style as your reference) ----
            if raw_left.ndim == 1:
                raw_left = raw_left[None, :]
            if raw_right.ndim == 1:
                raw_right = raw_right[None, :]

            pad_mask = np.zeros((self.action_horizon,), dtype=np.bool_)

            T = min(raw_left.shape[0], raw_right.shape[0])
            if T < self.action_horizon:
                pad_mask[T:] = True

                raw_left = np.concatenate(
                    [raw_left[:T], np.repeat(raw_left[T-1:T], self.action_horizon - T, axis=0)], axis=0
                )
                raw_right = np.concatenate(
                    [raw_right[:T], np.repeat(raw_right[T-1:T], self.action_horizon - T, axis=0)], axis=0
                )
            else:
                raw_left = raw_left[: self.action_horizon]
                raw_right = raw_right[: self.action_horizon]

            # ---- Build absolute target mats from raw next-state (pos+rotvec) ----
            left_tgt_mat = pose6_to_mat(
                np.concatenate([raw_left[:, :3], raw_left[:, 3:6]], axis=-1)
            )  # (H,4,4)
            right_tgt_mat = pose6_to_mat(
                np.concatenate([raw_right[:, :3], raw_right[:, 3:6]], axis=-1)
            )  # (H,4,4)

            # ---- Convert ABS targets to REL wrt current pose (inv(T_cur) @ T_tgt) ----
            left_rel_mat = convert_pose_mat_rep(
                left_tgt_mat,
                l_cur_mat,
                pose_rep="relative",
                backward=False,
            )  # (H,4,4)

            right_rel_mat = convert_pose_mat_rep(
                right_tgt_mat,
                r_cur_mat,
                pose_rep="relative",
                backward=False,
            )  # (H,4,4)

            # ---- mat -> pose9d (pos3 + rot6d6) ----
            left_pose9 = mat_to_pose10d(left_rel_mat).astype(np.float32)   # (H,9)
            right_pose9 = mat_to_pose10d(right_rel_mat).astype(np.float32) # (H,9)

            # ---- append gripper ----
            left_grip = raw_left[:, 6:7].astype(np.float32)    # (H,1)
            right_grip = raw_right[:, 6:7].astype(np.float32)  # (H,1)

            left_act10 = np.concatenate([left_pose9, left_grip], axis=-1)    # (H,10)
            right_act10 = np.concatenate([right_pose9, right_grip], axis=-1) # (H,10)

            inputs["actions"] = np.concatenate([left_act10, right_act10], axis=-1).astype(np.float32)  # (H,20)

        # --------------------------
        # 4) Prompt
        # --------------------------
        if "prompt" in data:
            inputs["prompt"] = str(data["prompt"])
        elif "task" in data:
            inputs["prompt"] = str(data["task"])

        # Passthrough LeRobot row indices for offline analysis (ignored by model forward).
        for _k in ("episode_index", "frame_index"):
            if _k in data:
                _v = np.asarray(data[_k])
                if _v.size >= 1:
                    inputs[_k] = int(_v.reshape(-1)[0])

        return inputs


@dataclasses.dataclass(frozen=True)
class XVDualOutputs(transforms.DataTransformFn):
    """
    Inference-only transform.

    Model output expected: actions (H,20):
      left:  pose9d(9)+grip(1)  => 10
      right: pose9d(9)+grip(1)  => 10

    Convert per-hand pose9d -> pose6d (pos3 + rotvec3) using pose10d_to_mat + mat_to_pose6,
    then append gripper => 7D per hand.

    Returns:
      - actions: (H,14) = concat(left7, right7)
      - left_actions: (H,7)
      - right_actions: (H,7)
    """

    def __call__(self, data: dict) -> dict:
        act = np.asarray(data["actions"], dtype=np.float32)  # (H,20) or (20,)

        if act.ndim == 1:
            act = act[None, :]

        if act.shape[-1] != 20:
            raise ValueError(f"Expected model actions last-dim=20, got {act.shape}")

        l = act[..., :10]   # (H,10)
        r = act[..., 10:20] # (H,10)

        l_pose9 = l[..., :9]      # (H,9)
        l_grip = l[..., 9:10]     # (H,1)
        l_grip = l_grip / 88.0  # 添加缩放
        r_pose9 = r[..., :9]
        r_grip = r[..., 9:10]
        r_grip = r_grip / 88.0  # 添加缩放
        # 在 xv_dual_policy.py XVDualOutputs 中
        # pose9 -> mat -> pose6 (pos3 + rotvec3)
        l_mat = pose10d_to_mat(l_pose9)     # (H,4,4)   (relative transform)
        r_mat = pose10d_to_mat(r_pose9)

        l_pose6 = mat_to_pose6(l_mat)       # (H,6)
        r_pose6 = mat_to_pose6(r_mat)

        l7 = np.concatenate([l_pose6, l_grip], axis=-1).astype(np.float32)  # (H,7)
        r7 = np.concatenate([r_pose6, r_grip], axis=-1).astype(np.float32)  # (H,7)

        both14 = np.concatenate([l7, r7], axis=-1).astype(np.float32)       # (H,14)

        return {
            "actions": both14,
            "left_actions": l7,
            "right_actions": r7,
        }
        