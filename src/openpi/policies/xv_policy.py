import dataclasses
import einops
import numpy as np
from scipy.spatial.transform import Rotation as R

from openpi import transforms
from openpi.policies.pose_util import pose6_to_mat, mat_to_pose10d, mat_to_pose6,pos_rot_to_mat,pose10d_to_mat
from openpi.policies.pose_repr_util import convert_pose_mat_rep
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class XVInputs(transforms.DataTransformFn):

    model_type: _model.ModelType
    action_dim: int                  # should be 11 for your setup
    action_horizon: int              # should match model_config.action_horizon
    action_down_sample_steps: int = 3

    image_down_sample_steps: tuple[int, ...] = (3, 15)

    def __call__(self, data: dict) -> dict:
    
        # --------------------------
        # 0) gather obs (single or sequence)
        # Expect LeRobot raw keys from your conversion:
        # image, eef_pos, eef_rot_axis_angle, gripper_width, demo_start_pose, actions(7D) :contentReference[oaicite:5]{index=5}
        # --------------------------
        img = data.get("image", None)
        img = _parse_image(img)

        eef_pos = np.asarray(data["eef_pos"], np.float32)                  # (3,) 
        eef_rot = np.asarray(data["eef_rot_axis_angle"], np.float32)       # (3,) 
        gripper = np.asarray(data["gripper_width"], np.float32)            # (1,) 
        demo_start_pose = np.asarray(data["demo_start_pose"], np.float32)  # (6,)


        # --------------------------
        # 1) image
        # --------------------------
        wrist_image = img
        zeros = np.zeros_like(wrist_image)

        mask_padding = (self.model_type == _model.ModelType.PI0)
        inputs = {
            "image": {
                "base_0_rgb": zeros,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": zeros,
            },
            "image_mask": {
                "base_0_rgb": np.False_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_ ,
            },
        }

        # --------------------------
        # 2) build low-dim state like UMI dataset
        # 后面再补充其他state
        pos_cur = eef_pos
        rot_cur = eef_rot


        cur_pose_mat = pose6_to_mat(np.concatenate([pos_cur, rot_cur], axis=-1))  # (4,4)
        start_pose_mat = pose6_to_mat(demo_start_pose)  # (4,4)
        rel_obs_pose_mat = convert_pose_mat_rep(
            cur_pose_mat,
            start_pose_mat,
            pose_rep='relative',
            backward=False,
        )  # (4,4)
        rel_obs_pose10d = mat_to_pose10d(rel_obs_pose_mat)  # (9,)
        rel_obs_rot6d = rel_obs_pose10d[3:]  # (6,)

        # inputs["state"] = rel_obs_rot6d  # model_transforms will PadStatesAndActions(action_dim)
        inputs["state"] = np.concatenate([pos_cur, rot_cur], axis=-1)

        # --------------------------
        # 3) action chunk: raw abs 7D -> relative pose10d + gripper (H,11)
        #    raw actions are [pos3, rotvec3, gripper] in your dataset :contentReference[oaicite:8]{index=8}
        # --------------------------
        raw_actions = data.get("actions", None)
        if raw_actions is not None:
            # Make it (H,7)
            if raw_actions.ndim == 1:
                # single-step fallback: repeat to horizon, all pad=True except first
                raw_actions = raw_actions[None, :]
                pad_mask = np.ones((self.action_horizon,), dtype=np.bool_)
                pad_mask[0] = False
                raw_actions = np.repeat(raw_actions, self.action_horizon, axis=0)
            else:
                # assume already a sequence (T,7); pad/truncate to action_horizon
                pad_mask = np.zeros((self.action_horizon,), dtype=np.bool_)
                T = raw_actions.shape[0]
                if T < self.action_horizon:
                    pad_mask[T:] = True
                    raw_actions = np.concatenate(
                        [raw_actions, np.repeat(raw_actions[-1:], self.action_horizon - T, axis=0)], axis=0
                    )
                else:
                    raw_actions = raw_actions[: self.action_horizon]

            # 下一步的位置
            action_mat = pose6_to_mat(np.concatenate([raw_actions[:, :3], raw_actions[:, 3:6]], axis=-1))  # (H,4,4)
            action_pose_mat = convert_pose_mat_rep(
                action_mat,
                cur_pose_mat,
                pose_rep='relative',
                backward=False,
            )  # (H,4,4)

            action_pose = mat_to_pose10d(action_pose_mat)  
            action_gripper = raw_actions[:, 6:7]  # (H,1)
            # 实际上 10 维
            act11 = np.concatenate([action_pose, action_gripper], axis=-1).astype(np.float32)

            # inputs["actions"] = act11
            inputs["actions"] = raw_actions

                # --------------------------
        # 4) prompt
        # --------------------------
        if "prompt" in data:
            inputs["prompt"] = str(data["prompt"])
        elif "task" in data:
            inputs["prompt"] = str(data["task"])

        
        return inputs
    
@dataclasses.dataclass(frozen=True)
class XVOutputs(transforms.DataTransformFn):
    """
    Inference-only:
    model output (H,11):
      [dx, dy, dz, rot6d(6), gripper]
    where dx,drot are in EEF/body frame (because training used T_rel = inv(T_cur) @ T_t)
    ->
    franka delta action (H,7) in BASE frame:
      [dx, dy, dz, drx, dry, drz, gripper]
    """

    def __call__(self, data: dict) -> dict:
        act = np.asarray(data["actions"], dtype=np.float32)  # (H,11)
        # ---- gripper ----
        # gripper = act[..., 9:10].astype(np.float32) / 88 # (H,1)

        # # =========================================================
        # # Convert EEF-frame delta -> BASE-frame delta for Franka
        # # Need current end-effector rotation R_cur (base <- eef)
        # # Get it from state: [pos(3), rotvec_cur(3), ...]
        # # =========================================================

        # pose10d = act[..., :9]   # (H,9)

        # mat = pose10d_to_mat(pose10d)
        # pose6d = mat_to_pose6(mat)
        # action_7d = np.concatenate([pose6d,gripper], axis=-1).astype(np.float32)

        # return {"actions": action_7d}        # (H,7)
        return {"actions": act}
