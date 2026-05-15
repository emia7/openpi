"""
LeRobot 样本 → openpi ``pi05_dual_franka_finetune_*`` 的 **RepackTransform 输入**。

针对外部「三相机 + 68D state」格式（与 ``dual_scripts/convert_external_to_intermediate.py`` 一致）：
  - ``image`` → ``observation.images.right_wrist_lumos``
  - ``extra_view_image-0`` → ``observation.images.third_d455``
  - ``extra_view_image-1`` → ``observation.images.left_wrist_lumos``

state 前 20 维布局与 ``extract_state_from_68d`` 相同（含左右 swap）。

**重要**：dual_franka 的 Repack 使用圆点路径字符串（如 ``observation.images.third_d455``）作为
``flatten_dict`` 后的键；须输出 **扁平圆点键**，不能只用深度嵌套 dict（否则 slash 展开后与 Repack 不匹配）。
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from scipy.spatial.transform import Rotation as R
except ImportError as e:
    raise ImportError("robot_dataset_adapters 需要 scipy") from e


def _to_numpy_img(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        arr = x
    else:
        try:
            import torch

            if isinstance(x, torch.Tensor):
                arr = x.detach().cpu().numpy()
            else:
                arr = np.asarray(x)
        except Exception:
            arr = np.asarray(x)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    return np.asarray(arr)


def rot6d_to_quat(rot6d: np.ndarray) -> np.ndarray:
    """6D rotation → 四元数 [qx,qy,qz,qw]。"""
    rot6d = np.asarray(rot6d, dtype=np.float64).reshape(-1)
    a1, a2 = rot6d[:3], rot6d[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    rot_mat = np.stack([b1, b2, b3], axis=1)
    quat = R.from_matrix(rot_mat).as_quat()
    return quat.astype(np.float32)


def extract_state_from_68d(state_68: np.ndarray) -> dict[str, np.ndarray]:
    """
    与 ``dual_scripts/convert_external_to_intermediate.py`` 保持一致；
    返回 swap 后的 ``left_ee_pos_quat`` / ``right_ee_pos_quat`` / ``gripper_pose``。
    """
    state = np.asarray(state_68, dtype=np.float32).reshape(-1)
    if state.shape[0] < 20:
        raise ValueError(f"state 维度不足 20，当前 {state.shape[0]}")
    s = state[:20]
    ext_l_grip = float(s[0])
    ext_r_grip = float(s[1])
    ext_l_xyz = s[2:5]
    ext_l_rot6d = s[5:11]
    ext_r_xyz = s[11:14]
    ext_r_rot6d = s[14:20]
    ext_l_quat = rot6d_to_quat(ext_l_rot6d)
    ext_r_quat = rot6d_to_quat(ext_r_rot6d)
    left_ee_pos_quat = np.concatenate([ext_r_xyz, ext_r_quat]).astype(np.float32)
    right_ee_pos_quat = np.concatenate([ext_l_xyz, ext_l_quat]).astype(np.float32)
    gripper_pose = np.array([ext_r_grip, ext_l_grip], dtype=np.float32)
    return {
        "left_ee_pos_quat": left_ee_pos_quat,
        "right_ee_pos_quat": right_ee_pos_quat,
        "gripper_pose": gripper_pose,
    }


def _pose7_quat_to_pose6(pos_quat7: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[xyz + qx,qy,qz,qw] -> pos3, rotvec3。"""
    p = np.asarray(pos_quat7, dtype=np.float32).reshape(7)
    pos = p[:3]
    quat = p[3:7].astype(np.float64)
    rotvec = R.from_quat(quat).as_rotvec().astype(np.float32)
    return pos, rotvec


def _get_cam(sample: dict, *names: str) -> np.ndarray | None:
    for n in names:
        if n in sample:
            return _to_numpy_img(sample[n])
    return None


# 与 convert_external_to_intermediate.CAMERA_MAPPING 一致
DEFAULT_CAMERA_MAP_TO_PI_SLOTS = {
    "image": "right_wrist_lumos",
    "extra_view_image-0": "third_d455",
    "extra_view_image-1": "left_wrist_lumos",
}


def adapt_external_three_cam_state68_to_dual_franka_flat(
    sample: dict,
    *,
    camera_map: dict[str, str] | None = None,
) -> dict:
    """
    单帧样本 → **扁平圆点路径键**，可直接喂 ``policy_robot._input_transform``。

    若无 ``left_action``/``right_action``，尝试 ``actions`` 前 14 维拆成 7+7；否则填零。
    """
    cam_map = camera_map or DEFAULT_CAMERA_MAP_TO_PI_SLOTS
    out: dict[str, Any] = {}

    for src_key, slot in cam_map.items():
        img = _get_cam(
            sample,
            src_key,
            src_key.replace("-", "_"),
            src_key.replace("-", ""),
        )
        if img is None:
            raise KeyError(f"样本缺少图像键（尝试过 {src_key}）：可用键 {sorted(sample.keys())[:40]}...")
        out[f"observation.images.{slot}"] = img

    st = extract_state_from_68d(sample["state"])
    l7 = st["left_ee_pos_quat"]
    r7 = st["right_ee_pos_quat"]
    gl, gr = float(st["gripper_pose"][0]), float(st["gripper_pose"][1])

    lp, lrv = _pose7_quat_to_pose6(l7)
    rp, rrv = _pose7_quat_to_pose6(r7)

    demo_l = np.concatenate([lp, lrv], axis=-1).astype(np.float32)
    demo_r = np.concatenate([rp, rrv], axis=-1).astype(np.float32)

    out["observation.state.left_eef_pos"] = lp.astype(np.float32)
    out["observation.state.left_eef_rotvec"] = lrv.astype(np.float32)
    out["observation.state.left_gripper"] = np.array([gl], dtype=np.float32)
    out["observation.state.right_eef_pos"] = rp.astype(np.float32)
    out["observation.state.right_eef_rotvec"] = rrv.astype(np.float32)
    out["observation.state.right_gripper"] = np.array([gr], dtype=np.float32)
    out["observation.state.demo_start_pose_left"] = demo_l
    out["observation.state.demo_start_pose_right"] = demo_r

    la = sample.get("left_action")
    ra = sample.get("right_action")
    if la is None or ra is None:
        act = sample.get("actions")
        if act is not None:
            act = np.asarray(act, dtype=np.float32).reshape(-1)
            if act.shape[0] >= 14:
                la = act[:7].astype(np.float32)
                ra = act[7:14].astype(np.float32)
        if la is None:
            la = np.zeros(7, dtype=np.float32)
        if ra is None:
            ra = np.zeros(7, dtype=np.float32)
    else:
        la = np.asarray(la, dtype=np.float32).reshape(7)
        ra = np.asarray(ra, dtype=np.float32).reshape(7)

    out["left_action"] = la
    out["right_action"] = ra

    task = sample.get("task", "")
    if hasattr(task, "item"):
        task = task.item()
    out["task"] = str(task) if task is not None else ""

    return out


ADAPTER_REGISTRY = {
    "external_three_cam_state68": adapt_external_three_cam_state68_to_dual_franka_flat,
}


def get_robot_adapter(name: str | None):
    if not name:
        return None
    if name not in ADAPTER_REGISTRY:
        raise ValueError(f"未知 robot-repo-adapter: {name!r}；可选: {list(ADAPTER_REGISTRY)}")
    return ADAPTER_REGISTRY[name]
