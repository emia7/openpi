from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


def pose7_to_pos_rotvec(pose7: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert [x,y,z,qx,qy,qz,qw] to (pos, rotvec)."""
    pos = pose7[:3].astype(np.float32)
    quat = pose7[3:7].astype(np.float32)
    rotvec = R.from_quat(quat).as_rotvec().astype(np.float32)
    return pos, rotvec


def load_episode_json(path: Path, *, default_fps: float, require_clamp: bool) -> tuple[np.ndarray, np.ndarray | None, float, int]:
    meta = json.loads(path.read_text(encoding="utf-8"))
    records = meta["records"]
    if len(records) < 2:
        raise ValueError(f"{path} has <2 records.")

    poses = np.asarray([r["pose"] for r in records], dtype=np.float32)
    has_clamp = "clamp" in records[0]
    if require_clamp and not has_clamp:
        raise ValueError(f"{path} missing required clamp field.")
    clamp = np.asarray([r["clamp"] for r in records], dtype=np.float32).reshape(-1, 1) if has_clamp else None

    fps = float(meta.get("fps", default_fps))
    return poses, clamp, fps, len(records)


def compute_stride(orig_fps: float, target_fps: float) -> tuple[int, float]:
    if target_fps <= 0:
        raise ValueError("target_fps must be > 0")
    stride = int(round(orig_fps / target_fps))
    stride = max(1, stride)
    achieved_fps = orig_fps / stride
    return stride, achieved_fps
