#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Render a moving coordinate triad (xyz axes) following pose trajectory and export to MP4.

Expected JSON:
{
  "fps": <float, optional>,
  "records": [
    {"pose": [x,y,z,qx,qy,qz,qw], "timestamp": <float, optional>, ...},
    ...
  ]
}

Output:
- MP4 animation with:
  - 3D trajectory
  - Triad attached to the pose (position + orientation)

Usage:
  python render_triad_mp4.py --json episode0100.json --out pose_triad.mp4
  python render_triad_mp4.py --json episode0100.json --out pose_triad.mp4 --stride 3
  python render_triad_mp4.py --json episode0100.json --out pose_triad.mp4 --speed 2.0
  python render_triad_mp4.py --json episode0100.json --out pose_triad.mp4 --no_traj_tail
'''

import argparse
import json
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.spatial.transform import Rotation as R
except Exception as e:
    raise RuntimeError("scipy is required for quaternion->rotation conversion") from e

import imageio.v2 as imageio


def load_episode(json_path: Path):
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    records = meta.get("records", [])
    if not records:
        raise ValueError(f"No 'records' found in {json_path}")
    poses = np.asarray([r["pose"] for r in records], dtype=np.float32)  # (T,7)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"Expected pose shape (T,7), got {poses.shape}")
    fps = float(meta.get("fps", 0.0)) if meta.get("fps", None) is not None else 0.0
    ts = np.asarray([r.get("timestamp", np.nan) for r in records], dtype=np.float64)
    return poses, fps, ts


def pick_fps(T: int, fps_meta: float, ts: np.ndarray, stride: int, speed: float):
    # Use timestamp-derived fps if possible; else fps_meta; else default 30.
    fps = None
    if ts.size == T and np.isfinite(ts).sum() >= 5:
        t = ts[np.isfinite(ts)]
        dt = np.median(np.diff(t))
        if dt > 1e-6:
            fps = 1.0 / dt
    if fps is None and fps_meta > 0:
        fps = fps_meta
    if fps is None:
        fps = 30.0
    fps = fps / max(1, stride)
    fps = fps * float(speed)
    # Keep within a reasonable range for MP4 encoders
    fps = float(np.clip(fps, 5.0, 60.0))
    return fps


def set_equal_3d(ax, xyz: np.ndarray, pad: float = 0.05):
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    ranges = maxs - mins
    max_range = float(np.max(ranges))
    if max_range <= 0:
        max_range = 1.0
    center = (mins + maxs) / 2.0
    half = max_range / 2.0
    half *= (1.0 + pad)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def fig_to_rgb(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return buf.reshape(h, w, 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json", type=str, required=True, help="Path to episode JSON")
    p.add_argument("--out", type=str, required=True, help="Output MP4 path")
    p.add_argument("--stride", type=int, default=1, help="Downsample by taking every Nth record")
    p.add_argument("--speed", type=float, default=1.0, help="Playback speed factor (e.g., 2.0 faster)")
    p.add_argument("--triad_len", type=float, default=None, help="Triad axis length in meters (default: 10%% of bbox)")
    p.add_argument("--dpi", type=int, default=130, help="Render DPI")
    p.add_argument("--width", type=int, default=960, help="Frame width (pixels)")
    p.add_argument("--height", type=int, default=720, help="Frame height (pixels)")
    p.add_argument("--no_traj_tail", action="store_true", help="Do not draw trajectory tail")
    p.add_argument("--tail_len", type=int, default=200, help="How many last points to show in tail")
    args = p.parse_args()

    json_path = Path(args.json)
    poses, fps_meta, ts = load_episode(json_path)

    stride = max(1, int(args.stride))
    poses = poses[::stride]
    ts = ts[::stride]

    pos = poses[:, 0:3]
    quat = poses[:, 3:7]
    # normalize
    qn = np.linalg.norm(quat, axis=1, keepdims=True)
    qn[qn == 0] = 1.0
    quat = quat / qn

    T = poses.shape[0]
    fps_out = pick_fps(T, fps_meta, ts, stride, args.speed)

    # Determine triad length
    if args.triad_len is None:
        bbox = np.ptp(pos, axis=0)
        scale = float(np.max(bbox))
        if not np.isfinite(scale) or scale <= 0:
            scale = 0.1
        triad_len = 0.10 * scale
    else:
        triad_len = float(args.triad_len)

    # Setup figure
    plt.ioff()
    fig = plt.figure(figsize=(args.width / args.dpi, args.height / args.dpi), dpi=args.dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(f"Pose triad replay ({json_path.name})")
    set_equal_3d(ax, pos, pad=0.10)

    # Static full trajectory (lightweight)
    if args.no_traj_tail:
        traj_line, = ax.plot(pos[:, 0], pos[:, 1], pos[:, 2])
        tail_line = None
    else:
        traj_line = None
        tail_line, = ax.plot([], [], [])  # updated tail

    # Triad: 3 axis lines from current position
    x_line, = ax.plot([], [], [])
    y_line, = ax.plot([], [], [])
    z_line, = ax.plot([], [], [])
    # Current point marker
    cur_pt = ax.scatter([], [], [])

    # Render & write
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out_path),
        fps=fps_out,
        format="FFMPEG",                 # ✅ 强制使用 ffmpeg
        codec="libx264",
        ffmpeg_params=[
            "-pix_fmt", "yuv420p",       # ✅ Windows 播放器最兼容
            "-movflags", "+faststart",
            "-profile:v", "baseline",
            "-level", "3.0",
        ],
    )
    import imageio_ffmpeg
    print("Using ffmpeg:", imageio_ffmpeg.get_ffmpeg_exe(), flush=True)
    print("Output:", out_path, flush=True)
    print("Frames:", T, "fps_out:", fps_out, flush=True)



    try:
        for i in range(T):
            p0 = pos[i]
            rot = R.from_quat(quat[i]).as_matrix()  # 3x3, columns are axis in world frame
            ex = rot[:, 0] * triad_len
            ey = rot[:, 1] * triad_len
            ez = rot[:, 2] * triad_len

            # Update triad lines (as simple arrows via line segments)
            x_line.set_data([p0[0], p0[0] + ex[0]], [p0[1], p0[1] + ex[1]])
            x_line.set_3d_properties([p0[2], p0[2] + ex[2]])

            y_line.set_data([p0[0], p0[0] + ey[0]], [p0[1], p0[1] + ey[1]])
            y_line.set_3d_properties([p0[2], p0[2] + ey[2]])

            z_line.set_data([p0[0], p0[0] + ez[0]], [p0[1], p0[1] + ez[1]])
            z_line.set_3d_properties([p0[2], p0[2] + ez[2]])

            # Update tail
            if tail_line is not None:
                j0 = max(0, i - int(args.tail_len))
                tail = pos[j0:i+1]
                tail_line.set_data(tail[:, 0], tail[:, 1])
                tail_line.set_3d_properties(tail[:, 2])

            # Update current point (re-create offsets for 3D scatter)
            cur_pt._offsets3d = ([p0[0]], [p0[1]], [p0[2]])

            frame = fig_to_rgb(fig)
            writer.append_data(frame)
    finally:
        writer.close()
        plt.close(fig)

    print(f"Saved MP4: {out_path}  (frames={T}, fps={fps_out:.2f}, stride={stride}, triad_len={triad_len:.4f}m)")


if __name__ == "__main__":
    main()
