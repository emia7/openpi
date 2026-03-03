#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import rosbag
import imageio.v3 as iio
import cv2
from cv_bridge import CvBridge

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial.transform import Rotation as R


# =========================
# CONFIG (edit topics here)
# =========================
CONFIG: Dict[str, Any] = {
    # Left hand (xv device)
    "left": {
        "serial": "250801DR48FP25002268",
        "image_topic": "/xv_sdk/{serial}/color_camera/image_10hz",
        "pose_topic":  "/xv_sdk/{serial}/slam/pose",
        "clamp_topic": "/xv_sdk/{serial}/clamp/Data",
    },
    # Right hand (xv device)
    "right": {
        "serial": "250801DR48FP25002565",
        "image_topic": "/xv_sdk/{serial}/color_camera/image_10hz",
        "pose_topic":  "/xv_sdk/{serial}/slam/pose",
        "clamp_topic": "/xv_sdk/{serial}/clamp/Data",
    },
    # Third view (D435)
    "third": {
        # use compressed_10hz if you recorded it; otherwise switch to "/camera/color/image_raw/compressed"
        "image_topic": "/camera/color/image_raw/compressed_10hz",
    },

    # Timestamp rule: use header if looks like epoch; else bag time
    "header_stamp_valid_epoch_sec": 1.5e9,

    # If timestamps are weird, fallback fps
    "fallback_fps": 10.0,

    # Plots
    "make_plots": True,
    "plots": {
        "unwrap_euler": True,
        "dpi": 120,
        "alignment_hist_bins": 80,
    },
}


# =========================
# Utilities
# =========================
def stamp_to_sec(stamp) -> float:
    if hasattr(stamp, "secs") and hasattr(stamp, "nsecs"):  # ROS1
        return float(stamp.secs) + float(stamp.nsecs) * 1e-9
    if hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9
    return float(stamp)


def get_valid_timestamp(msg, bag_time_sec: float, valid_epoch: float) -> float:
    header_ts = 0.0
    if hasattr(msg, "header"):
        try:
            header_ts = stamp_to_sec(msg.header.stamp)
        except Exception:
            header_ts = 0.0
    return header_ts if header_ts > valid_epoch else bag_time_sec


def estimate_fps(ts: np.ndarray, default: float) -> float:
    if len(ts) < 3:
        return float(default)
    d = np.diff(ts)
    d = d[d > 0]
    if len(d) == 0:
        return float(default)
    return float(1.0 / np.median(d))


def nearest_idx(q_ts: np.ndarray, ref_ts: np.ndarray) -> np.ndarray:
    if len(ref_ts) == 0:
        return np.zeros((len(q_ts),), dtype=np.int64)
    i = np.searchsorted(ref_ts, q_ts, side="left")
    i = np.clip(i, 0, len(ref_ts) - 1)
    j = np.clip(i - 1, 0, len(ref_ts) - 1)
    pick_j = np.abs(q_ts - ref_ts[j]) <= np.abs(q_ts - ref_ts[i])
    return np.where(pick_j, j, i).astype(np.int64)


def decode_image_msg(msg, bridge: CvBridge) -> Optional[np.ndarray]:
    # CompressedImage has .format and .data
    if hasattr(msg, "format") and hasattr(msg, "data"):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_bgr is None:
                return None
            return cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            return None
    # Raw Image
    try:
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        return np.asarray(cv_img, dtype=np.uint8)
    except Exception:
        return None


def pose7_from_msg(msg) -> np.ndarray:
    ps = msg.poseMsg
    p = ps.pose.position
    q = ps.pose.orientation
    return np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float32)


# =========================
# Plotting
# =========================
def quaternion_to_euler(quat_xyzw: np.ndarray) -> np.ndarray:
    return R.from_quat(quat_xyzw).as_euler("xyz")


def plot_states(pose_data: np.ndarray, clamp_data: np.ndarray, output_path: str, unwrap_euler: bool = True):
    T = int(len(pose_data))
    steps = np.arange(T, dtype=np.int64)

    pos = pose_data[:, :3] if T > 0 else np.zeros((0, 3), dtype=np.float32)
    if T > 0:
        euler = np.array([quaternion_to_euler(q) for q in pose_data[:, 3:7]], dtype=np.float64)
        if unwrap_euler:
            for i in range(3):
                euler[:, i] = np.unwrap(euler[:, i])
    else:
        euler = np.zeros((0, 3), dtype=np.float64)

    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

    pos_labels = ["X (m)", "Y (m)", "Z (m)"]
    rot_labels = ["Roll (rad)", "Pitch (rad)", "Yaw (rad)"]

    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(steps, pos[:, i], linewidth=1.5)
        ax.set_title(f"Position {pos_labels[i]}")
        ax.set_xlabel("Frame")
        ax.grid(True, alpha=0.3)

    for i in range(3):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(steps, euler[:, i], linewidth=1.5)
        ax.set_title(f"Rotation {rot_labels[i]}")
        ax.set_xlabel("Frame")
        ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, :])
    ax.plot(steps, clamp_data, linewidth=1.5, label="Clamp")
    ax.set_title("Clamp / Gripper State")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Value")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Trajectory States ({T} frames)", fontsize=14)
    plt.savefig(output_path, dpi=CONFIG["plots"]["dpi"], bbox_inches="tight")
    plt.close()


def plot_alignment(out_path: str,
                   left_ts: np.ndarray, right_ts: np.ndarray, third_ts: np.ndarray,
                   left_pose_ts: np.ndarray, left_clamp_ts: np.ndarray,
                   right_pose_ts: np.ndarray, right_clamp_ts: np.ndarray,
                   bins: int = 80):
    def align_err(frame_ts: np.ndarray, ref_ts: np.ndarray) -> np.ndarray:
        if len(frame_ts) == 0 or len(ref_ts) == 0:
            return np.zeros((0,), dtype=np.float64)
        idx = nearest_idx(frame_ts, ref_ts)
        return np.abs(frame_ts - ref_ts[idx])

    def nearest_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if len(a) == 0 or len(b) == 0:
            return np.zeros((0,), dtype=np.float64)
        idx = nearest_idx(a, b)
        return (a - b[idx])

    lr = nearest_delta(left_ts, right_ts)
    lt = nearest_delta(left_ts, third_ts)
    rt = nearest_delta(right_ts, third_ts)

    l_pose_err = align_err(left_ts, left_pose_ts)
    l_clamp_err = align_err(left_ts, left_clamp_ts)
    r_pose_err = align_err(right_ts, right_pose_ts)
    r_clamp_err = align_err(right_ts, right_clamp_ts)

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(lr * 1000.0, label="Left-Right (ms)")
    ax.plot(lt * 1000.0, label="Left-Third (ms)")
    ax.plot(rt * 1000.0, label="Right-Third (ms)")
    ax.set_title("Cross-view nearest timestamp deltas")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Delta (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = fig.add_subplot(gs[0, 1])
    if len(lr) > 0: ax.hist(lr * 1000.0, bins=bins, alpha=0.6, label="L-R")
    if len(lt) > 0: ax.hist(lt * 1000.0, bins=bins, alpha=0.6, label="L-Third")
    if len(rt) > 0: ax.hist(rt * 1000.0, bins=bins, alpha=0.6, label="R-Third")
    ax.set_title("Histogram of cross-view deltas (ms)")
    ax.set_xlabel("Delta (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = fig.add_subplot(gs[1, 0])
    if len(l_pose_err) > 0: ax.plot(l_pose_err * 1000.0, label="Left pose err (ms)")
    if len(l_clamp_err) > 0: ax.plot(l_clamp_err * 1000.0, label="Left clamp err (ms)")
    ax.set_title("Left alignment errors (nearest)")
    ax.set_xlabel("Left frame index")
    ax.set_ylabel("Abs error (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = fig.add_subplot(gs[1, 1])
    if len(r_pose_err) > 0: ax.plot(r_pose_err * 1000.0, label="Right pose err (ms)")
    if len(r_clamp_err) > 0: ax.plot(r_clamp_err * 1000.0, label="Right clamp err (ms)")
    ax.set_title("Right alignment errors (nearest)")
    ax.set_xlabel("Right frame index")
    ax.set_ylabel("Abs error (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = fig.add_subplot(gs[2, :])
    if len(left_ts) > 0: ax.plot(left_ts - left_ts[0], np.zeros_like(left_ts), ".", label="Left frames")
    if len(right_ts) > 0: ax.plot(right_ts - right_ts[0], np.ones_like(right_ts), ".", label="Right frames")
    if len(third_ts) > 0: ax.plot(third_ts - third_ts[0], 2*np.ones_like(third_ts), ".", label="Third frames")
    ax.set_title("Frame timelines (relative seconds)")
    ax.set_xlabel("t - t0 (sec)")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Left", "Right", "Third"])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    plt.suptitle("Alignment diagnostics", fontsize=14)
    plt.savefig(out_path, dpi=CONFIG["plots"]["dpi"], bbox_inches="tight")
    plt.close()


# =========================
# Conversion per bag
# =========================
def fmt_topic(tmpl: str, serial: str) -> str:
    return tmpl.format(serial=serial)


def read_bag_one(bag_path: Path) -> Dict[str, Any]:
    cfg = CONFIG
    valid_epoch = float(cfg["header_stamp_valid_epoch_sec"])
    bridge = CvBridge()

    left = cfg["left"]
    right = cfg["right"]
    third = cfg["third"]

    left_img = fmt_topic(left["image_topic"], left["serial"])
    left_pose = fmt_topic(left["pose_topic"], left["serial"])
    left_clamp = fmt_topic(left["clamp_topic"], left["serial"])

    right_img = fmt_topic(right["image_topic"], right["serial"])
    right_pose = fmt_topic(right["pose_topic"], right["serial"])
    right_clamp = fmt_topic(right["clamp_topic"], right["serial"])

    third_img = third["image_topic"]

    topics = [left_img, left_pose, left_clamp, right_img, right_pose, right_clamp, third_img]

    left_frames: List[Tuple[float, np.ndarray]] = []
    right_frames: List[Tuple[float, np.ndarray]] = []
    third_frames: List[Tuple[float, np.ndarray]] = []
    left_poses: List[Tuple[float, np.ndarray]] = []
    right_poses: List[Tuple[float, np.ndarray]] = []
    left_clamps: List[Tuple[float, float]] = []
    right_clamps: List[Tuple[float, float]] = []

    with rosbag.Bag(str(bag_path), "r") as bag:
        for topic, msg, t in bag.read_messages(topics=topics):
            bag_ts = t.to_sec()
            ts = get_valid_timestamp(msg, bag_ts, valid_epoch)

            if topic == left_img:
                img = decode_image_msg(msg, bridge)
                if img is not None:
                    left_frames.append((ts, img))
            elif topic == right_img:
                img = decode_image_msg(msg, bridge)
                if img is not None:
                    right_frames.append((ts, img))
            elif topic == third_img:
                img = decode_image_msg(msg, bridge)
                if img is not None:
                    third_frames.append((ts, img))
            elif topic == left_pose:
                try:
                    left_poses.append((ts, pose7_from_msg(msg)))
                except Exception:
                    pass
            elif topic == right_pose:
                try:
                    right_poses.append((ts, pose7_from_msg(msg)))
                except Exception:
                    pass
            elif topic == left_clamp:
                try:
                    left_clamps.append((ts, float(msg.data)))
                except Exception:
                    pass
            elif topic == right_clamp:
                try:
                    right_clamps.append((ts, float(msg.data)))
                except Exception:
                    pass

    # sort
    left_frames.sort(key=lambda x: x[0])
    right_frames.sort(key=lambda x: x[0])
    third_frames.sort(key=lambda x: x[0])
    left_poses.sort(key=lambda x: x[0])
    right_poses.sort(key=lambda x: x[0])
    left_clamps.sort(key=lambda x: x[0])
    right_clamps.sort(key=lambda x: x[0])

    return {
        "left_frames": left_frames,
        "right_frames": right_frames,
        "third_frames": third_frames,
        "left_poses": left_poses,
        "right_poses": right_poses,
        "left_clamps": left_clamps,
        "right_clamps": right_clamps,
    }


def align_pose_clamp_to_frames(frame_ts: np.ndarray,
                              poses: List[Tuple[float, np.ndarray]],
                              clamps: List[Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    T = len(frame_ts)

    pose_ts = np.array([t for t, _ in poses], dtype=np.float64)
    pose_vals = np.stack([v for _, v in poses], axis=0).astype(np.float32, copy=False) if len(poses) else np.zeros((0, 7), dtype=np.float32)

    clamp_ts = np.array([t for t, _ in clamps], dtype=np.float64)
    clamp_vals = np.array([v for _, v in clamps], dtype=np.float32) if len(clamps) else np.zeros((0,), dtype=np.float32)

    if len(pose_ts) > 0 and len(pose_vals) > 0:
        idx = nearest_idx(frame_ts, pose_ts)
        aligned_pose = pose_vals[idx]
        err_p = float(np.max(np.abs(frame_ts - pose_ts[idx]))) if T > 0 else 0.0
    else:
        aligned_pose = np.zeros((T, 7), dtype=np.float32)
        aligned_pose[:, 6] = 1.0
        err_p = -1.0

    if len(clamp_ts) > 0 and len(clamp_vals) > 0:
        idx = nearest_idx(frame_ts, clamp_ts)
        aligned_clamp = clamp_vals[idx].astype(np.float32, copy=False)
        err_c = float(np.max(np.abs(frame_ts - clamp_ts[idx]))) if T > 0 else 0.0
    else:
        aligned_clamp = np.zeros((T,), dtype=np.float32)
        err_c = -1.0

    return aligned_pose, aligned_clamp, {"max_pose_err_sec": err_p, "max_clamp_err_sec": err_c}


def write_mp4(mp4_path: Path, frames: List[np.ndarray], fps: float):
    iio.imwrite(mp4_path, frames, fps=float(fps))


def convert_one_bag(bag_path: Path, out_dir: Path, episode_idx: int):
    ep6 = f"{int(episode_idx):06d}"
    data = read_bag_one(bag_path)

    if len(data["left_frames"]) == 0 or len(data["right_frames"]) == 0 or len(data["third_frames"]) == 0:
        raise RuntimeError(f"Missing frames in bag {bag_path.name}: "
                           f"left={len(data['left_frames'])}, right={len(data['right_frames'])}, third={len(data['third_frames'])}")

    fallback_fps = float(CONFIG["fallback_fps"])
    make_plots = bool(CONFIG["make_plots"])

    # ---- Left ----
    left_ts = np.array([t for t, _ in data["left_frames"]], dtype=np.float64)
    left_fps = estimate_fps(left_ts, default=fallback_fps)
    left_imgs = [img for _, img in data["left_frames"]]
    left_pose_al, left_clamp_al, left_stats = align_pose_clamp_to_frames(left_ts, data["left_poses"], data["left_clamps"])

    left_mp4 = out_dir / f"episode_{ep6}_left.mp4"
    left_json = out_dir / f"episode_{ep6}_left.json"
    write_mp4(left_mp4, left_imgs, left_fps)

    left_records = [{
        "timestamp": float(left_ts[i]),
        "pose": left_pose_al[i].tolist(),
        "clamp": float(left_clamp_al[i]),
    } for i in range(len(left_ts))]
    left_json.write_text(json.dumps({
        "view": "left",
        "fps": float(left_fps),
        "num_frames": int(len(left_ts)),
        "alignment": left_stats,
        "records": left_records
    }, indent=2), encoding="utf-8")

    if make_plots:
        plot_states(left_pose_al, left_clamp_al, str(out_dir / f"episode_{ep6}_left_states.png"),
                    unwrap_euler=CONFIG["plots"]["unwrap_euler"])

    # ---- Right ----
    right_ts = np.array([t for t, _ in data["right_frames"]], dtype=np.float64)
    right_fps = estimate_fps(right_ts, default=fallback_fps)
    right_imgs = [img for _, img in data["right_frames"]]
    right_pose_al, right_clamp_al, right_stats = align_pose_clamp_to_frames(right_ts, data["right_poses"], data["right_clamps"])

    right_mp4 = out_dir / f"episode_{ep6}_right.mp4"
    right_json = out_dir / f"episode_{ep6}_right.json"
    write_mp4(right_mp4, right_imgs, right_fps)

    right_records = [{
        "timestamp": float(right_ts[i]),
        "pose": right_pose_al[i].tolist(),
        "clamp": float(right_clamp_al[i]),
    } for i in range(len(right_ts))]
    right_json.write_text(json.dumps({
        "view": "right",
        "fps": float(right_fps),
        "num_frames": int(len(right_ts)),
        "alignment": right_stats,
        "records": right_records
    }, indent=2), encoding="utf-8")

    if make_plots:
        plot_states(right_pose_al, right_clamp_al, str(out_dir / f"episode_{ep6}_right_states.png"),
                    unwrap_euler=CONFIG["plots"]["unwrap_euler"])

    # ---- Third (mp4 only, no json) ----
    third_ts = np.array([t for t, _ in data["third_frames"]], dtype=np.float64)
    third_fps = estimate_fps(third_ts, default=fallback_fps)
    third_imgs = [img for _, img in data["third_frames"]]
    third_mp4 = out_dir / f"episode_{ep6}_third.mp4"
    write_mp4(third_mp4, third_imgs, third_fps)

    # ---- Alignment plot ----
    if make_plots:
        left_pose_ts = np.array([t for t, _ in data["left_poses"]], dtype=np.float64)
        left_clamp_ts = np.array([t for t, _ in data["left_clamps"]], dtype=np.float64)
        right_pose_ts = np.array([t for t, _ in data["right_poses"]], dtype=np.float64)
        right_clamp_ts = np.array([t for t, _ in data["right_clamps"]], dtype=np.float64)

        plot_alignment(
            out_path=str(out_dir / f"episode_{ep6}_alignment.png"),
            left_ts=left_ts, right_ts=right_ts, third_ts=third_ts,
            left_pose_ts=left_pose_ts, left_clamp_ts=left_clamp_ts,
            right_pose_ts=right_pose_ts, right_clamp_ts=right_clamp_ts,
            bins=int(CONFIG["plots"]["alignment_hist_bins"]),
        )

    return {
        "episode": ep6,
        "left_fps": float(left_fps),
        "right_fps": float(right_fps),
        "third_fps": float(third_fps),
        "left_align": left_stats,
        "right_align": right_stats,
    }


# =========================
# CLI: minimal args for batch conversion
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="Convert all .bag in a folder to left/right MP4+JSON and third MP4.")
    p.add_argument("--bag_dir", required=True, help="Folder containing .bag files")
    p.add_argument("--out_dir", required=True, help="Output folder")
    p.add_argument("--start_idx", type=int, required=True, help="Starting episode index (e.g. 1)")
    p.add_argument("--pattern", default="*.bag", help="Bag filename pattern (default: *.bag)")
    return p.parse_args()


def main():
    args = parse_args()
    bag_dir = Path(args.bag_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bags = sorted(bag_dir.glob(args.pattern))
    if not bags:
        raise SystemExit(f"No bags found in {bag_dir} with pattern {args.pattern}")

    idx = int(args.start_idx)
    print(f"[INFO] Found {len(bags)} bag(s). start_idx={idx}")
    print(f"[INFO] Output dir: {out_dir}")

    for bag_path in bags:
        ep = f"{idx:06d}"
        print("\n============================================================")
        print(f"[INFO] Bag: {bag_path}")
        print(f"[INFO] Episode: {ep}")
        print("============================================================")

        summary = convert_one_bag(bag_path, out_dir, idx)
        print(f"[OK] episode_{summary['episode']} "
              f"left_fps={summary['left_fps']:.2f} right_fps={summary['right_fps']:.2f} third_fps={summary['third_fps']:.2f} "
              f"left_pose_err={summary['left_align']['max_pose_err_sec']:.6f}s right_pose_err={summary['right_align']['max_pose_err_sec']:.6f}s")
        idx += 1


if __name__ == "__main__":
    main()
