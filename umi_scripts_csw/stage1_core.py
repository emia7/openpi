from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import rosbag
from cv_bridge import CvBridge
from matplotlib.gridspec import GridSpec
from scipy.spatial.transform import Rotation as R


THREE_VIEW_CONFIG: dict[str, Any] = {
    "left": {
        "serial": "250801DR48FP25002268",
        "image_topic": "/xv_sdk/{serial}/color_camera/image_10hz",
        "pose_topic": "/xv_sdk/{serial}/slam/pose",
        "clamp_topic": "/xv_sdk/{serial}/clamp/Data",
    },
    "right": {
        "serial": "250801DR48FP25002565",
        "image_topic": "/xv_sdk/{serial}/color_camera/image_10hz",
        "pose_topic": "/xv_sdk/{serial}/slam/pose",
        "clamp_topic": "/xv_sdk/{serial}/clamp/Data",
    },
    "third": {
        "image_topic": "/camera/color/image_raw/compressed_10hz",
    },
    "header_stamp_valid_epoch_sec": 1.5e9,
    "fallback_fps": 10.0,
    "plots": {
        "unwrap_euler": True,
        "dpi": 120,
        "alignment_hist_bins": 80,
    },
}


def stamp_to_sec(stamp) -> float:
    if hasattr(stamp, "secs") and hasattr(stamp, "nsecs"):
        return float(stamp.secs) + float(stamp.nsecs) * 1e-9
    if hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9
    return float(stamp)


def pose7_from_msg(msg) -> np.ndarray:
    ps = msg.poseMsg
    p = ps.pose.position
    q = ps.pose.orientation
    return np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float32)


def nearest_idx(q_ts: np.ndarray, ref_ts: np.ndarray) -> np.ndarray:
    if len(ref_ts) == 0:
        return np.zeros((len(q_ts),), dtype=np.int64)
    i = np.searchsorted(ref_ts, q_ts, side="left")
    i = np.clip(i, 0, len(ref_ts) - 1)
    j = np.clip(i - 1, 0, len(ref_ts) - 1)
    pick_j = np.abs(q_ts - ref_ts[j]) <= np.abs(q_ts - ref_ts[i])
    return np.where(pick_j, j, i).astype(np.int64)


def estimate_fps(ts: np.ndarray, default: float) -> float:
    if len(ts) < 3:
        return float(default)
    d = np.diff(ts)
    d = d[d > 0]
    if len(d) == 0:
        return float(default)
    return float(1.0 / np.median(d))


def decode_image_msg(msg, bridge: CvBridge) -> np.ndarray | None:
    if hasattr(msg, "format") and hasattr(msg, "data"):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_bgr is None:
                return None
            return cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            return None
    try:
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        return np.asarray(cv_img, dtype=np.uint8)
    except Exception:
        return None


def get_valid_timestamp(msg, bag_time_sec: float, valid_epoch: float = 1.5e9) -> float:
    header_ts = 0.0
    if hasattr(msg, "header"):
        try:
            header_ts = stamp_to_sec(msg.header.stamp)
        except Exception:
            header_ts = 0.0
    return header_ts if header_ts > valid_epoch else bag_time_sec


def quaternion_to_euler(quat_xyzw: np.ndarray) -> np.ndarray:
    return R.from_quat(quat_xyzw).as_euler("xyz")


def plot_states(pose_data: np.ndarray, clamp_data: np.ndarray, output_path: str, unwrap_euler: bool = True) -> None:
    t = int(len(pose_data))
    steps = np.arange(t, dtype=np.int64)
    pos = pose_data[:, :3] if t > 0 else np.zeros((0, 3), dtype=np.float32)
    if t > 0:
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
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.suptitle(f"Trajectory States ({t} frames)", fontsize=14)
    plt.savefig(output_path, dpi=THREE_VIEW_CONFIG["plots"]["dpi"], bbox_inches="tight")
    plt.close()


def plot_alignment(
    out_path: str,
    left_ts: np.ndarray,
    right_ts: np.ndarray,
    third_ts: np.ndarray,
    left_pose_ts: np.ndarray,
    left_clamp_ts: np.ndarray,
    right_pose_ts: np.ndarray,
    right_clamp_ts: np.ndarray,
    bins: int = 80,
) -> None:
    def align_err(frame_ts: np.ndarray, ref_ts: np.ndarray) -> np.ndarray:
        if len(frame_ts) == 0 or len(ref_ts) == 0:
            return np.zeros((0,), dtype=np.float64)
        idx = nearest_idx(frame_ts, ref_ts)
        return np.abs(frame_ts - ref_ts[idx])

    def nearest_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if len(a) == 0 or len(b) == 0:
            return np.zeros((0,), dtype=np.float64)
        idx = nearest_idx(a, b)
        return a - b[idx]

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
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax = fig.add_subplot(gs[0, 1])
    if len(lr) > 0:
        ax.hist(lr * 1000.0, bins=bins, alpha=0.6, label="L-R")
    if len(lt) > 0:
        ax.hist(lt * 1000.0, bins=bins, alpha=0.6, label="L-Third")
    if len(rt) > 0:
        ax.hist(rt * 1000.0, bins=bins, alpha=0.6, label="R-Third")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax = fig.add_subplot(gs[1, 0])
    if len(l_pose_err) > 0:
        ax.plot(l_pose_err * 1000.0, label="Left pose err (ms)")
    if len(l_clamp_err) > 0:
        ax.plot(l_clamp_err * 1000.0, label="Left clamp err (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax = fig.add_subplot(gs[1, 1])
    if len(r_pose_err) > 0:
        ax.plot(r_pose_err * 1000.0, label="Right pose err (ms)")
    if len(r_clamp_err) > 0:
        ax.plot(r_clamp_err * 1000.0, label="Right clamp err (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax = fig.add_subplot(gs[2, :])
    if len(left_ts) > 0:
        ax.plot(left_ts - left_ts[0], np.zeros_like(left_ts), ".", label="Left frames")
    if len(right_ts) > 0:
        ax.plot(right_ts - right_ts[0], np.ones_like(right_ts), ".", label="Right frames")
    if len(third_ts) > 0:
        ax.plot(third_ts - third_ts[0], 2 * np.ones_like(third_ts), ".", label="Third frames")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Left", "Right", "Third"])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    plt.savefig(out_path, dpi=THREE_VIEW_CONFIG["plots"]["dpi"], bbox_inches="tight")
    plt.close()


def convert_single_view_bag(bag: str, serial: str, out_dir: str, data_idx: int | str, *, with_plot: bool) -> None:
    bag_path = Path(bag)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    img_topic = f"/xv_sdk/{serial}/color_camera/image"
    pose_topic = f"/xv_sdk/{serial}/slam/pose"
    clamp_topic = f"/xv_sdk/{serial}/clamp/Data"
    bridge = CvBridge()
    images: list[tuple[float, np.ndarray]] = []
    poses: list[tuple[float, np.ndarray]] = []
    clamps: list[tuple[float, float]] = []
    with rosbag.Bag(str(bag_path), "r") as bag_obj:
        for topic, msg, t in bag_obj.read_messages(topics=[img_topic, pose_topic, clamp_topic]):
            ts = t.to_sec()
            if topic == img_topic:
                cv = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                images.append((ts, np.asarray(cv, dtype=np.uint8)))
            elif topic == pose_topic:
                poses.append((ts, pose7_from_msg(msg)))
            else:
                try:
                    clamps.append((ts, float(msg.data)))
                except AttributeError:
                    pass
    if not images:
        raise RuntimeError("No images found.")
    images.sort(key=lambda x: x[0])
    poses.sort(key=lambda x: x[0])
    clamps.sort(key=lambda x: x[0])
    img_ts = np.array([x[0] for x in images], dtype=np.float64)
    if poses:
        pose_ts = np.array([x[0] for x in poses], dtype=np.float64)
        pose_vals = np.stack([x[1] for x in poses], axis=0)
        idx = nearest_idx(img_ts, pose_ts)
        aligned_pose = pose_vals[idx]
        err = np.abs(img_ts - pose_ts[idx])
    else:
        aligned_pose = np.zeros((len(images), 7), dtype=np.float32)
        aligned_pose[:, 6] = 1.0
        err = np.zeros(len(images))
    if clamps:
        clamp_ts = np.array([x[0] for x in clamps], dtype=np.float64)
        clamp_vals = np.array([x[1] for x in clamps], dtype=np.float32)
        idx_c = nearest_idx(img_ts, clamp_ts)
        aligned_clamp = clamp_vals[idx_c]
        clamp_err = np.abs(img_ts - clamp_ts[idx_c])
    else:
        aligned_clamp = np.zeros(len(images), dtype=np.float32)
        clamp_err = np.zeros(len(images))
    fps = estimate_fps(img_ts, 20.0)
    video_path = out / f"episode{data_idx}.mp4"
    json_path = out / f"episode{data_idx}.json"
    iio.imwrite(video_path, [img for _, img in images], fps=fps)
    records = [
        {"timestamp": float(img_ts[i]), "pose": aligned_pose[i].astype(float).tolist(), "clamp": float(aligned_clamp[i])}
        for i in range(len(img_ts))
    ]
    json_path.write_text(json.dumps({"fps": fps, "records": records}, indent=2), encoding="utf-8")
    if with_plot:
        plot_states(aligned_pose, aligned_clamp, str(out / f"episode{data_idx}_states.png"))


def convert_dual_view_bag(bag: str, serial: str, out_dir: str, data_idx: int | str, head_topic: str) -> None:
    bag_path = Path(bag)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wrist_topic = f"/xv_sdk/{serial}/color_camera/image"
    pose_topic = f"/xv_sdk/{serial}/slam/pose"
    clamp_topic = f"/xv_sdk/{serial}/clamp/Data"
    bridge = CvBridge()
    head_images: list[tuple[float, np.ndarray]] = []
    wrist_images: list[tuple[float, np.ndarray]] = []
    poses: list[tuple[float, np.ndarray]] = []
    clamps: list[tuple[float, float]] = []
    with rosbag.Bag(str(bag_path), "r") as bag_data:
        for topic, msg, t in bag_data.read_messages(topics=[head_topic, wrist_topic, pose_topic, clamp_topic]):
            bag_ts = t.to_sec()
            ts = get_valid_timestamp(msg, bag_ts)
            if topic == head_topic:
                img = decode_image_msg(msg, bridge)
                if img is not None:
                    head_images.append((ts, img))
            elif topic == wrist_topic:
                img = decode_image_msg(msg, bridge)
                if img is not None:
                    wrist_images.append((ts, img))
            elif topic == pose_topic:
                poses.append((ts, pose7_from_msg(msg)))
            elif topic == clamp_topic:
                try:
                    clamps.append((ts, float(msg.data)))
                except Exception:
                    pass
    if not head_images:
        raise RuntimeError("No head images found.")
    head_images.sort(key=lambda x: x[0])
    wrist_images.sort(key=lambda x: x[0])
    poses.sort(key=lambda x: x[0])
    clamps.sort(key=lambda x: x[0])
    head_ts = np.array([x[0] for x in head_images], dtype=np.float64)
    wrist_ts = np.array([x[0] for x in wrist_images], dtype=np.float64)
    pose_ts = np.array([x[0] for x in poses], dtype=np.float64)
    clamp_ts = np.array([x[0] for x in clamps], dtype=np.float64)
    wrist_vals = [x[1] for x in wrist_images]
    pose_vals = np.stack([x[1] for x in poses], axis=0) if poses else np.zeros((0, 7))
    clamp_vals = np.array([x[1] for x in clamps], dtype=np.float32) if clamps else np.zeros((0,))
    fps = estimate_fps(head_ts, 30.0)
    if len(wrist_ts) > 0:
        idx = nearest_idx(head_ts, wrist_ts)
        aligned_wrist = [wrist_vals[i] for i in idx]
    else:
        aligned_wrist = [np.zeros_like(head_images[0][1])] * len(head_ts)
    if len(pose_ts) > 0:
        idx = nearest_idx(head_ts, pose_ts)
        aligned_pose = pose_vals[idx]
    else:
        aligned_pose = np.zeros((len(head_ts), 7))
        aligned_pose[:, 6] = 1.0
    if len(clamp_ts) > 0:
        idx = nearest_idx(head_ts, clamp_ts)
        aligned_clamp = clamp_vals[idx]
    else:
        aligned_clamp = np.zeros((len(head_ts),), dtype=np.float32)
    idx_str = f"{int(data_idx):04d}"
    video_head = out / f"episode{idx_str}_head.mp4"
    video_left = out / f"episode{idx_str}_left.mp4"
    json_path = out / f"episode{idx_str}.json"
    iio.imwrite(video_head, [img for _, img in head_images], fps=fps)
    iio.imwrite(video_left, aligned_wrist, fps=fps)
    records = [{"timestamp": float(head_ts[i]), "pose": aligned_pose[i].tolist(), "clamp": float(aligned_clamp[i])} for i in range(len(head_ts))]
    json_path.write_text(json.dumps({"fps": fps, "records": records}, indent=2), encoding="utf-8")
    plot_states(aligned_pose, aligned_clamp, str(out / f"episode{idx_str}_states.png"))


def _fmt_topic(tmpl: str, serial: str) -> str:
    return tmpl.format(serial=serial)


def convert_three_view_bag(bag_path: Path, out_dir: Path, episode_idx: int) -> dict[str, Any]:
    cfg = THREE_VIEW_CONFIG
    valid_epoch = float(cfg["header_stamp_valid_epoch_sec"])
    bridge = CvBridge()
    left_img = _fmt_topic(cfg["left"]["image_topic"], cfg["left"]["serial"])
    left_pose = _fmt_topic(cfg["left"]["pose_topic"], cfg["left"]["serial"])
    left_clamp = _fmt_topic(cfg["left"]["clamp_topic"], cfg["left"]["serial"])
    right_img = _fmt_topic(cfg["right"]["image_topic"], cfg["right"]["serial"])
    right_pose = _fmt_topic(cfg["right"]["pose_topic"], cfg["right"]["serial"])
    right_clamp = _fmt_topic(cfg["right"]["clamp_topic"], cfg["right"]["serial"])
    third_img = cfg["third"]["image_topic"]
    topics = [left_img, left_pose, left_clamp, right_img, right_pose, right_clamp, third_img]

    left_frames = []
    right_frames = []
    third_frames = []
    left_poses = []
    right_poses = []
    left_clamps = []
    right_clamps = []
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
                left_poses.append((ts, pose7_from_msg(msg)))
            elif topic == right_pose:
                right_poses.append((ts, pose7_from_msg(msg)))
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

    left_frames.sort(key=lambda x: x[0])
    right_frames.sort(key=lambda x: x[0])
    third_frames.sort(key=lambda x: x[0])
    left_poses.sort(key=lambda x: x[0])
    right_poses.sort(key=lambda x: x[0])
    left_clamps.sort(key=lambda x: x[0])
    right_clamps.sort(key=lambda x: x[0])
    ep6 = f"{int(episode_idx):06d}"
    if not left_frames or not right_frames or not third_frames:
        raise RuntimeError(f"Missing frames in bag {bag_path.name}")

    def align_pose_clamp(frame_ts, poses, clamps):
        t = len(frame_ts)
        pose_ts = np.array([x[0] for x in poses], dtype=np.float64)
        pose_vals = np.stack([x[1] for x in poses], axis=0).astype(np.float32, copy=False) if poses else np.zeros((0, 7), dtype=np.float32)
        clamp_ts = np.array([x[0] for x in clamps], dtype=np.float64)
        clamp_vals = np.array([x[1] for x in clamps], dtype=np.float32) if clamps else np.zeros((0,), dtype=np.float32)
        if len(pose_ts) > 0:
            idx = nearest_idx(frame_ts, pose_ts)
            aligned_pose = pose_vals[idx]
            err_p = float(np.max(np.abs(frame_ts - pose_ts[idx]))) if t > 0 else 0.0
        else:
            aligned_pose = np.zeros((t, 7), dtype=np.float32)
            aligned_pose[:, 6] = 1.0
            err_p = -1.0
        if len(clamp_ts) > 0:
            idx = nearest_idx(frame_ts, clamp_ts)
            aligned_clamp = clamp_vals[idx].astype(np.float32, copy=False)
            err_c = float(np.max(np.abs(frame_ts - clamp_ts[idx]))) if t > 0 else 0.0
        else:
            aligned_clamp = np.zeros((t,), dtype=np.float32)
            err_c = -1.0
        return aligned_pose, aligned_clamp, {"max_pose_err_sec": err_p, "max_clamp_err_sec": err_c}

    left_ts = np.array([t for t, _ in left_frames], dtype=np.float64)
    right_ts = np.array([t for t, _ in right_frames], dtype=np.float64)
    third_ts = np.array([t for t, _ in third_frames], dtype=np.float64)
    left_imgs = [img for _, img in left_frames]
    right_imgs = [img for _, img in right_frames]
    third_imgs = [img for _, img in third_frames]
    left_fps = estimate_fps(left_ts, cfg["fallback_fps"])
    right_fps = estimate_fps(right_ts, cfg["fallback_fps"])
    third_fps = estimate_fps(third_ts, cfg["fallback_fps"])
    left_pose_al, left_clamp_al, left_stats = align_pose_clamp(left_ts, left_poses, left_clamps)
    right_pose_al, right_clamp_al, right_stats = align_pose_clamp(right_ts, right_poses, right_clamps)

    left_mp4 = out_dir / f"episode_{ep6}_left.mp4"
    right_mp4 = out_dir / f"episode_{ep6}_right.mp4"
    third_mp4 = out_dir / f"episode_{ep6}_third.mp4"
    left_json = out_dir / f"episode_{ep6}_left.json"
    right_json = out_dir / f"episode_{ep6}_right.json"
    iio.imwrite(left_mp4, left_imgs, fps=float(left_fps))
    iio.imwrite(right_mp4, right_imgs, fps=float(right_fps))
    iio.imwrite(third_mp4, third_imgs, fps=float(third_fps))
    left_json.write_text(json.dumps({"view": "left", "fps": float(left_fps), "num_frames": int(len(left_ts)), "alignment": left_stats, "records": [
        {"timestamp": float(left_ts[i]), "pose": left_pose_al[i].tolist(), "clamp": float(left_clamp_al[i])} for i in range(len(left_ts))
    ]}, indent=2), encoding="utf-8")
    right_json.write_text(json.dumps({"view": "right", "fps": float(right_fps), "num_frames": int(len(right_ts)), "alignment": right_stats, "records": [
        {"timestamp": float(right_ts[i]), "pose": right_pose_al[i].tolist(), "clamp": float(right_clamp_al[i])} for i in range(len(right_ts))
    ]}, indent=2), encoding="utf-8")
    plot_states(left_pose_al, left_clamp_al, str(out_dir / f"episode_{ep6}_left_states.png"), unwrap_euler=cfg["plots"]["unwrap_euler"])
    plot_states(right_pose_al, right_clamp_al, str(out_dir / f"episode_{ep6}_right_states.png"), unwrap_euler=cfg["plots"]["unwrap_euler"])
    plot_alignment(
        str(out_dir / f"episode_{ep6}_alignment.png"),
        left_ts,
        right_ts,
        third_ts,
        np.array([t for t, _ in left_poses], dtype=np.float64),
        np.array([t for t, _ in left_clamps], dtype=np.float64),
        np.array([t for t, _ in right_poses], dtype=np.float64),
        np.array([t for t, _ in right_clamps], dtype=np.float64),
        bins=int(cfg["plots"]["alignment_hist_bins"]),
    )
    return {
        "episode": ep6,
        "left_fps": float(left_fps),
        "right_fps": float(right_fps),
        "third_fps": float(third_fps),
        "left_align": left_stats,
        "right_align": right_stats,
    }
