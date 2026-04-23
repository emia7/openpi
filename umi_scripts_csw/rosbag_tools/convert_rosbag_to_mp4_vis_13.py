import json
import argparse
from pathlib import Path
import numpy as np
import rosbag
import imageio.v3 as iio
import cv2
from cv_bridge import CvBridge

# === [新增] 绘图与数学库 ===
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial.transform import Rotation as R

# ==========================================
# 辅助函数
# ==========================================

def stamp_to_sec(stamp) -> float:
    if hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9
    if hasattr(stamp, "secs") and hasattr(stamp, "nsecs"):
        return float(stamp.secs) + float(stamp.nsecs) * 1e-9
    return float(stamp)

def pose7_from_msg(msg) -> np.ndarray:
    ps = msg.poseMsg
    p = ps.pose.position
    q = ps.pose.orientation
    return np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float32)

def nearest_idx(q_ts: np.ndarray, ref_ts: np.ndarray) -> np.ndarray:
    i = np.searchsorted(ref_ts, q_ts, side="left")
    i = np.clip(i, 0, len(ref_ts) - 1)
    j = np.clip(i - 1, 0, len(ref_ts) - 1)
    pick_j = np.abs(q_ts - ref_ts[j]) <= np.abs(q_ts - ref_ts[i])
    return np.where(pick_j, j, i)

def estimate_fps(ts: np.ndarray, default=30.0) -> float:
    if len(ts) < 3: return default
    d = np.diff(ts)
    d = d[d > 0]
    if len(d) == 0: return default
    return float(1.0 / np.median(d))

def decode_image_msg(msg, bridge: CvBridge) -> np.ndarray:
    """自动处理 Raw 或 Compressed 图像"""
    if hasattr(msg, 'format') and "compressed" in msg._type.lower():
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB) if cv_bgr is not None else None
        except: return None
    try:
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        return np.asarray(cv_img, dtype=np.uint8)
    except: return None

def get_valid_timestamp(msg, bag_time_sec):
    """
    智能时间戳检查：如果 header 时间戳合法(>2017年)，则用 header；
    否则（如为0或硬件启动时间），回退到 rosbag 录制时间。
    """
    header_ts = 0.0
    if hasattr(msg, "header"):
        try:
            header_ts = stamp_to_sec(msg.header.stamp)
        except: pass

    if header_ts > 1.5e9:
        return header_ts
    else:
        return bag_time_sec

# ==========================================
# [新增] 可视化功能函数
# ==========================================

def quaternion_to_euler(quat):
    """Convert quaternion [qx, qy, qz, qw] to euler [rx, ry, rz] (xyz order)."""
    r = R.from_quat(quat)
    return r.as_euler('xyz')

def plot_states(pose_data, clamp_data, output_path):
    """
    绘制位置、姿态(欧拉角)和夹爪状态曲线
    pose_data: (T, 7) [x, y, z, qx, qy, qz, qw]
    clamp_data: (T,)
    """
    T = len(pose_data)
    steps = np.arange(T)

    # 提取位置
    pos = pose_data[:, :3]

    # 提取旋转并转换为欧拉角 (处理四元数可能为空的情况)
    if T > 0:
        euler = np.array([quaternion_to_euler(q) for q in pose_data[:, 3:7]])
        # Unwrap angles (解决 -pi 到 pi 的跳变，使曲线平滑)
        for i in range(3):
            euler[:, i] = np.unwrap(euler[:, i])
    else:
        euler = np.zeros((T, 3))

    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

    pos_labels = ['X (m)', 'Y (m)', 'Z (m)']
    rot_labels = ['Roll (rad)', 'Pitch (rad)', 'Yaw (rad)']

    # 1. Plot Position
    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(steps, pos[:, i], 'b-', linewidth=1.5)
        ax.set_title(f'Position {pos_labels[i]}')
        ax.set_xlabel('Frame')
        ax.grid(True, alpha=0.3)

    # 2. Plot Rotation
    for i in range(3):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(steps, euler[:, i], 'g-', linewidth=1.5)
        ax.set_title(f'Rotation {rot_labels[i]}')
        ax.set_xlabel('Frame')
        ax.grid(True, alpha=0.3)

    # 3. Plot Clamp
    ax = fig.add_subplot(gs[2, :])
    ax.plot(steps, clamp_data, 'r-', linewidth=1.5, label='Clamp')
    ax.set_title('Clamp / Gripper State')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Value')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Trajectory States ({T} frames)', fontsize=14)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved state plot: {output_path}")

# ==========================================
# 主逻辑
# ==========================================

def main(bag: str, serial: str, out_dir: str, data_idx: int, head_topic: str):
    bag = Path(bag)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wrist_topic = f"/xv_sdk/{serial}/color_camera/image"
    pose_topic = f"/xv_sdk/{serial}/slam/pose"
    clamp_topic = f"/xv_sdk/{serial}/clamp/Data"

    bridge = CvBridge()
    head_images, wrist_images, poses, clamps = [], [], [], []

    print(f"Reading bag: {bag}")

    with rosbag.Bag(str(bag), "r") as bag_data:
        topics = [head_topic, wrist_topic, pose_topic, clamp_topic]
        for topic, msg, t in bag_data.read_messages(topics=topics):

            # 使用智能时间戳检查（解决 FastUMI 时间戳为 0 或不准的问题）
            bag_ts = t.to_sec()
            ts = get_valid_timestamp(msg, bag_ts)

            if topic == head_topic:
                img = decode_image_msg(msg, bridge)
                if img is not None: head_images.append((ts, img))

            elif topic == wrist_topic:
                img = decode_image_msg(msg, bridge)
                if img is not None: wrist_images.append((ts, img))

            elif topic == pose_topic:
                poses.append((ts, pose7_from_msg(msg)))

            elif topic == clamp_topic:
                try:
                    val = float(msg.data)
                    clamps.append((ts, val))
                except: pass

    # Sort
    head_images.sort(key=lambda x: x[0])
    wrist_images.sort(key=lambda x: x[0])
    poses.sort(key=lambda x: x[0])
    clamps.sort(key=lambda x: x[0])

    if not head_images:
        print("[Error] No head images found.")
        return

    # Extract Arrays
    head_ts = np.array([x[0] for x in head_images], dtype=np.float64)
    wrist_ts = np.array([x[0] for x in wrist_images], dtype=np.float64)
    pose_ts = np.array([x[0] for x in poses], dtype=np.float64)
    clamp_ts = np.array([x[0] for x in clamps], dtype=np.float64)

    wrist_vals = [x[1] for x in wrist_images]
    pose_vals = np.stack([x[1] for x in poses], axis=0) if poses else np.zeros((0, 7))
    clamp_vals = np.array([x[1] for x in clamps], dtype=np.float32) if clamps else np.zeros((0,))

    fps = estimate_fps(head_ts, default=30.0)

    # Alignment (All aligned to Head timestamp)
    # 1. Wrist
    if len(wrist_ts) > 0:
        nn = nearest_idx(head_ts, wrist_ts)
        aligned_wrist = [wrist_vals[i] for i in nn]
        err_w = float(np.abs(head_ts - wrist_ts[nn]).max())
    else:
        aligned_wrist = [np.zeros_like(head_images[0][1])] * len(head_ts)
        err_w = -1.0

    # 2. Pose
    if len(pose_ts) > 0:
        nn = nearest_idx(head_ts, pose_ts)
        aligned_pose = pose_vals[nn]
        err_p = float(np.abs(head_ts - pose_ts[nn]).max())
    else:
        # Fill identity pose if missing
        aligned_pose = np.zeros((len(head_ts), 7))
        aligned_pose[:, 6] = 1.0 
        err_p = -1.0

    # 3. Clamp
    if len(clamp_ts) > 0:
        nn = nearest_idx(head_ts, clamp_ts)
        aligned_clamp = clamp_vals[nn]
        err_c = float(np.abs(head_ts - clamp_ts[nn]).max())
    else:
        aligned_clamp = np.zeros((len(head_ts),))
        err_c = -1.0

    # Output Files
    idx_str = f"{int(data_idx):04d}"
    video_head = out_dir / f"episode{idx_str}_head.mp4"
    video_left = out_dir / f"episode{idx_str}_left.mp4"
    json_path = out_dir / f"episode{idx_str}.json"
    plot_path = out_dir / f"episode{idx_str}_states.png"

    print(f"Writing Head MP4: {video_head}")
    iio.imwrite(video_head, [img for _, img in head_images], fps=fps)

    print(f"Writing Left MP4: {video_left}")
    iio.imwrite(video_left, aligned_wrist, fps=fps)

    # Write JSON
    records = []
    for i in range(len(head_ts)):
        records.append({
            "timestamp": float(head_ts[i]),
            "pose": aligned_pose[i].tolist(),
            "clamp": float(aligned_clamp[i])
        })
    json_path.write_text(json.dumps({"fps": fps, "records": records}, indent=2), encoding="utf-8")
    print(f"Writing JSON: {json_path}")

    # === [新增] 生成可视化图表 ===
    print("Generating state plot...")
    plot_states(aligned_pose, aligned_clamp, str(plot_path))

    print(f"Done.")
    print(f"Summary:")
    print(f"  FPS: {fps:.2f}")
    print(f"  Max Align Error (Wrist): {err_w:.4f} sec")
    print(f"  Max Align Error (Pose):  {err_p:.4f} sec")
    print(f"  Max Align Error (Clamp): {err_c:.4f} sec")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bag", required=True)
    p.add_argument("--serial", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--data_idx", required=True)
    p.add_argument("--head_topic", default="/camera/color/image_raw/compressed")
    args = p.parse_args()
    main(args.bag, args.serial, args.out_dir, args.data_idx, args.head_topic)