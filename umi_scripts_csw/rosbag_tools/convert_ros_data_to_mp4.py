import json
from pathlib import Path
import numpy as np


from cv_bridge import CvBridge
import imageio.v3 as iio
import rosbag


def stamp_to_sec(stamp) -> float:
    if hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9
    if hasattr(stamp, "secs") and hasattr(stamp, "nsecs"):
        return float(stamp.secs) + float(stamp.nsecs) * 1e-9
    return float(stamp)


def msg_time_sec(msg, fallback_time_sec: float) -> float:
    if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
        try:
            return stamp_to_sec(msg.header.stamp)
        except Exception:
            pass
    return float(fallback_time_sec)


def decode_rgb(msg) -> np.ndarray:
    h, w = int(msg.height), int(msg.width)
    enc = (msg.encoding or "").lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if enc == "rgb8":
        return data.reshape(h, w, 3)
    if enc == "bgr8":
        return data.reshape(h, w, 3)[:, :, ::-1]
    if enc == "rgba8":
        return data.reshape(h, w, 4)[:, :, :3]
    if enc == "bgra8":
        return data.reshape(h, w, 4)[:, :, :3][:, :, ::-1]
    if enc == "mono8":
        g = data.reshape(h, w, 1)
        return np.repeat(g, 3, axis=2)

    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


def pose7_from_msg(msg) -> np.ndarray:
    """
    xv_sdk/PoseStampedConfidence:
      - msg.poseMsg is geometry_msgs/PoseStamped
      - msg.confidence is float64
    """
    ps = msg.poseMsg  # geometry_msgs/PoseStamped
    p = ps.pose.position
    q = ps.pose.orientation
    return np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float32)


def nearest_idx(q_ts: np.ndarray, ref_ts: np.ndarray) -> np.ndarray:
    i = np.searchsorted(ref_ts, q_ts, side="left")
    i = np.clip(i, 0, len(ref_ts) - 1)
    j = np.clip(i - 1, 0, len(ref_ts) - 1)
    pick_j = np.abs(q_ts - ref_ts[j]) <= np.abs(q_ts - ref_ts[i])
    return np.where(pick_j, j, i)


def estimate_fps(ts: np.ndarray, default=20.0) -> float:
    if len(ts) < 3:
        return default
    d = np.diff(ts)
    d = d[d > 0]
    if len(d) == 0:
        return default
    return float(1.0 / np.median(d))


def main(bag: str, serial: str, out_dir: str, data_idx:int):
    bag = Path(bag)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_topic = f"/xv_sdk/{serial}/color_camera/image"
    pose_topic = f"/xv_sdk/{serial}/slam/pose"
    clamp_topic = f"/xv_sdk/{serial}/clamp/Data"

    bridge = CvBridge()

    images = []  # (t, rgb)
    poses = []   # (t, pose7)
    clamps = []   # (t, float)

    with rosbag.Bag(str(bag), "r") as bag:
        for topic, msg, t in bag.read_messages(topics=[img_topic, pose_topic,clamp_topic]):
            # Prefer header.stamp if present, else bag time t
            ts = t.to_sec()

            if topic == img_topic:
                cv = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                images.append((ts, np.asarray(cv, dtype=np.uint8)))
            elif topic == pose_topic:
                poses.append((ts, pose7_from_msg(msg)))
            else:  # clamp_topic
                # xv_sdk/Clamp: float64 data; float64 timestamp; Header header
                clamps.append((ts, float(msg.data)))



    images.sort(key=lambda x: x[0])
    poses.sort(key=lambda x: x[0])

    img_ts = np.array([x[0] for x in images], dtype=np.float64)
    pose_ts = np.array([x[0] for x in poses], dtype=np.float64)
    pose_vals = np.stack([x[1] for x in poses], axis=0)

    fps = estimate_fps(img_ts)

    nn = nearest_idx(img_ts, pose_ts)
    aligned_pose = pose_vals[nn]  # (T,7)
    err = np.abs(img_ts - pose_ts[nn])

    clamps.sort(key=lambda x: x[0])
    clamp_ts = np.array([x[0] for x in clamps], dtype=np.float64)
    clamp_vals = np.array([x[1] for x in clamps], dtype=np.float32)  # (C,)

    nn_c = nearest_idx(img_ts, clamp_ts)
    aligned_clamp = clamp_vals[nn_c]  # (T,)
    clamp_err = np.abs(img_ts - clamp_ts[nn_c])


    # write mp4
    video_path = out_dir / f"episode{data_idx}.mp4"
    frames = [img for _, img in images]
    iio.imwrite(video_path, frames, fps=fps)

    # write json
    json_path = out_dir / f"episode{data_idx}.json"
    records = [
        {
            "timestamp": float(img_ts[i]),
            "pose": aligned_pose[i].astype(float).tolist(),
            "clamp": float(aligned_clamp[i]),
        }
        for i in range(len(img_ts))
    ]

    json_path.write_text(json.dumps({"fps": fps, "records": records}, indent=2), encoding="utf-8")

    print(f"OK -> {video_path}")
    print(f"OK -> {json_path}")
    print(f"fps={fps:.3f}, max_pose_align_error_sec={float(err.max()):.6f}, "
        f"max_clamp_align_error_sec={float(clamp_err.max()):.6f}")



if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--bag", required=True)
    p.add_argument("--serial", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--data_idx",required=True)
    args = p.parse_args()
    main(args.bag, args.serial, args.out_dir,args.data_idx)
