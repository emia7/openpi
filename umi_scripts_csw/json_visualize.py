import os
import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def infer_frames(data):
    """
    尝试生成 x 轴（帧号）。
    优先：用 timestamp + fps 推算相对帧号；否则就用索引 0..N-1。
    """
    records = data.get("records", [])
    n = len(records)
    if n == 0:
        return np.array([])

    fps = data.get("fps", None)
    ts = [r.get("timestamp", None) for r in records]
    if fps is not None and all(t is not None for t in ts):
        t0 = ts[0]
        frames = np.rint((np.array(ts) - t0) * float(fps)).astype(int)
        # 防止出现重复/倒退导致曲线不好看：退化为顺序索引
        if np.any(np.diff(frames) < 0):
            frames = np.arange(n)
        return frames

    return np.arange(n)

def extract_series(data):
    records = data.get("records", [])
    if not records:
        return None

    # pose: N x D
    poses = []
    clamps = []
    for r in records:
        pose = r.get("pose", None)
        if pose is None:
            continue
        poses.append(pose)
        clamps.append(r.get("clamp", np.nan))

    if not poses:
        return None

    poses = np.array(poses, dtype=float)          # (N, D)
    clamps = np.array(clamps, dtype=float)        # (N,)

    return poses, clamps

def plot_one_json(json_path: str, out_dir: str, layout=(2, 4), dpi=160):
    data = load_json(json_path)
    series = extract_series(data)
    if series is None:
        print(f"[SKIP] no pose data: {json_path}")
        return

    poses, clamps = series
    n, d = poses.shape
    frames = infer_frames(data)

    # 要求：8 张图。默认：pose 7 维 + clamp 1 维
    # 如果 pose 维度不是 7，也照画 pose 的每一维；第 8 张仍画 clamp（若存在）
    num_plots = 8
    rows, cols = layout
    if rows * cols != num_plots:
        raise ValueError(f"layout {layout} must multiply to 8, e.g. (2,4) or (4,2)")

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.2), dpi=dpi)
    axes = axes.flatten()

    # 1..D：pose 各维
    for i in range(min(d, 7)):
        ax = axes[i]
        ax.plot(frames, poses[:, i], linewidth=1.2)
        ax.set_title(f"pose[{i}]")
        ax.set_xlabel("frame")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.3)

    # 如果 pose 维度 < 7，把缺的子图标注出来
    for i in range(d, 7):
        ax = axes[i]
        ax.axis("off")
        ax.text(0.5, 0.5, f"pose[{i}] (missing)", ha="center", va="center")
    print("pos done")

    # 第 8 张：clamp
    ax = axes[7]
    if np.all(np.isnan(clamps)):
        ax.axis("off")
        ax.text(0.5, 0.5, "clamp (missing)", ha="center", va="center")
    else:
        ax.plot(frames, clamps, linewidth=1.2)
        ax.set_title("clamp")
        ax.set_xlabel("frame")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.3)
    print("clamp done")

    # 总标题：文件名 + fps
    fps = data.get("fps", None)
    base = os.path.basename(json_path)
    fig.suptitle(f"{base} | fps={fps}", y=0.995, fontsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.splitext(base)[0] + ".png")
    print("out_path",out_path)
    fig.savefig(out_path)
    print("fig saved")
    plt.close(fig)

    print(f"[OK] {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="包含 json 的目录")
    ap.add_argument("--out_dir", required=True, help="输出 png 的目录")
    ap.add_argument("--layout", default="2x4", help="子图布局：2x4 或 4x2 等（必须=8张）")
    args = ap.parse_args()

    layout = args.layout.lower().split("x")
    layout = (int(layout[0]), int(layout[1]))

    json_files = sorted(glob.glob(os.path.join(args.in_dir, "*.json")))
    if not json_files:
        print(f"No json files found in: {args.in_dir}")
        return
    print("test position 1")
    print("json files",json_files)
    for jp in json_files:
        plot_one_json(jp, args.out_dir, layout=layout)
    print("All done.")
if __name__ == "__main__":
    main()
