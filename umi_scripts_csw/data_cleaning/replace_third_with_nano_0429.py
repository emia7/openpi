#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 handover_umi_0429 的第三视角替换为 DJI nano 切分片段。

对齐规则（与计划一致）：
  1. nano clip 先按 10Hz 做时间戳采样：t_k = k * 0.1 s，取原始帧序列中与 t_k 最近的帧（索引 round(t_k * fps)）。
  2. 在 10Hz 序列上取前 N 帧，N = 对应 episode 的 left 腕部视频帧数（末尾多读截断）。
  3. 写出 *_third.mp4，fps=10，无音轨。
  4. 复制数据集后删除全部 *_alignment.png。

Usage:
    python replace_third_with_nano_0429.py \\
        --source_dataset ~/Downloads/handover_umi_0429 \\
        --nano_dir ~/path/to/out_freq_dji0429_recut \\
        --output_dataset ~/Downloads/handover_umi_0429_nano

    # nano 片段从 clip_0001 开始时：
    python replace_third_with_nano_0429.py \\
        --source_dataset ~/Downloads/bagging_0430 \\
        --nano_dir ~/path/to/out_freq_dji0430_edited2 \\
        --output_dataset ~/Downloads/bagging_0430_nano \\
        --clip_index_base 1
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm


def count_frames(path: Path, limit: int = 1_000_000) -> int:
    r = imageio.get_reader(str(path))
    n = 0
    try:
        for _ in r:
            n += 1
            if n >= limit:
                break
    finally:
        r.close()
    return n


def read_all_frames(path: Path, limit: int = 1_000_000) -> Tuple[List[np.ndarray], float, Tuple[int, int]]:
    """顺序读取全部帧；返回 frames, fps, (h,w)。"""
    r = imageio.get_reader(str(path))
    meta = r.get_meta_data()
    fps = float(meta.get("fps") or 30.0)
    size = meta.get("size") or (0, 0)
    frames: List[np.ndarray] = []
    try:
        for frame in r:
            frames.append(np.asarray(frame))
            if len(frames) >= limit:
                break
    finally:
        r.close()
    if frames:
        h, w = frames[0].shape[:2]
        size = (w, h)  # meta size is (W,H)
    return frames, fps, size


def ten_hz_nearest_indices(fps: float, n_target: int, num_raw: int) -> List[int]:
    """
    对每个 k=0..n_target-1，t_k = k*0.1，取最近帧索引 round(t_k*fps) 并钳制到 [0, num_raw-1]。
    """
    if num_raw <= 0:
        return []
    out: List[int] = []
    for k in range(n_target):
        t = k * 0.1
        idx = int(round(t * fps))
        idx = max(0, min(idx, num_raw - 1))
        out.append(idx)
    return out


def pick_frames_by_indices(frames: List[np.ndarray], indices: List[int]) -> List[np.ndarray]:
    return [frames[i].copy() for i in indices]


def write_video_mp4(frames: List[np.ndarray], out_path: Path, fps: float = 10.0) -> None:
    if not frames:
        raise ValueError("empty frames")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    first = frames[0]
    if first.ndim != 3 or first.shape[2] not in (3, 4):
        raise ValueError(f"unexpected frame shape {first.shape}")
    kwargs = {"fps": fps, "macro_block_size": 1, "ffmpeg_params": ["-pix_fmt", "yuv420p"]}
    try:
        writer = imageio.get_writer(str(out_path), codec="libx264", quality=8, **kwargs)
    except Exception:
        writer = imageio.get_writer(str(out_path), **kwargs)
    try:
        for fr in frames:
            writer.append_data(np.asarray(fr[..., :3], dtype=np.uint8))
    finally:
        writer.close()


def remove_alignment_pngs(root: Path) -> int:
    n = 0
    for p in root.glob("episode_*_alignment.png"):
        p.unlink()
        n += 1
    return n


def copy_dataset(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def process_episode(
    nano_clip: Path,
    left_mp4: Path,
    out_third: Path,
) -> Dict[str, Any]:
    raw_frames, fps, _size = read_all_frames(nano_clip)
    raw_n = len(raw_frames)
    n_need = count_frames(left_mp4)

    if raw_n == 0:
        raise ValueError("nano clip has zero frames")
    last_idx = int(round((n_need - 1) * 0.1 * fps))
    if last_idx >= raw_n:
        raise ValueError(
            f"nano clip too short for {n_need} frames at 10Hz: need frame index <= {raw_n - 1}, last_idx={last_idx}"
        )

    indices = ten_hz_nearest_indices(fps, n_need, raw_n)

    picked = pick_frames_by_indices(raw_frames, indices)
    if len(picked) != n_need:
        raise RuntimeError(f"internal: picked {len(picked)} != N {n_need}")

    write_video_mp4(picked, out_third, fps=10.0)

    return {
        "nano_raw_frames": raw_n,
        "nano_fps_meta": fps,
        "ten_hz_frames_M": len(indices),
        "N": n_need,
        "note": None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source_dataset",
        type=Path,
        default=Path.home() / "Downloads" / "handover_umi_0429",
    )
    ap.add_argument(
        "--nano_dir",
        type=Path,
        default=None,
        help="默认: openpi 仓库下 umi_scripts_csw/out_freq_dji0429_recut",
    )
    ap.add_argument(
        "--output_dataset",
        type=Path,
        default=Path.home() / "Downloads" / "handover_umi_0429_nano",
    )
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument(
        "--clip_index_base",
        type=int,
        choices=(0, 1),
        default=0,
        help=(
            "nano clip 编号与 episode 编号对齐方式："
            "0=episode_N 对应 clip_{N-1}（如 episode_000001→clip_0000，默认）；"
            "1=episode_N 对应 clip_{N}（如 episode_000001→clip_0001，适用于 clip 从 0001 开始命名）"
        ),
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    nano_dir = args.nano_dir or (repo_root / "umi_scripts_csw" / "out_freq_dji0429_recut")

    src = args.source_dataset.expanduser().resolve()
    dst = args.output_dataset.expanduser().resolve()

    if not src.is_dir():
        raise SystemExit(f"source not found: {src}")
    if not nano_dir.is_dir():
        raise SystemExit(f"nano_dir not found: {nano_dir}")

    left_jsons = sorted(src.glob("episode_*_left.json"))
    n_ep = len(left_jsons)
    print(f"[INFO] episodes in source: {n_ep}")
    print(f"[INFO] nano_dir: {nano_dir}")
    print(f"[INFO] output: {dst}")

    if args.dry_run:
        print("[INFO] dry_run: skip copy/process")
        return

    print("[INFO] copying dataset...")
    copy_dataset(src, dst)

    removed = remove_alignment_pngs(dst)
    print(f"[INFO] removed alignment pngs: {removed}")

    mapping: Dict[str, Any] = {
        "source_dataset": str(src),
        "nano_dir": str(nano_dir),
        "output_dataset": str(dst),
        "clip_index_base": args.clip_index_base,
        "episodes": [],
        "failed": [],
    }

    for jf in tqdm(left_jsons, desc="replace third"):
        stem = jf.stem.replace("_left", "")  # episode_000001
        ep_idx = int(stem.split("_")[1])
        clip_i = ep_idx - 1 + args.clip_index_base
        clip_path = nano_dir / f"clip_{clip_i:04d}.mp4"
        left_mp4 = dst / f"{stem}_left.mp4"
        out_third = dst / f"{stem}_third.mp4"

        entry: Dict[str, Any] = {
            "episode_id": stem,
            "clip_id": f"clip_{clip_i:04d}",
            "clip_path": str(clip_path),
        }

        if not clip_path.is_file():
            mapping["failed"].append({**entry, "error": "clip missing"})
            continue
        if not left_mp4.is_file():
            mapping["failed"].append({**entry, "error": "left mp4 missing"})
            continue

        try:
            stats = process_episode(clip_path, left_mp4, out_third)
            entry.update(stats)
            mapping["episodes"].append(entry)
        except Exception as e:  # noqa: BLE001
            mapping["failed"].append({**entry, "error": str(e)})

    out_map = dst / "mapping.json"
    out_map.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] mapping written: {out_map}")
    print(f"[INFO] ok episodes: {len(mapping['episodes'])}, failed: {len(mapping['failed'])}")


if __name__ == "__main__":
    main()
