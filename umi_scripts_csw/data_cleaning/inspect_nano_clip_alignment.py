#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对照腕部 episode 数检查 nano clip 目录：数量、markers 间隔、时长异常、相邻首尾帧相似度。
输出 JSON + Markdown 报告（不写计划文件）。

Usage:
    python inspect_nano_clip_alignment.py \\
        --nano_dir umi_scripts_csw/out_freq_dji0430_edited2 \\
        --episode_dir ~/Downloads/bagging_0430 \\
        --clip_index_base 1 \\
        --report_out umi_scripts_csw/out_freq_dji0430_edited2/nano_alignment_report
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np


def clip_paths_sorted(nano_dir: Path) -> List[Path]:
    def key(p: Path) -> int:
        return int(p.stem.split("_")[1])

    return sorted(nano_dir.glob("clip_*.mp4"), key=key)


def count_episodes(episode_dir: Path) -> int:
    return len(list(episode_dir.glob("episode_*_left.json")))


def scan_clip_frames(path: Path) -> Tuple[int, float, np.ndarray, np.ndarray]:
    """返回 (帧数, fps, 首帧, 尾帧) RGB。优先 OpenCV seek，失败则回退全量解码。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return _scan_clip_frames_imageio(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        n_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ok, first_bgr = cap.read()
        if not ok or first_bgr is None:
            return _scan_clip_frames_imageio(path)
        first = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB)
        last = first.copy()
        n = n_meta
        if n_meta > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n_meta - 1))
            ok2, last_bgr = cap.read()
            if ok2 and last_bgr is not None:
                last = cv2.cvtColor(last_bgr, cv2.COLOR_BGR2RGB)
        if n_meta <= 0:
            return _scan_clip_frames_imageio(path)
        return n, fps, first, last
    finally:
        cap.release()


def _scan_clip_frames_imageio(path: Path) -> Tuple[int, float, np.ndarray, np.ndarray]:
    r = imageio.get_reader(str(path))
    meta = r.get_meta_data()
    fps = float(meta.get("fps") or 30.0)
    first = None
    last = None
    n = 0
    try:
        for fr in r:
            arr = np.asarray(fr)
            if first is None:
                first = arr.copy()
            last = arr.copy()
            n += 1
    finally:
        r.close()
    if first is None:
        raise ValueError(f"empty video {path}")
    return n, fps, first, last


def count_frames_video(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if n > 0:
            return n
    r = imageio.get_reader(str(path))
    n = 0
    try:
        for _ in r:
            n += 1
    finally:
        r.close()
    return n


def mse_rgb(a: np.ndarray, b: np.ndarray) -> float:
    a = a[..., :3].astype(np.float64)
    b = b[..., :3].astype(np.float64)
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a = a[:h, :w]
    b = b[:h, :w]
    return float(np.mean((a - b) ** 2))


def ten_hz_last_idx(n_need: int, fps: float) -> int:
    return int(round((n_need - 1) * 0.1 * fps))


def load_markers(path: Optional[Path]) -> Optional[List[float]]:
    if path is None or not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("start_times", []))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nano_dir", type=Path, required=True)
    ap.add_argument("--episode_dir", type=Path, required=True)
    ap.add_argument("--markers_json", type=Path, default=None)
    ap.add_argument("--clip_index_base", type=int, choices=(0, 1), default=1)
    ap.add_argument("--report_out", type=Path, required=True)
    ap.add_argument(
        "--write_trimmed_copy",
        type=Path,
        default=None,
        help="若指定，则复制 nano 目录到此路径，删除首尾多余 clip，并写入裁剪后的 markers.json",
    )
    args = ap.parse_args()

    nano_dir = args.nano_dir.expanduser().resolve()
    episode_dir = args.episode_dir.expanduser().resolve()
    markers_path = (
        args.markers_json.expanduser().resolve()
        if args.markers_json
        else nano_dir / "markers.json"
    )
    report_base = args.report_out.expanduser().resolve()

    clips = clip_paths_sorted(nano_dir)
    n_clips = len(clips)
    n_eps = count_episodes(episode_dir)
    ids = [int(p.stem.split("_")[1]) for p in clips]
    id_min, id_max = min(ids), max(ids)
    gaps = [i for i in range(id_min, id_max + 1) if i not in set(ids)]

    start_times = load_markers(markers_path)
    marker_analysis: Dict[str, Any] = {}
    if start_times:
        deltas = [start_times[i + 1] - start_times[i] for i in range(len(start_times) - 1)]
        med = statistics.median(deltas) if deltas else 0.0
        short_thr = max(2.0, 0.25 * med)
        short_idx = [(i, deltas[i]) for i in range(len(deltas)) if deltas[i] < short_thr]
        marker_analysis = {
            "start_times_count": len(start_times),
            "delta_sec_median": med,
            "delta_sec_min": min(deltas) if deltas else None,
            "delta_sec_max": max(deltas) if deltas else None,
            "suspicious_short_gap_indices": [{"between_clip_idx": i, "delta_sec": d} for i, d in short_idx],
            "clip_count_vs_markers": n_clips == len(start_times),
        }

    # per-clip: 单次解码取帧数/fps/首尾帧
    clip_scan: List[Dict[str, Any]] = []
    for p in clips:
        nf, fps, first, last = scan_clip_frames(p)
        clip_scan.append(
            {
                "path": p,
                "file": p.name,
                "clip_index": int(p.stem.split("_")[1]),
                "frames": nf,
                "fps_meta": fps,
                "duration_sec_approx": nf / fps if fps else None,
                "first": first,
                "last": last,
            }
        )

    clip_stats = [{k: v for k, v in c.items() if k not in ("path", "first", "last")} for c in clip_scan]

    frame_counts = [c["frames"] for c in clip_scan]
    med_f = statistics.median(frame_counts) if frame_counts else 0.0

    # outliers: very short vs median
    short_clips = [clip_stats[i] for i, c in enumerate(clip_scan) if c["frames"] < max(30, 0.5 * med_f)]

    # boundary MSE: (last of prev, first of cur)
    boundaries: List[Dict[str, Any]] = []
    for i in range(1, len(clip_scan)):
        try:
            la = clip_scan[i - 1]["last"]
            fb = clip_scan[i]["first"]
            boundaries.append(
                {
                    "left_file": clip_scan[i - 1]["file"],
                    "right_file": clip_scan[i]["file"],
                    "mse_last_first": mse_rgb(la, fb),
                }
            )
        except Exception as e:
            boundaries.append(
                {
                    "left_file": clip_scan[i - 1]["file"],
                    "right_file": clip_scan[i]["file"],
                    "error": str(e),
                }
            )

    # highlight boundaries involving clip_0000 or last clip
    boundary_high_mse = sorted(
        [b for b in boundaries if isinstance(b.get("mse_last_first"), float)],
        key=lambda x: x["mse_last_first"],
        reverse=True,
    )[:8]
    boundary_low_mse = sorted(
        [b for b in boundaries if isinstance(b.get("mse_last_first"), float)],
        key=lambda x: x["mse_last_first"],
    )[:8]

    by_clip_name = {c["file"]: c for c in clip_scan}

    # pairing simulation: current folder + clip_index_base vs episodes
    left_jsons = sorted(episode_dir.glob("episode_*_left.json"))
    pairing_failures: List[Dict[str, Any]] = []
    for jf in left_jsons:
        stem = jf.stem.replace("_left", "")
        ep_idx = int(stem.split("_")[1])
        clip_i = ep_idx - 1 + args.clip_index_base
        name = f"clip_{clip_i:04d}.mp4"
        cp = nano_dir / name
        lp = episode_dir / f"{stem}_left.mp4"
        if not cp.is_file():
            pairing_failures.append({"episode": stem, "reason": "clip missing", "expected": name})
            continue
        sc = by_clip_name.get(name)
        if sc:
            raw_n = sc["frames"]
            fps = sc["fps_meta"]
        else:
            raw_n = count_frames_video(cp)
            r = imageio.get_reader(str(cp))
            fps = float(r.get_meta_data().get("fps") or 30.0)
            r.close()
        n_need = count_frames_video(lp)
        last_idx = ten_hz_last_idx(n_need, fps)
        if last_idx >= raw_n:
            pairing_failures.append(
                {
                    "episode": stem,
                    "reason": "nano too short at 10Hz",
                    "N": n_need,
                    "nano_frames": raw_n,
                    "last_idx": last_idx,
                    "clip": name,
                }
            )

    report: Dict[str, Any] = {
        "nano_dir": str(nano_dir),
        "episode_dir": str(episode_dir),
        "episode_count": n_eps,
        "clip_count": n_clips,
        "clip_id_min": id_min,
        "clip_id_max": id_max,
        "clip_id_gaps": gaps,
        "expected_clip_count_if_one_to_one": n_eps,
        "surplus_clips": n_clips - n_eps,
        "clip_index_base": args.clip_index_base,
        "markers": marker_analysis,
        "duration_outliers_short": short_clips,
        "boundary_mse_highest_8": boundary_high_mse,
        "boundary_mse_lowest_8": boundary_low_mse,
        "pairing_failures_current_layout": pairing_failures,
        "recommendation": [],
    }

    extra = n_clips - n_eps
    if extra == 2 and id_min == 0 and id_max == n_eps + 1:
        report["recommendation"].append(
            "检测到 clip 编号自 0 起且总数比 episode 多 2：通常应删除 clip_0000（片头多余）与 clip_%04d（片尾多余），并同步 markers start_times 去掉首尾各一项。"
            % (n_eps + 1)
        )
    elif extra != 0:
        report["recommendation"].append(
            f"clip 数 ({n_clips}) 与 episode 数 ({n_eps}) 不一致，请人工核对分段或 clip_index_base。"
        )

    report_base.parent.mkdir(parents=True, exist_ok=True)
    json_path = report_base.with_suffix(".json")
    md_path = report_base.with_suffix(".md")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Nano clip 对齐检查报告",
        "",
        f"- nano: `{nano_dir}`",
        f"- episodes: `{episode_dir}`",
        f"- episode 数: **{n_eps}**",
        f"- clip 数: **{n_clips}**（编号 {id_min}–{id_max}）",
        f"- clip_index_base: **{args.clip_index_base}**",
        f"- 当前配对下 10Hz 不足失败数: **{len(pairing_failures)}**",
        "",
        "## Markers",
        "",
        "```json",
        json.dumps(marker_analysis, indent=2, ensure_ascii=False),
        "```",
        "",
        "## 时长偏短的 clip（帧数 < max(30, 0.5×中位数)）",
        "",
    ]
    for c in short_clips[:20]:
        lines.append(f"- {c['file']}: {c['frames']} frames")
    lines.extend(["", "## 相邻边界 MSE（首帧 vs 上一段尾帧）最高 8 组（切换最大）", ""])
    for b in boundary_high_mse:
        lines.append(f"- {b['left_file']} -> {b['right_file']}: MSE={b['mse_last_first']:.2f}")
    lines.extend(["", "## 相邻边界 MSE 最低 8 组（可能重复切 / 粘滞）", ""])
    for b in boundary_low_mse:
        lines.append(f"- {b['left_file']} -> {b['right_file']}: MSE={b['mse_last_first']:.2f}")
    lines.extend(["", "## 建议", ""])
    for r in report["recommendation"]:
        lines.append(f"- {r}")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] JSON: {json_path}")
    print(f"[OK] MD:   {md_path}")

    if args.write_trimmed_copy:
        trim_dst = args.write_trimmed_copy.expanduser().resolve()
        if trim_dst.exists():
            shutil.rmtree(trim_dst)
        trim_dst.mkdir(parents=True)
        # keep clip_0001 .. clip_{n_eps:04d} when base=1 and had 0..n_eps+1
        kept = []
        for ep in range(1, n_eps + 1):
            clip_i = ep - 1 + args.clip_index_base
            src = nano_dir / f"clip_{clip_i:04d}.mp4"
            if not src.is_file():
                print(f"[WARN] missing {src.name} for trim copy")
                continue
            shutil.copy2(src, trim_dst / src.name)
            kept.append(src.name)
        if start_times and len(start_times) >= n_eps + 2:
            # trim first and last boundary if surplus was head+tail
            new_st = start_times[1 : 1 + n_eps]
            if len(new_st) != n_eps:
                new_st = start_times[1 : n_eps + 1]
            meta = json.loads(markers_path.read_text(encoding="utf-8"))
            meta["start_times"] = new_st
            (trim_dst / "markers.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"[OK] trimmed markers: {len(new_st)} start_times -> {trim_dst / 'markers.json'}")
        print(f"[OK] trimmed clips copied: {len(kept)} -> {trim_dst}")


if __name__ == "__main__":
    main()
