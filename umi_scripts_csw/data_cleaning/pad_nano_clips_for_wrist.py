#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
若 nano clip 按 replace_third_with_nano_0429 的 10Hz 规则「不够长」，
在末尾用最后一帧补足到最小所需帧数（freeze-last-frame），再写出 mp4。

不改腕部数据；输出目录为 nano_dir 的完整副本，仅重编码需要补帧的 clip。

Usage:
    python pad_nano_clips_for_wrist.py \\
        --nano_dir umi_scripts_csw/out_freq_dji0430_trim_tail2 \\
        --episode_dir ~/Downloads/bagging_0430 \\
        --clip_index_base 0 \\
        --out_dir umi_scripts_csw/out_freq_dji0430_trim_tail2_padded
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from replace_third_with_nano_0429 import read_all_frames, write_video_mp4


def clip_index_for_episode(ep_num: int, clip_index_base: int) -> int:
    return ep_num - 1 + clip_index_base


def min_raw_frames_needed(n_wrist: int, fps: float) -> int:
    last_idx = int(round((n_wrist - 1) * 0.1 * fps))
    return last_idx + 1


def pad_frames(frames: List[np.ndarray], target_len: int) -> Tuple[List[np.ndarray], int]:
    if len(frames) >= target_len:
        return frames, 0
    last = frames[-1]
    n_add = target_len - len(frames)
    out = list(frames)
    for _ in range(n_add):
        out.append(np.array(last, copy=True))
    return out, n_add


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nano_dir", type=Path, required=True)
    ap.add_argument("--episode_dir", type=Path, required=True)
    ap.add_argument("--clip_index_base", type=int, choices=(0, 1), default=0)
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()

    nano_src = args.nano_dir.expanduser().resolve()
    ep_dir = args.episode_dir.expanduser().resolve()
    out_root = args.out_dir.expanduser().resolve()

    if not nano_src.is_dir():
        raise SystemExit(f"nano_dir not found: {nano_src}")
    if not ep_dir.is_dir():
        raise SystemExit(f"episode_dir not found: {ep_dir}")

    if out_root.exists():
        shutil.rmtree(out_root)
    shutil.copytree(nano_src, out_root)

    left_jsons = sorted(ep_dir.glob("episode_*_left.json"))
    padded_report: list[dict] = []

    for jf in tqdm(left_jsons, desc="check/pad clips"):
        m = re.match(r"episode_(\d+)_left\.json", jf.name)
        if not m:
            continue
        ep_num = int(m.group(1))
        data = json.loads(jf.read_text(encoding="utf-8"))
        n_wrist = len(data.get("records") or [])
        if n_wrist <= 0:
            continue

        ci = clip_index_for_episode(ep_num, args.clip_index_base)
        clip_name = f"clip_{ci:04d}.mp4"
        clip_dst = out_root / clip_name
        if not clip_dst.is_file():
            padded_report.append(
                {"episode": f"episode_{ep_num:06d}", "clip": clip_name, "error": "clip missing"}
            )
            continue

        frames, fps, _ = read_all_frames(clip_dst)
        need = min_raw_frames_needed(n_wrist, fps)
        if len(frames) >= need:
            continue

        new_frames, n_add = pad_frames(frames, need)
        write_video_mp4(new_frames, clip_dst, fps=fps)
        padded_report.append(
            {
                "episode": f"episode_{ep_num:06d}",
                "clip": clip_name,
                "wrist_frames_N": n_wrist,
                "fps": fps,
                "before_frames": len(frames),
                "after_frames": len(new_frames),
                "padded_tail_duplicate_frames": n_add,
            }
        )

    rep_path = out_root / "pad_report.json"
    rep_path.write_text(json.dumps(padded_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] wrote {out_root}")
    print(f"[OK] padded {sum(1 for x in padded_report if 'padded_tail_duplicate_frames' in x)} clips")
    print(f"[OK] report: {rep_path}")


if __name__ == "__main__":
    main()
