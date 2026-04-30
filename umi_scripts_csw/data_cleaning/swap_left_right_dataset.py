#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
同一数据集内互换每条 episode 的 left/right（文件名层面互换 mp4/png，
JSON 交换内容并修正 \"view\" 字段）。

*_third.mp4 不变；单文件的 *_alignment.png 不变（若需对齐可视化请自行重生成）。

Usage:
    # 全目录所有 episode
    python swap_left_right_dataset.py --data_dir ~/Downloads/handover_umi_0429

    # 仅 handover_umi_0429_mix 中来自 0429 的段（合并顺序：先 0423_mix 再 0429 → 第 465–593 条）
    python swap_left_right_dataset.py --data_dir ~/Downloads/handover_umi_0429_mix \\
        --episode_index_min 465 --episode_index_max 593
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional, Set, Tuple

import tqdm


EPISODE_RE = re.compile(r"^episode_(\d{6})$")


def episode_stems(data_dir: Path) -> List[str]:
    stems: Set[str] = set()
    for p in data_dir.glob("episode_*_left.json"):
        stem = p.stem.replace("_left", "")
        if EPISODE_RE.match(stem):
            stems.add(stem)
    return sorted(stems)


def parse_episode_index(stem: str) -> int:
    m = EPISODE_RE.match(stem)
    if not m:
        raise ValueError(stem)
    return int(m.group(1))


def in_range(idx: int, lo: Optional[int], hi: Optional[int]) -> bool:
    if lo is not None and idx < lo:
        return False
    if hi is not None and idx > hi:
        return False
    return True


def swap_two_files(a: Path, b: Path) -> bool:
    """三向重命名交换内容，要求 a、b 均存在。"""
    if not a.is_file() or not b.is_file():
        return False
    tmp = a.with_name(a.name + ".__swap__")
    if tmp.exists():
        tmp.unlink()
    a.rename(tmp)
    b.rename(a)
    tmp.rename(b)
    return True


def fix_json_views(left_path: Path, right_path: Path) -> None:
    """将左右 JSON 内容对调，并令 view 与文件名一致。"""
    with open(left_path, "r", encoding="utf-8") as f:
        left_data = json.load(f)
    with open(right_path, "r", encoding="utf-8") as f:
        right_data = json.load(f)
    new_left = json.loads(json.dumps(right_data))  # deep copy
    new_right = json.loads(json.dumps(left_data))
    new_left["view"] = "left"
    new_right["view"] = "right"
    with open(left_path, "w", encoding="utf-8") as f:
        json.dump(new_left, f, indent=2, ensure_ascii=False)
        f.write("\n")
    with open(right_path, "w", encoding="utf-8") as f:
        json.dump(new_right, f, indent=2, ensure_ascii=False)
        f.write("\n")


def process_episode(data_dir: Path, stem: str) -> Tuple[bool, str]:
    left_mp4 = data_dir / f"{stem}_left.mp4"
    right_mp4 = data_dir / f"{stem}_right.mp4"
    left_json = data_dir / f"{stem}_left.json"
    right_json = data_dir / f"{stem}_right.json"
    left_states = data_dir / f"{stem}_left_states.png"
    right_states = data_dir / f"{stem}_right_states.png"

    if not left_json.is_file() or not right_json.is_file():
        return False, "missing left/right json"

    # 先交换二进制视频与配图（若存在）
    if left_mp4.is_file() and right_mp4.is_file():
        swap_two_files(left_mp4, right_mp4)
    elif left_mp4.is_file() or right_mp4.is_file():
        return False, "partial mp4"

    if left_states.is_file() and right_states.is_file():
        swap_two_files(left_states, right_states)
    elif left_states.is_file() or right_states.is_file():
        return False, "partial states png"

    # JSON：内容与 view 一起修正（等价于交换后再改 view）
    fix_json_views(left_json, right_json)

    return True, "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description="互换数据集中 left/right 标注（0429 纠错）")
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument(
        "--episode_index_min",
        type=int,
        default=None,
        help="仅处理 episode 编号 >= 该值（如 mix 中 0429 段从 465 开始）",
    )
    ap.add_argument(
        "--episode_index_max",
        type=int,
        default=None,
        help="仅处理 episode 编号 <= 该值",
    )
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"not a directory: {data_dir}")

    stems = episode_stems(data_dir)
    selected: List[str] = []
    for s in stems:
        idx = parse_episode_index(s)
        if in_range(idx, args.episode_index_min, args.episode_index_max):
            selected.append(s)

    print(f"[INFO] data_dir={data_dir}")
    print(f"[INFO] episodes total (with left.json): {len(stems)}")
    print(f"[INFO] selected for swap: {len(selected)}")

    if args.dry_run:
        for s in selected[:20]:
            print(f"  would swap: {s}")
        if len(selected) > 20:
            print(f"  ... +{len(selected) - 20} more")
        return

    ok = 0
    failed: List[Tuple[str, str]] = []
    for stem in tqdm.tqdm(selected, desc="swap L/R"):
        good, msg = process_episode(data_dir, stem)
        if good:
            ok += 1
        else:
            failed.append((stem, msg))

    log_path = data_dir / "swap_left_right_log.json"
    payload = {
        "data_dir": str(data_dir),
        "episode_index_min": args.episode_index_min,
        "episode_index_max": args.episode_index_max,
        "count_selected": len(selected),
        "count_ok": ok,
        "failed": [{"episode": a, "reason": b} for a, b in failed],
    }
    log_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] swapped ok={ok} failed={len(failed)} log={log_path}")


if __name__ == "__main__":
    main()
