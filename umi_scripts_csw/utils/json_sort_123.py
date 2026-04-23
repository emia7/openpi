#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import argparse
from pathlib import Path

# 支持 episode_000001_left / episode_000001_left_states / episode_000001_alignment 等
EP_RE = re.compile(r"^episode[_-]?(\d+)(.*)$")

# 长的放前面，防止误匹配（例如 _left_states 要先于 _left）
SPECIAL_SUFFIXES = [
    "_left_states",
    "_right_states",
    "_alignment",
    "_left",
    "_right",
    "_third",
]

def split_stem(stem: str):
    """
    把 stem 拆成:
      ep_num_str: '000001' 或 None
      base:      'episode_000001' (不含 view/plot 后缀)
      extra:     '_left' / '_right' / '_third' / '_left_states' / '_alignment' / '' (未知后缀原样保留)
    """
    m = EP_RE.match(stem)
    if not m:
        return None

    num = m.group(1)
    tail = m.group(2) or ""

    # 规范 base 前缀为 episode_ + num（保持下划线风格）
    base = f"episode_{num}"

    extra = ""
    for s in SPECIAL_SUFFIXES:
        if tail == s:
            extra = s
            return num, base, extra
        if tail.endswith(s) and tail == s:
            extra = s
            return num, base, extra

    # 如果 tail 不是已知 suffix（比如你未来加了别的），也保留原 tail
    extra = tail
    return num, base, extra


def extract_sort_key(num_str: str):
    # 主要按数字排序，次级按原字符串
    return (int(num_str), num_str)


def two_phase_rename(mapping, dry_run: bool):
    """
    mapping: list of (src_path, dst_path)
    """
    # 冲突检查
    dsts = [dst for _, dst in mapping]
    if len(set(dsts)) != len(dsts):
        raise RuntimeError("目标文件名冲突：请检查是否会生成重名文件")

    # 构建临时映射
    pid = os.getpid()
    tmp_mapping = []
    for src, dst in mapping:
        if src == dst:
            continue
        tmp = src.with_name(f".__tmp__{src.stem}__{pid}{src.suffix}")
        tmp_mapping.append((src, tmp, dst))

    if not tmp_mapping:
        print("[INFO] 没有文件需要改名（已经连续/或没有匹配到 episode 文件）")
        return

    print("\n=== Rename Plan ===")
    for src, _, dst in tmp_mapping:
        print(f"{src.name}  ->  {dst.name}")
    print("===================\n")

    if dry_run:
        print("[DRY RUN] 未执行改名，请加 --apply 执行")
        return

    # Phase 1: src -> tmp
    for src, tmp, _ in tmp_mapping:
        if tmp.exists():
            raise RuntimeError(f"临时文件已存在：{tmp.name}")
        src.rename(tmp)

    # Phase 2: tmp -> dst
    for _, tmp, dst in tmp_mapping:
        if dst.exists():
            raise RuntimeError(f"目标文件已存在：{dst.name}")
        tmp.rename(dst)

    print(f"[OK] 重命名完成，共处理 {len(tmp_mapping)} 个文件")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="目标文件夹")
    ap.add_argument("--start", type=int, default=0, help="新的起始编号（默认 0）")
    ap.add_argument("--digits", type=int, default=6, help="编号位数（默认 6 -> 000001）")
    ap.add_argument("--exts", default="json,mp4,png", help="参与重命名的后缀，用逗号分隔")
    ap.add_argument("--apply", action="store_true", help="真正执行改名（默认仅预览）")
    args = ap.parse_args()

    folder = Path(args.dir)
    if not folder.is_dir():
        raise SystemExit(f"不是目录: {folder}")

    exts = [e.strip().lower() for e in args.exts.split(",")]
    exts = [("." + e) if not e.startswith(".") else e for e in exts]

    # 1) 扫描并按 episode 号分组：num_str -> list[(path, extra)]
    groups = {}  # num_str -> list[(Path, extra)]
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue

        res = split_stem(p.stem)
        if res is None:
            continue
        num_str, base, extra = res
        groups.setdefault(num_str, []).append((p, extra))

    if not groups:
        print("[INFO] 未找到 episode_*.(json/mp4/png) 文件")
        return

    # 2) episode 号排序
    nums_sorted = sorted(groups.keys(), key=extract_sort_key)

    # 3) 生成 rename 映射：旧 -> 新
    mapping = []
    cur = int(args.start)

    for old_num in nums_sorted:
        new_num = str(cur).zfill(int(args.digits))
        cur += 1

        for src_path, extra in groups[old_num]:
            # 统一输出命名：episode_{new_num}{extra}{ext}
            dst_name = f"episode_{new_num}{extra}{src_path.suffix}"
            dst_path = src_path.with_name(dst_name)
            mapping.append((src_path, dst_path))

    # 4) 两阶段改名
    two_phase_rename(mapping, dry_run=(not args.apply))


if __name__ == "__main__":
    main()
