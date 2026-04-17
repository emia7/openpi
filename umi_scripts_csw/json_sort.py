#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

EP_RE = re.compile(r"^episode[_-]?(\d+)(.*)$")
SPECIAL_SUFFIXES = [
    "_left_states",
    "_right_states",
    "_alignment",
    "_head",
    "_left",
    "_right",
    "_third",
    "_states",
]


def _normalize_exts(exts: str) -> list[str]:
    raw = [x.strip().lower() for x in exts.split(",") if x.strip()]
    return [f".{x}" if not x.startswith(".") else x for x in raw]


def _split_stem(stem: str) -> tuple[str, str] | None:
    m = EP_RE.match(stem)
    if not m:
        return None
    num = m.group(1)
    tail = m.group(2) or ""
    for suffix in SPECIAL_SUFFIXES:
        if tail == suffix:
            return num, suffix
    return num, tail


def _target_stem(prefix: str, num: int, digits: int, style: str, extra: str) -> str:
    padded = str(num).zfill(digits)
    base = f"{prefix}{padded}" if style == "compact" else f"{prefix}_{padded}"
    return f"{base}{extra}"


def _two_phase_rename(mapping: list[tuple[Path, Path]], dry_run: bool) -> None:
    targets = [dst for _, dst in mapping]
    if len(set(targets)) != len(targets):
        raise RuntimeError("目标文件名冲突，请检查参数")

    pid = os.getpid()
    staged: list[tuple[Path, Path, Path]] = []
    for src, dst in mapping:
        if src == dst:
            continue
        tmp = src.with_name(f".__tmp__{src.stem}__{pid}{src.suffix}")
        staged.append((src, tmp, dst))

    if not staged:
        print("[INFO] 没有文件需要改名")
        return

    print("\n=== Rename Plan ===")
    for src, _, dst in staged:
        print(f"{src.name}  ->  {dst.name}")
    print("===================\n")

    if dry_run:
        print("[DRY RUN] 未执行改名，加 --apply 执行")
        return

    for src, tmp, _ in staged:
        if tmp.exists():
            raise RuntimeError(f"临时文件已存在：{tmp.name}")
        src.rename(tmp)
    for _, tmp, dst in staged:
        if dst.exists():
            raise RuntimeError(f"目标文件已存在：{dst.name}")
        tmp.rename(dst)
    print(f"[OK] 重命名完成，共处理 {len(staged)} 个文件")


def main() -> None:
    ap = argparse.ArgumentParser(description="统一的 episode 文件重命名工具")
    ap.add_argument("--dir", required=True, help="目标文件夹")
    ap.add_argument("--start", type=int, default=0, help="起始编号，默认 0")
    ap.add_argument("--digits", type=int, default=6, help="编号位数，默认 6")
    ap.add_argument("--prefix", default="episode", help="前缀，默认 episode")
    ap.add_argument("--style", choices=["auto", "compact", "underscore"], default="auto", help="compact=episode0001, underscore=episode_000001")
    ap.add_argument("--exts", default="json,mp4,png", help="参与重命名的后缀")
    ap.add_argument("--apply", action="store_true", help="真正执行改名（默认预览）")
    args = ap.parse_args()

    folder = Path(args.dir)
    if not folder.is_dir():
        raise SystemExit(f"不是目录: {folder}")

    exts = _normalize_exts(args.exts)
    groups: dict[str, list[tuple[Path, str, str]]] = {}
    for p in folder.iterdir():
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        parsed = _split_stem(p.stem)
        if parsed is None:
            continue
        num, extra = parsed
        groups.setdefault(num, []).append((p, num, extra))

    if not groups:
        print("[INFO] 未找到 episode 命名文件")
        return

    nums_sorted = sorted(groups.keys(), key=lambda x: (int(x), x))
    style = args.style
    if style == "auto":
        style = "underscore" if any("_" in p.stem for files in groups.values() for p, _, _ in files) else "compact"

    mapping: list[tuple[Path, Path]] = []
    cur = int(args.start)
    for old_num in nums_sorted:
        for src, _, extra in groups[old_num]:
            stem = _target_stem(args.prefix, cur, int(args.digits), style, extra)
            mapping.append((src, src.with_name(f"{stem}{src.suffix}")))
        cur += 1

    _two_phase_rename(mapping, dry_run=not args.apply)


if __name__ == "__main__":
    main()
