import os
import re
import argparse
from pathlib import Path

NUM_RE = re.compile(r"(\d+)")

def extract_key(stem: str):
    """
    用于排序：
    - 优先取文件名中最后一段数字
    - 没有数字就按字符串
    """
    m = NUM_RE.findall(stem)
    if m:
        return (0, int(m[-1]), stem)
    return (1, stem)

def two_phase_rename(pairs, prefix, digits, start, dry_run: bool):
    """
    pairs: [
      {
        "base": 原 stem,
        "paths": [Path(...), ...]
      }
    ]
    """
    mapping = []
    cur = start

    for item in pairs:
        new_stem = f"{prefix}{str(cur).zfill(digits)}"
        cur += 1
        for p in item["paths"]:
            mapping.append((p, p.with_name(new_stem + p.suffix)))

    # 冲突检查
    targets = [dst for _, dst in mapping]
    if len(set(targets)) != len(targets):
        raise RuntimeError("目标文件名冲突，请检查后缀或参数")

    # 两阶段改名（避免覆盖）
    tmp_mapping = []
    pid = os.getpid()
    for src, dst in mapping:
        tmp = src.with_name(f".__tmp__{src.stem}__{pid}{src.suffix}")
        tmp_mapping.append((src, tmp, dst))

    # 预览
    print("\n=== Rename Plan ===")
    for src, _, dst in tmp_mapping:
        print(f"{src.name}  ->  {dst.name}")
    print("===================\n")

    if dry_run:
        print("[DRY RUN] 未执行改名，加 --apply 才会真正改名")
        return

    # src -> tmp
    for src, tmp, _ in tmp_mapping:
        if tmp.exists():
            raise RuntimeError(f"临时文件已存在：{tmp}")
        src.rename(tmp)

    # tmp -> dst
    for _, tmp, dst in tmp_mapping:
        if dst.exists():
            raise RuntimeError(f"目标文件已存在：{dst}")
        tmp.rename(dst)

    print("[OK] 重命名完成")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="目标文件夹")
    ap.add_argument("--prefix", default="episode", help="文件名前缀（默认 episode）")
    ap.add_argument("--digits", type=int, default=4, help="编号位数，默认 4 -> episode0001")
    ap.add_argument("--start", type=int, default=1, help="起始编号，默认 1")
    ap.add_argument("--exts", default="json,mp4", help="参与重命名的后缀，用逗号分隔")
    ap.add_argument("--apply", action="store_true", help="真正执行改名（默认仅预览）")
    args = ap.parse_args()

    folder = Path(args.dir)
    if not folder.is_dir():
        raise SystemExit(f"不是目录: {folder}")

    exts = [e.strip().lower() for e in args.exts.split(",")]
    exts = [("." + e) if not e.startswith(".") else e for e in exts]

    # 按 stem 分组（json/mp4 同组）
    groups = {}
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        groups.setdefault(p.stem, []).append(p)

    if not groups:
        print("未找到匹配文件")
        return

    items = [{"base": stem, "paths": paths} for stem, paths in groups.items()]
    items.sort(key=lambda x: extract_key(x["base"]))

    two_phase_rename(
        items,
        prefix=args.prefix,
        digits=args.digits,
        start=args.start,
        dry_run=not args.apply,
    )

if __name__ == "__main__":
    main()
