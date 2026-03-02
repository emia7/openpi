import os
import re
import argparse
from pathlib import Path

NUM_RE = re.compile(r"(\d+)")

# 定义需要识别并保留的特殊后缀（长尾缀在前，防止误匹配）
SPECIAL_SUFFIXES =["_head", "_left", "_states"]

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

def get_base_and_extra(stem: str):
    """
    从文件名中分离基础部分和特殊后缀
    例如: 
    episode0001_head -> (episode0001, _head)
    episode0001      -> (episode0001, "")
    """
    for s in SPECIAL_SUFFIXES:
        if stem.endswith(s):
            return stem[:-len(s)], s
    return stem, ""

def two_phase_rename(items, prefix, digits, start, dry_run: bool):
    """
    items: [
      {
        "base": 原 base_stem (用于排序),
        "files": [ (Path对象, extra_suffix), ... ]
      }
    ]
    """
    mapping = []
    cur = start

    for item in items:
        # 生成新的基础名，例如 episode0001
        new_base = f"{prefix}{str(cur).zfill(digits)}"
        cur += 1
        
        for (src_path, extra) in item["files"]:
            # 拼接: 新基础名 + 原有的特殊后缀 + 原扩展名
            # e.g. episode0001 + _head + .mp4
            new_name = f"{new_base}{extra}{src_path.suffix}"
            mapping.append((src_path, src_path.with_name(new_name)))

    # 冲突检查
    targets = [dst for _, dst in mapping]
    if len(set(targets)) != len(targets):
        raise RuntimeError("目标文件名冲突，请检查是否有重名文件生成")

    # 两阶段改名（避免覆盖：A->tmp, B->A, tmp->B）
    tmp_mapping = []
    pid = os.getpid()
    for src, dst in mapping:
        # 如果源文件名和目标文件名完全一致，则跳过
        if src == dst:
            continue
        tmp = src.with_name(f".__tmp__{src.stem}__{pid}{src.suffix}")
        tmp_mapping.append((src, tmp, dst))

    if not tmp_mapping:
        print("[INFO] 没有文件需要改名 (文件名已符合规则)")
        return

    # 预览
    print(f"\n=== Rename Plan (Start from {start}) ===")
    for src, _, dst in tmp_mapping:
        print(f"{src.name}  ->  {dst.name}")
    print("===================\n")

    if dry_run:
        print("[DRY RUN] 未执行改名，请添加 --apply 参数执行")
        return

    # Phase 1: src -> tmp
    for src, tmp, _ in tmp_mapping:
        if tmp.exists():
            raise RuntimeError(f"临时文件已存在：{tmp}")
        src.rename(tmp)

    # Phase 2: tmp -> dst
    for _, tmp, dst in tmp_mapping:
        if dst.exists():
            raise RuntimeError(f"目标文件已存在：{dst}")
        tmp.rename(dst)

    print(f"[OK] 重命名完成，共处理 {len(tmp_mapping)} 个文件")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="目标文件夹")
    ap.add_argument("--prefix", default="episode", help="文件名前缀（默认 episode）")
    ap.add_argument("--digits", type=int, default=4, help="编号位数，默认 4 -> episode0001")
    ap.add_argument("--start", type=int, default=0, help="起始编号，默认 0")
    ap.add_argument("--exts", default="json,mp4,png", help="参与重命名的后缀，用逗号分隔")
    ap.add_argument("--apply", action="store_true", help="真正执行改名（默认仅预览）")
    args = ap.parse_args()

    folder = Path(args.dir)
    if not folder.is_dir():
        raise SystemExit(f"不是目录: {folder}")

    exts = [e.strip().lower() for e in args.exts.split(",")]
    exts = [("." + e) if not e.startswith(".") else e for e in exts]

    # 1. 扫描并分组
    # groups 结构: base_stem -> list of (path, extra_suffix)
    groups = {}
    
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
            
        # 分离出 episodeXXXX 和 _head/_left
        base, extra = get_base_and_extra(p.stem)
        groups.setdefault(base, []).append((p, extra))

    if not groups:
        print("未找到匹配文件")
        return

    # 2. 构造排序列表
    items = [{"base": base, "files": files} for base, files in groups.items()]
    
    # 按基础名中的数字排序 (确保 episode2 排在 episode10 前面)
    items.sort(key=lambda x: extract_key(x["base"]))

    # 3. 执行重命名
    two_phase_rename(
        items,
        prefix=args.prefix,
        digits=args.digits,
        start=args.start,
        dry_run=not args.apply,
    )

if __name__ == "__main__":
    main()
