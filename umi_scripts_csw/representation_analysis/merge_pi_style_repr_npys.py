#!/usr/bin/env python3
"""
将多台机器分别跑出的 ``teleop_vs_umi_vlm_repr_pi_style.py --groups ... --skip-tsne``
产生的 ``*.npy`` 拷到同一目录（或两次 ``--extra-dir``），再联合做 t-SNE。

示例::

    # 在 gxzx2 得到 teleop_model_*.npy，在 zx1 得到 umi_model_*.npy，拷到 merge_out/
    PYTHONPATH=src:. python umi_scripts_csw/representation_analysis/merge_pi_style_repr_npys.py \\
      --input-dir ./merge_out --seed 0 --out-dir ./merge_out_merged
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from umi_scripts_csw.representation_analysis.legend_config import legend_labels_for_groups

ALL_GROUPS = (
    "teleop_model_robot_data",
    "teleop_model_umi_data",
    "umi_model_robot_data",
    "umi_model_umi_data",
    "umi_model_umi_other_data",
    "umi_model_umi_bagging_data",
)


def main():
    ap = argparse.ArgumentParser(description="合并分机器导出的 PI-style .npy 并画 t-SNE")
    ap.add_argument(
        "--input-dir",
        action="append",
        dest="input_dirs",
        default=None,
        help="可多次指定：从这些目录收集同名 .npy（后者覆盖前者）",
    )
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dirs = args.input_dirs or []
    if not dirs:
        raise SystemExit("请至少指定一次 --input-dir")

    collected: dict[str, Path] = {}
    for d in dirs:
        base = Path(d).resolve()
        for name in ALL_GROUPS:
            p = base / f"{name}.npy"
            if p.is_file():
                collected[name] = p

    if len(collected) < 2:
        raise SystemExit(f"至少需要 2 个 .npy，当前找到: {sorted(collected.keys())}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = sorted(collected.keys())
    plot_labels = legend_labels_for_groups(names)
    blocks = [np.load(collected[n]) for n in names]
    lens = [b.shape[0] for b in blocks]
    Z = np.concatenate(blocks, axis=0)
    n_z = Z.shape[0]
    perp = min(30, max(2, n_z // 4), n_z - 1)
    Z2 = TSNE(n_components=2, perplexity=perp, random_state=args.seed).fit_transform(Z)

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = matplotlib.colormaps["tab10"].resampled(max(len(names), 1))
    start = 0
    for i, n in enumerate(names):
        m = slice(start, start + lens[i])
        ax.scatter(Z2[m, 0], Z2[m, 1], s=14, alpha=0.75, color=cmap(i), label=plot_labels[i])
        start += lens[i]
    ax.legend(markerscale=2, fontsize=8)
    ax.set_title("t-SNE merged (VLM/vision .npy from teleop_vs_umi_vlm_repr_pi_style.py)")
    plt.tight_layout()
    plt.savefig(out_dir / "tsne_pi_style_vlm_merged.png", dpi=150)
    plt.close()

    for n in names:
        np.save(out_dir / f"{n}.npy", np.load(collected[n]))

    report = {
        "sources": {k: str(collected[k]) for k in names},
        "counts": dict(zip(names, lens)),
        "seed": args.seed,
    }
    (out_dir / "merge_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Wrote", out_dir / "tsne_pi_style_vlm_merged.png")


if __name__ == "__main__":
    main()
