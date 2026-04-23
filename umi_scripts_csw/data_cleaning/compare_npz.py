import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


D_KEEP = 32  # ✅ 只取前10维做比较（state 和 actions 都一样）


def _load_npz(path: str):
    d = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    return out


def _summarize(name, x):
    x = np.asarray(x)
    if x.dtype == object:
        return f"{name}: object array, shape={x.shape}"
    if x.size == 0:
        return f"{name}: empty, shape={x.shape}"
    flat = x.reshape(-1)
    return (
        f"{name}: shape={x.shape}, dtype={x.dtype}, "
        f"mean={flat.mean():.6g}, std={flat.std():.6g}, "
        f"p1={np.percentile(flat,1):.6g}, p50={np.percentile(flat,50):.6g}, p99={np.percentile(flat,99):.6g}"
    )


def _plot_series(out_dir: Path, title: str, t, a, b, ylabel: str, fname: str):
    plt.figure(figsize=(12, 4))
    plt.plot(t, a, label="A", linewidth=1.5)
    plt.plot(t, b, label="B", linewidth=1.5, linestyle="--")
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=150)
    plt.close()


def _heatmap(out_dir: Path, title: str, mat, fname: str):
    # mat: (T, D)
    plt.figure(figsize=(12, 5))
    plt.imshow(mat.T, aspect="auto", origin="lower")
    plt.title(title)
    plt.xlabel("time index")
    plt.ylabel("dim")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir / fname, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Path to npz A")
    ap.add_argument("--b", required=True, help="Path to npz B")
    ap.add_argument("--out", default="npz_compare_out", help="Output dir")

    ap.add_argument("--max-h", type=int, default=-1, help="Plot first N horizon steps (<=0 means all)")
    ap.add_argument(
        "--save-action-heatmaps",
        action="store_true",
        help="Save heatmaps for each horizon step (A and B separately).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    A = _load_npz(args.a)
    B = _load_npz(args.b)

    print("=== Keys A ===", sorted(A.keys()))
    print("=== Keys B ===", sorted(B.keys()))

    # --- choose t ---
    tA = A.get("t", None)
    tB = B.get("t", None)
    if tA is None or tB is None:
        TA = A["state"].shape[0]
        TB = B["state"].shape[0]
        T = min(TA, TB)
        t = np.arange(T, dtype=np.float32)
        print("[WARN] Missing t; using index as time.")
    else:
        T = min(len(tA), len(tB))
        t = np.asarray(tA[:T], dtype=np.float32)

    # --- print summaries ---
    print(_summarize("A.state", A.get("state")))
    print(_summarize("B.state", B.get("state")))
    print(_summarize("A.actions", A.get("actions")))
    print(_summarize("B.actions", B.get("actions")))

    # --- STATE plots (只取前10维) ---
    '''
    stateA = np.asarray(A["state"][:T], dtype=np.float32)
    stateB = np.asarray(B["state"][:T], dtype=np.float32)
    Ds = min(stateA.shape[1], stateB.shape[1])
    Ds_plot = min(Ds, D_KEEP)

    for d in range(Ds_plot):
        _plot_series(
            out_dir,
            title=f"STATE dim {d}",
            t=t,
            a=stateA[:, d],
            b=stateB[:, d],
            ylabel=f"state[{d}]",
            fname=f"state_dim{d:02d}.png",
        )
    '''
    # --- ACTION plots: ALL horizon steps, 只取前10维 ---
    actionsA = A.get("actions", None)
    actionsB = B.get("actions", None)

    if actionsA is None or actionsB is None:
        print("[WARN] actions missing; skipped action comparisons.")
        return

    if np.asarray(actionsA).dtype == object or np.asarray(actionsB).dtype == object:
        print("[WARN] actions is object dtype; skipped action comparisons.")
        return

    actA = np.asarray(actionsA[:T], dtype=np.float32)  # (T,H,D)
    actB = np.asarray(actionsB[:T], dtype=np.float32)

    H = min(actA.shape[1], actB.shape[1])
    Da = min(actA.shape[2], actB.shape[2])
    Da_plot = min(Da, D_KEEP)

    if args.max_h and args.max_h > 0:
        H = min(H, args.max_h)

    action_root = out_dir / "actions_by_h"
    action_root.mkdir(parents=True, exist_ok=True)

    if args.save_action_heatmaps:
        (action_root / "heatmaps_A").mkdir(parents=True, exist_ok=True)
        (action_root / "heatmaps_B").mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Plotting actions: T={T}, H={H}, D_plot={Da_plot} (only first {D_KEEP} dims)")
    for h in range(H):
        h_dir = action_root / f"h{h:02d}"
        h_dir.mkdir(parents=True, exist_ok=True)

        if args.save_action_heatmaps:
            _heatmap(
                action_root / "heatmaps_A",
                f"A action heatmap @ h={h} (first {Da_plot} dims)",
                actA[:, h, :Da_plot],
                f"A_h{h:02d}.png",
            )
            _heatmap(
                action_root / "heatmaps_B",
                f"B action heatmap @ h={h} (first {Da_plot} dims)",
                actB[:, h, :Da_plot],
                f"B_h{h:02d}.png",
            )

        for d in range(Da_plot):
            _plot_series(
                h_dir,
                title=f"ACTION h={h} dim {d}",
                t=t,
                a=actA[:, h, d],
                b=actB[:, h, d],
                ylabel=f"act[{h},{d}]",
                fname=f"dim{d:02d}.png",
            )

        if h % 2 == 0:
            print(f"done h={h}/{H-1}", end="\r")

    print("\n[OK] Wrote outputs to:", out_dir.resolve())


if __name__ == "__main__":
    main()
