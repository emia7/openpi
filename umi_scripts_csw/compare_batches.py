#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Compare two UMI episode JSON batches and diagnose why one trains and the other doesn't.

Expected file pattern:
  episodeXXXXX.json (X digits)
Expected JSON structure (typical):
{
  "fps": <float, optional>,
  "records": [
    {"pose": [x,y,z,qx,qy,qz,qw], "clamp": <optional>, "timestamp": <optional>, ...},
    ...
  ]
}

This script produces:
- A text summary (printed to stdout)
- A JSON report saved to --out (default: compare_report.json)
- Histograms saved to --plot_dir (default: compare_plots)

Key diagnostics:
- Pose range and scale (pos, quat norm, Euler / rot magnitude)
- Step-to-step delta statistics (Δpos, Δrot)
- Timestamp/fps sanity
- Missing fields consistency (clamp present? timestamps present?)
- Episode length distribution
- Outlier detection (flags) and counts

Usage:
  python compare_batches.py --dir_a /path/to/batchA --dir_b /path/to/batchB
'''

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt

try:
    from scipy.spatial.transform import Rotation as R
except Exception:
    R = None


def list_eps(dir_path: Path, pattern: str) -> List[Path]:
    return sorted([p for p in dir_path.glob(pattern) if p.is_file()])


def load_episode(json_path: Path) -> Dict[str, Any]:
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    records = meta.get("records", None)
    if not isinstance(records, list) or len(records) == 0:
        raise ValueError("missing/empty records")

    poses_obj = [r.get("pose", None) for r in records]
    if any(p is None for p in poses_obj):
        raise ValueError("some records missing pose")

    poses = np.asarray(poses_obj, dtype=np.float32)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"pose shape {poses.shape} != (T,7)")

    clamp_present = "clamp" in records[0]
    if clamp_present:
        clamp = np.asarray([r.get("clamp", np.nan) for r in records], dtype=np.float32).reshape(-1)
    else:
        clamp = None

    ts = np.asarray([r.get("timestamp", np.nan) for r in records], dtype=np.float64)
    fps = float(meta.get("fps", np.nan)) if meta.get("fps", None) is not None else np.nan

    return {
        "path": str(json_path),
        "T": int(len(records)),
        "fps": fps,
        "ts": ts,
        "poses": poses,
        "clamp": clamp,
        "clamp_present": clamp_present,
    }


def quat_norm_stats(quat: np.ndarray) -> Dict[str, float]:
    n = np.linalg.norm(quat, axis=1)
    return {
        "quat_norm_mean": float(np.mean(n)),
        "quat_norm_std": float(np.std(n)),
        "quat_norm_min": float(np.min(n)),
        "quat_norm_max": float(np.max(n)),
        "quat_norm_p99": float(np.percentile(n, 99)),
    }


def delta_stats(pos: np.ndarray, quat_xyzw: np.ndarray) -> Dict[str, float]:
    dpos = np.diff(pos, axis=0)
    dpos_norm = np.linalg.norm(dpos, axis=1) if dpos.shape[0] else np.array([0.0], dtype=np.float32)

    out = {
        "dpos_norm_mean": float(np.mean(dpos_norm)),
        "dpos_norm_std": float(np.std(dpos_norm)),
        "dpos_norm_p99": float(np.percentile(dpos_norm, 99)),
        "dpos_norm_max": float(np.max(dpos_norm)),
    }

    if R is not None and quat_xyzw.shape[0] >= 2:
        q = quat_xyzw.copy()
        qn = np.linalg.norm(q, axis=1, keepdims=True)
        qn[qn == 0] = 1.0
        q = q / qn
        r = R.from_quat(q)   # SciPy expects xyzw
        rel = r[:-1].inv() * r[1:]
        drot = rel.as_rotvec()
        drot_norm = np.linalg.norm(drot, axis=1)
        out.update({
            "drot_norm_mean": float(np.mean(drot_norm)),
            "drot_norm_std": float(np.std(drot_norm)),
            "drot_norm_p99": float(np.percentile(drot_norm, 99)),
            "drot_norm_max": float(np.max(drot_norm)),
        })
    else:
        out.update({
            "drot_norm_mean": float("nan"),
            "drot_norm_std": float("nan"),
            "drot_norm_p99": float("nan"),
            "drot_norm_max": float("nan"),
        })
    return out


def time_stats(ts: np.ndarray, fps: float) -> Dict[str, float]:
    finite = np.isfinite(ts)
    out: Dict[str, float] = {}
    if finite.sum() >= 5:
        t = ts[finite]
        dt = np.diff(t)
        out["ts_dt_median"] = float(np.median(dt))
        out["ts_dt_p99"] = float(np.percentile(dt, 99))
        out["ts_dt_min"] = float(np.min(dt))
        out["ts_dt_max"] = float(np.max(dt))
        if out["ts_dt_median"] > 1e-9:
            out["ts_fps_est"] = float(1.0 / out["ts_dt_median"])
    else:
        out["ts_dt_median"] = float("nan")
        out["ts_fps_est"] = float("nan")
    out["fps_meta"] = float(fps) if np.isfinite(fps) else float("nan")
    return out


def summarize_batch(files: List[Path], max_episodes: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    per_eps: List[Dict[str, Any]] = []
    failures = 0
    for p in files[:max_episodes]:
        try:
            ep = load_episode(p)
            poses = ep["poses"]
            pos = poses[:, :3]
            quat = poses[:, 3:7]

            qn = np.linalg.norm(quat, axis=1, keepdims=True)
            qn[qn == 0] = 1.0
            quat_n = quat / qn

            stats = {
                "file": p.name,
                "T": ep["T"],
                "fps": float(ep["fps"]) if np.isfinite(ep["fps"]) else float("nan"),
                "clamp_present": bool(ep["clamp_present"]),
                "pos_min": pos.min(axis=0).tolist(),
                "pos_max": pos.max(axis=0).tolist(),
                "pos_std": pos.std(axis=0).tolist(),
            }
            stats.update(quat_norm_stats(quat))
            stats.update(delta_stats(pos, quat_n))
            stats.update(time_stats(ep["ts"], ep["fps"]))

            stats["flag_quat_not_unit_p99"] = bool(stats["quat_norm_p99"] > 1.05 or stats["quat_norm_p99"] < 0.95)
            stats["flag_big_step_pos"] = bool(stats["dpos_norm_p99"] > 0.05)  # >5cm per step suspicious
            stats["flag_big_step_rot"] = bool(np.isfinite(stats["drot_norm_p99"]) and stats["drot_norm_p99"] > 0.5)  # >0.5rad per step
            per_eps.append(stats)
        except Exception as e:
            failures += 1
            per_eps.append({"file": p.name, "error": repr(e)})

    ok = [r for r in per_eps if "error" not in r]
    agg: Dict[str, Any] = {
        "episodes_total": len(files),
        "episodes_analyzed": min(len(files), max_episodes),
        "episodes_failed_to_parse": failures,
        "clamp_present_ratio": float(np.mean([r.get("clamp_present", False) for r in ok])) if ok else float("nan"),
        "T_mean": float(np.mean([r["T"] for r in ok])) if ok else float("nan"),
        "T_min": int(np.min([r["T"] for r in ok])) if ok else 0,
        "T_max": int(np.max([r["T"] for r in ok])) if ok else 0,
        "fps_meta_unique": sorted({round(r["fps"], 6) for r in ok if np.isfinite(r["fps"])}),
        "ts_fps_est_median": float(np.nanmedian([r.get("ts_fps_est", np.nan) for r in ok])) if ok else float("nan"),
        "dpos_norm_p99_median": float(np.nanmedian([r.get("dpos_norm_p99", np.nan) for r in ok])) if ok else float("nan"),
        "drot_norm_p99_median": float(np.nanmedian([r.get("drot_norm_p99", np.nan) for r in ok])) if ok else float("nan"),
        "quat_norm_p99_median": float(np.nanmedian([r.get("quat_norm_p99", np.nan) for r in ok])) if ok else float("nan"),
        "flags": {
            "quat_not_unit_count": int(np.sum([r.get("flag_quat_not_unit_p99", False) for r in ok])),
            "big_step_pos_count": int(np.sum([r.get("flag_big_step_pos", False) for r in ok])),
            "big_step_rot_count": int(np.sum([r.get("flag_big_step_rot", False) for r in ok])),
        },
    }
    return agg, per_eps


def plot_hist(values, title, xlabel, out_png: Path):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(values, bins=50)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir_a", type=str, required=True, help="Batch A directory (e.g., 0-199)")
    p.add_argument("--dir_b", type=str, required=True, help="Batch B directory (e.g., 1-101)")
    p.add_argument("--pattern", type=str, default="episode*.json", help="Glob pattern")
    p.add_argument("--max_episodes", type=int, default=999999, help="Analyze at most N episodes from each batch")
    p.add_argument("--out", type=str, default="compare_report.json", help="Output JSON report")
    p.add_argument("--plot_dir", type=str, default="compare_plots", help="Directory to save histograms")
    args = p.parse_args()

    dir_a = Path(args.dir_a)
    dir_b = Path(args.dir_b)

    files_a = list_eps(dir_a, args.pattern)
    files_b = list_eps(dir_b, args.pattern)

    if not files_a:
        raise FileNotFoundError(f"No files in {dir_a} matching {args.pattern}")
    if not files_b:
        raise FileNotFoundError(f"No files in {dir_b} matching {args.pattern}")

    agg_a, per_a = summarize_batch(files_a, args.max_episodes)
    agg_b, per_b = summarize_batch(files_b, args.max_episodes)

    report = {
        "batch_a": {"dir": str(dir_a), "aggregate": agg_a, "per_episode": per_a},
        "batch_b": {"dir": str(dir_b), "aggregate": agg_b, "per_episode": per_b},
    }

    def fmt(x):
        if x is None:
            return "NA"
        if isinstance(x, float):
            if math.isnan(x):
                return "nan"
            return f"{x:.6g}"
        return str(x)

    print("=" * 80)
    print("Batch comparison summary")
    print("=" * 80)
    print(f"A: {dir_a}  files={len(files_a)} analyzed={agg_a['episodes_analyzed']} parse_fail={agg_a['episodes_failed_to_parse']}")
    print(f"B: {dir_b}  files={len(files_b)} analyzed={agg_b['episodes_analyzed']} parse_fail={agg_b['episodes_failed_to_parse']}")
    print("-" * 80)
    print("Key stats (median across episodes):")
    print(f"  clamp_present_ratio:  A={fmt(agg_a['clamp_present_ratio'])}  B={fmt(agg_b['clamp_present_ratio'])}")
    print(f"  T_mean:               A={fmt(agg_a['T_mean'])}  B={fmt(agg_b['T_mean'])}")
    print(f"  ts_fps_est_median:    A={fmt(agg_a['ts_fps_est_median'])}  B={fmt(agg_b['ts_fps_est_median'])}")
    print(f"  dpos_norm_p99_median: A={fmt(agg_a['dpos_norm_p99_median'])}  B={fmt(agg_b['dpos_norm_p99_median'])}")
    print(f"  drot_norm_p99_median: A={fmt(agg_a['drot_norm_p99_median'])}  B={fmt(agg_b['drot_norm_p99_median'])}")
    print(f"  quat_norm_p99_median: A={fmt(agg_a['quat_norm_p99_median'])}  B={fmt(agg_b['quat_norm_p99_median'])}")
    print("-" * 80)
    print("Red-flag counts (episodes):")
    print(f"  quat_not_unit: A={agg_a['flags']['quat_not_unit_count']}  B={agg_b['flags']['quat_not_unit_count']}")
    print(f"  big_step_pos:  A={agg_a['flags']['big_step_pos_count']}  B={agg_b['flags']['big_step_pos_count']}")
    print(f"  big_step_rot:  A={agg_a['flags']['big_step_rot_count']}  B={agg_b['flags']['big_step_rot_count']}")
    print("=" * 80)

    out_path = Path(args.out)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved report: {out_path.resolve()}")

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    ok_a = [r for r in per_a if "error" not in r]
    ok_b = [r for r in per_b if "error" not in r]

    def get(series, key):
        return [r.get(key, np.nan) for r in series]

    plot_hist(get(ok_a, "T"), "Episode length T (Batch A)", "T (steps)", plot_dir / "A_T.png")
    plot_hist(get(ok_b, "T"), "Episode length T (Batch B)", "T (steps)", plot_dir / "B_T.png")
    plot_hist(get(ok_a, "dpos_norm_p99"), "Δpos norm p99 per episode (Batch A)", "meters", plot_dir / "A_dpos_p99.png")
    plot_hist(get(ok_b, "dpos_norm_p99"), "Δpos norm p99 per episode (Batch B)", "meters", plot_dir / "B_dpos_p99.png")
    plot_hist(get(ok_a, "drot_norm_p99"), "Δrot norm p99 per episode (Batch A)", "radians", plot_dir / "A_drot_p99.png")
    plot_hist(get(ok_b, "drot_norm_p99"), "Δrot norm p99 per episode (Batch B)", "radians", plot_dir / "B_drot_p99.png")
    plot_hist(get(ok_a, "quat_norm_p99"), "Quaternion norm p99 per episode (Batch A)", "norm", plot_dir / "A_quatnorm_p99.png")
    plot_hist(get(ok_b, "quat_norm_p99"), "Quaternion norm p99 per episode (Batch B)", "norm", plot_dir / "B_quatnorm_p99.png")
    plot_hist(get(ok_a, "ts_fps_est"), "Timestamp-estimated FPS (Batch A)", "Hz", plot_dir / "A_ts_fps.png")
    plot_hist(get(ok_b, "ts_fps_est"), "Timestamp-estimated FPS (Batch B)", "Hz", plot_dir / "B_ts_fps.png")

    print(f"Saved plots to: {plot_dir.resolve()}")


if __name__ == "__main__":
    main()
