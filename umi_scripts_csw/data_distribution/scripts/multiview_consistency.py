"""
多视角嵌入一致性：冻结 SigLIP，计算成对余弦与（可选）单路时序平滑度。

重要说明（勿当作唯一质检标准）：
  左腕、右腕、第三视角视场与内容差异大，不得期待两两余弦普遍很高。
  本脚本依赖 (1) 批内相对分位、(2) 同 episode 内时间突刺、(3) 单路 t–t+1 连续性；
  不要用单一绝对阈值宣布「好/坏」。L–B 与 L–R 等指标须分开解读，不得混为单一「相似度」。

用法（仓库根目录，PYTHONPATH 含 src）::

  python -m umi_scripts_csw.data_distribution.scripts.multiview_consistency \\
    --config-name pi05_xv_dual_finetune \\
    --max-samples 20000 --seed 0 --encoder openpi_siglip --temporal \\
    --out-dir ./reports/multiview_consistency_run001
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pathlib
import sys
from typing import Any
import math

import jax
import numpy as np
import pandas as pd

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

from umi_scripts_csw.data_distribution.scripts import _embeddings

logger = logging.getLogger(__name__)

_EP_KEYS = ("episode_index", "episode_id", "episode")
_FRAME_KEYS = ("frame_index", "index", "frame")


def _collate(items: list[dict]) -> dict:
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _episode_frame_from_sample(s: dict) -> tuple[int | None, int | None]:
    ep: int | None = None
    for k in _EP_KEYS:
        if k in s:
            ep = int(_to_numpy(s[k]))
            break
    fi: int | None = None
    for k in _FRAME_KEYS:
        if k in s:
            fi = int(_to_numpy(s[k]))
            break
    return ep, fi


def _same_episode_consecutive(ds, g: int, n_total: int) -> bool:
    if g < 0 or g + 1 >= n_total:
        return False
    e0, _ = _episode_frame_from_sample(ds[int(g)])
    e1, _ = _episode_frame_from_sample(ds[int(g + 1)])
    if e0 is not None and e1 is not None:
        return e0 == e1
    # Transforms often drop episode_index; assume dataset row order is chronological (may blur episode boundaries).
    return True


def _nan_fraction(a: np.ndarray) -> float:
    return float(np.mean(np.isnan(a)))


def _percentiles(a: np.ndarray, qs: tuple[float, ...]) -> dict[str, float]:
    out = {}
    for q in qs:
        key = f"p{int(q * 100)}"
        out[key] = float(np.nanpercentile(a, q * 100))
    return out


def _robust_z(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad + 1e-12
    return (x - med) / scale


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _mad_1d(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description=(
            "Multiview embedding consistency (frozen SigLIP): pairwise cosines and optional "
            "per-view temporal cosine at t vs t+1. "
            "Do NOT interpret high pairwise cosine as universally expected — see --help long text."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "读数说明：第三视角与手腕内容差异大，L–B、R–B 余弦绝对值常偏低、方差大属预期。"
            "请使用批内分位、episode 内突刺与单路时序连续性联合判断；"
            "不用几何重投影时本指标弱于标定；禁止默认用绝对余弦阈值断言好坏。"
        ),
    )
    p.add_argument("--config-name", type=str, default="pi05_xv_dual_finetune")
    p.add_argument("--max-samples", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--encoder", choices=("openpi_siglip",), default="openpi_siglip")
    p.add_argument("--temporal", action="store_true", help="Compute s_L,s_R,s_B vs next frame in same episode.")
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--repo-id", type=str, default=None)
    p.add_argument("--asset-id", type=str, default=None)
    p.add_argument("--assets-base-dir", type=str, default=None)
    p.add_argument("--inference-batch-size", type=int, default=8)
    p.add_argument("--every-k-frames", type=int, default=1)
    p.add_argument(
        "--global-quantile",
        type=float,
        default=0.1,
        help="Episode flagged if min(c_LB) is below this quantile of all per-frame c_LB.",
    )
    p.add_argument(
        "--episode-mad-k",
        type=float,
        default=0.0,
        help="If >0, also flag episode when min(c_LB) < median(c_LB in ep) - k * MAD(c_LB in ep).",
    )
    p.add_argument(
        "--frame-z-threshold",
        type=float,
        default=4.0,
        help="Robust z-score threshold (negative tail) for per-frame c_LB / s_B outliers.",
    )
    p.add_argument(
        "--absolute-threshold",
        type=float,
        default=None,
        help="If set, also flag frames with c_LR below this value (NOT recommended; default off).",
    )
    args = p.parse_args(argv)

    if args.encoder != "openpi_siglip":
        raise SystemExit("Only --encoder openpi_siglip is implemented.")
    if args.absolute_threshold is not None:
        logger.warning(
            "--absolute-threshold is set; absolute cos thresholds are discouraged (see plan §5–6)."
        )

    train_cfg = _config.get_config(args.config_name)
    if args.assets_base_dir is not None:
        train_cfg = dataclasses.replace(train_cfg, assets_base_dir=args.assets_base_dir)

    data_factory = train_cfg.data
    if args.repo_id is not None:
        data_factory = dataclasses.replace(data_factory, repo_id=args.repo_id)
    if args.repo_id is not None or args.asset_id is not None:
        asset_id = args.asset_id
        if asset_id is None and args.repo_id is not None:
            asset_id = pathlib.Path(args.repo_id.rstrip("/")).name
        if asset_id is not None:
            new_assets = dataclasses.replace(data_factory.assets, asset_id=asset_id)
            data_factory = dataclasses.replace(data_factory, assets=new_assets)
        train_cfg = dataclasses.replace(train_cfg, data=data_factory)

    model_cfg = train_cfg.model
    data_cfg = train_cfg.data.create(train_cfg.assets_dirs, model_cfg)

    base_ds = _data_loader.create_torch_dataset(data_cfg, model_cfg.action_horizon, model_cfg)
    try:
        ds = _data_loader.transform_dataset(base_ds, data_cfg, skip_norm_stats=False)
    except ValueError as e:
        logger.error("%s", e)
        sys.exit(
            "Normalization stats missing. Run:\n"
            "  uv run scripts/compute_norm_stats.py --config-name=<your-config>\n"
            "with the same data / assets paths as this analysis."
        )

    n_total = len(ds)
    rng = np.random.default_rng(args.seed)
    stride = max(1, args.every_k_frames)
    pool = np.arange(0, n_total, stride, dtype=np.int64)
    n_take = min(args.max_samples, pool.size)
    if n_take < 2:
        raise SystemExit(f"Not enough samples after stride/max-samples (got {n_take}, need >= 2).")

    perm = rng.permutation(pool.size)[:n_take]
    indices = pool[perm].tolist()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_cfg = {
        "config_name": args.config_name,
        "repo_id": data_cfg.repo_id,
        "asset_id": data_cfg.asset_id,
        "max_samples": n_take,
        "seed": args.seed,
        "encoder": args.encoder,
        "temporal": bool(args.temporal),
        "every_k_frames": stride,
        "global_quantile": args.global_quantile,
        "episode_mad_k": args.episode_mad_k,
        "frame_z_threshold": args.frame_z_threshold,
        "absolute_threshold": args.absolute_threshold,
        "image_keys": list(_model.IMAGE_KEYS),
        "weight_loader": str(train_cfg.weight_loader),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    logger.info("Loading JAX Pi0 + checkpoint weights (SigLIP tower)...")
    model = _embeddings.load_pi_model_from_train_config(train_cfg)

    bs = max(1, min(args.inference_batch_size, n_take))

    rows: list[dict[str, Any]] = []

    for start in range(0, n_take, bs):
        chunk = indices[start : start + bs]
        batch_dict = _collate([ds[int(i)] for i in chunk])
        obs = _model.Observation.from_dict(batch_dict)
        obs = _model.preprocess_observation(None, obs, train=False)
        z_l, z_r, z_b = _embeddings.encode_lr_base_l2_normalized(model, obs)
        c_lr = np.sum(z_l * z_r, axis=-1)
        c_lb = np.sum(z_l * z_b, axis=-1)
        c_rb = np.sum(z_r * z_b, axis=-1)

        s_l = np.full(len(chunk), np.nan)
        s_r = np.full(len(chunk), np.nan)
        s_b = np.full(len(chunk), np.nan)

        if args.temporal:
            need_next: list[int] = []
            pos_for_next: list[tuple[int, int]] = []
            for row_k, g in enumerate(chunk):
                if _same_episode_consecutive(ds, int(g), n_total):
                    need_next.append(int(g) + 1)
                    pos_for_next.append((row_k, int(g) + 1))
            if need_next:
                uniq = sorted(set(need_next))
                next_batch = _collate([ds[int(j)] for j in uniq])
                obs_n = _model.Observation.from_dict(next_batch)
                obs_n = _model.preprocess_observation(None, obs_n, train=False)
                zn_l, zn_r, zn_b = _embeddings.encode_lr_base_l2_normalized(model, obs_n)
                # map global index -> row in uniq batch
                idx_map = {uniq[r]: r for r in range(len(uniq))}
                for row_k, gn in pos_for_next:
                    r = idx_map[gn]
                    s_l[row_k] = float(np.dot(z_l[row_k], zn_l[r]))
                    s_r[row_k] = float(np.dot(z_r[row_k], zn_r[r]))
                    s_b[row_k] = float(np.dot(z_b[row_k], zn_b[r]))

        for row_k, g in enumerate(chunk):
            ep, fi = _episode_frame_from_sample(ds[int(g)])
            row: dict[str, Any] = {
                "global_index": int(g),
                "episode_id": ep,
                "frame_in_episode": fi,
                "c_LR": float(c_lr[row_k]),
                "c_LB": float(c_lb[row_k]),
                "c_RB": float(c_rb[row_k]),
            }
            if args.temporal:
                row["s_L"] = float(s_l[row_k])
                row["s_R"] = float(s_r[row_k])
                row["s_B"] = float(s_b[row_k])
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "per_frame_metrics.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Wrote %s", csv_path)

    # Summary stats
    qs = (0.1, 0.5, 0.9)
    summary: dict[str, Any] = {
        "n_samples": int(n_take),
        "n_total_dataset": int(n_total),
        "percentiles": {},
        "nan_fraction": {},
    }
    for name, col in [("c_LR", "c_LR"), ("c_LB", "c_LB"), ("c_RB", "c_RB")]:
        a = df[col].to_numpy()
        summary["nan_fraction"][col] = _nan_fraction(a)
        summary["percentiles"][col] = _percentiles(a, qs)

    if args.temporal:
        summary["percentiles_temporal"] = {}
        summary["nan_fraction_temporal"] = {}
        for col in ("s_L", "s_R", "s_B"):
            a = df[col].to_numpy()
            summary["nan_fraction_temporal"][col] = _nan_fraction(a)
            summary["percentiles_temporal"][col] = _percentiles(a, qs)

    (out_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_dir / "summary.json")

    # Outliers
    c_lb_all = df["c_LB"].to_numpy(dtype=np.float64)
    global_cut = float(np.nanquantile(c_lb_all, args.global_quantile))

    ep_rows: list[dict[str, Any]] = []
    if df["episode_id"].notna().any():
        for eid, sub in df.groupby("episode_id"):
            if pd.isna(eid):
                continue
            mlb = sub["c_LB"].to_numpy()
            entry = {
                "episode_id": int(eid),
                "mean_c_LB": float(np.nanmean(mlb)),
                "min_c_LR": float(np.nanmin(sub["c_LR"].to_numpy())),
                "min_c_LB": float(np.nanmin(mlb)),
                "min_c_RB": float(np.nanmin(sub["c_RB"].to_numpy())),
            }
            reasons: list[str] = []
            if entry["min_c_LB"] < global_cut:
                reasons.append(f"min_c_LB < global_p{int(args.global_quantile * 100):d} ({global_cut:.6f})")
            if args.episode_mad_k > 0:
                mad = _mad_1d(mlb)
                med = float(np.nanmedian(mlb))
                thr = med - args.episode_mad_k * mad
                if entry["min_c_LB"] < thr:
                    reasons.append(f"min_c_LB < median - k*MAD ({thr:.6f})")
            if reasons:
                entry["reasons"] = reasons
                ep_rows.append(entry)

    frame_rows: list[dict[str, Any]] = []
    z_lb = _robust_z(df["c_LB"].to_numpy())
    df["_z_c_LB"] = z_lb
    mask_lb = z_lb < -args.frame_z_threshold
    for i in np.where(mask_lb)[0]:
        ep_i = df.iloc[i]["episode_id"]
        frame_rows.append(
            {
                "kind": "low_c_LB_robust_z",
                "global_index": int(df.iloc[i]["global_index"]),
                "episode_id": None if pd.isna(ep_i) else int(ep_i),
                "c_LB": float(df.iloc[i]["c_LB"]),
                "robust_z_c_LB": float(z_lb[i]),
            }
        )

    if args.temporal:
        for col, label in [("s_B", "low_s_B_robust_z"), ("s_L", "low_s_L_robust_z"), ("s_R", "low_s_R_robust_z")]:
            z = _robust_z(df[col].to_numpy())
            mask = z < -args.frame_z_threshold
            for i in np.where(mask)[0]:
                ep_i = df.iloc[i]["episode_id"]
                frame_rows.append(
                    {
                        "kind": label,
                        "global_index": int(df.iloc[i]["global_index"]),
                        "episode_id": None if pd.isna(ep_i) else int(ep_i),
                        col: float(df.iloc[i][col]),
                        f"robust_z_{col}": float(z[i]),
                    }
                )

    if args.absolute_threshold is not None:
        m = df["c_LR"].to_numpy() < args.absolute_threshold
        for i in np.where(m)[0]:
            frame_rows.append(
                {
                    "kind": "absolute_c_LR",
                    "global_index": int(df.iloc[i]["global_index"]),
                    "c_LR": float(df.iloc[i]["c_LR"]),
                    "threshold": args.absolute_threshold,
                }
            )

    outliers = {"episodes": ep_rows, "frames": frame_rows, "global_cut_c_LB": global_cut}
    (out_dir / "outliers.json").write_text(json.dumps(_json_safe(outliers), indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_dir / "outliers.json")

    # Plots
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for ax, col, title in zip(
            axes,
            ("c_LR", "c_LB", "c_RB"),
            ("cos L–R", "cos L–B", "cos R–B"),
        ):
            a = df[col].to_numpy()
            a = a[np.isfinite(a)]
            ax.hist(a, bins=50, range=(-1.05, 1.05), color="#4477aa", alpha=0.85)
            ax.set_title(title)
            ax.set_xlabel("cosine")
        fig.tight_layout()
        fig.savefig(out_dir / "hist_pairwise.png", dpi=150)
        plt.close(fig)
        logger.info("Wrote %s", out_dir / "hist_pairwise.png")

        if args.temporal:
            fig2, axes2 = plt.subplots(1, 3, figsize=(12, 3.5))
            for ax, col, title in zip(
                axes2,
                ("s_L", "s_R", "s_B"),
                ("s_L (t,t+1)", "s_R (t,t+1)", "s_B (t,t+1)"),
            ):
                a = df[col].to_numpy()
                a = a[np.isfinite(a)]
                ax.hist(a, bins=50, range=(-1.05, 1.05), color="#aa7744", alpha=0.85)
                ax.set_title(title)
                ax.set_xlabel("cosine")
            fig2.tight_layout()
            fig2.savefig(out_dir / "hist_temporal.png", dpi=150)
            plt.close(fig2)
            logger.info("Wrote %s", out_dir / "hist_temporal.png")
    except Exception as e:
        logger.warning("Could not save histograms: %s", e)


if __name__ == "__main__":
    main()
