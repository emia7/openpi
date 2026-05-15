"""
Episode-level trajectory similarity (DTW) for XV dual-arm LeRobot data.

See ``umi_scripts_csw/data_distribution/PLAN_TRAJECTORY_SIMILARITY.md`` and
``umi_scripts_csw/data_distribution/TRAJECTORY_SIMILARITY_WORKFLOW.md`` (reader-oriented).

Run from repo root (with ``src`` on ``PYTHONPATH``, same as training)::

    PYTHONPATH=src:. python -m umi_scripts_csw.data_distribution.scripts.trajectory_similarity \\
      --config-name pi05_xv_dual_finetune \\
      --representation policy20 \\
      --dataset-root /path/to/lerobot_dataset \\
      --out-dir ./reports/traj_sim_run001

For datasets with heavy MP4 features, add ``--parquet-lowdim-only`` (requires ``--dataset-root``)
to read only ``data/chunk-*/episode_*.parquet`` and skip video decode.

Use **one task per run** (or a pre-filtered subset). Mixed-task datasets yield misleading global distances.

**Primary outputs** (default ``--cluster-mode none``): pairwise distances ``distance_matrix.npz`` (``D``, ``episodes``),
row-sum **outliers**, and **MDS / dendrogram / t-SNE** (``tsne2d.png``) from ``D`` when plots are on — use ``--no-plot-tsne`` to skip t-SNE (requires scikit-learn). No discrete cluster labels by default.

Optional partitioning: ``--cluster-mode auto`` (silhouette sweep on ``D``) or ``manual`` with ``--n-clusters K``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from openpi.policies import xv_dual_policy
from openpi.training import config as _config
import openpi.transforms as transforms


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _dtw_multivariate(
    x: np.ndarray,
    y: np.ndarray,
    *,
    p: float = 2.0,
    band_ratio: float | None = None,
) -> float:
    """Multivariate DTW cost; symmetric. x: (T1, D), y: (T2, D)."""
    from scipy.spatial.distance import cdist

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    t1, d1 = x.shape
    t2, d2 = y.shape
    if d1 != d2:
        raise ValueError(f"Feature dim mismatch: {d1} vs {d2}")
    local = cdist(x, y, metric="minkowski", p=p)
    inf = 1e30
    acc = np.full((t1 + 1, t2 + 1), inf, dtype=np.float64)
    acc[0, 0] = 0.0
    if band_ratio is None or band_ratio <= 0:
        for i in range(1, t1 + 1):
            for j in range(1, t2 + 1):
                c = local[i - 1, j - 1]
                acc[i, j] = c + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
        return float(acc[t1, t2])

    r = max(1, int(np.ceil(band_ratio * max(t1, t2))))
    for i in range(1, t1 + 1):
        j_lo = max(1, i - r)
        j_hi = min(t2, i + r)
        for j in range(1, t2 + 1):
            if j < j_lo or j > j_hi:
                continue
            c = local[i - 1, j - 1]
            acc[i, j] = c + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    return float(acc[t1, t2])


def _resample_uniform(traj: np.ndarray, l_fixed: int) -> np.ndarray:
    t, d = traj.shape
    if t == l_fixed:
        return traj.astype(np.float32)
    t_old = np.linspace(0.0, 1.0, t)
    t_new = np.linspace(0.0, 1.0, l_fixed)
    out = np.empty((l_fixed, d), dtype=np.float32)
    for di in range(d):
        out[:, di] = np.interp(t_new, t_old, traj[:, di].astype(np.float64)).astype(np.float32)
    return out


def _mean_lp_distance(x: np.ndarray, y: np.ndarray, p: float = 2.0) -> float:
    d = np.linalg.norm(x - y, ord=p, axis=-1)
    return float(np.mean(d))


def _episode_indices(ds, episode_index: int, length: int) -> list[int]:
    ep_key_candidates = ("episode_index", "episode_id", "episode")
    found_start = None
    found_key = None
    for i in range(len(ds)):
        s = ds[i]
        for k in ep_key_candidates:
            if k in s and int(_to_numpy(s[k])) == int(episode_index):
                found_start = i
                found_key = k
                break
        if found_start is not None:
            break
    if found_start is None:
        raise ValueError(f"Episode {episode_index} not found when scanning dataset.")

    idxs: list[int] = []
    for j in range(found_start, len(ds)):
        s = ds[j]
        if found_key not in s or int(_to_numpy(s[found_key])) != int(episode_index):
            break
        idxs.append(j)
        if length > 0 and len(idxs) >= length:
            break
    return idxs


def _list_episodes(ds) -> list[tuple[int, int]]:
    meta = ds.meta
    episodes = meta["episodes"] if isinstance(meta, dict) else getattr(meta, "episodes", None)
    out: list[tuple[int, int]] = []
    if episodes is None:
        return out
    if hasattr(episodes, "to_dicts"):
        for row in episodes.to_dicts():
            eid = int(row.get("episode_index", row.get("episode", len(out))))
            ln = int(row.get("length", 0))
            out.append((eid, ln))
        return out
    if hasattr(episodes, "iter_rows"):
        for row in episodes.iter_rows(named=True):
            eid = int(row.get("episode_index", row.get("episode", len(out))))
            ln = int(row.get("length", 0))
            out.append((eid, ln))
        return out
    if hasattr(episodes, "iloc"):
        import pandas as pd

        pdf = episodes if isinstance(episodes, pd.DataFrame) else episodes.to_pandas()
        for i in range(len(pdf)):
            r = pdf.iloc[i]
            eid = int(r.get("episode_index", r.get("episode", i)))
            ln = int(r.get("length", 0))
            out.append((eid, ln))
        return out
    if isinstance(episodes, dict):
        for k, v in episodes.items():
            if isinstance(v, dict):
                eid = int(v.get("episode_index", k))
                ln = int(v.get("length", 0))
            else:
                eid = int(k)
                ln = int(v) if isinstance(v, (int, float)) else 0
            out.append((eid, ln))
        return sorted(out, key=lambda x: x[0])
    return out


def _pose6_from_sample(s: dict, side: str) -> np.ndarray:
    pos = np.asarray(s[f"{side}_eef_pos"], dtype=np.float32).reshape(3)
    rot = np.asarray(s[f"{side}_eef_rotvec"], dtype=np.float32).reshape(3)
    return np.concatenate([pos, rot], axis=-1)


def _build_trajectory_policy20(
    ds,
    idxs: list[int],
    *,
    time_downsample: int,
    apply_training_norm: bool,
    norm_stats_tree: dict | None,
    use_quantile_norm: bool,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for step_i, gi in enumerate(idxs):
        if time_downsample > 1 and step_i % time_downsample != 0:
            continue
        s = ds[gi]
        l6 = _pose6_from_sample(s, "left")
        r6 = _pose6_from_sample(s, "right")
        la = _to_numpy(s["left_action"])
        ra = _to_numpy(s["right_action"])
        if la.ndim == 2:
            la = la[0]
        if ra.ndim == 2:
            ra = ra[0]
        la = np.asarray(la, dtype=np.float32).reshape(7)
        ra = np.asarray(ra, dtype=np.float32).reshape(7)
        row = xv_dual_policy.abs_next_targets_to_policy20_row(l6, r6, la, ra)
        rows.append(row)
    traj = np.stack(rows, axis=0).astype(np.float32)
    if apply_training_norm and norm_stats_tree is not None:
        n = transforms.Normalize(norm_stats_tree, use_quantiles=use_quantile_norm, strict=False)
        traj = n({"actions": traj})["actions"]
    return traj


def _build_trajectory_geom(
    ds,
    idxs: list[int],
    *,
    mode: str,
    time_downsample: int,
    skip_translation: bool,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for step_i, gi in enumerate(idxs):
        if time_downsample > 1 and step_i % time_downsample != 0:
            continue
        s = ds[gi]
        lp = np.asarray(s["left_eef_pos"], dtype=np.float32).reshape(3)
        lr = np.asarray(s["left_eef_rotvec"], dtype=np.float32).reshape(3)
        lg = np.asarray(s["left_gripper"], dtype=np.float32).reshape(1)
        rp = np.asarray(s["right_eef_pos"], dtype=np.float32).reshape(3)
        rr = np.asarray(s["right_eef_rotvec"], dtype=np.float32).reshape(3)
        rg = np.asarray(s["right_gripper"], dtype=np.float32).reshape(1)
        if mode == "geom14":
            row = np.concatenate([lp, lr, lg, rp, rr, rg], axis=-1)
        elif mode == "geom8":
            row = np.concatenate([lp, lg, rp, rg], axis=-1)
        else:
            raise ValueError(mode)
        rows.append(row.astype(np.float32))
    traj = np.stack(rows, axis=0)
    if not skip_translation and mode == "geom14":
        traj = traj.copy()
        traj[:, 0:3] -= traj[0, 0:3]
        traj[:, 7:10] -= traj[0, 7:10]
    elif not skip_translation and mode == "geom8":
        traj = traj.copy()
        traj[:, 0:3] -= traj[0, 0:3]
        traj[:, 4:7] -= traj[0, 4:7]
    return traj


def _parquet_episode_paths(dataset_root: Path) -> list[tuple[int, Path]]:
    pat = re.compile(r"episode_(\d+)\.parquet$")
    out: list[tuple[int, Path]] = []
    for p in sorted(dataset_root.glob("data/chunk-*/episode_*.parquet")):
        m = pat.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


def _build_trajectory_policy20_parquet(
    df,
    *,
    time_downsample: int,
    apply_training_norm: bool,
    norm_stats_tree: dict | None,
    use_quantile_norm: bool,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for step_i in range(len(df)):
        if time_downsample > 1 and step_i % time_downsample != 0:
            continue
        r = df.iloc[step_i]
        l6 = np.concatenate(
            [
                np.asarray(r["left_eef_pos"], dtype=np.float32).reshape(3),
                np.asarray(r["left_eef_rotvec"], dtype=np.float32).reshape(3),
            ]
        )
        r6 = np.concatenate(
            [
                np.asarray(r["right_eef_pos"], dtype=np.float32).reshape(3),
                np.asarray(r["right_eef_rotvec"], dtype=np.float32).reshape(3),
            ]
        )
        la = np.asarray(r["left_action"], dtype=np.float32).reshape(7)
        ra = np.asarray(r["right_action"], dtype=np.float32).reshape(7)
        rows.append(xv_dual_policy.abs_next_targets_to_policy20_row(l6, r6, la, ra))
    traj = np.stack(rows, axis=0).astype(np.float32)
    if apply_training_norm and norm_stats_tree is not None:
        n = transforms.Normalize(norm_stats_tree, use_quantiles=use_quantile_norm, strict=False)
        traj = n({"actions": traj})["actions"]
    return traj


def _build_trajectory_geom_parquet(
    df,
    *,
    mode: str,
    time_downsample: int,
    skip_translation: bool,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for step_i in range(len(df)):
        if time_downsample > 1 and step_i % time_downsample != 0:
            continue
        r = df.iloc[step_i]
        lp = np.asarray(r["left_eef_pos"], dtype=np.float32).reshape(3)
        lr = np.asarray(r["left_eef_rotvec"], dtype=np.float32).reshape(3)
        lg = np.asarray(r["left_gripper"], dtype=np.float32).reshape(1)
        rp = np.asarray(r["right_eef_pos"], dtype=np.float32).reshape(3)
        rr = np.asarray(r["right_eef_rotvec"], dtype=np.float32).reshape(3)
        rg = np.asarray(r["right_gripper"], dtype=np.float32).reshape(1)
        if mode == "geom14":
            row = np.concatenate([lp, lr, lg, rp, rr, rg], axis=-1)
        elif mode == "geom8":
            row = np.concatenate([lp, lg, rp, rg], axis=-1)
        else:
            raise ValueError(mode)
        rows.append(row.astype(np.float32))
    traj = np.stack(rows, axis=0)
    if not skip_translation and mode == "geom14":
        traj = traj.copy()
        traj[:, 0:3] -= traj[0, 0:3]
        traj[:, 7:10] -= traj[0, 7:10]
    elif not skip_translation and mode == "geom8":
        traj = traj.copy()
        traj[:, 0:3] -= traj[0, 0:3]
        traj[:, 4:7] -= traj[0, 4:7]
    return traj


def _zscore_pool(trajs: list[np.ndarray], grip_idx: list[int]) -> list[np.ndarray]:
    if not trajs:
        return trajs
    d = trajs[0].shape[-1]
    grip_set = set(grip_idx)
    pos_idx = [i for i in range(d) if i not in grip_set]
    if not pos_idx:
        pos_idx = list(range(d))
    parts = []
    for idxs in (pos_idx, list(grip_set)):
        if not idxs:
            continue
        flat = np.concatenate([t[:, idxs].reshape(-1) for t in trajs], axis=0)
        mu = float(np.mean(flat))
        sig = float(np.std(flat)) + 1e-6
        parts.append((idxs, mu, sig))
    out = []
    for t in trajs:
        u = t.copy().astype(np.float32)
        for idxs, mu, sig in parts:
            u[:, idxs] = (u[:, idxs] - mu) / sig
        out.append(u)
    return out


def _pairwise_matrix(
    trajs: list[np.ndarray],
    *,
    metric: str,
    p: float,
    band_ratio: float | None,
    resample: int | None,
) -> np.ndarray:
    e = len(trajs)
    dmat = np.zeros((e, e), dtype=np.float64)
    for i in range(e):
        dmat[i, i] = 0.0
        for j in range(i + 1, e):
            xi, xj = trajs[i], trajs[j]
            if resample is not None and resample > 0:
                xi = _resample_uniform(xi, resample)
                xj = _resample_uniform(xj, resample)
                dist = _mean_lp_distance(xi, xj, p=p)
            else:
                if metric != "dtw":
                    raise ValueError("Without --resample, only --metric dtw is supported.")
                dist = _dtw_multivariate(xi, xj, p=p, band_ratio=band_ratio)
            dmat[i, j] = dist
            dmat[j, i] = dist
    return dmat


def _silhouette_precomputed(D: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient using precomputed pairwise distances D (n,n)."""
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    labels = np.asarray(labels, dtype=np.int64)
    sil = np.zeros(n, dtype=np.float64)
    unique_labels = np.unique(labels)
    for i in range(n):
        li = labels[i]
        same = (labels == li) & (np.arange(n) != i)
        if not np.any(same):
            sil[i] = 0.0
            continue
        a = float(D[i, same].mean())
        other_means: list[float] = []
        for c in unique_labels:
            if int(c) == int(li):
                continue
            mask = labels == c
            if not np.any(mask):
                continue
            other_means.append(float(D[i, mask].mean()))
        if not other_means:
            sil[i] = 0.0
            continue
        b = min(other_means)
        m = max(a, b)
        sil[i] = (b - a) / m if m > 1e-12 else 0.0
    return float(sil.mean())


def _hierarchical_cluster_result(
    dmat: np.ndarray,
    episode_ids: list[int],
    *,
    cluster_mode: str,
    n_clusters: int | None,
    cluster_k_max: int,
) -> dict:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    n = len(episode_ids)
    row_sums = np.sum(dmat, axis=1)
    gmid = int(episode_ids[int(np.argmin(row_sums))])

    def _pack(labels_arr: np.ndarray, meta: dict) -> dict:
        labels_arr = np.asarray(labels_arr, dtype=np.int64)
        medoids: dict[str, int] = {}
        clusters: dict[str, list[int]] = {}
        for c in np.unique(labels_arr):
            members = [episode_ids[k] for k in np.where(labels_arr == c)[0]]
            clusters[str(int(c))] = members
            sub = dmat[np.ix_(labels_arr == c, labels_arr == c)]
            if sub.size == 0:
                continue
            local_idx = int(np.argmin(np.sum(sub, axis=1)))
            global_idx = int(np.where(labels_arr == c)[0][local_idx])
            medoids[str(int(c))] = int(episode_ids[global_idx])
        return {
            "linkage": "average",
            "labels": {int(episode_ids[k]): int(labels_arr[k]) for k in range(n)},
            "medoids": medoids,
            "clusters": clusters,
            "global_medoid_episode_id": gmid,
            **meta,
        }

    if n < 3:
        return {
            "cluster_mode": cluster_mode,
            "note": "n<3: no partition; use distance_matrix.npz for pairwise D",
            "global_medoid_episode_id": gmid,
            "labels": {},
            "medoids": {},
            "clusters": {},
        }

    if cluster_mode == "none":
        return {
            "cluster_mode": "none",
            "note": "No discrete partition; pairwise distances in distance_matrix.npz (D, episodes).",
            "global_medoid_episode_id": gmid,
            "labels": {},
            "medoids": {},
            "clusters": {},
        }

    condensed = squareform(dmat, checks=False)
    z = linkage(condensed, method="average")

    if cluster_mode == "manual":
        if n_clusters is None:
            raise ValueError("--cluster-mode manual requires --n-clusters")
        k = int(n_clusters)
        if k < 2 or k > n:
            raise ValueError(f"--n-clusters must be in [2, {n}], got {k}")
        labels = fcluster(z, t=k, criterion="maxclust")
        return _pack(labels, {"cluster_mode": "manual", "chosen_k": k})

    k_hi = max(2, min(int(cluster_k_max), n - 1))
    best_k = 2
    best_sil = -np.inf
    silhouette_by_k: dict[str, float] = {}
    for k in range(2, k_hi + 1):
        labels_k = fcluster(z, t=k, criterion="maxclust")
        sil = _silhouette_precomputed(dmat, labels_k)
        silhouette_by_k[str(k)] = sil
        if sil > best_sil + 1e-12 or (abs(sil - best_sil) <= 1e-12 and k < best_k):
            best_sil = sil
            best_k = k
    labels = fcluster(z, t=best_k, criterion="maxclust")
    return _pack(
        labels,
        {
            "cluster_mode": "auto",
            "chosen_k": best_k,
            "best_mean_silhouette": best_sil,
            "silhouette_by_k": silhouette_by_k,
            "cluster_k_max": cluster_k_max,
        },
    )


def _labels_for_plot(cl: dict) -> dict[int, int] | None:
    raw = cl.get("labels") or {}
    if not raw:
        return None
    return {int(k): int(v) for k, v in raw.items()}


def _outliers_from_matrix(dmat: np.ndarray, episode_ids: list[int], topk: int) -> list[dict]:
    s = np.sum(dmat, axis=1)
    order = np.argsort(-s)
    out = []
    for rank, k in enumerate(order[:topk]):
        out.append({"episode_id": int(episode_ids[k]), "row_sum_distance": float(s[k]), "rank": rank})
    return out


def _maybe_plots(
    out_dir: Path,
    dmat: np.ndarray,
    episode_ids: list[int],
    labels: dict[int, int] | None,
    *,
    plot_tsne: bool,
    seed: int,
):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform
    except Exception as e:
        print(f"[WARN] Skipping plots: {e}", file=sys.stderr)
        return

    z = linkage(squareform(dmat, checks=False), method="average")
    fig, ax = plt.subplots(figsize=(10, 4))
    dendrogram(z, labels=[str(x) for x in episode_ids], ax=ax, no_labels=len(episode_ids) > 40)
    ax.set_title("Hierarchical clustering (average linkage)")
    fig.tight_layout()
    fig.savefig(out_dir / "dendrogram.png", dpi=120)
    plt.close(fig)

    # Classical MDS (eigendecomposition), no sklearn
    n = dmat.shape[0]
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ (dmat**2) @ j
    w, v = np.linalg.eigh(b)
    order = np.argsort(w)[::-1]
    w = w[order]
    v = v[:, order]
    pos = np.where(w > 1e-8)[0]
    if len(pos) < 2:
        print("[WARN] MDS: not enough positive eigenvalues.", file=sys.stderr)
        return
    lam = np.sqrt(np.maximum(w[pos[:2]], 0.0))
    xy = v[:, pos[:2]] * lam
    fig, ax = plt.subplots(figsize=(6, 6))
    if labels:
        labs = np.array([labels.get(int(e), 0) for e in episode_ids])
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=labs, cmap="tab10", s=36)
        fig.colorbar(sc, ax=ax, label="cluster")
    else:
        ax.scatter(xy[:, 0], xy[:, 1], s=36)
    for i, e in enumerate(episode_ids):
        if n <= 30:
            ax.annotate(str(e), (xy[i, 0], xy[i, 1]), fontsize=7)
    ax.set_title("MDS 2D (classical, distance matrix)")
    fig.tight_layout()
    fig.savefig(out_dir / "mds2d.png", dpi=120)
    plt.close(fig)

    if plot_tsne:
        if n < 5:
            print("[WARN] t-SNE: need at least 5 episodes; skipping.", file=sys.stderr)
        else:
            try:
                from sklearn.manifold import TSNE
            except ImportError as e:
                print(
                    f"[WARN] t-SNE skipped (install scikit-learn for optional tsne2d.png): {e}",
                    file=sys.stderr,
                )
            else:
                perplexity = int(min(30, max(5, n // 4), n - 1))
                try:
                    xy_tsne = TSNE(
                        n_components=2,
                        metric="precomputed",
                        perplexity=perplexity,
                        init="random",
                        random_state=seed,
                    ).fit_transform(np.asarray(dmat, dtype=np.float64))
                except Exception as e:
                    print(f"[WARN] t-SNE failed: {e}", file=sys.stderr)
                else:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    if labels:
                        labs = np.array([labels.get(int(e), 0) for e in episode_ids])
                        sc = ax.scatter(xy_tsne[:, 0], xy_tsne[:, 1], c=labs, cmap="tab10", s=36)
                        fig.colorbar(sc, ax=ax, label="cluster")
                    else:
                        ax.scatter(xy_tsne[:, 0], xy_tsne[:, 1], s=36)
                    for i, e in enumerate(episode_ids):
                        if n <= 30:
                            ax.annotate(str(e), (xy_tsne[i, 0], xy_tsne[i, 1]), fontsize=7)
                    ax.set_title(f"t-SNE 2D (precomputed D, perplexity={perplexity})")
                    fig.tight_layout()
                    fig.savefig(out_dir / "tsne2d.png", dpi=120)
                    plt.close(fig)


def _emit_outputs(
    args: argparse.Namespace,
    out_dir: Path,
    trajs_concat: list[np.ndarray],
    trajs_left: list[np.ndarray],
    trajs_right: list[np.ndarray],
    used_ids: list[int],
    *,
    run_cfg_extra: dict | None = None,
) -> None:
    def finish(trajs: list[np.ndarray], suffix: str):
        if not args.skip_zscore:
            if args.representation == "geom14":
                grip_idx = [6, 13]
                trajs = _zscore_pool(trajs, grip_idx)
            elif args.representation == "geom8":
                trajs = _zscore_pool(trajs, [3, 7])
            else:
                grip_idx = [9, 19]
                trajs = _zscore_pool(trajs, grip_idx)
        dmat = _pairwise_matrix(
            trajs,
            metric=args.metric,
            p=2.0,
            band_ratio=args.dtw_band_ratio,
            resample=args.resample,
        )
        if np.any(~np.isfinite(dmat)):
            raise RuntimeError("Non-finite values in distance matrix.")
        np.savez(
            out_dir / f"distance_matrix{suffix}.npz",
            episodes=np.array(used_ids, dtype=np.int64),
            D=dmat.astype(np.float32),
        )
        return dmat

    def cluster_result(dmat: np.ndarray, eids: list[int]) -> dict:
        return _hierarchical_cluster_result(
            dmat,
            eids,
            cluster_mode=args.cluster_mode,
            n_clusters=args.n_clusters,
            cluster_k_max=args.cluster_k_max,
        )

    if args.arm_mode == "separate" and args.representation != "policy20":
        d_left = finish(trajs_left, "_left")
        d_right = finish(trajs_right, "_right")
        cl = cluster_result(d_left, used_ids)
        cr = cluster_result(d_right, used_ids)
        (out_dir / "clusters.json").write_text(
            json.dumps({"left": cl, "right": cr}, indent=2), encoding="utf-8"
        )
        ol = _outliers_from_matrix(d_left, used_ids, args.outlier_topk)
        or_ = _outliers_from_matrix(d_right, used_ids, args.outlier_topk)
        (out_dir / "outliers.json").write_text(json.dumps({"left": ol, "right": or_}, indent=2), encoding="utf-8")
        if not args.no_plots:
            (out_dir / "left").mkdir(parents=True, exist_ok=True)
            (out_dir / "right").mkdir(parents=True, exist_ok=True)
            _maybe_plots(
                out_dir / "left",
                d_left,
                used_ids,
                _labels_for_plot(cl),
                plot_tsne=args.plot_tsne,
                seed=args.seed,
            )
            _maybe_plots(
                out_dir / "right",
                d_right,
                used_ids,
                _labels_for_plot(cr),
                plot_tsne=args.plot_tsne,
                seed=args.seed,
            )
    else:
        dmat = finish(trajs_concat, "")
        cl = cluster_result(dmat, used_ids)
        (out_dir / "clusters.json").write_text(json.dumps(cl, indent=2), encoding="utf-8")
        ol = _outliers_from_matrix(dmat, used_ids, args.outlier_topk)
        (out_dir / "outliers.json").write_text(json.dumps(ol, indent=2), encoding="utf-8")
        labels_map = _labels_for_plot(cl)
        if not args.no_plots:
            _maybe_plots(
                out_dir,
                dmat,
                used_ids,
                labels_map,
                plot_tsne=args.plot_tsne,
                seed=args.seed,
            )

    run_cfg = {
        "config_name": args.config_name,
        "representation": args.representation,
        "arm_mode": args.arm_mode,
        "resample": args.resample,
        "dtw_band_ratio": args.dtw_band_ratio,
        "E": len(used_ids),
        "seed": args.seed,
        "episode_ids": used_ids,
        "cluster_mode": args.cluster_mode,
        "cluster_k_max": args.cluster_k_max,
        "plot_tsne": args.plot_tsne,
        "note_geom_vs_policy": (
            "geom14/geom8 are end-effector geometry; policy20 matches XVDualInputs supervision (see xv_dual_policy.abs_next_targets_to_policy20_row)."
        ),
    }
    if args.n_clusters is not None:
        run_cfg["n_clusters"] = args.n_clusters
    if run_cfg_extra:
        run_cfg.update(run_cfg_extra)
    (out_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")
    print(f"[OK] Wrote outputs under {out_dir.resolve()}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Episode-level trajectory similarity (XV dual-arm).")
    ap.add_argument("--config-name", type=str, default="pi05_xv_dual_finetune")
    ap.add_argument("--representation", choices=("policy20", "geom14", "geom8"), default="policy20")
    ap.add_argument("--arm-mode", choices=("concat", "separate"), default="concat")
    ap.add_argument("--time-downsample", type=int, default=1)
    ap.add_argument("--max-episodes", type=int, default=0, help="0 = all episodes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--metric", choices=("dtw",), default="dtw")
    ap.add_argument("--dtw-band-ratio", type=float, default=None)
    ap.add_argument("--resample", type=int, default=None, help="Fixed length L (>0) or omit for none")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--dataset-root", type=str, default=None, help="Override LeRobot root (local folder with meta/).")
    ap.add_argument(
        "--parquet-lowdim-only",
        action="store_true",
        help="Read trajectories from data/chunk-*/episode_*.parquet only (no video decode). Requires --dataset-root.",
    )
    ap.add_argument("--skip-translation", action="store_true", help="Disable per-episode translation normalize (geom).")
    ap.add_argument("--skip-zscore", action="store_true")
    ap.add_argument("--skip-training-norm", action="store_true", help="Do not apply norm_stats.json to policy20.")
    ap.add_argument("--outlier-topk", type=int, default=5)
    ap.add_argument(
        "--cluster-mode",
        choices=("auto", "manual", "none"),
        default="none",
        help="none (default): only D + outliers + plots from D; no cluster labels. auto/manual: partition via hierarchy.",
    )
    ap.add_argument("--n-clusters", type=int, default=None, help="With --cluster-mode manual (required there).")
    ap.add_argument("--cluster-k-max", type=int, default=15, help="Upper bound on k when scanning in auto mode.")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument(
        "--no-plot-tsne",
        action="store_false",
        dest="plot_tsne",
        help="Skip tsne2d.png (default: run t-SNE on D when plots enabled; needs scikit-learn, slower for large E).",
    )
    args = ap.parse_args()

    if args.cluster_mode == "manual" and args.n_clusters is None:
        raise SystemExit("--cluster-mode manual requires --n-clusters")

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = _config.get_config(args.config_name)
    data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)

    if args.parquet_lowdim_only:
        import pandas as pd
        from openpi.shared import normalize as openpi_normalize

        if not args.dataset_root:
            raise SystemExit("--parquet-lowdim-only requires --dataset-root")
        root_ds = Path(args.dataset_root)
        paths = _parquet_episode_paths(root_ds)
        if not paths:
            raise RuntimeError(f"No episode parquet under {root_ds}/data/chunk-*/episode_*.parquet")
        episode_ids_sel = [e for e, _ in paths]
        if args.max_episodes > 0 and len(episode_ids_sel) > args.max_episodes:
            pick = rng.choice(len(episode_ids_sel), size=args.max_episodes, replace=False)
            pick_set = {episode_ids_sel[i] for i in pick}
            paths = [(e, p) for e, p in paths if e in pick_set]

        ds_norm = None
        if not args.skip_training_norm:
            try:
                ds_norm = openpi_normalize.load(root_ds)
            except FileNotFoundError:
                ds_norm = None
        norm_stats = None if args.skip_training_norm else (ds_norm if ds_norm is not None else data_config.norm_stats)
        if args.representation == "policy20" and norm_stats is None and not args.skip_training_norm:
            print(
                "[WARN] No norm_stats (dataset or config); using raw policy20. Use --skip-training-norm to silence.",
                file=sys.stderr,
            )

        trajs_concat, trajs_left, trajs_right, used_ids = [], [], [], []
        for eid, pth in paths:
            df = pd.read_parquet(pth)
            if len(df) < 2:
                continue
            try:
                if args.representation == "policy20":
                    if args.arm_mode != "concat":
                        print("[WARN] policy20 forces concat; ignoring --arm-mode separate.", file=sys.stderr)
                    traj = _build_trajectory_policy20_parquet(
                        df,
                        time_downsample=args.time_downsample,
                        apply_training_norm=bool(norm_stats),
                        norm_stats_tree=norm_stats,
                        use_quantile_norm=data_config.use_quantile_norm,
                    )
                    trajs_concat.append(traj)
                else:
                    traj = _build_trajectory_geom_parquet(
                        df,
                        mode=args.representation,
                        time_downsample=args.time_downsample,
                        skip_translation=args.skip_translation,
                    )
                    if args.arm_mode == "concat":
                        trajs_concat.append(traj)
                    else:
                        if args.representation == "geom8":
                            trajs_left.append(traj[:, :4])
                            trajs_right.append(traj[:, 4:8])
                        else:
                            trajs_left.append(traj[:, :7])
                            trajs_right.append(traj[:, 7:14])
                used_ids.append(int(eid))
            except Exception as ex:
                print(f"[WARN] Skip episode {eid}: {ex}", file=sys.stderr)

        if not used_ids:
            raise RuntimeError("No episodes loaded from parquet.")
        _emit_outputs(
            args,
            out_dir,
            trajs_concat,
            trajs_left,
            trajs_right,
            used_ids,
            run_cfg_extra={
                "parquet_lowdim_only": True,
                "dataset_root": str(root_ds.resolve()),
            },
        )
        return

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("config data has no repo_id")
    root = Path(args.dataset_root) if args.dataset_root else None
    meta_fps = 10.0
    try:
        md = LeRobotDatasetMetadata(repo_id, root=str(root)) if root is not None else LeRobotDatasetMetadata(repo_id)
        meta_fps = float(md.fps)
    except Exception:
        pass
    delta_ts = {
        k: [t / meta_fps for t in range(train_cfg.model.action_horizon)]
        for k in (data_config.action_sequence_keys or ("left_action", "right_action"))
    }
    if root is not None:
        ds = LeRobotDataset(repo_id=repo_id, root=str(root), delta_timestamps=delta_ts)
    else:
        ds = LeRobotDataset(repo_id=repo_id, delta_timestamps=delta_ts)

    ep_list = _list_episodes(ds)
    if not ep_list:
        s0 = ds[0]
        k = next((kk for kk in ("episode_index", "episode_id", "episode") if kk in s0), "episode_index")
        uniq = sorted({int(_to_numpy(ds[i][k])) for i in range(len(ds))})
        ep_list = [(int(u), 0) for u in uniq]

    episode_ids = [eid for eid, _ln in ep_list]
    if args.max_episodes > 0 and len(episode_ids) > args.max_episodes:
        pick = rng.choice(len(episode_ids), size=args.max_episodes, replace=False)
        episode_ids = sorted(int(episode_ids[i]) for i in pick)
        ep_list = [(e, ln) for e, ln in ep_list if e in set(episode_ids)]

    norm_stats = None if args.skip_training_norm else data_config.norm_stats
    if args.representation == "policy20" and norm_stats is None and not args.skip_training_norm:
        print(
            "[WARN] No norm_stats on DataConfig; run compute_norm_stats or use --skip-training-norm. "
            "Using raw policy20 (first 20 dims before Pad).",
            file=sys.stderr,
        )

    trajs_concat = []
    trajs_left = []
    trajs_right = []
    used_ids = []

    for eid, ln in ep_list:
        if eid not in set(episode_ids):
            continue
        try:
            idxs = _episode_indices(ds, eid, ln)
        except Exception as ex:
            print(f"[WARN] Skip episode {eid}: {ex}", file=sys.stderr)
            continue
        if len(idxs) < 2:
            continue
        if args.representation == "policy20":
            if args.arm_mode != "concat":
                print("[WARN] policy20 forces concat; ignoring --arm-mode separate.", file=sys.stderr)
            traj = _build_trajectory_policy20(
                ds,
                idxs,
                time_downsample=args.time_downsample,
                apply_training_norm=bool(norm_stats) and not args.skip_training_norm,
                norm_stats_tree=norm_stats,
                use_quantile_norm=data_config.use_quantile_norm,
            )
            trajs_concat.append(traj)
        else:
            traj = _build_trajectory_geom(
                ds,
                idxs,
                mode=args.representation,
                time_downsample=args.time_downsample,
                skip_translation=args.skip_translation,
            )
            if args.arm_mode == "concat":
                trajs_concat.append(traj)
            else:
                if args.representation == "geom8":
                    trajs_left.append(traj[:, :4])
                    trajs_right.append(traj[:, 4:8])
                else:
                    trajs_left.append(traj[:, :7])
                    trajs_right.append(traj[:, 7:14])
        used_ids.append(int(eid))

    if not used_ids:
        raise RuntimeError("No episodes loaded; check dataset path and meta.")

    _emit_outputs(args, out_dir, trajs_concat, trajs_left, trajs_right, used_ids)


if __name__ == "__main__":
    main()
