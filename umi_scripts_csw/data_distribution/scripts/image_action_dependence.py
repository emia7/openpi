"""
冻结 SigLIP 图像塔 + 与训练一致的 (图像, actions) 配对，估计 Z–A 依赖性（HSIC）。

用法（在仓库根目录，且已安装 openpi / 设置 PYTHONPATH=src）:
  python -m umi_scripts_csw.data_distribution.scripts.image_action_dependence \\
    --config-name pi05_xv_dual_finetune --max-samples 10000 --seed 0 \\
    --view-mode concat --encoder openpi_siglip --metric hsic \\
    --out-dir ./reports/image_action_mi_run001
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pathlib
import sys
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

logger = logging.getLogger(__name__)

IMAGE_KEYS = _model.IMAGE_KEYS


def _collate(items: list[dict]) -> dict:
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _rbf_gram(X: np.ndarray, sigma: float) -> np.ndarray:
    """X: (n, d), isotropic RBF Gram K_ij = exp(-||xi-xj||^2 / (2 sigma^2))."""
    sq = np.sum(X**2, axis=1, keepdims=True)
    dist2 = sq + sq.T - 2.0 * (X @ X.T)
    dist2 = np.maximum(dist2, 0.0)
    return np.exp(-dist2 / (2.0 * sigma**2 + 1e-12))


def _median_bandwidth(X: np.ndarray, rng: np.random.Generator, max_rows: int = 800) -> float:
    """Median heuristic on a row subsample (squared Euclidean distances)."""
    n = X.shape[0]
    if n > max_rows:
        idx = rng.choice(n, size=max_rows, replace=False)
        Xs = X[idx]
    else:
        Xs = X
    m = Xs.shape[0]
    if m < 2:
        return 1.0
    sq = np.sum(Xs.astype(np.float64) ** 2, axis=1, keepdims=True)
    dist2 = sq + sq.T - 2.0 * (Xs.astype(np.float64) @ Xs.astype(np.float64).T)
    dist2 = np.maximum(dist2, 0.0)
    iu = np.triu_indices(m, k=1)
    dflat = dist2[iu]
    dflat = dflat[dflat > 0]
    if dflat.size == 0:
        return 1.0
    med = float(np.median(dflat))
    return float(np.sqrt(med / 2.0) + 1e-8)


def _hsic_biased(K: np.ndarray, L: np.ndarray) -> float:
    """HSIC with centered Gram matrices, common finite-sample form."""
    n = K.shape[0]
    if n < 4:
        return float("nan")
    H = np.eye(n, dtype=np.float64) - 1.0 / n
    Kc = H @ K @ H
    Lc = H @ L @ H
    return float(np.sum(Kc * Lc) / ((n - 1) ** 2))


def _pca_projection(A: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """A: (n, d) -> (n, k)."""
    A = A.astype(np.float64)
    A = A - A.mean(axis=0, keepdims=True)
    # SVD on (possibly wide) matrix: n×d, keep k components
    u, s, vt = np.linalg.svd(A, full_matrices=False)
    k = min(k, vt.shape[0])
    return (A @ vt[:k].T).astype(np.float64)


def _siglip_module(model: nnx.Module):
    pg = model.PaliGemma
    return pg.img if hasattr(pg, "img") else pg["img"]


def _embed_one_view(model: nnx.Module, img_bhwc: jnp.ndarray) -> jnp.ndarray:
    """Mean-pool SigLIP patch tokens -> (B, D)."""
    tokens, _ = _siglip_module(model)(img_bhwc, train=False)
    return jnp.mean(tokens, axis=1)


def _encode_z(
    model: nnx.Module,
    obs: _model.Observation,
    view_mode: Literal["concat", "third_only", "mean"],
) -> np.ndarray:
    """obs images must be JAX arrays BHWC in [-1,1], 224."""
    imgs = {k: obs.images[k] for k in IMAGE_KEYS}
    if view_mode == "third_only":
        z = _embed_one_view(model, imgs["base_0_rgb"])
        return np.array(z, dtype=np.float64)

    z_left = _embed_one_view(model, imgs["left_wrist_0_rgb"])
    z_right = _embed_one_view(model, imgs["right_wrist_0_rgb"])
    z_base = _embed_one_view(model, imgs["base_0_rgb"])
    if view_mode == "mean":
        z = (z_left + z_right + z_base) / 3.0
    else:
        z = jnp.concatenate([z_left, z_right, z_base], axis=-1)
    return np.array(z, dtype=np.float64)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Image–action dependence (HSIC) with frozen Pi SigLIP.")
    p.add_argument("--config-name", type=str, default="pi05_xv_dual_finetune")
    p.add_argument("--max-samples", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--view-mode", choices=("concat", "third_only", "mean"), default="concat")
    p.add_argument("--encoder", choices=("openpi_siglip",), default="openpi_siglip")
    p.add_argument("--metric", choices=("hsic",), default="hsic")
    p.add_argument("--out-dir", type=pathlib.Path, required=True)
    p.add_argument("--repo-id", type=str, default=None, help="Override TrainConfig.data.repo_id for local runs.")
    p.add_argument(
        "--asset-id",
        type=str,
        default=None,
        help="Override norm_stats asset_id (under data factory assets_dir). "
        "If --repo-id is set and this is omitted, defaults to the last path segment of --repo-id.",
    )
    p.add_argument("--assets-base-dir", type=str, default=None, help="Override TrainConfig.assets_base_dir.")
    p.add_argument("--inference-batch-size", type=int, default=8)
    p.add_argument("--pca-action-dim", type=int, default=0, help="If >0, also report HSIC(Z, PCA(A)) with this k.")
    p.add_argument("--every-k-frames", type=int, default=1, help="Take every k-th frame index before subsampling.")
    p.add_argument("--save-embeddings", action="store_true")
    args = p.parse_args(argv)

    if args.encoder != "openpi_siglip":
        raise SystemExit("Only --encoder openpi_siglip is implemented.")
    if args.metric != "hsic":
        raise SystemExit("Only --metric hsic is implemented.")

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
        "view_mode": args.view_mode,
        "encoder": args.encoder,
        "metric": args.metric,
        "every_k_frames": stride,
        "pca_action_dim": args.pca_action_dim,
        "action_horizon": model_cfg.action_horizon,
        "action_dim_effective": 20,
        "weight_loader": str(train_cfg.weight_loader),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    logger.info("Loading JAX Pi0 + checkpoint weights (SigLIP tower)...")
    init = train_cfg.model.create(jax.random.key(0))
    graphdef, state = nnx.split(init)
    merged = train_cfg.weight_loader.load(state.to_pure_dict())
    model = train_cfg.model.load(merged)

    bs = max(1, min(args.inference_batch_size, n_take))
    Z_list: list[np.ndarray] = []
    A_list: list[np.ndarray] = []

    for start in range(0, n_take, bs):
        chunk_idx = indices[start : start + bs]
        batch_dict = _collate([ds[int(i)] for i in chunk_idx])
        obs = _model.Observation.from_dict(batch_dict)
        obs = _model.preprocess_observation(None, obs, train=False)
        z_b = _encode_z(model, obs, args.view_mode)
        Z_list.append(z_b)
        act = np.asarray(batch_dict["actions"], dtype=np.float64)
        # (B, H, 32) -> effective (B, H*20)
        a_b = act[..., :20].reshape(act.shape[0], -1)
        A_list.append(a_b)

    Z = np.concatenate(Z_list, axis=0)
    A = np.concatenate(A_list, axis=0)
    assert Z.shape[0] == A.shape[0] == n_take

    rng_hsic = np.random.default_rng(args.seed + 17)
    sigma_z = _median_bandwidth(Z, rng_hsic)
    sigma_a = _median_bandwidth(A, rng_hsic)
    Kz = _rbf_gram(Z, sigma_z)
    Ka = _rbf_gram(A, sigma_a)
    hsic_val = _hsic_biased(Kz, Ka)

    metrics: dict = {
        "n_samples": int(n_take),
        "hsic": hsic_val,
        "sigma_z_median_heuristic": sigma_z,
        "sigma_a_median_heuristic": sigma_a,
        "z_dim": int(Z.shape[1]),
        "a_dim": int(A.shape[1]),
        "hsic_formula": "trace(H K H L H) / (n-1)^2 with RBF Grams, median bandwidth per variable",
    }

    if args.pca_action_dim > 0:
        Ap = _pca_projection(A, args.pca_action_dim, rng_hsic)
        sigma_ap = _median_bandwidth(Ap, rng_hsic)
        Kap = _rbf_gram(Ap, sigma_ap)
        metrics["hsic_z_apca"] = _hsic_biased(Kz, Kap)
        metrics["pca_action_dim"] = int(min(args.pca_action_dim, A.shape[1]))

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_dir / "metrics.json")

    if args.save_embeddings:
        np.savez_compressed(out_dir / "z_cache.npz", Z=Z.astype(np.float32), A=A.astype(np.float32))
        logger.info("Wrote %s", out_dir / "z_cache.npz")


if __name__ == "__main__":
    main()
