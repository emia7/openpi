#!/usr/bin/env python3
"""
两种表征模式（``--representation``）：

- **vlm**（默认）：PI 附录 C 风格 — **PaliGemma ``language_model``** 对前缀序列最后一层隐状态做 mean-pool，
  默认只对前 ``pool_first_k`` 个 prefix 位置池化（论文约 200）。
- **vision**：**SigLIP ``vision_tower`` → ``multi_modal_projector``**（与 ``embed_image`` 一致），
  每相机 patch 加权 mean，再三路相机算术平均。

对比两类微调 checkpoint：
  - **仅机器人遥操**（通常为 ``pi05_dual_franka_finetune_*``）
  - **仅 UMI / XV 双臂数据**（通常为 ``pi05_xv_dual_finetune``）

对 **机器人 LeRobot** 与 **UMI LeRobot** 各采样一批帧，分别用两套 policy 的 ``_input_transform`` 做预处理，
同一物理含义的样本在两种键格式之间用本脚本内的 **XV ↔ dual_franka 扁平键** 互转（仅图像槽位与 state 对齐）。

输出：若干组向量及联合 **2D 降维图**（默认 ``tsne`` + ``pca`` + ``umap``，可用 ``--plots`` 选择）；
并在 ``report.json`` 写入 **embedding_separation**（组间质心 L2、组内到质心距离）及 PCA 方差解释比例。

PI 论文图示往往对比 **跨物种/大规模轨迹**，样本量与域间差异更大；本仓库可与论文并列呈现多种降维作定性对照。

仓库根目录执行::

    PYTHONPATH=src:. python umi_scripts_csw/representation_analysis/teleop_vs_umi_vlm_repr_pi_style.py \\
      --teleop-config pi05_dual_franka_finetune_high_lumos \\
      --umi-config pi05_xv_dual_finetune \\
      --teleop-checkpoint /path/teleop_only_ft \\
      --umi-checkpoint /path/umi_only_ft \\
      --robot-repo /path/dual_franka_lerobot \\
      --umi-repo /path/umi_lerobot \\
      --out-dir ./analysis_results/pi_style_teleop_vs_umi
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from openpi.models import model as openpi_model
from openpi.models_pytorch import pi0_pytorch
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.policies import policy_config
from openpi.training import checkpoints as ckpt_lib
from openpi.training import config as train_config_lib
from openpi.training.config import (
    _DUAL_FRANKA_IMG_LEFT_LUMOS,
    _DUAL_FRANKA_IMG_LEFT_RS,
    _DUAL_FRANKA_IMG_RIGHT_LUMOS,
    _DUAL_FRANKA_IMG_RIGHT_RS,
    _DUAL_FRANKA_IMG_THIRD_DEFAULT,
)

from umi_scripts_csw.representation_analysis.human_robot_representation_transfer import (
    _openpi_dataset_sample_to_raw_row,
    load_norm_stats_for_config,
    load_policy_for_transforms,
    processed_dict_to_observation,
    resolve_checkpoint_model,
)
from umi_scripts_csw.representation_analysis.legend_config import legend_labels_for_groups
from umi_scripts_csw.representation_analysis.robot_dataset_adapters import get_robot_adapter

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    try:
        import jax.numpy as jnp

        if isinstance(x, jnp.ndarray):
            return np.asarray(x)
    except Exception:
        pass
    try:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def teleop_data_config_image_keys(teleop_cfg) -> tuple[str, str, str]:
    """由 TrainConfig.data（LeRobotDualFrankaDataConfig）得到三路图像在样本里的扁平键。"""
    d = teleop_cfg.data
    if not hasattr(d, "variant"):
        raise ValueError(
            "--teleop-config 须对应双臂 Franka LeRobot（含 variant），例如 pi05_dual_franka_finetune_high_lumos"
        )
    var = getattr(d, "variant", "")
    if str(var).endswith("_lumos"):
        lk, rk = _DUAL_FRANKA_IMG_LEFT_LUMOS, _DUAL_FRANKA_IMG_RIGHT_LUMOS
    elif str(var).endswith("_rs"):
        lk, rk = _DUAL_FRANKA_IMG_LEFT_RS, _DUAL_FRANKA_IMG_RIGHT_RS
    else:
        lk, rk = _DUAL_FRANKA_IMG_LEFT_LUMOS, _DUAL_FRANKA_IMG_RIGHT_LUMOS
    tk = getattr(d, "third_image_key", None) or _DUAL_FRANKA_IMG_THIRD_DEFAULT
    return lk, rk, tk


def xv_row_to_dual_franka_flat(row: dict, *, lk: str, rk: str, tk: str) -> dict:
    """policy 中间格式（left_view…）→ dual_franka Repack 期望的扁平圆点键。"""
    lp = np.asarray(row["left_eef_pos"], np.float32).reshape(3)
    lr = np.asarray(row["left_eef_rotvec"], np.float32).reshape(3)
    rp = np.asarray(row["right_eef_pos"], np.float32).reshape(3)
    rr = np.asarray(row["right_eef_rotvec"], np.float32).reshape(3)
    la = row.get("left_action")
    ra = row.get("right_action")
    if la is None or ra is None:
        la = np.zeros(7, np.float32)
        ra = np.zeros(7, np.float32)
    else:
        la = np.asarray(la, np.float32).reshape(7)
        ra = np.asarray(ra, np.float32).reshape(7)
    out = {
        lk: _to_numpy(row["left_view"]),
        rk: _to_numpy(row["right_view"]),
        tk: _to_numpy(row["third_view"]),
        "observation.state.left_eef_pos": lp,
        "observation.state.left_eef_rotvec": lr,
        "observation.state.left_gripper": np.asarray(row["left_gripper"], np.float32).reshape(1),
        "observation.state.right_eef_pos": rp,
        "observation.state.right_eef_rotvec": rr,
        "observation.state.right_gripper": np.asarray(row["right_gripper"], np.float32).reshape(1),
        "observation.state.demo_start_pose_left": np.asarray(row["demo_start_pose_left"], np.float32).reshape(6),
        "observation.state.demo_start_pose_right": np.asarray(row["demo_start_pose_right"], np.float32).reshape(6),
        "left_action": la,
        "right_action": ra,
        "task": str(row.get("task", "") or ""),
    }
    return out


def dual_franka_flat_to_xv_row(flat: dict, *, lk: str, rk: str, tk: str) -> dict:
    """dual_franka 扁平键 → XV Repack 期望的短键。"""
    return {
        "left_view": _to_numpy(flat[lk]),
        "right_view": _to_numpy(flat[rk]),
        "third_view": _to_numpy(flat[tk]),
        "left_eef_pos": np.asarray(flat["observation.state.left_eef_pos"], np.float32).reshape(3),
        "left_eef_rotvec": np.asarray(flat["observation.state.left_eef_rotvec"], np.float32).reshape(3),
        "left_gripper": np.asarray(flat["observation.state.left_gripper"], np.float32).reshape(1),
        "right_eef_pos": np.asarray(flat["observation.state.right_eef_pos"], np.float32).reshape(3),
        "right_eef_rotvec": np.asarray(flat["observation.state.right_eef_rotvec"], np.float32).reshape(3),
        "right_gripper": np.asarray(flat["observation.state.right_gripper"], np.float32).reshape(1),
        "demo_start_pose_left": np.asarray(flat["observation.state.demo_start_pose_left"], np.float32).reshape(6),
        "demo_start_pose_right": np.asarray(flat["observation.state.demo_start_pose_right"], np.float32).reshape(6),
        "left_action": np.asarray(flat["left_action"], np.float32).reshape(7),
        "right_action": np.asarray(flat["right_action"], np.float32).reshape(7),
        "task": str(flat.get("task", "") or ""),
    }


def _is_dual_franka_flat(sample: dict) -> bool:
    return _DUAL_FRANKA_IMG_LEFT_LUMOS in sample or _DUAL_FRANKA_IMG_LEFT_RS in sample


def robot_sample_to_dual_flat(sample: dict, robot_adapter, lk: str, rk: str, tk: str) -> dict:
    if robot_adapter is not None:
        return robot_adapter(sample)
    if _is_dual_franka_flat(sample):
        return dict(sample)
    row = _openpi_dataset_sample_to_raw_row(sample)
    return xv_row_to_dual_franka_flat(row, lk=lk, rk=rk, tk=tk)


def robot_sample_to_xv_row(sample: dict, robot_adapter, lk: str, rk: str, tk: str) -> dict:
    """机器人数据集帧 → XV 风格 raw（供 UMI policy transform）。"""
    flat = robot_sample_to_dual_flat(sample, robot_adapter, lk, rk, tk)
    return dual_franka_flat_to_xv_row(flat, lk=lk, rk=rk, tk=tk)


def umi_sample_to_xv_row(sample: dict) -> dict:
    return _openpi_dataset_sample_to_raw_row(sample)


def umi_sample_to_dual_flat(sample: dict, lk: str, rk: str, tk: str) -> dict:
    row = umi_sample_to_xv_row(sample)
    return xv_row_to_dual_franka_flat(row, lk=lk, rk=rk, tk=tk)


@torch.no_grad()
def extract_vlm_last_layer_mean_first_k(
    model: pi0_pytorch.PI0Pytorch,
    observation: openpi_model.Observation,
    *,
    pool_first_k: int,
) -> np.ndarray:
    """
    PI 论文附录：对 VLM（``paligemma.language_model``）最后一层 ``last_hidden_state``，
    取前缀序列 **前 min(K, T)** 个 token，在有效 mask 内 mean-pool → (D,) float32。
    """
    device = next(model.parameters()).device
    observation = jax.tree.map(lambda x: x.to(device) if torch.is_tensor(x) else x, observation)

    images, img_masks, lang_tokens, lang_masks, _state = model._preprocess_observation(observation, train=False)
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

    out = model.paligemma_with_expert.paligemma.language_model.forward(
        inputs_embeds=prefix_embs,
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        use_cache=False,
        adarms_cond=None,
    )
    h = out.last_hidden_state.float()
    tlen = h.shape[1]
    k = min(pool_first_k, tlen)
    h_k = h[:, :k, :]
    m = prefix_pad_masks[:, :k].unsqueeze(-1).float()
    pooled = (h_k * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)
    return pooled[0].detach().cpu().numpy()


@torch.no_grad()
def extract_vision_encoder_mean_pooled(
    model: pi0_pytorch.PI0Pytorch,
    observation: openpi_model.Observation,
) -> np.ndarray:
    """
    **图像编码器路径**（非 ``language_model``）：与 ``PI0Pytorch.embed_prefix`` 相同，
    对每个相机调用 ``paligemma_with_expert.embed_image`` → SigLIP ``vision_tower`` +
    ``multi_modal_projector``，得到每帧 ``(B, L_patch, D)``；
    在 patch 维按 ``img_mask`` 做加权 mean，得到每视图一条 ``(B, D)``，
    再对左/右/第三视角 **算术平均** → 单条 ``(D,)``（与三相机贡献均等）。
    """
    device = next(model.parameters()).device
    observation = jax.tree.map(lambda x: x.to(device) if torch.is_tensor(x) else x, observation)

    images, img_masks, _, _, _ = model._preprocess_observation(observation, train=False)

    pooled_views: list[torch.Tensor] = []
    for img, img_mask in zip(images, img_masks, strict=True):

        def image_embed_func(img_t):
            return model.paligemma_with_expert.embed_image(img_t)

        img_emb = model._apply_checkpoint(image_embed_func, img)
        bsize, num_img_embs, _dim = img_emb.shape
        m = img_mask[:, None].expand(bsize, num_img_embs).float().unsqueeze(-1)
        pooled = (img_emb * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)
        pooled_views.append(pooled)

    out = torch.stack(pooled_views, dim=0).mean(dim=0)
    return out[0].detach().cpu().numpy()


def extract_repr_vector(
    model: pi0_pytorch.PI0Pytorch,
    observation: openpi_model.Observation,
    *,
    representation: str,
    pool_first_k: int,
) -> np.ndarray:
    if representation == "vision":
        return extract_vision_encoder_mean_pooled(model, observation)
    if representation == "vlm":
        return extract_vlm_last_layer_mean_first_k(model, observation, pool_first_k=pool_first_k)
    raise ValueError(f"未知 representation={representation!r}，可选: vlm, vision")


def load_ns_safe(cfg, ckpt_dir: Path, asset_id_override: str | None, assets_from: Path | None):
    root = Path(assets_from).resolve() if assets_from else ckpt_dir
    return load_norm_stats_for_config(cfg, root, asset_id_override=asset_id_override)


def _collect_indices(ds, max_samples: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    n = len(ds)
    if n <= max_samples:
        return list(range(n))
    return sorted(rng.choice(n, size=max_samples, replace=False).tolist())


def _sample_cap_for_dataset(ds_len: int, *, max_samples: int, use_all: bool) -> int:
    """``use_all`` 时用满该集 ``ds_len``；否则上限为 ``max_samples``。"""
    return ds_len if use_all else max_samples


def discover_locally_complete_episodes(root: Path, repo_id: str = "local_eval") -> list[int]:
    """
    仅根据磁盘上已有文件判断 episode 是否完整（parquet + 该集所需全部视频），
    避免 LeRobotDataset 在缺文件时走 HF 下载分支（离线环境会失败）。
    parquet 须能通过 pyarrow 读 metadata（过滤拷一半的损坏文件）。
    """
    import pyarrow.parquet as pq

    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id, str(root.resolve()))
    ok: list[int] = []
    for ep_idx in range(meta.total_episodes):
        dp = meta.root / meta.get_data_file_path(ep_idx)
        if not dp.is_file() or dp.stat().st_size < 8:
            continue
        try:
            pq.read_metadata(str(dp))
        except Exception:
            continue
        vid_ok = True
        for vk in meta.video_keys:
            vp = meta.root / meta.get_video_file_path(ep_idx, vk)
            if not vp.is_file() or vp.stat().st_size < 32:
                vid_ok = False
                break
        if vid_ok:
            ok.append(ep_idx)
    return ok


def _embedding_separation_stats(names: list[str], blocks: list[np.ndarray]) -> dict:
    """原始表征空间（未 t-SNE）内：组间质心距离 vs 组内到质心距离，用于解读远近。"""
    import itertools

    centroids = {n: b.mean(axis=0) for n, b in zip(names, blocks)}
    between = {}
    for a, b in itertools.combinations(names, 2):
        between[f"{a}_vs_{b}"] = float(np.linalg.norm(centroids[a] - centroids[b]))
    within = {}
    for n, b in zip(names, blocks):
        c = centroids[n]
        d = np.linalg.norm(b - c.astype(np.float64), axis=1)
        within[f"{n}_mean_dist_to_centroid"] = float(d.mean())
        within[f"{n}_std_dist_to_centroid"] = float(d.std())
    return {
        "embedding_dim": int(blocks[0].shape[1]),
        "centroid_l2_between_groups": between,
        "within_group_dist_to_centroid": within,
    }


def _plot_group_scatter_2d(
    Z2: np.ndarray,
    group_names: list[str],
    lens: list[int],
    *,
    title: str,
    out_path: Path,
    legend_labels: list[str] | None = None,
    dpi: int = 200,
) -> None:
    """与 PI 类论文一致的清晰散点：略大点 + 白描边，便于屏幕与印刷。"""
    labels = legend_labels if legend_labels is not None else group_names
    if len(labels) != len(group_names):
        raise ValueError("legend_labels 与 group_names 长度须一致")
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = matplotlib.colormaps["tab10"].resampled(max(len(group_names), 1))
    start = 0
    for i, name in enumerate(group_names):
        sl = slice(start, start + lens[i])
        ax.scatter(
            Z2[sl, 0],
            Z2[sl, 1],
            s=42,
            alpha=0.85,
            color=cmap(i),
            label=labels[i],
            edgecolors="white",
            linewidths=0.4,
        )
        start += lens[i]
    ax.legend(markerscale=1.15, fontsize=9, framealpha=0.94, loc="best")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


ALL_GROUPS = (
    "teleop_model_robot_data",
    "teleop_model_umi_data",
    "umi_model_robot_data",
    "umi_model_umi_data",
    "umi_model_umi_other_data",
    "umi_model_umi_bagging_data",
)

_GROUPS_NOT_IN_ALL = frozenset({"umi_model_umi_other_data", "umi_model_umi_bagging_data"})


def _parse_groups(s: str) -> tuple[str, ...]:
    s = s.strip().lower()
    if s == "all":
        return tuple(x for x in ALL_GROUPS if x not in _GROUPS_NOT_IN_ALL)
    parts = tuple(x.strip() for x in s.split(",") if x.strip())
    for p in parts:
        if p not in ALL_GROUPS:
            raise SystemExit(f"未知 --groups 项 {p!r}；可选: {ALL_GROUPS} 或 all")
    return parts


def run():
    ap = argparse.ArgumentParser(
        description="PI 风格 VLM 最后一层表征：仅遥操 vs 仅 UMI 模型 × 机器人/UMI 数据"
    )
    ap.add_argument("--teleop-config", type=str, required=True, help="双臂 Franka / 遥操微调 TrainConfig 名")
    ap.add_argument("--umi-config", type=str, required=True, help="XV/UMI 微调 TrainConfig 名")
    ap.add_argument(
        "--teleop-checkpoint",
        type=str,
        default=None,
        help="遥操微调 checkpoint（含 model.safetensors）；若 --groups 不含 teleop_model_* 可省略",
    )
    ap.add_argument(
        "--umi-checkpoint",
        type=str,
        default=None,
        help="UMI 微调 checkpoint；若 --groups 不含 umi_model_* 可省略",
    )
    ap.add_argument(
        "--robot-repo",
        type=str,
        default=None,
        help="机器人 LeRobot root；若 --groups 不含 *_robot_data 可省略",
    )
    ap.add_argument(
        "--umi-repo",
        type=str,
        default=None,
        help="UMI handover LeRobot root（图例作 umi_handover_data；常为 handover_umi_0429 / mix）；若含 teleop_model_umi_data / umi_model_umi_data",
    )
    ap.add_argument(
        "--umi-other-repo",
        type=str,
        default=None,
        help="其它任务 UMI（如 duck，非 handover）；仅当 --groups 含 umi_model_umi_other_data",
    )
    ap.add_argument(
        "--umi-bagging-repo",
        type=str,
        default=None,
        help="bagging 任务 UMI LeRobot root；仅当 --groups 含 umi_model_umi_bagging_data",
    )
    ap.add_argument(
        "--groups",
        type=str,
        default="all",
        help=f"逗号分隔子集或 all；便于 zx1 / gxzx2 分机器跑。可选: {', '.join(ALL_GROUPS)}",
    )
    ap.add_argument(
        "--skip-tsne",
        action="store_true",
        help="跳过所有 2D 降维图（t-SNE / PCA / UMAP 等），只导出 .npy 与 report",
    )
    ap.add_argument(
        "--plots",
        type=str,
        default="tsne,pca,umap",
        help="逗号分隔，可选: tsne, pca, umap；在不使用 --skip-tsne 时生效",
    )
    ap.add_argument("--max-samples", type=int, default=128, help="每个数据集最多采样帧数（若使用 --use-all-samples 则不再截断）")
    ap.add_argument(
        "--use-all-samples",
        action="store_true",
        help="每个用到的数据集遍历全部帧（在磁盘/LeRobot 可见范围内；partial-repo 则为「可读的完整 episode」内全部帧）。"
        "帧数大时前向与 t-SNE/UMAP 会很慢，建议 tmux/nohup 或先 --skip-tsne 只导出 .npy。",
    )
    ap.add_argument("--pool-first-k", type=int, default=200, help="仅 representation=vlm：池化 prefix 前 K 个 token（PI 附录）")
    ap.add_argument(
        "--representation",
        type=str,
        choices=("vlm", "vision"),
        default="vlm",
        help="vlm=PaliGemma language_model 前缀 mean-pool（PI 附录）；vision=SigLIP+projector（embed_image，多相机 patch mean 再视图均值）",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="./analysis_results/pi_style_teleop_vs_umi")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--teleop-assets-from", type=str, default=None)
    ap.add_argument("--teleop-asset-id", type=str, default=None)
    ap.add_argument("--umi-assets-from", type=str, default=None)
    ap.add_argument("--umi-asset-id", type=str, default=None)
    ap.add_argument(
        "--robot-repo-adapter",
        type=str,
        default=None,
        help="机器人集若为外部三相机+68D，填 external_three_cam_state68",
    )
    ap.add_argument(
        "--robot-partial-repo",
        action="store_true",
        help="机器人 repo 未拷完：扫描磁盘上 parquet+视频齐全的 episode，仅用这些 episode 构造 "
        "LeRobotDataset（离线可用，避免缺文件时触发 HF）",
    )
    ap.add_argument(
        "--robot-partial-max-tries",
        type=int,
        default=8000,
        help="与 --robot-partial-repo 配合：最多尝试多少次 ds[i]",
    )
    args = ap.parse_args()

    groups = _parse_groups(args.groups)
    need_teleop_model = any(g.startswith("teleop_model") for g in groups)
    need_umi_model = any(g.startswith("umi_model") for g in groups)
    need_robot_repo = any(g.endswith("robot_data") for g in groups)
    need_umi_repo = any(g in ("teleop_model_umi_data", "umi_model_umi_data") for g in groups)
    need_umi_other_repo = "umi_model_umi_other_data" in groups
    need_umi_bagging_repo = "umi_model_umi_bagging_data" in groups

    if need_teleop_model and not args.teleop_checkpoint:
        raise SystemExit("当前 --groups 需要遥操模型：请提供 --teleop-checkpoint")
    if need_umi_model and not args.umi_checkpoint:
        raise SystemExit("当前 --groups 需要 UMI 模型：请提供 --umi-checkpoint")
    if need_robot_repo and not args.robot_repo:
        raise SystemExit("当前 --groups 需要机器人数据：请提供 --robot-repo")
    if need_umi_repo and not args.umi_repo:
        raise SystemExit("当前 --groups 需要主 UMI 数据：请提供 --umi-repo")
    if need_umi_other_repo and not args.umi_other_repo:
        raise SystemExit("当前 --groups 含 umi_model_umi_other_data：请提供 --umi-other-repo")
    if need_umi_bagging_repo and not args.umi_bagging_repo:
        raise SystemExit("当前 --groups 含 umi_model_umi_bagging_data：请提供 --umi-bagging-repo")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    teleop_cfg = train_config_lib.get_config(args.teleop_config)
    umi_cfg = train_config_lib.get_config(args.umi_config)
    lk, rk, tk = teleop_data_config_image_keys(teleop_cfg)

    teleop_ckpt = None
    umi_ckpt = None
    if args.teleop_checkpoint:
        teleop_ckpt = Path(args.teleop_checkpoint).resolve()
        if teleop_ckpt.is_file():
            teleop_ckpt = teleop_ckpt.parent
    if args.umi_checkpoint:
        umi_ckpt = Path(args.umi_checkpoint).resolve()
        if umi_ckpt.is_file():
            umi_ckpt = umi_ckpt.parent

    transform_teleop = transform_umi = None
    model_teleop = model_umi = None
    if need_teleop_model:
        assert teleop_ckpt is not None
        ns_t = load_ns_safe(
            teleop_cfg,
            teleop_ckpt,
            args.teleop_asset_id,
            Path(args.teleop_assets_from) if args.teleop_assets_from else None,
        )
        policy_teleop = load_policy_for_transforms(
            teleop_cfg, teleop_ckpt, pytorch_device=device, norm_stats=ns_t
        )
        transform_teleop = policy_teleop._input_transform
        model_teleop = resolve_checkpoint_model(teleop_cfg, teleop_ckpt, pytorch_device=device)
    if need_umi_model:
        assert umi_ckpt is not None
        ns_u = load_ns_safe(
            umi_cfg,
            umi_ckpt,
            args.umi_asset_id,
            Path(args.umi_assets_from) if args.umi_assets_from else None,
        )
        policy_umi = load_policy_for_transforms(umi_cfg, umi_ckpt, pytorch_device=device, norm_stats=ns_u)
        transform_umi = policy_umi._input_transform
        model_umi = resolve_checkpoint_model(umi_cfg, umi_ckpt, pytorch_device=device)

    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as e:
        raise ImportError("需要 lerobot：`pip install lerobot`") from e

    ds_robot = ds_umi = ds_umi_other = ds_umi_bagging = None
    idx_r = idx_u = idx_uo = idx_ub = []
    robot_adapter = get_robot_adapter(args.robot_repo_adapter)
    if need_robot_repo:
        root_robot = Path(args.robot_repo).resolve()
        episodes_robot = None
        if args.robot_partial_repo:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

            meta_probe = LeRobotDatasetMetadata("local_eval", str(root_robot))
            episodes_robot = discover_locally_complete_episodes(root_robot)
            logger.info(
                "robot-partial-repo：磁盘完整 episode %d / meta.total_episodes=%d",
                len(episodes_robot),
                meta_probe.total_episodes,
            )
            if not episodes_robot:
                raise SystemExit(
                    "robot-partial-repo：未发现任何完整 episode（请检查 meta/ 与 data/、videos/ 是否拷齐）"
                )
        ds_robot = LeRobotDataset(
            repo_id="local_eval",
            root=str(root_robot),
            episodes=episodes_robot,
        )
        cap_r = _sample_cap_for_dataset(len(ds_robot), max_samples=args.max_samples, use_all=args.use_all_samples)
        idx_r = _collect_indices(ds_robot, cap_r, args.seed)
    if need_umi_repo:
        ds_umi = LeRobotDataset(repo_id="local_eval", root=args.umi_repo)
        cap_u = _sample_cap_for_dataset(len(ds_umi), max_samples=args.max_samples, use_all=args.use_all_samples)
        idx_u = _collect_indices(ds_umi, cap_u, args.seed + 1)
    if need_umi_other_repo:
        ds_umi_other = LeRobotDataset(repo_id="local_eval", root=args.umi_other_repo)
        cap_uo = _sample_cap_for_dataset(len(ds_umi_other), max_samples=args.max_samples, use_all=args.use_all_samples)
        idx_uo = _collect_indices(ds_umi_other, cap_uo, args.seed + 2)
    if need_umi_bagging_repo:
        ds_umi_bagging = LeRobotDataset(repo_id="local_eval", root=args.umi_bagging_repo)
        cap_ub = _sample_cap_for_dataset(len(ds_umi_bagging), max_samples=args.max_samples, use_all=args.use_all_samples)
        idx_ub = _collect_indices(ds_umi_bagging, cap_ub, args.seed + 3)

    total_scheduled = len(idx_r) + len(idx_u) + len(idx_uo) + len(idx_ub)
    if args.use_all_samples:
        logger.info("--use-all-samples：各集计划前向总帧数 ≈ %d（实际成功数以 counts_per_group 为准）", total_scheduled)
    if total_scheduled > 8000 and not args.skip_tsne:
        logger.warning(
            "总帧数 %d 较大，联合 t-SNE/UMAP 可能极慢或内存不足；可改用 --plots pca 或 --skip-tsne 仅导出向量",
            total_scheduled,
        )

    logger.info(
        "groups=%s | robot=%d umi=%d umi_other=%d umi_bagging=%d | teleop图像槽: %s %s %s",
        groups,
        len(idx_r),
        len(idx_u),
        len(idx_uo),
        len(idx_ub),
        lk.split(".")[-1],
        rk.split(".")[-1],
        tk.split(".")[-1],
    )

    feats: dict[str, list[np.ndarray]] = {k: [] for k in ALL_GROUPS}

    if need_robot_repo and ds_robot is not None:
        for i in idx_r:
            sample = ds_robot[i]
            try:
                flat = robot_sample_to_dual_flat(sample, robot_adapter, lk, rk, tk)
                row_xv = dual_franka_flat_to_xv_row(flat, lk=lk, rk=rk, tk=tk)
                if "teleop_model_robot_data" in groups:
                    proc_tt = transform_teleop(flat.copy())
                    obs_tt = processed_dict_to_observation(proc_tt, device=device)
                    feats["teleop_model_robot_data"].append(
                        extract_repr_vector(
                            model_teleop,
                            obs_tt,
                            representation=args.representation,
                            pool_first_k=args.pool_first_k,
                        )
                    )
                if "umi_model_robot_data" in groups:
                    proc_ut = transform_umi(row_xv.copy())
                    obs_ut = processed_dict_to_observation(proc_ut, device=device)
                    feats["umi_model_robot_data"].append(
                        extract_repr_vector(
                            model_umi,
                            obs_ut,
                            representation=args.representation,
                            pool_first_k=args.pool_first_k,
                        )
                    )
            except Exception as e:
                logger.warning("跳过 robot idx=%s: %s", i, e)

    if need_umi_repo and ds_umi is not None:
        for i in idx_u:
            sample = ds_umi[i]
            try:
                row_xv = umi_sample_to_xv_row(sample)
                flat = umi_sample_to_dual_flat(sample, lk, rk, tk)
                if "teleop_model_umi_data" in groups:
                    proc_tu = transform_teleop(flat.copy())
                    obs_tu = processed_dict_to_observation(proc_tu, device=device)
                    feats["teleop_model_umi_data"].append(
                        extract_repr_vector(
                            model_teleop,
                            obs_tu,
                            representation=args.representation,
                            pool_first_k=args.pool_first_k,
                        )
                    )
                if "umi_model_umi_data" in groups:
                    proc_uu = transform_umi(row_xv.copy())
                    obs_uu = processed_dict_to_observation(proc_uu, device=device)
                    feats["umi_model_umi_data"].append(
                        extract_repr_vector(
                            model_umi,
                            obs_uu,
                            representation=args.representation,
                            pool_first_k=args.pool_first_k,
                        )
                    )
            except Exception as e:
                logger.warning("跳过 umi idx=%s: %s", i, e)

    if need_umi_other_repo and ds_umi_other is not None:
        for i in idx_uo:
            sample = ds_umi_other[i]
            try:
                row_xv = umi_sample_to_xv_row(sample)
                if "umi_model_umi_other_data" in groups:
                    proc_uo = transform_umi(row_xv.copy())
                    obs_uo = processed_dict_to_observation(proc_uo, device=device)
                    feats["umi_model_umi_other_data"].append(
                        extract_repr_vector(
                            model_umi,
                            obs_uo,
                            representation=args.representation,
                            pool_first_k=args.pool_first_k,
                        )
                    )
            except Exception as e:
                logger.warning("跳过 umi_other idx=%s: %s", i, e)

    if need_umi_bagging_repo and ds_umi_bagging is not None:
        for i in idx_ub:
            sample = ds_umi_bagging[i]
            try:
                row_xv = umi_sample_to_xv_row(sample)
                if "umi_model_umi_bagging_data" in groups:
                    proc_ub = transform_umi(row_xv.copy())
                    obs_ub = processed_dict_to_observation(proc_ub, device=device)
                    feats["umi_model_umi_bagging_data"].append(
                        extract_repr_vector(
                            model_umi,
                            obs_ub,
                            representation=args.representation,
                            pool_first_k=args.pool_first_k,
                        )
                    )
            except Exception as e:
                logger.warning("跳过 umi_bagging idx=%s: %s", i, e)

    saved_names: list[str] = []
    lens: list[int] = []
    for k in groups:
        if not feats[k]:
            logger.error("本组无有效样本 %s", k)
        else:
            stacked = np.stack(feats[k], axis=0)
            np.save(out_dir / f"{k}.npy", stacked)
            saved_names.append(k)
            lens.append(stacked.shape[0])

    if not saved_names:
        raise SystemExit("未保存任何表征向量。")

    blocks_saved = [np.load(out_dir / f"{k}.npy") for k in saved_names]
    embedding_separation = _embedding_separation_stats(saved_names, blocks_saved)

    rs = args.representation
    repr_short = (
        "vision encoder (SigLIP→projector, mean patches / mean views)"
        if rs == "vision"
        else "VLM prefix (language_model, mean first-K)"
    )
    repr_note = (
        "SigLIP vision_tower → multi_modal_projector（与 embed_image 一致）；每相机 patch 加权 mean，再三相机算术平均。"
        if rs == "vision"
        else "PaliGemma language_model last_hidden_state；prefix 前 min(K,T) token 加权 mean（PI 附录）。"
    )

    do_plots = not args.skip_tsne and len(saved_names) >= 1
    want_plots = {p.strip().lower() for p in args.plots.split(",") if p.strip()}
    plots_info: dict = {"requested": sorted(want_plots), "files": []}
    plot_legend_labels = legend_labels_for_groups(saved_names)

    if do_plots:
        lens = [b.shape[0] for b in blocks_saved]
        Z = np.concatenate(blocks_saved, axis=0)
        n_z = Z.shape[0]

        if "pca" in want_plots:
            pca = PCA(n_components=2, random_state=args.seed)
            Z_pca = pca.fit_transform(Z)
            ev = pca.explained_variance_ratio_.tolist()
            plots_info["pca_explained_variance_ratio"] = ev
            title_pca = (
                f"PCA — {repr_short}\n"
                f"PC1={ev[0]:.1%}   PC2={ev[1]:.1%} cumulative={(ev[0]+ev[1]):.1%}"
            )
            pca_path = out_dir / f"pca_pi_style_{rs}.png"
            _plot_group_scatter_2d(
                Z_pca,
                saved_names,
                lens,
                title=title_pca,
                out_path=pca_path,
                legend_labels=plot_legend_labels,
            )
            plots_info["files"].append(str(pca_path.name))

        if "tsne" in want_plots:
            perp = min(30, max(2, n_z // 4), n_z - 1)
            Z_tsne = TSNE(n_components=2, perplexity=perp, random_state=args.seed).fit_transform(Z)
            plots_info["tsne_perplexity"] = int(perp)
            _plot_group_scatter_2d(
                Z_tsne,
                saved_names,
                lens,
                title=f"t-SNE — {repr_short}",
                out_path=out_dir / f"tsne_pi_style_{rs}.png",
                legend_labels=plot_legend_labels,
            )
            plots_info["files"].append(f"tsne_pi_style_{rs}.png")

        if "umap" in want_plots:
            try:
                import umap

                n_neighbors = max(2, min(15, n_z - 1))
                Z_umap = umap.UMAP(
                    n_components=2,
                    random_state=args.seed,
                    n_neighbors=n_neighbors,
                    min_dist=0.15,
                    metric="euclidean",
                ).fit_transform(Z)
                plots_info["umap_n_neighbors"] = int(n_neighbors)
                _plot_group_scatter_2d(
                    Z_umap,
                    saved_names,
                    lens,
                    title=f"UMAP — {repr_short}",
                    out_path=out_dir / f"umap_pi_style_{rs}.png",
                    legend_labels=plot_legend_labels,
                )
                plots_info["files"].append(f"umap_pi_style_{rs}.png")
            except ImportError:
                logger.warning("未安装 umap-learn，跳过 UMAP（pip install umap-learn）")
    elif args.skip_tsne:
        logger.info("已 --skip-tsne，未生成降维图；可用 merge 脚本合并多机 .npy 后再画。")

    report = {
        "groups_requested": list(groups),
        "saved_groups": saved_names,
        "plot_legend_labels": dict(zip(saved_names, plot_legend_labels)),
        "teleop_config": args.teleop_config,
        "umi_config": args.umi_config,
        "teleop_checkpoint": str(teleop_ckpt) if teleop_ckpt else None,
        "umi_checkpoint": str(umi_ckpt) if umi_ckpt else None,
        "robot_repo": args.robot_repo,
        "umi_repo": args.umi_repo,
        "umi_other_repo": args.umi_other_repo,
        "umi_bagging_repo": args.umi_bagging_repo,
        "representation": args.representation,
        "representation_note": repr_note,
        "max_samples_requested": args.max_samples,
        "use_all_samples": bool(args.use_all_samples),
        "counts_per_group": dict(zip(saved_names, lens)),
        "pool_first_k": args.pool_first_k if args.representation == "vlm" else None,
        "dual_franka_image_keys": {"left": lk, "right": rk, "third": tk},
        "robot_partial_repo": bool(args.robot_partial_repo),
        "robot_partial_max_tries": args.robot_partial_max_tries if args.robot_partial_repo else None,
        "plots_generated": do_plots,
        "plots_detail": plots_info if do_plots else {},
        "embedding_separation": embedding_separation,
        "reference": (
            "Physical Intelligence appendix: VLM prefix mean-pool"
            if args.representation == "vlm"
            else "openpi PaliGemma: SigLIP vision_tower + projector (embed_image), per-view patch mean then mean over cameras"
        ),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("已写入 %s", out_dir)


if __name__ == "__main__":
    run()
