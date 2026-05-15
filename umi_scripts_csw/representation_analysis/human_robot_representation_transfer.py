#!/usr/bin/env python3
"""
可选 CLI：Human↔Robot 的 2×2 表征对比（真实前向）。

共享逻辑亦被 ``teleop_vs_umi_vlm_repr_pi_style.py`` import；论文主线「多组数据 × vlm/vision」请优先使用该脚本。

目标（与你的论文表述一致）：
  对每个观测 O，用「人类域预处理」与「机器人域预处理」分别得到模型输入，再分别送入
  「人类数据上微调的模型」与「机器人数据上微调的模型」，提取中间表征并比较：

  - 同一模型：人类域输入 vs 机器人域输入（域偏移）
  - 同一域输入：人类模型 vs 机器人模型（训练差异）

默认表征：PaliGemma 语言模型在 **prefix**（图像 token + 语言 token）上的 last_hidden_state，
按有效 token mask 做均值池化，得到固定维度向量。

依赖：
  - openpi（本仓库 PYTHONPATH=src）
  - JAX 检查点须先转为 PyTorch：见 README 「convert_jax_model_to_pytorch」
  - LeRobot 数据集（本地 root）；人类侧键与 XV 一致；机器人侧若为外部「三相机+68D」格式可加
    ``--robot-repo-adapter external_three_cam_state68``（见 robot_dataset_adapters.py）。

用法示例（在仓库根目录）::

    PYTHONPATH=src:. python umi_scripts_csw/representation_analysis/human_robot_representation_transfer.py \\
      --human-config pi05_xv_dual_finetune \\
      --robot-config pi05_dual_franka_finetune_high_lumos \\
      --human-checkpoint /path/to/human_ckpt_with_model.safetensors \\
      --robot-checkpoint /path/to/robot_ckpt_with_model.safetensors \\
      --human-repo /path/to/handover_umi_lerobot \\
      --robot-repo /path/to/dual_franka_lerobot \\
      --max-samples 64 \\
      --out-dir ./analysis_results/human_robot_repr_run1
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
from sklearn.manifold import TSNE

# 仓库根：representation_analysis -> umi_scripts_csw -> openpi
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from openpi.models import model as openpi_model
from openpi.models_pytorch import pi0_pytorch
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.policies import policy_config
from openpi.training import checkpoints as ckpt_lib
from openpi.training import config as train_config_lib

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


def _tree_to_torch_batch(tree, *, device: str):
    """Add batch dim and move tensors to device (same idea as Policy.infer for PyTorch)."""

    def one(x):
        a = np.asarray(x)
        t = torch.from_numpy(a).to(device)
        if t.dim() == 0:
            return t.unsqueeze(0)
        return t.unsqueeze(0)

    return jax.tree.map(one, tree)


def _openpi_dataset_sample_to_raw_row(sample: dict) -> dict:
    """Build raw dict for XVDualInputs / DualFranka Inputs (eval_dual_relative 风格)."""
    lp = np.asarray(sample["left_eef_pos"], np.float32).reshape(3)
    lr = np.asarray(sample["left_eef_rotvec"], np.float32).reshape(3)
    rp = np.asarray(sample["right_eef_pos"], np.float32).reshape(3)
    rr = np.asarray(sample["right_eef_rotvec"], np.float32).reshape(3)
    dsl = sample.get("demo_start_pose_left")
    dsr = sample.get("demo_start_pose_right")
    if dsl is None:
        dsl = np.concatenate([lp, lr], axis=-1).astype(np.float32)
    else:
        dsl = np.asarray(dsl, np.float32).reshape(6)
    if dsr is None:
        dsr = np.concatenate([rp, rr], axis=-1).astype(np.float32)
    else:
        dsr = np.asarray(dsr, np.float32).reshape(6)
    return {
        "third_view": _to_numpy(sample["third_view"]),
        "left_view": _to_numpy(sample["left_view"]),
        "right_view": _to_numpy(sample["right_view"]),
        "left_eef_pos": lp,
        "left_eef_rotvec": lr,
        "left_gripper": np.asarray(sample["left_gripper"], np.float32).reshape(1),
        "right_eef_pos": rp,
        "right_eef_rotvec": rr,
        "right_gripper": np.asarray(sample["right_gripper"], np.float32).reshape(1),
        "demo_start_pose_left": dsl,
        "demo_start_pose_right": dsr,
        "task": str(sample.get("task", "")),
    }


def load_norm_stats_for_config(train_cfg, assets_ckpt_dir: Path, *, asset_id_override: str | None = None):
    """从 ``assets_ckpt_dir/assets/<asset_id>/`` 加载 norm_stats。"""
    data_cfg = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    aid = asset_id_override or data_cfg.asset_id
    if aid is None:
        raise ValueError(f"Config {train_cfg.name} has no asset_id; cannot load norm stats.")
    return ckpt_lib.load_norm_stats(assets_ckpt_dir / "assets", aid)


def load_policy_for_transforms(
    train_cfg,
    checkpoint_dir: Path,
    *,
    pytorch_device: str,
    norm_stats=None,
):
    """加载 Policy（可为 JAX 或 PyTorch），仅用于获得与训练一致的 _input_transform。"""
    kwargs = dict(pytorch_device=pytorch_device)
    if norm_stats is not None:
        kwargs["norm_stats"] = norm_stats
    return policy_config.create_trained_policy(train_cfg, checkpoint_dir, **kwargs)


def load_pi05_pytorch(train_cfg, checkpoint_dir: Path, *, pytorch_device: str) -> pi0_pytorch.PI0Pytorch:
    """
    仅加载 PI0Pytorch 权重。目录下必须有 model.safetensors。
    若为纯 JAX params，请先运行 examples/convert_jax_model_to_pytorch.py 转换。
    """
    ckpt = Path(checkpoint_dir)
    weight_path = ckpt / "model.safetensors"
    if not weight_path.exists():
        raise FileNotFoundError(
            f"未找到 {weight_path}。\n"
            "请先将 JAX 检查点转为 PyTorch（生成 model.safetensors），例如：\n"
            "  uv run examples/convert_jax_model_to_pytorch.py \\\n"
            f"    --checkpoint_dir <jax_ckpt_parent_or_self> \\\n"
            "    --config_name <与训练一致的 config> \\\n"
            "    --output_path <输出目录>\n"
            "然后将 --human-checkpoint / --robot-checkpoint 指向含 model.safetensors 的目录。"
        )
    model = train_cfg.model.load_pytorch(train_cfg, str(weight_path))
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model = model.to(pytorch_device)
    model.eval()
    return model


def load_pi05_pytorch_from_pt(train_cfg, pt_path: Path, *, pytorch_device: str) -> pi0_pytorch.PI0Pytorch:
    """从单个 .pt / .pth 加载（如 RLinf full_weights.pt），strict=False 并打印缺失键。"""
    model = pi0_pytorch.PI0Pytorch(train_cfg.model)
    raw = torch.load(pt_path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict):
        if "model_state_dict" in raw:
            raw = raw["model_state_dict"]
        elif "model" in raw:
            raw = raw["model"]
        elif "actor" in raw and isinstance(raw["actor"], dict) and "model_state_dict" in raw["actor"]:
            raw = raw["actor"]["model_state_dict"]
    missing, unexpected = model.load_state_dict(raw, strict=False)
    if missing:
        logger.warning("load_state_dict missing keys (%d): %s ...", len(missing), missing[:8])
    if unexpected:
        logger.warning("load_state_dict unexpected keys (%d): %s ...", len(unexpected), unexpected[:8])
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model = model.to(pytorch_device)
    model.eval()
    return model


def resolve_checkpoint_model(train_cfg, path: str | Path, *, pytorch_device: str) -> pi0_pytorch.PI0Pytorch:
    p = Path(path).resolve()
    if p.is_file() and p.suffix in (".pt", ".pth", ".bin"):
        return load_pi05_pytorch_from_pt(train_cfg, p, pytorch_device=pytorch_device)
    return load_pi05_pytorch(train_cfg, p, pytorch_device=pytorch_device)


def processed_dict_to_observation(processed: dict, *, device: str) -> openpi_model.Observation:
    """apply transforms 之后的 dict -> Observation（单 batch）。"""
    proc = jax.tree.map(lambda x: np.asarray(x), processed)
    batch = _tree_to_torch_batch(proc, device=device)
    # Drop keys not used by Observation.from_dict
    keep = {
        k: batch[k]
        for k in batch
        if k in ("image", "image_mask", "state", "tokenized_prompt", "tokenized_prompt_mask")
    }
    return openpi_model.Observation.from_dict(keep)


@torch.no_grad()
def extract_prefix_llm_pooled(model: pi0_pytorch.PI0Pytorch, observation: openpi_model.Observation) -> np.ndarray:
    """PaliGemma LM last_hidden_state，按 prefix_pad_masks 有效位置均值池化 -> (D,) numpy。"""
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
    m = prefix_pad_masks.unsqueeze(-1).float()
    pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)
    return pooled[0].detach().cpu().numpy()


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA，X,Y: (n, d)。样本维一致。"""
    n = X.shape[0]
    if Y.shape[0] != n:
        raise ValueError(f"CKA sample mismatch: {X.shape[0]} vs {Y.shape[0]}")
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xt_y = X.T @ Y
    hsic = float(np.sum(xt_y * xt_y))
    a = float(np.sum((X.T @ X) ** 2))
    b = float(np.sum((Y.T @ Y) ** 2))
    return hsic / (np.sqrt(a * b) + 1e-12)


def linear_cka_subsample(X: np.ndarray, Y: np.ndarray, seed: int) -> float:
    """两集合样本数不同时，随机抽 ``min(nx,ny)`` 行（带替换的风险：独立抽样，适合集合相似度）。"""
    m = min(X.shape[0], Y.shape[0])
    rng = np.random.default_rng(seed)
    if X.shape[0] != m:
        xi = rng.choice(X.shape[0], size=m, replace=False)
        X = X[xi]
    if Y.shape[0] != m:
        yi = rng.choice(Y.shape[0], size=m, replace=False)
        Y = Y[yi]
    return linear_cka(X, Y)


def _collect_indices(ds, max_samples: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    n = len(ds)
    if n <= max_samples:
        return list(range(n))
    return sorted(rng.choice(n, size=max_samples, replace=False).tolist())


def run():
    ap = argparse.ArgumentParser(description="Human/Robot 双模型 × 双域输入 表征分析（真实前向）")
    ap.add_argument("--human-config", type=str, default="pi05_xv_dual_finetune")
    ap.add_argument("--robot-config", type=str, default="pi05_dual_franka_finetune_high_lumos")
    ap.add_argument("--human-checkpoint", type=str, required=True, help="含 model.safetensors 的目录，或单个 .pt/.pth")
    ap.add_argument("--robot-checkpoint", type=str, required=True)
    ap.add_argument("--human-repo", type=str, required=True, help="LeRobot 本地 root（人类/UMI 风格）")
    ap.add_argument(
        "--robot-repo",
        type=str,
        default=None,
        help="LeRobot 本地 root（双臂 Franka / openpi repack 格式）。与 --human-only 互斥时可省略",
    )
    ap.add_argument(
        "--human-only",
        action="store_true",
        help="仅人类域数据：两模型都只做人类预处理后的前向（跳过机器人数据集；无需 robot norm_stats）",
    )
    ap.add_argument("--max-samples", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="./analysis_results/human_robot_repr_transfer")
    ap.add_argument("--device", type=str, default=None, help="cuda / cuda:0 / cpu，默认自动")
    ap.add_argument(
        "--human-assets-from",
        type=str,
        default=None,
        help="人类模型 checkpoint 若无 assets 目录，指定另一含 assets/<asset_id>/ 的检查点根目录以加载 norm_stats",
    )
    ap.add_argument(
        "--human-asset-id",
        type=str,
        default=None,
        help="覆盖默认 TrainConfig 中的 asset_id（须与 checkpoint 内 assets/<name>/ 目录名一致，例如 handover_umi_0429_mix）",
    )
    ap.add_argument(
        "--robot-assets-from",
        type=str,
        default=None,
        help="机器人模型同上",
    )
    ap.add_argument(
        "--robot-asset-id",
        type=str,
        default=None,
        help="覆盖 robot TrainConfig 的 asset_id（须与 assets/<name>/ 目录一致，如 handover_mix_8sets）",
    )
    ap.add_argument(
        "--robot-repo-adapter",
        type=str,
        default=None,
        help="机器人 LeRobot 样本适配器：external_three_cam_state68（image/extra_view_* + 68D state）",
    )
    args = ap.parse_args()
    if not args.human_only and not args.robot_repo:
        raise SystemExit("缺少 --robot-repo；若暂无机器人格式数据集可加 --human-only")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    human_cfg = train_config_lib.get_config(args.human_config)
    robot_cfg = train_config_lib.get_config(args.robot_config)

    # Policies：仅用于各域的 input transform（可与推理一致）
    human_ckpt_dir = Path(args.human_checkpoint).resolve()
    robot_ckpt_dir = Path(args.robot_checkpoint).resolve()
    if human_ckpt_dir.is_file():
        human_ckpt_dir = human_ckpt_dir.parent
    if robot_ckpt_dir.is_file():
        robot_ckpt_dir = robot_ckpt_dir.parent

    logger.info("加载 Policy（变换链） human_ckpt_dir=%s robot_ckpt_dir=%s", human_ckpt_dir, robot_ckpt_dir)
    human_assets_root = Path(args.human_assets_from).resolve() if args.human_assets_from else human_ckpt_dir
    ns_human = load_norm_stats_for_config(human_cfg, human_assets_root, asset_id_override=args.human_asset_id)
    policy_human = load_policy_for_transforms(
        human_cfg, human_ckpt_dir, pytorch_device=device, norm_stats=ns_human
    )
    transform_human = policy_human._input_transform
    transform_robot = None
    robot_adapter = get_robot_adapter(args.robot_repo_adapter)
    if not args.human_only:
        robot_assets_root = Path(args.robot_assets_from).resolve() if args.robot_assets_from else robot_ckpt_dir
        ns_robot = load_norm_stats_for_config(
            robot_cfg, robot_assets_root, asset_id_override=args.robot_asset_id
        )
        policy_robot = load_policy_for_transforms(
            robot_cfg, robot_ckpt_dir, pytorch_device=device, norm_stats=ns_robot
        )
        transform_robot = policy_robot._input_transform

    logger.info("加载 PI0 PyTorch 权重（表征提取）")
    model_human = resolve_checkpoint_model(human_cfg, args.human_checkpoint, pytorch_device=device)
    model_robot = resolve_checkpoint_model(robot_cfg, args.robot_checkpoint, pytorch_device=device)

    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as e:
        raise ImportError("需要 lerobot 包以读取数据集：`pip install lerobot`") from e

    ds_h = LeRobotDataset(repo_id="local_eval", root=args.human_repo)
    ds_r = None
    idx_r = []
    if not args.human_only:
        ds_r = LeRobotDataset(repo_id="local_eval", root=args.robot_repo)

    idx_h = _collect_indices(ds_h, args.max_samples, args.seed)
    if ds_r is not None:
        idx_r = _collect_indices(ds_r, args.max_samples, args.seed + 1)

    feats = {
        "human_model_on_human_domain": [],
        "human_model_on_robot_domain": [],
        "robot_model_on_human_domain": [],
        "robot_model_on_robot_domain": [],
    }

    logger.info("人类数据集帧数=%d，采样 %d 条", len(ds_h), len(idx_h))
    for i in idx_h:
        sample = ds_h[i]
        raw = _openpi_dataset_sample_to_raw_row(sample)
        proc_h = transform_human(raw.copy())
        obs_h = processed_dict_to_observation(proc_h, device=device)
        feats["human_model_on_human_domain"].append(extract_prefix_llm_pooled(model_human, obs_h))
        feats["robot_model_on_human_domain"].append(extract_prefix_llm_pooled(model_robot, obs_h))

    if args.human_only:
        for k in ("human_model_on_robot_domain", "robot_model_on_robot_domain"):
            del feats[k]
    else:
        assert transform_robot is not None and ds_r is not None
        logger.info("机器人数据集帧数=%d，采样 %d 条", len(ds_r), len(idx_r))
        for i in idx_r:
            sample = ds_r[i]
            if robot_adapter is not None:
                raw = robot_adapter(sample)
            else:
                raw = _openpi_dataset_sample_to_raw_row(sample)
            proc_r = transform_robot(raw.copy())
            obs_r = processed_dict_to_observation(proc_r, device=device)
            feats["human_model_on_robot_domain"].append(extract_prefix_llm_pooled(model_human, obs_r))
            feats["robot_model_on_robot_domain"].append(extract_prefix_llm_pooled(model_robot, obs_r))

    for k in feats:
        feats[k] = np.stack(feats[k], axis=0)
        np.save(out_dir / f"{k}_prefix_llm.npy", feats[k])

    names = list(feats.keys())
    cka_mat = np.eye(len(names))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            c = linear_cka_subsample(feats[names[i]], feats[names[j]], seed=args.seed + 17 * i + 31 * j)
            cka_mat[i, j] = cka_mat[j, i] = c

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cka_mat, vmin=0.0, vmax=1.0, cmap="magma")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_yticklabels(names)
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.title("Linear CKA (subsampled to min count when n differs)")
    plt.tight_layout()
    plt.savefig(out_dir / "cka_matrix.png", dpi=150)
    plt.close()

    # t-SNE：合并四类（带标签）
    labels = []
    blocks = []
    label_ids = {n: i for i, n in enumerate(names)}
    for n in names:
        blocks.append(feats[n])
        labels.extend([label_ids[n]] * feats[n].shape[0])
    Z = np.concatenate(blocks, axis=0)
    labels = np.array(labels)
    n_z = Z.shape[0]
    if n_z < 8:
        logger.warning("样本过少 (%d)，跳过 t-SNE 图", n_z)
    else:
        perp = min(30, max(2, n_z // 4), n_z - 1)
        Z2 = TSNE(n_components=2, perplexity=perp, random_state=args.seed).fit_transform(Z)
        fig, ax = plt.subplots(figsize=(8, 6))
        for n in names:
            m = labels == label_ids[n]
            ax.scatter(Z2[m, 0], Z2[m, 1], s=12, alpha=0.7, label=n)
        ax.legend(markerscale=2)
        ax.set_title("t-SNE on pooled prefix LM representations")
        plt.tight_layout()
        plt.savefig(out_dir / "tsne_four_way.png", dpi=150)
        plt.close()

    report = {
        "human_only": bool(args.human_only),
        "human_config": args.human_config,
        "robot_config": args.robot_config,
        "human_checkpoint": str(Path(args.human_checkpoint).resolve()),
        "robot_checkpoint": str(Path(args.robot_checkpoint).resolve()),
        "human_repo": args.human_repo,
        "robot_repo": args.robot_repo,
        "robot_repo_adapter": args.robot_repo_adapter,
        "max_samples": args.max_samples,
        "representation": "paligemma.language_model prefix last_hidden_state (masked mean pool)",
        "cka_labels": names,
        "cka_matrix": cka_mat.tolist(),
        "notes": (
            [
                "human_only: 仅人类域数据；比较两模型在人类预处理输入下的表征差异（无机器人数据集分支）。",
                "CKA across unequal sample counts uses subsampling to min(n1,n2) (seeded).",
            ]
            if args.human_only
            else [
                "CKA across unequal sample counts uses subsampling to min(n1,n2) (seeded).",
                "Human-domain vs robot-domain preprocessing follows each train config's data transforms.",
            ]
        ),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary_lines = [
        "Human↔Robot 表征迁移分析（真实前向）",
        "",
        f"输出目录: {out_dir.resolve()}",
        f"样本数: 人类域 {len(idx_h)}"
        + ("" if args.human_only else f" / 机器人域 {len(idx_r)}"),
        "",
        "Linear CKA（集合之间，样本维对齐）:",
    ]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            summary_lines.append(f"  {names[i]}  vs  {names[j]} : {cka_mat[i, j]:.4f}")
    summary_lines += ["", "阅读顺序建议:"]
    if args.human_only:
        summary_lines.append(
            "  - human_only：仅「人类预处理」输入；比较 human_model vs robot_model 在该输入下的表征差异。"
        )
    else:
        summary_lines += [
            "  - 同行「*_on_human_domain」vs「*_on_robot_domain」→ 域偏移",
            "  - 同列「human_model_*」vs「robot_model_*」→ 两微调模型差异",
        ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    logger.info("完成。见 %s", out_dir.resolve())


if __name__ == "__main__":
    run()
