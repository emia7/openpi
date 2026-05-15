# 表征可视化（VLM 前缀 vs 纯视觉）

本目录只做一件事：**用同一批 LeRobot 帧，经 openpi policy 预处理后，抽取向量并画 PCA / t-SNE / UMAP**，便于论文对比。

---

## 流程（端到端）

```
LeRobot 数据集
    → policy._input_transform（与训练一致的 norm / resize / 键）
    → PI0Pytorch 前向
         ├─ representation=vlm   : embed_prefix → language_model.last_hidden_state → 前 K token mean-pool
         └─ representation=vision: 各相机 embed_image（SigLIP→projector）→ patch mean → 三相机 mean
    → 按 --groups 分组堆叠向量 → 联合降维 → PNG + report.json + 各组 *.npy
```

**不含**：本体 `state` 走 suffix/action expert，此流水线只看前缀里的 **图像 + 任务文本（vlm）** 或 **仅图像塔（vision）**。

---

## 入口脚本

| 文件 | 作用 |
|------|------|
| **`teleop_vs_umi_vlm_repr_pi_style.py`** | **主入口**：`--representation {vlm, vision}`，四组或自定义 `--groups` |
| **`legend_config.py`** | 内部组名 → 图例显示名（如 `umi_handover_data`），不改 `.npy` 文件名 |
| **`human_robot_representation_transfer.py`** | **共享库**（加载 checkpoint、Observation、`load_policy_for_transforms`）；亦可单独跑另一套 2×2 CLI（与主线二选一写论文） |
| **`robot_dataset_adapters.py`** | 外部机器人数据格式适配 |
| **`merge_pi_style_repr_npys.py`** | 多机 `--skip-tsne` 只产 `.npy` 后合并画一张 t-SNE |

---

## 常用命令（在仓库根）

```bash
export PYTHONPATH=src:.
# 建议使用服务器上与训练一致的 venv，例如：
# /path/to/openpi/.venv/bin/python

python umi_scripts_csw/representation_analysis/teleop_vs_umi_vlm_repr_pi_style.py \
  --teleop-config pi05_dual_franka_finetune_high_lumos \
  --umi-config pi05_xv_dual_finetune \
  --umi-checkpoint /path/to/checkpoint_dir_with_model.safetensors \
  --umi-asset-id YOUR_ASSET_ID \
  --robot-repo /path/to/dual_franka_lerobot \
  --umi-repo /path/to/handover_umi_lerobot \
  --groups umi_model_robot_data,umi_model_umi_data \
  --representation vlm \
  --max-samples 128 \
  --out-dir ./umi_scripts_csw/representation_analysis/analysis_results/my_run
```

- **`--representation vision`**：纯图像编码器路径；输出文件名含 `_pi_style_vision.png`。
- **`--use-all-samples`**：每个用到的数据集用满全部帧（慎用：很慢）。
- **`--robot-partial-repo`**：数据未拷全时只扫描磁盘上完整 episode。
- **`--plots tsne,pca,umap`**、`--skip-tsne`：控制是否画图。

运维向一键脚本见 **`scripts/README.md`**。

---

## 输出

- `pca_pi_style_{vlm|vision}.png`、`tsne_*`、`umap_*`
- `report.json`：`representation`、`plot_legend_labels`、`embedding_separation`、PCA 方差比等
- `*.npy`：每组一行样本一行向量

生成的大文件位于 **`analysis_results/`**（已由仓库根 `.gitignore` 忽略）。

---

## 依赖

`requirements.txt` + 本仓库 **openpi**（`PYTHONPATH=src:.`）+ **LeRobot** 可读本地数据集。

若服务器 **根分区 `/` 空间紧张**，LeRobot 构建 HuggingFace `datasets` 缓存可能报错 `No space left on device`。可先指向大容量盘再跑主脚本，例如：

```bash
export HF_HOME=/mnt/public/$USER/.cache/hf_home_runner
export HF_DATASETS_CACHE=/mnt/public/$USER/.cache/hf_datasets_cache_runner
export TMPDIR=/mnt/public/$USER/.cache/tmp_openpi_repr
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR"
```

---

## 后续（可选）

- **显著性 / 注意力图**：在视觉 backbone 上对 patch 归因（需单独脚本，见课题讨论）。
