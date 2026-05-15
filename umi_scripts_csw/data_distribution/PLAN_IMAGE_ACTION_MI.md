# 执行计划：冻结视觉特征 + 图像–动作依赖性分析

本文档供子 agent **直接按步骤实现脚本**，目标是在 **与 `pi05_xv_dual_finetune` 训练一致的 (图像, 动作 chunk)** 配对上，用 **冻结编码器** 得到 \(Z=f(X)\)，再估计 **\(Z\) 与动作向量 \(A\)** 之间的统计依赖性（互信息或其稳健替代）。

---

## 1. 目标与范围

| 项目 | 说明 |
|------|------|
| **目的** | 量化「当前帧（或多视角）图像表征」与「训练监督动作 chunk」之间的关联强度，用于数据质检 / 批次对比（非训练策略）。 |
| **不做** | 不更新视觉编码器；不把结果等同于理论 \(I(\text{像素}; \text{动作})\)（见 §6）。 |
| **默认训练配置** | `TrainConfig` 名：`pi05_xv_dual_finetune`（[`src/openpi/training/config.py`](../../src/openpi/training/config.py)）。 |
| **动作语义** | 与 [`XVDualInputs`](../../src/openpi/policies/xv_dual_policy.py) + `PadStatesAndActions` 一致：`actions` 形状 `(action_horizon, action_dim)`，有效前两维为 **20**（左 10 + 右 10），其余为 pad 至 **32**。 |

---

## 2. 配对定义（必须与 DataLoader 一致）

训练时 LeRobot 使用 `delta_timestamps`：对 keys `left_action`、`right_action`，在 **当前帧时刻** 取未来 `t = 0, 1, …, H-1`（以 `fps` 换算为秒）。参考 [`create_torch_dataset`](../../src/openpi/training/data_loader.py)。

**每条样本（一行数据）定义为：**

- **图像 \(X\)**：与训练相同的 **当前帧** 三路图像（经与训练相同的 transform 后），键名与 policy 一致：`base_0_rgb`（third）、`left_wrist_0_rgb`、`right_wrist_0_rgb`（见 `XVDualInputs` 映射）。
- **动作 \(A\)**：同一条样本在 **完整 transform 链末尾** 的 `actions`，即 **已归一化（若启用 Normalize）且已 pad 到 32 维** 的张量，形状 `(H, 32)`，其中 **`A_eff = actions[:, :20]`** 为与策略相关的有效部分（实现时可 **flatten 为长度 `H*20=200`** 作为向量；若需减轻估计难度，可只对 **`actions[:, :20]`** flatten，不拼接 pad 维）。

**禁止**：用原始 parquet 列直接当 \(A\)，而不跑 `repack → XVDualInputs → Normalize → PadStatesAndActions`。

---

## 3. 前置条件

1. **环境与仓库**：在 openpi 根目录；可 import `openpi.training.config`、`openpi.training.data_loader` 等。
2. **数据集**：`TrainConfig.data.repo_id` 指向的 LeRobot 数据集本地可用（与配置一致）。
3. **归一化统计**：`data_config.norm_stats` 已存在（通常需事先对同一 config 跑过 `scripts/compute_norm_stats.py`）。若缺失，计划应先报错并提示运行该脚本（与训练相同）。
4. **硬件**：推理冻结编码器建议 **GPU**（CPU 仅适合小抽样）。

---

## 4. 推荐实现路径（复用训练 DataLoader）

**原则**：与训练共用 **`transform_dataset`** + **`create_torch_dataset`**，保证 `(X, A)` 与一步训练 batch 定义一致。

### 4.1 伪代码流程

```
1. train_cfg = config.get_config("pi05_xv_dual_finetune")
2. model_cfg = train_cfg.model
3. data_cfg = train_cfg.data.create(assets_dirs, model_cfg)   # 或沿用训练入口里等价调用
4. base_ds = create_torch_dataset(data_cfg, model_cfg.action_horizon, model_cfg)
5. ds = transform_dataset(base_ds, data_cfg, skip_norm_stats=False)
6. loader = DataLoader(ds, batch_size=B, shuffle=False, num_workers=..., collate_fn=官方若需要则自定义)

7. For each batch item (或 collate 后逐样本):
       sample = apply transforms 的单条 dict  # 若自定义迭代，需与框架一致
       imgs = sample["image"]   # dict of HWC uint8 或 float，以 transform 后为准
       actions = sample["actions"]  # (H, 32)

8. Z = FrozenEncoder(imgs)   # §5
   a = flatten(actions[:, :20])  # (H*20,)

9. 累积所有 (Z, a) 到内存或磁盘 shard（大数据集须流式写 npz）
10. 在全体样本或分层子集上计算依赖性指标 §7
```

**注意**：若默认 `DataLoader` collate 将 dict 结构拍平，需对照 [`create_data_loader`](../../src/openpi/training/data_loader.py) 或写一个 **仅用于分析的简单 Iterable**，逐条 `dataset[i]` 应用 transform（与 `TransformedDataset` 行为一致）。

### 4.2 抽样策略（必做）

全量遍历千万帧成本高。实现 **CLI 参数**：

- `--max-samples N`（默认例如 10000）
- `--seed`
- 可选 `--every-k-frames` 或按 episode 均匀抽样（若可从 dataset 取 episode id）

保证 **可复现**。

---

## 5. 冻结视觉编码器 \(f(X)\rightarrow Z\)

### 5.1 方案 A（推荐：与 Pi 图像塔一致）

- 使用 openpi 内 **SigLIP / PaliGemma image tower**（[`pi0.py`](../../src/openpi/models/pi0.py) 中 `PaliGemma.img`）加载 **冻结权重**，对每路 `224×224` 图像提 **patch/token 池化后的向量**（具体维宽度由 variant 决定）。
- **预处理**：必须与训练一致：`ResizeImages(224,224)` 已在 `model_transforms` 中；若模型 forward 还有 `preprocess_observation` 中的归一化，应用同一套（避免手写另一种 ImageNet mean/std 除非确认等价）。

**子 agent 任务**：查清 `_model.preprocess_observation` 对 `images` 的 dtype/范围，分析脚本输出与训练一步中的 image 张量一致。

### 5.2 方案 B（备选：快速原型）

- **PyTorch + OpenCLIP / timm-SigLIP**，冻结参数；输入 **224 RGB**，使用库自带 preprocess。
- **缺点**：与 Pi05 实际 SigLIP 变体可能不一致，**数值不可与方案 A 横向对比预训练策略**，仅适合粗筛。

### 5.3 多视角融合（必选其一并写死在配置里）

| 模式 | 做法 |
|------|------|
| `concat` | 三路分别 \(f(X)\)，再 `concat([z_left, z_right, z_base])` → 单一 \(Z\) |
| `third_only` | 仅用 `base_0_rgb`，与「主要依赖第三视角」假设一致 |
| `mean` | 三路 \(z\) 平均（仅在维数相同时） |

CLI：`--view-mode concat|third_only|mean`。

---

## 6. 依赖性度量（\(Z\) 与 \(A\)）

高维连续变量上 **精确 MI 难求**。实现时 **至少一种主指标 + 一种对照**：

### 6.1 主指标：HSIC（推荐）

- **HSIC**（Hilbert-Schmidt Independence Criterion）用核衡量依赖性，**无需密度估计**，对多维 \(A\) 较稳。
- 实现：对 \(Z\in\mathbb{R}^{d_z}\)、\(A\in\mathbb{R}^{d_a}\)（\(d_a=200\)）用 **RBF 核**，带宽 **median heuristic**（基于成对距离的中位数）。
- 输出：`HSIC(Z, A)` 标量；可按 **episode / task 分层** 各算一个。

依赖：`jax`/`torch`/`sklearn` 自写 20 行或用小库（需 pin 版本）。

### 6.2 对照：降维 + kNN 互信息（可选）

- 对 \(A\) 做 **PCA** 至 `k_a` 维（如 10），或与 \(Z` concat 后对联合分布用 **Kraskov kNN MI**（若引入 `skmisc`/`minepy` 等，须在 `requirements` 或 README 注明）。
- 报告 **`I(Z; A_pca)`** 或 **`I(Z_pc; a_pc)`** 时写明预处理，避免与 HSIC 混读。

### 6.3 禁止误认为「跨模型可比绝对值」

- 不同编码器、不同 \(N\) 的 HSIC/MI **不可直接比绝对大小**；批次对比时应 **固定 \(f\)、固定 \(N\)、固定 view-mode**。

---

## 7. 输出物（验收）

| 输出 | 格式 |
|------|------|
| 配置快照 | `run_config.json`：`repo_id`、`config_name`、`N`、`seed`、`view_mode`、编码器方案 |
| 汇总指标 | `metrics.json`：`hsic`、`n_samples`、可选分层键 |
| 可选 | `z_cache.npz` 的 `--save-embeddings`（大数据慎用） |
| 可选图 | `task_id` vs HSIC 条形图（若分层） |

---

## 8. CLI 建议接口（子 agent 实现时对齐）

```
python -m umi_scripts_csw.data_distribution.scripts.image_action_dependence \
  --config-name pi05_xv_dual_finetune \
  --max-samples 10000 \
  --seed 0 \
  --view-mode concat \
  --encoder openpi_siglip \
  --metric hsic \
  --out-dir ./reports/image_action_mi_run001
```

（脚本路径可调整；若放在 `scripts/` 下需 `__main__` 入口。）

---

## 9. 验收检查清单

- [ ] `actions` 来自 **完整训练 transform**，形状 `(10, 32)`，分析使用 **`[:, :20]`**。
- [ ] 图像时间索引与 `delta_timestamps` **一致**（当前帧图像对未来 chunk）。
- [ ] 编码器 **eval / no_grad**，权重冻结。
- [ ] 固定 `seed` 与 `max-samples`，同一数据重复运行指标一致。
- [ ] `discrete_state_input` 与本分析无关；**不在 MI 中 concat state**（与用户意图一致）。

---

## 10. 已知风险与规避

| 风险 | 规避 |
|------|------|
| `norm_stats` 缺失 | 与训练相同方式生成 |
| 内存爆炸 | 流式累积统计量或分批 HSIC（或先降采样 \(Z\)） |
| JAX/Torch 混用 | 选定一条栈；方案 A 可能偏 JAX |
| 标签泄漏误解 | 明确写的是 **监督目标 \(A\)** 与 **图像** 的依赖性，不是因果 |

---

## 11. 参考文献（概念）

- 数据处理不等式：\(I(X;A) \geq I(f(X);A)\)，冻特征 MI/HSIC 为代理。
- HSIC：Gretton et al.（核方法与独立性检验）。

---

**文档版本**：与 [`DATA_DISTRIBUTION_ROADMAP.md`](DATA_DISTRIBUTION_ROADMAP.md) 中「图像–动作」讨论一致；实现完成后可把脚本路径回链到该路线图。
