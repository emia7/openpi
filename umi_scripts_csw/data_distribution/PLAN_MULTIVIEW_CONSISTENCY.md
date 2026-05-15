# 执行计划：多视角嵌入一致性（冻结编码器）

本文档供子 agent **按步骤实现脚本**。目标是在 **与 `pi05_xv_dual_finetune` 训练一致的三路图像** 上，用**同一冻结编码器**得到 \(z_L, z_R, z_B\)，计算 **成对余弦相似度** 与 **单路时序平滑度**，用于批内质检（标定/同步/遮挡/坏帧的**弱信号**探针）。

**重要读数说明（必须写进脚本 `--help` 或 README 片段）**：左腕、右腕、第三视角 **视场与内容差异大**，**不得**期待两两余弦「普遍很高」。本计划依赖 **(1) 批内相对分位**、**(2) 同 episode 内时间突刺**、**(3) 单路 \(t\)–\(t+1\) 连续性**；**不要**用单一绝对阈值宣布「好/坏」。论证见本文 §6。

---

## 1. 目标与范围

| 项目 | 说明 |
|------|------|
| **目的** | 同一时刻三路图在**固定语义嵌入空间**中是否**异常不一致**；单路嵌入是否**时间突跳**。 |
| **默认训练配置** | `pi05_xv_dual_finetune`；图像键与 [`XVDualInputs`](../../src/openpi/policies/xv_dual_policy.py) 一致：`base_0_rgb`（third）、`left_wrist_0_rgb`、`right_wrist_0_rgb`。 |
| **不做** | 不训练多视角不变模型；不用几何重投影（无标定假设）；不把余弦当「像素应相似」。 |
| **强相关参考** | 编码器与预处理与 [`PLAN_IMAGE_ACTION_MI.md`](PLAN_IMAGE_ACTION_MI.md) **方案 A 对齐**（同骨干、同 `preprocess_observation` / resize），便于三条分析线可比。 |

---

## 2. 数据与 tensor 来源

1. 复用 **`create_torch_dataset` + `transform_dataset`**（与 [`data_loader.py`](../../src/openpi/training/data_loader.py) 一致），与 `PLAN_IMAGE_ACTION_MI` §4 相同。
2. 从每条 transform 后样本取 **`data["image"]`** 中三路；dtype/布局以训练一步为准（通常经 `ResizeImages(224,224)` 等）。
3. 需能映射 **frame / episode_id**（若 DataLoader 不直接返回，从 `dataset` 元数据或 `episode_index` 列拼接；与 `trajectory_similarity` 实现可共享索引逻辑）。

---

## 3. 冻结编码器 \(f\)

- **与 `PLAN_IMAGE_ACTION_MI` 方案 A 相同**：openpi 内 **SigLIP / PaliGemma image tower**，**eval、无梯度**。
- 对 **每路** 分别前向，得到 \(z_L, z_R, z_B \in \mathbb{R}^d\)，再 **L2 归一化**为 \(\hat z\)。
- **禁止**三路拼成一张图过单 forward（除非做额外消融）；默认 **三个独立 forward** 或 **batch 维堆叠**一次 forward。

---

## 4. 逐帧标量指标

对每一有效帧 \(t\)（在 `--max-samples` 或全量内）计算：

### 4.1 成对互视角（归一化后点积 = 余弦）

- \(c_{LR} = \hat z_L \cdot \hat z_R\)
- \(c_{LB} = \hat z_L \cdot \hat z_B\)
- \(c_{RB} = \hat z_R \cdot \hat z_B\)

**输出**：三列与 `episode_id` / `frame_index` / `global_index` 写入 `per_frame_metrics.csv`（或 parquet）。

### 4.2 单路时序（可选但推荐，`--temporal`）

- \(s_L = \hat z_{L,t} \cdot \hat z_{L,t+1}\)（同 episode 内相邻帧；边界帧跳过或标 NaN）
- 同理 \(s_R, s_B\)

用于抓 **单路黑屏跳变、时间戳错位、极端运动模糊**，不依赖「跨视角要像」。

---

## 5. 聚合与离群

1. **全局**：对 \(c_{LR}, c_{LB}, c_{RB}\) 各画 **直方图** + 报告 **p10/p50/p90**（分开展，不合并）。
2. **按 episode**：对每对指标求 **mean、min**；`min` 过低可标为「该条某时刻多视角最不一致」。
3. **离群规则（仅建议默认，可 CLI 调）**  
   - `episode` 级：`min(c_LB) < 全局 p10`（或比该 episode 内 **median−k·MAD**）则列入 `outliers.json`。  
   - `frame` 级：\(c_{LB}\) 或 \(s_B\) 单日 **Z-score** 超阈（需稳健估计，防误报）。
4. **禁止**：写死「cos < 0.5 即坏」类魔法数；若提供 `--absolute-threshold`，须 **默认关闭** 并打印警告。

---

## 6. 读数与局限性（产品说明）

| 事实 | 含义 |
|------|------|
| 第三视角与手腕内容差大 | **L–B、R–B** 余弦**绝对值常偏低**、方差大，属预期。 |
| 共享语义空间 | 同一场景下通常**非无关**，故仍可比 **时间突刺** 与 **批内尾部分位**。 |
| 弱于几何 | 有外参时 **重投影误差** 更硬；本 plan 不替代标定流程。 |

---

## 7. 输出物

| 文件 | 内容 |
|------|------|
| `run_config.json` | `config_name`, `max_samples`, `seed`, `encoder`, `temporal` on/off |
| `per_frame_metrics.csv` | 至少：`episode_id`, `frame_in_episode`, `c_LR`, `c_LB`, `c_RB`, 可选 `s_L,s_R,s_B` |
| `summary.json` | 各指标分位数、样本数、NaN 比例 |
| `outliers.json` | episode / frame 列表及分数 |
| `hist_pairwise.png` | 三路成对余弦三个子图 |
| `hist_temporal.png` | 可选：三路 \(s\) 直方图 |

---

## 8. CLI 建议

```
python -m umi_scripts_csw.data_distribution.scripts.multiview_consistency \
  --config-name pi05_xv_dual_finetune \
  --max-samples 20000 \
  --seed 0 \
  --encoder openpi_siglip \
  --temporal \
  --out-dir ./reports/multiview_consistency_run001
```

（模块路径可随仓库实际调整。）

---

## 9. 验收检查清单

- [ ] 三路图像来自 **与训练相同** transform；键名 `base_0_rgb` / `left_wrist_0_rgb` / `right_wrist_0_rgb`。
- [ ] 编码器 **冻结**；\(z\) **L2 归一化**后再算点积。
- [ ] 文档/帮助中注明 **勿用高余弦当唯一标准**（§6）。
- [ ] 输出中 **L–B 与 L–R 分开报告**，不混为单一「相似度」。
- [ ] 固定 `seed` 与 `max-samples` 可复现（相同缓存统计）。

---

## 10. 依赖与复用

- 与 `PLAN_IMAGE_ACTION_MI` **共享**编码器加载与图像预处理工具函数（建议抽到 `umi_scripts_csw.data_distribution.scripts._embeddings` 或等价模块）。
- `numpy` / `pandas` / `matplotlib`；训练侧 `jax`+模型权重路径按 openpi 惯例。

---

## 11. 与另两条分析线的关系

| 分析 | 关系 |
|------|------|
| **图像–动作** | 问「图能否解释动作」；本 plan 问「几路图像表征是否自洽/是否突变」。 |
| **轨迹相似度** | 低维动作几何；本 plan 仅视觉嵌入，**不混算**。 |

---

**文档版本**：试跑性质；若效果弱，优先依赖 **§4.2 时序项** 与 **episode 内 min 排名**，再考虑 Ridge \(z_L\to z_B\) 等升级版（可另开 plan）。
