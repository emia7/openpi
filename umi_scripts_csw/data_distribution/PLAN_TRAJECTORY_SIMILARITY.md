# 执行计划：轨迹相似度（Episode 级）

**给读者的一页说明**（目标、原理、实现、注意）见 [`TRAJECTORY_SIMILARITY_WORKFLOW.md`](TRAJECTORY_SIMILARITY_WORKFLOW.md)。  
本文档供子 agent **按步骤实现** 与细节对照，目标是对 **同一 LeRobot / 训练口径下的双臂轨迹** 构造 **episode×episode 相似度矩阵**，用于发现 **多种「做法」**、**离群演示**、**簇结构**（后者为可选）。与 [`DATA_DISTRIBUTION_ROADMAP.md`](DATA_DISTRIBUTION_ROADMAP.md) §4.6 B 节一致。

---

## 1. 目标与范围

| 项目 | 说明 |
|------|------|
| **目的** | 在无任务标签前提下，量化 **整条 episode** 之间的形状相似度；输出距离矩阵 + 可选聚类/离群列表。 |
| **默认训练配置对齐** | `pi05_xv_dual_finetune`（[`src/openpi/training/config.py`](../../src/openpi/training/config.py)）：`action_horizon=10`，双序列键 `left_action`, `right_action`。 |
| **不做** | 不训练模型；不替代 [`compare_batches.py`](../data_cleaning/compare_batches.py) 的两目录边际对比（本计划侧重 **批内 episode–episode**）。 |

---

## 2. 轨迹表征（二选一，CLI 固定）

### 方案 A（推荐）：与策略监督空间一致

对 **每个时间步 \(t\)**，使用与训练相同的变换得到 **长度 \(T\) 的多维序列**（\(T\) 随 episode 变化）：

- 数据管道：`create_torch_dataset` + `transform_dataset`（与 [`data_loader.py`](../../src/openpi/training/data_loader.py) 一致）。
- **每条 episode**：按帧顺序遍历，对每一帧取 **`actions[t]`** 在 **完整 transform 之后** 的结果：
  - 若 DataLoader 样本是「当前帧图像 + **未来 chunk**」，则对本计划需改为：**逐帧构造「以 \(t\) 为起点」的 chunk** 或使用 **仅依赖当前观测与 future actions** 的 API；**更简单做法**：从 LeRobot 按索引读取 `left_action/right_action` **绝对 7D** 序列，**在脚本内调用与 `XVDualInputs` 等价的单步逻辑** 生成每时刻的 **20 维目标向量**（左 10 + 右 10），避免与训练错位。
- **实现约束**：子 agent 应 **复用** [`XVDualInputs`](../../src/openpi/policies/xv_dual_policy.py) 中「绝对 next → 相对当前 → pose9d+grip」的逻辑（可抽成函数），对 episode 内每个 \(t\) 输出一行 **20 维**，得到序列 **`S_A[t] ∈ R^{20}`**，长度 = episode 帧数（或有效长度）。

**优点**：相似度直接对应 **Pi05 学习的动作空间**。  
**缺点**：实现略重，必须与 `XVDualInputs` 数值一致。

### 方案 B（备选）：几何原生空间

直接使用 LeRobot 中 **逐帧**（对齐时间戳）的：

- `left_eef_pos(3)+left_eef_rotvec(3)+left_gripper(1)` 与右臂同理 → **每时刻 14 维**（或仅 xyz+clamp **8 维** 快速版）。

**优点**：实现快，适合粗筛。  
**缺点**：与 **相对 chunk 监督** 不一致，解释时只能说「末端轨迹几何相似」。

**CLI**：`--representation policy20 | geom14 | geom8`。

---

## 3. 预处理（在 DTW 之前）

对所有方案 **统一建议**（可按 `--skip-*` 关闭以便消融）：

| 步骤 | 说明 |
|------|------|
| **平移归一** | 每个 episode、每只手：位置减去 **该手第一帧位置**（方案 B）；方案 A 若已是相对变换可按需跳过或仅做尺度标准化。 |
| **旋转** | 方案 B 优先用 **rotvec** 序列或改为 **逐帧相对首帧的增量**（通过 `pose6_to_mat` / `convert_pose_mat_rep` 与 xv_dual 一致），避免四元数 wrap。 |
| **尺度** | 各维 **标准化**：在 **整批 episode 池** 上算均值方差（或稳健 median/MAD），再 z-score；**clamp/gripper** 维度单独统计，防止碾压位置维。 |
| **长度归一** | **可选**：均匀重采样到固定长度 `L_fixed`（例如 64），则可用 **欧氏距离 + Procrustes** 替代 DTW；CLI：`--resample L \| none`。 |

---

## 4. 相似度算法

### 4.1 主算法：多变量 DTW

- 输入：两条等维序列 `X ∈ R^{T1×D}`, `Y ∈ R^{T2×D}`（同一 `representation`）。
- 距离：逐步用 **闵可夫斯基距离**（默认 **L2**）在 \(D\) 维上算代价。
- 实现：`tslearn.metrics.dtw_path` / `softdtw` 或 `fastdtw`（大数据）；**必须**支持 **多元序列**（沿特征维聚合）。
- **Sakoe–Chiba 带**（可选）：`--dtw-band-ratio` 限制扭曲，防极端对齐。

### 4.2 备选（若 `--resample` 固定长度）

- 对齐长度后对整条序列算 **Fréchet 离散近似** 或 **逐帧 L2 均值**（仅当重采样后时间已对齐）。

### 4.3 双臂处理

| 模式 | 说明 |
|------|------|
| `separate` | **左臂一条矩阵、右臂一条矩阵**（方案 B 取左 7 维、右 7 维分别 DTW）；报告两次聚类结果。 |
| `concat` | 每时刻 **拼接** 左右特征（方案 A 已为 20 维天然拼接）。 |

CLI：`--arm-mode separate | concat`（`policy20` 建议固定 `concat`）。

---

## 5. 计算流程

```
1. 解析 TrainConfig / DataConfig，定位 LeRobot root。
2. 枚举 episode id（或从 dataset meta 读取 episode 边界）。
3. 对每个 episode：
       读取帧序列 → 构造 representation → 预处理 → 缓存为 list[np.ndarray], shape (T_i, D)
4. 若 episode 数 E 过大：
       --max-episodes E_max（均匀抽样或随机 seed）
5. 计算 E×E 对称距离矩阵 D（DTW 对称化：d_ij = d_ji；对角为 0）
       可用并行 / 上三角枚举
6. 后处理：
       - 层次聚类（scipy linkage + dendrogram）
       - 可选 HDBSCAN（距离矩阵输入 sklearn 兼容形式）
       - medoid：每簇 argmin sum_j d_ij
       - 离群：距离矩阵行和 Top-k 或 isolation on embedded space（可选 MDS 2D 可视化）
7. 写出报告
```

**复杂度**：\(O(E^2 \cdot f(T))\)；必须提供 **降采样时间步**（`--time-downsample k` 每 k 帧取 1）或 **E_max**。

---

## 6. 输出物

| 文件 | 内容 |
|------|------|
| `run_config.json` | `config_name`, `representation`, `arm_mode`, `resample`, `dtw_band`, `E`, `seed` |
| `distance_matrix.npz` | `episodes`（id 列表）, `D` shape `(E,E)` |
| `clusters.json` | 可选：簇标签、medoid episode id、参数 |
| `outliers.json` | 离群 episode id 及分数 |
| `mds2d.png` | 可选：距离矩阵 MDS 散点，按簇着色 |
| `dendrogram.png` | 可选：层次聚类树 |

---

## 7. CLI 建议

```
python -m umi_scripts_csw.data_distribution.scripts.trajectory_similarity \
  --config-name pi05_xv_dual_finetune \
  --representation policy20 \
  --arm-mode concat \
  --time-downsample 2 \
  --max-episodes 200 \
  --seed 0 \
  --metric dtw \
  --dtw-band-ratio 0.12 \
  --out-dir ./reports/traj_sim_run001
```

---

## 8. 验收检查清单

- [ ] `representation=policy20` 时，每帧 20 维与 **`XVDualInputs` 数值路径**一致（抽样若干帧与训练 batch 对比 rtol/atol）。
- [ ] 距离矩阵 **对称**，对角线为 0；**无 NaN**。
- [ ] 同一 `seed` + 同一数据，**距离矩阵可复现**。
- [ ] 文档中写明 **方案 B 与策略空间不一致**，避免误读。

---

## 9. 依赖建议

- `numpy`, `scipy`, `matplotlib`
- `tslearn`（DTW）或 `fastdtw` + 自写累积矩阵（二选一，写死版本）
- LeRobot / openpi 已有依赖；复用 `lerobot` dataset 读 parquet 或官方 episode 索引

---

## 10. 与现有脚本关系

- [`compare_batches.py`](../data_cleaning/compare_batches.py)：两批 JSON **边际/Δ 统计**；本计划为 **episode–episode 整体序列距离**。
- [`check_trajectory_anomalies.py`](../data_cleaning/check_trajectory_anomalies.py)：离群 **规则检测**；本计划离群来自 **几何/策略空间 DTW**，可交叉验证。

---

## 11. 风险

| 风险 | 缓解 |
|------|------|
| policy20 实现与训练漂移 | 单元测试：对比单 episode 与 DataLoader 中单步 `actions[:,:20]` |
| \(E^2\) DTW 过慢 | `max-episodes` + `time-downsample` + `fastdtw` |
| separate 双臂语义 | 输出两份矩阵并在 README 注明 |

---

**文档版本**：与 [`PLAN_IMAGE_ACTION_MI.md`](PLAN_IMAGE_ACTION_MI.md) 并列；分别交给不同子 agent 执行。
