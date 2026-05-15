# 数据分布可视化讨论与路线图

本文档沉淀「多批次 / 单批次具身采集数据」可视化方案的讨论结论：口径分层、方法分层、文献启发、与仓库脚本的关系，以及待尝试路径 backlog。**不包含**具体脚本实现与运行命令；面向 subagent 的逐步执行计划在讨论稿确认后**另档撰写**。

---

## 1. 背景与目标

### 1.1 单批次内（优先）

- 关心的问题：**episode 之间有多像、有多分散**，**时间轴上运动是否集中在相近相位**，工作空间内轨迹是否过窄或多模态，是否存在离群演示与对齐质量问题。
- 注意：单任务重复采集时，帧池边际分布「看起来窄」**不一定是坏事**；更要区分工艺问题（对齐差、夹爪几乎不变、极短/极长 episode 拖尾等）。

### 1.2 跨批次

- 关心的问题：分布偏移、长尾动作、对齐退化、任务阶段占比变化、不同采集日/配置下的可比性等。

---

## 2. 数据契约（示例：bagging 风格）

参考路径占位：`~/Downloads/bagging_0430`（本机示例；实际分析时替换为你的批次根目录）。

### 2.1 文件模式

- `episode_XXXXXX_left.json` / `episode_XXXXXX_right.json`（双臂成对）。

### 2.2 JSON 结构要点

- **顶层**：`view`、`fps`、`num_frames`、`alignment`（对齐误差摘要）、`records`。
- **每条 record**：`timestamp`、`pose`（7 维：位置 xyz + 四元数 xyzw）、`clamp`。

### 2.3 可视化单元（四层口径）

统计前务必固定口径，避免混用：

| 层级 | 含义 |
|------|------|
| 帧 | 单条 `records` 中的单个时间步 |
| Episode | 单个 `episode_*` 文件对应的整条轨迹 |
| 批次 | 同一目录/同一采集批次下的 episode 集合 |
| 臂别 | `left` / `right` — **建议默认分列**，必要时再做成对差分视图 |

与仓库脚本对齐：该结构与 [`../data_cleaning/compare_batches.py`](../data_cleaning/compare_batches.py) 文档中的期望一致；双臂批次可按臂别分子目录或同一目录下用文件名区分后分别聚合。

---

## 3. 单批次内数据分布（讨论要点）

在同一采集批次内，重点不是「与另一批次是否偏移」，而是 **批内异质性** 与 **相位/几何覆盖**。

### 3.1 统计口径

- **帧池级**：批次内所有时间步摊平 → 全局边际（xyz、四元数各分量、`clamp`、帧间隔）。
- **Episode 级**：每条 episode 汇总为标量（时长、`num_frames`、路径弧长、平均末端速度、`clamp` 极差、`alignment` 摘要等）→ **演示间差异**、长尾与离群 id。
- **双臂**：`left` / `right` **分开出图**；可选「成对 episode」的同步/位姿差视图。

### 3.2 建议的分析维度（由易到难）

1. **边际与尺度**：位置各轴、`clamp` 直方图或 KDE；四元数模长（应贴近 1）；`Δpos`、`Δrot`（平滑度与偶发大步跳变）。
2. **采集与时间**：`fps`、相邻 `timestamp` 间隔；episode 长度分布。
3. **工作空间与几何**：末端位置 2D 投影 + 密度/hexbin，或 3D 密度；多 episode 轨迹叠加（按 episode 上色）→ 窄带 vs 多簇。
4. **相位 / 归一化时间**：每条 episode 时间归一化到 \[0,1\]，叠加 z、夹爪或 ‖v‖ → 节奏与阶段是否对齐。
5. **Episode 异质性**：episode 级标量的箱线/小提琴图 + 离群标注；可与 [`../data_cleaning/check_trajectory_anomalies.py`](../data_cleaning/check_trajectory_anomalies.py) 报告对照。
6. **流形探索（可选）**：对 `(pose 或 Δpose, clamp)` 做 PCA/UMAP/t-SNE，按 episode 或时间段着色 → **探索用**，解释需谨慎。

### 3.3 批内易误判点

- 勿把「帧池边际」直接等同于「状态覆盖」叙事。
- 四元数边际可读性弱时，可改用 **轴角幅度** 或 **相对起始姿态的增量** 辅助解释。

---

## 4. 可视化方法分层（批内 + 跨批）

### 4.1 边际分布

pose 各维度、`clamp`、`Δpose`/`Δrot`、episode 长度、fps/帧间隔、`alignment` 相关标量。

### 4.2 时间与轨迹

按 episode 叠加末端 3D 轨迹；相位图（速度–位置等）；夹爪事件前后窗口统计（若有事件定义）。

### 4.3 跨批次对比

重叠直方图、ECDF；摘要表中带 KS、EMD（Wasserstein）等（辅助定量，不必过度解读 p 值）。

### 4.4 关节/位姿流形

对 `(pose, clamp)` 或差分特征降维；强调探索性用途。

### 4.5 图像 / 多模态（若下游有 LeRobot / 视频）

帧级 embedding 分布、按任务或批次着色；与「仅 JSON 轨迹」层级区分清楚。

### 4.6 更深层次数据分析（调研摘要）

**约定**：直方图、分位数、episode 时长与简单 Δ 统计等「第一层」描述统计，团队已做过；**交给 subagent 实现管线时在入口脚本里顺手跑一遍作为卫生检查即可**，本路线图的重点放在下面几类 **结构 / 分布 / 依赖** 更强的分析。

#### A. 分段与变点（时间结构）

- **目的**：在无任务标签时，看每条轨迹是否呈现稳定的「若干相位」（接近—操作—回撤等），以及不同 episode 的分段边界是否一致。
- **做法示例**：在标量序列上检测变点——如 ‖v‖、`clamp`、角速度范数、或末端到某参考点的距离；可用贝叶斯在线变点（例：关节运动建模）、PELT、或简单的滑动窗口统计突变。
- **文献线索**：在线贝叶斯变点与运动模型 [CHAMP / UMass](https://people.cs.umass.edu/~sniekum/pubs/CPD15.pdf)；演示分段与技能树 [CST](https://cs.brown.edu/people/gdk/pubs/cst-ws.pdf)；从演示做物体运动分段（与操纵阶段相关）[DITTO](https://arxiv.org/html/2403.15203v1)。

#### B. 轨迹相似度、对齐与无监督簇结构

- **目的**：回答「这批里到底有几种**做法**」比只看边际分布更直接；找 **离群轨迹**、 **主流模板**、 **少数替代策略**。
- **做法示例**：
  - 时间规整：**DTW**（或导数 DTW）在 **xyz 序列**、或与 `clamp` 拼接的低维序列上；得到 episode×episode 距离矩阵。
  - 形状对齐：**Procrustes**（平移归一化后比对形状）适合「几何轮廓相似但起点不同」的比法。
  - 聚类：在 DTW 距离或短时窗特征（统计量、PCA 系数）上做 **层次聚类 / HDBSCAN / 谱聚类**；簇大小与簇心轨迹可视化。
- **文献线索**：轨迹相似度与聚类在轨迹数据挖掘中的综述性讨论见 [轨迹大数据综述（科学出版社链接示例）](https://www.sciengine.com/parse/pdf/1000-9825/F0617EC7328646E08610DE674DC01E4F.pdf)（通用方法，非机器人专用）。

#### C. 分布级距离（超越一维 KS）

- **目的**：比较「**轨迹片段集合**」或「**整条轨迹特征向量集合**」在高维上是否同分布，而不只是一维边际。
- **做法示例**：将每条 episode 压缩为固定维特征（例如分段统计量、DTW 到 k 个 medoid 的距离、或小型编码器 embedding），对两组样本算 **MMD**（多核或高斯核）、**Energy distance**，或对嵌入做 **Sliced Wasserstein**；也可对 **滑动窗口** 特征做分布漂移监测。
- **文献线索**：MMD 在模仿学习与轨迹分布对齐中的经典讨论 [Maximum Mean Discrepancy Imitation Learning (RSS)](https://www.roboticsproceedings.org/rss09/p38.pdf)。

#### D. 信息流与依赖性（数据「有没有教」）

- **目的**：衡量 **状态（或位姿变化）与动作/夹爪** 之间可预测性与多样性——过低可能「僵死」或同步错；过高未必好但可提示噪声或混杂。
- **做法示例**：在 `(s_t, a_t)` 或 `(pose_t, Δpose_t, clamp_t)` 上估计 **互信息**（kNN、MINE、或论文中的 DemInf 流程）；或对「运动强度 vs clamp」算 **HSIC** / 条件独立性检验。
- **文献线索**：基于互信息估计的机器人演示策展 [Robot Data Curation with Mutual Information Estimators](https://arxiv.org/abs/2502.08623)；HSIC 概述见 [Gretton 讲义 PDF](https://www.gatsby.ucl.ac.uk/~gretton/coursefiles/lecture5_distribEmbed_2018.pdf)。

#### E. 潜在变量与「技能式」分解（探索性强）

- **目的**：把长轨迹投影到低维潜在空间，再看簇或选项结构；适合 **生成假设**，不适合单独写结论。
- **做法示例**：时序 VAE / Transformer 编码器 + 潜在空间聚类；与 RL 中无监督技能发现思路类比（若只做离线演示分析，可弱化奖励相关部分）。
- **文献线索（类比）**：[Unsupervised Skill Discovery for Robotic Manipulation…](https://arxiv.org/html/2410.04855v1)；选项与瓶颈 [Bottleneck Option Learning (ICML 2021)](http://proceedings.mlr.press/v139/kim21j/kim21j.pdf)。

#### F. 实现时注意

- **时间长度不一**：先 **时间归一化重采样** 或 **用 DTW / 固定数量统计特征**，避免强行堆叠张量。
- **旋转**：相似度与分段尽量用 **相对首帧旋转** 或 **角速度**，避免四元数_wrap Artifact。
- **双臂**：相似度、MMD、互信息 **按臂分别** 或 **拼接维数明确标注**，不要静默混合。
- **计算量**：全配对 DTW / 全流形为大 O；可先抽样 episode 或先用均匀重采样降长度。

---

## 5. 文献与实践启发（链接）

摘录可借用概念；正文不必复述论文全文。

| 主题 | 说明 | 链接 |
|------|------|------|
| 模仿学习数据质量 | 状态偏移、数据质量度量思路（可作「该画哪些标量」的来源） | [NeurIPS 2023 Poster](https://neurips.cc/virtual/2023/poster/72235) |
| 数据选择与价值 | DataMIL / Datamodels：数据选择与下游成功率 | [OpenReview](https://openreview.net/forum?id=AcTsKglDdh) |
| 分布偏移 | 跨 embodiment / 批次混合与偏移叙事 | [Ablett et al. PDF](https://starslab.ca/wp-content/papercite-data/pdf/2025_ablett_addressing.pdf) |
| 多视角与数据效率 | 视角增广与表征分布（若对比多相机批次） | [arXiv 2604.00557](https://arxiv.org/html/2604.00557v1) |
| 多模态数据集可视化类比 | 触觉等模态常用 t-SNE 类展示 | [arXiv 2510.25725](https://arxiv.org/html/2510.25725v2) |
| 轨迹分布 / MMD 与模仿学习 | 用嵌入分布距离刻画演示 vs 策略差异 | [RSS: MMD IL](https://www.roboticsproceedings.org/rss09/p38.pdf) |
| 互信息与演示策展 | DemInf：状态–动作互信息估计用于数据价值 | [arXiv 2502.08623](https://arxiv.org/abs/2502.08623) |
| 变点 / 演示分段 | 贝叶斯变点、技能树、演示变换中的分段思想 | [CPD15 PDF](https://people.cs.umass.edu/~sniekum/pubs/CPD15.pdf)、[CST PDF](https://cs.brown.edu/people/gdk/pubs/cst-ws.pdf)、[DITTO](https://arxiv.org/html/2403.15203v1) |
| 无监督技能发现（类比） | 离线分析潜在「模式」时的概念参照 | [arXiv 2410.04855](https://arxiv.org/html/2410.04855v1) |

---

## 6. 与仓库现有能力的关系

| 能力 | 路径 | 备注 |
|------|------|------|
| 两批 JSON 对比：pose/Δ、直方图、报告 | [`../data_cleaning/compare_batches.py`](../data_cleaning/compare_batches.py) | 双臂需按 left/right 分别指向目录或扩展调用方式 |
| 轨迹异常检测与汇总报告 | [`../data_cleaning/check_trajectory_anomalies.py`](../data_cleaning/check_trajectory_anomalies.py) | 可与 episode 级可视化对照 |

路线图实施时标注 **复用 / 扩展 / 新建**，避免重复造轮子。

---

## 7. 待尝试路径（Backlog）

可用勾选框维护；建议增加「日期 / 负责人 / 结论」列（表格或在 PR 中记录）。

### 批内

- [ ] **批内 P0（卫生检查，顺手）**：帧池边际（pose / clamp / Δ）+ episode 级时长与对齐摘要分布；left/right 分列。
- [ ] **批内 P1**：工作空间密度 / 轨迹叠加；归一化时间相位曲线。
- [ ] **批内 P2**：帧级特征降维散点（按 episode 着色）。
- [ ] **批内 D1（结构）**：标量序列变点 / 分段一致性（跨 episode 对齐分段边界或统计分段数）。
- [ ] **批内 D2（相似度）**：DTW 或 Procrustes episode 距离矩阵 + 聚类 / medoid 轨迹可视化。
- [ ] **批内 D3（分布）**：episode 级 embedding 的 MMD / Energy distance（批内子集 vs 子集或 vs 参考 medoids）。
- [ ] **批内 D4（依赖）**：状态–动作或 pose–clamp 互信息 / HSIC 的 episode 级或批次级摘要。

### 跨批

- [ ] **跨批 P0**：不同批次间 left/right 同一指标并列或差分。
- [ ] **跨批 P1**：重叠直方图、ECDF、KS/EMD 等摘要表。
- [ ] **跨批 P2**：降维散点按批次着色；与训练管线一致的特征空间（如 LeRobot 转换后）再画一遍。

---

## 8. 留给 Subagent 执行计划阶段的内容（占位）

以下内容**不在本文档展开**，待讨论稿确认后由负责人触发，**单独撰写** subagent 任务书：

- 输入路径约定、输出目录命名、是否生成静态 HTML。
- 随机种子、全量 vs 每批抽样 N 条 episode。
- 依赖与环境（Python 版本、可选库）。
- 脚本清单与验收标准（生成哪些图、哪些 JSON 摘要）。

---

## 9. 可选示意图（文档维护用）

```mermaid
flowchart LR
  subgraph batch_level [BatchLevel]
    meta[EpisodeMeta]
    marginals[MarginalsAndECDF]
    compare[CrossBatchTests]
  end
  subgraph traj_level [TrajectoryLevel]
    path3d[EEPaths3D]
    temporal[TimeSeriesPlots]
  end
  subgraph manifold [EmbeddingLevel]
    dimred[PCA_UMAP]
  end
  batch_level --> traj_level
  traj_level --> manifold
```

---

## 10. 验收（讨论稿）

- [x] 数据契约与四层可视化单元已写明。
- [x] 单批次内要点与跨批次分层已区分。
- [x] 方法分层、文献链接、backlog、与现有脚本关系已覆盖。
- [x] 「更深层次」分析方向（分段、相似度、MMD/依赖、文献）已写入 §4.6 与 §5、§7。
- [x] Subagent 执行细节仅占位，不与讨论稿混写。

全文以中文为主，专有名词保留英文。
