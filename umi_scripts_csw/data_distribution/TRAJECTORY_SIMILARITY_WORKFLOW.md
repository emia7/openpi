# 双臂 Episode 轨迹相似度分析：流程说明（速读）

面向 **快速理解「在算什么、怎么用、注意什么」**；实现细节与历史计划见 [`PLAN_TRAJECTORY_SIMILARITY.md`](PLAN_TRAJECTORY_SIMILARITY.md)。  
**入口脚本**：[`scripts/trajectory_similarity.py`](scripts/trajectory_similarity.py)（模块名 `umi_scripts_csw.data_distribution.scripts.trajectory_similarity`）。

---

## 1. 目标（要回答的问题）

| 目标 | 说明 |
|------|------|
| **成对比较整条演示** | 不只看单帧或统计量，而是把 **一条 episode 当成一条时序轨迹**，和另一条比「整体像不像」。 |
| **可复用的距离数据** | 核心产物是 **对称距离矩阵 `D`**（`D[i,j]` = episode i 与 j 的标量距离），便于后续自己筛离群、降维、画图或接聚类。 |
| **与训练空间可对齐（可选）** | 使用 `policy20` 时，轨迹处在与 **XVDual / Pi 微调动作监督** 一致（或尽量一致）的 20 维空间里，解释时可以说「在策略学习空间里有多像」。 |

**不做的**：不训练模型；不替代批与批之间的边际对比（那是别的脚本的事）。

---

## 2. 原理（在算什么）

1. **每条 episode** 在统一Representation 下变成矩阵 **`(T, D)`**：`T` 为时间步数（可下采样），`D` 为每步特征维数。  
2. **两条 episode** 之间定义一个标量 **距离** `d(·,·)`：  
   - **设了 `--resample L`（L>0）**（推荐用于大数据集）：先把两条轨迹 **按时间轴均匀重采样到固定长度 L**，使时间轴一一对应，再对每步算 L2 范数、**对时间取平均** → 即 **「对齐时间后的平均逐帧 L2」**（**不是** DTW，但快、可复现）。  
   - **未设 `--resample`**：当前实现走 **`--metric dtw`** 的多维 DTW 路径（更慢，适合小 `E` 或实验）。  
3. 对所有 episode 两两算距离，得到 **`D ∈ R^{E×E}`**，对角线为 0，**对称**。  
4. **离群（与分簇无关）**：用 **行和** `sum_j D[i,j]` 很大的 episode 作为「相对全体都远」的候选（`outliers.json`）。  
5. **（可选）层次聚类 / 图**：在 **同一张 `D`** 上可以做 linkage → **树状图**；**默认不输出离散簇**（`--cluster-mode none`），避免「一大团 + 单条轨迹」这种被指标误导的切分；需要簇时再开 `auto` 或 `manual`。

**直觉**：`D` 小 = 两条演示在选定特征、选定时间分辨率下 **整体走势接近**；`D` 大 = 节奏/位形/抓放等模式差得远（在 mix 数据里也可能只是 **任务不同** 而非「坏轨迹」）。

---

## 3. 实现方式（代码在干什么）

| 环节 | 实现要点 |
|------|----------|
| **配置** | 默认 `--config-name pi05_xv_dual_finetune`，用 openpi 的 `DataConfig` 取 `norm_stats` 等（与训练数据配置一致）。 |
| **读数据** | **全量 / 有视频** 时走 `LeRobotDataset`；**仅低维、免解码** 时用 `--parquet-lowdim-only` + `--dataset-root` 只读 `data/chunk-*/episode_*.parquet`。 |
| **policy20** | 与 [`xv_dual_policy.abs_next_targets_to_policy20_row`](../../src/openpi/policies/xv_dual_policy.py) 一致：把每步 eef + action 等压成 **20 维** 一行；可配合数据侧 `norm_stats.json` 做归一。 |
| **geom14 / geom8** | 末端位姿 + 夹爪的 **几何空间**（与监督空间不完全一致，但实现轻）。 |
| **预处理** | 默认对 **grip 等维** 在 episode 池上做 **z-score**（可用 `--skip-zscore` 等关闭）。几何方案里还可选 **按首帧平移归一**（`--skip-translation`）。 |
| **双臂** | `policy20` 固定 **concat**（20 维已含左右）；几何可选 `--arm-mode separate` 分左右两张矩阵。 |
| **输出** | `distance_matrix.npz`（`episodes`, `D`）、`outliers.json`、`run_config.json`；`clusters.json` 在 **默认 none** 下主要记录 **全局 medoid** 与说明；开图时默认写 **`mds2d.png`**、**`dendrogram.png`**、**`tsne2d.png`**（需 **scikit-learn**；与 MDS 对照，强调局部结构）。用 **`--no-plot-tsne`** 跳过 t-SNE；**`--no-plots`** 关闭全部图。 |

**复杂度**：\(O(E^2)\) 次距离计算；大数据务必 **`--time-downsample`**、**`--resample`**，必要时 **`--max-episodes`**。

---

## 4. 推荐用法（让别人能一键复现）

从仓库根目录、**`PYTHONPATH=src`**（与训练一致）：

```bash
PYTHONPATH=src python -m umi_scripts_csw.data_distribution.scripts.trajectory_similarity \
  --dataset-root /path/to/lerobot_dataset_root \
  --parquet-lowdim-only \
  --representation policy20 \
  --time-downsample 8 \
  --resample 64 \
  --out-dir ./reports/traj_sim_example

# 默认会尝试输出 tsne2d.png（需 pip install scikit-learn）；不需要时加 --no-plot-tsne
```

- **解读结果**：优先打开 **`distance_matrix.npz`** 和 **`outliers.json`**；需要簇时再显式加 `--cluster-mode auto` 或 `manual --n-clusters K`。  
- **论文/报告**：写清 **representation、是否 resample、time_downsample、是否 z-score**，否则不同设定下的 `D` **不可比**。

---

## 5. 使用注意（避免误读）

1. **混合多任务的 dataset（如 `*_mix`）**：全局 `D` 会混合不同任务动力学，**大距离不一定等于失败**，可能只是任务不同；若要做「同一种做法内的离群」，应在 **单任务子集** 上单独跑。  
2. **树状图 vs 簇**：dendrogram 展示 **整棵合并树**；**默认不给出离散簇**。若曾用 `auto` 聚类，可能出现 **「一大簇 + 单条」** 被 silhouette 偏好——这是指标问题，不是树「错了」。  
3. **`resample` + 平均 L2** 与 **DTW** 量纲与数值都不同；对比实验请固定同一套参数。  
4. **MDS vs t-SNE**：二者都是对 **`D`** 的 2D 嵌入；**MDS** 更倾向保留全局远近关系；**t-SNE（默认与 MDS 一起出图）** 强调邻域、便于看「团」，平面距离勿直接当作 `D[i,j]`。依赖 **scikit-learn**；缺包或 `--no-plot-tsne` 时跳过；\(E\) 大时较慢。

---

## 6. 文档与代码索引

| 资源 | 用途 |
|------|------|
| [`PLAN_TRAJECTORY_SIMILARITY.md`](PLAN_TRAJECTORY_SIMILARITY.md) | 原始设计、方案 A/B、CLI 草案（实现时可略有出入，以脚本 `--help` 为准）。 |
| [`scripts/trajectory_similarity.py`](scripts/trajectory_similarity.py) | 唯一主入口；行为以源码为准。 |
| [`DATA_DISTRIBUTION_ROADMAP.md`](DATA_DISTRIBUTION_ROADMAP.md) | 与数据分布分析路线图的关系。 |

---

**版本说明**：本文档描述 **以距离矩阵为中心、默认不分簇** 的工作流；若脚本 CLI 变更，请以 `trajectory_similarity.py` 内 argparse 与模块 docstring 为准。
