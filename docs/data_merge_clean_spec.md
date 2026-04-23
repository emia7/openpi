# 数据合并与清理规范

> 本文档定义了无本体采集数据集的合并、清理与质量检查规范。
> 
> 版本: 1.0  
> 最后更新: 2025-04-22

## 目录

1. [概述](#概述)
2. [数据源管理](#数据源管理)
3. [合并规则](#合并规则)
4. [异常检测规则](#异常检测规则)
5. [清理流程](#清理流程)
6. [命名约定](#命名约定)
7. [质量检查清单](#质量检查清单)
8. [版本历史](#版本历史)

---

## 概述

### 目标
- 确保合并后的数据集质量一致
- 标准化不同批次数据的格式
- 自动化异常检测与清理流程
- 维护可追溯的数据血缘关系

### 适用范围
- XV双采集的任务数据
- 输入格式: MP4 + JSON (stage1格式)
- 输出格式: MP4 + JSON (stage1格式，保持与输入一致)

---

## 数据源管理

### 数据源命名规范

| 字段 | 格式 | 示例 |
|------|------|------|
| 日期 | YYYYMMDD | 20250421 |
| 批次号 | 可选，从1开始 | _b1, _b2 |
| 数据类型 | 原始/清洗后 | _raw, _cleaned |
| 混合标记 | 混合数据集标记 | _mix, _good_mix |

**完整命名示例**:
```
handover_umi_20250421_raw          # 原始数据
handover_umi_20250421_cleaned      # 清洗后
handover_umi_20250421_b2_raw       # 第二批原始数据
handover_umi_0422_0421_mix         # 0422+0421混合
handover_umi_0422_0421_good_mix    # 优质子集混合
```

### 数据源注册表 (Data Source Registry)

所有数据源统一在 `data_source_registry.json` 中维护：

```json
{
  "registry_version": "1.0",
  "last_updated": "2025-04-22T12:00:00",
  "sources": [
    {
      "source_id": "0421",
      "path": "~/Downloads/handover_umi_0422_aruco",
      "collection_date": "2025-04-21",
      "robot_platform": "XV_DUAL",
      "task": "handover",
      "num_episodes": 19,
      "total_frames": 1286,
      "avg_frames_per_episode": 67.7,
      "status": "cleaned",
      "notes": "第一批ArUco检测数据"
    },
    {
      "source_id": "0422",
      "path": "~/Downloads/handover_umi_0422",
      "collection_date": "2025-04-22",
      "robot_platform": "XV_DUAL",
      "task": "handover",
      "num_episodes": 210,
      "total_frames": 12344,
      "avg_frames_per_episode": 58.8,
      "status": "raw",
      "notes": "第二批原始数据"
    }
  ]
}
```

**维护职责**: 每次新增数据源时，必须更新此注册表。

### 数据源元信息 (单数据源)

每个数据源目录可选包含 `source_info.json`，由注册表自动生成：

```json
{
  "source_id": "0422",
  "collection_date": "2025-04-22",
  "robot_platform": "XV_DUAL",
  "task": "handover",
  "num_episodes": 210,
  "total_frames": 12344,
  "camera_config": {
    "left_resolution": [1280, 1280],
    "right_resolution": [1280, 1280],
    "third_resolution": [1280, 720]
  },
  "notes": "采集备注信息"
}
```

---

## 合并规则

### 基础合并 (Simple Merge)

**场景**: 将多个批次数据直接拼接

**规则**:
1. 重新编号: 从1开始连续编号 (episode_000001, episode_000002, ...)
2. 保持原始文件结构
3. 生成合并映射表 `merge_mapping.json`

**映射表格式**:
```json
{
  "merge_timestamp": "2025-04-22T10:00:00",
  "sources": [
    {"source_id": "0421", "num_episodes": 19, "episode_range": [0, 18]},
    {"source_id": "0422", "num_episodes": 210, "episode_range": [19, 228]}
  ],
  "mapping": {
    "episode_000000": {"source": "0421", "original_name": "episode_000001"},
    "episode_000019": {"source": "0422", "original_name": "episode_000001"}
  }
}
```

### 选择性合并 (Selective Merge)

**场景**: 从多个源中筛选优质数据子集合并

**规则**:
1. 先对各源独立执行质量检查
2. 根据评分选择优质episodes
3. 可设置各源采样比例

**评分维度**:
- 帧数充足性 (≥20帧: 满分, <10帧: 0分)
- 动作多样性 (轨迹方差)
- 左右手同步性 (时间戳对齐误差)

### 混合采样 (Stratified Sampling)

**场景**: 确保不同策略/场景的数据均衡

**规则**:
1. 定义分类维度 (如: 起始位置、物体类型)
2. 按类别分层采样
3. 记录采样比例

---

## 异常检测规则

### 1. 文件完整性检查

**规则ID**: CHECK-001  
**严重程度**: 阻塞性 (必须修复)

| 检查项 | 通过标准 | 失败处理 |
|--------|---------|---------|
| 必要文件存在 | left/right/third MP4 + JSON | 标记为损坏，跳过该episode |
| JSON可解析 | 无语法错误 | 尝试修复，否则跳过 |
| 视频可读取 | 能读取首帧 | 使用ffprobe检查，跳过损坏文件 |

**检测代码示例**:
```python
def check_file_completeness(episode_dir, episode_name):
    required_files = [
        f"{episode_name}_left.mp4",
        f"{episode_name}_left.json",
        f"{episode_name}_right.mp4",
        f"{episode_name}_right.json",
        f"{episode_name}_third.mp4",
        # third.json 可选 (ArUco检测后才有)
    ]
    missing = [f for f in required_files if not (episode_dir / f).exists()]
    return len(missing) == 0, missing
```

### 2. 帧数异常检测 (动态阈值)

**规则ID**: CHECK-002  
**严重程度**: 警告性 (建议清理)

**动态阈值计算方法**:
1. 计算数据集平均帧数 `avg_frames`
2. 设定剔除阈值 `threshold = avg_frames * 0.3` (保留至少30%的帧数)
3. 设定警告阈值 `warning_threshold = avg_frames * 0.5`

| 等级 | 判定条件 | 处理建议 |
|------|---------|---------|
| 优秀 | ≥ avg_frames | 保留，高质量数据 |
| 正常 | warning_threshold ~ avg_frames | 保留 |
| 偏短 | threshold ~ warning_threshold | 可选保留，记录警告 |
| 异常 | < threshold | 建议剔除 |
| 损坏 | < 5帧 | 必须剔除 |

**示例**:
- 数据集平均帧数 60帧
- 剔除阈值 = 60 * 0.3 = 18帧
- 警告阈值 = 60 * 0.5 = 30帧
- <18帧建议剔除，18-30帧记录警告，>30帧正常保留

### 3. 分辨率一致性检查

**规则ID**: CHECK-003  
**严重程度**: 警告性

**批量内检查**:
- 同一数据集中所有episode分辨率应一致
- 不一致时需统一到最低或最高分辨率

**跨批次检查**:
- 记录各批次分辨率
- 混合时统一到相同分辨率 (建议统一到640×480或1280×1280)

### 4. 时间同步性检查 (帧数相关)

**规则ID**: CHECK-004  
**严重程度**: 警告性

**检测指标** (基于帧数的相对差异):

| 检查项 | 判定标准 | 说明 |
|--------|---------|------|
| 左右手帧数差 | `abs(left_frames - right_frames) < 0.05 * avg_frames` | 差异<平均帧数的5% |
| 第三视角帧数差 | `abs(third_frames - avg(left,right)) < 0.1 * avg_frames` | 差异<平均帧数的10% |
| 时间戳对齐误差 | 最大误差< 100ms 或 1帧时长 | 取较小值 |

**计算方式**:
```python
left_frames = count_frames(episode_left_mp4)
right_frames = count_frames(episode_right_mp4)
third_frames = count_frames(episode_third_mp4)
avg_frames = (left_frames + right_frames) / 2

# 检查左右手同步
if abs(left_frames - right_frames) > 0.05 * avg_frames:
    mark_warning("左右手帧数差异过大")

# 检查第三视角同步
if abs(third_frames - avg_frames) > 0.1 * avg_frames:
    mark_warning("第三视角帧数差异过大")
```

**示例** (平均60帧):
- 左右手差异应 < 3帧 (60*0.05)
- 第三视角差异应 < 6帧 (60*0.1)

### 5. 静态帧检测

**规则ID**: CHECK-005  
**严重程度**: 警告性

**检测方法**:
- 计算相邻帧光流或像素差异
- 连续5帧以上无变化视为静态
- 静态占比>30%标记为低质量

### 6. 动作范围异常

**规则ID**: CHECK-006  
**严重程度**: 警告性

**检测指标**:
- 单步位移>0.5米: 异常
- 单帧旋转>45度: 异常
- 轨迹总长度<0.1米: 可能为静态演示

---

## 清理流程

### 阶段1: 预处理检查

1. **扫描数据源**
   - 列出所有episodes
   - 生成初始清单 `episodes_inventory.csv`

2. **完整性检查** (CHECK-001)
   - 标记文件缺失的episodes
   - 生成损坏列表 `damaged_episodes.txt`

3. **基础元数据提取**
   - 帧数、分辨率、FPS
   - 保存到 `metadata_summary.json`

### 阶段2: 质量评估

1. **帧数筛选** (CHECK-002)
   - 剔除 <10帧的episodes
   - 标记 10-20帧的episodes

2. **分辨率检查** (CHECK-003)
   - 检测分辨率不一致
   - 决定统一策略

3. **同步性检查** (CHECK-004)
   - 标记帧数严重不一致的episodes

4. **高级质量检查** (CHECK-005, 006)
   - 静态帧检测
   - 动作范围分析
   - 生成质量评分

### 阶段3: 清洗操作

1. **剔除确认**
   - 根据评分剔除低质量episodes
   - 保留剔除记录 `removed_episodes_log.json`

2. **分辨率统一** (如需要)
   - 使用ffmpeg缩放
   - 保持纵横比
   - 生成操作记录

3. **重新编号**
   - 连续编号从0开始
   - 生成编号映射表

### 阶段4: 验证与输出

1. **清理后验证**
   - 再次运行基础检查
   - 确认无损坏文件

2. **生成清理报告**
   - 原始数量、剔除数量、剩余数量
   - 各检查项通过率
   - 建议的报告格式见下方

---

## 命名约定

### Episode编号

- 格式: `episode_{6位数字}`
- 示例: `episode_000001`, `episode_000210`
- 起始: **从1开始** (`episode_000001`)
- 编号必须连续，不允许跳号

### 输出目录命名

```
{task}_{robot}_{date}_{type}

type选项:
- raw: 原始数据
- cleaned: 清洗后
- mix: 简单混合
- good_mix: 优质数据混合
- aruco: 已检测ArUco
- lerobot: LeRobot格式
```

### 元数据文件命名

```
{dataset_name}_metadata.json      # 数据集元数据
{dataset_name}_mapping.json       # 编号映射表
{dataset_name}_quality_report.json # 质量报告
{dataset_name}_cleaning_log.json   # 清理日志
```

---

## 质量检查清单

### 执行检查前准备

- [ ] 确认数据源路径
- [ ] 确认输出路径
- [ ] 备份原始数据 (建议)
- [ ] 定义接受/拒绝阈值

### 基础检查 (必须通过)

- [ ] 所有必要文件存在 (CHECK-001)
- [ ] JSON文件可解析
- [ ] 视频文件可读取
- [ ] 帧数≥10 (建议≥20) (CHECK-002)

### 质量检查 (推荐执行)

- [ ] 分辨率一致性检查 (CHECK-003)
- [ ] 时间同步性检查 (CHECK-004)
- [ ] 静态帧检测 (CHECK-005)
- [ ] 动作范围检查 (CHECK-006)

### 清理后验证

- [ ] 清理后文件完整性
- [ ] 编号连续性检查
- [ ] 元数据文件生成
- [ ] 清理报告生成

---

## 清理报告模板

```json
{
  "report_version": "1.0",
  "generated_at": "2025-04-22T12:00:00",
  "source": {
    "path": "/path/to/source",
    "num_episodes_original": 210
  },
  "cleaning_summary": {
    "num_removed": 5,
    "num_retained": 205,
    "removal_reasons": {
      "incomplete_files": 1,
      "too_few_frames": 3,
      "resolution_mismatch": 1
    }
  },
  "quality_metrics": {
    "avg_frames_per_episode": 58.5,
    "min_frames": 22,
    "max_frames": 92,
    "resolution_consistency": "100%",
    "avg_quality_score": 8.5
  },
  "output": {
    "path": "/path/to/output",
    "num_episodes": 205,
    "naming_scheme": "episode_{6位数字}",
    "start_index": 0
  }
}
```

---

## 版本历史

| 版本 | 日期 | 修改内容 | 作者 |
|------|------|---------|------|
| 1.0 | 2025-04-22 | 初始版本，定义基础规则和检查项 | csw |

---

## 待补充规则 (TODO)

- [ ] ArUco检测质量评估标准
- [ ] 轨迹相似度计算方法
- [ ] 多批次混合时的norm stats处理
- [ ] 异常episode的人工审核流程
- [ ] 数据血缘追踪详细规范

---

**文档维护指南**:
1. 每次发现新的异常类型，添加新的CHECK规则
2. 规则编号按顺序递增
3. 更新版本历史表
4. 修改后通知团队成员
