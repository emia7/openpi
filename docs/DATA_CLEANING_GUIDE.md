# 数据筛选操作指南

> 本文档提供完整的数据筛选操作步骤，可直接复制命令执行。

## 环境准备

### 1. 克隆代码仓库

```bash
git clone https://github.com/emia7/openpi.git
cd openpi
git checkout dev_csw
```

### 2. 安装依赖

```bash
pip install numpy imageio tqdm
```

## 标准筛选流程

### 第一步：检测数据质量

```bash
cd ~/openpi

python umi_scripts_csw/data_cleaning/check_trajectory_anomalies.py \
    --data_dir /path/to/your/dataset \
    --output_dir ./anomaly_reports
```

**参数说明**:
- `--data_dir`: 你的数据目录路径 (包含 MP4 + JSON)
- `--output_dir`: 检测报告输出目录

**输出文件**:
- `trajectory_anomaly_report.json` - 完整检测报告

### 第二步：生成可视化报告

```bash
python umi_scripts_csw/data_cleaning/generate_visual_report.py \
    --json_report ./anomaly_reports/trajectory_anomaly_report.json \
    --output_dir ./anomaly_reports \
    --format both
```

**查看报告**:
```bash
# 在浏览器中打开 HTML 报告
open ./anomaly_reports/trajectory_anomaly_visual.html
```

**报告内容**:
- 总episodes数量
- 各严重程度统计 (CRITICAL/WARNING/MINOR/OK)
- 具体异常列表
- 处理建议

### 第三步：执行筛选

根据报告结果选择筛选策略：

#### 策略A - 保守筛选 (只保留OK)
```bash
python umi_scripts_csw/data_cleaning/execute_data_cleaning.py \
    --data_dir /path/to/your/dataset \
    --anomaly_report ./anomaly_reports/trajectory_anomaly_report.json \
    --severity critical \
    --dry_run
```

#### 策略B - 标准筛选 (保留OK+MINOR)
```bash
python umi_scripts_csw/data_cleaning/execute_data_cleaning.py \
    --data_dir /path/to/your/dataset \
    --anomaly_report ./anomaly_reports/trajectory_anomaly_report.json \
    --severity warning \
    --dry_run
```

**预览模式** (`--dry_run`): 不实际删除，只显示将要删除的episodes

### 第四步：正式执行

确认预览结果无误后，去掉 `--dry_run` 正式执行：

```bash
python umi_scripts_csw/data_cleaning/execute_data_cleaning.py \
    --data_dir /path/to/your/dataset \
    --anomaly_report ./anomaly_reports/trajectory_anomaly_report.json \
    --severity critical \
    --output_dir /path/to/your/dataset_cleaned
```

**输入 `yes` 确认执行**

### 第五步：验证结果

```bash
# 查看清理后的episodes数量
ls /path/to/your/dataset_cleaned/ | grep "_left.json$" | wc -l

# 查看清理日志
cat /path/to/your/dataset_cleaned/cleaning_log.json
```

## 完整示例

### 示例1：筛选高质量数据

```bash
# 1. 检测
cd ~/openpi
python umi_scripts_csw/data_cleaning/check_trajectory_anomalies.py \
    --data_dir ~/Downloads/handover_umi_0422 \
    --output_dir ./reports_0422

# 2. 生成报告
python umi_scripts_csw/data_cleaning/generate_visual_report.py \
    --json_report ./reports_0422/trajectory_anomaly_report.json \
    --format both

# 3. 预览 (只保留OK的)
python umi_scripts_csw/data_cleaning/execute_data_cleaning.py \
    --data_dir ~/Downloads/handover_umi_0422 \
    --anomaly_report ./reports_0422/trajectory_anomaly_report.json \
    --severity critical \
    --dry_run

# 4. 正式执行
python umi_scripts_csw/data_cleaning/execute_data_cleaning.py \
    --data_dir ~/Downloads/handover_umi_0422 \
    --anomaly_report ./reports_0422/trajectory_anomaly_report.json \
    --severity critical \
    --output_dir ~/Downloads/handover_umi_0424

# 5. 验证
ls ~/Downloads/handover_umi_0424/ | grep "_left.json$" | wc -l
```

### 示例2：合并多个数据集

```bash
# 合并 handover_umi_0424 和 handover_umi_0422_good_mix
cd ~/openpi

python umi_scripts_csw/data_cleaning/merge_datasets.py \
    --sources ~/Downloads/handover_umi_0424 ~/Downloads/handover_umi_0422_good_mix \
    --output ~/Downloads/handover_umi_0424_mix \
    --dataset_name "handover_umi_0424_mix"

# 验证
ls ~/Downloads/handover_umi_0424_mix/ | grep "_left.json$" | wc -l
cat ~/Downloads/handover_umi_0424_mix/merge_log.json
```

## 筛选策略选择

| 策略 | 保留等级 | 适用场景 | 预期保留率 |
|------|----------|----------|------------|
| **保守** | OK only | 追求最高质量 | ~60% |
| **标准** | OK + MINOR | 平衡质量与数量 | ~80% |
| **宽松** | OK + MINOR + WARNING | 最大化数据量 | ~95% |

### 严重程度说明

```
🔴 CRITICAL (必须剔除)
  - 文件缺失
  - 单手<5帧
  - 静态轨迹

🟡 WARNING (建议剔除)
  - 末端跳变
  - 帧数差异>10帧
  - 动作范围异常

🟢 MINOR (可保留)
  - 轻微同步问题
  - Clamp变化少
```

## 常见问题

### Q1: 如何只筛选特定严重程度的异常？

```bash
# 只剔除 CRITICAL
--severity critical

# 剔除 CRITICAL + WARNING
--severity warning

# 剔除所有非OK (CRITICAL + WARNING + MINOR)
--severity all
```

### Q2: 如何处理分辨率不一致？

```bash
# 统一分辨率 (将480x640提升到720x1280)
python umi_scripts_csw/data_cleaning/unify_resolution.py \
    --data_dir ~/Downloads/handover_umi_0423_mix \
    --output_dir ~/Downloads/handover_umi_0423_mix_unified \
    --target_resolution 720 1280 \
    --views third
```

### Q3: 如何查看某个episode的具体问题？

```bash
# 查看检测报告中的详细信息
cat ./anomaly_reports/trajectory_anomaly_report.json | python3 -m json.tool | grep -A 5 "episode_000007"
```

## 输出文件说明

### 检测报告 (`trajectory_anomaly_report.json`)

```json
{
  "summary": {
    "total_episodes": 206,
    "statistics": {
      "critical": 4,    // 严重异常数
      "warning": 77,    // 警告异常数
      "minor": 48,      // 轻微异常数
      "ok": 77          // 正常数
    }
  },
  "classified_reports": {
    "critical": [...],  // 严重异常列表
    "warning": [...],   // 警告异常列表
    "minor": [...],     // 轻微异常列表
    "ok": [...]         // 正常列表
  }
}
```

### 清理日志 (`cleaning_log.json`)

```json
{
  "statistics": {
    "total_original": 206,
    "removed": 4,
    "retained": 202
  },
  "episodes_removed": ["episode_000007", ...],
  "renumber_mapping": {
    "episode_000001": "episode_000001",
    "episode_000008": "episode_000007"
  }
}
```

## 完整流程图

```
原始数据
    ↓
Step 1: 检测质量
  python check_trajectory_anomalies.py
    ↓
Step 2: 生成报告
  python generate_visual_report.py
    ↓ (查看HTML报告)
Step 3: 预览筛选
  python execute_data_cleaning.py --dry_run
    ↓ (确认删除列表)
Step 4: 正式执行
  python execute_data_cleaning.py
    ↓
清理后数据 (cleaned/)
    ↓ (可选)
Step 5: 合并数据集
  python merge_datasets.py
    ↓
最终数据集
```

## 参考文档

- [数据合并与清理规范](./data_merge_clean_spec.md) - 详细的规则定义
- `umi_scripts_csw/README.md` - 脚本说明
- `umi_scripts_csw/INDEX.md` - 功能索引