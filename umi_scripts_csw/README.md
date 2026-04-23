# UMI Scripts 目录说明

> 本目录包含数据处理、转换、评估的各类工具脚本，按功能分类组织。

## 目录结构

```
umi_scripts_csw/
├── README.md                          # 本文件
├── INDEX.md                           # 脚本功能索引
├── data_detection/                    # 数据检测 (ArUco/Foundation Pose)
├── data_cleaning/                     # 数据清理与质量检查
├── data_conversion/                   # 数据格式转换
├── data_evaluation/                   # 数据评估与可视化
├── rosbag_tools/                      # ROSBag处理工具
├── nano_sync/                         # 音视频同步工具 (原有)
└── utils/                             # 通用工具脚本
```

## 各目录说明

### data_detection/ - 数据检测
**用途**: 第三视角视觉检测，提取机械臂位姿

| 脚本 | 功能 |
|------|------|
| add_aruco_pose_to_json.py | ArUco码检测，输出left_in_cam/right_in_cam到third.json |
| visualize_aruco_detection.py | 可视化ArUco检测结果，生成检测框和坐标轴预览图 |

**输入**: third.mp4  
**输出**: third.json (含检测位姿)

---

### data_cleaning/ - 数据清理
**用途**: 数据质量检查、异常筛选、合并清理

| 脚本 | 功能 | 推荐用法 |
|------|------|---------|
| **check_trajectory_anomalies.py** | **轨迹异常检测** (完整分析，支持CHECK-004+~010) | Step 1: 检测异常 |
| **generate_visual_report.py** | **可视化报告生成** (HTML/Markdown) | Step 2: 生成报告 |
| **execute_data_cleaning.py** | **执行数据清理** (删+重编号) | Step 3: 正式清理 |
| clean_data_0422_preview.py | 旧版清理预览 (仅供参考) | - |
| check_consistency.py | 检查数据集内部一致性 (视频分辨率、文件完整性) | - |
| check_dataset_actions.py | 检查数据集动作统计分布 | - |
| compare_batches.py | 对比两个数据批次的差异 | - |
| compare_npz.py | 对比NPZ数据文件 | - |

**快速开始 - 标准清理流程**:
```bash
# Step 1: 检测异常
python data_cleaning/check_trajectory_anomalies.py \
    --data_dir ~/Downloads/handover_umi_0422 \
    --output_dir ./anomaly_reports

# Step 2: 生成可视化报告
python data_cleaning/generate_visual_report.py \
    --json_report ./anomaly_reports/trajectory_anomaly_report.json \
    --format both

# Step 3: 预览清理
python data_cleaning/execute_data_cleaning.py \
    --data_dir ~/Downloads/handover_umi_0422 \
    --anomaly_report ./anomaly_reports/trajectory_anomaly_report.json \
    --severity critical \
    --dry_run

# Step 4: 正式清理
python data_cleaning/execute_data_cleaning.py \
    --data_dir ~/Downloads/handover_umi_0422 \
    --anomaly_report ./anomaly_reports/trajectory_anomaly_report.json \
    --severity critical \
    --output_dir ~/Downloads/handover_umi_0422_cleaned
```

---

### data_conversion/ - 数据转换
**用途**: 转换为LeRobot格式或其他格式

| 脚本 | 功能 |
|------|------|
| convert_mp4_data_to_lerobot_123.py | 原转换脚本 (支持相对旋转state) |
| convert_mp4_data_to_lerobot_123_aruco.py | **ArUco版本** (支持hands_rel_xyz + hands_rel_rot6d 9D state) |
| convert_mp4_data_to_lerobot_downsample.py | 下采样版本 |
| convert_mp4_data_to_lerobot_downsample_13.py | 下采样版本 (13fps) |

**输入**: MP4 + JSON (stage1格式)  
**输出**: LeRobot格式数据集

---

### data_evaluation/ - 数据评估
**用途**: 评估数据质量、可视化轨迹

| 脚本 | 功能 |
|------|------|
| eval_actions.py | 动作序列评估 |
| eval_relative.py | 相对位姿评估 |
| eval_dual_relative.py | 双手相对位姿评估 |
| eval_relative_visualize.py | 相对位姿可视化 |

---

### rosbag_tools/ - ROSBag工具
**用途**: ROSBag录制数据处理

| 脚本 | 功能 |
|------|------|
| convert_rosbag_to_mp4_vis_123.py | 主版本: rosbag转MP4+JSON (3视角) |
| convert_rosbag_to_mp4_vis.py | 早期版本 |
| convert_rosbag_to_mp4_vis_13.py | 13fps版本 |
| convert_ros_data_to_mp4.py | 简化版本 |

**输入**: .bag  
**输出**: MP4 + JSON (stage1格式)

---

### nano_sync/ - 音视频同步
**用途**: 录音棚同步信号处理 (原有目录，保持不变)

| 脚本 | 功能 |
|------|------|
| detect_beep_in_video.py | 检测视频中的beep信号 |
| generate_beep.py | 生成beep音频 |
| keyboard_beep_listener.py | 键盘监听触发beep |
| ... | ... |

---

### utils/ - 通用工具
**用途**: 各类辅助工具

| 脚本 | 功能 |
|------|------|
| json_sort.py / json_sort_123.py / json_sort_13.py | JSON文件排序整理 |
| json_visualize.py | JSON数据可视化 |
| transform_pose.py | 位姿转换工具 |
| camera_test_analyzer.py | 相机测试分析 |
| tri_image_sampler_10hz.py | 三视角图像采样 |
| render_triad_mp4.py | 渲染triad视频 |
| replay_data_fastumi.py | FastUMI数据回放 |
| fastumi_helper.py | FastUMI辅助工具 |
| test.py | 测试脚本 |

---

## 数据处理流程

### 完整流程 (含清理)

```
原始数据 (MP4+JSON)
    ↓ [data_cleaning/check_trajectory_anomalies.py]
异常检测报告 (JSON)
    ↓ [data_cleaning/generate_visual_report.py]
可视化报告 (HTML)
    ↓ (查看报告，确认清理级别)
执行清理 [data_cleaning/execute_data_cleaning.py]
    ↓
清理后数据 (cleaned/)
    ↓ [data_conversion/]
LeRobot格式数据集
```

### 旧流程 (ArUco检测，非必须)

```
rosbag (.bag)
    ↓ [rosbag_tools/]
MP4 + JSON (stage1)
    ↓ [data_detection/] (可选)
MP4 + JSON (含检测数据)
    ↓ [data_conversion/]
LeRobot格式数据集
    ↓ [data_evaluation/]
质量评估报告
```

## 快速开始

### 标准数据清理流程 (推荐，从原始数据到清理后数据)

```bash
# Step 1: 轨迹异常检测 (CHECK-004+ ~ CHECK-010)
python data_cleaning/check_trajectory_anomalies.py \
    --data_dir ~/Downloads/handover_umi_0422 \
    --output_dir ./anomaly_reports

# Step 2: 生成可视化报告 (查看HTML决定清理级别)
python data_cleaning/generate_visual_report.py \
    --json_report ./anomaly_reports/trajectory_anomaly_report.json \
    --output_dir ./anomaly_reports \
    --format both
# 查看: anomaly_reports/trajectory_anomaly_visual.html

# Step 3: 预览清理 (确认要删除的episodes)
python data_cleaning/execute_data_cleaning.py \
    --data_dir ~/Downloads/handover_umi_0422 \
    --anomaly_report ./anomaly_reports/trajectory_anomaly_report.json \
    --severity critical \
    --dry_run

# Step 4: 正式执行清理 (删除异常并重新编号)
python data_cleaning/execute_data_cleaning.py \
    --data_dir ~/Downloads/handover_umi_0422 \
    --anomaly_report ./anomaly_reports/trajectory_anomaly_report.json \
    --severity critical \
    --output_dir ~/Downloads/handover_umi_0422_cleaned

# Step 5: 验证结果
ls ~/Downloads/handover_umi_0422_cleaned/ | grep "_left.json$" | wc -l
# 应显示清理后的episodes数量
cat ~/Downloads/handover_umi_0422_cleaned/cleaning_log.json
# 查看清理日志
```

### 完整数据处理流程 (含ArUco，可选)

```bash
# 1. rosbag转MP4 (如果是rosbag源数据)
python rosbag_tools/convert_rosbag_to_mp4_vis_123.py \
    --bag_dir /path/to/bags \
    --out_dir /path/to/stage1 \
    --start_idx 1

# 2. ArUco检测 (如果需要hands_rel数据)
python data_detection/add_aruco_pose_to_json.py \
    --data_dir /path/to/stage1 \
    --output_dir /path/to/stage1_aruco

# 3-7. 清理流程 (同上)
# ...

# 8. 转换为LeRobot
python data_conversion/convert_mp4_data_to_lerobot_123_aruco.py \
    --stage1_dir /path/to/stage1_aruco \
    --repo my_dataset
```

### 2. 数据质量检查

```bash
# 检查一致性
python data_cleaning/check_consistency.py --data_dir /path/to/data

# 评估数据
python data_evaluation/eval_dual_relative.py --data_dir /path/to/data
```

## 规范文档

- [数据合并与清理规范](../docs/data_merge_clean_spec.md) - 定义数据清理规则
- [清理报告示例](../docs/cleaning_report_0422_preview.json) - 0422数据清理预览报告

## 注意事项

1. **新脚本放置**: 按功能放入对应目录，不要在根目录直接创建
2. **命名规范**: 使用 `{功能}_{描述}_{版本/变体}.py` 格式
3. **版本管理**: 不同变体用 `_123`, `_13` 等后缀区分
4. **文档维护**: 修改本README和INDEX.md记录新脚本
