# UMI Scripts 功能索引

> 详细记录每个脚本的功能、输入输出、依赖关系

## 索引表

### 核心工作流脚本 (我创建的)

| 脚本路径 | 核心功能 | 输入 | 输出 | 依赖 |
|---------|---------|------|------|------|
| data_detection/add_aruco_pose_to_json.py | ArUco检测 | third.mp4 | third.json (检测位姿) | opencv, numpy |
| data_detection/visualize_aruco_detection.py | 检测可视化 | third.mp4 + third.json | preview图片 | opencv, imageio |
| data_cleaning/clean_data_0422_preview.py | 清理预览/报告 | MP4+JSON目录 | cleaning_report.json | numpy, imageio |
| data_conversion/convert_mp4_data_to_lerobot_123_aruco.py | **主转换脚本** (ArUco版) | MP4+JSON (stage1) | LeRobot数据集 | lerobot, imageio |

---

### ArUco检测流程详细说明

**脚本**: `data_detection/add_aruco_pose_to_json.py`

```bash
# 使用示例
python data_detection/add_aruco_pose_to_json.py \
    --data_dir ~/Downloads/handover_umi_0422 \
    --output_dir ~/Downloads/handover_umi_0422_aruco
```

**功能**:
- 读取third.mp4
- 检测ArUco码 (ID=1左手, ID=2右手)
- 支持多帧独立检测 (左右手可不同帧检测到)
- 输出位姿到third.json

**输出JSON格式**:
```json
{
  "left_in_cam": {
    "translation": [x, y, z],
    "quaternion": [qx, qy, qz, qw],
    "rotation_matrix": [...]
  },
  "right_in_cam": {...},
  "metadata": {
    "detection_frame_left": 77,
    "detection_frame_right": 28,
    "marker_size_m": 0.02,
    "camera_intrinsics": {...}
  }
}
```

---

### 数据转换流程详细说明

**脚本**: `data_conversion/convert_mp4_data_to_lerobot_123_aruco.py`

**与原版区别**:
| 特性 | 原版 (123.py) | ArUco版 (123_aruco.py) |
|------|--------------|----------------------|
| State维度 | 12D (相对旋转) | **9D** (hands_rel_xyz + hands_rel_rot6d) |
| 检测假设 | 第0帧同时检测 | 支持独立多帧检测 |
| 坐标系 | SLAM本地坐标系 | ArUco世界坐标系 |
| 适用场景 | 传统双臂 | **无本体双臂** |

**使用示例**:
```bash
python data_conversion/convert_mp4_data_to_lerobot_123_aruco.py \
    --stage1_dir ~/Downloads/handover_umi_0422_aruco \
    --repo handover_umi_0422_aruco_v2 \
    --robot_type XV_DUAL \
    --fps 10
```

---

### 清理流程详细说明

**脚本**: `data_cleaning/clean_data_0422_preview.py`

**检查规则 (按data_merge_clean_spec.md v1.0)**:

| CHECK规则 | 实现功能 | 阈值计算 |
|----------|---------|---------|
| CHECK-001 | 文件完整性 | 必要文件存在性检查 |
| CHECK-002 | 帧数异常 | `threshold = avg_frames * 0.3` |
| CHECK-003 | 分辨率一致性 | 批量分辨率统计 |
| CHECK-004 | 时间同步性 | 差异 < 5%平均帧数 |

**输出报告**:
```json
{
  "summary": {
    "total_episodes_scanned": 210,
    "episodes_passed_basic_check": 210,
    "episodes_with_issues": 0
  },
  "cleaning_preview": {
    "episodes_to_remove": ["episode_000007"],
    "episodes_to_warn": [...],
    "final_episode_count": 209
  }
}
```

---

## 脚本依赖关系图

```
rosbag_tools/
├── convert_rosbag_to_mp4_vis_123.py
│   └── 输出: stage1格式 (MP4+JSON)
│
data_detection/
├── add_aruco_pose_to_json.py
│   ├── 输入: stage1 (需third.mp4)
│   └── 输出: stage1_aruco (含third.json检测信息)
│
data_cleaning/
├── clean_data_0422_preview.py
│   ├── 输入: stage1或stage1_aruco
│   └── 输出: cleaning_report.json
│
data_conversion/
├── convert_mp4_data_to_lerobot_123_aruco.py
│   ├── 输入: stage1_aruco (必须有ArUco检测数据)
│   └── 输出: LeRobot格式
│
data_evaluation/
├── eval_dual_relative.py
│   └── 输入: LeRobot格式或stage1
```

---

## 版本对照表

| 功能 | 标准版 | 变体版本 | 说明 |
|------|--------|---------|------|
| rosbag转换 | _123.py | _13.py | 13fps版本 |
| 数据转换 | _123.py | _aruco.py | ArUco 9D state版本 |
| JSON排序 | json_sort.py | _123.py, _13.py | 不同episode编号格式 |

---

## 状态跟踪

### 我的脚本状态

| 脚本 | 状态 | 测试情况 | 备注 |
|------|------|---------|------|
| add_aruco_pose_to_json.py | ✅ 稳定 | 已测试 | ArUco检测核心 |
| visualize_aruco_detection.py | ✅ 稳定 | 已测试 | 可视化验证 |
| convert_mp4_data_to_lerobot_123_aruco.py | ✅ 稳定 | 已测试 | 转换核心 |
| clean_data_0422_preview.py | ✅ 稳定 | 已测试0422 | 清理预览 |

### 待优化/补充

- [ ] 正式清理脚本 (带实际删除和重新编号)
- [ ] 多数据源合并脚本
- [ ] 批量ArUco检测脚本
- [ ] 数据血缘追踪工具

---

## 使用频率统计

高频使用 (每次新数据必用):
1. `convert_rosbag_to_mp4_vis_123.py` - rosbag转换
2. `add_aruco_pose_to_json.py` - ArUco检测
3. `clean_data_0422_preview.py` - 质量检查
4. `convert_mp4_data_to_lerobot_123_aruco.py` - 最终转换

中频使用 (调试/分析):
1. `visualize_aruco_detection.py` - 检测验证
2. `check_consistency.py` - 一致性检查
3. `eval_dual_relative.py` - 数据评估

低频使用 (特定场景):
1. `compare_batches.py` - 批次对比
2. `compare_npz.py` - NPZ对比
3. `json_sort*.py` - 排序整理

---

*最后更新: 2025-04-23*
