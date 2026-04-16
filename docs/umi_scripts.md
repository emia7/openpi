# UMI 相关脚本说明（`umi_scripts_qiuyi` / `umi_scripts_csw`）

仓库中有两个以 `umi` 开头的目录，用于 **FastUMI / XV / 多相机 ROS 数据** 到 **LeRobot** 的转换，以及 **质检、评测**。本文按数据流说明各文件职责与推荐用法；安装与 openpi 训练通用步骤仍以 [README.md](../README.md) 为准。

---

## 总览：两套目录分工

| 目录 | 定位 |
|------|------|
| **`umi_scripts_qiuyi/`** | 在训练机/服务器上，从 **已落盘的轻量格式**（单路 `mp4+json`，或 Franka 的 `parquet+mp4+meta`）**构建 LeRobot 数据集**；含随机 episode 子采样等。 |
| **`umi_scripts_csw/`** | **ROS bag → MP4/JSON → LeRobot** 的完整链路，外加 **文件重命名、JSON/轨迹可视化、批次对比、openpi 上简单评测**；部分 **shell 路径与 replay 脚本** 指向其他机器/工程，需自行修改后才能用。 |

---

## `umi_scripts_qiuyi/`

### `convert_umi_data_to_lerobot_sampling.py`

- **输入**：一个或多个目录中的 `episode*.mp4` 及同 stem 的 `.json`。JSON 内需有 `records[].pose`（7D：`x,y,z,qx,qy,qz,qw`），可选 `records[].clamp`；元数据里可有 `fps`。
- **行为**：按 **整数 stride** 将原始帧率降到 `target_fps`（默认 10Hz）；用 **LeRobotDataset** 写入 `HF_LEROBOT_HOME / <repo_id>`。观测为当前步图像与位姿；**`actions` 为下一采样时刻的绝对位姿 + 夹爪**（next-state 监督，与部分 VLA 数据约定一致）。
- **采样**：`--num_episodes` + `--seed` 可在全部 episode 中 **随机子集**；不指定则用全部（顺序经 shuffle）。
- **常用参数**：`--input_dirs`（多个目录）、`--repo_id`、`--robot_type`（默认 `XV`）、`--target_fps`、`--task`（语言任务字符串）。

**使用建议**：Stage1 已是 **单路对齐好的 mp4+json** 时，用本脚本直接进 LeRobot；改 `--target_fps` 前确认与后续 `training/config` 中 fps 一致。

### `convert_franka_data_to_lerobot.py`

- **输入**：目录内 `*.parquet`、同名 `*_meta.json`，以及 meta 中列出的各相机 **`{base}_{cam}.mp4`**。
- **行为**：读取 meta 中的 `cameras`、`fps`、`task_description`、`action_scale`、`episode_failed` 等；按 `scale_action_for_lerobot` 将 7D 动作缩放到 LeRobot 约定（含夹爪映射到 `[0,1]`）。特征中包含 **`observation.state.demo_start_tcp_pose`**（首帧 TCP，四元数形式）。
- **入口**：**absl**（`--input_dirs` 可多值、`--output_dir`、`--repo_id`、`--skip_failed`、`--num_episodes`、`--seed`）。若输出目录下已存在同名 repo，会 **删除后重建**。

### `convert_franka_data_to_lerobot_v1.py`

- 与 `convert_franka_data_to_lerobot.py` 逻辑基本相同，但 **LeRobot schema 不含 `demo_start_tcp_pose`**。当训练配置与 policy **不依赖** demo 起始位姿时用 v1 即可。

**使用建议**：Franka 侧导出为 **parquet + 多路 mp4 + meta** 的「上传包」时二选一；需要 `demo_start_tcp_pose` 对齐 openpi 某条 config 时用 **非 v1**。

---

## `umi_scripts_csw/`

### Stage 1：Rosbag → MP4 + JSON

| 文件 | 说明 |
|------|------|
| `convert_ros_data_to_mp4.py` | 单个 rosbag：按 XV **serial** 对齐图像与 `PoseStampedConfidence`，导出单 episode 的 **mp4 + json**。 |
| `convert_rosbag_to_mp4_vis.py` | 同上，并带 **matplotlib** 可视化（对齐/轨迹等），便于调试。 |
| `convert_rosbag_to_mp4_vis_13.py` | **双视角**（如 XV + head/D435），支持压缩图像与更稳健的时间戳；输出命名习惯含 `_head`、`_left` 等。 |
| `convert_rosbag_to_mp4_vis_123.py` | **三视角**（左/右 XV + 第三路相机）；**topic 与序列号**在文件顶部 **`CONFIG`** 中修改。 |

**批量 shell**（使用前务必改其中的 **`BAG_DIR` / `OUT_DIR` / `SERIALS` / `PY_SCRIPT`**）：

- `batch_stage1.sh` → 调用 `convert_ros_data_to_mp4.py`
- `batch_stage1_vis.sh` → 调用 `convert_rosbag_to_mp4_vis.py`
- `batch_stage1_vis_13.sh` → 调用 `convert_rosbag_to_mp4_vis_13.py`
- `batch_stage1_vis_123.sh`：通用包装，参数传入 bag 目录、输出目录、起始序号、Python 脚本路径

### Stage 2：MP4 + JSON → LeRobot

| 文件 | 说明 |
|------|------|
| `convert_mp4_data_to_lerobot_downsample.py` | **单路** `episode*.mp4` + `episode*.json`，降采样后写入 LeRobot（无多目录随机采样）。 |
| `convert_mp4_data_to_lerobot_downsample_13.py` | **双路**：`episode*_head.mp4`、`episode*_left.mp4` 与对应 json。 |
| `convert_mp4_data_to_lerobot_123.py` | **三路**：`episode_*_left` / `_right` / `_third` 的 mp4 与 json，构建多图像特征的数据集。 |

### 文件整理与可视化、坐标变换

| 文件 | 说明 |
|------|------|
| `json_sort.py` | 对 `episode*.mp4/.json` 等 **两阶段重命名** 为连续编号，避免覆盖；支持 dry-run / `--apply`。 |
| `json_sort_13.py` | 带 **`_head` / `_left` / `_states`** 等后缀的 **成组** 重命名。 |
| `json_sort_123.py` | 处理 `episode_000001_left`、`…_right`、`…_third`、`…_alignment` 等命名规则的排序与改名。 |
| `json_visualize.py` | 绘制单条 json 中的 **pose / clamp** 等时间序列。 |
| `render_triad_mp4.py` | 由 pose 序列渲染 **3D 坐标系 + 轨迹 MP4**。 |
| `transform_pose.py` | 对 json 内每条 `pose` 做 **固定轴系变换**，输出新 json；使用前需与 **policy / 训练坐标约定** 一致。 |

### ROS 录制辅助

| 文件 | 说明 |
|------|------|
| `tri_image_sampler_10hz.py` | **ROS1 节点**：三路相机高频订阅，**定时 10Hz 发布**最新帧，并打印时间戳差统计。需放入 **catkin 包** 的 `scripts/`，并修改 `CONFIG` 中的 topic 与设备序列号。 |

### 与 openpi / 结果对比

| 文件 | 说明 |
|------|------|
| `eval_actions.py` | 使用 **openpi** 的 `config`、`policy_config` 与 **LeRobotDataset**，在指定 episode 上推理并 **绘制动作** 等（JAX）。 |
| `eval_relative_new.py` | 在相对动作 / horizon 设定下评测（如 `--dims`），结果写入目录。 |
| `eval_relative_visualize.py` | 评测并导出 **视频类** 可视化。 |
| `compare_npz.py` | 对比两个 **`.npz`**（如两次导出的 state/actions），曲线与可选热力图。 |
| `compare_batches.py` | 对比两个目录下的 **episode JSON 批次**：位姿范围、步进差分、时间戳、clamp 是否一致等，用于排查「一批能训一批不能训」。 |

### 机侧回放与其它

| 文件 | 说明 |
|------|------|
| `replay_data_fastumi.py` | 设计为 **绝对位姿离线转 delta 后在真机回放**；依赖 **`scripts.pose_util`**、`experiments.get_config` 等，且含 **写死的 parquet 路径**。在 **仅 openpi 仓库** 中 **通常无法直接运行**，仅作参考或需迁回完整 FastUMI 工程。 |
| `replay_data_fastumi.sh` | 调用上述 replay 的包装，同样依赖原工程布局。 |
| `test.py` | 与 README 类似的代码片段（含非法占位符），**不可当作可运行测试**。 |

---

## 推荐流水线（简图）

1. **录包**（可选）`tri_image_sampler_10hz.py` 统一三路 10Hz → `rosbag record`。  
2. **Stage1**：按相机路数选择 `convert_ros_*.py` + 对应 `batch_stage1*.sh`。  
3. **整理命名**：`json_sort.py` / `json_sort_13.py` / `json_sort_123.py`。  
4. **质检**：`json_visualize.py`、`render_triad_mp4.py`、`compare_batches.py`；坐标与训练不一致时再考虑 `transform_pose.py`。  
5. **Stage2**：`convert_mp4_data_to_lerobot_downsample*.py` 或 `convert_mp4_data_to_lerobot_123.py`。  
6. **替代路径**：若 Stage1 已是单路 `episode*.mp4+json` 且需多目录随机采样，可用 **`umi_scripts_qiuyi/convert_umi_data_to_lerobot_sampling.py`** 代替 csw 单路 Stage2。  
7. **Franka 云端包**：直接用 **`convert_franka_data_to_lerobot*.py`**。

---

## 注意事项

- **路径硬编码**：`umi_scripts_csw` 内多处 **`/home/ubuntu/qiuyi/...`**，必须在本地改为你的路径。  
- **语义一致**：`convert_umi_data_to_lerobot_sampling.py` 中 **action 为下一时刻绝对位姿 + 夹爪**，须与 **`src/openpi/training/config.py` 中对应 DataConfig 与 policy** 一致。  
- **破坏性写入**：若干转换脚本会 **删除已存在的输出 repo 目录**；运行前注意备份。  
- **依赖**：ROS 相关脚本需要 **rosbag、cv_bridge、rospy** 等 ROS 环境；LeRobot 脚本需要本仓库 **`uv sync`** 后的 Python 环境。

---

## 相关文档

- [codebase_overview.md](codebase_overview.md)：openpi 整体结构  
- [norm_stats.md](norm_stats.md)：训练前归一化统计  
- [remote_inference.md](remote_inference.md)：远端策略服务
- [../umi_scripts_csw/README.md](../umi_scripts_csw/README.md)：`umi_scripts_csw` 目录内快速入口与分组索引
