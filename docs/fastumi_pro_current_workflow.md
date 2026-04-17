# FastUMI Pro 安装与使用（完整流程手册）

本文是 **FastUMI Pro + XV + openpi** 的端到端操作手册，覆盖从硬件环境、采集、数据转换、训练、离线评估到部署与数据分析的完整链路。  
数据转换与评测部分以本仓库 `umi_scripts_csw/` 的 **统一入口脚本**为准（`dev-csw-reorganize` 分支）。

---

## 0. 你需要先搞清楚的三件事

1. **你训练用的 policy / TrainConfig 期望什么输入键**（单路 `wrist_view` 还是 `image`，双臂 `left_view/right_view/third_view` 等）。  
2. **你的数据在磁盘上的组织形式**（rosbag 原始包、Stage1 输出的 `mp4+json`、Stage2 输出的 LeRobot repo）。  
3. **控制语义**：很多 XV/UMI 训练配置使用 **relative / delta action**；实机侧必须与策略输出语义一致，否则“能 offline 拟合但真机不对”。

---

## 1. 推荐总流程（从采集到上线）

1. **ROS + XV SDK +（可选）RealSense D435** 环境就绪。  
2. **采集**：`rosbag record` 录关键话题（建议图像走 10Hz 话题以降低带宽）。  
3. **Stage1**：`rosbag -> mp4 + json`（`stage1_convert.py --views 1|2|3`）。  
4. **整理与质检**：`json_sort.py` + `json_visualize.py` / `render_triad_mp4.py` / `compare_batches.py`。  
5. **传输**：把 Stage1 输出打包 `scp/rsync` 到训练机（路径自定）。  
6. **Stage2**：`mp4+json -> LeRobot`（`stage2_convert.py --views 1|2|3`）。  
7. **对齐训练配置**：在 `src/openpi/training/config.py` 选择/新增对应 `TrainConfig`，并确认 `repo_id`、fps、键名一致。  
8. **norm**：`scripts/compute_norm_stats.py`（必要时使用变体脚本，见第 10 节）。  
9. **训练**：`scripts/train.py`。  
10. **离线评估**：`eval_relative.py --mode single|dual|transform`。  
11. **部署**：`serve_policy.py` / `serve_policy_dual.py` + RealRL 侧 `eval_policy.sh`。  
12. **持续数据分析**：`compare_npz.py`、`compute_norm_stats_w_image.py` 等。

---

## 2. 基础环境

### 2.1 操作系统与 ROS

- Ubuntu 20.04  
- ROS Noetic  

### 2.2 代码仓库

- openpi（本仓库）：`https://github.com/emia7/openpi`

### 2.3 OpenPI Python 环境（训练机）

在 openpi 根目录：

```bash
cd /path/to/openpi
uv sync
```

---

## 3. 硬件 SDK 与 ROS 启动（采集机）

### 3.1 安装 FastUMI 硬件 SDK（示例流程）

> 说明：不同机器上 deb 包路径可能不同，以你本机 `FastUMI_Hardware_SDK` 目录为准。

```bash
git clone https://github.com/FastUMIData/FastUMI_Hardware_SDK.git
cd FastUMI_Hardware_SDK/xv/scripts/
sudo -E bash install-ros1.sh ../sdk/<YOUR_XVSDK_DEB>.deb
```

安装完成后通常会在 `~/catkin_ws` 生成/编译工作空间。

### 3.2 启动 XV SDK

```bash
cd ~/catkin_ws/
roslaunch xv_sdk xv_sdk.launch
```

### 3.3 USB 带宽（强烈建议）

若出现相机掉帧、录制带宽不足等问题，执行 SDK 提供的 USB 多路支持脚本（脚本会提示重启终端/机器，按提示操作）：

```bash
sudo -E bash multi-support.sh
```

### 3.4（可选）头部 D435i / RealSense

若使用 D435i，需要安装 ROS 驱动包（示例）：

```bash
sudo apt-get install ros-noetic-realsense2-camera-*
```

启动相机（示例）：

```bash
roslaunch realsense2_camera rs_camera.launch
```

### 3.5（可选）从 gitee 更新 `xv_sdk`（示例）

```bash
cd ~/catkin_ws/src
git clone https://gitee.com/nics-robot/xv_sdk
cd ~/catkin_ws
catkin_make -DXVSDK_INCLUDE_DIRS="/usr/include/xvsdk" -DXVSDK_LIBRARIES="/usr/lib/libxvsdk.so"
```

---

## 4. Monitor 工具（采集机）

用于查看传感器参数、图像、以及做采集前的 sanity check。

```bash
conda create -n fastumi python=3.8.5
conda activate fastumi
git clone https://github.com/FastUMIData/FastUMI_Monitor_Tool.git
cd FastUMI_Monitor_Tool
pip install -r requirements.txt

# 先启动 xv_sdk.launch，再启动 monitor
roslaunch xv_sdk xv_sdk.launch
chmod +x fastumi_monitor_menu.sh
bash fastumi_monitor_menu.sh
```

---

## 5. 典型频率与常见问题（采集机）

经验上：

- 主鱼眼相机常见 **60Hz**（取决于 USB/主机负载）  
- 多路相机同时录制时更容易掉帧  
- 若怀疑 USB 口问题：优先换 **USB3.0 口**、换线、换工控机

当录制带宽成为瓶颈时，推荐走 **10Hz 图像话题**（见第 6.4 节 `tri_image_sampler_10hz.py`）。

---

## 6. 采集：rosbag

### 6.1 单臂：建议录制话题集合

你需要根据设备序列号替换 `SERIAL`（示例字段来自历史工程记录）：

- `/tf`、`/tf_static`  
- `/xv_sdk/SERIAL/clamp/Data`  
- `/xv_sdk/SERIAL/color_camera/image`（或压缩图话题，按你实际配置）  
- `/xv_sdk/SERIAL/slam/pose`  
- （可选）点云、rgbd、markers 等调试话题

录制示例：

```bash
mkdir -p ~/data/bags
cd ~/data
rosbag record -O bags/ep_$(date +%Y%m%d_%H%M%S).bag \
  /xv_sdk/SERIAL/clamp/Data \
  /xv_sdk/SERIAL/color_camera/image \
  /xv_sdk/SERIAL/slam/pose
```

采集完成后建议写一个 `README`：记录 **序列号、任务名、episode 数、时间**。

### 6.2 双臂 + 第三视角（10Hz 录制，推荐）

典型做法是先把三路图像降到 10Hz，再录包（降低写盘压力）。

#### 6.2.1 启动采集链路

Terminal 1：

```bash
cd ~/catkin_ws/
roslaunch xv_sdk xv_sdk.launch
```

Terminal 2（monitor，按你团队流程打开对应菜单项）：

```bash
conda activate fastumi
cd ./FastUMI_Monitor_Tool/
bash fastumi_monitor_menu.sh
```

Terminal 3（D435）：

```bash
roslaunch realsense2_camera rs_camera.launch
```

#### 6.2.2 使用 `tri_image_sampler_10hz.py` 生成 `_10hz` 图像话题

脚本位置：`umi_scripts_csw/tri_image_sampler_10hz.py`

你需要：

1. 把脚本放入某个 catkin 包的 `scripts/` 并 `chmod +x`  
2. 修改脚本内 `CONFIG`：对齐你机器上的 image topic  
3. `rosrun <your_pkg> tri_image_sampler_10hz.py`

然后 `rostopic hz` 检查 `_10hz` 话题频率。

#### 6.2.3 录包（示例模板）

把左右手 `SERIAL_L/SERIAL_R` 与第三视角压缩图 topic 换成你机器上的真实名字：

```bash
mkdir -p ~/data/bags
cd ~/data
rosbag record -O bags/all_10hz_$(date +%Y%m%d_%H%M%S).bag \
  /xv_sdk/SERIAL_L/color_camera/image_10hz \
  /xv_sdk/SERIAL_R/color_camera/image_10hz \
  /camera/color/image_raw/compressed_10hz \
  /xv_sdk/SERIAL_L/slam/pose \
  /xv_sdk/SERIAL_L/clamp/Data \
  /xv_sdk/SERIAL_R/slam/pose \
  /xv_sdk/SERIAL_R/clamp/Data \
  --lz4 --buffsize=2048 --chunksize=512
```

### 6.3 双臂采集（简化 bringup）

仓库提供：`umi_scripts_csw/start_bringup.sh`（通常需要在你们的采集机上按实际路径调整）。

### 6.4（可选）Web 采集平台路线

历史流程里存在 **FastUMI_Data_Platform_Web** 的双臂采集方式（`deploy.sh` + Chrome 页面）。  
如果你团队仍在使用该平台：采集完成后通常仍建议 **落盘为 rosbag 或统一转 Stage1**，再走本文 Stage1/Stage2 主线，避免“平台私有格式”与训练侧键名漂移。

### 6.5（可选）采集热键辅助

`umi_scripts_csw/fastumi_helper.py`：桌面热键触发开始/停止采集（依赖 `keyboard` 库）。是否使用取决于你的采集流程。

---

## 7. Stage1：rosbag -> mp4 + json（统一入口）

统一脚本：`umi_scripts_csw/stage1_convert.py`

### 7.1 运行环境注意

Stage1 需要能 `import rosbag`（以及 ROS 相关依赖）。实践上常见做法是：

- 在 **配置好 ROS 的 Python 环境**中运行（不一定是 `uv run`），只要 `python` 能找到 `rosbag` 即可。

### 7.2 单臂（1 视角）

```bash
python3 umi_scripts_csw/stage1_convert.py \
  --views 1 \
  --bag /path/to/one.bag \
  --serial SERIAL \
  --out_dir /path/to/stage1_out \
  --data_idx 1
```

可视化（状态对齐/检查）：

```bash
python3 umi_scripts_csw/stage1_convert.py \
  --views 1 --mode vis \
  --bag /path/to/one.bag \
  --serial SERIAL \
  --out_dir /path/to/stage1_out \
  --data_idx 00001
```

### 7.3 双臂（2 视角）

```bash
python3 umi_scripts_csw/stage1_convert.py \
  --views 2 \
  --bag /path/to/one.bag \
  --serial SERIAL_LEFT,SERIAL_RIGHT \
  --out_dir /path/to/stage1_out \
  --data_idx 1
```

批处理 + 并行：

```bash
python3 umi_scripts_csw/stage1_convert.py \
  --views 2 \
  --bag_dir /path/to/bags \
  --out_dir /path/to/stage1_out \
  --start_idx 0 \
  --serials SERIAL_LEFT,SERIAL_RIGHT \
  --jobs 4 \
  --skip_existing
```

### 7.4 三视角（2 腕部 + 1 第三视角）

```bash
python3 umi_scripts_csw/stage1_convert.py \
  --views 3 \
  --bag_dir /path/to/bags \
  --out_dir /path/to/stage1_out \
  --start_idx 0
```

### 7.5 单臂 + 固定第三视角（D435）采集与转换

采集时额外录第三视角压缩图（示例）：

```bash
rosbag record -O bags/with_third_$(date +%Y%m%d_%H%M%S).bag \
  /camera/color/image_raw/compressed \
  /xv_sdk/SERIAL/clamp/Data \
  /xv_sdk/SERIAL/color_camera/image \
  /xv_sdk/SERIAL/slam/pose
```

Stage1 仍用 `--views 3`（前提是 bag 内话题与 `stage1_core.py` 的三视角配置一致；如不一致需要先对齐 `stage1_core.py` 或你们的录制 topic 命名）。

---

## 8. Stage1 后：整理、质检、可视化

### 8.1 episode 命名整理（统一）

统一脚本：`umi_scripts_csw/json_sort.py`

预览：

```bash
python3 umi_scripts_csw/json_sort.py --dir /path/to/stage1_out
```

执行：

```bash
python3 umi_scripts_csw/json_sort.py --dir /path/to/stage1_out --apply
```

### 8.2 曲线质检

```bash
python3 umi_scripts_csw/json_visualize.py --help
```

### 8.3 三维轨迹视频

```bash
python3 umi_scripts_csw/render_triad_mp4.py --json /path/to/episode00001.json --out /path/to/out.mp4
```

### 8.4 两批数据分布对比（强烈推荐）

```bash
python3 umi_scripts_csw/compare_batches.py --dir_a /path/to/batch_a --dir_b /path/to/batch_b
```

### 8.5（可选）数据集动作检查

```bash
python3 umi_scripts_csw/check_dataset_actions.py --help
```

### 8.6（可选）坐标变换（谨慎）

`umi_scripts_csw/transform_pose.py`：对 json 内 `pose` 做固定轴系变换。  
**只有在明确知道训练/policy 坐标约定时才使用**，否则容易把数据“改对坐标但改错语义”。

---

## 9. 传输到训练机

把 Stage1 输出目录打包：

```bash
cd /path/to
zip -r stage1_out.zip stage1_out/
```

传输示例：

```bash
scp stage1_out.zip user@train_host:/path/to/umi_data_raw/
```

在训练机解压：

```bash
unzip stage1_out.zip -d /path/to/umi_data_raw/
```

---

## 10. Stage2：mp4 + json -> LeRobot（统一入口）

统一脚本：`umi_scripts_csw/stage2_convert.py`

### 10.1 单臂（views=1）

```bash
cd /path/to/openpi
uv run python umi_scripts_csw/stage2_convert.py \
  --views 1 \
  --stage1_dir /path/to/stage1_out \
  --repo /path/to/hf_home/your_lerobot_repo \
  --target_fps 10
```

### 10.2 双臂（views=2）

```bash
uv run python umi_scripts_csw/stage2_convert.py \
  --views 2 \
  --stage1_dir /path/to/stage1_out \
  --repo /path/to/hf_home/your_lerobot_repo \
  --target_fps 10
```

### 10.3 三视角（views=3）

```bash
uv run python umi_scripts_csw/stage2_convert.py \
  --views 3 \
  --stage1_dir /path/to/stage1_out \
  --repo /path/to/hf_home/your_lerobot_repo \
  --fps 10
```

### 10.4 结果检查

Stage2 完成后应检查：

- `repo` 目录下 `meta/`、`data/`、`videos/` 是否齐全  
- `info.json` 里的 `fps` 是否与训练配置一致  
- feature keys 是否与 policy 输入一致（这是最常见的离线/在线不一致来源）

---

## 11. 训练：openpi（JAX）

### 11.1 选择 TrainConfig / policy

训练配置集中在：

- `src/openpi/training/config.py`

常见示例（按任务替换）：

- 单臂 XV：`pi05_xv_finetune`  
- 双臂 XV：`pi05_xv_dual_finetune`  
- 第三视角/混合策略：以你们实际新增 config 为准（同文件检索关键词 `xv13`、`umi`、`hybrid`）

### 11.2 对齐 policy 输入输出

策略实现目录：

- `src/openpi/policies/`

你需要保证：

- LeRobot 数据集的 observation keys  
- policy 的 `*_policy.py` 输入 transform  
- TrainConfig 的 DataConfig  

三者一致。

### 11.3 checkpoint 保存位置

不要把大量 checkpoint 写到 home 配额受限的位置。  
在对应 `TrainConfig` 中修改 `checkpoint_base_dir`（具体字段名以 config 为准）。

### 11.4 计算 norm 统计

默认：

```bash
uv run scripts/compute_norm_stats.py --config-name YOUR_CONFIG
```

变体（仓库内存在，按需要选用）：

- `scripts/compute_norm_stats_skip_action7.py`  
- `scripts/compute_norm_stats_skip_action10.py`  
- `scripts/compute_norm_stats_w_image.py`（更偏分析与可视化输出）

> 注意：历史经验里 numpy / datasets 版本会影响统计与加载一致性；若出现奇怪统计，优先锁定你们验证过的版本组合。

### 11.5 开训（tmux 强烈建议）

```bash
tmux new -s YOUR_EXP
tmux attach -t YOUR_EXP
```

训练示例：

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py YOUR_CONFIG --exp-name=YOUR_EXP --overwrite
```

训练结束后，把 `norm_stats.json` 放到 checkpoint assets 目录（与你们历史流程一致）。

---

## 12. 离线评估（统一入口）

统一脚本：`umi_scripts_csw/eval_relative.py`

它支持三类模式：

- `--mode single`：单臂 relative 评估（滑动窗口构造 GT）  
- `--mode dual`：双臂 relative 评估  
- `--mode transform`：导出 transform 后的视频与 `npz`（用于检查训练输入对齐）

### 12.1 checkpoint 路径

默认 `--ckpt-base` 为：

- `/mnt/public1/chenshuaiwen/checkpoints`

若你的 checkpoint 不在该目录，请显式传入：

```bash
--ckpt-base /path/to/checkpoints_root
```

checkpoint 目录布局约定为：

`<ckpt-base>/<config>/<exp-name>/<step>/`

### 12.2 单臂 relative 评估

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run python umi_scripts_csw/eval_relative.py \
  --mode single \
  --config pi05_xv_finetune \
  --exp-name YOUR_EXP \
  --step 20000 \
  --ckpt-base /path/to/checkpoints_root \
  --repo /path/to/lerobot_repo_root \
  --episode 0 \
  --compare-steps "0,1,2,3" \
  --dims 11 \
  --use-time-axis \
  --out-dir eval_results/relative/single_ep0
```

### 12.3 双臂 relative 评估

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run python umi_scripts_csw/eval_relative.py \
  --mode dual \
  --config pi05_xv_dual_finetune \
  --exp-name YOUR_EXP \
  --step 20000 \
  --ckpt-base /path/to/checkpoints_root \
  --repo /path/to/lerobot_repo_root \
  --episode 0 \
  --compare-steps "0,1,2,3" \
  --dims 20 \
  --use-time-axis \
  --out-dir eval_results/relative/dual_ep0
```

`--compare-steps all` 会对比 horizon 内所有步（注意输出图数量）。

### 12.4 transform 导出（训练输入对齐检查）

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run python umi_scripts_csw/eval_relative.py \
  --mode transform \
  --config pi05_xv_finetune \
  --exp-name YOUR_EXP \
  --step 20000 \
  --ckpt-base /path/to/checkpoints_root \
  --repo /path/to/lerobot_repo_root \
  --episode 0 \
  --out-dir eval_results/relative/transform_ep0
```

输出：

- `epXXXX_transformed.mp4`  
- `epXXXX_transformed_inputs.npz`

### 12.5 对比两份 transform 导出（可选）

```bash
uv run python umi_scripts_csw/compare_npz.py \
  --a /path/to/a/ep0001_transformed_inputs.npz \
  --b /path/to/b/ep0001_transformed_inputs.npz \
  --out /path/to/compare_out \
  --save-action-heatmaps
```

---

## 13. 部署（openpi serve + RealRL eval）

### 13.1 openpi 启动策略服务（单臂/通用）

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=YOUR_CONFIG \
  --policy.dir=/path/to/checkpoint_dir
```

要求：`checkpoint_dir/assets/norm_stats.json` 等资产齐全。

### 13.2 openpi 启动策略服务（双臂）

```bash
uv run scripts/serve_policy_dual.py policy:checkpoint \
  --policy.config=YOUR_CONFIG \
  --policy.dir=/path/to/checkpoint_dir
```

### 13.3 RealRL 侧联调

RealRL 侧通常使用：

- `scripts/eval_policy.sh`（具体路径以你们 realRL 工程为准）

重点检查：

- policy host / port  
- 图像流键（如 `left_wrist_lumos,right_wrist_lumos,third_d455`）  
- **控制器是否使用 relative/delta**（与训练标签语义一致）

---

## 14. 相机发布（ZMQ）（可选）

仓库提供 `scripts/camera_zmq_pub.py` 用于把第三视角/腕部相机以 ZMQ 方式发布给下游系统。  
不同相机类型与 serial 需要按设备修改参数，典型命令形态如下（示例）：

```bash
python scripts/camera_zmq_pub.py \
  --name third_d455 --type realsense --serial YOUR_RS_SERIAL \
  --port 7000 --width 640 --height 480 --fps 30 --hz_pub 30 \
  --out_width 420 --out_height 240 \
  --jpeg_quality 80
```

---

## 15. 回放（真机 / 参考脚本）

仓库内存在 `umi_scripts_csw/replay_data_fastumi.py` 与 `replay_data_fastumi.sh`。  
在“仅 openpi 仓库”环境下**往往不能直接跑通**（依赖外部工程路径与模块）。  
若你要做真机回放，建议以你们完整的 FastUMI/控制栈工程为准，把 openpi 当作 **数据与训练**侧。

---

## 16. 经验法则（强烈建议保留为团队规范）

1. **relative 的本质**：减少/消除对某个固定 base 坐标系的过拟合依赖，但前提是数据与 policy 的相对定义一致。  
2. **图像频率与带宽**：多相机 + 高分辨率时，录制与训练侧频率要一致且可解释。  
3. **每批数据抽查**：至少做 `json_visualize` / `render_triad_mp4` / 小样本 offline eval。  
4. **loss 只能做粗指示**：最终以 offline eval + 真机表现为准。  
5. **retry 与干扰数据**：对泛化与稳定性影响非常大。

---

## 17. 统一入口（可选）

`umi_scripts_csw/run.py` 用于发现脚本与打印常用模板：

```bash
python3 umi_scripts_csw/run.py --examples
python3 umi_scripts_csw/run.py --list
python3 umi_scripts_csw/run.py --check
```

脚本分组清单：`umi_scripts_csw/scripts_index.json`

---

## 18. 旧脚本名 -> 当前统一入口（迁移表）

> 这节用于对照历史命令与文档片段，避免团队里旧链接失效后找不到替代。

| 历史脚本/入口 | 现在用 |
|---|---|
| `convert_ros_data_to_mp4.py` / `convert_rosbag_to_mp4_vis*.py` / `batch_stage1*.sh` | `stage1_convert.py --views 1|2|3`（`--mode vis`） |
| `json_sort_13.py` / `json_sort_123.py` | `json_sort.py` |
| `convert_mp4_data_to_lerobot_downsample*.py` / `convert_mp4_data_to_lerobot_123.py` | `stage2_convert.py --views 1|2|3` |
| `eval_actions.py` | 已移除；请使用 `eval_relative.py`（relative/transform） |
| `eval_dual_relative.py` / `eval_relative_visualize.py` | `eval_relative.py --mode dual|transform` |
