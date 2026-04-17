# FastUMI Pro 安装与使用（当前流程版）

本文基于 `docs/FastUMI Pro安装使用.pdf` 的原始流程重写，命令已经更新为当前重构后的统一入口。  
适用于当前分支：`dev-csw-reorganize`。

---

## 1. 目标与总流程

当前推荐的数据与训练链路：

1. 设备与 ROS 环境准备（FastUMI/XV + 可选 D435）。  
2. 采集 rosbag（建议单臂/双臂都统一录到 10Hz 图像话题）。  
3. Stage1：`rosbag -> mp4 + json`（统一脚本：`stage1_convert.py`）。  
4. 质检与整理：`json_sort.py`、`json_visualize.py`、`compare_batches.py`。  
5. Stage2：`mp4 + json -> LeRobot`（统一脚本：`stage2_convert.py`）。  
6. 训练：`scripts/train.py` + `scripts/compute_norm_stats.py`。  
7. 离线评估：`eval_relative.py`（`single|dual|transform`）。  
8. 在线部署：`serve_policy*.py` + RealRL 侧 eval。

---

## 2. 环境准备

### 2.1 基础环境

- Ubuntu 20.04
- ROS Noetic
- 本仓库代码：`https://github.com/emia7/openpi`

### 2.2 OpenPI Python 环境

```bash
cd /path/to/openpi
uv sync
```

### 2.3 FastUMI/XV 硬件 SDK（沿用原流程）

硬件 SDK、monitor、USB 带宽扩展、D435 驱动安装步骤，继续参考原 PDF。  
本文不重复抄写硬件驱动细节，只更新 openpi 数据链路与命令入口。

---

## 3. 数据采集（rosbag）

建议保留原文档的采集实践：

- 启动 `xv_sdk`
- 使用 monitor 检查传感器状态
- 需要第三视角时，启动 D435
- 录制图像 + pose + clamp 等关键话题

若录制带宽紧张，推荐先用 `tri_image_sampler_10hz.py` 降采样后再录包。

---

## 4. Stage1：rosbag -> mp4+json（统一）

统一入口：`umi_scripts_csw/stage1_convert.py`

### 4.1 单臂（1 视角）

```bash
python umi_scripts_csw/stage1_convert.py \
  --views 1 \
  --bag /path/to/one.bag \
  --serial YOUR_SERIAL \
  --out_dir /path/to/stage1_out \
  --data_idx 1
```

可视化模式（保留旧 `vis` 能力）：

```bash
python umi_scripts_csw/stage1_convert.py \
  --views 1 --mode vis \
  --bag /path/to/one.bag \
  --serial YOUR_SERIAL \
  --out_dir /path/to/stage1_out \
  --data_idx 00001
```

### 4.2 双臂（2 视角）

```bash
python umi_scripts_csw/stage1_convert.py \
  --views 2 \
  --bag /path/to/one.bag \
  --serial SERIAL_LEFT,SERIAL_RIGHT \
  --out_dir /path/to/stage1_out \
  --data_idx 1
```

批处理：

```bash
python umi_scripts_csw/stage1_convert.py \
  --views 2 \
  --bag_dir /path/to/bags \
  --out_dir /path/to/stage1_out \
  --start_idx 0 \
  --serials SERIAL_LEFT,SERIAL_RIGHT \
  --jobs 4 \
  --skip_existing
```

### 4.3 三视角（2 腕部 + 1 第三视角）

```bash
python umi_scripts_csw/stage1_convert.py \
  --views 3 \
  --bag_dir /path/to/bags \
  --out_dir /path/to/stage1_out \
  --start_idx 0
```

---

## 5. 质检与整理（Stage1 后）

### 5.1 统一重命名（替代 `json_sort_13.py/json_sort_123.py`）

预览：

```bash
python umi_scripts_csw/json_sort.py --dir /path/to/stage1_out
```

执行：

```bash
python umi_scripts_csw/json_sort.py --dir /path/to/stage1_out --apply
```

### 5.2 可视化/对比建议

- `json_visualize.py`：看单条 episode 的 pose/clamp 曲线
- `render_triad_mp4.py`：看三维轨迹
- `compare_batches.py`：看两批数据统计差异

---

## 6. Stage2：mp4+json -> LeRobot（统一）

统一入口：`umi_scripts_csw/stage2_convert.py`

### 6.1 单臂

```bash
python umi_scripts_csw/stage2_convert.py \
  --views 1 \
  --stage1_dir /path/to/stage1_out \
  --repo /path/to/hf_home/fastumi_repo \
  --target_fps 10
```

### 6.2 双臂

```bash
python umi_scripts_csw/stage2_convert.py \
  --views 2 \
  --stage1_dir /path/to/stage1_out \
  --repo /path/to/hf_home/fastumi_repo \
  --target_fps 10
```

### 6.3 三视角

```bash
python umi_scripts_csw/stage2_convert.py \
  --views 3 \
  --stage1_dir /path/to/stage1_out \
  --repo /path/to/hf_home/fastumi_repo \
  --fps 10
```

---

## 7. 训练流程（openpi）

### 7.1 计算归一化统计

```bash
uv run scripts/compute_norm_stats.py --config-name YOUR_CONFIG
```

### 7.2 训练

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py YOUR_CONFIG --exp-name=YOUR_EXP --overwrite
```

常见配置示例（按你当前任务选）：

- 单臂：`pi05_xv_finetune`
- 双臂：`pi05_xv_dual_finetune`
- 其它 UMI/Hybrid：见 `src/openpi/training/config.py`

---

## 8. 离线评估（统一）

统一脚本：`umi_scripts_csw/eval_relative.py`

### 8.1 相对动作评估（单臂/双臂）

```bash
uv run python umi_scripts_csw/eval_relative.py \
  --mode single \
  --config pi05_xv_finetune \
  --exp-name YOUR_EXP \
  --step 20000 \
  --repo /path/to/lerobot_repo \
  --episode 0 \
  --compare-steps "0,1,2,3" \
  --out-dir eval_results/relative/single_ep0
```

双臂只需改 `--mode dual` + 对应 config。

### 8.2 transform 导出（替代旧 `eval_relative_visualize.py`）

```bash
uv run python umi_scripts_csw/eval_relative.py \
  --mode transform \
  --config pi05_xv_finetune \
  --exp-name YOUR_EXP \
  --step 20000 \
  --repo /path/to/lerobot_repo \
  --episode 0 \
  --out-dir eval_results/relative/transform_ep0
```

该模式会导出：

- `epXXXX_transformed.mp4`
- `epXXXX_transformed_inputs.npz`

---

## 9. 在线部署与实机验证

### 9.1 openpi 侧服务

单臂/通用：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=YOUR_CONFIG \
  --policy.dir=/path/to/checkpoint
```

双臂：

```bash
uv run scripts/serve_policy_dual.py policy:checkpoint \
  --policy.config=YOUR_CONFIG \
  --policy.dir=/path/to/checkpoint
```

### 9.2 RealRL 侧

继续沿用你原工作流中的 `eval_policy.sh` 与任务配置。  
重点确保控制端使用 relative/delta 控制语义，并与策略输入键对齐。

---

## 10. 常见问题（迁移到统一入口后）

- 旧脚本名（如 `*_13`、`*_123`、`eval_dual_relative.py`）已下线，改用统一入口参数控制。
- Stage1/Stage2 只建议走：
  - `stage1_convert.py --views 1|2|3`
  - `stage2_convert.py --views 1|2|3`
- eval 只建议走：
  - `eval_relative.py --mode single|dual|transform`
- 如果 `run.py --check` 失败，先确认 `scripts_index.json` 与实际脚本是否一致。

---

## 11. 快速入口（可选）

可以使用统一调度器 `run.py` 快速查看命令模板：

```bash
python umi_scripts_csw/run.py --examples
python umi_scripts_csw/run.py --list
python umi_scripts_csw/run.py --check
```
