# Franka 双臂数据 → LeRobot（简要）

## 1. 数据长什么样

每个 episode 同一目录下需有：

- `{base}.parquet`、 `{base}_meta.json`
- `{base}_{相机名}.mp4`，相机名与 `meta.json` 里 `cameras` 列表一致（例如五路腕部 + 第三视角）

标签来自 parquet 的 `next_observation/state/*`，**不用** `action` 列；最后一帧动作为当前观测复制。详见 [`convert_dual_franka_data_to_lerobot.py`](convert_dual_franka_data_to_lerobot.py) 文件头注释。

## 2. 转换（在 openpi 仓库根目录执行）

```bash
cd /path/to/openpi

uv run python dual_scripts/convert_dual_franka_data_to_lerobot.py \
  --input_dirs=/你的数据目录/handover_high \
  --repo_id=local/handover_high_dual
```

`--repo_id` 也可以是**绝对路径**（整段作为数据集根）。训练时 [`config.py`](../src/openpi/training/config.py) 里对应 `TrainConfig` 的 `repo_id` 必须与转换时**完全一致**（例如本机示例：`/share/chenshuaiwen-local/.cache/hf_home/dual_franka/handover_high`）。

- 默认输出目录：`HF_LEROBOT_HOME`（常由 `HF_HOME` 决定，子路径为 `repo_id`）。指定自定义根目录：

```bash
uv run python dual_scripts/convert_dual_franka_data_to_lerobot.py \
  --input_dirs=/你的数据目录/handover_high \
  --repo_id=local/handover_high_dual \
  --output_dir=/你想放的父目录
```

实际数据集路径：`{output_dir}/{repo_id}`（`repo_id` 可含 `local/xxx`）。

## 3. 常用可选参数

```bash
# 多目录合并
uv run python dual_scripts/convert_dual_franka_data_to_lerobot.py \
  --input_dirs=/data/A --input_dirs=/data/B \
  --repo_id=local/my_dual_set

# 随机抽 N 条 episode（先打乱，seed 控制可复现）
uv run python dual_scripts/convert_dual_franka_data_to_lerobot.py \
  --input_dirs=/data/handover_high \
  --repo_id=local/subset \
  --num_episodes=50 \
  --seed=42

# 包含 meta 里 episode_failed=true 的条（默认跳过）
uv run python dual_scripts/convert_dual_franka_data_to_lerobot.py \
  --input_dirs=/data/handover_high \
  --repo_id=local/all_including_failed \
  --noskip_failed
```

## 4. 转出来的 LeRobot 里有什么（易懂版）

**落盘位置**：`{output_dir}/{repo_id}` 这一整个文件夹就是一个 LeRobot 数据集。里面会有 `meta/`（含 `info.json` 描述所有字段）、按块存的数据表、以及各路相机对应的视频文件等；**具体子目录名**以你当前 LeRobot 版本为准。打开 `meta/info.json` 可看到完整字段列表（LeRobot 还会自动加上帧号、episode 号、时间戳等索引类列，不必手写）。

**每一帧（一条样本）在训练里主要会用到这些键**（名字与脚本写入一致）：

**图像（有几路相机就有几个键）**

- 键名形如：`observation.images.left_wrist_d435`、`…left_wrist_lumos`、`…right_wrist_d435`、`…right_wrist_lumos`、`…third_d455`（与原始 `meta.json` 里 `cameras` 列表一致）。
- 内容：**RGB 彩色图**，形状约为 `高 × 宽 × 3`（你当前数据多为 240×420，若某路分辨率不同会按该路单独记录）。

**当前状态（两只手 + 夹爪）**

- `observation.state.left_eef_pos`：左手末端 **位置**（3 个数：x,y,z）。
- `observation.state.left_eef_rotvec`：左手末端 **朝向**（3 个数：旋转向量，由四元数换算而来）。
- `observation.state.left_gripper`：左夹爪（1 个数）。
- `observation.state.right_eef_pos` / `right_eef_rotvec` / `right_gripper`：右手同上。

**本段演示的起点（方便以后做「相对起点」的训练）**

- `observation.state.demo_start_pose_left`：本 episode **第一帧** 左手的「位置+旋转向量」共 6 个数；**每一帧都重复写同一个值**。
- `observation.state.demo_start_pose_right`：右手同理。

**要学习的「下一步目标」（监督信号）**

- `left_action`：左手 **下一时刻** 的「位置 3 + 朝向 3 + 夹爪 1」共 7 个数；来自 parquet 里 **下一时刻状态**（`next_observation`）。
- `right_action`：右手同理。
- **最后一帧**没有真正的「下一帧」时：这两维 **等于当前这一帧的左右手状态**（等价于「停在原地」）。

**其它**

- `task`：语言指令字符串（来自 `meta.json` 的 `task_description`）。
- `task_idx` / `subtask_idx` / `done`：从原始 parquet 带上来的任务编号与是否 episode 结束。

**刻意没有用的列**：原始 parquet 里的 `action` / `logged_action` **不会**写进上述监督里（标签只看 `next_observation` + 最后一帧复制规则）。

## 5. 训练 config 与策略（`config.py` / `dual_franka_policy.py`）

- **数据侧**：[`src/openpi/training/config.py`](../src/openpi/training/config.py) 中的 `LeRobotDualFrankaDataConfig` 用 `RepackTransform` 把 LeRobot 带点字段映射到 `left_view` / `right_view` / `third_view` 与双手状态、`left_action` / `right_action`；`base_config` 里已设 `action_sequence_keys=("left_action", "right_action")`。
- **策略侧**：[`src/openpi/policies/dual_franka_policy.py`](../src/openpi/policies/dual_franka_policy.py) 的 `DualFranka*Inputs` / `DualFrankaDualHandOutputs`（不继承 `xv_dual_policy`）。

当前注册了 **四个** `TrainConfig`（名称即选用哪套「第三视角槽位 + 腕部相机键」）：

| `TrainConfig.name` | 含义（简要） |
| --- | --- |
| `pi05_dual_franka_finetune_high_lumos` | 第三视角用 *High* 类 Inputs；腕部 Repack 用 `observation.images.left_wrist_lumos` / `right_wrist_lumos` |
| `pi05_dual_franka_finetune_high_rs` | 同上；腕部用 `left_wrist_d435` / `right_wrist_d435` |
| `pi05_dual_franka_finetune_mask_lumos` | 第三视角用 *Mask* 类 Inputs；腕部 Lumos |
| `pi05_dual_franka_finetune_mask_rs` | *Mask* 类 Inputs；腕部 D435 |

（注：当前 `DualFrankaMask*` 与 `DualFrankaHigh*` 的 `Inputs` 前向实现相同，区别主要在命名与后续可扩展；**腕部选 Lumos 还是 D435** 由 Repack 键决定，效果不同。）

四个 config 默认共用**同一条** `repo_id`（在 `TrainConfig` 里写死，按你的机器改成与转换时 `--repo_id` **完全相同**的路径）。**多个任务、不同 episode 数量**应合并进**同一个** LeRobot 数据集，而不是再拆四个 train 名字。

**`asset_id`**：这四个 entry **未**单独设置 `AssetsConfig(asset_id=…)`，因此 `asset_id` 会回落为 **`repo_id` 字符串**。这样 `scripts/compute_norm_stats.py` 的写出目录与训练时加载 `norm_stats.json` 的路径一致。若 `repo_id` 为绝对路径，统计文件通常写在**数据集根目录**下的 `norm_stats.json`。

## 6. 归一化统计与开训（仓库根目录）

**算 norm（同一 `repo_id` 只需跑一次）**，任选其一 config 名即可，例如：

```bash
cd /path/to/openpi
uv run scripts/compute_norm_stats.py --config-name pi05_dual_franka_finetune_high_lumos
```

可选：`--max-frames N` 先做小样本试跑。若算 norm 时也要固定 GPU，可在命令前加 `CUDA_VISIBLE_DEVICES=0`（与训练时习惯一致即可）。

**训练**（选用与腕部/第三视角设定一致的那一个 config）。多 GPU 机器上请用 **`CUDA_VISIBLE_DEVICES`** 指定要用的物理卡（下面示例为卡 `0`，按机器改成 `1` 或 `0,1` 等）：

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_dual_franka_finetune_high_lumos --exp-name=my_run --overwrite
```

也可先 `export CUDA_VISIBLE_DEVICES=0` 再执行同一行里的 `uv run ...`（仅保留 `XLA_PYTHON_CLIENT_MEM_FRACTION=...` 前缀即可）。

`checkpoint_base_dir` 等在对应 `TrainConfig` 里已设；若需改路径，直接改 `config.py` 中该 entry。

更多与 openpi 通用的说明见仓库根目录 [`README.md`](../README.md) 的 Fine-Tuning 小节。
