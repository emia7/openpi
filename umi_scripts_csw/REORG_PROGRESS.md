# Reorganization Progress (dev-csw-reorganize)

## 已完成（兼容优先）

- 新增统一入口 `run.py`：
  - `--list / --group / --find / --check`
  - `--stage1`（转发到 `stage1_convert.py`）
  - `--stage2`（转发到 `stage2_convert.py`）
- 新增脚本索引 `scripts_index.json`，将分组信息配置化。
- 新增 Stage wrapper：
  - `stage1_convert.py`（`--views 1|2|3`）
  - `stage2_convert.py`（`--views 1|2|3`）
- 更新 `README.md` 与 `docs/umi_scripts.md`，补充统一入口说明。

## 新旧入口映射

### Stage1（rosbag -> mp4/json）

- 1 视角：
  - 新：`run.py --stage1 -- --views 1 ...`
  - 旧：`convert_ros_data_to_mp4.py`
- 2 视角：
  - 新：`run.py --stage1 -- --views 2 ...`
  - 旧：`convert_rosbag_to_mp4_vis_13.py`
- 3 视角：
  - 新：`run.py --stage1 -- --views 3 ...`
  - 旧：`convert_rosbag_to_mp4_vis_123.py`

### Stage2（mp4/json -> LeRobot）

- 1 视角：
  - 新：`run.py --stage2 -- --views 1 ...`
  - 旧：`convert_mp4_data_to_lerobot_downsample.py`
- 2 视角：
  - 新：`run.py --stage2 -- --views 2 ...`
  - 旧：`convert_mp4_data_to_lerobot_downsample_13.py`
- 3 视角：
  - 新：`run.py --stage2 -- --views 3 ...`
  - 旧：`convert_mp4_data_to_lerobot_123.py`

## 下一步建议

- 在 wrapper 中逐步吸收公共参数与日志格式，减少底层脚本差异。
- 为 `stage1_convert.py` 增加 `--mode vis/plain`，收敛 `convert_rosbag_to_mp4_vis.py` 与 `convert_ros_data_to_mp4.py`。
- 把 batch shell（`batch_stage1*.sh`）逐步转成 Python CLI 子命令。
