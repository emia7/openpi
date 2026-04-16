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
  - `stage1_convert.py` 已支持 views=1/2 批处理（自动按 serial 候选重试并写日志），并支持 `--jobs` 并行。
  - `stage1_convert.py` 已支持 `--mode vis`（views=1）保留单路状态可视化流程。
- 更新 `README.md` 与 `docs/umi_scripts.md`，补充统一入口说明。
- 删除 legacy 批处理 shell：`batch_stage1*.sh`（流程已迁移到 Python 入口）。
- 新增 `stage2_core.py`，抽离 Stage2 三个转换脚本共享的 `pose`/`json`/`stride` 逻辑。
- `convert_mp4_data_to_lerobot_{downsample,downsample_13,123}.py` 现直接调用 `stage2_core.py`；旧文件主要作为按视角分流的适配入口。

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

## 合并前建议检查

- 用一小批 bag 验证 `stage1_convert.py`：
  - `--views 1 --mode plain`
  - `--views 1 --mode vis`
  - `--views 2 --jobs N --serials S1,S2`
- 用同一份 stage1 输出验证 `stage2_convert.py --views 1/2/3` 的字段与下游配置一致。
- 若团队确认稳定，可在后续 PR 再考虑把更多 legacy 脚本迁移到统一 wrapper（当前保持兼容，不做侵入式合并）。
