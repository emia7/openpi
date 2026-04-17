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
- `stage2_convert.py` 从子进程分发改为进程内分发，成为 Stage2 实际主执行入口。
- 已移除 `convert_mp4_data_to_lerobot_{downsample,downsample_13,123}.py`，Stage2 只保留 `stage2_convert.py` + `stage2_core.py`。
- `stage1_convert.py` 也从子进程分发改为进程内调用（views=1/2/3 统一由一个入口执行）。
- 已移除 `convert_ros{,_vis,_vis_13,_vis_123}.py`，Stage1 只保留 `stage1_convert.py` + `stage1_core.py`。
- `json_sort.py` 已统一吸收 `json_sort_13.py/json_sort_123.py` 的主要能力，重命名工具收敛为单脚本入口。
- 相对评估脚本已合并为 `eval_relative.py`（`--mode single|dual`），移除 `eval_dual_relative.py` 和 `eval_relative_visualize.py`。

## 新旧入口映射

### Stage1（rosbag -> mp4/json）

- 1 视角：
  - 新：`run.py --stage1 -- --views 1 ...`
- 2 视角：
  - 新：`run.py --stage1 -- --views 2 ...`
- 3 视角：
  - 新：`run.py --stage1 -- --views 3 ...`

### Stage2（mp4/json -> LeRobot）

- 1 视角：
  - 新：`run.py --stage2 -- --views 1 ...`
- 2 视角：
  - 新：`run.py --stage2 -- --views 2 ...`
- 3 视角：
  - 新：`run.py --stage2 -- --views 3 ...`

## 合并前建议检查

- 用一小批 bag 验证 `stage1_convert.py`：
  - `--views 1 --mode plain`
  - `--views 1 --mode vis`
  - `--views 2 --jobs N --serials S1,S2`
- 用同一份 stage1 输出验证 `stage2_convert.py --views 1/2/3` 的字段与下游配置一致。
- 若团队确认稳定，可在后续 PR 再考虑把更多 legacy 脚本迁移到统一 wrapper（当前保持兼容，不做侵入式合并）。
