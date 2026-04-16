# umi_scripts_csw

这个目录主要承载 FastUMI / XV / Franka 的数据处理与评测脚本。  
为降低使用门槛，新增统一入口：

```bash
python umi_scripts_csw/run.py --list
python umi_scripts_csw/run.py --list --group evaluation
python umi_scripts_csw/run.py --find eval
python umi_scripts_csw/run.py --examples
python umi_scripts_csw/run.py --check
python umi_scripts_csw/run.py --stage1 -- --views 1 --bag /data/ep001.bag --serial XV_SERIAL --out_dir /data/stage1 --data_idx 1
python umi_scripts_csw/run.py --stage1 -- --views 1 --mode vis --bag /data/ep001.bag --serial XV_SERIAL --out_dir /data/stage1 --data_idx 00001
python umi_scripts_csw/run.py --stage2 -- --views 2 --stage1_dir /data/stage1 --repo local/my_repo --target_fps 10
python umi_scripts_csw/run.py --stage1 -- --views 2 --bag_dir /data/bags --out_dir /data/stage1 --start_idx 0 --serials S1,S2 --jobs 4 --skip_existing
python umi_scripts_csw/run.py --script convert_rosbag_to_mp4_vis_13.py -- --help
python umi_scripts_csw/stage1_convert.py --views 2 --bag /data/ep001.bag --serial XV_SERIAL --out_dir /data/stage1 --data_idx 1
python umi_scripts_csw/stage2_convert.py --views 2 --stage1_dir /data/stage1 --repo local/my_repo --target_fps 10
```

## 功能分组（不改旧脚本路径）

- `stage1_ros_to_mp4_json`: ROS bag 转 mp4/json。
- `stage2_mp4_json_to_lerobot`: mp4/json 转 LeRobot 数据集。
- `dataset_tools`: 重命名、可视化、比对、坐标变换等工具。
- `evaluation`: 评测与可视化脚本。
- `runtime_helpers`: 采集/回放/便捷控制脚本。
- 分组清单维护在 `umi_scripts_csw/scripts_index.json`，后续增减脚本优先改这个文件。

## 说明

- 本次重构第一阶段保持 **兼容优先**：旧脚本仍按原路径直接可运行。
- 新增 `stage1_convert.py` 作为统一入口：`--views 1|2|3` 分发到原有 stage1 脚本（1/2 单 bag，3 批处理）。
- `stage1_convert.py` 已支持 views=1/2 的批处理模式（`--bag_dir + --serials + --jobs`），用于替代原先 `batch_stage1*.sh` 的核心流程。
- `--views 1` 可用 `--mode vis` 切到 `convert_rosbag_to_mp4_vis.py`（用于保留状态可视化输出）。
- 新增 `stage2_convert.py` 作为统一入口：`--views 1|2|3` 分发到原有三个 stage2 转换脚本。
- `stage2_convert.py` 现在是 Stage2 主入口（进程内按 `--views` 调用对应实现），建议日常只用它。
- 后续可逐步把同类脚本合并为参数化入口，再逐步清理重复变体（如 `*_13` / `*_123`）。
- 重构进度与新旧映射见 `umi_scripts_csw/REORG_PROGRESS.md`。
