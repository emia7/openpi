# umi_scripts_csw

这个目录主要承载 FastUMI / XV / Franka 的数据处理与评测脚本。  
为降低使用门槛，新增统一入口：

```bash
python umi_scripts_csw/run.py --list
python umi_scripts_csw/run.py --list --group evaluation
python umi_scripts_csw/run.py --find eval
python umi_scripts_csw/run.py --check
python umi_scripts_csw/run.py --script convert_rosbag_to_mp4_vis_13.py -- --help
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
- 后续可逐步把同类脚本合并为参数化入口，再逐步清理重复变体（如 `*_13` / `*_123`）。
