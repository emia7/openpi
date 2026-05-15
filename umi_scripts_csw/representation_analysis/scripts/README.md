# 运维脚本（数据集同步、一键跑 zx1）

与表征算法无关；分析入口与参数见上级 [**README.md**](../README.md)。

| 文件 | 说明 |
|------|------|
| `run_pi_style_four_way_vision_zx1.sh` | zx1 四路 **`--representation vision`**；默认 `OPENPI_ROOT`、`CACHE_ROOT`；`PYTHON_BIN` 默认 `.venv/bin/python`；需 **`--umi-asset-id`**（脚本内默认 `handover_umi_0429_mix`） |
| `run_pi_style_vlm_on_public1_example.sh` | 通过环境变量包一层调用主脚本 |
| `run_pi_style_vlm_split_servers.sh` | gxzx2 / zx1 分机器 **`--skip-tsne`** 产出 `.npy` |
| `sync_dataset_gxzx2_to_zx1_tar.sh`、`start_dataset_sync_tmux.sh`、`rsync_gxzx2_to_zx1.example.sh` | 数据拷贝示例 |
