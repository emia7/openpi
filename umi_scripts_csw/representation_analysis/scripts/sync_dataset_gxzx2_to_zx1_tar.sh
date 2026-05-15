#!/usr/bin/env bash
# 在「你自己的终端」里跑（Cursor 里超长传输常被中断）。
# 进度：另开一个终端 watch： ssh chenshuaiwen-zx1 'du -sh ...handover_mix_8sets'
set -euo pipefail
LOG="${SYNC_LOG:-$HOME/Desktop/openpi_sync_dataset.log}"
echo "日志: $LOG"
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date) 开始 tar 流 gxzx2 -> zx1 handover_mix_8sets ==="
ssh -o ServerAliveInterval=30 chenshuaiwen-gxzx2 \
  "cd /mnt/public1/chenshuaiwen/.cache/hf_home/dual_franka && tar cf - handover_mix_8sets" \
| ssh -o ServerAliveInterval=30 chenshuaiwen-zx1 \
  "mkdir -p /mnt/public/chenshuaiwen/.cache/hf_home/dual_franka && cd /mnt/public/chenshuaiwen/.cache/hf_home/dual_franka && tar xf -"
echo "=== $(date) 结束 ==="
ssh chenshuaiwen-zx1 "du -sh /mnt/public/chenshuaiwen/.cache/hf_home/dual_franka/handover_mix_8sets"
