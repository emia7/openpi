#!/usr/bin/env bash
# 在 tmux 会话里跑 gxzx2 -> zx1 数据集 tar 同步（断线、关 Cursor 也不中断本机 ssh 子进程，Mac 睡眠仍会断）。
# 依赖: brew install tmux
#
# 用法:
#   bash umi_scripts_csw/representation_analysis/scripts/start_dataset_sync_tmux.sh
#   tmux attach -t gxzx_dataset_sync

set -euo pipefail

SESSION="${TMUX_SESSION:-gxzx_dataset_sync}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYNC_SCRIPT="$HERE/sync_dataset_gxzx2_to_zx1_tar.sh"
REPO="$(cd "$HERE/../../.." && pwd)"
LOG="${SYNC_LOG:-$REPO/umi_scripts_csw/representation_analysis/sync_dataset_live.log}"

if [[ ! -f "$SYNC_SCRIPT" ]]; then
  echo "未找到 $SYNC_SCRIPT"
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "未安装 tmux，请先执行: brew install tmux"
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "会话已存在: $SESSION"
  echo "接入: tmux attach -t $SESSION"
  echo "若仍占用旧同步，可先: tmux kill-session -t $SESSION"
  exit 0
fi

mkdir -p "$(dirname "$LOG")"
export SYNC_LOG="$LOG"

# 会话内跑同步；结束后保留 pane 方便看最后一屏日志
tmux new-session -d -s "$SESSION" "bash -lc '
  export SYNC_LOG=\"$LOG\"
  echo 日志: \"\$SYNC_LOG\"
  bash \"$SYNC_SCRIPT\"; ec=\$?;
  echo; echo === 结束 exit=\$ec ===;
  echo 按回车关闭本 pane; read
'"

echo "已在 tmux 会话 [$SESSION] 中启动同步。"
echo "  接入: tmux attach -t $SESSION"
echo "  脱离: Ctrl+B 松手后按 D"
echo "  日志: $LOG"
echo "  另开终端盯体积:"
echo "    while sleep 30; do date; ssh chenshuaiwen-zx1 \"du -sh /mnt/public/chenshuaiwen/.cache/hf_home/dual_franka/handover_mix_8sets\"; done"
