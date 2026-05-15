#!/usr/bin/env bash
# 在集群节点（如 chenshuaiwen-gxzx2）上运行 PI 风格 VLM 表征对比。
# 用法（登录服务器后）：
#   cd /path/to/openpi
#   export TELEOP_CKPT=... UMI_CKPT=...
#   bash umi_scripts_csw/representation_analysis/scripts/run_pi_style_vlm_on_public1_example.sh
#
# 说明：Cursor/本助手无法替你 SSH；请在终端自行 ssh 到 gxzx2 再执行。

set -euo pipefail

ROOT="${PUBLIC1:-/mnt/public1/chenshuaiwen}"
# 脚本位于 openpi/umi_scripts_csw/representation_analysis/scripts/
REPO="${OPENPI_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"

# 与 src/openpi/training/config.py 中 pi05_* 默认路径一致（可按实际微调）
ROBOT_REPO="${ROBOT_REPO:-$ROOT/.cache/hf_home/dual_franka/handover_mix_8sets}"
UMI_REPO="${UMI_REPO:-$ROOT/.cache/hf_home/fastumi/handover_umi}"

# 训练产物目录：请改成你在该机上存在的 step 目录（内含 model.safetensors 或 weights）
TELEOP_CKPT="${TELEOP_CKPT:?请 export TELEOP_CKPT=/path/to/pi05_dual_franka_finetune/某exp/某step}"
UMI_CKPT="${UMI_CKPT:?请 export UMI_CKPT=/path/to/pi05_xv_dual_finetune/某exp/某step}"

TELEOP_ASSET_ID="${TELEOP_ASSET_ID:-handover_mix_8sets}"
UMI_ASSET_ID="${UMI_ASSET_ID:-handover_umi}"

OUT="${OUT_DIR:-$REPO/umi_scripts_csw/representation_analysis/analysis_results/pi_style_gxzx2_run}"

export PYTHONPATH="$REPO/src:$REPO"

cd "$REPO"
exec python umi_scripts_csw/representation_analysis/teleop_vs_umi_vlm_repr_pi_style.py \
  --teleop-config pi05_dual_franka_finetune_high_lumos \
  --umi-config pi05_xv_dual_finetune \
  --teleop-checkpoint "$TELEOP_CKPT" \
  --umi-checkpoint "$UMI_CKPT" \
  --robot-repo "$ROBOT_REPO" \
  --umi-repo "$UMI_REPO" \
  --teleop-asset-id "$TELEOP_ASSET_ID" \
  --umi-asset-id "$UMI_ASSET_ID" \
  --max-samples "${MAX_SAMPLES:-128}" \
  --out-dir "$OUT" \
  "$@"
