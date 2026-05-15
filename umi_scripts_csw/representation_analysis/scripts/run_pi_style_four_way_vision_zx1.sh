#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# 在 zx1 上验证「仅图像编码器」表征（SigLIP → projector，`--representation vision`），
# 与已生成的 `analysis_results/umi_model_four_way_duck_bagging/`（默认 VLM 前缀）对照。
# 只依赖 openpi 仓库内脚本与路径；勿改系统目录。
#
# 用法（SSH 登录服务器后）：
#   cd /home/chenshuaiwen/openpi
#   bash umi_scripts_csw/representation_analysis/scripts/run_pi_style_four_way_vision_zx1.sh
#
# 可选环境变量（覆盖默认）：
#   OPENPI_ROOT   默认 /home/chenshuaiwen/openpi
#   PYTHON_BIN    默认 $OPENPI_ROOT/.venv/bin/python
#   UMI_CKPT      默认 $OPENPI_ROOT/checkpoints_pt/handover_umi_0429_mix_38000_pt
#   UMI_ASSET_ID  默认 handover_umi_0429_mix（须与 checkpoint 下 assets/ 中目录一致）
#   CACHE_ROOT    默认 /mnt/public/chenshuaiwen/.cache/hf_home（其下 dual_franka / fastumi）
#   MAX_SAMPLES   默认 64
#   OUT_DIR       默认 .../analysis_results/umi_model_four_way_duck_bagging_vision
# 脚本末尾 "$@" 可再追加 CLI 参数，例如 --device cuda:0
# -----------------------------------------------------------------------------

set -euo pipefail

ROOT="${OPENPI_ROOT:-/home/chenshuaiwen/openpi}"
PY="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
UMI_CKPT="${UMI_CKPT:-$ROOT/checkpoints_pt/handover_umi_0429_mix_38000_pt}"
CACHE_ROOT="${CACHE_ROOT:-/mnt/public/chenshuaiwen/.cache/hf_home}"
ROBOT_REPO="${ROBOT_REPO:-$CACHE_ROOT/dual_franka/handover_mix_8sets}"
UMI_REPO="${UMI_REPO:-$CACHE_ROOT/fastumi/handover_umi_0429_mix}"
UMI_OTHER_REPO="${UMI_OTHER_REPO:-$CACHE_ROOT/fastumi/duck_0430}"
UMI_BAGGING_REPO="${UMI_BAGGING_REPO:-$CACHE_ROOT/fastumi/bagging_0430}"
OUT_DIR="${OUT_DIR:-$ROOT/umi_scripts_csw/representation_analysis/analysis_results/umi_model_four_way_duck_bagging_vision}"
MAX_SAMPLES="${MAX_SAMPLES:-64}"
UMI_ASSET_ID="${UMI_ASSET_ID:-handover_umi_0429_mix}"

export PYTHONPATH="$ROOT/src:$ROOT"
cd "$ROOT"

exec "$PY" umi_scripts_csw/representation_analysis/teleop_vs_umi_vlm_repr_pi_style.py \
  --teleop-config pi05_dual_franka_finetune_high_lumos \
  --umi-config pi05_xv_dual_finetune \
  --umi-checkpoint "$UMI_CKPT" \
  --umi-asset-id "$UMI_ASSET_ID" \
  --robot-repo "$ROBOT_REPO" \
  --umi-repo "$UMI_REPO" \
  --umi-other-repo "$UMI_OTHER_REPO" \
  --umi-bagging-repo "$UMI_BAGGING_REPO" \
  --groups umi_model_robot_data,umi_model_umi_data,umi_model_umi_other_data,umi_model_umi_bagging_data \
  --robot-partial-repo \
  --max-samples "$MAX_SAMPLES" \
  --representation vision \
  --out-dir "$OUT_DIR" \
  "$@"
