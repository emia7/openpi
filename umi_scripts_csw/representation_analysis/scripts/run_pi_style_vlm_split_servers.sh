#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# 双机分工：zx1 常有 UMI 数据 + ckpt；gxzx2 常有双臂 Franka 数据 + ckpt。
# SSH 实测（本机密钥）：zx1 用 /mnt/public/chenshuaiwen；gxzx2 用 /mnt/public1/chenshuaiwen。
#
# 用法：在对应机器上 source 本文件，再调用下列函数。
# 各函数使用 teleop_vs_umi_vlm_repr_pi_style.py 的 --groups + --skip-tsne，
# 将两台产出的 out 目录拷到一起后在本机执行 merge_pi_style_repr_npys.py。
# -----------------------------------------------------------------------------

set -euo pipefail

REPO="${OPENPI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export PYTHONPATH="$REPO/src:$REPO"
cd "$REPO"

run_gxzx2_teleop_both () {
  export TELEOP_CKPT TELEOP_ASSET_ID ROBOT_REPO UMI_REPO OUT_DIR MAX_SAMPLES SEED
  : "${TELEOP_CKPT:?}"
  : "${ROBOT_REPO:?}"
  : "${UMI_REPO:?}"
  OUT_DIR="${OUT_DIR:-$REPO/umi_scripts_csw/representation_analysis/analysis_results/partial_gxzx2_teleop}"
  python umi_scripts_csw/representation_analysis/teleop_vs_umi_vlm_repr_pi_style.py \
    --teleop-config pi05_dual_franka_finetune_high_lumos \
    --umi-config pi05_xv_dual_finetune \
    --teleop-checkpoint "$TELEOP_CKPT" \
    --groups teleop_model_robot_data,teleop_model_umi_data \
    --robot-repo "$ROBOT_REPO" \
    --umi-repo "$UMI_REPO" \
    --skip-tsne \
    --teleop-asset-id "${TELEOP_ASSET_ID:-handover_mix_8sets}" \
    --max-samples "${MAX_SAMPLES:-128}" \
    --seed "${SEED:-0}" \
    --out-dir "$OUT_DIR" \
    "$@"
}

run_gxzx2_teleop_robot_only () {
  export TELEOP_CKPT TELEOP_ASSET_ID ROBOT_REPO OUT_DIR MAX_SAMPLES SEED
  : "${TELEOP_CKPT:?}"
  : "${ROBOT_REPO:?}"
  OUT_DIR="${OUT_DIR:-$REPO/umi_scripts_csw/representation_analysis/analysis_results/partial_gxzx2_teleop_robot}"
  python umi_scripts_csw/representation_analysis/teleop_vs_umi_vlm_repr_pi_style.py \
    --teleop-config pi05_dual_franka_finetune_high_lumos \
    --umi-config pi05_xv_dual_finetune \
    --teleop-checkpoint "$TELEOP_CKPT" \
    --groups teleop_model_robot_data \
    --robot-repo "$ROBOT_REPO" \
    --skip-tsne \
    --teleop-asset-id "${TELEOP_ASSET_ID:-handover_mix_8sets}" \
    --max-samples "${MAX_SAMPLES:-128}" \
    --seed "${SEED:-0}" \
    --out-dir "$OUT_DIR" \
    "$@"
}

run_zx1_umi_both () {
  export UMI_CKPT UMI_ASSET_ID ROBOT_REPO UMI_REPO OUT_DIR MAX_SAMPLES SEED
  : "${UMI_CKPT:?}"
  : "${ROBOT_REPO:?}"
  : "${UMI_REPO:?}"
  OUT_DIR="${OUT_DIR:-$REPO/umi_scripts_csw/representation_analysis/analysis_results/partial_zx1_umi}"
  python umi_scripts_csw/representation_analysis/teleop_vs_umi_vlm_repr_pi_style.py \
    --teleop-config pi05_dual_franka_finetune_high_lumos \
    --umi-config pi05_xv_dual_finetune \
    --umi-checkpoint "$UMI_CKPT" \
    --groups umi_model_robot_data,umi_model_umi_data \
    --robot-repo "$ROBOT_REPO" \
    --umi-repo "$UMI_REPO" \
    --skip-tsne \
    --umi-asset-id "${UMI_ASSET_ID:-handover_umi}" \
    --max-samples "${MAX_SAMPLES:-128}" \
    --seed "${SEED:-0}" \
    --out-dir "$OUT_DIR" \
    "$@"
}

run_zx1_umi_umi_only () {
  export UMI_CKPT UMI_ASSET_ID UMI_REPO OUT_DIR MAX_SAMPLES SEED
  : "${UMI_CKPT:?}"
  : "${UMI_REPO:?}"
  OUT_DIR="${OUT_DIR:-$REPO/umi_scripts_csw/representation_analysis/analysis_results/partial_zx1_umi_umi}"
  python umi_scripts_csw/representation_analysis/teleop_vs_umi_vlm_repr_pi_style.py \
    --teleop-config pi05_dual_franka_finetune_high_lumos \
    --umi-config pi05_xv_dual_finetune \
    --umi-checkpoint "$UMI_CKPT" \
    --groups umi_model_umi_data \
    --umi-repo "$UMI_REPO" \
    --skip-tsne \
    --umi-asset-id "${UMI_ASSET_ID:-handover_umi}" \
    --max-samples "${MAX_SAMPLES:-128}" \
    --seed "${SEED:-0}" \
    --out-dir "$OUT_DIR" \
    "$@"
}

echo "已加载函数: run_gxzx2_teleop_both | run_gxzx2_teleop_robot_only | run_zx1_umi_both | run_zx1_umi_umi_only"
echo "合并示例: python umi_scripts_csw/representation_analysis/merge_pi_style_repr_npys.py --input-dir DIR1 --input-dir DIR2 --out-dir merged"
