#!/usr/bin/env bash
# 将 gxzx2（/mnt/public1/chenshuaiwen）上的双臂 Franka 数据与遥操 checkpoint 拷到 zx1（/mnt/public/chenshuaiwen）。
# 在你本机执行（需已配置 SSH：chenshuaiwen-gxzx2、chenshuaiwen-zx1）。
#
# 用法：
#   SRC_DATA=... SRC_CKPT=... DST_DATA=... DST_CKPT=... bash rsync_gxzx2_to_zx1.example.sh
# 或编辑默认值后：
#   bash rsync_gxzx2_to_zx1.example.sh
#
# 说明：用「gxzx2 -> 本机临时目录 -> zx1」两段 rsync，不要求 gxzx2 能直连 zx1。

set -euo pipefail

GXZX2="${GXZX2_HOST:-chenshuaiwen-gxzx2}"
ZX1="${ZX1_HOST:-chenshuaiwen-zx1}"

PUB1="/mnt/public1/chenshuaiwen"
PUB="/mnt/public/chenshuaiwen"

SRC_DATA="${SRC_DATA:-$PUB1/.cache/hf_home/dual_franka/handover_mix_8sets}"
DST_DATA="${DST_DATA:-$PUB/.cache/hf_home/dual_franka/handover_mix_8sets}"

SRC_CKPT="${SRC_CKPT:-$PUB1/checkpoints/pi05_dual_franka_finetune_high_lumos/handover_third_lumos_400}"
DST_CKPT="${DST_CKPT:-$PUB/checkpoints/pi05_dual_franka_finetune_high_lumos/handover_third_lumos_400}"

RSYNC_OPTS=( -avz --progress )

TMP="${TMPDIR:-/tmp}/gxzx2_to_zx1_$$"
mkdir -p "$TMP"

echo ">>> 1) $GXZX2:$SRC_DATA -> 本机 $TMP/data"
rsync "${RSYNC_OPTS[@]}" -e ssh "$GXZX2:$SRC_DATA/" "$TMP/data/"

echo ">>> 2) $GXZX2:$SRC_CKPT -> 本机 $TMP/ckpt"
rsync "${RSYNC_OPTS[@]}" -e ssh "$GXZX2:$SRC_CKPT/" "$TMP/ckpt/"

echo ">>> 3) 本机 -> $ZX1（创建远端目录）"
ssh "$ZX1" "mkdir -p $(dirname "$DST_DATA") $(dirname "$DST_CKPT")"

echo ">>> 4) 本机 -> $ZX1:$DST_DATA"
rsync "${RSYNC_OPTS[@]}" -e ssh "$TMP/data/" "$ZX1:$DST_DATA/"

echo ">>> 5) 本机 -> $ZX1:$DST_CKPT"
rsync "${RSYNC_OPTS[@]}" -e ssh "$TMP/ckpt/" "$ZX1:$DST_CKPT/"

rm -rf "$TMP"
echo "完成。"

echo ""
echo "之后在 zx1 上请统一用 zx1 路径，例如："
echo "  --robot-repo $DST_DATA"
echo "  --teleop-checkpoint $DST_CKPT"
echo "并在 src/openpi/training/config.py 里把 pi05_dual_franka_* 的 repo_id / assets 改成 zx1 下的目录（若你希望默认训练也指向 zx1）。"
