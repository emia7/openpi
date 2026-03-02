#!/usr/bin/env bash
set -euo pipefail

# ====== [配置区域] 请修改这里 ======
# 1. 原始 rosbag 文件夹路径
BAG_DIR="/home/ubuntu/qiuyi/fastumi_data/rosbag/pick-place-cubes-0"

# 2. 输出文件夹路径
OUT_DIR="/home/ubuntu/qiuyi/fastumi_data/batch_dual_cam_vis"

# 3. Python 脚本路径 (即刚才那个带绘图功能的 v3 脚本)
PY_SCRIPT="./convert_rosbag_to_mp4_vis_13.py"

# 4. Head 相机的 Topic (根据你的实际情况修改，例如 compressed)
HEAD_TOPIC="/camera/color/image_raw/compressed"

# 5. 候选 Serial (FastUMI 的序列号，优先尝试第一个)
SERIALS=(
  "250801DR48FP25002268"
  "250801DR48FP25002565"
)

# 6. 并行任务数 (建议设置为 CPU 核心数的一半，以免 IO 爆炸)
JOBS="${JOBS:-1}"

# ====== [新增] 获取起始索引 ======
# 获取第一个参数作为起始 Index，如果未提供则默认为 0
START_IDX="${1:-0}"
# ================================

mkdir -p "$OUT_DIR"

# 查找所有 bag 文件并排序
mapfile -t BAGS < <(find "$BAG_DIR" -maxdepth 1 -type f -name "*.bag" | sort)
if [[ ${#BAGS[@]} -eq 0 ]]; then
  echo "[ERROR] No .bag files found in: $BAG_DIR" >&2
  exit 1
fi

echo "[INFO] Found ${#BAGS[@]} bag files. Starting batch processing from index: $START_IDX"

run_one () {
  local bag="$1"
  local idx="$2" # 传入的是 4 位字符串，如 0001
  local base mp4_head mp4_left jsn chosen serial_file log

  base="$(basename "$bag")"
  
  # 定义输出文件名 (必须与 Python 脚本中的命名逻辑一致)
  mp4_head="$OUT_DIR/episode${idx}_head.mp4"
  mp4_left="$OUT_DIR/episode${idx}_left.mp4"
  jsn="$OUT_DIR/episode${idx}.json"
  
  # 记录该 episode 使用的 serial，用于跳过检查
  serial_file="$OUT_DIR/episode${idx}.serial.txt"

  # [检查] 跳过已生成的 (必须三个主要文件都存在)
  if [[ -f "$mp4_head" && -f "$mp4_left" && -f "$jsn" && -f "$serial_file" ]]; then
    # 仅打印简略信息，避免刷屏
    # echo "[SKIP] idx=$idx $base"
    return 0
  fi

  echo "[RUN ] idx=$idx $base (auto-pick serial)"

  # 清理旧的残余文件
  rm -f "$mp4_head" "$mp4_left" "$jsn" "$serial_file"

  chosen=""
  for s in "${SERIALS[@]}"; do
    log="$OUT_DIR/stage1_${idx}_try_${s}.log"
    # echo "[TRY ] idx=$idx serial=$s" | tee "$log" >/dev/null

    # 执行 Python 脚本
    if python3 "$PY_SCRIPT" \
        --bag "$bag" \
        --serial "$s" \
        --out_dir "$OUT_DIR" \
        --data_idx "$((10#$idx))" \
        --head_topic "$HEAD_TOPIC" \
        >> "$log" 2>&1; then  # 注意：data_idx 转为十进制传给 python

      # [验证] 确认是否真的产出了核心文件
      if [[ -f "$mp4_head" && -f "$mp4_left" && -f "$jsn" ]]; then
        chosen="$s"
        echo "$chosen" > "$serial_file"
        echo "[OK  ] idx=$idx $base serial=$chosen"
        # 删除日志以节省空间（可选）
        rm -f "$log"
        break
      else
        echo "[WARN] idx=$idx script return 0 but files missing for serial=$s" >> "$log"
        rm -f "$mp4_head" "$mp4_left" "$jsn"
      fi
    else
      # 失败清理
      rm -f "$mp4_head" "$mp4_left" "$jsn"
      echo "[FAIL] idx=$idx serial=$s check log: $log"
    fi
  done

  if [[ -z "$chosen" ]]; then
    echo "[ERROR] idx=$idx $base: ALL serials failed." >&2
    return 1
  fi
}

export -f run_one
export OUT_DIR PY_SCRIPT HEAD_TOPIC
# 导出数组供子进程使用
export SERIALS_STR="$(printf "%q " "${SERIALS[@]}")"

run_one_wrapper () {
  # 重建数组
  eval "SERIALS=($SERIALS_STR)"
  run_one "$@"
}
export -f run_one_wrapper

# ---------------------------------------------------------
# 执行调度逻辑
# ---------------------------------------------------------

if [[ "$JOBS" -le 1 ]]; then
  # 串行执行
  i=$START_IDX   # [修改] 使用传入的起始索引
  for b in "${BAGS[@]}"; do
    # 格式化为 4 位数字 (匹配 Python 脚本 f"{int(data_idx):04d}")
    printf -v idx "%04d" "$i"
    run_one_wrapper "$b" "$idx"
    i=$((i+1))
  done
else
  # 并行执行
  tmp_list="$(mktemp)"
  i=$START_IDX   # [修改] 使用传入的起始索引
  for b in "${BAGS[@]}"; do
    printf -v idx "%04d" "$i"
    # 将 bag路径 和 idx 写入临时文件
    printf "%s\t%s\n" "$b" "$idx" >> "$tmp_list"
    i=$((i+1))
  done

  # xargs 并行调用
  cat "$tmp_list" | xargs -P "$JOBS" -n 1 bash -c '
    line="$0"
    bag="$(echo "$line" | cut -f1)"
    idx="$(echo "$line" | cut -f2)"
    run_one_wrapper "$bag" "$idx"
  '

  rm -f "$tmp_list"
fi

echo "[INFO] All done. Output dir: $OUT_DIR"
