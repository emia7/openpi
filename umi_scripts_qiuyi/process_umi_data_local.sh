#!/usr/bin/env bash
set -euo pipefail

# ====== EDIT THESE ======
BAG_DIR="/home/ubuntu/qiuyi/fastumi_data/rosbag/pick-place-cubes-0"     # 放所有 .bag 的目录
OUT_DIR="/home/ubuntu/qiuyi/fastumi_data/batch_data_0125_vis"        # 输出目录：episodeXXXXX.mp4/json
PY_SCRIPT="./convert_rosbag_to_mp4_vis.py"                   # 写死脚本名

# 候选 serial（按优先级从高到低填）
SERIALS=(
  "250801DR48FP25002268"
  "250801DR48FP25002565"
)

# 可选：并行数（=1 就串行）
JOBS="${JOBS:-1}"

mkdir -p "$OUT_DIR"

mapfile -t BAGS < <(find "$BAG_DIR" -maxdepth 1 -type f -name "*.bag" | sort)
if [[ ${#BAGS[@]} -eq 0 ]]; then
  echo "[ERROR] No .bag files found in: $BAG_DIR" >&2
  exit 1
fi

echo "[INFO] Found ${#BAGS[@]} bag files."

run_one () {
  local bag="$1"
  local idx="$2"
  local base mp4 jsn chosen serial_file
  base="$(basename "$bag")"
  mp4="$OUT_DIR/episode${idx}.mp4"
  jsn="$OUT_DIR/episode${idx}.json"
  serial_file="$OUT_DIR/episode${idx}.serial.txt"

  # 跳过已生成的
  if [[ -f "$mp4" && -f "$jsn" && -f "$serial_file" ]]; then
    echo "[SKIP] idx=$idx $base (serial=$(cat "$serial_file" 2>/dev/null || true))"
    return 0
  fi

  echo "[RUN ] idx=$idx $base (auto-pick serial)"

  # 清理可能的半成品（避免误判）
  rm -f "$mp4" "$jsn" "$serial_file"

  chosen=""
  for s in "${SERIALS[@]}"; do
    local log="$OUT_DIR/stage1_${idx}_try_${s}.log"
    echo "[TRY ] idx=$idx $base serial=$s" | tee "$log" >/dev/null

    # 用 subshell 执行，避免 set -e 直接把整个脚本带走
    if python3 "$PY_SCRIPT" \
        --bag "$bag" \
        --serial "$s" \
        --out_dir "$OUT_DIR" \
        --data_idx "$idx" \
        >> "$log" 2>&1; then

      # 判断是否真的产出了文件（避免脚本“返回0但没产出”的情况）
      if [[ -f "$mp4" && -f "$jsn" ]]; then
        chosen="$s"
        echo "$chosen" > "$serial_file"
        echo "[OK  ] idx=$idx $base serial=$chosen"
        break
      else
        echo "[WARN] idx=$idx produced no mp4/json for serial=$s" >> "$log"
        rm -f "$mp4" "$jsn"
      fi
    else
      # 常见：serial 不匹配导致 poses 为空/脚本报错
      rm -f "$mp4" "$jsn"
      echo "[FAIL] idx=$idx $base serial=$s" >> "$log"
    fi
  done

  if [[ -z "$chosen" ]]; then
    echo "[ERROR] idx=$idx $base: no serial worked. See logs: $OUT_DIR/stage1_${idx}_try_*.log" >&2
    return 1
  fi

  echo "[DONE] idx=$idx $base (serial=$chosen)"
}

export -f run_one
export OUT_DIR PY_SCRIPT
# 让子进程也能拿到 SERIALS（bash 数组需要特殊处理：用字符串传递再 eval）
export SERIALS_STR="$(printf "%q " "${SERIALS[@]}")"

# 让 run_one 在子进程里也能访问 SERIALS
# （xargs 并行会起新的 bash -c）
run_one_wrapper () {
  # 重建 SERIALS 数组
  eval "SERIALS=($SERIALS_STR)"
  run_one "$@"
}
export -f run_one_wrapper

if [[ "$JOBS" -le 1 ]]; then
  i=0
  for b in "${BAGS[@]}"; do
    printf -v idx "%05d" "$i"
    run_one_wrapper "$b" "$idx"
    i=$((i+1))
  done
else
  tmp_list="$(mktemp)"
  i=0
  for b in "${BAGS[@]}"; do
    printf -v idx "%05d" "$i"
    printf "%s\t%s\n" "$b" "$idx" >> "$tmp_list"
    i=$((i+1))
  done

  # 每行一个任务：bag + idx
  cat "$tmp_list" | xargs -P "$JOBS" -n 1 bash -c '
    line="$0"
    bag="$(echo "$line" | cut -f1)"
    idx="$(echo "$line" | cut -f2)"
    run_one_wrapper "$bag" "$idx"
  '

  rm -f "$tmp_list"
fi

echo "[INFO] All done. Output dir: $OUT_DIR"
