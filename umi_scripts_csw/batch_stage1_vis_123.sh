#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./convert_all.sh <bag_dir> <out_dir> <start_idx> <py_script>
#
# Example:
#   ./convert_all.sh /home/ubuntu/qiuyi/FastUMI/data/bags \
#                    /home/ubuntu/qiuyi/FastUMI/data/converted \
#                    100 \
#                    /home/ubuntu/qiuyi/FastUMI/tools/bag_folder_to_mp4_json.py

BAG_DIR="${1:?bag_dir required}"
OUT_DIR="${2:?out_dir required}"
START_IDX="${3:?start_idx required}"
PY_SCRIPT="${4:?python script required}"

mkdir -p "${OUT_DIR}"

python3 "${PY_SCRIPT}" \
  --bag_dir "${BAG_DIR}" \
  --out_dir "${OUT_DIR}" \
  --start_idx "${START_IDX}"
