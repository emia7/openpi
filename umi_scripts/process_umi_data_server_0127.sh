#!/bin/bash
# [Server Machine] Reconstruct LeRobot dataset from uploaded UMI lightweight data (MP4 + JSON).
#
# Usage: ./umi_scripts/process_umi_data_server.sh
#
# Environment variables:
#   INPUT_DIR         - Directory containing uploaded mp4/json files (default: umi_ready_to_upload)
#   REPO_ID           - LeRobot repo ID (default: local/umi_demo)
#   OUTPUT_DIR        - LeRobot output directory (default: INPUT_DIR/lerobot_dataset)
#   TARGET_FPS        - Target FPS for downsampling (default: 10)
#   TASK_TEXT         - Language instruction for the task (default: "do the task")

set -e

# 获取脚本所在目录的上一级目录作为项目根目录 (假设脚本在 umi_scripts/ 下)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
INPUT_DIR="${INPUT_DIR:-$PROJECT_ROOT/umi_ready_to_upload}"
REPO_ID="${REPO_ID:-local/umi_demo}"
# 如果没指定 OUTPUT_DIR，默认放在 INPUT_DIR 下面的 lerobot_dataset 文件夹
OUTPUT_DIR="${OUTPUT_DIR:-$INPUT_DIR/lerobot_dataset}"
TARGET_FPS="${TARGET_FPS:-10}"
TASK_TEXT="${TASK_TEXT:-do the task}"

echo "========================================"
echo "UMI Server Data Reconstruction Config"
echo "========================================"
echo "Input Directory: $INPUT_DIR"
echo "Repo ID: $REPO_ID"
echo "Output Directory: $OUTPUT_DIR"
echo "Target FPS: $TARGET_FPS"
echo "Task Text: $TASK_TEXT"
echo "========================================"
echo ""

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

ARGS=(
    --input_dir="$INPUT_DIR"
    --repo_id="$REPO_ID"
    --output_dir="$OUTPUT_DIR"
    --target_fps="$TARGET_FPS"
    --task_text="$TASK_TEXT"
)

# 执行 Python 脚本
# 注意：这里指向 umi_scripts 目录下的脚本
"$PROJECT_ROOT/.venv/bin/python" umi_scripts/process_umi_data_server.py "${ARGS[@]}"