#!/bin/bash
# [Server Machine] Reconstruct LeRobot dataset from uploaded lightweight data.
#
# Usage: ./process_data_server.sh
#
# Environment variables:
#   INPUT_DIR         - Directory containing uploaded data (default: ready_to_upload)
#   REPO_ID           - LeRobot repo ID (default: local/franka_pick_place)
#   OUTPUT_DIR        - LeRobot output directory (default: INPUT_DIR/lerobot_dataset)
#   SKIP_FAILED       - Skip failed episodes in LeRobot export (default: false)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
INPUT_DIR="${INPUT_DIR:-$PROJECT_ROOT/ready_to_upload}"
REPO_ID="${REPO_ID:-local/franka_pick_place}"
# 如果没指定 OUTPUT_DIR，默认放在 INPUT_DIR 下面的 lerobot_dataset 文件夹
OUTPUT_DIR="${OUTPUT_DIR:-$INPUT_DIR/lerobot_dataset}"
SKIP_FAILED="${SKIP_FAILED:-false}"

echo "========================================"
echo "Server Data Reconstruction Configuration"
echo "========================================"
echo "Input Directory: $INPUT_DIR"
echo "Repo ID: $REPO_ID"
echo "Output Directory: $OUTPUT_DIR"
echo "Skip Failed Episodes: $SKIP_FAILED"
echo "========================================"
echo ""

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

ARGS=(
    --input_dir="$INPUT_DIR"
    --repo_id="$REPO_ID"
    --output_dir="$OUTPUT_DIR"
)

[[ "$SKIP_FAILED" == "true" ]] && ARGS+=(--skip_failed) || ARGS+=(--noskip_failed)

# 执行 Python 脚本
"$PROJECT_ROOT/.venv/bin/python" scripts/process_server.py "${ARGS[@]}"