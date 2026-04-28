#!/bin/bash
# ArUco detection script for 0424 data

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE1_DIR="$HOME/Downloads/handover_umi_0424"
OUTPUT_DIR="$HOME/Downloads/handover_umi_0424_aruco"
LOG_FILE="$SCRIPT_DIR/aruco_detection_0424.log"

echo "=========================================="
echo "ArUco Detection for handover_umi_0424"
echo "=========================================="
echo "Input:  $STAGE1_DIR"
echo "Output: $OUTPUT_DIR"
echo "Log:    $LOG_FILE"
echo "=========================================="

# Check if input exists
if [ ! -d "$STAGE1_DIR" ]; then
    echo "ERROR: Input directory not found: $STAGE1_DIR"
    exit 1
fi

# Count episodes
EPISODE_COUNT=$(ls "$STAGE1_DIR"/episode_*_left.json 2>/dev/null | wc -l)
echo "Found $EPISODE_COUNT episodes to process"
echo ""

# Run detection
cd "$SCRIPT_DIR"
python3 umi_scripts_csw/detect_aruco_from_video.py \
    --stage1_dir "$STAGE1_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --max_frames 100 2>&1 | tee "$LOG_FILE"

echo ""
echo "=========================================="
echo "Detection complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "Log saved to: $LOG_FILE"
echo "=========================================="
