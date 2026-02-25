#!/bin/bash
# Replay recorded demo episodes on the robot (offline ABS->DELTA)
#
# Usage:
#   ./replay_data.sh <task_name> <episode>
#
# Examples:
#   ./replay_data.sh pick-place-0 1
#   FILTERED=true ./replay_data.sh pick-place-0 1
#   DRY_RUN=true SPEED=0.5 ./replay_data.sh pick-place-0 1
#
# Environment variables:
#   EXP_NAME      - Experiment name from experiments/ directory (default: souhu_droid_base)
#   DATA_DIR      - Directory containing parquet files (default: demo_data)
#   FILTERED      - Use filtered episode (default: false)
#   DRY_RUN       - Dry run without robot (default: false)
#   SKIP_RESET    - Skip initial reset (default: false)
#   SPEED         - Speed factor for replay (default: 1.0)
#   CONFIRM       - Confirm each step (default: false)

set -e

# Resolve paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Required arguments
TASK_NAME="${1:?Usage: $0 <task_name> <episode>}"
EPISODE="${2:?Usage: $0 <task_name> <episode>}"

# Defaults (can be overridden by env vars)
EXP_NAME="${EXP_NAME:-souhu_fastumi_base}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/demo_data}"
FILTERED="${FILTERED:-false}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_RESET="${SKIP_RESET:-false}"
SPEED="${SPEED:-1.0}"
CONFIRM="${CONFIRM:-false}"

echo "=================================================="
echo "Episode Replay (offline ABS -> DELTA)"
echo "=================================================="
echo "Experiment       : $EXP_NAME"
echo "Task             : $TASK_NAME"
echo "Episode          : $EPISODE"
echo "Data directory   : $DATA_DIR"
echo "Use filtered     : $FILTERED"
echo "Dry run          : $DRY_RUN"
echo "Skip reset       : $SKIP_RESET"
echo "Speed factor     : $SPEED"
echo "Confirm per step : $CONFIRM"
echo "=================================================="
echo ""

# Ensure correct working directory and PYTHONPATH
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Base args
ARGS=(
    --exp_name="$EXP_NAME"
    --task_name="$TASK_NAME"
    --episode="$EPISODE"
    --data_dir="$DATA_DIR"
    --speed_factor="$SPEED"
)

# Optional flags
[[ "$FILTERED" == "true" ]]   && ARGS+=(--use_filtered)
[[ "$DRY_RUN" == "true" ]]    && ARGS+=(--dry_run)
[[ "$SKIP_RESET" == "true" ]] && ARGS+=(--skip_reset)
[[ "$CONFIRM" == "true" ]]    && ARGS+=(--confirm_each_step)

# Run
"$PROJECT_ROOT/.venv/bin/python" umi_scripts_csw/replay_data_fastumi.py "${ARGS[@]}"
