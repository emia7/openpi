#!/usr/bin/env python3
"""Unified entrypoint for umi_scripts_csw utilities.

This keeps legacy scripts untouched while providing a discoverable command
index for day-to-day usage.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_GROUPS: dict[str, list[str]] = {
    "stage1_ros_to_mp4_json": [
        "convert_ros_data_to_mp4.py",
        "convert_rosbag_to_mp4_vis.py",
        "convert_rosbag_to_mp4_vis_13.py",
        "convert_rosbag_to_mp4_vis_123.py",
    ],
    "stage2_mp4_json_to_lerobot": [
        "convert_mp4_data_to_lerobot_downsample.py",
        "convert_mp4_data_to_lerobot_downsample_13.py",
        "convert_mp4_data_to_lerobot_123.py",
    ],
    "dataset_tools": [
        "json_sort.py",
        "json_sort_13.py",
        "json_sort_123.py",
        "json_visualize.py",
        "render_triad_mp4.py",
        "transform_pose.py",
        "check_dataset_actions.py",
        "compare_batches.py",
        "compare_npz.py",
    ],
    "evaluation": [
        "eval_actions.py",
        "eval_relative.py",
        "eval_dual_relative.py",
        "eval_relative_visualize.py",
    ],
    "runtime_helpers": [
        "tri_image_sampler_10hz.py",
        "replay_data_fastumi.py",
        "replay_data_fastumi.sh",
        "fastumi_helper.py",
        "start_bringup.sh",
    ],
}


def _scripts_dir() -> Path:
    return Path(__file__).resolve().parent


def _print_index() -> None:
    print("UMI scripts index\n")
    for group, scripts in SCRIPT_GROUPS.items():
        print(f"[{group}]")
        for script in scripts:
            print(f"  - {script}")
        print()
    print("Use --script <name> [-- <args...>] to execute one script.")


def _validate_script(script_name: str) -> Path:
    scripts_dir = _scripts_dir()
    script_path = scripts_dir / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    return script_path


def _run_script(script_name: str, passthrough: list[str]) -> int:
    script_path = _validate_script(script_name)

    if script_path.suffix == ".py":
        cmd = [sys.executable, str(script_path), *passthrough]
    else:
        cmd = ["bash", str(script_path), *passthrough]

    print("Running:", " ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified launcher for umi_scripts_csw")
    parser.add_argument("--list", action="store_true", help="List scripts by functional group")
    parser.add_argument("--script", type=str, help="Script file name to execute")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Args passed to target script")
    parsed = parser.parse_args()

    if parsed.list or not parsed.script:
        _print_index()
        return 0

    passthrough = parsed.args
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    return _run_script(parsed.script, passthrough)


if __name__ == "__main__":
    raise SystemExit(main())
