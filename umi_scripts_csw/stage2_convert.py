#!/usr/bin/env python3
"""Unified Stage2 launcher: mp4/json -> LeRobot.

This script dispatches to existing stage2 converters to preserve behavior.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _scripts_dir() -> Path:
    return Path(__file__).resolve().parent


def _run(cmd: list[str]) -> int:
    print("Running:", " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified stage2 converter launcher")
    parser.add_argument("--views", type=int, choices=[1, 2, 3], required=True, help="Number of camera views")
    parser.add_argument("--stage1_dir", required=True, help="Stage1 output directory")
    parser.add_argument("--repo", required=True, help="LeRobot repo id/name")
    parser.add_argument("--robot_type", default=None, help="Override robot_type")
    parser.add_argument("--task", default="put the purple cup into the plate", help="Task text")
    parser.add_argument("--target_fps", type=float, default=10.0, help="Target fps for views=1/2")
    parser.add_argument("--fps", type=int, default=0, help="Override dataset fps for views=3")
    args = parser.parse_args()

    scripts_dir = _scripts_dir()
    py = sys.executable

    if args.views == 1:
        script = scripts_dir / "convert_mp4_data_to_lerobot_downsample.py"
        cmd = [
            py,
            str(script),
            "--stage1_dir",
            args.stage1_dir,
            "--repo",
            args.repo,
            "--robot_type",
            args.robot_type or "XV",
            "--target_fps",
            str(args.target_fps),
            "--task",
            args.task,
        ]
        return _run(cmd)

    if args.views == 2:
        script = scripts_dir / "convert_mp4_data_to_lerobot_downsample_13.py"
        cmd = [
            py,
            str(script),
            "--stage1_dir",
            args.stage1_dir,
            "--repo",
            args.repo,
            "--robot_type",
            args.robot_type or "XV",
            "--target_fps",
            str(args.target_fps),
            "--task",
            args.task,
        ]
        return _run(cmd)

    script = scripts_dir / "convert_mp4_data_to_lerobot_123.py"
    cmd = [
        py,
        str(script),
        "--stage1_dir",
        args.stage1_dir,
        "--repo",
        args.repo,
        "--robot_type",
        args.robot_type or "XV_DUAL",
        "--task",
        args.task,
    ]
    if args.fps > 0:
        cmd.extend(["--fps", str(args.fps)])
    return _run(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
