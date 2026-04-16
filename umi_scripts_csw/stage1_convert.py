#!/usr/bin/env python3
"""Unified Stage1 launcher: rosbag -> mp4/json."""

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


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.views in (1, 2):
        if not args.bag or not args.serial:
            parser.error("--views 1/2 require --bag and --serial")
    else:
        if not args.bag_dir or args.start_idx is None:
            parser.error("--views 3 requires --bag_dir and --start_idx")


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified stage1 converter launcher")
    parser.add_argument("--views", type=int, choices=[1, 2, 3], required=True, help="Number of camera views")
    parser.add_argument("--out_dir", required=True, help="Output directory")

    # Single-bag mode (views=1/2)
    parser.add_argument("--bag", default=None, help="Input rosbag path for views=1/2")
    parser.add_argument("--serial", default=None, help="XV serial for views=1/2")
    parser.add_argument("--data_idx", type=int, default=None, help="Episode index for views=1/2")
    parser.add_argument("--head_topic", default="/camera/color/image_raw/compressed", help="Head topic for views=2")

    # Batch mode (views=3)
    parser.add_argument("--bag_dir", default=None, help="Bag directory for views=3")
    parser.add_argument("--start_idx", type=int, default=None, help="Start index for views=3")
    parser.add_argument("--pattern", default="*.bag", help="Bag glob pattern for views=3")

    args = parser.parse_args()
    _validate_args(args, parser)

    scripts_dir = _scripts_dir()
    py = sys.executable

    if args.views == 1:
        script = scripts_dir / "convert_ros_data_to_mp4.py"
        data_idx = args.data_idx if args.data_idx is not None else 1
        cmd = [
            py,
            str(script),
            "--bag",
            args.bag,
            "--serial",
            args.serial,
            "--out_dir",
            args.out_dir,
            "--data_idx",
            str(data_idx),
        ]
        return _run(cmd)

    if args.views == 2:
        script = scripts_dir / "convert_rosbag_to_mp4_vis_13.py"
        data_idx = args.data_idx if args.data_idx is not None else 1
        cmd = [
            py,
            str(script),
            "--bag",
            args.bag,
            "--serial",
            args.serial,
            "--out_dir",
            args.out_dir,
            "--data_idx",
            str(data_idx),
            "--head_topic",
            args.head_topic,
        ]
        return _run(cmd)

    script = scripts_dir / "convert_rosbag_to_mp4_vis_123.py"
    cmd = [
        py,
        str(script),
        "--bag_dir",
        args.bag_dir,
        "--out_dir",
        args.out_dir,
        "--start_idx",
        str(args.start_idx),
        "--pattern",
        args.pattern,
    ]
    return _run(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
