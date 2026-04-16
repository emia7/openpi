#!/usr/bin/env python3
"""Unified Stage2 launcher: mp4/json -> LeRobot.

This script is the single Stage2 entrypoint. It directly invokes per-view
conversion implementations in-process.
"""

from __future__ import annotations

import argparse


def run_stage2(
    *,
    views: int,
    stage1_dir: str,
    repo: str,
    robot_type: str | None,
    task: str,
    target_fps: float,
    fps: int,
) -> int:
    if views == 1:
        import convert_mp4_data_to_lerobot_downsample as stage2_v1

        stage2_v1.main(
            stage1_dir=stage1_dir,
            repo_name=repo,
            robot_type=robot_type or "XV",
            target_fps=target_fps,
            task_text=task,
        )
        return 0

    if views == 2:
        import convert_mp4_data_to_lerobot_downsample_13 as stage2_v2

        stage2_v2.main(
            stage1_dir=stage1_dir,
            repo_name=repo,
            robot_type=robot_type or "XV",
            target_fps=target_fps,
            task_text=task,
        )
        return 0

    import convert_mp4_data_to_lerobot_123 as stage2_v3

    stage2_v3.main(
        stage1_dir=stage1_dir,
        repo=repo,
        robot_type=robot_type or "XV_DUAL",
        task=task,
        fps_override=fps if fps > 0 else 0,
    )
    return 0


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

    return run_stage2(
        views=args.views,
        stage1_dir=args.stage1_dir,
        repo=args.repo,
        robot_type=args.robot_type,
        task=args.task,
        target_fps=args.target_fps,
        fps=args.fps,
    )


if __name__ == "__main__":
    raise SystemExit(main())
