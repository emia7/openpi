#!/usr/bin/env python3
"""Unified Stage1 launcher: rosbag -> mp4/json."""

from __future__ import annotations

import argparse
import concurrent.futures
from pathlib import Path

import stage1_core


def _scripts_dir() -> Path:
    return Path(__file__).resolve().parent


def _run_single(
    args: argparse.Namespace,
    bag: str,
    serial: str,
    data_idx: str,
    log_path: Path | None = None,
) -> int:
    if log_path is None:
        if args.views == 2:
            stage1_core.convert_dual_view_bag(bag, serial, args.out_dir, data_idx, args.head_topic)
        else:
            stage1_core.convert_single_view_bag(bag, serial, args.out_dir, data_idx, with_plot=args.mode == "vis")
        return 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        import contextlib
        import traceback

        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            try:
                if args.views == 2:
                    stage1_core.convert_dual_view_bag(bag, serial, args.out_dir, data_idx, args.head_topic)
                else:
                    stage1_core.convert_single_view_bag(
                        bag,
                        serial,
                        args.out_dir,
                        data_idx,
                        with_plot=args.mode == "vis",
                    )
                return 0
            except Exception:
                traceback.print_exc()
                return 1


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.views in (1, 2):
        if not args.bag and not args.bag_dir:
            parser.error("--views 1/2 require --bag or --bag_dir")
        if args.bag and args.bag_dir:
            parser.error("Use either --bag or --bag_dir for --views 1/2")
        if args.bag and not args.serial:
            parser.error("--views 1/2 single mode requires --serial")
    else:
        if not args.bag_dir or args.start_idx is None:
            parser.error("--views 3 requires --bag_dir and --start_idx")
    if args.views != 1 and args.mode != "plain":
        parser.error("--mode is only supported for --views 1")


def _format_batch_data_idx(views: int, idx: int) -> str:
    # Keep legacy naming conventions used by old batch shell scripts.
    if views == 1:
        return f"{idx:05d}"
    return f"{idx:04d}"


def _expected_outputs(views: int, out_dir: Path, data_idx: str) -> list[Path]:
    if views == 1:
        return [out_dir / f"episode{data_idx}.mp4", out_dir / f"episode{data_idx}.json"]
    return [out_dir / f"episode{data_idx}_head.mp4", out_dir / f"episode{data_idx}_left.mp4", out_dir / f"episode{data_idx}.json"]


def _parse_serials(args: argparse.Namespace) -> list[str]:
    serials: list[str] = []
    if args.serial:
        serials.append(args.serial)
    if args.serials:
        serials.extend([x.strip() for x in args.serials.split(",") if x.strip()])
    # Preserve order while deduplicating.
    unique: list[str] = []
    for serial in serials:
        if serial not in unique:
            unique.append(serial)
    return unique


def _run_batch_views12(args: argparse.Namespace) -> int:
    if args.start_idx is None:
        args.start_idx = 0
    serials = _parse_serials(args)
    if not serials:
        raise SystemExit("Batch mode for views 1/2 requires --serial or --serials.")

    bag_dir = Path(args.bag_dir)
    bags = sorted(bag_dir.glob(args.pattern))
    if not bags:
        raise SystemExit(f"No bags found in {bag_dir} with pattern {args.pattern}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Found {len(bags)} bag files. start_idx={args.start_idx}")

    indexed_bags = [(args.start_idx + i, bag) for i, bag in enumerate(bags)]

    def run_one(item: tuple[int, Path]) -> tuple[int, str]:
        idx, bag = item
        data_idx = _format_batch_data_idx(args.views, idx)
        outputs = _expected_outputs(args.views, out_dir, data_idx)
        serial_file = out_dir / f"episode{idx}.serial.txt"
        if args.skip_existing and all(path.exists() for path in outputs) and serial_file.exists():
            return idx, f"[SKIP] idx={idx} {bag.name}"

        chosen = None
        for serial in serials:
            for path in outputs:
                if path.exists():
                    path.unlink()
            if serial_file.exists():
                serial_file.unlink()
            log_path = out_dir / f"stage1_{idx}_try_{serial}.log"
            code = _run_single(args, str(bag), serial, data_idx, log_path=log_path)
            if code == 0 and all(path.exists() for path in outputs):
                chosen = serial
                serial_file.write_text(serial, encoding="utf-8")
                return idx, f"[OK] idx={idx} {bag.name} serial={serial}"
            print(f"[FAIL] idx={idx} {bag.name} serial={serial} (see {log_path.name})")

        return idx, f"[ERROR] idx={idx} {bag.name} all serial attempts failed"

    failures = 0
    completed: list[tuple[int, str]] = []
    max_workers = max(1, args.jobs)
    if max_workers == 1:
        for item in indexed_bags:
            completed.append(run_one(item))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_one, item) for item in indexed_bags]
            for future in concurrent.futures.as_completed(futures):
                completed.append(future.result())

    for _, msg in sorted(completed, key=lambda x: x[0]):
        print(msg)
        if msg.startswith("[ERROR]"):
            failures += 1
            if not args.continue_on_error:
                raise SystemExit("Batch aborted because --continue_on_error is not set.")

    print(f"[INFO] Batch summary: total={len(indexed_bags)} failures={failures} jobs={max_workers}")
    if failures > 0:
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified stage1 converter launcher")
    parser.add_argument("--views", type=int, choices=[1, 2, 3], required=True, help="Number of camera views")
    parser.add_argument("--mode", choices=["plain", "vis"], default="plain", help="Conversion mode for views=1")
    parser.add_argument("--out_dir", required=True, help="Output directory")

    # Single-bag mode (views=1/2)
    parser.add_argument("--bag", default=None, help="Input rosbag path for views=1/2")
    parser.add_argument("--serial", default=None, help="XV serial for views=1/2")
    parser.add_argument("--serials", default=None, help="Comma-separated serial candidates for batch views=1/2")
    parser.add_argument("--data_idx", default=None, help="Episode index for views=1/2 (string allowed, e.g. 0001)")
    parser.add_argument("--head_topic", default="/camera/color/image_raw/compressed", help="Head topic for views=2")

    # Batch mode (views=3)
    parser.add_argument("--bag_dir", default=None, help="Bag directory for views=3")
    parser.add_argument("--start_idx", type=int, default=None, help="Start index for views=3")
    parser.add_argument("--pattern", default="*.bag", help="Bag glob pattern for views=3")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel workers for views=1/2 batch mode")
    parser.add_argument("--skip_existing", action="store_true", help="Skip batch items with existing outputs")
    parser.add_argument("--continue_on_error", action="store_true", help="Continue batch when one bag fails")

    args = parser.parse_args()
    _validate_args(args, parser)

    if args.views == 1:
        if args.bag_dir:
            return _run_batch_views12(args)
        data_idx = args.data_idx if args.data_idx is not None else "1"
        return _run_single(args, args.bag, args.serial, str(data_idx))

    if args.views == 2:
        if args.bag_dir:
            return _run_batch_views12(args)
        data_idx = args.data_idx if args.data_idx is not None else "1"
        return _run_single(args, args.bag, args.serial, str(data_idx))

    bag_dir = Path(args.bag_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bags = sorted(bag_dir.glob(args.pattern))
    if not bags:
        raise SystemExit(f"No bags found in {bag_dir} with pattern {args.pattern}")
    idx = int(args.start_idx)
    for bag_path in bags:
        stage1_core.convert_three_view_bag(bag_path, out_dir, idx)
        idx += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
