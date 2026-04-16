#!/usr/bin/env python3
"""Unified entrypoint for umi_scripts_csw utilities.

This keeps legacy scripts untouched while providing a discoverable command
index for day-to-day usage.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

def _scripts_dir() -> Path:
    return Path(__file__).resolve().parent


def _load_script_groups() -> dict[str, list[str]]:
    index_path = _scripts_dir() / "scripts_index.json"
    with index_path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid index format in {index_path}")
    return {str(key): [str(item) for item in value] for key, value in data.items()}


def _print_index(script_groups: dict[str, list[str]]) -> None:
    print("UMI scripts index\n")
    for group, scripts in script_groups.items():
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
    script_groups = _load_script_groups()

    if parsed.list or not parsed.script:
        _print_index(script_groups)
        return 0

    passthrough = parsed.args
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    return _run_script(parsed.script, passthrough)


if __name__ == "__main__":
    raise SystemExit(main())
