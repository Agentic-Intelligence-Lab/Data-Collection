#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


PYTHON_MODULES = [
    "yaml",
    "numpy",
    "PIL",
    "torch",
    "datasets",
    "pyarrow",
    "av",
    "torchvision",
    "huggingface_hub",
    "jsonlines",
    "teleoperation",
]

HARDWARE_MODULES = [
    "pyrealsense2",
    "piper_sdk",
]

SYSTEM_COMMANDS = [
    "ffmpeg",
    "ffprobe",
    "rs-enumerate-devices",
    "ip",
    "ethtool",
    "candump",
]


def _check_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Project: {PROJECT_ROOT}")

    failed = []
    print("\nPython modules:")
    for module_name in PYTHON_MODULES:
        ok = _check_module(module_name)
        print(f"  {'ok' if ok else 'missing'}  {module_name}")
        if not ok:
            failed.append(module_name)

    print("\nHardware-specific Python modules:")
    for module_name in HARDWARE_MODULES:
        ok = _check_module(module_name)
        print(f"  {'ok' if ok else 'missing'}  {module_name}")

    print("\nSystem commands:")
    for command in SYSTEM_COMMANDS:
        path = shutil.which(command)
        print(f"  {'ok' if path else 'missing'}  {command}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

