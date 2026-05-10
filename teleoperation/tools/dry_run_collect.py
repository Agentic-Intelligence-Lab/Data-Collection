#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from teleoperation.recorder.config import load_runtime_config


CORE_IMPORTS = [
    "teleoperation.gui.collect_data_gui",
    "teleoperation.dataset.lerobot_dataset",
    "teleoperation.dataset.validate_realtime_episode",
    "teleoperation.robot.piper",
    "teleoperation.cameras.realsense",
]


def _import_modules() -> dict[str, str]:
    result = {}
    for module_name in CORE_IMPORTS:
        importlib.import_module(module_name)
        result[module_name] = "ok"
    return result


def _detect_cameras() -> dict:
    if importlib.util.find_spec("pyrealsense2") is None:
        return {
            "status": "skipped",
            "reason": "pyrealsense2 is not importable in this Python environment.",
            "detected": [],
        }

    from teleoperation.cameras.realsense import find_cameras

    detected = find_cameras(raise_when_empty=False)
    return {
        "status": "ok",
        "detected": detected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dry-run the Piper teleoperation collector without activating CAN or connecting robot arms."
    )
    parser.add_argument("--config", default=PROJECT_ROOT / "configs" / "piper_4cam.yaml", type=Path)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    runtime_config = load_runtime_config(args.config, PROJECT_ROOT)
    import_result = _import_modules()
    camera_result = _detect_cameras()

    expected_serials = {
        key: cfg["serial_number"]
        for key, cfg in runtime_config["cameras"].items()
    }
    detected_serials = {
        int(item["serial_number"])
        for item in camera_result.get("detected", [])
    }
    missing_cameras = {
        key: serial
        for key, serial in expected_serials.items()
        if camera_result["status"] == "ok" and serial not in detected_serials
    }

    report = {
        "config": str(args.config.resolve()),
        "dataset_root": runtime_config["dataset_root"],
        "camera_keys": list(runtime_config["cameras"]),
        "expected_can_interfaces": runtime_config["expected_can_interfaces"],
        "imports": import_result,
        "camera_detection": camera_result,
        "missing_configured_cameras": missing_cameras,
        "robot_connection": "not attempted",
        "can_activation": "not attempted",
    }

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    print("Dry-run passed for config and core imports.")
    print(f"Config: {report['config']}")
    print(f"Dataset root: {report['dataset_root']}")
    print(f"Camera keys: {', '.join(report['camera_keys'])}")
    print(f"Expected CAN interfaces: {' '.join(report['expected_can_interfaces'])}")
    if camera_result["status"] == "skipped":
        print(f"Camera detection skipped: {camera_result['reason']}")
    else:
        print(f"Detected RealSense cameras: {len(camera_result['detected'])}")
        if missing_cameras:
            print(f"Missing configured cameras: {missing_cameras}")
    print("No CAN activation, Piper SDK connection, or robot motion was attempted.")


if __name__ == "__main__":
    main()

