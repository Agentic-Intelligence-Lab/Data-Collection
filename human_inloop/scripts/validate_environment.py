#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPENPI_ROOT = PROJECT_ROOT / "external" / "openpi"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ[key] = value


def env_value(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip() != "":
            return value
    return default


def resolve_path(value: str | None) -> Path | None:
    if value is None or value.strip() == "":
        return None
    path = Path(os.path.expandvars(os.path.expanduser(value))).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


class Reporter:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def pass_(self, message: str) -> None:
        print(f"PASS {message}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"WARN {message}")

    def fail(self, message: str) -> None:
        self.failures.append(message)
        print(f"FAIL {message}")

    def maybe_fail(self, message: str, *, strict: bool) -> None:
        if strict:
            self.fail(message)
        else:
            self.warn(message)


def check_python(reporter: Reporter) -> None:
    version = sys.version_info
    if version[:2] == (3, 11):
        reporter.pass_(f"Python version is {version.major}.{version.minor}.{version.micro}")
    else:
        reporter.fail(
            f"Python version is {version.major}.{version.minor}.{version.micro}; OpenPI runtime expects Python 3.11. "
            "Run with python3.11 or set OPENPI_PYTHON_BIN for the launch script."
        )


def check_imports(reporter: Reporter, *, strict_hardware: bool) -> None:
    required = {
        "numpy": "numpy",
        "torch": "torch",
        "cv2": "opencv-python-headless / opencv-python",
        "datasets": "datasets",
        "huggingface_hub": "huggingface-hub",
    }
    hardware = {
        "pyrealsense2": "pyrealsense2",
        "can": "python-can",
        "piper_sdk": "piper_sdk",
    }
    for module, package in required.items():
        if importlib.util.find_spec(module) is None:
            reporter.fail(f"Cannot import {module}; install {package} from requirements-collection.txt.")
        else:
            reporter.pass_(f"Import available: {module}")
    for module, package in hardware.items():
        if importlib.util.find_spec(module) is None:
            reporter.maybe_fail(
                f"Cannot import {module}; install {package} for robot/camera collection.",
                strict=strict_hardware,
            )
        else:
            reporter.pass_(f"Import available: {module}")


def check_openpi(reporter: Reporter, openpi_root: Path | None) -> None:
    if openpi_root is None:
        reporter.fail("OPENPI_ROOT is not set. Set OPENPI_ROOT or clone the reference checkout to external/openpi.")
        return
    if not openpi_root.exists():
        reporter.fail(f"OPENPI_ROOT does not exist: {openpi_root}. Set OPENPI_ROOT to the OpenPI checkout.")
        return
    reporter.pass_(f"OPENPI_ROOT exists: {openpi_root}")
    for relative in ("src/openpi", "packages/openpi-client/src"):
        candidate = openpi_root / relative
        if candidate.exists():
            reporter.pass_(f"OpenPI source path exists: {relative}")
        else:
            reporter.fail(f"Missing OpenPI source path: {candidate}. OPENPI_ROOT must contain src/openpi and packages/openpi-client/src.")


def check_runtime(reporter: Reporter, robot_runtime_root: Path | None, policy_runtime_script: Path | None) -> None:
    if robot_runtime_root is None:
        robot_runtime_root = PROJECT_ROOT / "vendor" / "lerobot_piper"
    if (robot_runtime_root / "lerobot").exists():
        reporter.pass_(f"Robot runtime package exists: {robot_runtime_root / 'lerobot'}")
    else:
        reporter.fail(f"Missing robot runtime package. Set ROBOT_RUNTIME_ROOT or restore {PROJECT_ROOT / 'vendor' / 'lerobot_piper'}")

    if policy_runtime_script is None:
        policy_runtime_script = PROJECT_ROOT / "runtime" / "piper_openpi_runtime.py"
    if policy_runtime_script.exists():
        reporter.pass_(f"Policy runtime script exists: {policy_runtime_script}")
    else:
        reporter.fail(f"Missing policy runtime script. Set POLICY_RUNTIME_SCRIPT: {policy_runtime_script}")


def check_checkpoint(reporter: Reporter, checkpoint_dir: Path | None, norm_stats_path: Path | None) -> None:
    if checkpoint_dir is None:
        reporter.fail("CHECKPOINT_DIR is not set. Set CHECKPOINT_DIR to the policy checkpoint directory.")
        return
    if not checkpoint_dir.exists():
        reporter.fail(f"CHECKPOINT_DIR does not exist: {checkpoint_dir}")
        return
    reporter.pass_(f"CHECKPOINT_DIR exists: {checkpoint_dir}")

    for name in ("model.safetensors", "metadata.pt"):
        candidate = checkpoint_dir / name
        if candidate.exists():
            reporter.pass_(f"Checkpoint file exists: {name}")
        else:
            reporter.fail(f"Missing checkpoint file: {candidate}")

    if norm_stats_path is not None:
        if norm_stats_path.exists():
            reporter.pass_(f"NORM_STATS_PATH exists: {norm_stats_path}")
        else:
            reporter.fail(f"NORM_STATS_PATH does not exist: {norm_stats_path}")
        return

    matches = sorted(checkpoint_dir.glob("assets/**/norm_stats.json"))
    if matches:
        reporter.pass_(f"Found checkpoint norm stats: {matches[0]}")
    else:
        reporter.fail(
            "Could not infer norm_stats.json under CHECKPOINT_DIR/assets. "
            "Set NORM_STATS_PATH if the stats live outside the checkpoint directory."
        )


def check_can(reporter: Reporter, can_name: str, *, strict_hardware: bool) -> None:
    interface_path = Path("/sys/class/net") / can_name
    if interface_path.exists():
        reporter.pass_(f"CAN/network interface exists: {can_name}")
    else:
        reporter.maybe_fail(
            f"CAN/network interface not found: {can_name}. Bring it up or set PIPER_CAN_NAME.",
            strict=strict_hardware,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the local collection environment without moving the robot.")
    parser.add_argument("--skip-hardware", action="store_true", help="Downgrade hardware import/CAN failures to warnings.")
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env.local")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file)
    reporter = Reporter()
    strict_hardware = not args.skip_hardware

    openpi_root = resolve_path(env_value("OPENPI_ROOT", default=str(DEFAULT_OPENPI_ROOT)))
    checkpoint_dir = resolve_path(env_value("CHECKPOINT_DIR"))
    norm_stats_path = resolve_path(env_value("NORM_STATS_PATH"))
    robot_runtime_root = resolve_path(env_value("ROBOT_RUNTIME_ROOT", "PIPER_RUNTIME_ROOT", default=str(PROJECT_ROOT / "vendor" / "lerobot_piper")))
    policy_runtime_script = resolve_path(env_value("POLICY_RUNTIME_SCRIPT", "PIPER_OPENPI_RUNTIME", default=str(PROJECT_ROOT / "runtime" / "piper_openpi_runtime.py")))
    can_name = env_value("PIPER_CAN_NAME", default="can1") or "can1"

    print(f"Project root: {PROJECT_ROOT}")
    check_python(reporter)
    check_runtime(reporter, robot_runtime_root, policy_runtime_script)
    check_openpi(reporter, openpi_root)
    check_checkpoint(reporter, checkpoint_dir, norm_stats_path)
    check_imports(reporter, strict_hardware=strict_hardware)
    check_can(reporter, can_name, strict_hardware=strict_hardware)

    print()
    print(f"Summary: {len(reporter.failures)} fail(s), {len(reporter.warnings)} warning(s)")
    if reporter.failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
