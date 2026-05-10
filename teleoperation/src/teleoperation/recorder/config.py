from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

import yaml


VALID_ROTATIONS = {None, "none", -90, 90, 180}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _resolve_path(path_value: str | Path, *, project_root: Path, base_dir: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate.resolve()
    return (project_root / path).resolve()


def _coerce_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean value.")


def _normalize_rotation(value: Any) -> int | None:
    if value == "none":
        value = None
    if value not in VALID_ROTATIONS:
        raise ValueError(f"camera rotation must be one of none, -90, 90, 180 (got {value!r}).")
    return None if value is None else int(value)


def _validate_cameras(cameras: dict[str, dict]) -> dict[str, dict]:
    if not cameras:
        raise ValueError("At least one camera must be configured.")

    normalized = {}
    serials = []
    for key, raw in cameras.items():
        if not key.replace("_", "").isalnum():
            raise ValueError(f"Camera key {key!r} must contain only letters, numbers, and underscores.")
        camera_type = raw.get("type", "intelrealsense")
        if camera_type != "intelrealsense":
            raise ValueError(f"Unsupported camera type for {key}: {camera_type}")
        serial = raw.get("serial_number")
        if serial is None or not str(serial).isdigit():
            raise ValueError(f"Camera {key} requires a numeric RealSense serial_number.")
        serial = int(serial)
        serials.append(serial)
        normalized[key] = {
            "type": "intelrealsense",
            "serial_number": serial,
            "fps": int(raw["fps"]) if raw.get("fps") is not None else None,
            "width": int(raw["width"]) if raw.get("width") is not None else None,
            "height": int(raw["height"]) if raw.get("height") is not None else None,
            "force_hardware_reset": bool(raw.get("force_hardware_reset", False)),
            "rotation": _normalize_rotation(raw.get("rotation")),
        }

    duplicates = sorted({serial for serial in serials if serials.count(serial) > 1})
    if duplicates:
        raise ValueError(f"Duplicate RealSense serial numbers are not allowed: {duplicates}")
    return normalized


def _load_referenced_yaml(
    main_config: dict[str, Any],
    key: str,
    *,
    default_path: str,
    project_root: Path,
    config_dir: Path,
) -> dict[str, Any]:
    path_value = main_config.get(key, default_path)
    path = _resolve_path(path_value, project_root=project_root, base_dir=config_dir)
    return _load_yaml(path)


def load_runtime_config(config_path: Path, project_root: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    project_root = project_root.resolve()
    main_config = _load_yaml(config_path)
    config_dir = config_path.parent

    dataset = main_config.get("dataset", {})
    runtime = main_config.get("runtime", {})
    camera_mapping = _load_referenced_yaml(
        main_config,
        "camera_mapping",
        default_path="camera_mapping.yaml",
        project_root=project_root,
        config_dir=config_dir,
    )
    robot_config = _load_referenced_yaml(
        main_config,
        "robot_config",
        default_path="robot_config.yaml",
        project_root=project_root,
        config_dir=config_dir,
    )

    dataset_name = str(dataset.get("name", "piper_collection"))
    dataset_root_value = dataset.get("root")
    if dataset_root_value is None:
        dataset_root = project_root / "data" / dataset_name
    else:
        dataset_root = _resolve_path(dataset_root_value, project_root=project_root, base_dir=project_root)

    cameras = _validate_cameras(camera_mapping.get("cameras", {}))
    expected_can_interfaces = runtime.get("expected_can_interfaces", ["can0", "can1"])
    if isinstance(expected_can_interfaces, str):
        expected_can_interfaces = expected_can_interfaces.split()

    return {
        "dataset_name": dataset_name,
        "dataset_root": str(dataset_root),
        "single_task": str(dataset.get("task", "piper teleoperation task")),
        "fps": int(dataset.get("fps", 30)),
        "num_episodes": int(dataset.get("num_episodes", 1)),
        "episode_time_s": float(dataset.get("episode_time_s", 10.0)),
        "resume": _coerce_bool(dataset.get("resume", True), name="dataset.resume"),
        "validate_after_record": _coerce_bool(
            dataset.get("validate_after_record", True),
            name="dataset.validate_after_record",
        ),
        "can_settle_time_s": float(runtime.get("can_settle_time_s", 5.0)),
        "expected_can_interfaces": [str(item) for item in expected_can_interfaces],
        "cameras": cameras,
        "robot": robot_config.get("robot", robot_config),
    }


def shell_exports(runtime_config: dict[str, Any]) -> str:
    values = {
        "DATASET_NAME": runtime_config["dataset_name"],
        "DATASET_ROOT": runtime_config["dataset_root"],
        "SINGLE_TASK": runtime_config["single_task"],
        "FPS": str(runtime_config["fps"]),
        "NUM_EPISODES": str(runtime_config["num_episodes"]),
        "EPISODE_TIME_S": str(runtime_config["episode_time_s"]),
        "CAN_SETTLE_TIME_S": str(runtime_config["can_settle_time_s"]),
        "RESUME": "true" if runtime_config["resume"] else "false",
        "VALIDATE_AFTER_RECORD": "true" if runtime_config["validate_after_record"] else "false",
        "EXPECTED_CAN_INTERFACES": " ".join(runtime_config["expected_can_interfaces"]),
        "ROBOT_CAMERAS_JSON": json.dumps(runtime_config["cameras"], separators=(",", ":")),
        "ROBOT_CONFIG_JSON": json.dumps(runtime_config["robot"], separators=(",", ":")),
        "CAMERA_KEYS": ",".join(runtime_config["cameras"].keys()),
    }
    return "\n".join(f"{key}={shlex.quote(value)}" for key, value in values.items())


def main() -> None:
    parser = argparse.ArgumentParser(description="Load teleoperation recorder YAML config.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--format", choices=("json", "shell"), default="json")
    args = parser.parse_args()

    runtime_config = load_runtime_config(args.config, args.project_root)
    if args.format == "shell":
        print(shell_exports(runtime_config))
    else:
        print(json.dumps(runtime_config, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

