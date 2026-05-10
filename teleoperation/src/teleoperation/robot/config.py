from __future__ import annotations

from dataclasses import dataclass, field

from teleoperation.cameras.configs import CameraConfig, IntelRealSenseCameraConfig
from teleoperation.robot.motor_config import MotorsBusConfig, PiperMotorsBusConfig


DEFAULT_PIPER_MOTORS = {
    "joint_1": (1, "agilex_piper"),
    "joint_2": (2, "agilex_piper"),
    "joint_3": (3, "agilex_piper"),
    "joint_4": (4, "agilex_piper"),
    "joint_5": (5, "agilex_piper"),
    "joint_6": (6, "agilex_piper"),
    "gripper": (7, "agilex_piper"),
}


def _piper_bus(can_name: str) -> PiperMotorsBusConfig:
    return PiperMotorsBusConfig(can_name=can_name, motors=dict(DEFAULT_PIPER_MOTORS))


@dataclass
class PiperRobotConfig:
    inference_time: bool = False
    passive_recording: bool = True
    leader_arms: dict[str, MotorsBusConfig] = field(default_factory=dict)
    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "left": _piper_bus("can1"),
            "right": _piper_bus("can0"),
        }
    )
    follower_arm: dict[str, MotorsBusConfig] = field(default_factory=lambda: {"main": _piper_bus("can0")})
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "head": IntelRealSenseCameraConfig(
                serial_number=254622073267,
                fps=30,
                width=640,
                height=480,
                force_hardware_reset=False,
            ),
            "left_wrist": IntelRealSenseCameraConfig(
                serial_number=244222071617,
                fps=30,
                width=640,
                height=480,
                force_hardware_reset=False,
            ),
            "right_wrist": IntelRealSenseCameraConfig(
                serial_number=317622070857,
                fps=30,
                width=640,
                height=480,
                force_hardware_reset=False,
            ),
        }
    )

    @property
    def type(self) -> str:
        return "piper"


def _normalize_motor_tuple(value) -> tuple[int, str]:
    motor_id, model = value
    return int(motor_id), str(model)


def piper_bus_from_dict(raw: dict) -> PiperMotorsBusConfig:
    return PiperMotorsBusConfig(
        can_name=str(raw["can_name"]),
        motors={name: _normalize_motor_tuple(value) for name, value in raw["motors"].items()},
    )


def realsense_camera_from_dict(raw: dict) -> IntelRealSenseCameraConfig:
    camera_type = raw.get("type", "intelrealsense")
    if camera_type != "intelrealsense":
        raise ValueError(f"Unsupported camera type: {camera_type}")
    rotation = raw.get("rotation")
    if rotation == "none":
        rotation = None
    return IntelRealSenseCameraConfig(
        name=raw.get("name"),
        serial_number=int(raw["serial_number"]) if raw.get("serial_number") is not None else None,
        fps=int(raw["fps"]) if raw.get("fps") is not None else None,
        width=int(raw["width"]) if raw.get("width") is not None else None,
        height=int(raw["height"]) if raw.get("height") is not None else None,
        color_mode=raw.get("color_mode", "rgb"),
        use_depth=bool(raw.get("use_depth", False)),
        force_hardware_reset=bool(raw.get("force_hardware_reset", False)),
        rotation=rotation if rotation is None else int(rotation),
        mock=bool(raw.get("mock", False)),
    )


def piper_robot_config_from_dict(
    raw: dict | None,
    cameras: dict[str, CameraConfig] | None = None,
) -> PiperRobotConfig:
    if raw is None:
        return PiperRobotConfig(cameras=cameras or PiperRobotConfig().cameras)

    follower_arms = {
        name: piper_bus_from_dict(bus_cfg)
        for name, bus_cfg in raw.get("follower_arms", {}).items()
    }
    leader_arms = {
        name: piper_bus_from_dict(bus_cfg)
        for name, bus_cfg in raw.get("leader_arms", {}).items()
    }
    follower_arm = {
        name: piper_bus_from_dict(bus_cfg)
        for name, bus_cfg in raw.get("follower_arm", {}).items()
    }

    return PiperRobotConfig(
        inference_time=bool(raw.get("inference_time", False)),
        passive_recording=bool(raw.get("passive_recording", True)),
        leader_arms=leader_arms,
        follower_arms=follower_arms or PiperRobotConfig().follower_arms,
        follower_arm=follower_arm or PiperRobotConfig().follower_arm,
        cameras=cameras if cameras is not None else PiperRobotConfig().cameras,
    )
