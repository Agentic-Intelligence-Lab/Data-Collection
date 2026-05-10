from typing import Protocol

from teleoperation.robot.motor_config import MotorsBusConfig, PiperMotorsBusConfig


class MotorsBus(Protocol):
    def motor_names(self): ...
    def set_calibration(self): ...
    def apply_calibration(self): ...
    def revert_calibration(self): ...
    def read(self): ...
    def write(self): ...


def make_motors_buses_from_configs(motors_bus_configs: dict[str, MotorsBusConfig]) -> list[MotorsBus]:
    motors_buses = {}

    for key, cfg in motors_bus_configs.items():
        if cfg.type == "piper":
            from teleoperation.robot.piper_motor import PiperMotorsBus

            motors_buses[key] = PiperMotorsBus(cfg)

        else:
            raise ValueError(f"The motor type '{cfg.type}' is not supported by this project.")

    return motors_buses


def make_motors_bus(motor_type: str, **kwargs) -> MotorsBus:
    if motor_type == "piper":
        from teleoperation.robot.piper_motor import PiperMotorsBus

        config = PiperMotorsBusConfig(**kwargs)
        return PiperMotorsBus(config)

    else:
        raise ValueError(f"The motor type '{motor_type}' is not supported by this project.")

def get_motor_names(arm: dict[str, MotorsBus]) -> list:
        return [f"{arm}_{motor}" for arm, bus in arm.items() for motor in bus.motors]
