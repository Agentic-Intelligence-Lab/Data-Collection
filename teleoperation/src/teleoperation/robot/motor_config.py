from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PiperMotorsBusConfig:
    can_name: str
    motors: dict[str, tuple[int, str]]

    @property
    def type(self) -> str:
        return "piper"


MotorsBusConfig = PiperMotorsBusConfig

